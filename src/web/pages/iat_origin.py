#!/usr/bin/env python3
"""IAT › Análise Temporal › painel "Origem → Previsão" (alluvial / Sankey).

Usa o mesmo recorte da subaba (filtros, classes de interesse e intervalos) e
liga, imagem a imagem, duas ou três colunas escolhidas livremente: a pasta de
origem, a previsão do modelo escolhido e o rótulo atual (revisado no IAT).
Classes podem ser unidas ou renomeadas só no diagrama (grupos de exibição).
Cálculos e figura estática em ``iat_origin_flow``; o projeto não é alterado,
só são gravadas as exportações pedidas.
"""
from __future__ import annotations

import base64
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from iat_display_groups import count_renamed
from iat_origin_flow import (COLUMN_KEYS, DEFAULT_TITLES, EXPORT_DEFAULTS, NEUTRAL, NO_SOURCE, SPECIAL,
                             category_order, flow_palette, flow_summary, flow_table,
                             prediction_snapshots, render_alluvial, save_alluvial, snapshot_predictions,
                             visible_flows)

from .. import components as C
from ..headless import FiledialogShim, NOTIFY
from ..theme import GRAPH_CONFIG, STATUS, plot_layout
from .execution import act

STYLES = [('parcats', 'Alluvial'), ('sankey', 'Sankey')]
COLUMNS = [('origem', 'Origem (pasta)'), ('previsao', 'Previsão'), ('revisado', 'Rótulo revisado')]
COLORS = [('origem', 'Origem'), ('previsao', 'Previsão'), ('revisado', 'Rótulo revisado')]
SCOPES = [('cur', 'Intervalo atual'), ('all', 'Todos os intervalos')]
MIN_SHARES = [('0', '0 %'), ('0.005', '0,5 %'), ('0.01', '1 %'), ('0.02', '2 %'), ('0.05', '5 %')]
EXPORT_THEMES = {'light': ('Claro — fundo branco', '#ffffff', '#131b2a'),
                 'dark': ('Escuro — fundo preto', '#0b0b0d', '#f4f7f8'),
                 'custom': ('Personalizado', None, None)}
_VALID = {'style': dict(STYLES), 'color': dict(COLORS), 'scope': dict(SCOPES), 'min_share': dict(MIN_SHARES)}
TITLE = 'Origem → Previsão'
# Summary texts for the pair compared (the first two columns shown).
PAIR_TEXT = {
    ('origem', 'previsao'): {'agree': 'Origem = previsão', 'agree_sub': 'concordância com a pasta',
                             'worst': 'Maior confusão', 'first': 'Classe de origem',
                             'same': 'Prevista como a própria', 'of': 'da pasta'},
    ('origem', 'revisado'): {'agree': 'Origem = rótulo revisado', 'agree_sub': 'rótulos atuais iguais à pasta',
                             'worst': 'Maior mudança', 'first': 'Classe de origem',
                             'same': 'Rótulo igual à pasta', 'of': 'da pasta'},
    ('previsao', 'revisado'): {'agree': 'Previsão = rótulo revisado',
                               'agree_sub': 'previsões mantidas no rótulo atual', 'worst': 'Maior correção',
                               'first': 'Classe prevista', 'same': 'Mantida no rótulo atual',
                               'of': 'das previstas como'},
}


def _av(key):
    return {'type': 'alv', 'key': key}


def _ax(key):
    return {'type': 'alx', 'key': key}


def _fmt(n):
    return f'{int(n):,}'.replace(',', '.')


def _pct(a, b, digits=1):
    return f'{100.0 * a / b:.{digits}f} %'.replace('.', ',') if b else '—'


def _rgba(color, alpha):
    from matplotlib.colors import to_rgba
    r, g, b, _a = to_rgba(color or NEUTRAL)
    return f'rgba({round(r * 255)},{round(g * 255)},{round(b * 255)},{alpha})'


# ------------------------------------------------------------------ state
class FlowState:
    """Panel settings and caches; lives on the temporal state object."""

    def __init__(self):
        self.style, self.color, self.scope = 'parcats', 'origem', 'cur'
        self.cols = ['origem', 'previsao']   # any two or all three of COLUMN_KEYS
        self.min_share, self.merge, self.model = '0.005', True, 'current'
        self.idx = 0
        self.pending = ''          # model version being loaded in the background
        self.alt = {}              # version -> predictions aligned to the loaded data
        self.alt_found = {}        # version -> model versions found in that copy
        self.snapshots = None
        self.current_label = None
        self.class_names = None
        self.token = None
        self.frames_version = None
        self._key = None
        self._result = None
        self.export = None
        self.export_sig = None


def flow_state(st):
    fs = getattr(st, '_origin_flow', None)
    if fs is None:
        fs = FlowState()
        st._origin_flow = fs
    if fs.token != st.load_token:  # project (re)loaded: drop everything tied to the old rows
        fs.token = st.load_token
        fs.alt, fs.alt_found, fs.snapshots = {}, {}, None
        fs.current_label = fs.class_names = None
        fs.model, fs.pending, fs._key, fs._result = 'current', '', None, None
    if fs.frames_version != st.version:  # intervals regenerated
        fs.frames_version = st.version
        fs.idx = st.current if st.frames else 0
        fs._key, fs._result = None, None
    return fs


def ready(st):
    return st.data is not None and not st.data.empty and 'origem' in st.data.columns


def _sync_pending(fs, iat):
    if fs.pending and not getattr(iat, '_op_running', False) and fs.pending not in fs.alt:
        fs.pending = ''  # the background read ended without a result (error or stop)


def _snapshots(fs, iat):
    if fs.snapshots is None:
        project = getattr(iat, 'project', None)
        fs.snapshots = prediction_snapshots(project.project_dir) if project is not None else []
    return fs.snapshots


def _current_label(st, fs):
    if fs.current_label is None:
        vers = sorted(v for v in st.data['versao'].astype(str).unique() if v and v != 'Sem versão')
        if vers:
            fs.current_label = 'Atual (' + ', '.join(vers[:3]) + ('…' if len(vers) > 3 else '') + ')'
        else:
            fs.current_label = 'Atual (classificação)'
    return fs.current_label


def _model_short(fs):
    return 'atual' if fs.model == 'current' else f'antes de {fs.model}'


def _model_options(st, fs, iat):
    opts = [{'label': _current_label(st, fs), 'value': 'current'}]
    for ver in _snapshots(fs, iat):
        label = f'Antes de {ver}'
        if ver in fs.alt:
            found = fs.alt_found.get(ver) or []
            label += f" ({', '.join(found)})" if found else ' (classificação original)'
        opts.append({'label': label, 'value': ver})
    return opts


def _class_names(st, fs):
    """Model classes = every top-1 prediction of the loaded images (manual labels
    are not used: in audit mode they start as the folder names)."""
    if fs.class_names is None:
        col = st.data['previsto']
        vals = col.cat.categories if isinstance(col.dtype, pd.CategoricalDtype) else col.unique()
        fs.class_names = {str(v) for v in vals if str(v).strip()}
    return fs.class_names


def _mask(mask):
    if hasattr(mask, 'fillna'):
        mask = mask.fillna(False)
    return np.asarray(mask, dtype=bool)


def _scope(st, fs):
    view, frames = st.view, st.frames
    if view is None or view.empty:
        return view, 'sem imagens'
    if fs.scope == 'all' or not frames:
        if not frames:
            return view, 'imagens filtradas'
        sel = np.zeros(len(view), dtype=bool)
        for _label, mask in frames:
            sel |= _mask(mask)
        return view.loc[sel], f'todos os {len(frames)} intervalos'
    i = max(0, min(fs.idx, len(frames) - 1))
    label, mask = frames[i]
    return view.loc[_mask(mask)], f'intervalo {label}'


def _selected(fs):
    """Columns to draw, always in the order origem → previsao → revisado."""
    return [c for c in COLUMN_KEYS if c in fs.cols]


def _series(rows, fs, cols):
    """Per-column Series for flow_table (None = column not drawn)."""
    pred = None
    if 'previsao' in cols:
        pred = fs.alt[fs.model].loc[rows.index] if fs.model in fs.alt else rows['previsto']
    return (rows['origem'] if 'origem' in cols else None, pred,
            (rows['classe_orig'] if 'classe_orig' in rows.columns else rows['classe']) if 'revisado' in cols else None)


def _groups(st):
    """Display names per column, shared with every chart of the tab (iat_display_groups)."""
    return getattr(st, 'group_aliases', None) or {}


def _groups_key(st):
    return tuple((col, tuple(sorted(m.items()))) for col, m in sorted(_groups(st).items()))


def compute(st, fs):
    """(table, summary, scope label) of the current settings; cached."""
    cols = _selected(fs)
    key = (st.version, st.load_token, id(st.view), fs.scope, fs.idx if fs.scope == 'cur' else -1,
           fs.model, tuple(cols), fs.merge, _groups_key(st))
    if fs._key == key and fs._result is not None:
        return fs._result
    rows, scope_label = _scope(st, fs)
    if rows is None or rows.empty or len(cols) < 2:
        table = pd.DataFrame(columns=cols + ['imagens'])
    else:
        table = flow_table(*_series(rows, fs, cols), merge=fs.merge, aliases=_groups(st))
    pred_names = _groups(st).get('previsao') or {}
    names = {pred_names.get(c, c) for c in _class_names(st, fs)}
    summary = flow_summary(table, names, merge=fs.merge)
    fs._key, fs._result = key, (table, summary, scope_label)
    return fs._result


def _columns(table):
    return [c for c in COLUMN_KEYS if c in table.columns]


def _palette(st, table):
    names = set()
    for col in _columns(table):
        names.update(table[col].tolist())
    return flow_palette(names, st.palette)


def _titles(fs, cols):
    title = {'origem': DEFAULT_TITLES[0], 'previsao': f'Previsão ({_model_short(fs)})',
             'revisado': DEFAULT_TITLES[2]}
    return [title[c] for c in cols]


def _tint(fs, cols):
    """Column whose class colours the ribbons (first column when the chosen one is hidden)."""
    return fs.color if fs.color in cols else cols[0]


# ----------------------------------------------------------------- figure
def flow_figure(fs, theme, table, palette):
    """Plotly parcats (alluvial) or Sankey of the visible flows.
    Returns ``(figure, height_px, hidden_images)``."""
    kept, hidden = visible_flows(table, float(fs.min_share))
    ink = theme.get('ink', '#f4f7f8')
    display = f"{theme.get('font_display', 'Chakra Petch')}, DejaVu Sans, sans-serif"
    layout = plot_layout(theme, margin=dict(l=20, r=20, t=40, b=16), showlegend=False, separators=',.',
                         xaxis=dict(visible=False), yaxis=dict(visible=False))
    if len(_selected(fs)) < 2 or kept is None or kept.empty:
        text = 'Marque pelo menos duas colunas' if len(_selected(fs)) < 2 else 'Sem imagens neste recorte'
        layout['annotations'] = [dict(text=text, x=.5, y=.5, xref='paper', yref='paper',
                                      showarrow=False, font=dict(color=STATUS['muted'], size=13))]
        return {'data': [], 'layout': layout}, 460, hidden
    cols = _columns(kept)
    titles = _titles(fs, cols)
    order = category_order(kept, cols)
    n_max = max(int(kept[c].nunique()) for c in cols)
    height = int(min(1100, max(460, 130 + 24 * n_max)))
    tint_col = _tint(fs, cols)
    if fs.style == 'sankey':
        nodes, index = [], {}
        for j, col in enumerate(cols):
            present = set(kept[col])
            for name in order:
                if name in present:
                    index[(j, name)] = len(nodes)
                    nodes.append(name)
        src, tgt, val, colour, path = [], [], [], [], []
        values = {c: kept[c].tolist() for c in cols}
        tints = kept[tint_col].tolist()
        counts = kept['imagens'].astype(int).tolist()
        for i, n in enumerate(counts):
            key = [values[c][i] for c in cols]
            for j in range(len(cols) - 1):
                src.append(index[(j, key[j])])
                tgt.append(index[(j + 1, key[j + 1])])
                val.append(n)
                colour.append(_rgba(palette.get(tints[i]), .42))
                path.append(' → '.join(key))
        data = [dict(type='sankey', arrangement='snap', valueformat=',d',
                     node=dict(label=nodes, color=[palette.get(n, NEUTRAL) for n in nodes], pad=14, thickness=14,
                               line=dict(width=0), hovertemplate='%{label}: %{value:,} imagens<extra></extra>'),
                     link=dict(source=src, target=tgt, value=val, color=colour, customdata=path,
                               hovertemplate='%{customdata}<br>%{value:,} imagens<extra></extra>'),
                     textfont=dict(color=ink, size=12))]
        xs = [0, 1] if len(cols) == 2 else [0, .5, 1]
        layout['annotations'] = [dict(text=f'<b>{t}</b>', x=xs[j], y=1.0, yshift=26, xref='paper', yref='paper',
                                      showarrow=False, xanchor='left' if j == 0 else ('right' if j == len(cols) - 1
                                                                                      else 'center'),
                                      font=dict(family=display, size=13, color=ink))
                                 for j, t in enumerate(titles)]
    else:
        n = len(order)
        rank = {name: i for i, name in enumerate(order)}
        scale = []
        for i, name in enumerate(order):
            colour = palette.get(name, NEUTRAL)
            scale += [[i / n, colour], [(i + 1) / n, colour]]
        dims = []
        for j, col in enumerate(cols):
            present = set(kept[col])
            dims.append(dict(label=titles[j], values=kept[col].tolist(), categoryorder='array',
                             categoryarray=[name for name in order if name in present]))
        data = [dict(type='parcats', dimensions=dims, counts=kept['imagens'].astype(int).tolist(),
                     arrangement='freeform', hoveron='color', hoverinfo='count+probability',
                     line=dict(color=[rank[v] + .5 for v in kept[tint_col]], colorscale=scale, cmin=0, cmax=n,
                               shape='hspline'),
                     labelfont=dict(size=13, color=ink, family=display), tickfont=dict(size=12, color=ink))]
        # Category names of the outer columns are written outside the plot area.
        widest = lambda col: max((len(str(v)) for v in set(kept[col])), default=8)
        layout['margin'] = dict(l=int(min(260, max(60, 7.2 * widest(cols[0]) + 18))),
                                r=int(min(260, max(60, 7.2 * widest(cols[-1]) + 18))), t=40, b=16)
    return {'data': data, 'layout': layout}, height, hidden


# ------------------------------------------------------------- side panel
def _tile(label, value, sub='', color='var(--accent)', wide=False):
    style = {'--c': color}
    if wide:
        style['gridColumn'] = '1 / -1'
    return html.Div([html.Small(label), html.B(value, className='tv', title=value), html.Em(sub)],
                    className='tile', style=style)


def side_panel(fs, summary, scope_label, palette):
    pair = summary.get('pair')
    total = summary['total']
    if pair is None or pair not in PAIR_TEXT:
        return [C.hint('Marque pelo menos duas colunas para ver o resumo.')]
    if not total:
        return [C.hint('Sem imagens neste recorte.')]
    tx = PAIR_TEXT[pair]
    tiles = [_tile('Imagens', _fmt(total), scope_label),
             _tile(tx['agree'], _pct(summary['agree'], total), tx['agree_sub'], STATUS['ok'])]
    worst = summary['worst']
    if worst:
        tiles.append(_tile(tx['worst'], f"{worst['a']} → {worst['b']}",
                           f"{_fmt(worst['n'])} imagens ({_pct(worst['n'], worst['of'])} {tx['of']} {worst['a']})",
                           STATUS['warn'], wide=True))
    review = summary['review']
    if review is not None:
        if review['disagree']:
            tiles.append(_tile('Rótulo atual onde pasta ≠ previsão',
                               f"{_fmt(review['origin'])} = pasta • {_fmt(review['prediction'])} = previsão",
                               f"{_fmt(review['other'])} outra classe, de {_fmt(review['disagree'])} imagens", wide=True))
        else:
            tiles.append(_tile('Rótulo atual onde pasta ≠ previsão', '—', 'pasta e previsão sempre iguais', wide=True))
    rows = []
    for r in summary['rows'][:300]:
        known = r['name'] not in SPECIAL
        share = r['same'] / r['n'] if (r['n'] and known) else 0.0
        best = r['best']
        rows.append(html.Tr([
            html.Td([html.Span(className='alv-sw', style={'background': palette.get(r['name'], NEUTRAL)}),
                     r['name']], className='cls', title=r['name']),
            html.Td(_fmt(r['n']), className='num'),
            html.Td([_pct(r['same'], r['n']) if known else '—',
                     html.Div(html.I(style={'width': f'{100 * share:.1f}%'}), className='alv-bar')]),
            html.Td([best[0], html.Span(' ' + _pct(best[1], r['n']), className='dim')] if best else '—',
                    className='dest', title=best[0] if best else None),
        ]))
    table = html.Div(html.Table([
        html.Thead(html.Tr([html.Th(tx['first']), html.Th('Imagens', className='num'),
                            html.Th(tx['same']), html.Th('Principal outro destino')])),
        html.Tbody(rows)]), className='tbl-wrap alv-tbl')
    return [html.Div(tiles, className='tiles t2 alv-tiles'), table]


def _warning(summary):
    share = summary.get('class_share')
    rows = summary.get('rows') or []
    first = (summary.get('pair') or (None,))[0]
    if first == 'origem' and rows and all(r['name'] == NO_SOURCE for r in rows):
        return html.Div('As imagens deste recorte não têm pasta de origem registrada.', className='alv-warn')
    if share is not None and summary['total'] and share < 0.5:
        return html.Div(f'Só {_pct(share, 1)} das imagens vêm de pastas com o nome de uma classe prevista pelo '
                        'modelo. Se as pastas forem estações, datas ou lotes, o diagrama mostra a pasta dos arquivos '
                        'e a concordância não tem significado.', className='alv-warn')
    return None


def _note(fs, hidden):
    parts = []
    if hidden:
        share = dict(MIN_SHARES).get(fs.min_share, fs.min_share)
        parts.append(f'{_fmt(hidden)} imagens em fluxos menores que {share} do total não aparecem no gráfico '
                     '(entram nas contagens ao lado).')
    if fs.pending:
        parts.append(f'Lendo as previsões anteriores a {fs.pending}…')
    return ' '.join(parts)


def _interval_label(st, fs):
    if fs.scope == 'all' or not st.frames:
        return 'todos'
    return str(st.frames[max(0, min(fs.idx, len(st.frames) - 1))][0])


def _outputs(st, fs, theme):
    table, summary, scope_label = compute(st, fs)
    palette = _palette(st, table)
    fig, height, hidden = flow_figure(fs, theme, table, palette)
    locked = fs.scope == 'all' or len(st.frames) < 2
    return (fig, {'height': f'{height}px'}, side_panel(fs, summary, scope_label, palette), _note(fs, hidden),
            f'modelo {_model_short(fs)} • {scope_label}', _interval_label(st, fs), _warning(summary), locked, locked)


def _ctl(label, control):
    return html.Div([html.Span(label, className='lbl'), control], className='alv-ctl')


def panel(st, iat, theme):
    """The HUD card (None until the intervals exist)."""
    from .iat import iact
    if st.data is None or st.data.empty or not st.frames:
        return None
    if not ready(st):
        return C.hud(TITLE, C.hint('Clique em ⟳ Carregar projeção para ler a pasta de origem das imagens.'))
    fs = flow_state(st)
    _sync_pending(fs, iat)
    fig, style, side, note, meta, ilabel, warn, prev_off, next_off = _outputs(st, fs, theme)
    controls = html.Div([
        _ctl('Estilo', C.seg(_av('style'), STYLES, fs.style)),
        _ctl('Colunas', dcc.Checklist(id=_av('cols'), value=_selected(fs), inline=True, className='seg',
                                      inputClassName='seg-in', labelClassName='seg-lb',
                                      options=[{'label': lbl, 'value': key} for key, lbl in COLUMNS])),
        _ctl('Cores por', C.seg(_av('color'), COLORS, fs.color)),
        _ctl('Recorte', C.seg(_av('scope'), SCOPES, fs.scope)),
        _ctl('Intervalo', html.Div([
            C.btn('◀', 'alv-prev', 'sm icon', disabled=prev_off, title='Intervalo anterior'),
            html.Span(ilabel, id='alv-ilabel', className='alv-ilabel'),
            C.btn('▶', 'alv-next', 'sm icon', disabled=next_off, title='Próximo intervalo')], className='alv-step')),
        _ctl('Modelo', dcc.Dropdown(id=_av('model'), options=_model_options(st, fs, iat),
                                    value=fs.pending or fs.model, clearable=False, className='dd',
                                    style={'minWidth': '220px'})),
        _ctl('Ocultar fluxos menores que', dcc.Dropdown(id=_av('min_share'), options=[
            {'label': lbl, 'value': val} for val, lbl in MIN_SHARES], value=fs.min_share, clearable=False,
            className='dd', style={'width': '120px'})),
        _ctl('Nomes', C.switch(_av('merge'), 'Unir grafias (maiúsculas e espaços)', fs.merge)),
        _ctl('Grupos do diagrama', html.Div([
            C.btn('Agrupar / renomear…', iact('alv_groups'), 'sm'),
            html.Span(f'{count_renamed(_groups(st))} nome(s) alterado(s)' if count_renamed(_groups(st))
                      else 'nomes originais',
                      className='hint')], className='row nowrap')),
    ], className='alv-ctrls')
    body = [
        controls,
        html.Div(warn, id='alv-warn'),
        html.Div([
            html.Div([dcc.Graph(id='alv-graph', figure=fig, config=dict(GRAPH_CONFIG, displayModeBar=False),
                                style=style),
                      html.Div(note, id='alv-note', className='hint')], className='stack alv-plot'),
            html.Div(side, id='alv-side', className='alv-side'),
        ], className='alv-grid'),
        html.Div([C.btn('Salvar figura…', iact('alv_export_fig'), 'pri'),
                  C.btn('Exportar tabela (CSV)', iact('alv_export_csv'))], className='row'),
        C.hint('Marque duas ou três colunas. Passe o mouse sobre uma faixa para ver quantas imagens fazem aquele '
               'caminho; no estilo Alluvial as categorias podem ser arrastadas. Usa os mesmos filtros, classes de '
               'interesse (pelo rótulo atual) e intervalos da aba. Origem = pasta da imagem na classificação; '
               'Previsão = top-1 do modelo escolhido; Rótulo revisado = rótulo atual do IAT, que antes de qualquer '
               'revisão é a própria previsão (ou a pasta, no modo auditoria). O resumo compara as duas primeiras '
               'colunas marcadas. A concordância com a pasta só equivale a acurácia se as pastas foram conferidas '
               'por especialista. Os nomes de "Agrupar / renomear" valem para todos os gráficos da aba.'),
    ]
    return C.hud(TITLE, body, meta=html.Span(meta, id='alv-meta'))


# ------------------------------------------------- previous model versions
def _load_snapshot(iat, st, fs, version):
    project = getattr(iat, 'project', None)
    if project is None:
        return False
    if getattr(iat, '_op_running', False):
        NOTIFY.push('info', TITLE, 'Aguarde a operação em andamento terminar e escolha o modelo de novo.')
        return False
    project_dir, keys, token = project.project_dir, st.data[['batch_source', 'names']].copy(), st.load_token
    fs.pending = version

    def done(res):
        if token != st.load_token:
            return  # project reloaded meanwhile
        series, found = res
        fs.alt[version] = series
        fs.alt_found[version] = found
        if fs.pending == version:
            fs.model, fs.pending = version, ''
        fs._key = fs._result = None
        iat.web_version = getattr(iat, 'web_version', 0) + 1
    iat._run_operation(f'Previsões anteriores a {version}',
                       lambda progress: snapshot_predictions(project_dir, version, keys, progress), done)
    return True


# ------------------------------------------------------------ export
def _temporal():
    from . import iat_temporal
    return iat_temporal


def _export_form(fs, iat, scope_label, n_max, cols):
    T = _temporal()
    fonts = T.font_options()
    model = _model_short(fs)
    titles = _titles(fs, cols) + [DEFAULT_TITLES[2]] * 3
    words = {'origem': 'origem', 'previsao': 'previsão', 'revisado': 'rótulo revisado'}
    path = ' → '.join(words[c] for c in cols)
    title = path[:1].upper() + path[1:] + (f' • modelo {model}' if 'previsao' in cols else '') + f' • {scope_label}'
    return {
        'fmt': 'png', 'dpi': 300, 'transparent': [], 'trim': ['on'], 'folder': T._export_folder(iat),
        'filename': f'origem_previsao_{datetime.now():%Y%m%d_%H%M}',
        'width_cm': 25.0, 'height_cm': float(round(min(60.0, max(12.0, 5.0 + 0.62 * n_max)), 1)),
        'title': title, 'col_a': titles[0], 'col_b': titles[1], 'col_c': titles[2],
        'show_counts': ['on'], 'font': 'DejaVu Sans' if 'DejaVu Sans' in fonts else fonts[0],
        'title_size': EXPORT_DEFAULTS['title_size'], 'header_size': EXPORT_DEFAULTS['header_size'],
        'label_size': EXPORT_DEFAULTS['label_size'], 'theme': 'light', 'background': '#ffffff', 'ink': '#131b2a',
        'ribbon_alpha': EXPORT_DEFAULTS['ribbon_alpha'],
    }


def _style_from(form):
    from matplotlib.colors import is_color_like
    T = _temporal()
    style = dict(EXPORT_DEFAULTS)
    for key in ('title', 'col_a', 'col_b', 'col_c'):
        style[key] = str(form.get(key) or '').strip()
    family = str(form.get('font') or '')
    style['font'] = family if family in T.font_options() else 'DejaVu Sans'
    style['title_size'] = T._num(form.get('title_size'), 'Tamanho do título', 6, 36)
    style['header_size'] = T._num(form.get('header_size'), 'Tamanho dos títulos das colunas', 6, 36)
    style['label_size'] = T._num(form.get('label_size'), 'Tamanho dos nomes das classes', 5, 30)
    for key, name in (('background', 'Fundo'), ('ink', 'Texto')):
        value = str(form.get(key) or '').strip()
        if not is_color_like(value):
            raise ValueError(f'Cor inválida em {name}.')
        style[key] = value
    style['ribbon_alpha'] = T._num(form.get('ribbon_alpha'), 'Opacidade das faixas', 0.1, 1.0)
    style['show_counts'] = T._on(form.get('show_counts'))
    style['figure_width'] = T._num(form.get('width_cm'), 'Largura (cm)', 8, 137) / 2.54
    style['figure_height'] = T._num(form.get('height_cm'), 'Altura (cm)', 6, 137) / 2.54
    style['dpi'] = T._num(form.get('dpi'), 'Resolução (dpi)', 30, 1200)
    style['transparent'] = T._on(form.get('transparent'))
    style['trim'] = T._on(form.get('trim'))
    fmt = str(form.get('fmt') or 'png')
    out = {'fmt': fmt if fmt in dict(T.FIG_FORMATS) else 'png', 'folder': str(form.get('folder') or '').strip(),
           'filename': str(form.get('filename') or '').strip()}
    return style, out


def _render_preview(exp, style):
    """Same renderer and size as the file, at a low dpi (data URI)."""
    from matplotlib.figure import Figure
    from PIL import Image
    from iat_temporal_figure import _savefig_kwargs
    fig = Figure(figsize=(style['figure_width'], style['figure_height']), dpi=100, facecolor=style['background'])
    render_alluvial(fig, exp['table'], exp['palette'], exp['color_by'], style)
    w_in, _h_in = fig.get_size_inches()
    pdpi = max(30.0, min(110.0, 1150.0 / max(w_in, 1e-6)))
    buf = io.BytesIO()
    fig.savefig(buf, format='png', **_savefig_kwargs('preview.png', dict(style, dpi=pdpi, transparent=False)))
    data = buf.getvalue()
    with Image.open(io.BytesIO(data)) as im:
        px_w, px_h = im.size
    return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii'), (px_w / pdpi, px_h / pdpi)


def _status(style, out, final, exp):
    w, h = final
    dpi = style['dpi']
    msg = (f'Arquivo: {w * 2.54:.1f} × {h * 2.54:.1f} cm • {round(w * dpi):,} × {round(h * dpi):,} px a '
           f'{dpi:g} dpi').replace(',', '.')
    if out['fmt'] in ('svg', 'pdf'):
        msg += ' • vetorial'
    if style['trim']:
        msg += ' • margens recortadas'
    if exp.get('hidden'):
        msg += f" • {_fmt(exp['hidden'])} imagens em fluxos pequenos ocultas, como na tela"
    return msg


def _ex_row(label, control, hint=None):
    return html.Div([html.Label(label, className='hint'), control,
                     html.Div(hint, className='hint small') if hint else None], className='xf-row')


def _ex_text(form, key, placeholder=''):
    return dcc.Input(id=_ax(key), value=form.get(key, ''), debounce=0.7, className='inp sm', placeholder=placeholder)


def _ex_num(form, key, lo, hi, step):
    return dcc.Input(id=_ax(key), type='number', min=lo, max=hi, step=step, value=form.get(key), debounce=0.7,
                     className='inp sm')


def _ex_color(form, key):
    return dcc.Input(id=_ax(key), type='color', value=form.get(key, '#000000'), debounce=0.5, className='color-in')


def _ex_dd(form, key, options):
    return dcc.Dropdown(id=_ax(key), options=[{'label': lbl, 'value': val} for val, lbl in options],
                        value=form.get(key), clearable=False, className='dd')


def _ex_sw(form, key, label):
    return dcc.Checklist(id=_ax(key), value=form.get(key) or [], options=[{'label': label, 'value': 'on'}],
                         className='sw', inputClassName='sw-in', labelClassName='sw-lb')


def export_modal(fs, form):
    T = _temporal()
    exp = fs.export
    try:
        style, out = _style_from(form)
        src, final = _render_preview(exp, style)
        status = _status(style, out, final, exp)
    except (ValueError, RuntimeError) as exc:
        src, status = '', f'⚠ {exc}'
    fonts = [(f, f) for f in T.font_options()]
    three = exp['ncols'] == 3
    file_box = html.Div([
        html.Div('Arquivo', className='kicker'),
        _ex_row('Formato', _ex_dd(form, 'fmt', T.FIG_FORMATS)),
        _ex_row('Resolução (dpi)', _ex_num(form, 'dpi', 30, 1200, 10),
                'Revistas: 300 dpi (fotos) a 600 dpi (gráficos com texto).'),
        _ex_sw(form, 'transparent', 'Fundo transparente (PNG, TIFF, SVG, PDF)'),
        _ex_sw(form, 'trim', 'Recortar margens em branco'),
        _ex_row('Pasta', html.Div([_ex_text(form, 'folder'),
                                   C.btn('…', 'alx-browse', 'sm icon', title='Escolher pasta')], className='browse')),
        _ex_row('Nome do arquivo (sem extensão)', _ex_text(form, 'filename'),
                'Se já existir, um sufixo _2, _3… é acrescentado; nada é sobrescrito.'),
    ], className='subcard')
    size_box = html.Div([
        html.Div('Tamanho', className='kicker'),
        html.Div([_ex_row('Largura (cm)', _ex_num(form, 'width_cm', 8, 137, 0.1)),
                  _ex_row('Altura (cm)', _ex_num(form, 'height_cm', 6, 137, 0.1))], className='g g-2'),
        C.hint('Muitas classes pedem mais altura: categorias baixas demais para o texto ficam sem rótulo.'),
    ], className='subcard')
    text_box = html.Div([
        html.Div('Textos', className='kicker'),
        _ex_row('Título (vazio = sem título)', _ex_text(form, 'title')),
        html.Div([_ex_row('Coluna 1', _ex_text(form, 'col_a')), _ex_row('Coluna 2', _ex_text(form, 'col_b')),
                  _ex_row('Coluna 3', _ex_text(form, 'col_c')) if three else None],
                 className='g g-3' if three else 'g g-2'),
        _ex_sw(form, 'show_counts', 'Mostrar o número de imagens ao lado de cada classe'),
    ], className='subcard')
    font_box = html.Div([
        html.Div('Fonte e tamanhos (pt)', className='kicker'),
        _ex_row('Fonte', _ex_dd(form, 'font', fonts)),
        html.Div([_ex_row('Título', _ex_num(form, 'title_size', 6, 36, 0.5)),
                  _ex_row('Colunas', _ex_num(form, 'header_size', 6, 36, 0.5)),
                  _ex_row('Classes', _ex_num(form, 'label_size', 5, 30, 0.5))], className='g g-3'),
    ], className='subcard')
    color_box = html.Div([
        html.Div('Cores', className='kicker'),
        _ex_row('Tema', _ex_dd(form, 'theme', [(k, v[0]) for k, v in EXPORT_THEMES.items()])),
        html.Div([_ex_row('Fundo', _ex_color(form, 'background')), _ex_row('Texto', _ex_color(form, 'ink'))],
                 className='g g-2'),
        _ex_row('Opacidade das faixas', _ex_num(form, 'ribbon_alpha', 0.1, 1.0, 0.05)),
        C.hint('As cores das classes são as mesmas da tela.'),
    ], className='subcard')
    body = html.Div([
        html.Div([
            html.Div([html.Div(html.Img(id='alx-img', src=src, className='xf-img'), className='xf-preview'),
                      html.Div(status, id='alx-status', className='hint'),
                      C.hint('A prévia usa o mesmo desenho do arquivo; só a resolução é menor. A figura salva '
                             'mostra os mesmos fluxos e cores da tela.')], className='stack xf-left'),
            html.Div([file_box, size_box, text_box, font_box, color_box], className='stack xf-right'),
        ], className='xf-grid'),
    ])
    return C.modal_card(f'Salvar figura — {TITLE}', body, act('modal_close'), wide=True,
                        actions=[C.btn('Restaurar padrão', 'alx-reset'), C.btn('Fechar', act('modal_close')),
                                 C.btn('Salvar', 'alx-save', 'pri')])


def _save(fs, iat, form):
    T = _temporal()
    exp = fs.export
    style, out = _style_from(form)
    folder = Path(out['folder'] or T._export_folder(iat)).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    stem = T._safe_stem(out['filename']) or f'origem_previsao_{datetime.now():%Y%m%d_%H%M%S}'
    path = T._unique_path(folder / f"{stem}.{out['fmt']}")
    save_alluvial(str(path), exp['table'], exp['palette'], exp['color_by'], style)
    if iat is not None:
        iat._log(f'Figura salva: {path}', 'success')
    NOTIFY.push('info', 'Figura salva', str(path))
    return path


def _open_export(iat, lang):
    T = _temporal()
    st = T.state(iat._ctl)
    if not ready(st) or not st.frames:
        NOTIFY.push('info', 'Salvar figura', 'Primeiro carregue a projeção e gere os intervalos.')
        return None
    fs = flow_state(st)
    if len(_selected(fs)) < 2:
        NOTIFY.push('info', 'Salvar figura', 'Marque pelo menos duas colunas.')
        return None
    table, _summary, scope_label = compute(st, fs)
    if table.empty:
        NOTIFY.push('info', 'Salvar figura', 'Não há imagens neste recorte.')
        return None
    kept, hidden = visible_flows(table, float(fs.min_share))
    cols = _columns(kept)
    fs.export = {'table': kept, 'palette': _palette(st, table), 'color_by': _tint(fs, cols), 'ncols': len(cols),
                 'cols': cols, 'hidden': hidden, 'scope_label': scope_label,
                 'n_max': max(int(kept[c].nunique()) for c in cols)}
    fs.export_sig = None
    return {'modal': export_modal(fs, _export_form(fs, iat, scope_label, fs.export['n_max'], cols))}


def _export_csv(iat, lang):
    """Every flow of the current settings (also the small ones) → CSV."""
    from ilt_contracts import atomic_write_csv
    T = _temporal()
    st = T.state(iat._ctl)
    if not ready(st) or not st.frames:
        NOTIFY.push('info', 'Exportar tabela', 'Primeiro carregue a projeção e gere os intervalos.')
        return None
    fs = flow_state(st)
    if len(_selected(fs)) < 2:
        NOTIFY.push('info', 'Exportar tabela', 'Marque pelo menos duas colunas.')
        return None
    table, _summary, scope_label = compute(st, fs)
    if table.empty:
        NOTIFY.push('info', 'Exportar tabela', 'Não há imagens neste recorte.')
        return None
    names = {'origem': 'origem', 'previsao': 'previsao_modelo', 'revisado': 'rotulo_revisado'}
    cols = _columns(table)
    a, b = names[cols[0]], names[cols[1]]
    out = table.rename(columns=names).copy()
    total = float(out['imagens'].sum())
    out['pct_do_total'] = (100.0 * out['imagens'] / total).round(4)
    out[f'pct_da_{a}'] = (100.0 * out['imagens'] / out.groupby(a)['imagens'].transform('sum')).round(4)
    out[f'{a}_igual_{b}'] = out[a] == out[b]
    out['modelo'] = _model_short(fs) if 'previsao' in cols else ''
    out['recorte'] = scope_label
    out['eixo'] = st.vars['axis'].get()
    out['grafias_unidas'] = bool(fs.merge)
    out['grupos_de_exibicao'] = bool(count_renamed(_groups(st)))
    folder = Path(T._export_folder(iat))
    folder.mkdir(parents=True, exist_ok=True)
    path = T._unique_path(folder / f'origem_previsao_{datetime.now():%Y%m%d_%H%M%S}.csv')
    atomic_write_csv(out, path, index=False)
    if count_renamed(_groups(st)):
        # The names used, so the table can be reproduced (names only; labels unchanged).
        from ilt_contracts import atomic_write_json
        atomic_write_json({'nota': 'Nomes e grupos usados só nos gráficos; os rótulos das imagens não mudam.',
                           'nome_original_para_nome_no_grafico': {c: dict(sorted(m.items()))
                                                                  for c, m in _groups(st).items() if m}},
                          path.with_name(path.stem + '_grupos.json'), ensure_ascii=False, indent=2)
    iat._log(f'Tabela origem → previsão salva: {path}', 'success')
    NOTIFY.push('info', 'Tabela exportada', str(path))
    return None


ACTIONS = {'alv_export_fig': _open_export, 'alv_export_csv': _export_csv}


# ------------------------------------------------------------- callbacks
def register(app, ctl, get_iat):
    def _state():
        return _temporal().state(ctl)

    @app.callback(Output('alv-graph', 'figure'), Output('alv-graph', 'style'), Output('alv-side', 'children'),
                  Output('alv-note', 'children'), Output('alv-meta', 'children'), Output('alv-ilabel', 'children'),
                  Output('alv-warn', 'children'), Output('alv-prev', 'disabled'), Output('alv-next', 'disabled'),
                  Input(_av(ALL), 'value'), Input('alv-prev', 'n_clicks'), Input('alv-next', 'n_clicks'),
                  State('theme-store', 'data'), prevent_initial_call=True)
    def alv_update(_values, _prev, _next, theme):
        from ..theme import sanitize
        trig = ctx.triggered_id
        iat = get_iat()
        with ctl.lock:
            st = _state()
            if not ready(st) or not st.frames:
                raise PreventUpdate
            fs = flow_state(st)
            _sync_pending(fs, iat)
            if trig in ('alv-prev', 'alv-next'):
                if not ctx.triggered[0]['value'] or fs.scope == 'all' or len(st.frames) < 2:
                    raise PreventUpdate
                fs.idx = (fs.idx + (1 if trig == 'alv-next' else -1)) % len(st.frames)
            elif isinstance(trig, dict) and trig.get('type') == 'alv':
                key, value = trig['key'], ctx.triggered[0]['value']
                if key == 'merge':
                    value = bool(value) and 'on' in value
                    if value == fs.merge:
                        raise PreventUpdate
                    fs.merge = value
                elif key == 'cols':
                    chosen = [c for c in COLUMN_KEYS if c in (value or [])]
                    if chosen == _selected(fs):
                        raise PreventUpdate
                    fs.cols = chosen
                elif key == 'model':
                    value = value or 'current'
                    if value == (fs.pending or fs.model):
                        raise PreventUpdate
                    if value == 'current' or value in fs.alt:
                        fs.model, fs.pending = value, ''
                    elif value in _snapshots(fs, iat):
                        _load_snapshot(iat, st, fs, value)
                    else:
                        raise PreventUpdate
                elif key in _VALID:
                    if value not in _VALID[key] or value == getattr(fs, key):
                        raise PreventUpdate
                    setattr(fs, key, value)
                else:
                    raise PreventUpdate
            else:
                raise PreventUpdate
            return _outputs(st, fs, sanitize(theme or {}))

    @app.callback(Output('alx-img', 'src'), Output('alx-status', 'children'),
                  Input(_ax(ALL), 'value'), prevent_initial_call=True)
    def alx_preview(values):
        form = {o['id']['key']: v for o, v in zip(ctx.inputs_list[0], values)}
        with ctl.lock:
            fs = getattr(_state(), '_origin_flow', None)
            if fs is None or fs.export is None:
                raise PreventUpdate
            sig = repr(sorted(form.items(), key=lambda kv: kv[0]))
            if fs.export_sig == sig:
                raise PreventUpdate  # re-inserted inputs, nothing changed
            fs.export_sig = sig
            exp = fs.export
        try:
            style, out = _style_from(form)
            src, final = _render_preview(exp, style)
            return src, _status(style, out, final, exp)
        except (ValueError, RuntimeError) as exc:
            return no_update, f'⚠ {exc}'

    @app.callback(Output(_ax('background'), 'value'), Output(_ax('ink'), 'value'),
                  Input(_ax('theme'), 'value'), prevent_initial_call=True)
    def alx_theme(preset):
        if preset not in EXPORT_THEMES or preset == 'custom':
            raise PreventUpdate
        _label, bg, ink = EXPORT_THEMES[preset]
        return bg, ink

    @app.callback(Output(_ax('folder'), 'value'), Input('alx-browse', 'n_clicks'), State(_ax('folder'), 'value'),
                  prevent_initial_call=True)
    def alx_browse(n, current):
        if not n:
            raise PreventUpdate
        chosen = FiledialogShim.askdirectory(initialdir=current or str(Path.home()), title='Pasta da exportação')
        if not chosen:
            raise PreventUpdate
        return chosen

    @app.callback(Output('modal-host', 'children', allow_duplicate=True),
                  Output('alx-status', 'children', allow_duplicate=True),
                  Input('alx-save', 'n_clicks'), State(_ax(ALL), 'value'), prevent_initial_call=True)
    def alx_save(n, values):
        if not n:
            raise PreventUpdate
        form = {o['id']['key']: v for o, v in zip(ctx.states_list[0], values)}
        iat = get_iat()
        with ctl.lock:
            fs = getattr(_state(), '_origin_flow', None)
            if fs is None or fs.export is None:
                raise PreventUpdate
            try:
                _save(fs, iat, form)
            except (ValueError, RuntimeError, OSError) as exc:
                return no_update, f'⚠ {exc}'
        return [], no_update

    @app.callback(Output('modal-host', 'children', allow_duplicate=True), Input('alx-reset', 'n_clicks'),
                  prevent_initial_call=True)
    def alx_reset(n):
        if not n:
            raise PreventUpdate
        iat = get_iat()
        with ctl.lock:
            fs = getattr(_state(), '_origin_flow', None)
            if fs is None or fs.export is None:
                raise PreventUpdate
            fs.export_sig = None
            exp = fs.export
            return export_modal(fs, _export_form(fs, iat, exp['scope_label'], exp['n_max'], exp['cols']))

