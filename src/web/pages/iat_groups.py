#!/usr/bin/env python3
"""IAT › Análise Temporal › "Agrupar / renomear classes" (matriz com sugestões).

Para a coluna escolhida (rótulo revisado, origem ou previsão), cada linha é um
nome e cada coluna uma classe de referência: as previsões do modelo, ou os
rótulos atuais quando a própria coluna é a Previsão.  A cor mostra a fração
das imagens do nome em cada classe.  Clicar numa célula usa aquela classe como
nome nos gráficos; a sugestão é a classe dominante; "Aceitar sugestões" aplica
as que passam do limite.  As edições ficam num rascunho até "Aplicar"; fechar
a janela descarta.  Cálculos em ``iat_display_groups``.
"""
from __future__ import annotations

import pandas as pd
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from iat_display_groups import COLUMNS, clean_groups, count_renamed, match_matrix, save_groups

from .. import components as C
from ..headless import NOTIFY
from .execution import act

LABELS = {'revisado': 'Rótulo revisado', 'origem': 'Origem (pasta)', 'previsao': 'Previsão'}
USE = {'revisado': 'Vale para a projeção, a evolução entre intervalos, o resumo e a coluna Rótulo revisado do alluvial.',
       'origem': 'Vale para a coluna Origem do alluvial.',
       'previsao': 'Vale para a coluna Previsão do alluvial.'}
MAX_ROWS = 400


def _temporal():
    from . import iat_temporal
    return iat_temporal


def _fmt(n):
    return f'{int(n):,}'.replace(',', '.')


def _values(st, col):
    """(row values, reference values, reference title) over every loaded image."""
    data = st.data
    pred = data['previsto'] if 'previsto' in data.columns else None
    if col == 'revisado':
        return data['classe'], pred, 'classes previstas'
    if col == 'origem':
        return (data['origem'] if 'origem' in data.columns else None), pred, 'classes previstas'
    return pred, data['classe'], 'rótulos atuais'


def _matrix(st, col):
    cache = getattr(st, '_gmx_cache', None)
    if cache is None or cache.get('token') != st.load_token:
        cache = {'token': st.load_token}
        st._gmx_cache = cache
    if col not in cache:
        rows_v, ref_v, ref_title = _values(st, col)
        if rows_v is None:
            cache[col] = ([], [], {}, {}, {}, ref_title)
        else:
            if ref_v is None:
                ref_v = pd.Series('', index=rows_v.index, dtype=object)
            cache[col] = (*match_matrix(rows_v, ref_v), ref_title)
    return cache[col]


def _summary(st):
    g = st._gmx
    col = g['col']
    rows = _matrix(st, col)[0]
    draft = g['draft'][col]
    shown = {draft.get(n, n) for n in rows}
    extra = f' · lista limitada a {MAX_ROWS} nomes' if len(rows) > MAX_ROWS else ''
    return f'{len(rows)} nomes → {len(shown)} nos gráficos · {len(draft)} alterado(s){extra}'


def _table(st):
    g = st._gmx
    col = g['col']
    rows, refs, counts, totals, dominant, _ref_title = _matrix(st, col)
    if not rows:
        return C.hint('Não há nomes nesta coluna. Carregue a projeção do projeto de novo para ler origem e previsão.')
    draft = g['draft'][col]
    head = html.Thead(html.Tr([html.Th(LABELS[col]), html.Th('Imagens', className='num'),
                               *[html.Th(r, className='v', title=r) for r in refs],
                               html.Th('Sugestão'), html.Th('Nome nos gráficos')]))
    body = []
    for i, name in enumerate(rows[:MAX_ROWS]):
        cur = draft.get(name, name)
        row_counts, total = counts.get(name, {}), totals[name]
        cells = []
        for j, ref in enumerate(refs):
            n = row_counts.get(ref, 0)
            share = n / total if total else 0.0
            cells.append(html.Td(
                f'{round(100 * share)}' if share >= 0.05 else '',
                className='gmx-cell' + (' pick' if cur == ref else ''),
                title=f'{name} → {ref}: {_fmt(n)} imagens ({100 * share:.0f} %)',
                style={'background': f'rgba(34,211,238,{min(0.9, share * 0.95):.3f})',
                       'color': '#03151b' if share > 0.45 else 'var(--ink-2)'},
                **{'data-gmx': f'c:{i}:{j}'}))
        dom = dominant.get(name)
        sugg = (html.Span([html.I(className='on' if cur == dom[0] else None), dom[0],
                           html.Small(f'{100 * dom[1]:.0f} %')], className='gmx-sugg',
                          title='Usar a classe sugerida (clique de novo para voltar ao nome original)',
                          **{'data-gmx': f's:{i}'}) if dom else html.Span('—', className='hint'))
        body.append(html.Tr([
            html.Td(name, className='gmx-name', title=name),
            html.Td(_fmt(total), className='num'),
            *cells,
            html.Td(sugg),
            html.Td(dcc.Input(id={'type': 'gmx-name', 'i': i}, value=cur, debounce=True,
                              className='inp sm gmx-in' + (' renamed' if cur != name else ''),
                              placeholder=name)),
        ]))
    return html.Table([head, html.Tbody(body)], className='gmx')


def groups_modal(st):
    col = st._gmx['col']
    ref_title = _matrix(st, col)[5]
    body = html.Div([
        dcc.Store(id='gmx-click'),
        C.hint('Os nomes valem só para os gráficos e as exportações da Análise Temporal; os rótulos das imagens '
               'não mudam. Classes com o mesmo nome formam um grupo. Os nomes ficam salvos no projeto.'),
        html.Div([C.seg('gmx-col', [(c, LABELS[c]) for c in COLUMNS], col),
                  html.Span(USE[col], id='gmx-use', className='hint')], className='row'),
        html.Div([html.Label('Aceitar sugestões com pelo menos', className='hint', htmlFor='gmx-thr'),
                  dcc.Input(id='gmx-thr', type='number', min=0, max=100, step=5, value=70, className='inp sm',
                            style={'width': '76px'}),
                  html.Span('%', className='hint'),
                  C.btn('Aceitar sugestões', 'gmx-accept', 'sm pri'),
                  C.btn('Restaurar os nomes desta coluna', 'gmx-reset', 'sm'),
                  html.Span(_summary(st), id='gmx-sum', className='hint push')], className='row'),
        html.Div(_table(st), id='gmx-table', className='gmx-wrap'),
        html.Div(f'Colunas da matriz: {ref_title}. A cor e o número são a porcentagem das imagens de cada linha '
                 'em cada classe. Clique numa célula para usar aquela classe como nome, clique na sugestão para '
                 'aceitá-la ou desfazê-la, ou digite o nome.', id='gmx-help', className='hint'),
    ], className='stack')
    return C.modal_card('Agrupar e renomear classes', body, act('modal_close'), wide=True,
                        actions=[C.btn('Fechar sem aplicar', act('modal_close')),
                                 C.btn('Aplicar', 'gmx-apply', 'pri')])


def _first_alluvial_column(st):
    fs = getattr(st, '_origin_flow', None)
    cols = list(getattr(fs, 'cols', None) or ['origem'])
    for col in ('origem', 'previsao', 'revisado'):
        if col in cols:
            return col if (col != 'origem' or 'origem' in st.data.columns) else 'revisado'
    return 'revisado'


def _open(pick_column):
    def run(iat, lang):
        st = _temporal().state(iat._ctl)
        if st.data is None or st.data.empty:
            NOTIFY.push('info', 'Agrupar classes', 'Primeiro carregue a projeção.')
            return None
        st._gmx = {'col': pick_column(st), 'draft': clean_groups(st.group_aliases)}
        return {'modal': groups_modal(st)}
    return run


ACTIONS = {'tp_groups': _open(lambda st: 'revisado'), 'alv_groups': _open(_first_alluvial_column)}


def _set(draft, name, shown):
    shown = str(shown or '').strip()
    if shown and shown != name:
        draft[name] = shown
    else:
        draft.pop(name, None)


def register(app, ctl, get_iat):
    @app.callback(Output('gmx-table', 'children'), Output('gmx-sum', 'children'), Output('gmx-use', 'children'),
                  Output('gmx-help', 'children'),
                  Input('gmx-click', 'data'), Input('gmx-col', 'value'), Input('gmx-accept', 'n_clicks'),
                  Input('gmx-reset', 'n_clicks'), Input({'type': 'gmx-name', 'i': ALL}, 'value'),
                  State('gmx-thr', 'value'), prevent_initial_call=True)
    def gmx_update(click, col_value, _accept, _reset, _names, thr):
        trig = ctx.triggered_id
        with ctl.lock:
            st = _temporal().state(ctl)
            g = getattr(st, '_gmx', None)
            if g is None or st.data is None or st.data.empty:
                raise PreventUpdate
            col = g['col']
            rows, refs, _counts, _totals, dominant, _t = _matrix(st, col)
            draft = g['draft'][col]
            if trig == 'gmx-col':
                if col_value not in COLUMNS or col_value == col:
                    raise PreventUpdate
                g['col'] = col_value
            elif trig == 'gmx-click':
                key = str((click or {}).get('k') or '')
                parts = key.split(':')
                try:
                    i = int(parts[1])
                    name = rows[i]
                except (IndexError, ValueError):
                    raise PreventUpdate
                if parts[0] == 'c' and len(parts) == 3:
                    try:
                        ref = refs[int(parts[2])]
                    except (IndexError, ValueError):
                        raise PreventUpdate
                    _set(draft, name, name if draft.get(name, name) == ref else ref)
                elif parts[0] == 's' and name in dominant:
                    ref = dominant[name][0]
                    _set(draft, name, name if draft.get(name, name) == ref else ref)
                else:
                    raise PreventUpdate
            elif trig == 'gmx-accept':
                if not _accept:
                    raise PreventUpdate
                try:
                    limit = max(0.0, min(100.0, float(thr))) / 100.0
                except (TypeError, ValueError):
                    limit = 0.7
                for name in rows:
                    dom = dominant.get(name)
                    if dom and dom[1] >= limit:
                        _set(draft, name, dom[0])
            elif trig == 'gmx-reset':
                if not _reset:
                    raise PreventUpdate
                g['draft'][col] = {}
            elif isinstance(trig, dict) and trig.get('type') == 'gmx-name':
                try:
                    name = rows[int(trig['i'])]
                except (IndexError, ValueError, TypeError):
                    raise PreventUpdate
                value = str(ctx.triggered[0]['value'] or '').strip() or name
                if value == draft.get(name, name):
                    raise PreventUpdate
                _set(draft, name, value)
            else:
                raise PreventUpdate
            col = g['col']
            ref_title = _matrix(st, col)[5]
            help_text = (f'Colunas da matriz: {ref_title}. A cor e o número são a porcentagem das imagens de cada '
                         'linha em cada classe. Clique numa célula para usar aquela classe como nome, clique na '
                         'sugestão para aceitá-la ou desfazê-la, ou digite o nome.')
            return _table(st), _summary(st), USE[col], help_text

    @app.callback(Output('modal-host', 'children', allow_duplicate=True), Input('gmx-apply', 'n_clicks'),
                  prevent_initial_call=True)
    def gmx_apply(n):
        if not n:
            raise PreventUpdate
        iat = get_iat()
        with ctl.lock:
            st = _temporal().state(ctl)
            g = getattr(st, '_gmx', None)
            if g is None:
                raise PreventUpdate
            st.group_aliases = clean_groups(g['draft'])
            st._gmx = None
            fs = getattr(st, '_origin_flow', None)
            if fs is not None:
                fs._key = fs._result = None
            st.regroup()
            message = f'{count_renamed(st.group_aliases)} nome(s) alterado(s) nos gráficos.'
            if iat is not None and getattr(iat, 'project', None) is not None:
                try:
                    path = save_groups(iat.project.project_dir, st.group_aliases)
                    message += f'\n{path}'
                except OSError as exc:
                    message += f'\nNão foi possível salvar no projeto: {exc}'
            NOTIFY.push('info', 'Grupos aplicados', message)
            _temporal()._bump_iat()
        return []
