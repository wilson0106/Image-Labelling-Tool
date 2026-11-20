#!/usr/bin/env python3
"""IAT › Segmentation (same scopes, methods, metrics; equations and references)."""
from __future__ import annotations

import time
from urllib.parse import urlencode

import pandas as pd
from dash import dcc, html

from iat_backend import SEGMENTATION_METHODS
from segmentation_docs import METRIC_FIELDS_TEXT, METRIC_FORMULAS, method_doc_text

from .. import components as C
from ..pages import modals


def _p(lang, pt, en):
    return pt if lang == 'pt' else en


def on_show(iat):
    iat._refresh_seg_scope_ui()


def _scope_counts(iat, lang):
    bn = str(iat.vars['batch'].get() or 'Overview')
    counts = {'selected': len(iat.selected_gids), 'checked': len(iat.checked_gids),
              'batch': len(iat.current_df) if bn != 'Overview' else None,
              'project': len(iat.current_df) if bn == 'Overview' else None}
    enabled = {'selected': counts['selected'] > 0, 'checked': counts['checked'] > 0, 'batch': bn != 'Overview',
               'project': bool(iat.project and iat.project.csv_names())}

    def txt(key):
        n = counts.get(key)
        if n is None:
            return _p(lang, 'todos os batches', 'all batches') if key == 'project' and bn != 'Overview' else '—'
        return f'{n:,}'.replace(',', '.')
    return counts, enabled, txt


def _scope_rows(iat):
    """Rows the preview can browse: the segmentation scope (selection, checked
    images, batch or project), in dataframe order; the whole view if empty."""
    df = iat.current_df
    if df is None or df.empty or 'gid' not in df.columns:
        return df
    scope = str(iat.vars['seg_scope'].get() or 'selected')
    ids = iat.selected_gids if scope == 'selected' else iat.checked_gids if scope == 'checked' else None
    if ids:
        rows = df.loc[df['gid'].astype(str).isin(set(map(str, ids)))]
        if not rows.empty:
            return rows
    return df


def _preview_row(iat):
    if not iat.project:
        return None, 0, 0
    rows = _scope_rows(iat)
    if rows is None or rows.empty:
        return None, 0, 0
    pos = int(getattr(iat, '_seg_preview_pos', 0) or 0) % len(rows)
    iat._seg_preview_pos = pos
    return rows.iloc[pos], pos, len(rows)


def _histogram_figure(iat, row, method, theme, lang):
    """Grey levels of the preview image: object vs background pixels of the
    live mask, with the global threshold of the method when it has one."""
    import numpy as np
    from iat_backend import segmentation_threshold
    from ..theme import STATUS, plot_layout
    from .full_image import segmentation_live
    layout = plot_layout(theme, margin=dict(l=40, r=8, t=6, b=28), height=170, barmode='overlay', bargap=0,
                         showlegend=True, legend=dict(orientation='h', x=0, y=1.08, font=dict(size=10)),
                         xaxis=dict(range=[0, 255], title=dict(text=_p(lang, 'nível de cinza', 'grey level'))),
                         yaxis=dict(title=dict(text='px'), rangemode='tozero'))
    path = iat.project.resolve_image(str(row.get('csv_basename', '')), str(row.get('names', '')))
    try:
        radius = int(float(iat.vars['seg_radius'].get()))
    except (TypeError, ValueError):
        radius = 25
    gray, mask = segmentation_live(path, method, radius) if path else (None, None)
    if gray is None:
        layout['annotations'] = [dict(text='—', x=.5, y=.5, xref='paper', yref='paper', showarrow=False)]
        return {'data': [], 'layout': layout}
    edges = np.arange(0, 257, 4)
    centers = (edges[:-1] + 2).tolist()
    bg, _ = np.histogram(gray[~mask], bins=edges)
    obj, _ = np.histogram(gray[mask], bins=edges)
    data = [dict(type='bar', x=centers, y=bg.tolist(), name=_p(lang, 'fundo', 'background'),
                 marker=dict(color='#7f93aa'), opacity=.75, hovertemplate='%{x}: %{y} px<extra></extra>'),
            dict(type='bar', x=centers, y=obj.tolist(), name=_p(lang, 'objeto', 'object'),
                 marker=dict(color='#ff3048'), opacity=.75, hovertemplate='%{x}: %{y} px<extra></extra>')]
    thr, desc = segmentation_threshold(gray, method)
    if thr is not None:
        layout['shapes'] = [dict(type='line', xref='x', yref='paper', x0=thr, x1=thr, y0=0, y1=1,
                                 line=dict(color=STATUS['warn'], width=1.5, dash='dash'))]
        layout['annotations'] = [dict(text=f'{desc} = {thr:.0f}', x=thr, y=1, xref='x', yref='paper',
                                      showarrow=False, xanchor='left', yanchor='top',
                                      font=dict(color=STATUS['warn'], size=10))]
    else:
        layout['annotations'] = [dict(text=desc, x=1, y=1, xref='paper', yref='paper', showarrow=False,
                                      xanchor='right', yanchor='top', font=dict(color=STATUS['muted'], size=10))]
    return {'data': data, 'layout': layout}


def _step_preview(delta):
    def run(iat, lang):
        rows = _scope_rows(iat)
        n = 0 if rows is None else len(rows)
        if n:
            iat._seg_preview_pos = (int(getattr(iat, '_seg_preview_pos', 0) or 0) + delta) % n
        iat.seg_version = getattr(iat, 'seg_version', 0) + 1
    return run


def _metrics_lines(iat, row, method):
    bn, name = str(row.get('csv_basename', '')), str(row.get('names', ''))
    fallback = iat._segmentation_pixel_size(None, iat.vars['pixel_size'].get())
    seg_row = None
    if iat.project.segmentation_csv_path(bn, method).exists():
        try:
            sdf = iat.project.get_segmentation_table(bn, method)
            m = sdf[sdf['Image Name'].astype(str) == name] if not sdf.empty and 'Image Name' in sdf.columns else pd.DataFrame()
            if not m.empty:
                seg_row = m.iloc[0]
        except Exception:
            seg_row = None
    um = iat._segmentation_pixel_size(seg_row, fallback)
    lines = [f"{iat.tr('method')}: {SEGMENTATION_METHODS.get(method, method)}", f'Pixel size: {um:g} µm/pixel', '']
    lines.extend(iat._format_segmentation_metrics(seg_row, um) if seg_row is not None else [iat.tr('no_segmentation')])
    return lines


def render(iat, lang, theme):
    from .iat import iact, iat_input, iat_select, iat_switch
    method = iat.vars['seg_method'].get()
    counts, enabled, txt = _scope_counts(iat, lang)
    texts = {'selected': iat.tr('seg_scope_selected'), 'checked': iat.tr('seg_scope_checked'),
             'batch': iat.tr('seg_scope_batch'), 'project': iat.tr('seg_scope_project')}
    scope = str(iat.vars['seg_scope'].get() or 'selected')
    scope_ctl = dcc.RadioItems(
        id={'type': 'ivs', 'key': 'seg_scope'}, value=scope, className='clist', inputClassName='chk',
        labelClassName='clist-item',
        options=[{'label': html.Span([texts[k], html.B(txt(k), className='push accent-text')], className='row nowrap'),
                  'value': k, 'disabled': not enabled[k]} for k in ('selected', 'checked', 'batch', 'project')])
    radius_on = method in {'rolling_ball', 'black_hat_otsu', 'adaptive_sauvola'}
    eq, ref = method_doc_text(method, lang)
    row, pos, total = _preview_row(iat)
    if row is not None:
        bn, name = str(row.get('csv_basename', '')), str(row.get('names', ''))
        url = '/iat-segview?' + urlencode({'bn': bn, 'name': name, 'm': method,
                                           'r': str(iat.vars['seg_radius'].get()), 't': int(time.time() * 1000)})
        preview = [
            html.Div([C.btn('◀', iact('seg_prev'), 'sm icon'), C.btn('▶', iact('seg_next'), 'sm icon'),
                      html.Span(_p(lang, f'Imagem {pos + 1:,} de {total:,} do escopo', f'Image {pos + 1:,} of {total:,} in scope')
                                .replace(',', '.'), className='hint'),
                      html.Span(name, className='hint small push')], className='seg-nav'),
            html.Img(src=url, className='seg-view'),
            html.Div(_p(lang, 'Original  •  Máscara  •  Sobreposição — calculadas agora com o método e o raio escolhidos',
                        'Original  •  Mask  •  Overlay — computed now with the chosen method and radius'),
                     className='hint small'),
            dcc.Graph(figure=_histogram_figure(iat, row, method, theme, lang),
                      config={'displayModeBar': False, 'responsive': True}, style={'height': '170px'}),
        ]
        metrics = html.Pre('\n'.join(_metrics_lines(iat, row, method)), className='pre-block')
    else:
        preview, metrics = C.hint('—'), C.hint('—')
    fields = METRIC_FIELDS_TEXT['en' if lang == 'en' else 'pt']
    controls = html.Div([
        C.hud(iat.tr('seg_scope_title'), [
            scope_ctl,
            C.hint(f"{iat.tr('seg_scope_confirm')}: {texts.get(scope, scope)} • {txt(scope)}"),
        ]),
        C.hud(iat.tr('seg_params'), [
            iat_select(iat, 'seg_method', list(SEGMENTATION_METHODS.items())),
            html.Label(iat.tr('radius'), className='lbl' if radius_on else 'lbl muted'),
            dcc.Input(id={'type': 'iv', 'key': 'seg_radius'}, value=str(iat.vars['seg_radius'].get()), debounce=True,
                      className='inp', disabled=not radius_on),
            C.hint(_p(lang, 'Usado apenas em Rolling ball, Black-hat + Otsu e Adaptive / Sauvola.',
                      'Used only by Rolling ball, Black-hat + Otsu and Adaptive / Sauvola.')),
            html.Div([
                html.Div(_p(lang, 'Escala física das métricas', 'Physical metric scale'), className='kicker'),
                iat_input(iat, 'pixel_size', alt=True, class_name='inp sm', style={'width': '110px'}),
                C.hint(_p(lang, 'Não altera o limiar/máscara; é usada para µm, µm² e barra de escala.',
                          'Does not change threshold/mask; used for µm, µm² and the scale bar.')),
            ], className='subcard'),
            C.hint(_p(lang, 'A prévia mostra a máscara e o contorno com estes parâmetros antes de gerar.',
                      'The preview shows the mask and outline with these parameters before generating.')),
            C.btn(iat.tr('build_seg'), iact('build_seg'), 'pri'),
        ], variant='alt'),
    ], className='stack ctrl-col')
    center = html.Div([
        html.Div([
            C.hud(iat.tr('preview'), preview),
            C.hud(_p(lang, 'Métricas da segmentação gerada', 'Metrics of the generated segmentation'), metrics,
                  variant='alt'),
        ], className='g', style={'gridTemplateColumns': 'minmax(0,1.6fr) minmax(260px,1fr)'}),
        html.Div([
            C.hud(_p(lang, 'Método: equação e referência', 'Method: equation and reference'), [
                html.B(SEGMENTATION_METHODS.get(method, method)),
                html.Pre(eq, className='eq'),
                html.Div(ref, className='note'),
            ]),
            C.hud(_p(lang, 'Métricas: definições', 'Metrics: definitions'),
                  html.Pre(METRIC_FORMULAS['en' if lang == 'en' else 'pt'], className='eq'), variant='alt'),
        ], className='g g-2'),
        C.hud(iat.tr('gallery_order_metrics'), C.note(iat.tr('available_fields') + ':\n' + fields)),
    ], className='stack')
    return html.Div([controls, center], className='g iat-2col')


def _build(iat, lang):
    if str(iat.vars['seg_scope'].get()) == 'project' and iat.project:
        csvs = iat.project.csv_names()
        bn = str(iat.vars['batch'].get() or 'Overview')
        n = len(iat.current_df) if bn == 'Overview' else 0
        detail = (f'{n:,} imagens em {len(csvs)} batches'.replace(',', '.') if n else f'{len(csvs)} batches')
        q = _p(lang, f'Você está prestes a gerar/atualizar a segmentação para o projeto inteiro ({detail}).\n\n'
                     'A operação pode demorar bastante. Deseja continuar?',
               f'You are about to generate/update segmentation for the entire project ({detail}).\n\n'
               'This operation may take a long time. Continue?')
        from .execution import act
        from .iat import iact
        return {'modal': C.modal_card(iat.tr('iat_title'), html.P(q, className='note'), act('modal_close'),
                                      actions=[C.btn(_p(lang, 'Não', 'No'), act('modal_close')),
                                               C.btn(_p(lang, 'Sim', 'Yes'), iact('build_seg_yes'), 'pri')])}
    iat.build_segmentation()


def _build_yes(iat, lang):
    iat.build_segmentation()
    return {'modal': []}


ACTIONS = {'build_seg': _build, 'build_seg_yes': _build_yes,
           'seg_prev': _step_preview(-1), 'seg_next': _step_preview(1)}
