#!/usr/bin/env python3
"""IAT › Projection / Scatter (WebGL).

Selections, the class panel scale and point clicks update only the figure
they affect; the page is rebuilt only when data or options change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from .. import components as C
from ..iat_plots import class_abundance_figure, scatter_figure
from ..theme import GRAPH_CONFIG, sanitize
from .execution import dyn


def _p(lang, pt, en):
    return pt if lang == 'pt' else en


def _sel_text(iat):
    return f"{iat.tr('selected')}: {len(iat.selected_gids):,}".replace(',', '.')


def _points_text(lang, n_points, sampled):
    return (_p(lang, f'{n_points:,} pontos', f'{n_points:,} points').replace(',', '.')
            + (_p(lang, ' • exibindo amostra; a seleção usa todos', ' • showing a sample; selection uses all') if sampled else ''))


SCATTER_CFG = dict(GRAPH_CONFIG, modeBarButtonsToRemove=['autoScale2d', 'toggleSpikelines'], scrollZoom=True)


def render(iat, lang, theme):
    from .iat import iact, iat_select, iat_switch
    df = iat.current_df
    stale = 0
    if df is not None and 'projection_stale' in df.columns:
        s = df['projection_stale']
        stale = int(s.fillna(False).sum()) if s.dtype == bool else int(
            s.fillna(False).astype(str).str.lower().isin({'1', 'true', 't', 'yes', 'y'}).sum())
    pairs, _p2k, _k2p = iat._projection_display_maps()
    cpairs, _c2k, _k2c = iat._color_display_maps()
    labels = iat._scatter_class_labels()
    is_overview = iat.vars['batch'].get() == 'Overview'
    if not iat.visible_classes and not is_overview:
        iat.visible_classes = set(labels)
    multi = (not is_overview) or bool(iat.vars['overview_multi_classes'].get())
    fig, n_points, sampled = scatter_figure(iat, lang, theme)

    if multi:
        class_ctl = dcc.Checklist(id='sc-classes', options=[{'label': l, 'value': l} for l in labels],
                                  value=sorted(iat.visible_classes & set(labels)), className='clist',
                                  inputClassName='chk', labelClassName='clist-item')
    else:
        cur = sorted(iat.visible_classes & set(labels))
        class_ctl = dcc.RadioItems(id='sc-classes', options=[{'label': l, 'value': l} for l in labels],
                                   value=cur[0] if cur else None, className='clist', inputClassName='chk',
                                   labelClassName='clist-item')

    controls = C.hud(iat.tr('scatter'), [
        html.Div('⚠ ' + _p(lang, f'Projeção x/y desatualizada para {stale:,} imagem(ns). Recalcule a projeção antes de interpretar distâncias.',
                           f'x/y projection is stale for {stale:,} image(s). Recompute the projection before interpreting distances.').replace(',', '.'),
                 className='warn-text') if stale else None,
        html.Label(_p(lang, 'Projeção', 'Projection'), className='lbl'),
        iat_select(iat, 'projection_method', [(k, l) for l, k in pairs]),
        html.Div([C.btn(_p(lang, 'Calcular', 'Compute'), iact('run_projection'), 'sm pri'),
                  C.btn('Original', iact('restore_projection'), 'sm')], className='row'),
        C.hint(_p(lang, '* Supervisionado usa rótulos com peso moderado; mantenha o UMAP não supervisionado para procurar erros.',
                  '* Supervised mode uses labels with moderate weight; use unsupervised UMAP when looking for label errors.')),
        html.Label(_p(lang, 'Colorir por', 'Color by'), className='lbl'),
        iat_select(iat, 'scatter_color_mode', [(k, l) for l, k in cpairs]),
        iat_switch(iat, 'show_hulls', _p(lang, 'Mostrar contorno das manchas', 'Show stain outlines')),
        iat_switch(iat, 'hide_relabel', iat.tr('hide_relabel')),
        html.Label(iat.tr('class_filter'), className='lbl'),
        html.Div(class_ctl, className='clist-box tall'),
        iat_switch(iat, 'overview_multi_classes', iat.tr('overview_multi_classes')) if is_overview else None,
        C.hint(iat.tr('overview_multi_tip')) if is_overview else None,
        html.Div([C.btn(iat.tr('all_classes'), iact('all_classes'), 'sm', disabled=not multi),
                  C.btn(iat.tr('no_classes'), iact('no_classes'), 'sm')], className='row'),
        C.hint(iat.tr('class_click_tip')),
        C.btn(iat.tr('select_class_points'), iact('select_class_points'), 'dan'),
        C.hint(iat.tr('overview_info')) if is_overview else None,
        html.Label(iat.tr('selection_tools'), className='lbl'),
        C.hint(_p(lang, 'Use Laço ou Retângulo na barra do gráfico; clique em um ponto para marcá-lo/desmarcá-lo.',
                  'Use Lasso or Box in the chart toolbar; click a point to toggle it.')),
        html.Div([C.btn(iat.tr('clear_sel'), iact('clear_selection'), 'sm'),
                  C.btn(iat.tr('build_sel'), iact('build_selection'), 'sm pri')], className='row'),
        html.Div(_sel_text(iat), id='sc-selcount', className='big-count'),
        C.hint(iat.tr('scatter_tip')),
    ], class_name='ctrl-col')

    bars = class_abundance_figure(iat, lang, theme)
    plot = C.hud(iat.tr('scatter'), [
        html.Div(_points_text(lang, n_points, sampled), id='sc-points', className='hint'),
        dcc.Graph(id='sc-graph', figure=fig, config=SCATTER_CFG, className='plot', clear_on_unhover=True,
                  style={'height': '68vh', 'minHeight': '460px'}),
        dcc.Tooltip(id='sc-tip', direction='right', background_color='rgba(8,12,15,.96)',
                    border_color='var(--accent)', loading_text='…'),
        C.hint(_p(lang, 'Passe o mouse sobre um ponto para ver a imagem, o rótulo e a previsão.',
                  'Hover a point to see the image, label and prediction.')),
    ], meta='WebGL')
    side = C.hud(iat.tr('class_abundance'), [
        html.Div([
            html.Span(iat.tr('class_axis_mode'), className='hint'),
            C.seg('sc-mode', [('N', iat.tr('axis_count')), ('%', iat.tr('axis_percent'))],
                  iat.vars['class_axis_mode'].get() or 'N'),
        ], className='row'),
        html.Div([
            html.Span(iat.tr('axis_scale'), className='hint'),
            html.Div(dcc.Slider(id='sc-scale', min=45, max=220, step=5, value=float(iat.vars['class_axis_scale'].get() or 100),
                                marks=None, updatemode='drag', className='slider'), className='grow'),
            C.btn(iat.tr('axis_auto'), 'sc-auto', 'sm'),
        ], className='row nowrap'),
        html.Div(dcc.Graph(id='sc-bars', figure=bars, config={'displaylogo': False, 'displayModeBar': False,
                                                                  'responsive': True},
                           style={'height': f"{bars['layout'].get('height', 300)}px"}),
                 className='bars-scroll'),
    ], variant='alt')
    return html.Div([controls, html.Div([plot, side], className='g sc-main')], className='g iat-2col')


def _set_visible(iat, classes):
    iat._set_visible_classes(classes)


ACTIONS = {
    'run_projection': lambda iat, lang: iat._run_projection_analysis(),
    'restore_projection': lambda iat, lang: iat._restore_original_projection(),
    'all_classes': lambda iat, lang: _set_visible(iat, iat._scatter_class_labels()),
    'no_classes': lambda iat, lang: _set_visible(iat, []),
    'select_class_points': lambda iat, lang: iat.select_current_class(),
    'clear_selection': lambda iat, lang: iat.clear_selection() or iat._draw_scatter(),
    'build_selection': lambda iat, lang: iat.build_selection(),
}


def _selection_mask(df, sel):
    pts = np.c_[df['x'].to_numpy(float), df['y'].to_numpy(float)]
    if sel.get('lassoPoints'):
        lp = sel['lassoPoints']
        verts = np.c_[lp['x'], lp['y']]
        from matplotlib.path import Path as MplPath
        # Bounding-box prefilter keeps the polygon test cheap on large Overview sets.
        x0, y0 = verts.min(axis=0)
        x1, y1 = verts.max(axis=0)
        box = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        mask = np.zeros(len(pts), bool)
        if box.any():
            mask[box] = MplPath(verts).contains_points(pts[box])
        return mask, list(map(tuple, verts)) + [tuple(verts[0])]
    if sel.get('range'):
        (x0, x1), (y0, y1) = sorted(sel['range']['x']), sorted(sel['range']['y'])
        mask = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        return mask, [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return None, None


_HOVER_INDEX = {'key': None, 'index': None}


def _hover_row(df, gid):
    """Row of *gid* in the drawn points (index built once per dataframe)."""
    key = (id(df), len(df))
    if _HOVER_INDEX['key'] != key:
        _HOVER_INDEX['index'] = pd.Index(df['gid'].astype(str).to_numpy())
        _HOVER_INDEX['key'] = key
    pos = _HOVER_INDEX['index'].get_indexer([gid])[0]
    return None if pos < 0 else df.iloc[int(pos)]


def _fmt_num(value, digits=2):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return '—'
    return '—' if not np.isfinite(v) else f'{v:.{digits}f}'


def register(app, ctl, get_iat, respond):
    # Hover tooltip (item 64): thumbnail + labels of the point under the
    # cursor.  Read-only and lock-free: it reads the dataframe already drawn.
    @app.callback(Output('sc-tip', 'show'), Output('sc-tip', 'bbox'), Output('sc-tip', 'children'),
                  Input('sc-graph', 'hoverData'), State('lang-store', 'data'), prevent_initial_call=True)
    def scatter_hover(hover, lang):
        from .iat_gallery import thumb_url
        pts = (hover or {}).get('points') or []
        if not pts or pts[0].get('customdata') is None or not pts[0].get('bbox'):
            return False, no_update, no_update
        iat = get_iat()
        df = iat._scatter_points_df
        if df is None or df.empty or 'gid' not in df.columns:
            return False, no_update, no_update
        row = _hover_row(df, str(pts[0]['customdata']))
        if row is None:
            return False, no_update, no_update
        lang = lang or 'pt'
        name = str(row.get('names', ''))
        pred = row.get('preds1', row.get('pred', ''))
        lines = [html.B(name),
                 html.Div(f"{_p(lang, 'Rótulo manual', 'Manual label')}: {row.get('manual_label', '—')}"),
                 html.Div(f"{_p(lang, 'Previsão', 'Prediction')}: {pred} ({_fmt_num(row.get('confs1'))})")]
        if 'var_norm' in row.index:
            lines.append(html.Div(f"{_p(lang, 'Incerteza', 'Uncertainty')}: {_fmt_num(row.get('var_norm'))}"))
        src = str(row.get('source_label', '') or '')
        if src:
            lines.append(html.Div(f"{_p(lang, 'Origem', 'Source')}: {src}", className='hint'))
        body = html.Div([html.Img(src=thumb_url(str(row.get('csv_basename', '')), name)), *lines], className='sc-tip')
        return True, pts[0]['bbox'], body

    @app.callback(Output(dyn(ALL), 'children', allow_duplicate=True),
                  Input('sc-classes', 'value'), State('lang-store', 'data'), State('theme-store', 'data'),
                  prevent_initial_call=True)
    def classes_changed(value, lang, theme):
        from .iat import versions
        iat = get_iat()
        with ctl.lock:
            before = versions(iat)
            sel = value if isinstance(value, list) else ([value] if value else [])
            if set(sel) == (iat.visible_classes & set(iat._scatter_class_labels())):
                raise PreventUpdate
            if iat.vars['batch'].get() == 'Overview' and not iat.vars['overview_multi_classes'].get():
                iat.visible_classes = {sel[-1]} if sel else set()
            else:
                iat.visible_classes = set(sel)
            iat._draw_scatter()
            return respond(iat, before, lang, theme)

    # Selection / point click: redraw only the projection and the counter.
    @app.callback(Output('sc-graph', 'figure', allow_duplicate=True), Output('sc-selcount', 'children'),
                  Output('sc-points', 'children'),
                  Input('sc-graph', 'selectedData'), Input('sc-graph', 'clickData'),
                  State('lang-store', 'data'), State('theme-store', 'data'), prevent_initial_call=True)
    def graph_select(sel, click, lang, theme):
        from .iat import versions
        iat = get_iat()
        with ctl.lock:
            df = iat._scatter_points_df
            if ctx.triggered[0]['prop_id'].endswith('selectedData'):
                if not sel or df is None or df.empty:
                    raise PreventUpdate
                mask, outline = _selection_mask(df, sel)
                if mask is None:
                    raise PreventUpdate
                iat.selected_gids = set(df['gid'].to_numpy(object)[mask].astype(str))
                iat._selection_outline = outline
            else:
                pts = (click or {}).get('points') or []
                gids = [str(p.get('customdata')) for p in pts if p.get('customdata') is not None]
                if not gids:
                    raise PreventUpdate
                for gid in gids:
                    if gid in iat.selected_gids:
                        iat.selected_gids.discard(gid)
                    else:
                        iat.selected_gids.add(gid)
            fig, n_points, sampled = scatter_figure(iat, lang or 'pt', sanitize(theme or {}))
            iat.rendered_versions = versions(iat)
            return fig, _sel_text(iat), _points_text(lang or 'pt', n_points, sampled)

    # Class panel: mode, scale and "Auto" redraw only the bar chart.
    @app.callback(Output('sc-bars', 'figure'), Output('sc-bars', 'style'), Output('sc-scale', 'value'),
                  Input('sc-mode', 'value'), Input('sc-scale', 'value'), Input('sc-auto', 'n_clicks'),
                  State('lang-store', 'data'), State('theme-store', 'data'), prevent_initial_call=True)
    def bars_update(mode, scale, _auto, lang, theme):
        iat = get_iat()
        with ctl.lock:
            if ctx.triggered_id == 'sc-auto':
                scale = 100.0
            iat.vars['class_axis_mode'].set(mode or 'N')
            iat.vars['class_axis_scale'].set(float(scale or 100))
            fig = class_abundance_figure(iat, lang or 'pt', sanitize(theme or {}))
        return fig, {'height': f"{fig['layout'].get('height', 300)}px"}, float(scale or 100)

    @app.callback(Output(dyn(ALL), 'children', allow_duplicate=True),
                  Input('sc-bars', 'clickData'), State('lang-store', 'data'), State('theme-store', 'data'),
                  prevent_initial_call=True)
    def bar_click(click, lang, theme):
        from .iat import versions
        pts = (click or {}).get('points') or []
        if not pts:
            raise PreventUpdate
        lab = str(pts[0].get('customdata') or pts[0].get('y'))
        iat = get_iat()
        with ctl.lock:
            before = versions(iat)
            if iat.vars['batch'].get() == 'Overview' and not iat.vars['overview_multi_classes'].get():
                iat.visible_classes = {lab}
            elif lab in iat.visible_classes:
                iat.visible_classes.discard(lab)
            else:
                iat.visible_classes.add(lab)
            iat._draw_scatter()
            return respond(iat, before, lang, theme)
