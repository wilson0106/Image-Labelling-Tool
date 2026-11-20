#!/usr/bin/env python3
"""IAT shell: project bar, operation strip, sub-tab routing, shared callbacks.

Each sub-tab lives in its own module (``iat_<tab>.py``) exposing
``render(iat, lang, theme)`` and optionally ``on_show(iat)`` and
``register(app, ctl, get_iat)``.
"""
from __future__ import annotations

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from project_settings import format_pixel_size

from .. import components as C
from ..texts import t
from ..theme import sanitize
from . import (iat_clustering, iat_cycle, iat_export, iat_gallery, iat_logs, iat_scatter, iat_segmentation,
               iat_temporal)
from .execution import dyn

TABS = {
    'scatter': iat_scatter, 'gallery': iat_gallery, 'temporal': iat_temporal,
    'clustering': iat_clustering, 'model_cycle': iat_cycle, 'segmentation': iat_segmentation,
    'export': iat_export, 'logs': iat_logs,
}


def iact(name, **extra):
    return {'type': 'iact', 'name': name, **extra}


def iv(key):
    return {'type': 'iv', 'key': key}


def ivs(key):
    return {'type': 'ivs', 'key': key}


def ivb(key):
    return {'type': 'ivb', 'key': key}


def versions(iat):
    s = iat.op_state()
    return [s['view'], s['scatter'], s['gallery'], s['cluster'], s['seg'], s['web'], s['tab']]


def iat_input(iat, key, debounce=True, class_name='inp', alt=False, **kw):
    # ``alt`` gives a second field bound to the same variable (unique DOM id).
    return dcc.Input(id={'type': 'ivx', 'key': key} if alt else iv(key), value=str(iat.vars[key].get()),
                     debounce=debounce, className=class_name,
                     autoComplete='off', spellCheck=False, **kw)


def iat_switch(iat, key, label):
    return C.switch(ivb(key), label, bool(iat.vars[key].get()))


def iat_select(iat, key, options, clearable=False, placeholder=None):
    return dcc.Dropdown(id=ivs(key), options=[{'label': lbl, 'value': val} for val, lbl in options],
                        value=iat.vars[key].get(), clearable=clearable, placeholder=placeholder,
                        className='dd')


def render_body(iat, lang, theme):
    """Project bar + operation strip + current sub-tab (re-rendered together)."""
    mod = TABS.get(iat.current_tab, iat_scatter)
    iat.rendered_versions = versions(iat)
    try:
        content = mod.render(iat, lang, theme)
    except Exception as exc:  # never leave the tab blank silently
        iat._log(f'Falha ao montar a aba {iat.current_tab}: {exc}', 'error')
        content = C.hud('Erro', C.note(str(exc)))
    bar = project_bar(iat, lang) if iat.current_tab != 'model_cycle' else None
    return [bar, op_strip(iat, lang), content]


def project_bar(iat, lang):
    csvs = iat.project.csv_names() if iat.project else []
    batch_opts = [{'label': 'Overview', 'value': 'Overview'}] + [{'label': c, 'value': c} for c in csvs]
    count = len(iat.current_df) if iat.current_df is not None else 0
    mode = 'Overview' if iat.vars['batch'].get() == 'Overview' else 'Batch'
    return html.Div([
        html.Div([
            html.Label(iat.tr('project_folder'), className='hint'),
            html.Div([
                iat_input(iat, 'project'),
                C.btn('…', iact('browse_project'), 'sm icon', title=iat.tr('browse')),
                C.btn(iat.tr('reload') if iat.project else ('Carregar' if lang == 'pt' else 'Load'),
                      iact('load_project'), 'sm pri'),
            ], className='browse'),
        ], className='pb-cell grow'),
        html.Div([html.Label(iat.tr('batch'), className='hint'),
                  dcc.Dropdown(id=ivs('batch'), options=batch_opts, value=iat.vars['batch'].get(), clearable=False,
                               className='dd', disabled=not iat.project)], className='pb-cell w-batch'),
        html.Div([html.Label(iat.tr('mode'), className='hint'), html.B(mode if iat.project else '—')], className='pb-cell'),
        html.Div([html.Label(iat.tr('pixel_size'), className='hint'),
                  iat_input(iat, 'pixel_size', class_name='inp sm', style={'width': '96px'})], className='pb-cell'),
        html.Div([html.Label(iat.tr('images'), className='hint'),
                  html.B(f'{count:,}'.replace(',', '.') if iat.project else '—', style={'color': 'var(--accent)'})],
                 className='pb-cell'),
    ], className='hud pbar')


def op_strip(iat, lang):
    s = iat.op_state()
    return html.Div([
        html.Span(s['status'], id='iat-status', className='status'),
        html.Span(s['operation'], id='iat-op', className='hint op-text'),
        html.Div(html.I(id='iat-bar', style={'width': f"{s['pct']}%"}), className='bar grow'),
        html.Span(s['pct_text'], id='iat-pct', className='hint'),
        html.Span(s['elapsed'], id='iat-elapsed', className='hint mono'),
        C.btn('■ ' + iat.tr('stop'), iact('stop_operation'), 'sm dan iat-stop-btn', disabled=not s['running']),
        C.btn(iat.tr('open_folder'), iact('open_project_folder'), 'sm'),
    ], className='opstrip')


def layout(ctl, iat, lang, tab, theme):
    with ctl.lock:
        if tab in TABS and tab != iat.current_tab:
            iat.show_tab(tab)
        hook = getattr(TABS.get(iat.current_tab), 'on_show', None)
        if hook:
            hook(iat)
        body = render_body(iat, lang, theme)
    return html.Div(body, id=dyn('iat-body'), className='screen iat')


def _body_outputs(idx, content):
    specs = ctx.outputs_list
    # A single wildcard Output gives a flat list of specs; several Outputs give one list each.
    if not specs or not isinstance(specs[0], dict):
        specs = specs[idx]
    return [content if s['id']['name'] == 'iat-body' else no_update for s in specs]


# Side effects that the classic widgets triggered when a variable changed.
def _after_var(iat, key):
    tab = iat.current_tab
    if key == 'batch':
        iat.change_batch()
    elif key == 'scatter_color_mode':
        iat._refresh_scatter_class_selector(reset=True)
        iat._draw_scatter()
    elif key in ('name_filter_stains', 'show_hulls', 'class_axis_mode', 'class_axis_scale'):
        iat._draw_scatter()
    elif key == 'hide_relabel':
        if tab == 'gallery':
            iat._gallery_update_in_place(reset_page=True)
        else:
            iat._draw_scatter()
    elif key == 'overview_multi_classes':
        iat._on_overview_multi_toggle()
    elif key.startswith('review_use_') or key == 'review_threshold':
        iat._recompute_review_priority(refresh=True)
    elif key == 'page_size':
        iat._gallery_page_size_changed()
    elif key in ('conf_min', 'conf_max', 'var_min', 'var_max', 'order', 'order2', 'order3'):
        iat._gallery_update_in_place(reset_page=True)
    elif key == 'gallery_outline':
        iat._refresh_visible_thumbnails()
    elif key == 'similarity_mode':
        if getattr(iat, '_gallery_similarity_active', False):
            iat._clear_similarity_sort()
    elif key == 'seg_method':
        if tab == 'gallery':
            if getattr(iat, '_gallery_similarity_active', False) and iat.vars['similarity_mode'].get() == 'pca':
                iat._clear_gallery_similarity_state(clear_reference=False, refresh=False)
            iat._gallery_update_in_place(reset_page=True)
        else:
            iat._update_seg_param_state()
            iat._seg_update_method_doc()
            iat._seg_refresh_preview()
    elif key in ('seg_scope', 'seg_outline', 'seg_radius'):
        if key == 'seg_scope':
            iat._seg_preview_pos = 0
        iat._seg_refresh_preview()
    elif key == 'pixel_size':
        iat._persist_project_pixel_size(silent=True)
        cycle_px = getattr(iat, '_cycle_pixel_size', None)
        if cycle_px is not None:
            # The integrated cycle's default follows the project calibration.
            cycle_px.set(str(iat.vars['pixel_size'].get()))
        iat._seg_refresh_preview()
    elif key == 'cluster_cut':
        iat._schedule_cluster_view()
    elif key in ('cluster_metric', 'cluster_linkage'):
        iat.web_version = getattr(iat, 'web_version', 0) + 1
    elif key == 'export_all_classes':
        iat._export_toggle_all_classes()
        iat.web_version = getattr(iat, 'web_version', 0) + 1
    elif key in ('export_scope', 'export_class_search'):
        iat.web_version = getattr(iat, 'web_version', 0) + 1
    elif key == 'training_scope':
        iat._training_scope_changed()
        iat_export.on_show(iat)
        iat.web_version = getattr(iat, 'web_version', 0) + 1
    elif key == 'training_all_classes':
        iat._training_toggle_all()
        iat.web_version = getattr(iat, 'web_version', 0) + 1


def register(app, ctl, get_iat):
    def respond(iat, before, lang, theme, idx=0):
        if versions(iat) == before:
            # One entry per matched component (a bare no_update is invalid here).
            return _body_outputs(idx, no_update)
        return _body_outputs(idx, render_body(iat, lang or 'pt', sanitize(theme or {})))

    @app.callback(
        Output(dyn(ALL), 'children', allow_duplicate=True),
        Input(iv(ALL), 'value'), Input(ivs(ALL), 'value'), Input(ivb(ALL), 'value'),
        Input({'type': 'ivx', 'key': ALL}, 'value'),
        State('lang-store', 'data'), State('theme-store', 'data'), prevent_initial_call=True,
    )
    def iat_sync(_a, _b, _c, _d, lang, theme):
        trig = ctx.triggered_id
        if not isinstance(trig, dict):
            raise PreventUpdate
        value = ctx.triggered[0]['value']
        key = trig['key']
        iat = get_iat()
        with ctl.lock:
            var = iat.vars.get(key)
            if var is None:
                raise PreventUpdate
            if trig['type'] == 'ivb':
                value = bool(value)
            elif key == 'pixel_size' and value not in (None, ''):
                value = str(value)
            if value is None:
                value = ''
            if var.get() == value or (trig['type'] != 'ivb' and str(var.get()) == str(value)):
                raise PreventUpdate
            before = versions(iat)
            var.set(value)
            _after_var(iat, key)
            return respond(iat, before, lang, theme)

    @app.callback(
        Output(dyn(ALL), 'children', allow_duplicate=True),
        Output('modal-host', 'children', allow_duplicate=True),
        Input(iact(ALL), 'n_clicks'), State('lang-store', 'data'), State('theme-store', 'data'),
        prevent_initial_call=True,
    )
    def iat_action(_clicks, lang, theme):
        trig = ctx.triggered_id
        if not isinstance(trig, dict) or not ctx.triggered[0]['value']:
            raise PreventUpdate
        name = trig['name']
        iat = get_iat()
        modal = no_update
        with ctl.lock:
            before = versions(iat)
            handler = ACTIONS.get(name)
            if handler is None:
                for mod in TABS.values():
                    handler = getattr(mod, 'ACTIONS', {}).get(name)
                    if handler:
                        break
            if handler is None:
                raise PreventUpdate
            res = handler(iat, lang or 'pt')
            if isinstance(res, dict) and 'modal' in res:
                modal = res['modal']
            elif name in ('browse_project', 'load_project'):
                iat.web_version = getattr(iat, 'web_version', 0) + 1
            body = respond(iat, before, lang, theme)
        return body, modal

    @app.callback(
        Output(dyn(ALL), 'children', allow_duplicate=True), Input('iat-ver-poll', 'data'),
        State('lang-store', 'data'), State('theme-store', 'data'), State('url', 'pathname'),
        prevent_initial_call=True,
    )
    def iat_rerender(_ver, lang, theme, pathname):
        if not (pathname or '').startswith('/iat'):
            raise PreventUpdate
        iat = get_iat()
        with ctl.lock:
            if versions(iat) == getattr(iat, 'rendered_versions', None):
                raise PreventUpdate
            return _body_outputs(0, render_body(iat, lang or 'pt', sanitize(theme or {})))

    for mod in TABS.values():
        if hasattr(mod, 'register'):
            mod.register(app, ctl, get_iat, respond)


def _browse_project(iat, lang):
    iat.browse_project()


def _load_project(iat, lang):
    iat.load_project()


ACTIONS = {
    'browse_project': _browse_project,
    'load_project': _load_project,
    'stop_operation': lambda iat, lang: iat.stop_operation(),
    'open_project_folder': lambda iat, lang: iat.open_project_folder(),
}
