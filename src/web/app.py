#!/usr/bin/env python3
"""ILT web interface (Dash/Plotly) — shell, routing and callbacks.

Runs locally (127.0.0.1) and drives the same workers as the classic
interface through :mod:`web.controller`.  The classic Tk interface stays
available (``./run_gui.sh --classic``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import socket
import threading
import time
import webbrowser
from pathlib import Path

from dash import ALL, ClientsideFunction, Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from flask import abort, send_from_directory

from . import components as C
from .controller import get_controller
from .headless import NOTIFY
from . import iat_core
from .iat_core import get_iat
from .pages import appearance, classification, full_image, iat_bridge, modals, training
from .pages import iat as iat_page
from .pages.iat_logs import TAG_CLASS
from .pages.execution import act, dyn
from .sysmon import get_monitor
from .texts import t
from .theme import DEFAULT_THEME, PRESETS, contrast_ratio, sanitize, save_theme, theme_path, load_theme

ASSETS = Path(__file__).resolve().parent / 'assets'

ROUTES = {
    '/treino': 'training', '/classificacao': 'classification', '/aparencia': 'appearance',
}


def _route(pathname):
    path = (pathname or '/').rstrip('/') or '/'
    if path == '/':
        return 'training', None
    if path.startswith('/iat'):
        tab = path[5:] or 'scatter'
        return 'iat', tab
    return ROUTES.get(path, 'training'), None


def _sidebar(ctl, lang, page, iat_tab, theme):
    def link(href, label, icon, on):
        kids = [html.Img(src=f'/branding/{icon}', alt='')] if icon else []
        return dcc.Link([*kids, html.Span(label)], href=href, className='on' if on else '')

    subs = html.Div([
        dcc.Link(t(lang, key), href=f'/iat/{name}', className='on' if page == 'iat' and iat_tab == name else '')
        for name, key in iat_bridge.IAT_TABS
    ], className='sub')
    gpu = theme.get('gpu_monitor', True)
    sysbox = html.Div([
        html.Div([html.I(), html.Span(ctl.system_status_var.get(), id='sys-status')], className='st'),
        html.Div([
            html.Div(html.Div([html.B('—'), html.Small('GPU')]), id='gauge-a', className='gauge'),
            html.Div(html.Div([html.B('—'), html.Small('VRAM')]), id='gauge-b', className='gauge',
                     style={'--c': 'var(--accent-2)'}),
        ], className='gauges') if gpu else None,
        html.Div([C.btn(t(lang, 'quit'), act('quit_ask'), 'sm dan')], className='sysrow'),
    ], className='sysbox' + (' busy' if ctl.proc is not None else ''))
    mascots = _mascot_files(ctl)
    mascot = html.Div([
        html.Img(id='mascot-img', src=f'/mascot/{random.choice(mascots)}', alt='',
                 title=t(lang, 'mascot_tip'), **{'data-files': json.dumps(mascots)}),
        html.Small(t(lang, 'mascot_tip'), className='hint'),
    ], className='mascot') if mascots else None
    return [
        html.Div([html.Img(src='/branding/icon_global.png', alt=''),
                  html.Div([html.B('ILT Data Extraction'), html.Small('Treino · Classificação · IAT')])],
                 className='brand'),
        html.Div(t(lang, 'tasks'), className='grp'),
        html.Nav([
            link('/treino', t(lang, 'training'), 'icon_training.png', page == 'training'),
            link('/classificacao', t(lang, 'classification'), 'icon_classification.png', page == 'classification'),
            link('/iat/scatter', t(lang, 'iat'), 'icon_iat.png', page == 'iat'),
            subs,
            html.Div(t(lang, 'system'), className='grp'),
            html.A(t(lang, 'old_logs'), id=act('old_logs'), n_clicks=0),
            link('/aparencia', t(lang, 'appearance'), None, page == 'appearance'),
            html.A(t(lang, 'settings'), id=act('settings'), n_clicks=0),
            html.A(t(lang, 'about'), id=act('about'), n_clicks=0),
        ], className='nav'),
        mascot,
        sysbox,
    ]


def _mascot_dir(ctl):
    from platform_utils import resource_dir
    for d in (ctl.project_root / 'assets' / 'mascots', resource_dir() / 'assets' / 'mascots'):
        if d.is_dir() and any(d.glob('*.png')):
            return d
    return None


def _mascot_files(ctl):
    d = _mascot_dir(ctl)
    return sorted(p.name for p in d.glob('*.png')) if d else []


def _dyn_out(values, which):
    """Map ``{name: (children, style)}`` onto the matched ``dyn`` outputs."""
    out = []
    for spec in ctx.outputs_list[which] if isinstance(ctx.outputs_list[which], list) else []:
        name = spec['id']['name']
        if name in values:
            child, style = values[name]
            # allow_duplicate outputs carry a suffix: 'children@<hash>'.
            prop = str(spec['property']).split('@', 1)[0]
            out.append(child if prop == 'children' else (style if style is not None else no_update))
        else:
            out.append(no_update)
    return out


def create_app():
    ctl = get_controller()
    mon = get_monitor()
    theme0 = load_theme(ctl.project_root)
    mon.enabled = bool(theme0['gpu_monitor'])
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    app = Dash(__name__, assets_folder=str(ASSETS), title='ILT Data Extraction & IAT',
               suppress_callback_exceptions=True, update_title=None, serve_locally=True)
    server = app.server
    branding_dir = ctl.project_root / 'assets' / 'branding'

    @server.route('/mascot/<path:name>')
    def _mascot(name):
        d = _mascot_dir(ctl)
        if d is None or '/' in name or not (d / name).is_file():
            abort(404)
        return send_from_directory(d, name, max_age=86400)

    @server.route('/branding/<path:name>')
    def _branding(name):
        if '/' in name or not (branding_dir / name).is_file():
            abort(404)
        return send_from_directory(branding_dir, name, max_age=3600)

    app.layout = html.Div([
        html.Aside(id='sidebar', className='side'),
        html.Main([
            html.Header([
                html.Div([html.Span(id='top-date'), html.Span(id='top-device')], className='l'),
                html.Div(html.Span(id='plate-text'), className='plate'),
                html.Div([
                    html.Span([html.Button('PT', id='lang-pt', n_clicks=0, className='on'), '◉',
                               html.Button('EN', id='lang-en', n_clicks=0)], className='chip'),
                    html.Span([html.Button('A−', id='fs-minus', n_clicks=0),
                               html.Span(id='fs-val'),
                               html.Button('A+', id='fs-plus', n_clicks=0)], className='chip'),
                ], className='r'),
            ], className='top'),
            html.Nav(id='iat-tabs', className='tabs', hidden=True),
            html.Div(id='page'),
        ], className='main'),
        dcc.Location(id='url'),
        dcc.Store(id='lang-store', data=ctl.lang),
        dcc.Store(id='theme-store', data=theme0),
        dcc.Store(id='cs'), dcc.Store(id='exec-store'), dcc.Store(id='rows-store', data=ctl.rows_version),
        dcc.Store(id='labels-store'), dcc.Store(id='sink-a'), dcc.Store(id='sink-b'), dcc.Store(id='sink-c'),
        dcc.Store(id='ack-v'), dcc.Store(id='ack-d'), dcc.Store(id='iat-ver-poll'), dcc.Store(id='gclick'), dcc.Store(id='gsel'), dcc.Store(id='gopen'), dcc.Store(id='sink-g'),
        dcc.Interval(id='tick', interval=600),
        C.modal('modal-host'),
        dcc.Graph(id='plotly-loader', figure={'data': [], 'layout': {}}, style={'display': 'none'}),
    ], className='app')

    # ---------------- client-side renderers ----------------
    app.clientside_callback(ClientsideFunction('ilt', 'clientState'), Output('cs', 'data'), Input('tick', 'n_intervals'))
    app.clientside_callback(ClientsideFunction('ilt', 'render'), Output('sink-a', 'data'), Input('exec-store', 'data'))
    app.clientside_callback(ClientsideFunction('ilt', 'theme'), Output('sink-b', 'data'), Input('theme-store', 'data'))
    app.clientside_callback(ClientsideFunction('ilt', 'labels'), Output('sink-c', 'data'), Input('labels-store', 'data'))

    # ---------------- routing ----------------
    @app.callback(
        Output('page', 'children'), Output('plate-text', 'children'), Output('sidebar', 'children'),
        Output('iat-tabs', 'children'), Output('iat-tabs', 'hidden'), Output('labels-store', 'data'),
        Output('top-date', 'children'), Output('top-device', 'children'), Output('fs-val', 'children'),
        Output('lang-pt', 'className'), Output('lang-en', 'className'),
        Input('url', 'pathname'), Input('lang-store', 'data'), State('theme-store', 'data'),
    )
    def route(pathname, lang, theme):
        lang = lang or 'pt'
        theme = sanitize(theme or {})
        page, tab = _route(pathname)
        with ctl.lock:
            if page in ('training', 'classification') and ctl.proc is None:
                ctl.set_stage_task(page)
            if page == 'training':
                content, plate = training.layout(ctl, lang), t(lang, 'plate_training')
            elif page == 'classification':
                content, plate = classification.layout(ctl, lang), t(lang, 'plate_classification')
            elif page == 'appearance':
                content, plate = appearance.layout(ctl, lang, theme), t(lang, 'plate_appearance')
            else:
                iat = get_iat()
                content = iat_page.layout(ctl, iat, lang, tab, theme)
                tab = iat.current_tab
                plate = f"{t(lang, 'plate_iat')} · {t(lang, dict(iat_bridge.IAT_TABS).get(tab, 'iat_scatter'))}"
            side = _sidebar(ctl, lang, page, tab, theme)
        tabs = [dcc.Link(html.Span(t(lang, key)), href=f'/iat/{name}', className='on' if tab == name else '')
                for name, key in iat_bridge.IAT_TABS]
        labels = {k: t(lang, k) for k in ('train_loss', 'valid_loss', 'valid_acc', 'valid_macro')}
        labels['lang'] = lang
        device = mon.device or ''
        return (content, plate, side, tabs, page != 'iat', labels, time.strftime('%Y-%m-%d'), device,
                f"{int(round(theme['font_scale'] * 100))}%", 'on' if lang == 'pt' else '', 'on' if lang == 'en' else '')

    @app.callback(Output('lang-store', 'data'), Input('lang-pt', 'n_clicks'), Input('lang-en', 'n_clicks'),
                  prevent_initial_call=True)
    def set_lang(_pt, _en):
        lang = 'en' if ctx.triggered_id == 'lang-en' else 'pt'
        with ctl.lock:
            ctl.set_language(lang)
        return lang

    # ---------------- live execution state ----------------
    @app.callback(
        Output('exec-store', 'data'), Output('rows-store', 'data'),
        Output('modal-host', 'children', allow_duplicate=True),
        Output('iat-ver-poll', 'data'), Output('url', 'pathname', allow_duplicate=True),
        Input('cs', 'data'), State('rows-store', 'data'), State('lang-store', 'data'), State('theme-store', 'data'),
        State('url', 'pathname'),
        prevent_initial_call=True,
    )
    def poll(cs, rows_v, lang, theme, pathname):
        if not cs:
            raise PreventUpdate
        payload = {}
        modal = no_update
        with ctl.lock:
            st = ctl.exec_state()
            if cs.get('has_log'):
                epoch, reset, lines = ctl.log_since(int(cs.get('seq') or 0), int(cs.get('epoch') or -1))
                if lines or reset:
                    payload['log'] = {'epoch': epoch, 'reset': reset, 'lines': lines}
            if cs.get('has_charts') and cs.get('charts_v') != st['charts_version']:
                payload['charts'] = ctl.chart_data()
                payload['charts_v'] = st['charts_version']
            rows_out = ctl.rows_version if ctl.rows_version != rows_v else no_update
            req = ctl.results_request
            if req and req[0] > ctl.results_shown_seq:
                ctl.results_shown_seq = req[0]
                try:
                    modal = modals.training_results(lang or 'pt', req[1], req[2])
                except Exception as exc:
                    NOTIFY.push('error', t(lang, 'error'), str(exc))
            iat_ver, url_out, ist = no_update, no_update, None
            iat = iat_core._IAT
            if iat is not None:
                ist = iat.op_state()
                ist.pop('view', None)
                if cs.get('has_iat_log'):
                    epoch, reset, lines = iat.log_since(int(cs.get('iat_seq') or 0), int(cs.get('iat_epoch') or -1))
                    if lines or reset:
                        payload['iat_log'] = {'epoch': epoch, 'reset': reset,
                                              'lines': [(q, s_, TAG_CLASS.get(tg, 'normal'), m) for q, s_, tg, m in lines]}
                on_iat = (pathname or '').startswith('/iat')
                if on_iat:
                    url_tab = (pathname or '/iat/')[5:] or 'scatter'
                    if url_tab != iat.current_tab:
                        url_out = f'/iat/{iat.current_tab}'
                    elif iat_page.versions(iat) != getattr(iat, 'rendered_versions', None):
                        iat_ver = time.time()
        notifs = NOTIFY.since(int(cs.get('notif') or 0))
        state_hash = hashlib.md5((json.dumps([st, ist], sort_keys=True, default=str) + f'|{mon.version}').encode()).hexdigest()
        if not payload and not notifs and state_hash == cs.get('hash'):
            return no_update, rows_out, modal, iat_ver, url_out
        payload.update(state=st, iat=ist, hash=state_hash, notifs=notifs,
                       gauges=mon.data if (theme or {}).get('gpu_monitor', True) else None)
        return payload, rows_out, modal, iat_ver, url_out

    # ---------------- form variables ----------------
    @app.callback(
        Output('ack-v', 'data'),
        Input({'type': 'v', 'key': ALL}, 'value'), Input({'type': 'vs', 'key': ALL}, 'value'),
        Input({'type': 'vb', 'key': ALL}, 'value'),
        prevent_initial_call=True,
    )
    def sync_vars(_v, _vs, _vb):
        trig = ctx.triggered_id
        if not isinstance(trig, dict):
            raise PreventUpdate
        value = ctx.triggered[0]['value']
        if trig['type'] == 'vb':
            value = bool(value)
        with ctl.lock:
            ctl.set_var(trig['key'], value)
        return no_update

    @app.callback(Output('ack-d', 'data'), Input({'type': 'dist-def', 'split': ALL}, 'value'),
                  prevent_initial_call=True)
    def set_default(_values):
        trig = ctx.triggered_id
        if not isinstance(trig, dict):
            raise PreventUpdate
        with ctl.lock:
            ctl.set_distribution_default(trig['split'], ctx.triggered[0]['value'])
        return no_update

    # ---------------- dataset classes editor ----------------
    @app.callback(
        Output(dyn(ALL), 'children', allow_duplicate=True), Output(dyn(ALL), 'style', allow_duplicate=True),
        Input('rows-store', 'data'), Input({'type': 'act', 'name': ALL}, 'n_clicks'),
        Input({'type': 'dsel', 'name': ALL}, 'value'), State('lang-store', 'data'),
        prevent_initial_call=True,
    )
    def render_classes(_rows, _acts, _sel, lang):
        trig = ctx.triggered_id
        with ctl.lock:
            if isinstance(trig, dict) and trig.get('type') == 'act':
                if not ctx.triggered[0]['value']:
                    raise PreventUpdate
                name = trig['name']
                if name == 'refresh_ds':
                    ctl.refresh_dataset_now()
                elif name == 'reset_names':
                    ctl.reset_class_names()
                elif name == 'apply_dist':
                    ctl.apply_distribution_all()
                else:
                    raise PreventUpdate
            elif isinstance(trig, dict) and trig.get('type') == 'dsel':
                ctl.set_distribution_mode(ctx.triggered[0]['value'])
            parts = training.class_section(ctl, lang or 'pt')
            parts.update(classification.model_status(ctl))
        return _dyn_out(parts, 0), _dyn_out(parts, 1)

    @app.callback(
        Output({'type': 'cls-unused', 'cls': ALL}, 'children'),
        Output({'type': 'cls-split', 'cls': ALL, 'split': ALL}, 'className'),
        Output(dyn(ALL), 'children', allow_duplicate=True), Output(dyn(ALL), 'style', allow_duplicate=True),
        Input({'type': 'cls-use', 'cls': ALL}, 'value'),
        Input({'type': 'cls-split', 'cls': ALL, 'split': ALL}, 'value'),
        Input({'type': 'cls-target', 'cls': ALL}, 'value'),
        State('lang-store', 'data'),
        prevent_initial_call=True,
    )
    def edit_class(_use, _split, _target, lang):
        trig = ctx.triggered_id
        if not isinstance(trig, dict):
            raise PreventUpdate
        value = ctx.triggered[0]['value']
        with ctl.lock:
            if trig['type'] == 'cls-use':
                ctl.set_class_field(trig['cls'], 'use', bool(value))
            elif trig['type'] == 'cls-split':
                ctl.set_class_field(trig['cls'], trig['split'], value)
            else:
                ctl.set_class_field(trig['cls'], 'target', value)
            rows = {r['name']: r for r in ctl.class_rows()}
            parts = training.preview_section(ctl, lang or 'pt')
        unused = [rows.get(o['id']['cls'], {}).get('unused', '') for o in ctx.outputs_list[0]]
        classes = ['inp sm' + (' err' if rows.get(o['id']['cls'], {}).get('error') else '')
                   for o in ctx.outputs_list[1]]
        return unused, classes, _dyn_out(parts, 2), _dyn_out(parts, 3)

    # ---------------- actions (buttons) ----------------
    START_ACTIONS = ('start_training', 'start_classification', 'train_overwrite', 'train_resume')
    READONLY_FIELDS = ('model_file', 'labels_file')

    @app.callback(
        Output({'type': 'v', 'key': ALL}, 'value', allow_duplicate=True),
        Output({'type': 'vs', 'key': ALL}, 'value', allow_duplicate=True),
        Output({'type': 'vb', 'key': ALL}, 'value', allow_duplicate=True),
        Output(dyn(ALL), 'children', allow_duplicate=True), Output(dyn(ALL), 'style', allow_duplicate=True),
        Output('modal-host', 'children', allow_duplicate=True),
        Input({'type': 'act', 'name': ALL}, 'n_clicks'),
        State('lang-store', 'data'), State({'type': 'v', 'key': ALL}, 'value'),
        prevent_initial_call=True,
    )
    def action(_clicks, lang, field_values):
        trig = ctx.triggered_id
        if not isinstance(trig, dict) or not ctx.triggered[0]['value']:
            raise PreventUpdate
        lang = lang or 'pt'
        name = trig['name']
        modal = no_update
        vars_changed = False
        dyn_vals = {}
        with ctl.lock:
            if name in START_ACTIONS:
                # Text fields send their value when they lose focus; the click on
                # "Iniciar" may reach the server first.  Start with exactly what
                # the page shows (pixel size, batch size, paths...).
                for item, value in zip(ctx.states_list[1], field_values or []):
                    if item['id']['key'] not in READONLY_FIELDS:
                        ctl.set_var(item['id']['key'], value)
            if name == 'pick_train_dataset':
                ctl.pick_train_dataset(); vars_changed = True
            elif name == 'pick_train_output':
                ctl.pick_dir(ctl.v['train_output']); vars_changed = True
            elif name == 'pick_class_input':
                ctl.pick_dir(ctl.v['class_input']); vars_changed = True
            elif name == 'pick_class_output':
                ctl.pick_dir(ctl.v['class_output_parent']); vars_changed = True
            elif name == 'pick_model_folder':
                ctl.pick_model_folder(); vars_changed = True
                dyn_vals.update(classification.model_status(ctl))
            elif name in ('recipe_recommended', 'recipe_legacy'):
                ctl.apply_training_recipe('legacy' if name == 'recipe_legacy' else 'recommended')
                vars_changed = True
            elif name == 'save_config':
                ctl.save_train_config()
            elif name == 'load_config':
                ctl.load_train_config(); vars_changed = True
            elif name == 'results':
                ctl.open_training_results()
                req = ctl.results_request
                if req and req[0] > ctl.results_shown_seq:
                    ctl.results_shown_seq = req[0]
                    try:
                        modal = modals.training_results(lang, req[1], req[2])
                    except Exception as exc:
                        NOTIFY.push('error', t(lang, 'error'), str(exc))
            elif name == 'start_training' and ctl.proc is None and ctl.training_conflict() is not None:
                # The web message box cannot ask a question: decide in a modal
                # before the reused start_training() runs.
                modal = modals.existing_model(lang, *ctl.training_conflict())
            elif name in ('start_training', 'start_classification', 'train_overwrite', 'train_resume'):
                existing = {'train_overwrite': 'overwrite', 'train_resume': 'resume'}.get(name)
                ctl.start('classification' if name == 'start_classification' else 'training', existing=existing)
                vars_changed = True
                dyn_vals.update(classification.model_status(ctl))
                if existing:
                    modal = []
            elif name == 'stop_ask':
                if ctl.proc is not None:
                    modal = modals.confirm(lang, t(lang, 'confirm_stop'), 'stop_yes')
            elif name == 'stop_yes':
                ctl.stop(); modal = []
            elif name in ('modal_close', 'modal_x'):
                modal = []
            elif name == 'clear_log':
                ctl.clear_log()
            elif name == 'open_output':
                ctl.open_output()
            elif name == 'res_open_folder':
                if ctl.results_request:
                    ctl.open_folder(ctl.results_request[1])
            elif name == 'old_logs':
                res = ctl.open_old_log()
                if res:
                    modal = modals.old_log(lang, *res)
            elif name == 'settings':
                modal = modals.settings(ctl, lang)
            elif name == 'about':
                modal = modals.about(lang)
            elif name == 'quit_ask':
                modal = modals.confirm(lang, t(lang, 'quit_confirm'), 'quit_yes')
            elif name == 'quit_yes':
                if ctl.proc is not None:
                    try:
                        ctl._stopping = True
                        ctl.proc.terminate()
                    except Exception:
                        pass
                threading.Timer(0.6, lambda: os._exit(0)).start()
                modal = C.modal_card(t(lang, 'quit'), html.P('ILT encerrado. Pode fechar esta aba.' if lang == 'pt'
                                                             else 'ILT stopped. You can close this tab.'),
                                     act('modal_close'))
            else:
                raise PreventUpdate
            values = ctl.form_values()
        if vars_changed:
            v_out = [values.get(o['id']['key'], '') for o in ctx.outputs_list[0]]
            vs_out = [values.get(o['id']['key'], '') for o in ctx.outputs_list[1]]
            vb_out = [['on'] if values.get(o['id']['key']) else [] for o in ctx.outputs_list[2]]
        else:
            # Wildcard outputs need one entry per matched component.
            v_out = [no_update] * len(ctx.outputs_list[0])
            vs_out = [no_update] * len(ctx.outputs_list[1])
            vb_out = [no_update] * len(ctx.outputs_list[2])
        return v_out, vs_out, vb_out, _dyn_out(dyn_vals, 3), _dyn_out(dyn_vals, 4), modal

    @app.callback(Output('res-matrix', 'figure'), Input('res-mode', 'value'),
                  State('res-store', 'data'), State('lang-store', 'data'), State('theme-store', 'data'),
                  prevent_initial_call=True)
    def results_mode(mode, store, lang, theme):
        if not store:
            raise PreventUpdate
        return modals.matrix_figure(store['matrix'], store['names'], mode, lang or 'pt',
                                    (theme or DEFAULT_THEME).get('accent', '#22d3ee'))

    # ---------------- appearance ----------------
    ap_keys = ['accent', 'accent2', 'bg', 'ink', 'font_display', 'font_body', 'glass', 'radius', 'font_scale',
               'gpu_monitor']

    @app.callback(
        Output('theme-store', 'data', allow_duplicate=True), Output('ap-contrast', 'children'),
        Output('ap-contrast', 'style'),
        *[Input(f'ap-{k}', 'value') for k in ap_keys], State('theme-store', 'data'), State('lang-store', 'data'),
        prevent_initial_call=True,
    )
    def theme_live(*args):
        values, theme, lang = args[:len(ap_keys)], args[len(ap_keys)], args[len(ap_keys) + 1]
        new = dict(theme or DEFAULT_THEME)
        for k, val in zip(ap_keys, values):
            new[k] = bool(val) if k == 'gpu_monitor' else val
        new = sanitize(new)
        mon.enabled = bool(new['gpu_monitor'])
        ratio = contrast_ratio(new['ink'], new['bg'])
        level = 'AAA' if ratio >= 7 else 'AA' if ratio >= 4.5 else t(lang, 'contrast_low')
        msg = t(lang, 'contrast').format(ratio=f'{ratio:.1f}'.replace('.', ',') if lang == 'pt' else f'{ratio:.1f}',
                                         level=level)
        return new, msg, {'color': 'var(--ok)' if ratio >= 4.5 else 'var(--bad)'}

    @app.callback(Output('ap-accent', 'value', allow_duplicate=True), Output('ap-accent2', 'value', allow_duplicate=True),
                  Input({'type': 'preset', 'i': ALL}, 'n_clicks'), prevent_initial_call=True)
    def preset(_clicks):
        trig = ctx.triggered_id
        if not isinstance(trig, dict) or not ctx.triggered[0]['value']:
            raise PreventUpdate
        _name, a, b = PRESETS[int(trig['i'])]
        return a, b

    @app.callback(*[Output(f'ap-{k}', 'value', allow_duplicate=True) for k in ap_keys],
                  Input('ap-reset', 'n_clicks'), prevent_initial_call=True)
    def reset_theme(n):
        if not n:
            raise PreventUpdate
        d = DEFAULT_THEME
        return [(['on'] if d[k] else []) if k == 'gpu_monitor' else d[k] for k in ap_keys]

    @app.callback(Output('ap-msg', 'children'), Input('ap-save', 'n_clicks'),
                  State('theme-store', 'data'), State('lang-store', 'data'), prevent_initial_call=True)
    def save(n, theme, lang):
        if not n:
            raise PreventUpdate
        save_theme(ctl.project_root, theme or DEFAULT_THEME)
        return t(lang, 'theme_saved').format(path=theme_path(ctl.project_root))

    @app.callback(Output('theme-store', 'data', allow_duplicate=True), Output('fs-val', 'children', allow_duplicate=True),
                  Input('fs-minus', 'n_clicks'), Input('fs-plus', 'n_clicks'), State('theme-store', 'data'),
                  prevent_initial_call=True)
    def font_scale(_m, _p, theme):
        theme = sanitize(theme or {})
        step = -0.1 if ctx.triggered_id == 'fs-minus' else 0.1
        theme['font_scale'] = round(min(1.4, max(0.8, theme['font_scale'] + step)), 2)
        try:
            saved = load_theme(ctl.project_root)
            saved['font_scale'] = theme['font_scale']
            save_theme(ctl.project_root, saved)
        except Exception:
            pass
        return theme, f"{int(round(theme['font_scale'] * 100))}%"

    iat_page.register(app, ctl, get_iat)
    full_image.register_routes(server, get_iat)
    return app


def _free_port(preferred=8050):
    for port in [preferred, *range(8051, 8080)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    return 0


def main(open_browser=True):
    app = create_app()
    port = int(os.environ.get('ILT_WEB_PORT') or _free_port())
    url = f'http://127.0.0.1:{port}/treino'
    print(f'ILT web: {url}  (Ctrl+C para encerrar)', flush=True)
    if open_browser and os.environ.get('ILT_WEB_NO_BROWSER') != '1':
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
