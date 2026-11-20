#!/usr/bin/env python3
"""Execution panel shared by the Training and Classification pages."""
from __future__ import annotations

from dash import html

from .. import components as C
from ..texts import t


def act(name):
    return {'type': 'act', 'name': name}


def dyn(name):
    return {'type': 'dyn', 'name': name}


def initial_log(ctl):
    lines = list(ctl._log)[-1500:]
    children = [
        html.Div([html.Span(f'[{stamp}]', className='t'), ' ', html.Span(text, className=tag)])
        for _seq, stamp, tag, text in lines
    ]
    seq = lines[-1][0] if lines else 0
    return html.Div(children, id='exec-log', className='log',
                    **{'data-seq': str(seq), 'data-epoch': str(ctl.log_epoch)})


def panel(ctl, lang, task):
    s = ctl.exec_state()
    stages = html.Div(
        [html.Div([html.Small('✓' if st == 'done' else '—' if st == 'skipped' else str(i + 1)), html.B(label)],
                  className=f'stage {st}') for i, (label, st) in enumerate(s['stages'])],
        id='exec-stages', className='pipe')
    if task == 'training':
        metrics = html.Div([
            C.hud(t(lang, 'loss_history'), C.plot_div('chart-loss', 210), class_name='inner'),
            C.hud(t(lang, 'validation_history'), C.plot_div('chart-valid', 210), variant='alt', class_name='inner'),
        ], className='stack')
    else:
        metrics = html.Div([
            C.tile(v[0], v[1], color=col, id=f'm-{k}')
            for (k, v), col in zip(s['metrics'].items(),
                                   ['var(--accent)', 'var(--accent-2)', 'var(--ok)', 'var(--warn)'])
        ], className='tiles t2')
    return C.hud(
        t(lang, 'execution'),
        [
            stages,
            html.Div([html.Span(t(lang, 'general_progress'), className='hint'),
                      html.Span(f"{round(s['pct'])}%", id='exec-pct', className='push hint')], className='row'),
            html.Div(html.I(id='exec-bar', style={'width': f"{s['pct']}%"}), className='bar'),
            metrics,
            html.Div([html.H3(t(lang, 'logs'), className='sub-h'),
                      C.btn(t(lang, 'clear'), act('clear_log'), 'sm', class_name='push')], className='row'),
            html.Div(s['live'], id='exec-live', className='live'),
            initial_log(ctl),
            html.Div([
                C.btn('■ ' + t(lang, 'stop'), act('stop_ask'), 'dan stop-btn', disabled=not s['running']),
                C.btn(t(lang, 'open_output'), act('open_output'), class_name='push'),
            ], className='row'),
        ],
        meta=html.Span([
            html.Span(s['status'], id='exec-status', className='status', style={'color': s['status_color']}),
            html.Span(s['elapsed'], id='exec-elapsed'),
        ], className='row'),
        class_name='sticky',
    )
