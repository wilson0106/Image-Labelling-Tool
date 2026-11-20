#!/usr/bin/env python3
"""HUD layout primitives for the web interface.

Every page is assembled from these helpers so the look stays consistent and
no page needs its own styling code.  Purely presentational.
"""
from __future__ import annotations

from dash import dcc, html


def hud(title, children, meta=None, variant='', id=None, class_name='', head_extra=None):
    head = [html.H3(title)]
    if head_extra is not None:
        head.append(head_extra)
    if meta is not None:
        head.append(html.Span(meta, className='meta'))
    kwargs = {'id': id} if id else {}
    return html.Section(
        [html.Div(head, className='hh'), *(_as_list(children))],
        className=f'hud {variant} {class_name}'.strip(), **kwargs,
    )


def _as_list(children):
    if children is None:
        return []
    return children if isinstance(children, (list, tuple)) else [children]


def tile(label, value='—', sub='', color='var(--accent)', id=None):
    kwargs = {'id': id} if id else {}
    return html.Div(
        [html.Small(label), html.B(value, className='tv'), html.Em(sub)],
        className='tile', style={'--c': color}, **kwargs,
    )


def btn(label, id, kind='', disabled=False, title=None, class_name=''):
    kwargs = {'title': title} if title else {}
    return html.Button(label, id=id, n_clicks=0, disabled=disabled,
                       className=f'btn {kind} {class_name}'.strip(), **kwargs)


def form(rows, class_name=''):
    items = []
    for label, control in rows:
        items.append(html.Label(label) if isinstance(label, str) else label)
        items.append(control)
    return html.Div(items, className=f'form {class_name}'.strip())


def var_input(key, value, placeholder='', readonly=False, debounce=True, type='text'):
    return dcc.Input(id={'type': 'v', 'key': key}, value=value, placeholder=placeholder, type=type,
                     readOnly=readonly, debounce=debounce, className='inp', autoComplete='off',
                     spellCheck=False)


def browse_input(key, value, browse_id, browse_title):
    return html.Div([
        var_input(key, value),
        html.Button('…', id=browse_id, n_clicks=0, className='btn sm icon', title=browse_title),
    ], className='browse')


def seg(id, options, value, class_name=''):
    return dcc.RadioItems(
        id=id, value=value, options=[{'label': lbl, 'value': val} for val, lbl in options],
        className=f'seg {class_name}'.strip(), inputClassName='seg-in', labelClassName='seg-lb', inline=True,
    )


def switch(id, label, checked):
    return dcc.Checklist(
        id=id, value=['on'] if checked else [], options=[{'label': label, 'value': 'on'}],
        className='sw', inputClassName='sw-in', labelClassName='sw-lb',
    )


def hint(text, class_name=''):
    return html.Div(text, className=f'hint {class_name}'.strip())


def note(text):
    return html.Div(text, className='note')


def pre(text):
    return html.Pre(text, className='pre-block')


def stage_track(id):
    return html.Div(id=id, className='pipe')


def log_box(id, height=None):
    style = {'maxHeight': f'{height}px'} if height else None
    return html.Div(id=id, className='log', style=style)


def table(headers, rows, num_cols=(), class_name=''):
    head = html.Thead(html.Tr([html.Th(h, className='num' if i in num_cols else None)
                               for i, h in enumerate(headers)]))
    body = html.Tbody([
        html.Tr([html.Td(c, className='num' if i in num_cols else None) for i, c in enumerate(row)])
        for row in rows
    ])
    return html.Div(html.Table([head, body]), className=f'tbl-wrap {class_name}'.strip())


def modal(id, children=None):
    return html.Div(id=id, className='modal-host', children=children or [])


def modal_card(title, body, close_id, wide=False, actions=None):
    return html.Div([
        html.Div(className='modal-backdrop'),
        html.Div([
            html.Div([html.H3(title), html.Button('✕', id={'type': 'act', 'name': 'modal_x'}, n_clicks=0,
                                                   className='btn sm icon modal-x', title='Fechar')], className='hh'),
            html.Div(body, className='modal-body'),
            html.Div(actions or [], className='row modal-actions') if actions else None,
        ], className=f'hud modal-card {"wide" if wide else ""}'),
    ], className='modal-wrap')


def graph(id, height, figure=None, class_name=''):
    from .theme import GRAPH_CONFIG
    return dcc.Graph(id=id, figure=figure or {'data': [], 'layout': {}}, config=GRAPH_CONFIG,
                     className=f'plot {class_name}'.strip(), style={'height': f'{height}px'})


def plot_div(id, height):
    """Plain container drawn by the client renderer (Plotly.react)."""
    return html.Div(id=id, className='plot', style={'height': f'{height}px'})
