#!/usr/bin/env python3
"""IAT › Logs (operation status and the complete IAT event log)."""
from __future__ import annotations

from dash import html

from .. import components as C

TAG_CLASS = {'normal': 'normal', 'muted': 'muted', 'info': 'progress', 'success': 'success', 'warn': 'warning',
             'error': 'error', 'cluster': 'path'}


def log_box(iat, height=None):
    lines = iat._seq_lines[-1500:]
    kids = [html.Div([html.Span(f'[{stamp}]', className='t'), ' ', html.Span(msg, className=TAG_CLASS.get(tag, 'normal'))])
            for _seq, stamp, tag, msg in lines]
    style = {'height': height} if height else None
    return html.Div(kids, id='iat-log', className='log', style=style,
                    **{'data-seq': str(lines[-1][0] if lines else 0), 'data-epoch': str(iat.log_epoch)})


def render(iat, lang, theme):
    from .iat import iact
    return C.hud(iat.tr('iat_log'), [
        html.Div([C.btn(iat.tr('clear'), iact('clear_iat_log'), 'sm', class_name='push')], className='row'),
        log_box(iat, '64vh'),
    ])


ACTIONS = {'clear_iat_log': lambda iat, lang: (iat.clear_log(), setattr(iat, 'web_version', getattr(iat, 'web_version', 0) + 1))}
