#!/usr/bin/env python3
"""Appearance page: theme presets, colours, fonts and shape."""
from __future__ import annotations

from dash import dcc, html

from .. import components as C
from ..texts import t
from ..theme import FONT_CHOICES, PRESETS


def color_row(lang, key, label_key, value):
    return [
        html.Label(t(lang, label_key)),
        dcc.Input(id=f'ap-{key}', type='color', value=value, debounce=False, className='color-in'),
    ]


def layout(ctl, lang, theme):
    presets = html.Div([
        html.Button([
            html.I(style={'background': f'linear-gradient(90deg,{a},{b})'}),
            html.B(name), html.Small(f'{a} · {b}'),
        ], id={'type': 'preset', 'i': i}, n_clicks=0,
            className='pre' + (' on' if (a, b) == (theme['accent'], theme['accent2']) else ''))
        for i, (name, a, b) in enumerate(PRESETS)
    ], className='presets')
    colors = html.Div(
        color_row(lang, 'accent', 'accent', theme['accent'])
        + color_row(lang, 'accent2', 'accent2', theme['accent2'])
        + color_row(lang, 'bg', 'background', theme['bg'])
        + color_row(lang, 'ink', 'text_color', theme['ink']),
        className='colors')
    fonts = [(f, f) for f in FONT_CHOICES]
    shape = C.form([
        (t(lang, 'font_display'), C.seg('ap-font_display', fonts, theme['font_display'])),
        (t(lang, 'font_body'), C.seg('ap-font_body', fonts, theme['font_body'])),
        (t(lang, 'glass'), dcc.Input(id='ap-glass', type='range', min=0.2, max=0.95, step=0.05, value=theme['glass'])),
        (t(lang, 'radius'), dcc.Input(id='ap-radius', type='range', min=0, max=24, step=1, value=theme['radius'])),
        (t(lang, 'font_scale'), dcc.Input(id='ap-font_scale', type='range', min=0.8, max=1.4, step=0.05,
                                          value=theme['font_scale'])),
        ('', C.switch('ap-gpu_monitor', t(lang, 'gpu_monitor'), theme['gpu_monitor'])),
    ])
    return html.Div([
        C.hud(t(lang, 'presets'), presets),
        html.Div([
            C.hud(t(lang, 'colors'), [colors, html.Div(id='ap-contrast', className='hint'),
                                      C.hint(t(lang, 'class_colors_note'))]),
            C.hud(t(lang, 'font_shape'), [
                shape,
                html.Div([
                    C.btn(t(lang, 'save_theme'), 'ap-save', 'pri'),
                    C.btn(t(lang, 'reset_theme'), 'ap-reset'),
                    html.Span(id='ap-msg', className='hint'),
                ], className='row'),
            ], variant='alt'),
        ], className='g g-2'),
        C.hud('Prévia' if lang == 'pt' else 'Preview', [
            html.Div([
                C.tile('Acurácia' if lang == 'pt' else 'Accuracy', '94,2 %', color='var(--ok)'),
                C.tile('Época' if lang == 'pt' else 'Epoch', '12 / 40', color='var(--accent)'),
                C.tile('Perda' if lang == 'pt' else 'Loss', '0,214', color='var(--warn)'),
                C.tile('GPU', '62 %', color='var(--accent-2)'),
            ], className='tiles'),
            html.Div([html.Button('Primário', className='btn pri'), html.Button('Secundário', className='btn'),
                      html.Button('Perigo', className='btn dan')], className='row'),
        ]),
    ], className='screen')
