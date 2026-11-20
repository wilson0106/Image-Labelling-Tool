#!/usr/bin/env python3
"""Dialog contents: training results, settings, about, previous log, confirmations."""
from __future__ import annotations

import sys

import numpy as np
from dash import dcc, html

from training_evaluation import load_training_evaluation
from training_results import TEXT as RESULTS_TEXT

from .. import components as C
from ..texts import t
from ..theme import plot_layout
from .execution import act


def confirm(lang, message, yes_action):
    return C.modal_card(
        t(lang, 'stop') if yes_action == 'stop_yes' else t(lang, 'quit'),
        html.P(message, className='big-msg'),
        act('modal_close'),
        actions=[C.btn(t(lang, 'no'), act('modal_close')),
                 C.btn(t(lang, 'yes'), act(yes_action), 'dan')],
    )


def existing_model(lang, state, model_path):
    """Overwrite / resume / cancel when ``model_<project>`` already exists."""
    msg = t(lang, 'model_exists_msg').format(path=model_path)
    if state == 'partial':
        msg += '\n' + t(lang, 'model_exists_partial_hint')
    actions = [C.btn(t(lang, 'model_exists_cancel'), act('modal_close'))]
    if state == 'partial':
        actions.append(C.btn(t(lang, 'model_exists_resume'), act('train_resume')))
    actions.append(C.btn(t(lang, 'model_exists_overwrite'), act('train_overwrite'), 'dan'))
    return C.modal_card(t(lang, 'model_exists_title'), html.P(msg, className='big-msg'), act('modal_close'),
                        actions=actions)


def info(lang, title, text, extra=None):
    body = [html.Div(text, className='note')]
    if extra is not None:
        body.insert(0, extra)
    return C.modal_card(title, body, act('modal_close'),
                        actions=[C.btn(t(lang, 'close'), act('modal_close'))])


def old_log(lang, name, text):
    return C.modal_card(f"{t(lang, 'old_log_title')} · {name}",
                        html.Pre(text, className='log', style={'height': '62vh', 'whiteSpace': 'pre-wrap'}),
                        act('modal_close'), wide=True,
                        actions=[C.btn(t(lang, 'close'), act('modal_close'))])


def settings(ctl, lang):
    return info(lang, t(lang, 'settings'),
                t(lang, 'settings_text').format(python=sys.executable, src=ctl.src_dir, root=ctl.project_root))


def about(lang):
    logo = html.Img(src='/branding/logo_global.png', style={'maxWidth': '420px', 'width': '100%'})
    return info(lang, t(lang, 'about'), t(lang, 'about_text'), extra=logo)


def matrix_figure(matrix, names, mode, lang, accent='#22d3ee'):
    arr = np.asarray(matrix, dtype=float)
    rt = RESULTS_TEXT['en' if lang == 'en' else 'pt']
    if mode == 'normalized':
        sums = arr.sum(axis=1, keepdims=True)
        z = np.divide(arr, sums, out=np.zeros_like(arr), where=sums > 0) * 100.0
        text = [[f'{v:.1f}' for v in row] for row in z]
        hover = '%{y} → %{x}<br>%{z:.1f}%<extra></extra>'
    else:
        z = arr
        text = [[f'{int(v)}' for v in row] for row in z]
        hover = '%{y} → %{x}<br>%{z:.0f}<extra></extra>'
    n = len(names)
    show_text = n <= 30
    fig = {
        'data': [{
            'type': 'heatmap', 'z': z.tolist(), 'x': list(names), 'y': list(names),
            'colorscale': [[0, '#071014'], [0.12, 'rgba(34,211,238,.35)'], [1, accent]],
            'xgap': 1, 'ygap': 1, 'hovertemplate': hover,
            'text': text if show_text else None, 'texttemplate': '%{text}' if show_text else None,
            'textfont': {'size': 10 if n > 15 else 12},
            'colorbar': {'thickness': 10, 'tickfont': {'color': '#a3adb5'}},
        }],
        'layout': plot_layout(
            margin={'l': 140, 'r': 20, 't': 10, 'b': 120},
            xaxis={'title': {'text': rt['pred_class']}, 'tickangle': -40, 'showgrid': False, 'type': 'category'},
            yaxis={'title': {'text': rt['true_class']}, 'autorange': 'reversed', 'showgrid': False, 'type': 'category'},
        ),
    }
    return fig


def training_results(lang, output_dir, project):
    rt = RESULTS_TEXT['en' if lang == 'en' else 'pt']
    matrix, names, m = load_training_evaluation(output_dir, project)
    tiles = html.Div([
        C.tile(rt['accuracy'], f"{100 * m['accuracy']:.2f}%", color='var(--ok)'),
        C.tile(rt['macro_f1'], f"{m['macro_f1']:.4f}"),
        C.tile(rt['macro_recall'], f"{m['macro_recall']:.4f}", color='var(--accent-2)'),
        C.tile(rt['macro_precision'], f"{m['macro_precision']:.4f}", color='var(--accent-2)'),
        C.tile(rt['bias'], f"{m['mean_abs_bias_pp']:.2f} p.p.", rt['ideal_zero'], color='var(--warn)'),
        C.tile(rt['samples'], f"{m['n_samples']:,}".replace(',', '.'), f"{rt['classes_n']}: {m['n_classes']}"),
    ], className='tiles t3')
    loss = (html.Div(f"{rt['test_loss']}: {float(m['test_loss']):.6f}", className='hint')
            if 'test_loss' in m else None)
    rows = [(r['class'], r['support'], r['predicted'], f"{r['precision']:.4f}", f"{r['recall']:.4f}",
             f"{r['f1']:.4f}", f"{r['bias_pp']:+.2f}") for r in m['per_class']]
    table = C.table([rt['class'], rt['support'], rt['predicted'], rt['precision'], rt['recall'], rt['f1'],
                     rt['bias_pp']], rows, num_cols=(1, 2, 3, 4, 5, 6), class_name='tall')
    height = max(420, min(900, 26 * len(names) + 180))
    body = [
        html.Div(f"{rt['subtitle']}  •  {project}", className='hint'),
        tiles, loss,
        C.hud(rt['matrix'], [
            html.Div([C.seg('res-mode', [('normalized', rt['normalized']), ('counts', rt['count'])], 'normalized'),
                      html.Span(rt['matrix_help'], className='hint')], className='row'),
            dcc.Store(id='res-store', data={'matrix': np.asarray(matrix).tolist(), 'names': list(names)}),
            C.graph('res-matrix', height, matrix_figure(matrix, names, 'normalized', lang)),
        ]),
        C.hud(rt['classes'], [table, C.hint(rt['bias_help'])], variant='alt'),
        C.hud(rt['summary'], [C.note(rt['macro_help']), C.note(rt['bias_help'])]),
    ]
    return C.modal_card(rt['title'], body, act('modal_close'), wide=True,
                        actions=[C.btn(rt['open_folder'], act('res_open_folder')),
                                 C.btn(t(lang, 'close'), act('modal_close'))])
