#!/usr/bin/env python3
"""Classification page (main.py) — same fields and rules as the classic page."""
from __future__ import annotations

from dash import html

from .. import components as C
from ..texts import t
from .execution import act, dyn, panel


def model_status(ctl):
    m = ctl.model_state()
    return {'model-status': (m['status'], {'color': m['color'], 'fontWeight': 600})}


def layout(ctl, lang):
    v = ctl.form_values()
    ms = model_status(ctl)['model-status']

    def f(label_key, key, readonly=False):
        return (t(lang, label_key), C.var_input(key, v[key], readonly=readonly))

    left = html.Div([
        html.Div([
            C.hud(t(lang, 'dataset'), [
                C.form([(t(lang, 'dataset_root'),
                         C.browse_input('class_input', v['class_input'], act('pick_class_input'), t(lang, 'browse')))]),
                C.pre('dataset/\n  ├─ classe_A/  imagens\n  ├─ classe_B/  imagens\n  └─ ...'),
            ]),
            C.hud(t(lang, 'output'), C.form([
                (t(lang, 'output_folder'),
                 C.browse_input('class_output_parent', v['class_output_parent'], act('pick_class_output'),
                                t(lang, 'browse'))),
                f('project_name', 'class_project'),
            ]), variant='alt'),
        ], className='g g-2'),
        html.Div([
            C.hud(t(lang, 'model_folder'), [
                C.form([
                    (t(lang, 'model_folder_label'),
                     C.browse_input('model_folder', v['model_folder'], act('pick_model_folder'), t(lang, 'browse'))),
                    ('', html.Div(ms[0], id=dyn('model-status'), style=ms[1])),
                    f('model_file', 'model_file', readonly=True),
                    f('labels_file', 'labels_file', readonly=True),
                ]),
            ]),
            C.hud(t(lang, 'classification_options'), [
                C.form([
                    f('max_batch', 'max_batch'),
                    (t(lang, 'image_mode'), C.seg({'type': 'vs', 'key': 'img_mode'},
                                                  [(o, o) for o in ['auto', 'ln', 'hardlink', 'cp']], v['img_mode'])),
                    f('pixel_size', 'class_pixel_size'),
                ]),
                C.hint(t(lang, 'pixel_size_class_tip')),
                C.hint(t(lang, 'image_mode_tip')),
                html.Div([html.Span(t(lang, 'embeddings'), className='hint'),
                          html.Span(t(lang, 'always_enabled'), style={'color': 'var(--ok)', 'fontWeight': 600})],
                         className='row'),
                C.switch({'type': 'vb', 'key': 'audit_source_labels'}, t(lang, 'audit_source_labels'),
                         v['audit_source_labels']),
                C.hint(t(lang, 'audit_source_tip')),
                C.switch({'type': 'vb', 'key': 'run_class_clustering'}, t(lang, 'run_class_clustering'),
                         v['run_class_clustering']),
                C.hint(t(lang, 'cluster_note')),
            ], variant='alt'),
        ], className='g g-2'),
        C.hud(t(lang, 'description'), C.note(t(lang, 'class_desc'))),
        html.Div([
            C.btn('▶ ' + t(lang, 'start_classification'), act('start_classification'), 'pri start-btn',
                  class_name='push', disabled=ctl.proc is not None),
        ], className='row'),
    ], className='stack')
    return html.Div([left, panel(ctl, lang, 'classification')], className='g g-cfg screen')
