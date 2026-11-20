#!/usr/bin/env python3
"""Training page (train.py) — same fields and rules as the classic page."""
from __future__ import annotations

from dash import dcc, html

from training_dataset import SPLITS

from .. import components as C
from ..texts import t
from .execution import act, dyn, panel

SPLIT_LABEL = {'train': 'train_short', 'valid': 'valid_short', 'test': 'test_short'}


def _fmt(n):
    return f'{int(n):,}'.replace(',', '.')


def class_section(ctl, lang):
    """Dynamic parts of the dataset-classes editor, keyed by ``dyn`` name."""
    ds = ctl.dataset_state()
    rows = ctl.class_rows()
    mode = ds['distribution_mode']
    dist_box = html.Div([
        html.Div([
            html.Span(t(lang, 'distribution_by'), className='hint'),
            C.seg({'type': 'dsel', 'name': 'dist-mode'},
                  [('percent', t(lang, 'distribution_percent')), ('number', t(lang, 'distribution_number'))], mode),
        ], className='row'),
        html.Div([
            html.Span(t(lang, 'default_distribution'), className='hint'),
            *[html.Span([
                html.Span(t(lang, SPLIT_LABEL[s]), className='hint'),
                dcc.Input(id={'type': 'dist-def', 'split': s}, value=ds['defaults'][s], debounce=True,
                          className='inp sm', style={'width': '64px'}),
            ], className='row', style={'gap': '4px'}) for s in SPLITS],
            html.Span(ds['unit'], className='chip'),
            C.btn(t(lang, 'apply_distribution_all'), act('apply_dist'), 'sm'),
        ], className='row push'),
    ], className='row distbar')

    head = [t(lang, 'use_in_training'), t(lang, 'original_class'), t(lang, 'total_images'),
            t(lang, 'train_short'), t(lang, 'valid_short'), t(lang, 'test_short'),
            t(lang, 'unused_images'), t(lang, 'final_class')]
    body = []
    for r in rows:
        name = r['name']
        body.append(html.Tr([
            html.Td(dcc.Checklist(id={'type': 'cls-use', 'cls': name}, value=['on'] if r['use'] else [],
                                  options=[{'label': '', 'value': 'on'}], inputClassName='chk'), className='ctr'),
            html.Td(name),
            html.Td(_fmt(r['total']), className='num'),
            *[html.Td(dcc.Input(id={'type': 'cls-split', 'cls': name, 'split': s}, value=r['splits'][s],
                                className='inp sm' + (' err' if r['error'] else ''), debounce=False,
                                style={'width': '78px'}), className='ctr') for s in SPLITS],
            html.Td(r['unused'], id={'type': 'cls-unused', 'cls': name}, className='num'),
            html.Td(dcc.Input(id={'type': 'cls-target', 'cls': name}, value=r['target'], className='inp',
                              debounce=True, style={'minWidth': '150px'})),
        ], className='' if r['use'] else 'off'))
    if rows:
        table = html.Div(html.Table([
            html.Thead(html.Tr([html.Th(h, className='ctr' if i != 1 and i != 7 else None)
                                for i, h in enumerate(head)])),
            html.Tbody(body)]), className='tbl-wrap tall')
    else:
        table = C.hint(t(lang, 'dataset_classes_wait'))
    return {
        'ds-detect': (ds['detection'], {'color': ds['detection_color'], 'fontWeight': 600}),
        'dist-box': (dist_box, None),
        'cls-table': (table, None),
        **preview_section(ctl, lang, ds),
    }


def preview_section(ctl, lang, ds=None):
    ds = ds or ctl.dataset_state()
    head = [t(lang, 'model_class'), t(lang, 'total_images'), t(lang, 'train_short'),
            t(lang, 'valid_short'), t(lang, 'test_short'), t(lang, 'source_classes')]
    preview = C.table(head, ds['preview'], num_cols=(1, 2, 3, 4, 5), class_name='tall') if ds['preview'] else None
    return {
        'preview-table': (preview, None),
        'preview-summary': (ds['summary'], {'color': ds['summary_color'], 'fontWeight': 600}),
    }


def layout(ctl, lang):
    v = ctl.form_values()
    dyn_parts = class_section(ctl, lang)

    def d(name, tag=html.Div, class_name=''):
        child, style = dyn_parts.get(name, (None, None))
        return tag(child, id=dyn(name), style=style, className=class_name)

    def f(label_key, key):
        return (t(lang, label_key), C.var_input(key, v[key]))

    def sel(label_key, key, options):
        return (t(lang, label_key), C.seg({'type': 'vs', 'key': key}, [(o, o) for o in options], v[key]))

    left = html.Div([
        html.Div([
            C.hud(t(lang, 'dataset'), [
                C.form([(t(lang, 'dataset_root'),
                         C.browse_input('train_dataset', v['train_dataset'], act('pick_train_dataset'),
                                        t(lang, 'browse')))]),
                C.pre(t(lang, 'dataset_structure_help')),
                d('ds-detect'),
            ]),
            C.hud(t(lang, 'output'), C.form([
                (t(lang, 'output_folder'),
                 C.browse_input('train_output', v['train_output'], act('pick_train_output'), t(lang, 'browse'))),
                f('project_name', 'train_project'),
            ]), variant='alt'),
        ], className='g g-2'),
        C.hud(t(lang, 'dataset_classes'), [
            C.note(t(lang, 'dataset_classes_tip')),
            html.Div([
                C.btn(t(lang, 'restore_class_names'), act('reset_names'), 'sm'),
                C.btn(t(lang, 'refresh_dataset_classes'), act('refresh_ds'), 'sm', class_name='push'),
            ], className='row'),
            d('dist-box'),
            d('cls-table'),
        ]),
        C.hud(t(lang, 'model_classes_preview'), [
            C.note(t(lang, 'model_classes_preview_tip')),
            d('preview-table'),
            d('preview-summary'),
        ], variant='alt'),
        C.hud(t(lang, 'train_parameters'), [
            C.form([
                f('epochs', 'numEpochs'), f('batch_size', 'batchSize'), f('learning_rate', 'lr'),
                sel('loss', 'lossName', ['focal', 'ce']), f('focal_gamma', 'focalGamma'), f('patience', 'patience'),
                sel('validation_metric', 'validMetric', ['valid_loss', 'valid_acc', 'valid_macro']),
                f('seed', 'seed'),
                f('pixel_size', 'pixelSize'),
            ], class_name='f3'),
            C.hint(t(lang, 'pixel_size_train_tip')),
        ]),
        C.hud(t(lang, 'train_recipe'), [
            C.form([
                sel('input_normalize', 'inputNormalize', ['imagenet', 'none']),
                sel('input_resize', 'inputResize', ['pad', 'stretch']),
                sel('augment', 'augment', ['T', 'F']),
                sel('optimizer', 'optimizer', ['adamw', 'adam']), f('weight_decay', 'weightDecay'),
                sel('scheduler', 'scheduler', ['plateau', 'cosine', 'none']),
                f('warmup_epochs', 'warmupEpochs'), f('head_lr_mult', 'headLrMult'),
                f('label_smoothing', 'labelSmoothing'), f('stochastic_depth', 'stochasticDepth'),
            ], class_name='f3'),
            html.Div([
                C.btn(t(lang, 'recipe_recommended'), act('recipe_recommended'), 'sm'),
                C.btn(t(lang, 'recipe_legacy'), act('recipe_legacy'), 'sm'),
            ], className='row'),
            C.hint(t(lang, 'train_recipe_tip')),
        ]),
        C.hud(t(lang, 'advanced'), [
            C.form([
                f('train_folder', 'trainFolder'), f('valid_folder', 'validFolder'), f('test_folder', 'testFolder'),
                f('workers', 'numWorkers'), sel('resume', 'readModel', ['T', 'F']),
                (t(lang, 'amp'), html.Span(t(lang, 'amp_fixed'), className='hint')),
            ], class_name='f3'),
        ], variant='alt'),
        C.hud(t(lang, 'description'), C.note(t(lang, 'train_desc'))),
        html.Div([
            C.btn(t(lang, 'save_config'), act('save_config')),
            C.btn(t(lang, 'load_config'), act('load_config')),
            C.btn(t(lang, 'training_results'), act('results')),
            C.btn('▶ ' + t(lang, 'start_training'), act('start_training'), 'pri start-btn', class_name='push',
                  disabled=ctl.proc is not None),
        ], className='row'),
    ], className='stack')
    return html.Div([left, panel(ctl, lang, 'training')], className='g g-cfg screen')
