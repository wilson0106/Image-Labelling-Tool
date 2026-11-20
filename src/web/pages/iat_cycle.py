#!/usr/bin/env python3
"""IAT › Training + Classification (integrated cycle) — original cycle methods."""
from __future__ import annotations

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from project_model_cycle import allocate_counts, batch_id_from_csv

from .. import components as C
from ..theme import GRAPH_CONFIG, STATUS, plot_layout
from .execution import act, dyn
from .iat_logs import log_box

PALETTE = ['#ff6b6b', '#4bd37b', '#55a7ff', '#f2c14e', '#a56cff', '#f78fb3', '#63d4c7', '#ff9f43', '#7b8cff',
           '#8ed081', '#e07a5f', '#9d79bc']


def _p(lang, pt, en):
    return pt if lang == 'pt' else en


def on_show(iat):
    iat.prepare_cycle()
    from ..headless import HeadlessVar
    if getattr(iat, '_cycle_apply_detail', None) is None:
        iat._cycle_apply_detail = HeadlessVar(value='—')
    project = str(iat.project.project_dir) if getattr(iat, 'project', None) is not None else ''
    if getattr(iat, '_cycle_pixel_size', None) is None or getattr(iat, '_cycle_pixel_project', None) != project:
        # Default: the project calibration (same value the IAT uses); taken
        # again whenever another project is opened.
        iat._cycle_pixel_size = HeadlessVar(value=str(iat.vars['pixel_size'].get()))
        iat._cycle_pixel_project = project


def cyv(key):
    return {'type': 'cyv', 'key': key}


def _cinput(iat, attr, **kw):
    return dcc.Input(id=cyv(attr), value=str(getattr(iat, attr).get()), debounce=True, className='inp sm', **kw)


def _summary(iat):
    batches = iat._cycle_selected_batches() if hasattr(iat, '_cycle_batch_vars') else []
    meta = getattr(iat, '_cycle_batch_meta', {})
    batch_images = sum(int(meta.get(b, {}).get('total', 0) or 0) for b in batches)
    enabled = {c: s for c, s in iat._cycle_class_vars.items() if s['enabled'].get()}
    available = sum(s['available'] for s in enabled.values())
    finals = {s['final'].get().strip() for s in enabled.values() if s['final'].get().strip()}
    used = 0
    mode = iat._cycle_distribution_mode.get()
    for cls, s in iat._cycle_class_vars.items():
        try:
            counts = allocate_counts(s['available'], float(s['train'].get().replace(',', '.')),
                                     float(s['valid'].get().replace(',', '.')),
                                     float(s['test'].get().replace(',', '.')), mode)
            n = sum(int(v) for v in counts.values())
            s['unused'].set(str(max(0, s['available'] - n)) if s['enabled'].get() else str(s['available']))
            if s['enabled'].get():
                used += n
        except Exception:
            s['unused'].set('—')
    fmt = lambda v: f'{v:,}'.replace(',', '.')
    return [(_p(iat.lang(), 'Batches (treino)', 'Batches (training)'), fmt(len(batches))),
            (_p(iat.lang(), 'Imagens (treino)', 'Images (training)'), fmt(batch_images)),
            (_p(iat.lang(), 'Classes selecionadas', 'Selected classes'), fmt(len(enabled))),
            (_p(iat.lang(), 'Classes finais', 'Final classes'), fmt(len(finals))),
            (_p(iat.lang(), 'Imagens no dataset', 'Images in dataset'), fmt(used)),
            (_p(iat.lang(), 'Não utilizadas', 'Unused'), fmt(max(0, available - used)))]


def _stage(label, var):
    pct = float(var.get() or 0)
    return html.Div([html.Div([html.B(label, className='small'), html.Span(f'{pct:.0f}%', className='push accent-text')],
                              className='row nowrap'),
                     html.Div(html.I(style={'width': f'{pct}%'}), className='bar')], className='stage-tile')


def _train_chart(iat, theme):
    hist = iat._cycle_train_history or {}
    data = []
    for key, color, name in (('train', '#57b8ff', 'train acc'), ('valid', '#c168ff', 'valid acc')):
        vals = list(hist.get(key, []))
        if vals:
            data.append(dict(type='scatter', mode='lines+markers', x=list(range(1, len(vals) + 1)), y=vals, name=name,
                             line=dict(color=color, width=2, shape='spline'), marker=dict(size=4)))
    layout = plot_layout(theme, margin=dict(l=40, r=10, t=10, b=30), height=180,
                         yaxis=dict(range=[0, 1], tickformat='.0%'), xaxis=dict(title=dict(text='época')))
    return {'data': data, 'layout': layout}


def render(iat, lang, theme):
    from .iat import iact
    on_show(iat)
    sp = iat._cycle_stage_progress
    stages = html.Div([
        _stage(_p(lang, 'Referência', 'Reference'), sp['reference']),
        _stage(_p(lang, 'Treinamento', 'Training'), sp['training']),
        _stage(_p(lang, 'Classificação', 'Classification'), sp['classification']),
        _stage('Clustering', sp['cluster']),
    ], className='tiles')
    q = str(iat._cycle_batch_search.get() or '').strip().lower()
    rows = []
    for name, var in iat._cycle_batch_vars.items():
        if q and q not in name.lower():
            continue
        m = iat._cycle_batch_meta.get(name, {})
        total, validated, pct = m.get('total', 0), m.get('validated', 0), m.get('pct', 0)
        ok = total and validated >= total
        rows.append(html.Tr([
            html.Td('✓' if var.get() else '', className='ctr chkcell'),
            html.Td(batch_id_from_csv(name)), html.Td(f'{total:,}'.replace(',', '.'), className='num'),
            html.Td(f'{validated:,}'.replace(',', '.'), className='num'), html.Td(f'{pct:.0f}%', className='num'),
            html.Td(_p(lang, 'Validado', 'Validated') if ok else _p(lang, 'Pendente', 'Pending'),
                    style={'color': STATUS['ok'] if ok else STATUS['warn']}),
        ], id={'type': 'cyb', 'b': name}, n_clicks=0, className='clickrow' + (' on' if var.get() else '')))
    sel = iat._cycle_selected_batches()
    sel_images = sum(int(iat._cycle_batch_meta.get(b, {}).get('total', 0) or 0) for b in sel)
    batches = C.hud(_p(lang, 'Seleção de batches para treinamento', 'Training batch selection'), [
        C.hint(_p(lang, 'Escolha quais batches já validados serão usados como referência.',
                  'Choose which validated batches will be used as reference.')),
        html.Div([_cinput(iat, '_cycle_batch_search', placeholder=_p(lang, 'Buscar batch…', 'Search batch…')),
                  C.btn(_p(lang, 'Selecionar validados', 'Select validated'), iact('cy_validated'), 'sm'),
                  C.btn(_p(lang, 'Selecionar todos', 'Select all'), iact('cy_all_batches'), 'sm'),
                  C.btn(_p(lang, 'Limpar seleção', 'Clear selection'), iact('cy_no_batches'), 'sm')], className='row'),
        html.Div(html.Table([html.Thead(html.Tr([html.Th(''), html.Th('Batch'), html.Th('Total', className='num'),
                                                 html.Th(_p(lang, 'Validados', 'Validated'), className='num'),
                                                 html.Th('%', className='num'), html.Th('Status')])),
                             html.Tbody(rows)]), className='tbl-wrap tall'),
        html.Div(_p(lang, f'{len(sel)} selecionados • {sel_images:,} imagens', f'{len(sel)} selected • {sel_images:,} images')
                 .replace(',', '.'), className='accent-text small'),
    ])
    cq = str(iat._cycle_class_search.get() or '').strip().lower()
    summary = _summary(iat)
    crow = []
    for i, (cls, s) in enumerate(iat._cycle_class_vars.items()):
        if cq and cq not in cls.lower() and cq not in s['final'].get().lower():
            continue
        on = s['enabled'].get()
        crow.append(html.Tr([
            html.Td(C.btn('✓' if on else '', {'type': 'cyc', 'cls': cls}, 'sm icon' + (' pri' if on else ''))),
            html.Td(html.I(className='swatch', style={'background': PALETTE[i % len(PALETTE)]})),
            html.Td(cls),
            html.Td(dcc.Input(id={'type': 'cyf', 'cls': cls, 'k': 'final'}, value=s['final'].get(), debounce=True,
                              className='inp sm', style={'minWidth': '130px'})),
            html.Td(f"{s['available']:,}".replace(',', '.'), className='num accent-text'),
            *[html.Td(dcc.Input(id={'type': 'cyf', 'cls': cls, 'k': k}, value=s[k].get(), debounce=True,
                                className='inp sm', style={'width': '70px'})) for k in ('train', 'valid', 'test')],
            html.Td(s['unused'].get(), className='num'),
        ], className='' if on else 'off'))
    classes = C.hud(_p(lang, 'Classes e definição do dataset de treinamento', 'Classes and training dataset definition'), [
        html.Div([C.btn(_p(lang, 'Ler classes', 'Read classes'), iact('cy_scan'), 'sm pri'),
                  html.Span(_p(lang, 'Distribuição por classe:', 'Per-class distribution:'), className='hint'),
                  C.seg(cyv('_cycle_distribution_mode'), [('percent', '%'), ('number', 'N')], iat._cycle_distribution_mode.get()),
                  html.Span(_p(lang, 'Aplicar padrão:', 'Default:'), className='hint'),
                  _cinput(iat, '_cycle_default_train', style={'width': '60px'}),
                  _cinput(iat, '_cycle_default_valid', style={'width': '60px'}),
                  _cinput(iat, '_cycle_default_test', style={'width': '60px'}),
                  C.btn(_p(lang, 'Aplicar', 'Apply'), iact('cy_apply_defaults'), 'sm')], className='row'),
        _cinput(iat, '_cycle_class_search', placeholder=_p(lang, 'Buscar classe…', 'Search class…')),
        html.Div(html.Table([html.Thead(html.Tr([html.Th(_p(lang, 'Usar', 'Use')), html.Th(''),
                                                 html.Th(_p(lang, 'Classe original', 'Original class')),
                                                 html.Th(_p(lang, 'Nome no modelo', 'Model name')),
                                                 html.Th(_p(lang, 'Disponível', 'Available'), className='num'),
                                                 html.Th(_p(lang, 'Treino', 'Train')), html.Th(_p(lang, 'Validação', 'Validation')),
                                                 html.Th(_p(lang, 'Teste', 'Test')),
                                                 html.Th(_p(lang, 'Não usado', 'Unused'), className='num')])),
                             html.Tbody(crow)]), className='tbl-wrap tall') if crow else
        C.hint(_p(lang, 'Selecione batches e clique em “Ler classes”.', 'Select batches and click “Read classes”.')),
        html.Div([C.tile(lbl, val, color=c) for (lbl, val), c in zip(summary, ['var(--accent)', 'var(--accent-2)', 'var(--ok)',
                                                                               'var(--ok)', 'var(--accent)', 'var(--warn)'])],
                 className='tiles t3'),
    ], variant='alt')
    training = C.hud(_p(lang, 'Treinamento do novo modelo', 'New model training'), [
        C.hint(_p(lang, 'Treina um novo modelo usando apenas os batches e classes selecionados.',
                  'Trains a new model using only the selected batches and classes.')),
        C.hint(_p(lang, 'Modelo base: ConvNeXt-Tiny    |    Dispositivo: Automático (CUDA se disponível)',
                  'Base model: ConvNeXt-Tiny    |    Device: Automatic (CUDA when available)')),
        C.form([(_p(lang, 'Épocas', 'Epochs'), _cinput(iat, '_cycle_epochs')),
                (_p(lang, 'Tamanho do batch', 'Batch size'), _cinput(iat, '_cycle_batch_size')),
                (_p(lang, 'Taxa de aprendizado', 'Learning rate'), _cinput(iat, '_cycle_lr')),
                ('Patience', _cinput(iat, '_cycle_patience')), ('Seed', _cinput(iat, '_cycle_seed')),
                (_p(lang, 'Versão', 'Version'), _cinput(iat, '_cycle_version', placeholder='auto')),
                (_p(lang, 'Pixel size (µm/pixel)', 'Pixel size (µm/pixel)'), _cinput(iat, '_cycle_pixel_size'))],
               class_name='f3'),
        C.hint(_p(lang, 'Pixel size: calibração física das imagens. Fica registrado no modelo (training_config) e é usado '
                        'nas métricas em µm da segmentação; a rede trabalha em pixels, então não muda o treino em si.',
                  'Pixel size: physical calibration of the images. It is stored with the model (training_config) and used '
                  'for µm metrics in segmentation; the network works in pixels, so it does not change training itself.')),
        C.hint('✓ ' + _p(lang, 'Manter manual_label dos batches de treino (não alterar)',
                         'Keep manual_label of training batches (unchanged)')),
        html.Div([html.Span(_p(lang, 'Progresso do treinamento', 'Training progress'), className='hint'),
                  html.Span(iat._cycle_train_percent_text.get(), className='push accent-text')], className='row'),
        html.Div(html.I(style={'width': f"{float(sp['training'].get() or 0)}%"}), className='bar'),
        html.Div([html.Pre(iat._cycle_train_detail.get(), className='pre-block'),
                  dcc.Graph(figure=_train_chart(iat, theme), config={'displayModeBar': False}, style={'height': '180px'})],
                 className='g g-2'),
        html.Div([C.btn('⚙  ' + _p(lang, 'Iniciar treinamento', 'Start training'), iact('cy_train'), 'pri'),
                  C.btn(_p(lang, 'Ver avaliação do modelo', 'View model evaluation'), iact('cy_eval'),
                        disabled=not iat._cycle_trained_version),
                  C.btn(_p(lang, 'O que é cada campo?', 'What is each field?'), iact('cy_help'), 'sm')], className='row'),
    ])
    versions = list(iat._cycle_saved_model_entries())
    targets = None
    if iat._cycle_target_mode.get() == 'manual':
        targets = html.Div([
            html.Div([C.btn(_p(lang, 'Selecionar todos', 'Select all'), iact('cy_targets_all'), 'sm'),
                      C.btn(_p(lang, 'Desmarcar todos', 'Clear all'), iact('cy_targets_none'), 'sm')], className='row'),
            html.Div([html.Div(('✓ ' if v.get() else '   ') + batch_id_from_csv(n), id={'type': 'cyt', 'b': n}, n_clicks=0,
                               className='clickrow' + (' on' if v.get() else ''))
                      for n, v in iat._cycle_target_vars.items()], className='clist-box'),
        ])
    else:
        targets = C.hint(_p(lang, 'Modo automático: serão usados todos os batches que não fazem parte da referência.',
                            'Automatic mode: all batches outside the reference set will be used.'))
    apply = C.hud(_p(lang, 'Aplicação do novo modelo ao projeto', 'Applying the new model to the project'), [
        C.hint(_p(lang, 'Classifique automaticamente os batches escolhidos e aplique o clustering nas classes.',
                  'Classify the chosen batches automatically and run clustering on the classes.')),
        html.Div([
            html.Div([html.Div(_p(lang, 'Escopo da nova classificação', 'New classification scope'), className='kicker'),
                      dcc.RadioItems(id=cyv('_cycle_target_mode'), value=iat._cycle_target_mode.get(), className='clist',
                                     inputClassName='chk', labelClassName='clist-item',
                                     options=[{'label': _p(lang, 'Todos os batches pendentes para o novo modelo (recomendado)',
                                                           'All batches pending for the new model (recommended)'), 'value': 'pending'},
                                              {'label': _p(lang, 'Selecionar batches manualmente', 'Select batches manually'),
                                               'value': 'manual'}]),
                      targets], className='subcard'),
            html.Div([html.Div(_p(lang, 'Opções', 'Options'), className='kicker'),
                      html.Label(_p(lang, 'Modelo a aplicar', 'Model to apply'), className='hint'),
                      dcc.Dropdown(id=cyv('_cycle_apply_model'), options=versions, value=iat._cycle_apply_model.get() or None,
                                   clearable=False, className='dd'),
                      C.switch(cyv('_cycle_auto_cluster'), _p(lang, 'Aplicar clustering automaticamente após a classificação',
                                                              'Run clustering automatically after classification'),
                               iat._cycle_auto_cluster.get()),
                      C.switch(cyv('_cycle_update_reference_features'),
                               _p(lang, 'Atualizar também as features dos batches usados no treinamento',
                                  'Also update features of the batches used for training'),
                               iat._cycle_update_reference_features.get()),
                      html.Div(iat._cycle_status.get(), className='accent-text small')], className='subcard'),
        ], className='g g-2'),
        html.Div([html.Div([html.Span(_p(lang, 'Progresso da classificação', 'Classification progress'), className='hint'),
                            html.Div(html.I(style={'width': f"{float(sp['classification'].get() or 0)}%"}), className='bar')]),
                  html.Div([html.Span('Clustering', className='hint'),
                            html.Div(html.I(style={'width': f"{float(sp['cluster'].get() or 0)}%"}), className='bar')])],
                 className='g g-2'),
        html.Div(iat._cycle_apply_detail.get(), className='hint'),
        C.btn('▶  ' + _p(lang, 'Classificar e aplicar', 'Classify and apply'), iact('cy_apply'), 'pri',
              disabled=not iat._cycle_trained_version),
    ], variant='alt')
    log = C.hud(_p(lang, 'Log em tempo real', 'Live log'), [
        C.hint(_p(lang, 'Treino, classificação e clustering desta subaba em tempo real.',
                  'Training, classification and clustering of this sub-tab in real time.')),
        log_box(iat, '300px'),
        C.btn(_p(lang, 'Limpar log', 'Clear log'), iact('clear_iat_log'), 'sm'),
    ])
    title = html.Div([html.H2(_p(lang, 'IAT — Treino + Classificação (Integrado)', 'IAT — Training + Classification (Integrated)'),
                              className='page-h'),
                      html.Span(_p(lang, 'Ciclo de iteração', 'Iteration cycle'), className='chip')], className='row')
    return html.Div([title, stages, html.Div([batches, classes], className='g g-2'),
                     html.Div([training, apply], className='g g-2'), log], className='stack')


HELP = [
    ('Batches de referência', 'Somente os batches selecionados entram no novo treinamento.'),
    ('Renomear classes', 'Duas classes com o mesmo nome final são agrupadas no novo modelo.'),
    ('Distribuição do dataset', 'Train/Valid/Test podem ser definidos por N ou percentual, individualmente por classe.'),
    ('Modelo e parâmetros', 'Mantém o pipeline científico atual; esta tela apenas orquestra o ciclo.'),
    ('Manter manual label', 'Rótulos manuais dos batches de referência nunca são sobrescritos.'),
]


def _help(iat, lang):
    body = html.Div([html.Div([html.B('› ' + h), html.Div(t, className='note')]) for h, t in HELP], className='stack')
    return {'modal': C.modal_card('O que é cada campo?', body, act('modal_close'))}


def _eval(iat, lang):
    from ..headless import NOTIFY
    from .modals import training_results
    iat.cycle_eval_request = None
    iat._cycle_open_evaluation()
    req = getattr(iat, 'cycle_eval_request', None)
    if not req:
        NOTIFY.push('info', _p(lang, 'Avaliação', 'Evaluation'),
                    _p(lang, 'Escolha um modelo em “Modelo a aplicar” ou treine um novo.',
                       'Choose a model in “Model to apply” or train a new one.'))
        return None
    try:
        return {'modal': training_results(lang, req[0], req[1])}
    except Exception as exc:
        iat._log(str(exc), 'error')
        NOTIFY.push('error', _p(lang, 'Avaliação', 'Evaluation'), str(exc))


def _bump(fn):
    def run(iat, lang):
        fn(iat)
        iat.web_version = getattr(iat, 'web_version', 0) + 1
    return run


ACTIONS = {
    'cy_validated': _bump(lambda i: i._cycle_select_validated_batches()),
    'cy_all_batches': _bump(lambda i: i._cycle_set_all_batches(True)),
    'cy_no_batches': _bump(lambda i: i._cycle_set_all_batches(False)),
    'cy_scan': lambda i, lang: i._cycle_scan_classes(),
    'cy_apply_defaults': _bump(lambda i: i._cycle_apply_defaults()),
    'cy_train': lambda i, lang: i._cycle_start_training(),
    'cy_eval': _eval,
    'cy_help': _help,
    'cy_targets_all': _bump(lambda i: i._cycle_set_all_targets(True)),
    'cy_targets_none': _bump(lambda i: i._cycle_set_all_targets(False)),
    'cy_apply': lambda i, lang: i._cycle_start_apply(),
}


def register(app, ctl, get_iat, respond):
    @app.callback(Output(dyn(ALL), 'children', allow_duplicate=True),
                  Input(cyv(ALL), 'value'), Input({'type': 'cyb', 'b': ALL}, 'n_clicks'),
                  Input({'type': 'cyc', 'cls': ALL}, 'n_clicks'), Input({'type': 'cyf', 'cls': ALL, 'k': ALL}, 'value'),
                  Input({'type': 'cyt', 'b': ALL}, 'n_clicks'),
                  State('lang-store', 'data'), State('theme-store', 'data'), prevent_initial_call=True)
    def cycle_inputs(_a, _b, _c, _d, _e, lang, theme):
        from .iat import versions
        trig = ctx.triggered_id
        if not isinstance(trig, dict):
            raise PreventUpdate
        value = ctx.triggered[0]['value']
        iat = get_iat()
        with ctl.lock:
            iat.prepare_cycle()
            before = versions(iat)
            kind = trig['type']
            if kind in ('cyb', 'cyc', 'cyt') and not value:
                raise PreventUpdate
            if kind == 'cyv':
                attr = trig['key']
                var = getattr(iat, attr, None)
                if var is None:
                    raise PreventUpdate
                new = bool(value) if isinstance(var.get(), bool) else ('' if value is None else str(value))
                if var.get() == new:
                    raise PreventUpdate
                var.set(new)
                if attr == '_cycle_target_mode':
                    iat._cycle_target_mode_changed()
                elif attr == '_cycle_apply_model':
                    iat._cycle_select_saved_model()
                elif attr == '_cycle_distribution_mode':
                    iat._cycle_distribution_changed()
                elif attr not in ('_cycle_batch_search', '_cycle_class_search'):
                    raise PreventUpdate
            elif kind == 'cyb':
                iat._cycle_toggle_batch(trig['b'])
            elif kind == 'cyc':
                iat._cycle_toggle_class(trig['cls'])
            elif kind == 'cyt':
                iat._cycle_toggle_target(trig['b'])
            elif kind == 'cyf':
                s = iat._cycle_class_vars.get(trig['cls'])
                if s is None or s[trig['k']].get() == (value or ''):
                    raise PreventUpdate
                s[trig['k']].set(value or '')
            iat.web_version = getattr(iat, 'web_version', 0) + 1
            return respond(iat, before, lang, theme)
