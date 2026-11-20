#!/usr/bin/env python3
"""IAT › Export (CSV, images by label, training dataset) — original export code."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from iat_backend import SEGMENTATION_METHODS, sanitize_filename

from .. import components as C
from ..headless import HeadlessBoolVar, HeadlessVar, StubTree, StubWidget
from .execution import act, dyn


def _defaults(iat):
    """Same defaults ``_build_export`` sets before drawing the tab."""
    iat._ensure_export_field_vars()
    if iat.project:
        default_dir = str(iat.project.project_dir / 'exports')
        if not iat.vars['csv_output_dir'].get():
            iat.vars['csv_output_dir'].set(default_dir)
        if not iat.vars['export_dir'].get():
            iat.vars['export_dir'].set(default_dir)
        if not iat.vars['training_dir'].get():
            iat.vars['training_dir'].set(str(iat.project.project_dir / 'exports' / 'training'))
    if not iat.vars['save_name'].get():
        iat.vars['save_name'].set(f'iat_export_{datetime.now().strftime("%Y-%m-%d")}')
    for name, factory in (('_training_split_mode_var', lambda: HeadlessVar(value='%')),
                          ('_training_existing_mode', lambda: HeadlessVar(value='merge')),
                          ('_training_report_duplicates_var', lambda: HeadlessBoolVar(value=True)),
                          ('_training_duplicate_status_var', lambda: HeadlessVar(value='Duplicadas ignoradas: 0'))):
        if getattr(iat, name, None) is None:
            setattr(iat, name, factory())
    if not isinstance(getattr(iat, '_training_preview_tree', None), StubTree):
        iat._training_preview_tree = StubTree()
    if not isinstance(getattr(iat, '_training_summary_label', None), StubWidget):
        iat._training_summary_label = StubWidget(text='Configure as classes e clique em Pré-visualizar distribuição.')
    if getattr(iat, '_ecotaxa_vars', None) is None:
        project = iat.project.name if iat.project else 'projeto'
        iat._ecotaxa_vars = {'status': HeadlessVar(value='auto'), 'copy': HeadlessBoolVar(value=True),
                             'name': HeadlessVar(value=f'ecotaxa_{project}_{datetime.now():%Y%m%d}')}


def on_show(iat):
    _defaults(iat)
    if (iat.vars['export_view'].get() or 'csv') == 'training' and iat.project is not None:
        iat.vars['training_scope'].set(iat.vars['training_scope'].get() or 'all')
        if iat.vars['training_scope'].get() == 'current':
            iat.vars['training_scope'].set('all')
        sig = iat._training_context_signature()
        if iat._training_context_key != sig:
            iat._training_start_context_load(sig)


def _export_classes(iat):
    """Data part of the classic ``_refresh_export_classes`` (class vars + counts)."""
    df = iat._export_preview_df()
    counts = df['manual_label'].astype(str).value_counts().to_dict() if not df.empty and 'manual_label' in df.columns else {}
    for lab in sorted(counts):
        if lab not in iat._export_class_vars:
            iat._export_class_vars[lab] = HeadlessBoolVar(value=True)
    q = str(iat.vars['export_class_search'].get() or '').strip().lower()
    labels = [x for x in sorted(iat._export_class_vars) if not q or q in x.lower()]
    return counts, labels


def _class_box(iat, counts, labels):
    from .iat import iat_input, iat_switch
    return html.Div([
        html.Div([iat_input(iat, 'export_class_search', debounce=True, placeholder=iat._export_text('search_classes')),
                  iat_switch(iat, 'export_all_classes', iat._export_text('select_all'))], className='row nowrap'),
        C.hint(f"{len(counts)} {iat._export_text('available_here')}"),
        html.Div([
            html.Label([dcc.Checklist(id={'type': 'xcls', 'cls': lab}, value=['on'] if iat._export_class_vars[lab].get() else [],
                                      options=[{'label': '', 'value': 'on'}], inputClassName='chk', className='inline'),
                        html.Span(lab, className='grow'),
                        html.Span(f'{int(counts[lab]):,}'.replace(',', '.') if lab in counts else '', className='hint')],
                       className='clist-item row nowrap')
            for lab in labels], className='clist-box tall'),
    ])


def _scope_radio(iat, key, options):
    return dcc.RadioItems(id={'type': 'ivs', 'key': key}, value=iat.vars[key].get(), className='clist',
                          inputClassName='chk', labelClassName='clist-item',
                          options=[{'label': lbl, 'value': val} for val, lbl in options])


def _shared_box(iat):
    counts, labels = _export_classes(iat)
    T = iat._export_text
    scope = _scope_radio(iat, 'export_scope', [('current', T('current')), ('selected', T('selected')),
                                                ('checked', T('checked')), ('all', T('all_batches'))])
    return C.hud(T('shared'), [
        html.Div([html.Div([html.Div(T('scope'), className='kicker'), scope], className='subcard'),
                  html.Div([html.Div(T('classes'), className='kicker'), _class_box(iat, counts, labels)], className='subcard')],
                 className='g g-2'),
        C.hint(T('shared_note')),
    ])


def _ecotaxa_view(iat, lang):
    """Item 95: TSV in the EcoTaxa import format (+ the images, ready to upload)."""
    from ecotaxa_export import STATUS_MODES
    from .iat import iact, iat_input, iat_select
    v = iat._ecotaxa_vars
    box = C.hud('EcoTaxa (TSV)', [
        C.note('Gera o arquivo ecotaxa_*.tsv no formato de importação do EcoTaxa (cabeçalho com tipos [t]/[f], '
               'colunas img_, object_, sample_, acq_ e process_) para o escopo e as classes acima. Medidas em pixels '
               'com a calibração em process_pixel (µm/pixel), como o ZooProcess; data, hora e profundidade vêm do '
               'nome da imagem quando presentes.'),
        html.Div([html.Div([html.Label('Método de segmentação', className='hint'),
                            iat_select(iat, 'seg_method', list(SEGMENTATION_METHODS.items()))], className='grow'),
                  html.Div([html.Label('Pixel size (µm/pixel)', className='hint'),
                            iat_input(iat, 'pixel_size', alt=True, class_name='inp sm')])], className='row nowrap'),
        C.hint('Imagens sem segmentação desse método saem sem medidas (colunas vazias). Gere a segmentação antes '
               'para incluir área, perímetro, Feret, eixos, ESD etc.'),
        html.Label('Status da anotação (object_annotation_status)', className='hint'),
        dcc.Dropdown(id={'type': 'xeco', 'key': 'status'}, options=[{'label': lbl, 'value': k} for k, lbl in STATUS_MODES.items()],
                     value=v['status'].get(), clearable=False, className='dd'),
        C.switch({'type': 'xeco', 'key': 'copy'}, 'Copiar as imagens para a mesma pasta (pacote pronto para importar)',
                 v['copy'].get()),
        html.Div(iat._export_text('output'), className='kicker'),
        html.Div([html.Label(iat._export_text('output_dir'), className='hint'),
                  html.Div([iat_input(iat, 'csv_output_dir'), C.btn('…', iact('browse_csv_export'), 'sm icon')],
                           className='browse')]),
        html.Div([html.Label('Nome do pacote / arquivo (o prefixo ecotaxa_ é obrigatório no EcoTaxa)', className='hint'),
                  dcc.Input(id={'type': 'xeco', 'key': 'name'}, value=v['name'].get(), debounce=True, className='inp')]),
        C.btn('Exportar para EcoTaxa', iact('save_ecotaxa'), 'pri'),
        C.hint('Com a cópia das imagens ligada é criada a pasta <nome>/ com o TSV e as imagens; compacte-a em .zip e '
               'importe no EcoTaxa. Referência: Picheral et al. (2017), EcoTaxa.'),
    ], variant='alt')
    return [_shared_box(iat), box]


def _csv_view(iat, lang):
    from .iat import iact, iat_input, iat_select, iat_switch
    counts, labels = _export_classes(iat)
    T = iat._export_text
    specs = iat._export_field_specs()
    groups = [('basic', T('basic')), ('labels', T('labels')), ('general', T('general')), ('seg', T('seg'))]
    fields = html.Div([
        html.Div([html.Div(title, className='kicker')] + [
            html.Label([dcc.Checklist(id={'type': 'xfld', 'key': key}, value=['on'] if iat._export_field_vars[key].get() else [],
                                      options=[{'label': '', 'value': 'on'}], inputClassName='chk', className='inline'),
                        html.Span(label)], className='clist-item row nowrap')
            for key, label, _s, _o, _d in specs[g]], className='subcard')
        for g, title in groups], className='g g-2')
    scope = _scope_radio(iat, 'export_scope', [('current', T('current')), ('selected', T('selected')),
                                                ('checked', T('checked')), ('all', T('all_batches'))])
    shared = C.hud(T('shared'), [
        html.Div([html.Div([html.Div(T('scope'), className='kicker'), scope], className='subcard'),
                  html.Div([html.Div(T('classes'), className='kicker'), _class_box(iat, counts, labels)], className='subcard')],
                 className='g g-2'),
        C.hint(T('shared_note')),
    ])
    csv = C.hud(T('csv'), [
        C.note(T('csv_sub')),
        html.Div(T('fields'), className='kicker'), fields,
        html.Div([html.Div([html.Label(T('seg_method'), className='hint'),
                            iat_select(iat, 'seg_method', list(SEGMENTATION_METHODS.items()))], className='grow'),
                  html.Div([html.Label(T('pixel_size_export'), className='hint'),
                            iat_input(iat, 'pixel_size', alt=True, class_name='inp sm')])], className='row nowrap'),
        html.Div(T('output'), className='kicker'),
        html.Div([html.Label(T('output_dir'), className='hint'),
                  html.Div([iat_input(iat, 'csv_output_dir'), C.btn('…', iact('browse_csv_export'), 'sm icon')],
                           className='browse')]),
        html.Div([html.Label(T('filename'), className='hint'), iat_input(iat, 'save_name')]),
        iat_switch(iat, 'export_save_metadata', T('metadata')), C.hint(T('metadata_tip')),
        C.btn(T('csv_button'), iact('save_csv'), 'pri'),
    ], variant='alt')
    return [shared, csv]


def _images_view(iat, lang):
    from .iat import iact, iat_input, iat_select, iat_switch
    counts, labels = _export_classes(iat)
    T = iat._export_text
    scope = _scope_radio(iat, 'export_scope', [('current', T('current')), ('selected', T('selected')),
                                                ('checked', T('checked')), ('all', T('all_batches'))])
    return [C.hud(T('images'), [
        C.note(T('images_workflow_sub')),
        html.Div([
            html.Div([html.Div(T('scope'), className='kicker'), C.hint(T('images_scope_tip')), scope], className='subcard'),
            html.Div([html.Div(T('classes'), className='kicker'), C.hint(T('images_labels_tip')),
                      _class_box(iat, counts, labels)], className='subcard'),
            html.Div([
                html.Div(T('img_opts'), className='kicker'), C.hint(T('images_options_tip')),
                iat_switch(iat, 'export_create_subfolders', T('subfolders')),
                iat_switch(iat, 'export_keep_names', T('keep_names')),
                iat_switch(iat, 'export_include_outline', T('include_outline')),
                iat_switch(iat, 'export_only_segmented', T('only_segmented')),
                html.Label(T('seg_method'), className='hint'),
                iat_select(iat, 'seg_method', list(SEGMENTATION_METHODS.items())),
                html.Div(T('img_output'), className='kicker'),
                html.Div([iat_input(iat, 'export_dir'), C.btn('…', iact('browse_export'), 'sm icon')], className='browse'),
                C.btn(T('images_button'), iact('export_images'), 'pri'),
            ], className='subcard'),
        ], className='g g-3'),
    ])]


def _training_view(iat, lang):
    from .iat import iact, iat_input, iat_select, iat_switch
    T = iat._export_text
    df = iat._training_context_df
    if getattr(iat, '_training_context_loading', False) or (iat.project is not None and iat._training_context_key is None):
        return [C.hud(T('training_title'), [C.note(T('training_loading'))])]
    counts = df['manual_label'].fillna('').astype(str).value_counts().to_dict() if (df is not None and not df.empty and 'manual_label' in df.columns) else {}
    for lab in sorted(counts):
        iat._training_class_vars.setdefault(lab, HeadlessBoolVar(value=True))
        iat._training_group_vars.setdefault(lab, HeadlessVar(value=lab))
    pal = iat._training_palette()
    scope = _scope_radio(iat, 'training_scope', [('all', 'Todas as imagens do projeto'),
                                                  ('batches', 'Apenas batch(es) selecionados'),
                                                  ('selected', 'Apenas imagens selecionadas na galeria')])
    classes = html.Div([
        html.Label([dcc.Checklist(id={'type': 'tcls', 'cls': lab}, value=['on'] if iat._training_class_vars[lab].get() else [],
                                  options=[{'label': '', 'value': 'on'}], inputClassName='chk', className='inline'),
                    html.Span(lab, className='grow'), html.Span(f'{int(counts[lab]):,}'.replace(',', '.'), className='hint')],
                   className='clist-item row nowrap') for lab in sorted(counts)], className='clist-box tall')
    sel = iat._training_selected_labels()
    group_rows = [html.Tr([html.Td(html.I(className='swatch', style={'background': pal[i % len(pal)]})), html.Td(lab),
                           html.Td(dcc.Input(id={'type': 'tgrp', 'cls': lab}, value=iat._training_group_vars[lab].get(),
                                             debounce=True, className='inp sm')),
                           html.Td(f'{int(counts.get(lab, 0)):,}'.replace(',', '.'), className='num')])
                  for i, lab in enumerate(sel)]
    out_labels = iat._training_output_labels()
    for lab in out_labels:
        if lab not in iat._training_split_vars:
            mode = iat._training_split_mode_var.get() if iat._training_split_mode_var is not None else '%'
            iat._training_split_vars[lab] = {'mode': HeadlessVar(value=mode), 'train': HeadlessVar(value='70'),
                                             'valid': HeadlessVar(value='15'), 'test': HeadlessVar(value='15')}
    split_rows = []
    for i, lab in enumerate(out_labels):
        v = iat._training_split_vars[lab]
        split_rows.append(html.Tr([
            html.Td(html.I(className='swatch', style={'background': pal[i % len(pal)]})), html.Td(html.B(lab)),
            html.Td(dcc.Dropdown(id={'type': 'tspl', 'cls': lab, 'k': 'mode'}, options=['%', 'N'], value=v['mode'].get(),
                                 clearable=False, className='dd', style={'width': '70px'})),
            *[html.Td(dcc.Input(id={'type': 'tspl', 'cls': lab, 'k': k}, value=v[k].get(), debounce=True,
                                className='inp sm', style={'width': '70px'})) for k in ('train', 'valid', 'test')],
        ]))
    tree = iat._training_preview_tree
    prev = (C.table(['Classe', 'Disponível', 'Train', 'Valid', 'Test'], tree.rows, num_cols=(1, 2, 3, 4))
            if tree is not None and tree.rows else None)
    return [
        C.hud(T('training_title'), [
            C.note('Selecione as classes, crie agrupamentos, defina a divisão e revise a distribuição antes de exportar.'),
            html.Div([
                html.Div([html.Div('1. Escopo dos dados', className='kicker'), scope,
                          C.btn(T('training_choose_batches'), iact('training_choose_batches'), 'sm'),
                          C.hint(', '.join(sorted(iat._training_batch_selection)) or '—'),
                          html.Label('Seed do sorteio', className='hint'), iat_input(iat, 'training_seed', class_name='inp sm'),
                          html.Label('Unidade de sorteio', className='hint'),
                          iat_select(iat, 'training_unit', [('image', T('training_unit_image')), ('group', T('training_unit_group'))])],
                         className='subcard'),
                html.Div([html.Div('2. Classes incluídas', className='kicker'),
                          iat_switch(iat, 'training_all_classes', 'Todas'), classes], className='subcard'),
            ], className='g g-2'),
        ]),
        html.Div([
            C.hud('3. Agrupamento de classes (opcional)', [
                C.hint('Agrupe classes usando o mesmo nome de saída, ou renomeie cada classe individualmente.'),
                html.Div([C.btn('Usar nomes originais', iact('training_reset_groups'), 'sm'),
                          C.btn('Atualizar grupos', iact('training_update_groups'), 'sm')], className='row'),
                html.Div(html.Table([html.Thead(html.Tr([html.Th(''), html.Th('Classe original'), html.Th('Classe de saída'),
                                                          html.Th('Imagens', className='num')])), html.Tbody(group_rows)]),
                         className='tbl-wrap tall'),
            ]),
            C.hud('4. Configuração da divisão', [
                C.hint('Configure Train / Validação / Teste para cada grupo final, em percentual (%) ou quantidade absoluta (N).'),
                html.Div([html.Span('Modo global:', className='hint'),
                          C.seg({'type': 'tmode', 'k': 'all'}, [('%', 'Percentual (%)'), ('N', 'Quantidade (N)')],
                                iat._training_split_mode_var.get())], className='row'),
                html.Div(html.Table([html.Thead(html.Tr([html.Th(''), html.Th('Classe / Grupo'), html.Th('Modo'),
                                                          html.Th('Train'), html.Th('Valid'), html.Th('Test')])),
                                     html.Tbody(split_rows)]), className='tbl-wrap tall'),
            ], variant='alt'),
        ], className='g g-2'),
        html.Div([
            C.hud('5. Pré-visualização da distribuição', [
                C.hint('Resumo das imagens por classe ou grupo e por split.'),
                C.btn('Pré-visualizar distribuição', iact('training_preview'), 'sm pri'),
                prev, html.Div(iat._training_summary_label.state.get('text', ''), className='accent-text small'),
            ]),
            C.hud('6. Local de exportação', [
                html.Label('Pasta de destino', className='hint'),
                html.Div([iat_input(iat, 'training_dir'), C.btn('…', iact('training_browse'), 'sm icon')], className='browse'),
                html.Label('Nome da pasta', className='hint'), iat_input(iat, 'training_name'),
                *[iat_switch(iat, k, lbl) for k, lbl in (('training_keep_names', 'Manter nomes originais'),
                                                          ('training_manifest', 'Gerar manifesto (CSV)'),
                                                          ('training_config', 'Salvar configuração (JSON)'),
                                                          ('training_stats', 'Salvar estatísticas (CSV)'))],
                html.Label('Se a pasta já existir:', className='hint'),
                C.seg({'type': 'tmode', 'k': 'existing'}, [('merge', 'Mesclar com dataset existente'),
                                                            ('replace', 'Substituir conteúdo existente')],
                      iat._training_existing_mode.get()),
                C.switch({'type': 'tmode', 'k': 'dups'}, 'Gerar relatório de duplicadas', iat._training_report_duplicates_var.get()),
                html.Div(iat._training_duplicate_status_var.get(), className='hint'),
                C.btn('Exportar dataset de treinamento', iact('training_export'), 'pri'),
                html.Div(html.I(style={'width': f'{float(iat._training_progress_var.get() or 0)}%'}), className='bar'),
            ], variant='alt'),
        ], className='g g-2'),
    ]


def render(iat, lang, theme):
    _defaults(iat)
    view = iat.vars['export_view'].get() or 'csv'
    tab_labels = [('csv', iat._export_text('tab_csv')), ('images', iat._export_text('tab_images')),
                  ('training', iat._export_text('tab_training')), ('ecotaxa', 'EcoTaxa (TSV)')]
    tabs = html.Div([C.btn(label, {'type': 'iact', 'name': f'export_view_{mode}'}, 'pri' if mode == view else '')
                     for mode, label in tab_labels], className='row')
    if view == 'training':
        body = _training_view(iat, lang)
    elif view == 'ecotaxa':
        body = _ecotaxa_view(iat, lang)
    elif view == 'images':
        body = _images_view(iat, lang)
    else:
        body = _csv_view(iat, lang)
    return html.Div([tabs, *body], className='stack')


def _confirm(iat, text, yes):
    from .iat import iact
    return {'modal': C.modal_card(iat._export_text('training_title') if 'training' in yes else iat._export_text('csv'),
                                  html.P(text, className='note'), act('modal_close'),
                                  actions=[C.btn('Não', act('modal_close')), C.btn('Sim', iact(yes), 'pri')])}


def _save_csv(iat, lang):
    outdir = iat.vars['csv_output_dir'].get().strip()
    name = iat.vars['save_name'].get().strip()
    if outdir and name:
        name = Path(name).name
        if not name.lower().endswith('.csv'):
            name += '.csv'
        if (Path(outdir).expanduser() / name).exists():
            return _confirm(iat, f'{name} já existe.\n\nSubstituir?', 'save_csv_yes')
    iat.save_csv()


def _training_export(iat, lang):
    parent = iat.vars['training_dir'].get().strip()
    if parent and iat.project:
        name = sanitize_filename(Path(iat.vars['training_name'].get().strip() or 'ilt_dataset').name) or 'ilt_dataset'
        if (Path(parent).expanduser() / name).exists():
            msg = ('A pasta do dataset já existe.\nSubstituir todo o conteúdo dessa pasta?'
                   if iat._training_existing_mode.get() == 'replace' else
                   'A pasta do dataset já existe.\nMesclar com o dataset existente? Imagens com o mesmo nome original serão ignoradas.')
            return _confirm(iat, msg, 'training_export_yes')
    iat._training_export_dataset()


def _save_ecotaxa(iat, lang):
    """Write ecotaxa_*.tsv (and optionally copy the images) for the export scope."""
    import os
    from ecotaxa_export import append_rows, ecotaxa_filename, ecotaxa_frame, write_header
    from iat_io import CopyJob, copy_jobs, recommended_io_workers
    from ..headless import NOTIFY
    if not iat.project:
        return
    _defaults(iat)
    T = iat._export_text
    if not iat.vars['export_all_classes'].get() and not iat._export_selected_classes():
        NOTIFY.push('warning', 'EcoTaxa', T('no_classes'))
        return
    plan = iat._export_scope_plan()
    if not plan:
        NOTIFY.push('warning', 'EcoTaxa', T('empty_scope'))
        return
    outdir = Path(iat.vars['csv_output_dir'].get().strip() or (iat.project.project_dir / 'exports')).expanduser()
    v = iat._ecotaxa_vars
    filename = ecotaxa_filename(v['name'].get())
    copy_images = bool(v['copy'].get())
    status_mode = str(v['status'].get() or 'auto')
    base = outdir / filename[:-4] if copy_images else outdir
    if copy_images and base.exists():
        for i in range(2, 10000):
            cand = base.with_name(f'{base.name}_{i}')
            if not cand.exists():
                base = cand
                break
    path = base / filename
    if path.exists():
        path = path.with_name(f'{path.stem}_{datetime.now():%H%M%S}.tsv')
    method = str(iat.vars['seg_method'].get())
    px = iat._segmentation_pixel_size(None, iat.vars['pixel_size'].get())
    project = iat.project

    def work(progress):
        base.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tsv.part')
        rows = missing = 0
        jobs = []
        with open(tmp, 'w', encoding='utf-8', newline='') as fh:
            write_header(fh)
            for i, (bn, gids) in enumerate(plan, 1):
                df = iat._export_filter_frame(iat._export_read_batch(bn), gids)
                if not df.empty:
                    df = iat._export_attach_segmentation(df, bn, method)
                    rows += append_rows(fh, ecotaxa_frame(df, px, method, status_mode))
                    if copy_images:
                        for name in df['names'].astype(str):
                            src = project.resolve_image(bn, name)
                            if src is None or not Path(src).is_file():
                                missing += 1
                            else:
                                jobs.append(CopyJob(Path(src), base / name, None))
                progress(i * (40 if copy_images else 100) / len(plan), 100, bn)
        os.replace(tmp, path)
        copied = 0
        if copy_images and jobs:
            report = copy_jobs(jobs, workers=recommended_io_workers(),
                               progress=lambda c, n, _l='': progress(40 + 60 * c / max(n, 1), 100, f'Copiando {c}/{n}'))
            copied = report.copied
        return path, rows, copied, missing

    def done(res):
        p, rows, copied, missing = res
        msg = f'EcoTaxa: {rows:,} objetos em {p}'.replace(',', '.')
        if copy_images:
            msg += f' • {copied:,} imagens copiadas'.replace(',', '.')
            if missing:
                msg += f' • {missing:,} imagens não encontradas'.replace(',', '.')
        iat._log(msg, 'success' if not missing else 'warn')
        NOTIFY.push('info', 'EcoTaxa', msg)
    iat._run_operation('Exportando para EcoTaxa', work, done)


def _view(mode):
    def run(iat, lang):
        iat.vars['export_view'].set(mode)
        on_show(iat)
        iat.web_version = getattr(iat, 'web_version', 0) + 1
    return run


def _bump(fn):
    def run(iat, lang):
        res = fn(iat, lang)
        iat.web_version = getattr(iat, 'web_version', 0) + 1
        return res
    return run


def _choose_batches(iat, lang):
    from .iat import iact
    names = iat.project.csv_names() if iat.project else []
    body = html.Div([
        dcc.Checklist(id='tbatch-list', options=names, value=sorted(iat._training_batch_selection), className='clist',
                      inputClassName='chk', labelClassName='clist-item'),
    ], className='clist-box xtall')
    return {'modal': C.modal_card(iat._export_text('training_choose_batches'), body, act('modal_close'),
                                  actions=[C.btn('OK', iact('training_batches_ok'), 'pri')])}


def _batches_ok(iat, lang):
    sel = getattr(iat, '_pending_batches', None)
    if sel is not None:
        iat._training_batch_selection = set(sel)
    iat.vars['training_scope'].set('batches')
    iat._training_context_key = None
    on_show(iat)
    iat.web_version = getattr(iat, 'web_version', 0) + 1
    return {'modal': []}


ACTIONS = {
    'export_view_csv': _view('csv'), 'export_view_images': _view('images'), 'export_view_training': _view('training'),
    'export_view_ecotaxa': _view('ecotaxa'), 'save_ecotaxa': lambda iat, lang: _save_ecotaxa(iat, lang),
    'browse_csv_export': _bump(lambda iat, lang: iat.browse_csv_export()),
    'browse_export': _bump(lambda iat, lang: iat.browse_export()),
    'save_csv': _save_csv, 'save_csv_yes': lambda iat, lang: (iat.save_csv(), {'modal': []})[1],
    'export_images': lambda iat, lang: iat.export_images(),
    'training_choose_batches': _choose_batches, 'training_batches_ok': _batches_ok,
    'training_reset_groups': _bump(lambda iat, lang: iat._training_reset_group_names()),
    'training_update_groups': _bump(lambda iat, lang: iat._training_grouping_updated()),
    'training_preview': _bump(lambda iat, lang: iat._training_preview()),
    'training_browse': _bump(lambda iat, lang: iat._training_browse_output()),
    'training_export': _training_export,
    'training_export_yes': lambda iat, lang: (iat._training_export_dataset(), {'modal': []})[1],
}


def register(app, ctl, get_iat, respond):
    @app.callback(Output(dyn(ALL), 'children', allow_duplicate=True),
                  Input({'type': 'xcls', 'cls': ALL}, 'value'), Input({'type': 'xfld', 'key': ALL}, 'value'),
                  Input({'type': 'tcls', 'cls': ALL}, 'value'), Input({'type': 'tgrp', 'cls': ALL}, 'value'),
                  Input({'type': 'tspl', 'cls': ALL, 'k': ALL}, 'value'), Input({'type': 'tmode', 'k': ALL}, 'value'),
                  Input({'type': 'xeco', 'key': ALL}, 'value'),
                  State('lang-store', 'data'), State('theme-store', 'data'), prevent_initial_call=True)
    def export_inputs(_a, _b, _c, _d, _e, _f, _g, lang, theme):
        from .iat import versions
        trig = ctx.triggered_id
        if not isinstance(trig, dict):
            raise PreventUpdate
        value = ctx.triggered[0]['value']
        iat = get_iat()
        with ctl.lock:
            before = versions(iat)
            kind = trig['type']
            if kind == 'xeco':
                var = getattr(iat, '_ecotaxa_vars', {}).get(trig['key'])
                if var is None:
                    raise PreventUpdate
                var.set(bool(value) if trig['key'] == 'copy' else str(value or ''))
                raise PreventUpdate
            if kind == 'xcls':
                var = iat._export_class_vars.get(trig['cls'])
                if var is None or var.get() == bool(value):
                    raise PreventUpdate
                var.set(bool(value))
                iat._export_class_toggled()
            elif kind == 'xfld':
                var = iat._export_field_vars.get(trig['key'])
                if var is None or var.get() == bool(value):
                    raise PreventUpdate
                var.set(bool(value))
                raise PreventUpdate
            elif kind == 'tcls':
                var = iat._training_class_vars.get(trig['cls'])
                if var is None or var.get() == bool(value):
                    raise PreventUpdate
                var.set(bool(value))
                iat._training_class_toggled()
                iat._training_grouping_updated()
            elif kind == 'tgrp':
                var = iat._training_group_vars.get(trig['cls'])
                if var is None or var.get() == (value or ''):
                    raise PreventUpdate
                var.set(value or '')
                iat._training_grouping_updated()
            elif kind == 'tspl':
                v = iat._training_split_vars.get(trig['cls'], {}).get(trig['k'])
                if v is None or v.get() == (value or ''):
                    raise PreventUpdate
                v.set(value or '')
                iat._training_clear_preview()
                iat._training_preview_tree.rows = []
            elif kind == 'tmode':
                if trig['k'] == 'all':
                    iat._training_set_split_mode_all(value)
                elif trig['k'] == 'existing':
                    iat._training_existing_mode.set(value)
                    raise PreventUpdate
                elif trig['k'] == 'dups':
                    iat._training_report_duplicates_var.set(bool(value))
                    raise PreventUpdate
            iat.web_version = getattr(iat, 'web_version', 0) + 1
            return respond(iat, before, lang, theme)

    @app.callback(Output('ack-d', 'data', allow_duplicate=True), Input('tbatch-list', 'value'), prevent_initial_call=True)
    def batches_pending(value):
        iat = get_iat()
        with ctl.lock:
            iat._pending_batches = list(value or [])
        return no_update
