#!/usr/bin/env python3
"""Export UI/operations mixin for :mod:`iat_workspace`.

Keeping CSV/image export concerns outside the main workspace controller makes
that controller substantially easier to navigate without changing behavior.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from iat_backend import SEGMENTATION_METHODS, ensure_required_columns, sanitize_filename
from iat_io import CopyJob, copy_jobs, recommended_io_workers
from iat_widgets import ScrollFrame
from project_settings import (
    DEFAULT_PIXEL_SIZE_UM_PER_PX,
    load_project_settings,
    parse_pixel_size,
)
from ui_resources import IAT_COLORS as C


class IATExportMixin:
    def _export_text(self,key):
        pt={
            'title':'Exportação','subtitle':'Exporte dados e imagens da análise com filtros, classes e campos personalizados.',
            'shared':'Seleção e filtros (comum para CSV e imagens)','scope':'1. Escopo dos dados','classes':'2. Filtrar por classes (manual_label)',
            'current':'Batch/visão atual','selected':'Apenas imagens selecionadas na galeria','checked':'Apenas imagens marcadas','all_batches':'Todos os batches (Overview)',
            'search_classes':'Buscar classes...','select_all':'Selecionar todas','shared_note':'Os filtros de escopo e classes acima são aplicados tanto ao CSV quanto à exportação das imagens.',
            'csv':'Exportar CSV','csv_sub':'Escolha exatamente quais informações serão incluídas no arquivo CSV.','fields':'1. Campos para incluir no CSV',
            'basic':'Informações básicas','labels':'Rótulos e classes','general':'Classificação / métricas gerais','seg':'Métricas de segmentação (se disponível)',
            'seg_method':'Método de segmentação','pixel_size_export':'Pixel size fallback para conversões físicas','output':'2. Local e nome do arquivo',
            'output_dir':'Diretório de saída','filename':'Nome do arquivo (opcional)','metadata':'Salvar metadados da exportação (.json)',
            'metadata_tip':'O JSON registra projeto, escopo, classes, método de segmentação, pixel size e campos exportados.',
            'images':'Exportar imagens por label','images_sub':'Exporte as imagens do mesmo escopo, filtradas pelas mesmas classes.',
            'images_workflow_sub':'Fluxo visual em três etapas: escolha o escopo, selecione os labels e configure a exportação.',
            'images_scope_tip':'Escolha de onde virão as imagens. A seleção não altera nenhum dado do projeto.',
            'images_labels_tip':'Marque somente os labels que deseja exportar. A quantidade aparece ao lado de cada classe.',
            'images_options_tip':'Defina a organização dos arquivos e a pasta de destino antes de exportar.',
            'img_opts':'3. Opções de exportação','subfolders':'Criar subpastas por manual_label','keep_names':'Manter nomes originais dos arquivos',
            'include_outline':'Incluir imagem adicional com contorno da segmentação (se disponível)',
            'only_segmented':'Incluir apenas imagens com segmentação válida','img_output':'4. Local de exportação',
            'csv_button':'Salvar CSV','images_button':'Exportar por label','no_fields':'Selecione pelo menos um campo para o CSV.',
            'no_classes':'Nenhuma classe foi selecionada.','empty_scope':'O escopo selecionado não contém imagens.',
            'csv_done':'CSV exportado','metadata_done':'Metadados salvos','images_done':'Imagens exportadas',
            'rows':'linhas','current_batch':'Batch atual','overview':'Overview','available_here':'classes no contexto carregado',
            'tab_csv':'Exportar CSV','tab_images':'Exportar imagens por label','tab_training':'Exportar para Treinamento (novo)',
            'training_title':'Dataset para treinamento','training_sub':'Crie uma estrutura train/valid/test com seleção, agrupamento e amostragem reprodutível.',
            'training_scope':'1. Escopo dos dados','training_classes':'2. Seleção de classes','training_grouping':'3. Agrupamento de classes (opcional)',
            'training_split':'4. Configuração da divisão','training_options':'5. Opções adicionais','training_preview':'6. Pré-visualização da distribuição',
            'training_output':'7. Local de exportação','training_current':'Batch/visão atual','training_batches':'Apenas batch(es) selecionados',
            'training_gallery':'Apenas imagens selecionadas na galeria','training_all':'Todas as imagens do projeto',
            'training_choose_batches':'Selecionar batches...','training_seed':'Seed do sorteio','training_unit':'Unidade de sorteio',
            'training_unit_image':'Imagem individual','training_unit_group':'ID original (custom_data)',
            'training_original_class':'Classe original','training_output_class':'Classe de saída','training_available':'Disponíveis',
            'training_refresh_groups':'Atualizar agrupamentos e divisão','training_mode':'Modo','training_train':'Train','training_valid':'Valid','training_test':'Test',
            'training_keep_names':'Manter nomes originais dos arquivos','training_manifest':'Gerar manifesto das imagens (CSV)',
            'training_config_file':'Salvar configuração da exportação (JSON)','training_stats_file':'Salvar estatísticas da divisão (CSV)',
            'training_preview_button':'Pré-visualizar distribuição','training_export_button':'Exportar dataset de treinamento',
            'training_folder_name':'Nome da pasta','training_no_batches':'Selecione pelo menos um batch.',
            'training_no_classes':'Selecione pelo menos uma classe.','training_invalid':'Configuração de divisão inválida',
            'training_done':'Dataset de treinamento exportado','training_loading':'Carregando batches e classes do projeto…','training_group_note':'Para agrupar classes, use o mesmo nome de saída em duas ou mais classes.',
            'training_group_unit_note':'Com ID original, todas as imagens do mesmo ID permanecem no mesmo split; contagens podem ficar aproximadas.',
            'training_preview_empty':'Configure as classes e clique em Pré-visualizar distribuição.',
            'training_overwrite':'A pasta do dataset já existe. Substituir todo o conteúdo dessa pasta?',
            'training_missing':'imagens de origem não encontradas','training_copied':'imagens copiadas',
            'training_unused':'Não usadas','training_total':'Total (selecionadas)','training_browse':'Procurar...','training_batches_selected':'batches selecionados',
        }
        en={
            'title':'Export','subtitle':'Export analysis data and images with custom filters, classes, and fields.',
            'shared':'Selection and filters (shared by CSV and images)','scope':'1. Data scope','classes':'2. Filter by classes (manual_label)',
            'current':'Current batch/view','selected':'Only images selected in Gallery','checked':'Only checked images','all_batches':'All batches (Overview)',
            'search_classes':'Search classes...','select_all':'Select all','shared_note':'The scope and class filters above are applied to both CSV and image export.',
            'csv':'Export CSV','csv_sub':'Choose exactly which information will be included in the CSV file.','fields':'1. Fields to include in CSV',
            'basic':'Basic information','labels':'Labels and classes','general':'Classification / general metrics','seg':'Segmentation metrics (if available)',
            'seg_method':'Segmentation method','pixel_size_export':'Fallback pixel size for physical conversions','output':'2. Output location and filename',
            'output_dir':'Output directory','filename':'Filename (optional)','metadata':'Save export metadata (.json)',
            'metadata_tip':'The JSON records project, scope, classes, segmentation method, pixel size, and exported fields.',
            'images':'Export images by label','images_sub':'Export images from the same scope, filtered by the same classes.',
            'images_workflow_sub':'Three-step visual flow: choose the scope, select labels, then configure the export.',
            'images_scope_tip':'Choose where the images come from. This selection never changes project data.',
            'images_labels_tip':'Select only the labels you want to export. The image count appears beside each class.',
            'images_options_tip':'Choose file organization and the destination folder before exporting.',
            'img_opts':'3. Export options','subfolders':'Create subfolders by manual_label','keep_names':'Keep original filenames',
            'include_outline':'Include an additional image with segmentation outline (if available)',
            'only_segmented':'Include only images with valid segmentation','img_output':'4. Export location',
            'csv_button':'Save CSV','images_button':'Export by label','no_fields':'Select at least one CSV field.',
            'no_classes':'No class was selected.','empty_scope':'The selected scope contains no images.',
            'csv_done':'CSV exported','metadata_done':'Metadata saved','images_done':'Images exported',
            'rows':'rows','current_batch':'Current batch','overview':'Overview','available_here':'classes in loaded context',
            'tab_csv':'Export CSV','tab_images':'Export images by label','tab_training':'Export for Training (new)',
            'training_title':'Training dataset','training_sub':'Create a train/valid/test structure with selection, grouping, and reproducible sampling.',
            'training_scope':'1. Data scope','training_classes':'2. Class selection','training_grouping':'3. Class grouping (optional)',
            'training_split':'4. Split configuration','training_options':'5. Additional options','training_preview':'6. Distribution preview',
            'training_output':'7. Export location','training_current':'Current batch/view','training_batches':'Selected batch(es) only',
            'training_gallery':'Only images selected in Gallery','training_all':'All project images',
            'training_choose_batches':'Select batches...','training_seed':'Random seed','training_unit':'Sampling unit',
            'training_unit_image':'Individual image','training_unit_group':'Original ID (custom_data)',
            'training_original_class':'Original class','training_output_class':'Output class','training_available':'Available',
            'training_refresh_groups':'Refresh grouping and split','training_mode':'Mode','training_train':'Train','training_valid':'Valid','training_test':'Test',
            'training_keep_names':'Keep original filenames','training_manifest':'Create image manifest (CSV)',
            'training_config_file':'Save export configuration (JSON)','training_stats_file':'Save split statistics (CSV)',
            'training_preview_button':'Preview distribution','training_export_button':'Export training dataset',
            'training_folder_name':'Folder name','training_no_batches':'Select at least one batch.',
            'training_no_classes':'Select at least one class.','training_invalid':'Invalid split configuration',
            'training_done':'Training dataset exported','training_loading':'Loading project batches and classes…','training_group_note':'To group classes, use the same output name for two or more classes.',
            'training_group_unit_note':'With Original ID, all images sharing an ID remain in the same split; counts may be approximate.',
            'training_preview_empty':'Configure classes and click Preview distribution.',
            'training_overwrite':'The dataset folder already exists. Replace all content in that folder?',
            'training_missing':'source images not found','training_copied':'images copied',
            'training_unused':'Unused','training_total':'Total (selected)','training_browse':'Browse...','training_batches_selected':'batches selected',
        }
        return (pt if self.lang()=='pt' else en).get(key,key)

    def _export_field_specs(self):
        # key, UI label, source column, output column, default
        return {
            'basic':[
                ('gid','gid (ID global)','gid','gid',True),
                ('image_name','Image Name (nome do arquivo)','names','image_name',True),
                ('batch','batch (CSV)','batch','batch',True),
                ('custom_data','custom_data (ID original)','custom_data_orig','custom_data',True),
            ],
            'labels':[
                ('manual_label','manual_label','manual_label','manual_label',True),
                ('correct_label','correct_label (rótulo original)','correct_label','correct_label',False),
                ('cluster','cluster (se disponível)','cluster','cluster',False),
                ('color_id','colors (ID da cor)','colors','color_id',False),
            ],
            'general':[
                ('predicted_class','Predicted class (top-1)','preds1','predicted_class',True),
                ('confidence','Confidence (0–1)','confs1','confidence',True),
                ('variance','Uncertainty = 1 − (top-1 − top-2) (0–1)','var_norm','uncertainty_margin',True),
                ('predicted_class_2','Predicted class (top-2)','preds2','predicted_class_2',False),
                ('confidence_2','Confidence top-2 (0–1)','confs2','confidence_2',False),
                ('predicted_class_3','Predicted class (top-3)','preds3','predicted_class_3',False),
                ('confidence_3','Confidence top-3 (0–1)','confs3','confidence_3',False),
            ],
            'seg':[
                ('area','Área — px² + µm²','Area (pxl)',None,True),
                ('perimeter','Perímetro (Crofton) — px + µm','Perimeter (pxl)',None,False),
                ('esd','ESD — px + µm','ESD (pxl)',None,True),
                ('circularity','Circularity (adimensional)','circularity','circularity',True),
                ('elongation','Elongation = Feret_max/Feret_min (adimensional)','Elongation','elongation',True),
                ('rectangularity','Rectangularity (adimensional)','Rectangularity','rectangularity',True),
                ('solidity','Solidity (adimensional)','Solidity','solidity',False),
                ('feret_max','Feret_max — px + µm','Feret_max',None,False),
                ('feret_min','Feret_min — px + µm','Feret_min',None,False),
                ('major','Major axis (elipse de momentos) — px + µm','Major (pxl)',None,False),
                ('minor','Minor axis (elipse de momentos) — px + µm','Minor (pxl)',None,False),
                ('eccentricity','Eccentricity (adimensional)','Eccentricity','eccentricity',False),
                ('biovolume','Biovolume elipsoidal — px³ + µm³','Biovolume (pxl3)',None,False),
                ('mean_intensity','Mean Intensity (0–255)','Mean Intensity','mean_intensity_0_255',False),
                ('median_intensity','Median Intensity (0–255)','Median Intensity','median_intensity_0_255',False),
                ('std_intensity','Std Intensity (0–255)','Std Intensity','std_intensity_0_255',False),
                ('contrast','Contrast (0–255)','Contrast','contrast_0_255',False),
                ('border','Percent_on_border (%)','Percent_on_border','percent_on_border_pct',False),
                ('holes','Number Holes (contagem)','Number Holes','number_holes',False),
                ('validation','Validation State (booleano)','Validation State','validation_state',False),
                ('metrics_version','Metrics Version (1 = antigo, 2 = atual)','Metrics Version','metrics_version',False),
            ],
        }

    def _ensure_export_field_vars(self):
        for group in self._export_field_specs().values():
            for key,_,_,_,default in group:
                if key not in self._export_field_vars:
                    self._export_field_vars[key]=tk.BooleanVar(value=default)

    def _export_scope_gids(self):
        scope=str(self.vars['export_scope'].get() or 'current')
        if scope=='selected':
            return set(map(str,self.allb_gids or self.selected_gids))
        if scope=='checked':
            return set(map(str,self.checked_gids))
        return None

    def _export_scope_plan(self):
        if not self.project:return []
        scope=str(self.vars['export_scope'].get() or 'current')
        if scope=='all':
            return [(bn,None) for bn in self.project.csv_names()]
        if scope=='current':
            bn=self.vars['batch'].get() or 'Overview'
            if bn=='Overview':
                return [(x,None) for x in self.project.csv_names()]
            return [(bn,None)]
        gids=self._export_scope_gids() or set()
        if not gids:return []
        grouped={}
        fallback=self.vars['batch'].get()
        for gid in gids:
            if '::' in gid:
                bn=gid.split('::',1)[0]
            elif fallback and fallback!='Overview':
                bn=fallback
            else:
                continue
            grouped.setdefault(bn,set()).add(gid)
        return [(bn,gg) for bn,gg in sorted(grouped.items())]

    def _export_preview_df(self):
        df=self.current_df
        if df is None or df.empty:return pd.DataFrame()
        scope=str(self.vars['export_scope'].get() or 'current')
        if scope in ('selected','checked'):
            gids=self._export_scope_gids() or set()
            if not gids:return df.iloc[0:0]
            return df[df['gid'].astype(str).isin(gids)] if 'gid' in df.columns else df.iloc[0:0]
        return df

    def _export_selected_classes(self):
        if self.vars['export_all_classes'].get():
            return None
        return {lab for lab,var in self._export_class_vars.items() if bool(var.get())}

    def _export_scope_changed(self):
        self._refresh_export_classes()

    def _export_toggle_all_classes(self):
        value=bool(self.vars['export_all_classes'].get())
        # The master checkbox is a true two-way control: checking selects every
        # class and unchecking clears every class, including classes hidden by
        # the search box.
        for var in self._export_class_vars.values():
            var.set(value)
        self._refresh_export_classes()

    def _export_class_toggled(self):
        if self._export_class_vars:
            self.vars['export_all_classes'].set(all(bool(v.get()) for v in self._export_class_vars.values()))

    def _refresh_export_classes(self):
        host=self._export_class_host
        if host is None:
            return
        try:
            if not host.winfo_exists():return
        except Exception:return
        for w in host.winfo_children():w.destroy()
        df=self._export_preview_df()
        counts={}
        if not df.empty and 'manual_label' in df.columns:
            counts=df['manual_label'].astype(str).value_counts().to_dict()
        # Preserve class choices seen previously; add labels from the current context.
        for lab in sorted(counts):
            if lab not in self._export_class_vars:
                self._export_class_vars[lab]=tk.BooleanVar(value=True)
        query=str(self.vars['export_class_search'].get() or '').strip().lower()
        labels=[x for x in sorted(self._export_class_vars) if not query or query in x.lower()]
        if self._export_class_count_lbl is not None:
            try:self._export_class_count_lbl.config(text=f'{len(counts)} {self._export_text("available_here")}')
            except Exception:pass
        for i,lab in enumerate(labels):
            row=tk.Frame(host,bg=C['card2'])
            row.grid(row=i,column=0,sticky='ew',padx=2,pady=1)
            host.columnconfigure(0,weight=1)
            var=self._export_class_vars[lab]
            cb=tk.Checkbutton(row,text=lab,variable=var,command=self._export_class_toggled,bg=C['card2'],fg=C['text'],
                              selectcolor=C['red'],activebackground=C['card2'],activeforeground='white',anchor='w')
            cb.pack(side='left',fill='x',expand=True)
            n=counts.get(lab)
            if n is not None:
                tk.Label(row,text=f'{int(n):,}'.replace(',','.'),bg=C['card2'],fg=C['muted'],font=('Sans',8)).pack(side='right',padx=(5,4))

    def _export_build_class_selector(self,parent,height=150):
        top=tk.Frame(parent,bg=C['card']);top.pack(fill='x',pady=(0,4))
        search=self._entry(top,self.vars['export_class_search']);search.pack(side='left',fill='x',expand=True,ipady=3)
        search.insert(0,'') if False else None
        search.bind('<KeyRelease>',lambda e:self._refresh_export_classes())
        tk.Checkbutton(top,text=self._export_text('select_all'),variable=self.vars['export_all_classes'],command=self._export_toggle_all_classes,
                       bg=C['card'],fg=C['text'],selectcolor=C['red'],activebackground=C['card'],activeforeground='white').pack(side='left',padx=(8,0))
        self._export_class_count_lbl=tk.Label(parent,text='',bg=C['card'],fg=C['muted'],font=('Sans',7))
        self._export_class_count_lbl.pack(anchor='w',pady=(0,3))
        wrap=tk.Frame(parent,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1,height=height)
        wrap.pack(fill='both',expand=True);wrap.pack_propagate(False)
        canvas=tk.Canvas(wrap,bg=C['card2'],highlightthickness=0)
        sb=ttk.Scrollbar(wrap,orient='vertical',command=canvas.yview)
        inner=tk.Frame(canvas,bg=C['card2'])
        win=canvas.create_window((0,0),window=inner,anchor='nw')
        inner.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e:canvas.itemconfigure(win,width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left',fill='both',expand=True);sb.pack(side='right',fill='y')
        self._export_class_host=inner
        # Local wheel handling for the compact class list.
        def wheel(e):
            if getattr(e,'num',None)==4:step=-1
            elif getattr(e,'num',None)==5:step=1
            else:
                d=int(getattr(e,'delta',0) or 0)
                if not d:return
                step=-1 if d>0 else 1
            canvas.yview_scroll(step,'units');return 'break'
        for ev in ('<MouseWheel>','<Button-4>','<Button-5>'):
            canvas.bind(ev,wheel,add='+');inner.bind(ev,wheel,add='+')
        self._refresh_export_classes()

    def _export_field_group(self,parent,title,items,columns=1):
        box=tk.Frame(parent,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1)
        tk.Label(box,text=title,bg=C['card2'],fg='#ffffff',font=('Sans',8,'bold')).grid(row=0,column=0,columnspan=columns,sticky='w',padx=8,pady=(7,3))
        for j in range(columns):box.columnconfigure(j,weight=1)
        for i,(key,label,_,_,_) in enumerate(items):
            r=1+i//columns;c=i%columns
            tk.Checkbutton(box,text=label,variable=self._export_field_vars[key],bg=C['card2'],fg=C['text'],selectcolor=C['red'],
                           activebackground=C['card2'],activeforeground='white',anchor='w',justify='left').grid(row=r,column=c,sticky='w',padx=8,pady=1)
        return box

    def _export_switch_view(self, mode):
        mode=str(mode or 'csv')
        if mode not in {'csv','images','training'}:
            mode='csv'
        self.vars['export_view'].set(mode)
        if self.current_tab=='export':
            self.show_tab('export',rebuild=True)

    def _export_tabs(self,parent):
        row=tk.Frame(parent,bg=C['bg']);row.grid(row=0,column=0,columnspan=(5 if (self.vars['export_view'].get() or 'csv')=='training' else 4),sticky='ew',padx=4,pady=(4,6))
        current=self.vars['export_view'].get() or 'csv'
        for mode,key in [('csv','tab_csv'),('images','tab_images'),('training','tab_training')]:
            col=C['purple'] if mode==current else C['card2']
            btn=self._btn(row,self._export_text(key),lambda m=mode:self._export_switch_view(m),col)
            btn.pack(side='left',ipadx=12,ipady=5,padx=(0,5))

    def _build_export_shared_filters(self,host,row=1):
        c,b=self._card(host,self._export_text('shared'));c.grid(row=row,column=0,columnspan=4,sticky='ew',padx=4,pady=(0,5))
        shared=tk.Frame(b,bg=C['card']);shared.pack(fill='x');shared.columnconfigure(0,weight=2);shared.columnconfigure(1,weight=3);shared.columnconfigure(2,weight=1)
        sf=tk.Frame(shared,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1);sf.grid(row=0,column=0,sticky='nsew',padx=(0,6))
        tk.Label(sf,text=self._export_text('scope'),bg=C['card2'],fg='#ffffff',font=('Sans',9,'bold')).pack(anchor='w',padx=9,pady=(8,4))
        bn=self.vars['batch'].get() or 'Overview'
        scope_items=[
            ('current',f"{self._export_text('current')}: {bn}"),
            ('selected',f"{self._export_text('selected')} ({len(self.allb_gids or self.selected_gids):,})".replace(',','.')),
            ('checked',f"{self._export_text('checked')} ({len(self.checked_gids):,})".replace(',','.')),
            ('all',self._export_text('all_batches')),
        ]
        for val,label in scope_items:
            tk.Radiobutton(sf,text=label,value=val,variable=self.vars['export_scope'],command=self._export_scope_changed,
                           bg=C['card2'],fg=C['text'],selectcolor=C['entry'],activebackground=C['card2'],activeforeground='white',anchor='w').pack(fill='x',anchor='w',padx=9,pady=2)
        cf=tk.Frame(shared,bg=C['card']);cf.grid(row=0,column=1,sticky='nsew',padx=(0,6))
        tk.Label(cf,text=self._export_text('classes'),bg=C['card'],fg='#ffffff',font=('Sans',9,'bold')).pack(anchor='w',pady=(1,5))
        self._export_build_class_selector(cf)
        info=tk.Frame(shared,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1);info.grid(row=0,column=2,sticky='nsew')
        tk.Label(info,text='ⓘ',bg=C['card2'],fg=C['cyan'],font=('Sans',15,'bold')).pack(anchor='w',padx=10,pady=(9,2))
        tk.Label(info,text=self._export_text('shared_note'),wraplength=240,justify='left',bg=C['card2'],fg=C['muted'],font=('Sans',8)).pack(anchor='w',padx=10,pady=(0,10))

    def _build_export_csv_panel(self,host,row=2):
        c,b=self._card(host,self._export_text('csv'));c.grid(row=row,column=0,columnspan=4,sticky='nsew',padx=4,pady=4)
        tk.Label(b,text=self._export_text('csv_sub'),bg=C['card'],fg=C['muted'],font=('Sans',8)).pack(anchor='w',pady=(0,6))
        tk.Label(b,text=self._export_text('fields'),bg=C['card'],fg='#ffffff',font=('Sans',9,'bold')).pack(anchor='w',pady=(0,5))
        specs=self._export_field_specs()
        grid=tk.Frame(b,bg=C['card']);grid.pack(fill='x');grid.columnconfigure(0,weight=1);grid.columnconfigure(1,weight=1);grid.columnconfigure(2,weight=1)
        self._export_field_group(grid,self._export_text('basic'),specs['basic']).grid(row=0,column=0,sticky='nsew',padx=(0,5))
        self._export_field_group(grid,self._export_text('labels'),specs['labels']).grid(row=0,column=1,sticky='nsew',padx=(0,5))
        self._export_field_group(grid,self._export_text('general'),specs['general']).grid(row=0,column=2,sticky='nsew')
        segbox=tk.Frame(b,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1);segbox.pack(fill='x',pady=(6,4))
        head=tk.Frame(segbox,bg=C['card2']);head.pack(fill='x',padx=8,pady=(7,3))
        tk.Label(head,text=self._export_text('seg'),bg=C['card2'],fg='#ffffff',font=('Sans',8,'bold')).pack(side='left')
        methods=self.project.available_segmentation_methods() if self.project else []
        if methods and self.vars['seg_method'].get() not in methods:self.vars['seg_method'].set(methods[0])
        displays=[SEGMENTATION_METHODS.get(m,m) for m in methods]
        self._export_seg_method_rev={SEGMENTATION_METHODS.get(m,m):m for m in methods}
        if methods:
            disp=tk.StringVar(value=SEGMENTATION_METHODS.get(self.vars['seg_method'].get(),self.vars['seg_method'].get()))
            cb=ttk.Combobox(head,textvariable=disp,values=displays,state='readonly',width=24,style='Dark.TCombobox');cb.pack(side='right')
            cb.bind('<<ComboboxSelected>>',lambda e:self.vars['seg_method'].set(self._export_seg_method_rev.get(disp.get(),self.vars['seg_method'].get())))
        else:
            tk.Label(head,text=self.tr('no_segmentation'),bg=C['card2'],fg=C['muted'],font=('Sans',8)).pack(side='right')
        pxrow=tk.Frame(segbox,bg=C['card2']);pxrow.pack(fill='x',padx=8,pady=(1,4))
        tk.Label(pxrow,text=self._export_text('pixel_size_export')+':',bg=C['card2'],fg=C['muted'],font=('Sans',8)).pack(side='left')
        e=self._entry(pxrow,self.vars['pixel_size']);e.config(width=10);e.pack(side='left',padx=(6,3),ipady=2)
        e.bind('<Return>',lambda ev:self._persist_project_pixel_size())
        e.bind('<FocusOut>',lambda ev:self._persist_project_pixel_size(silent=True))
        tk.Label(pxrow,text='µm/pixel',bg=C['card2'],fg=C['muted'],font=('Sans',8)).pack(side='left')
        sg=tk.Frame(segbox,bg=C['card2']);sg.pack(fill='x',padx=2,pady=(0,7))
        for j in range(3):sg.columnconfigure(j,weight=1)
        for i,(key,label,_,_,_) in enumerate(specs['seg']):
            tk.Checkbutton(sg,text=label,variable=self._export_field_vars[key],bg=C['card2'],fg=C['text'],selectcolor=C['red'],
                           activebackground=C['card2'],activeforeground='white',anchor='w').grid(row=i//3,column=i%3,sticky='w',padx=6,pady=1)
        tk.Checkbutton(b,text=self._export_text('metadata'),variable=self.vars['export_save_metadata'],bg=C['card'],fg=C['text'],selectcolor=C['red'],
                       activebackground=C['card'],activeforeground='white').pack(anchor='w',pady=(3,0))
        tk.Label(b,text=self._export_text('metadata_tip'),bg=C['card'],fg=C['muted'],font=('Sans',7),wraplength=1050,justify='left').pack(anchor='w',pady=(0,6))
        tk.Label(b,text=self._export_text('output'),bg=C['card'],fg='#ffffff',font=('Sans',9,'bold')).pack(anchor='w',pady=(4,3))
        orow=tk.Frame(b,bg=C['card']);orow.pack(fill='x')
        left=tk.Frame(orow,bg=C['card']);left.pack(side='left',fill='x',expand=True)
        self._label(left,self._export_text('output_dir')).pack(anchor='w')
        rr=tk.Frame(left,bg=C['card']);rr.pack(fill='x')
        self._entry(rr,self.vars['csv_output_dir']).pack(side='left',fill='x',expand=True,ipady=4)
        self._btn(rr,self.tr('browse'),self.browse_csv_export).pack(side='left',padx=(5,0))
        right=tk.Frame(orow,bg=C['card']);right.pack(side='left',fill='x',expand=True,padx=(10,0))
        self._label(right,self._export_text('filename')).pack(anchor='w')
        self._entry(right,self.vars['save_name']).pack(fill='x',ipady=4)
        self._btn(b,self._export_text('csv_button'),self.save_csv,C['blue']).pack(fill='x',ipady=6,pady=(8,2))

    def _build_export_images_panel(self,host,row=1):
        # This view intentionally mirrors the successful training-export workflow:
        # one clear step per column, while reusing the exact same variables and
        # export_images() backend used by the previous layout.
        for col, weight in enumerate((4, 5, 5)):
            host.columnconfigure(col, weight=weight, minsize=0)
        host.columnconfigure(3, weight=0, minsize=0)

        hero = tk.Frame(host, bg='#202025', highlightbackground='#3c3c45', highlightthickness=1)
        hero.grid(row=row, column=0, columnspan=3, sticky='ew', padx=4, pady=(0, 7))
        hero_left = tk.Frame(hero, bg='#202025')
        hero_left.pack(side='left', fill='x', expand=True, padx=13, pady=10)
        tk.Label(
            hero_left, text=self._export_text('images'), bg='#202025', fg='#eef6ff',
            font=('Sans', 12, 'bold')
        ).pack(anchor='w')
        tk.Label(
            hero_left, text=self._export_text('images_workflow_sub'), bg='#202025',
            fg='#ceced3', font=('Sans', 8), justify='left'
        ).pack(anchor='w', pady=(2, 0))
        info = tk.Label(
            hero, text='IAT  •  manual_label', bg='#173d60', fg=C['cyan'],
            font=('Sans', 8, 'bold'), padx=11, pady=5
        )
        info.pack(side='right', padx=12)

        cards_row = row + 1

        # 1. Scope/origin
        c1, b1 = self._card(host, self._export_text('scope'))
        c1.grid(row=cards_row, column=0, sticky='nsew', padx=(4, 3), pady=4)
        tk.Label(
            b1, text=self._export_text('images_scope_tip'), bg=C['card'], fg=C['muted'],
            font=('Sans', 8), wraplength=300, justify='left'
        ).pack(anchor='w', pady=(0, 8))
        scope_box = tk.Frame(b1, bg=C['card2'], highlightbackground=C['border'], highlightthickness=1)
        scope_box.pack(fill='x')
        bn = self.vars['batch'].get() or 'Overview'
        scope_items = [
            ('current', f"{self._export_text('current')}: {bn}"),
            ('selected', f"{self._export_text('selected')} ({len(self.allb_gids or self.selected_gids):,})".replace(',', '.')),
            ('checked', f"{self._export_text('checked')} ({len(self.checked_gids):,})".replace(',', '.')),
            ('all', self._export_text('all_batches')),
        ]
        for val, label in scope_items:
            tk.Radiobutton(
                scope_box, text=label, value=val, variable=self.vars['export_scope'],
                command=self._export_scope_changed, bg=C['card2'], fg=C['text'],
                selectcolor=C['blue2'], activebackground=C['card2'],
                activeforeground='white', anchor='w', justify='left'
            ).pack(fill='x', anchor='w', padx=9, pady=4)
        tip_box = tk.Frame(b1, bg='#19191d', highlightbackground='#2f2f36', highlightthickness=1)
        tip_box.pack(fill='x', pady=(10, 0))
        tk.Label(tip_box, text='i', bg=C['cyan'], fg='#0c0c0e', width=2, font=('Sans', 9, 'bold')).pack(side='left', padx=8, pady=8)
        tk.Label(
            tip_box, text=self._export_text('shared_note'), bg='#19191d', fg='#ceced3',
            wraplength=250, justify='left', font=('Sans', 7)
        ).pack(side='left', fill='x', expand=True, padx=(0, 8), pady=8)

        # 2. Label selection
        c2, b2 = self._card(host, self._export_text('classes'))
        c2.grid(row=cards_row, column=1, sticky='nsew', padx=3, pady=4)
        tk.Label(
            b2, text=self._export_text('images_labels_tip'), bg=C['card'], fg=C['muted'],
            font=('Sans', 8), wraplength=390, justify='left'
        ).pack(anchor='w', pady=(0, 8))
        self._export_build_class_selector(b2, height=310)

        # 3. Options + destination
        c3, b3 = self._card(host, self._export_text('img_opts'))
        c3.grid(row=cards_row, column=2, sticky='nsew', padx=(3, 4), pady=4)
        tk.Label(
            b3, text=self._export_text('images_options_tip'), bg=C['card'], fg=C['muted'],
            font=('Sans', 8), wraplength=390, justify='left'
        ).pack(anchor='w', pady=(0, 8))

        opts = tk.Frame(b3, bg=C['card2'], highlightbackground=C['border'], highlightthickness=1)
        opts.pack(fill='x')
        option_rows = [
            (self._export_text('subfolders'), self.vars['export_create_subfolders'], C['purple']),
            (self._export_text('keep_names'), self.vars['export_keep_names'], C['green']),
            (self._export_text('include_outline'), self.vars['export_include_outline'], C['cyan']),
            (self._export_text('only_segmented'), self.vars['export_only_segmented'], C['amber']),
        ]
        for label, var, accent in option_rows:
            row_box = tk.Frame(opts, bg=C['card2'])
            row_box.pack(fill='x', padx=8, pady=2)
            tk.Label(row_box, text='●', bg=C['card2'], fg=accent, font=('Sans', 8, 'bold')).pack(side='left', padx=(0, 3))
            tk.Checkbutton(
                row_box, text=label, variable=var, bg=C['card2'], fg=C['text'],
                selectcolor=C['entry'], activebackground=C['card2'],
                activeforeground='white', wraplength=355, justify='left', anchor='w'
            ).pack(side='left', fill='x', expand=True)

        output_box = tk.Frame(b3, bg=C['card2'], highlightbackground=C['border'], highlightthickness=1)
        output_box.pack(fill='x', pady=(10, 0))
        tk.Label(
            output_box, text=self._export_text('img_output'), bg=C['card2'], fg='#ffffff',
            font=('Sans', 9, 'bold')
        ).pack(anchor='w', padx=9, pady=(8, 4))
        rr = tk.Frame(output_box, bg=C['card2'])
        rr.pack(fill='x', padx=9, pady=(0, 9))
        self._entry(rr, self.vars['export_dir']).pack(side='left', fill='x', expand=True, ipady=4)
        self._btn(rr, self.tr('browse'), self.browse_export).pack(side='left', padx=(5, 0))

        self._btn(b3, self._export_text('images_button'), self.export_images, C['green']).pack(fill='x', ipady=8, pady=(11, 3))

    def _build_export(self):
        self._ensure_export_field_vars()
        if self.project:
            default_dir=str(self.project.project_dir/'exports')
            if not self.vars['csv_output_dir'].get():self.vars['csv_output_dir'].set(default_dir)
            if not self.vars['export_dir'].get():self.vars['export_dir'].set(default_dir)
            if not self.vars['training_dir'].get():self.vars['training_dir'].set(str(self.project.project_dir/'exports'/'training'))
        if not self.vars['save_name'].get():
            self.vars['save_name'].set(f'iat_export_{datetime.now().strftime("%Y-%m-%d")}')
        page=ScrollFrame(self.content,C['bg']);page.pack(fill='both',expand=True)
        host=page.inner
        view=self.vars['export_view'].get() or 'csv'
        if view=='images':
            for i in range(3):host.columnconfigure(i,weight=1)
            host.columnconfigure(3,weight=0,minsize=0)
        else:
            for i in range(4):host.columnconfigure(i,weight=1)
        self._export_tabs(host)
        if view=='training':
            self._build_training_export(host,row=1)
            return
        if view=='images':
            self._build_export_images_panel(host,row=1)
        else:
            self._build_export_shared_filters(host,row=1)
            self._build_export_csv_panel(host,row=2)

    def _export_read_batch(self,bn,columns=None):
        if self.project is None:return pd.DataFrame()
        requested_list=list(dict.fromkeys(columns or ()))
        requested=set(requested_list)
        # Reuse cached batches so unsaved relabeling is included. For untouched
        # batches, read directly from disk instead of filling batch_cache during
        # an all-batches export.
        if bn in self.project.batch_cache:
            frame=self.project.batch_cache[bn].copy()
            if not requested:return frame
            keep=[c for c in requested_list if c in frame.columns]
            # ``gid`` is always available in cached project frames.
            return frame.loc[:,keep].copy() if keep else frame.iloc[:,0:0].copy()
        p=self.project.dataframes_dir/bn
        if not p.exists():return pd.DataFrame()
        if not requested:
            return ensure_required_columns(pd.read_csv(p,low_memory=False),bn)

        # Read only the columns needed by lightweight operations such as image
        # and training export.  Include legacy label/id dependencies so
        # ensure_required_columns can reconstruct the stable IAT contract.
        dependencies={
            'names','manual_label','correct_label','preds1','pred','custom_data',
            'custom_data_orig','source_label','klass','csv_basename','batch',
        }
        wanted=requested|dependencies
        frame=pd.read_csv(p,usecols=lambda c:c in wanted,low_memory=False)
        frame=ensure_required_columns(frame,bn)
        keep=[c for c in requested_list if c in frame.columns]
        return frame.loc[:,keep].copy()

    def _export_filter_frame(self,df,gids=None):
        if df is None or df.empty:return pd.DataFrame()
        out=df
        if gids is not None:
            if 'gid' not in out.columns:return out.iloc[0:0]
            out=out[out['gid'].astype(str).isin(gids)]
        classes=self._export_selected_classes()
        if classes is not None:
            if not classes or 'manual_label' not in out.columns:return out.iloc[0:0]
            out=out[out['manual_label'].astype(str).isin(classes)]
        return out.copy()

    def _export_attach_segmentation(self,df,bn,method):
        if df.empty or not self.project:return df
        p=self.project.segmentation_csv_path(bn,method)
        if not p.exists():return df
        try:
            seg=pd.read_csv(p,low_memory=False)
        except Exception:return df
        if seg.empty or 'Image Name' not in seg.columns:return df
        cols=[c for c in seg.columns if c!='Image Name']
        return df.merge(seg[['Image Name']+cols],how='left',left_on='names',right_on='Image Name')

    def _export_selected_fields(self):
        self._ensure_export_field_vars()
        specs=self._export_field_specs()
        return [(key,label,src,out,group) for group,items in specs.items() for key,label,src,out,_ in items if self._export_field_vars[key].get()]

    def _export_needs_segmentation(self):
        return any(group=='seg' for *_,group in self._export_selected_fields())

    def _export_prepare_frame(self,df,selected,px):
        out=pd.DataFrame(index=df.index)
        fallback_px=self._segmentation_pixel_size(None,px)
        settings=load_project_settings(self.project.project_dir) if self.project else {}
        if 'pixel_size_um_per_px' in settings:
            # Current project calibration is authoritative for physical-unit
            # conversion; stored segmentation pixel size remains provenance.
            px_series=pd.Series(fallback_px,index=df.index,dtype=float)
        elif 'Pixel Size (um/pixel)' in df.columns:
            px_series=pd.to_numeric(df['Pixel Size (um/pixel)'],errors='coerce')
            px_series=px_series.where(np.isfinite(px_series)&(px_series>0),fallback_px)
        else:
            px_series=pd.Series(fallback_px,index=df.index,dtype=float)
        for key,_,src,name,group in selected:
            if group!='seg':
                if name:
                    out[name]=df[src] if src in df.columns else np.nan
                continue
            vals=pd.to_numeric(df[src],errors='coerce') if src in df.columns else pd.Series(np.nan,index=df.index)
            # Physical units: lengths × s, areas × s², volumes × s³ (s = µm/pixel).
            linear={'perimeter':'perimeter','esd':'esd','feret_max':'feret_max','feret_min':'feret_min',
                    'major':'major_axis','minor':'minor_axis'}
            if key=='area':
                out['area_px2']=vals
                out['area_um2']=vals*(px_series**2)
            elif key=='biovolume':
                out['biovolume_px3']=vals
                out['biovolume_um3']=vals*(px_series**3)
            elif key in linear:
                out[f'{linear[key]}_px']=vals
                out[f'{linear[key]}_um']=vals*px_series
            elif name:
                if key=='validation' and src in df.columns:
                    out[name]=df[src]
                else:
                    out[name]=vals
        return out.reset_index(drop=True)

    def browse_csv_export(self):
        initial=self.vars['csv_output_dir'].get() or (str(self.project.project_dir/'exports') if self.project else str(self.app.project_root))
        p=filedialog.askdirectory(initialdir=initial)
        if p:self.vars['csv_output_dir'].set(p)

    def save_csv(self):
        if not self.project:return
        selected=self._export_selected_fields()
        if not selected:
            messagebox.showwarning(self._export_text('csv'),self._export_text('no_fields'));return
        if not self.vars['export_all_classes'].get() and not self._export_selected_classes():
            messagebox.showwarning(self._export_text('csv'),self._export_text('no_classes'));return
        plan=self._export_scope_plan()
        if not plan:
            messagebox.showwarning(self._export_text('csv'),self._export_text('empty_scope'));return
        outdir=self.vars['csv_output_dir'].get().strip()
        if not outdir:
            self.browse_csv_export();outdir=self.vars['csv_output_dir'].get().strip()
        if not outdir:return
        name=self.vars['save_name'].get().strip() or f'iat_export_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}'
        name=Path(name).name
        if not name.lower().endswith('.csv'):name+='.csv'
        outpath=Path(outdir).expanduser()/name
        outpath.parent.mkdir(parents=True,exist_ok=True)
        if outpath.exists() and not messagebox.askyesno(self._export_text('csv'),f'{outpath.name} já existe.\n\nSubstituir?'):
            return
        try:px=parse_pixel_size(self.vars['pixel_size'].get())
        except Exception:px=DEFAULT_PIXEL_SIZE_UM_PER_PX
        method=self.vars['seg_method'].get()
        need_seg=self._export_needs_segmentation()
        save_meta=bool(self.vars['export_save_metadata'].get())
        classes=self._export_selected_classes()
        scope=str(self.vars['export_scope'].get() or 'current')
        batch_or_view=str(self.vars['batch'].get())
        fields=[x[0] for x in selected]
        tmp=outpath.with_suffix(outpath.suffix+'.part')
        def work(progress):
            if tmp.exists():
                try:tmp.unlink()
                except Exception:pass
            first=True;rows=0
            total=len(plan)
            for i,(bn,gids) in enumerate(plan,1):
                df=self._export_filter_frame(self._export_read_batch(bn),gids)
                if need_seg and not df.empty:
                    df=self._export_attach_segmentation(df,bn,method)
                shaped=self._export_prepare_frame(df,selected,px)
                if not shaped.empty:
                    shaped.to_csv(tmp,index=False,mode='w' if first else 'a',header=first)
                    first=False;rows+=len(shaped)
                progress(i,total,bn)
            if first:
                # Still emit a valid header-only CSV when filters produce zero rows.
                empty=self._export_prepare_frame(pd.DataFrame(),selected,px)
                empty.to_csv(tmp,index=False)
            os.replace(tmp,outpath)
            meta_path=None
            if save_meta:
                meta_path=outpath.with_suffix('.json')
                meta={
                    'project':self.project.name,
                    'project_dir':str(self.project.project_dir),
                    'generated_at':datetime.now().astimezone().isoformat(),
                    'scope':scope,
                    'batch_or_view':batch_or_view,
                    'classes':'all' if classes is None else sorted(classes),
                    'rows_exported':int(rows),
                    'fields':fields,
                    'segmentation_method':method if need_seg else None,
                    'segmentation_method_label':SEGMENTATION_METHODS.get(method,method) if need_seg else None,
                    'pixel_size_fallback_um_per_px':float(px) if need_seg else None,
                    'pixel_size_source':('project calibration when configured; otherwise stored segmentation calibration/fallback' if need_seg else None),
                    'physical_unit_note':'area_um2 = area_px2 * pixel_size^2; feret_um = feret_px * pixel_size',
                    'csv_file':outpath.name,
                }
                meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
            return outpath,meta_path,rows
        def done(res):
            path,meta,rows=res;self.saved_count+=1
            msg=f"{self._export_text('csv_done')}: {path.name} • {rows:,} {self._export_text('rows')}".replace(',','.')
            if meta:msg+=f' • JSON: {meta.name}'
            self._log(msg,'success')
        self._run_operation(self.tr('saving'),work,done)

    def browse_export(self):
        p=filedialog.askdirectory(initialdir=self.vars['export_dir'].get() or str(self.project.project_dir if self.project else self.app.project_root))
        if p:self.vars['export_dir'].set(p)

    def _export_valid_segmented_names(self,bn,method):
        if not self.project:return set()
        p=self.project.segmentation_csv_path(bn,method)
        if not p.exists():return set()
        try:
            s=pd.read_csv(p,usecols=lambda c:c in ('Image Name','Validation State'))
            if 'Image Name' not in s.columns:return set()
            if 'Validation State' in s.columns:
                v=s['Validation State']
                if v.dtype==bool:mask=v
                else:mask=v.astype(str).str.lower().isin({'true','1','t','yes'})
                s=s[mask]
            return set(s['Image Name'].dropna().astype(str))
        except Exception:return set()

    def _export_outline_map(self,bn,method):
        if not self.project:return {}
        p=self.project.segmentation_cache_path(bn,method)
        if not p.exists():return {}
        try:return json.loads(p.read_text(encoding='utf-8'))
        except Exception:return {}

    def _export_unique_path(path):
        path=Path(path)
        if not path.exists():return path
        for i in range(2,1000000):
            p=path.with_name(f'{path.stem}_{i}{path.suffix}')
            if not p.exists():return p
        return path.with_name(f'{path.stem}_{int(time.time())}{path.suffix}')

    def export_images(self):
        if not self.project:return
        if not self.vars['export_all_classes'].get() and not self._export_selected_classes():
            messagebox.showwarning(self._export_text('images'),self._export_text('no_classes'));return
        plan=self._export_scope_plan()
        if not plan:
            messagebox.showwarning(self._export_text('images'),self._export_text('empty_scope'));return
        out=self.vars['export_dir'].get().strip()
        if not out:
            self.browse_export();out=self.vars['export_dir'].get().strip()
        if not out:return
        outdir=Path(out).expanduser();outdir.mkdir(parents=True,exist_ok=True)
        method=self.vars['seg_method'].get()
        subfolders=bool(self.vars['export_create_subfolders'].get())
        keep_names=bool(self.vars['export_keep_names'].get())
        include_outline=bool(self.vars['export_include_outline'].get())
        only_segmented=bool(self.vars['export_only_segmented'].get())
        selected_classes=self._export_selected_classes()

        def work(progress):
            # Phase 1: resolve source paths and destination names once.  Doing
            # this before copying lets the resolver reuse its caches/index and
            # lets the copy engine favor source-directory locality.
            self.project.reset_image_resolve_stats()
            jobs=[];outline_specs={};missing=0;seen_index=0;reserved=set()
            total_batches=max(1,len(plan))

            def reserve(path):
                path=Path(path)
                candidate=path
                key=os.path.normcase(os.path.abspath(str(candidate)))
                if not candidate.exists() and key not in reserved:
                    reserved.add(key);return candidate
                for i in range(2,1000000):
                    candidate=path.with_name(f'{path.stem}_{i}{path.suffix}')
                    key=os.path.normcase(os.path.abspath(str(candidate)))
                    if not candidate.exists() and key not in reserved:
                        reserved.add(key);return candidate
                candidate=path.with_name(f'{path.stem}_{int(time.time())}{path.suffix}')
                reserved.add(os.path.normcase(os.path.abspath(str(candidate))))
                return candidate

            for bi,(bn,gids) in enumerate(plan,1):
                df=self._export_read_batch(bn,columns=('gid','names','manual_label'))
                if gids is not None:
                    if 'gid' not in df.columns:
                        df=df.iloc[0:0]
                    else:
                        df=df[df['gid'].astype(str).isin(gids)]
                if selected_classes is not None:
                    if not selected_classes or 'manual_label' not in df.columns:
                        df=df.iloc[0:0]
                    else:
                        df=df[df['manual_label'].astype(str).isin(selected_classes)]
                valid=self._export_valid_segmented_names(bn,method) if (only_segmented or include_outline) else set()
                outlines=self._export_outline_map(bn,method) if include_outline else {}
                if only_segmented:
                    df=df[df['names'].astype(str).isin(valid)] if 'names' in df.columns else df.iloc[0:0]

                for row in df.itertuples(index=False):
                    name=str(getattr(row,'names',''))
                    lab=str(getattr(row,'manual_label','unlabeled') or 'unlabeled')
                    src=self.project.resolve_image(bn,name)
                    if not src or not src.is_file():
                        missing+=1
                        continue
                    dest_dir=outdir/(sanitize_filename(lab) or 'unlabeled') if subfolders else outdir
                    seen_index+=1
                    if keep_names:
                        dest=dest_dir/src.name
                        if dest.exists() or os.path.normcase(os.path.abspath(str(dest))) in reserved:
                            dest=dest_dir/f'{src.stem}__{Path(bn).stem}{src.suffix}'
                    else:
                        dest=dest_dir/f'image_{seen_index:08d}{src.suffix.lower()}'
                    dest=reserve(dest)
                    token=len(jobs)
                    jobs.append(CopyJob(Path(src),dest,token))
                    if include_outline and name in outlines:
                        outline_specs[token]=(Path(src),dest,outlines[name])
                progress(int((bi/total_batches)*20),100,f'Preparando {bi}/{total_batches}: {bn}')

            # Phase 2: bounded concurrent copies.  Two workers are deliberately
            # conservative for unknown storage and can be overridden internally
            # with ILT_EXPORT_WORKERS when benchmarking a specific workstation.
            def copy_progress(cur,total_files,label=''):
                pct=20.0+(float(cur)/max(float(total_files),1.0))*75.0
                progress(pct,100,f'Copiando {cur:,}/{total_files:,}'.replace(',','.'))

            report=copy_jobs(
                jobs,
                progress=copy_progress,
                workers=recommended_io_workers(),
                preserve_metadata=True,
            )

            # Phase 3: optional segmentation overlays.  This is intentionally
            # separate from the copy pool so ordinary export stays I/O-focused.
            outlined=0
            if include_outline and outline_specs and Image is not None and ImageDraw is not None:
                successful=report.succeeded_tokens
                todo=[(token,spec) for token,spec in outline_specs.items() if token in successful]
                for oi,(token,(src,dest,rec)) in enumerate(todo,1):
                    try:
                        pts=rec.get('points') or []
                        if len(pts)>1:
                            with Image.open(src) as raw:
                                im=raw.convert('RGB')
                            ImageDraw.Draw(im).line(
                                [tuple(p) for p in pts]+[tuple(pts[0])],
                                fill=(255,40,60),
                                width=max(1,round(max(im.size)/700)),
                            )
                            overlay=reserve(dest.parent/f'{dest.stem}__segmentation.png')
                            im.save(overlay);outlined+=1
                    except Exception:
                        pass
                    if oi==len(todo) or oi%25==0:
                        progress(95.0+(oi/max(len(todo),1))*5.0,100,f'Contornos {oi:,}/{len(todo):,}'.replace(',','.'))
            progress(100,100,'Concluído')
            return {
                'copied':report.copied,
                'failed':report.failed,
                'missing':missing,
                'outlined':outlined,
                'report':report,
                'resolver':self.project.image_resolve_stats(),
            }

        def done(res):
            n=int(res['copied']);outlined=int(res['outlined']);missing=int(res['missing']);failed=int(res['failed'])
            report=res['report'];resolver=res['resolver']
            msg=f"{self._export_text('images_done')}: {n:,}".replace(',','.')
            if include_outline:msg+=f' • contornos: {outlined:,}'.replace(',','.')
            if missing:msg+=f' • ausentes: {missing:,}'.replace(',','.')
            if failed:msg+=f' • falhas de cópia: {failed:,}'.replace(',','.')
            self._log(msg,'success' if not failed else 'warn')
            perf=(
                f"Export I/O: {report.files_per_second:.1f} arquivos/s • "
                f"{report.mib_per_second:.1f} MiB/s • {report.elapsed_seconds:.1f}s • "
                f"{report.workers} worker(s)"
            )
            self._log(perf,'info')
            if resolver.get('global_index_builds'):
                self._log(
                    "Resolução de imagens: índice global construído uma vez "
                    f"• hits índice={resolver.get('global_index_hit',0)} "
                    f"• ausentes={resolver.get('missing',0)}",
                    'info',
                )
        self._run_operation(self.tr('exporting'),work,done)

