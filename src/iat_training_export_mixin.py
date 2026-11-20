#!/usr/bin/env python3
"""Training-dataset export mixin for :mod:`iat_workspace`."""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from iat_backend import sanitize_filename
from iat_io import CopyJob, copy_jobs, recommended_io_workers
from training_export import plan_training_split
from ui_resources import IAT_COLORS as C


class IATTrainingExportMixin:
    @staticmethod
    def _training_batch_from_gid(gid, fallback=''):
        s=str(gid or '')
        return s.split('::',1)[0] if '::' in s else str(fallback or '')

    def _training_scope_frame(self,progress=None,snapshot=None):
        if self.project is None:return pd.DataFrame()
        snapshot=snapshot or {}
        scope=str(snapshot.get('scope', self.vars['training_scope'].get() if not snapshot else 'all') or 'all')
        current_batch=str(snapshot.get('current_batch', self.vars['batch'].get() if not snapshot else 'Overview') or 'Overview')
        current_df=snapshot.get('current_df',self.current_df)
        selected_ids=snapshot.get('selected_ids',self.allb_gids or self.selected_gids)
        batch_selection=set(snapshot.get('batch_selection',self._training_batch_selection))
        wanted=['gid','names','manual_label','custom_data','custom_data_orig','source_label','preds1','csv_basename','batch']

        def compact(df,fallback=''):
            if df is None or df.empty:return pd.DataFrame()
            cols=[c for c in wanted if c in df.columns]
            out=df.loc[:,cols].copy()
            if 'gid' not in out.columns:
                key=out['custom_data'].astype(str) if 'custom_data' in out.columns else out['names'].astype(str)
                out['gid']=str(fallback)+'::'+key
            if 'custom_data_orig' not in out.columns:
                if 'custom_data' in out.columns:out['custom_data_orig']=out['custom_data'].astype(str)
                else:out['custom_data_orig']=out['names'].astype(str)
            if fallback:
                out['source_batch']=str(fallback)
            else:
                out['source_batch']=out['gid'].astype(str).map(lambda x:self._training_batch_from_gid(x,''))
            return out

        if scope=='current':
            return compact(current_df,'' if current_batch=='Overview' else current_batch)

        if scope=='selected':
            gids=set(map(str,selected_ids))
            if not gids:return pd.DataFrame()
            base=compact(current_df,'' if current_batch=='Overview' else current_batch)
            found=base[base['gid'].astype(str).isin(gids)].copy() if ('gid' in base.columns and not base.empty) else pd.DataFrame()
            found_gids=set(found['gid'].astype(str)) if (not found.empty and 'gid' in found.columns) else set()
            missing=gids-found_gids
            extra=[]
            if missing:
                grouped={}
                for gid in missing:
                    bn=self._training_batch_from_gid(gid,'' if current_batch=='Overview' else current_batch)
                    if bn:grouped.setdefault(bn,set()).add(gid)
                for bn,gg in sorted(grouped.items()):
                    try:d=compact(self._export_read_batch(bn,columns=wanted),bn)
                    except Exception:continue
                    if 'gid' in d.columns:
                        part=d[d['gid'].astype(str).isin(gg)].copy()
                        if not part.empty:extra.append(part)
            parts=([found] if not found.empty else [])+extra
            if not parts:return pd.DataFrame()
            out=pd.concat(parts,ignore_index=True,sort=False,copy=False)
            return out.drop_duplicates('gid',keep='first').reset_index(drop=True)

        if scope=='batches':
            names=[bn for bn in self.project.csv_names() if bn in batch_selection]
            if not names:return pd.DataFrame()
        else:
            names=self.project.csv_names()

        # Reuse an already-loaded Overview when it exactly represents all batches.
        if scope=='all' and current_batch=='Overview' and current_df is not None and not current_df.empty:
            return compact(current_df,'')

        parts=[]
        total_names=max(1,len(names))
        for index,bn in enumerate(names,1):
            try:d=self._export_read_batch(bn,columns=wanted)
            except Exception:d=pd.DataFrame()
            if d is not None and not d.empty:parts.append(compact(d,bn))
            if progress:
                progress(index,total_names,f'{bn} ({index}/{total_names})')
        if not parts:return pd.DataFrame()
        out=pd.concat(parts,ignore_index=True,sort=False,copy=False)
        if 'gid' in out.columns:out=out.drop_duplicates('gid',keep='first')
        return out.reset_index(drop=True)

    def _training_context_signature(self):
        if self.project is None:
            return None
        scope=str(self.vars['training_scope'].get() or 'all')
        current_batch=str(self.vars['batch'].get() or 'Overview')
        selected=self.allb_gids or self.selected_gids or ()
        try:
            selected_hash=hash(frozenset(map(str,selected))) if scope=='selected' else 0
        except Exception:
            selected_hash=len(selected)
        dirty=tuple(sorted(map(str,getattr(self.project,'dirty_batches',()))))
        return (
            str(self.project.project_dir),scope,current_batch,
            tuple(sorted(map(str,self._training_batch_selection))),
            selected_hash,dirty,
        )

    def _training_render_loading(self,host,row=1):
        box=tk.Frame(host,bg=C['card'],highlightbackground=C['border'],highlightthickness=1)
        box.grid(row=row,column=0,columnspan=5,sticky='ew',padx=4,pady=4)
        tk.Label(
            box,text=self._export_text('training_title'),bg=C['card'],fg=C['text'],
            font=('Sans',13,'bold')
        ).pack(anchor='w',padx=14,pady=(14,4))
        tk.Label(
            box,text=self._export_text('training_loading'),
            bg=C['card'],fg=C['muted'],font=('Sans',9)
        ).pack(anchor='w',padx=14,pady=(0,8))
        ttk.Progressbar(box,maximum=100,variable=self.progress_var,style='Blue.Horizontal.TProgressbar').pack(
            fill='x',padx=14,pady=(0,14)
        )

    def _training_start_context_load(self,signature):
        if getattr(self,'_training_context_loading',False):
            return
        self._training_context_loading=True
        # Snapshot every Tk-dependent value on the UI thread.  Worker threads
        # must never call StringVar/BooleanVar.get() directly.
        snapshot={
            'scope':str(self.vars['training_scope'].get() or 'all'),
            'current_batch':str(self.vars['batch'].get() or 'Overview'),
            'selected_ids':tuple(map(str,self.allb_gids or self.selected_gids or ())),
            'batch_selection':tuple(map(str,self._training_batch_selection)),
            'current_df':self.current_df,
        }

        def work(progress):
            try:
                frame=self._training_scope_frame(progress=progress,snapshot=snapshot)
                return signature,frame,None
            except Exception as exc:
                return signature,pd.DataFrame(),exc

        def done(result):
            sig,frame,error=result
            self._training_context_loading=False
            if error is not None:
                self._training_context_df=pd.DataFrame()
                self._training_context_key=None
                self._log(f'Training export: {error}','error')
                return
            # Do not install stale data if the user changed project/scope while
            # the background load was running.
            if sig!=self._training_context_signature():
                self._training_context_key=None
                if self.current_tab=='export' and self.vars['export_view'].get()=='training':
                    self.show_tab('export',rebuild=True)
                return
            self._training_context_df=frame
            self._training_context_key=sig
            counts={}
            if frame is not None and not frame.empty and 'manual_label' in frame.columns:
                counts=frame['manual_label'].fillna('').astype(str).value_counts().to_dict()
            for lab in sorted(counts):
                self._training_class_vars.setdefault(lab,tk.BooleanVar(value=True))
                self._training_group_vars.setdefault(lab,tk.StringVar(value=lab))
            if self.current_tab=='export' and self.vars['export_view'].get()=='training':
                self.show_tab('export',rebuild=True)

        self._run_operation(self._export_text('training_loading'),work,done)

    def _training_scope_changed(self):
        self._training_context_key=None
        if self.current_tab=='export' and self.vars['export_view'].get()=='training':
            self.show_tab('export',rebuild=True)
        else:
            self._training_refresh_context(rebuild=False)

    def _training_select_batches(self):
        if not self.project:return
        win=tk.Toplevel(self);win.title(self._export_text('training_choose_batches'));win.configure(bg=C['card']);win.geometry('560x520')
        win.transient(self.winfo_toplevel());win.grab_set()
        tk.Label(win,text=self._export_text('training_choose_batches'),bg=C['card'],fg=C['text'],font=('Sans',11,'bold')).pack(anchor='w',padx=12,pady=(12,6))
        frame=tk.Frame(win,bg=C['card']);frame.pack(fill='both',expand=True,padx=12,pady=4)
        lb=tk.Listbox(frame,selectmode='multiple',bg=C['entry'],fg=C['text'],selectbackground=C['blue2'],selectforeground='white',relief='flat',font=('Sans',9))
        sb=ttk.Scrollbar(frame,orient='vertical',command=lb.yview);lb.configure(yscrollcommand=sb.set)
        lb.pack(side='left',fill='both',expand=True);sb.pack(side='right',fill='y')
        names=self.project.csv_names()
        for i,bn in enumerate(names):
            lb.insert('end',bn)
            if bn in self._training_batch_selection:lb.selection_set(i)
        buttons=tk.Frame(win,bg=C['card']);buttons.pack(fill='x',padx=12,pady=(6,12))
        def all_():
            lb.selection_set(0,'end')
        def none_():
            lb.selection_clear(0,'end')
        def ok():
            self._training_batch_selection={names[i] for i in lb.curselection()}
            win.destroy()
            self.vars['training_scope'].set('batches')
            self._training_context_key=None
            self.show_tab('export',rebuild=True)
        self._btn(buttons,self._export_text('select_all'),all_).pack(side='left')
        self._btn(buttons,self.tr('clear'),none_).pack(side='left',padx=(5,0))
        self._btn(buttons,'OK',ok,C['green']).pack(side='right')

    def _training_selected_labels(self):
        active=set()
        if self._training_context_df is not None and not self._training_context_df.empty and 'manual_label' in self._training_context_df.columns:
            active=set(self._training_context_df['manual_label'].fillna('').astype(str).unique())
        return [lab for lab,var in sorted(self._training_class_vars.items()) if lab in active and bool(var.get())]

    def _training_toggle_all(self):
        value=bool(self.vars['training_all_classes'].get())
        for lab,var in self._training_class_vars.items():
            if lab in set(self._training_context_df.get('manual_label',pd.Series(dtype=str)).astype(str).unique()):
                var.set(value)
        self._training_rebuild_group_table()

    def _training_class_toggled(self):
        labels=set(self._training_context_df.get('manual_label',pd.Series(dtype=str)).astype(str).unique())
        active=[self._training_class_vars[x].get() for x in labels if x in self._training_class_vars]
        self.vars['training_all_classes'].set(bool(active) and all(active))
        self._training_rebuild_group_table()

    def _training_refresh_context(self,rebuild=False):
        try:self._training_context_df=self._training_scope_frame()
        except Exception as e:
            self._training_context_df=pd.DataFrame();self._log(f'Training export: {e}','error')
        counts={}
        if not self._training_context_df.empty and 'manual_label' in self._training_context_df.columns:
            counts=self._training_context_df['manual_label'].fillna('').astype(str).value_counts().to_dict()
        for lab in sorted(counts):
            if lab not in self._training_class_vars:self._training_class_vars[lab]=tk.BooleanVar(value=True)
            if lab not in self._training_group_vars:self._training_group_vars[lab]=tk.StringVar(value=lab)
        if rebuild and self.current_tab=='export' and self.vars['export_view'].get()=='training':
            self.show_tab('export',rebuild=True)
            return
        self._training_render_class_list(counts)
        self._training_rebuild_group_table()

    def _training_render_class_list(self,counts=None):
        host=self._training_class_host
        if host is None:return
        try:
            if not host.winfo_exists():return
        except Exception:return
        for w in host.winfo_children():w.destroy()
        if counts is None:
            counts=self._training_context_df['manual_label'].fillna('').astype(str).value_counts().to_dict() if (not self._training_context_df.empty and 'manual_label' in self._training_context_df.columns) else {}
        for i,lab in enumerate(sorted(counts)):
            row=tk.Frame(host,bg=C['card2']);row.grid(row=i,column=0,sticky='ew',padx=2,pady=1);host.columnconfigure(0,weight=1)
            var=self._training_class_vars.setdefault(lab,tk.BooleanVar(value=True))
            tk.Checkbutton(row,text=lab,variable=var,command=self._training_class_toggled,bg=C['card2'],fg=C['text'],selectcolor=C['red'],
                           activebackground=C['card2'],activeforeground='white',anchor='w').pack(side='left',fill='x',expand=True)
            tk.Label(row,text=f"{int(counts[lab]):,}".replace(',','.'),bg=C['card2'],fg=C['muted'],font=('Sans',8)).pack(side='right',padx=(5,4))

    def _training_scroll_area(self,parent,height=210):
        wrap=tk.Frame(parent,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1,height=height)
        wrap.pack(fill='both',expand=True);wrap.pack_propagate(False)
        canvas=tk.Canvas(wrap,bg=C['card2'],highlightthickness=0)
        # Use an explicit classic scrollbar here: on Windows the themed ttk
        # scrollbar could become almost invisible against the dark card.
        sb=tk.Scrollbar(
            wrap,orient='vertical',command=canvas.yview,width=14,
            bg=C['blue2'],activebackground=C['blue'],troughcolor=C['entry'],
            relief='flat',bd=0,highlightthickness=0
        )
        inner=tk.Frame(canvas,bg=C['card2']);win=canvas.create_window((0,0),window=inner,anchor='nw')
        # Tag the training-only scroll area so the global wheel router can
        # recognize labels/entries nested inside it.
        inner._training_scroll_canvas=canvas
        wrap._training_scroll_canvas=canvas
        inner.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e:canvas.itemconfigure(win,width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right',fill='y')
        canvas.pack(side='left',fill='both',expand=True)
        return inner

    def _training_palette(self):
        return ['#ff4d8d','#23d5ff','#4cff72','#ffb020','#9a4dff','#f54dff','#00d0a8','#ff704d']

    def _training_reset_group_names(self):
        for lab in self._training_selected_labels():
            self._training_group_vars.setdefault(lab,tk.StringVar(value=lab)).set(lab)
        self._training_grouping_updated()

    def _training_rebuild_group_table(self):
        host=self._training_group_host
        if host is None:return
        try:
            if not host.winfo_exists():return
        except Exception:return
        for w in host.winfo_children():w.destroy()
        labels=self._training_selected_labels()
        counts=self._training_context_df['manual_label'].fillna('').astype(str).value_counts().to_dict() if (self._training_context_df is not None and not self._training_context_df.empty and 'manual_label' in self._training_context_df.columns) else {}
        host.columnconfigure(0,weight=4)
        host.columnconfigure(1,weight=4)
        host.columnconfigure(2,weight=0)
        tk.Label(host,text='Classe original',bg=C['card2'],fg='#ffffff',font=('Sans',8,'bold')).grid(row=0,column=0,sticky='w',padx=7,pady=(5,4))
        tk.Label(host,text='Classe de saída',bg=C['card2'],fg='#ffffff',font=('Sans',8,'bold')).grid(row=0,column=1,sticky='w',padx=7,pady=(5,4))
        tk.Label(host,text='Imagens',bg=C['card2'],fg='#ffffff',font=('Sans',8,'bold')).grid(row=0,column=2,sticky='e',padx=7,pady=(5,4))
        palette=self._training_palette()
        for i,lab in enumerate(labels,1):
            row=tk.Frame(host,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1)
            row.grid(row=i,column=0,columnspan=3,sticky='ew',padx=3,pady=2)
            row.columnconfigure(1,weight=5)
            row.columnconfigure(2,weight=5)
            tk.Frame(row,bg=palette[(i-1)%len(palette)],width=6).grid(row=0,column=0,sticky='nsw')
            tk.Label(
                row,text=lab,bg=C['card2'],fg=C['text'],font=('Sans',8),
                anchor='w',justify='left',wraplength=145
            ).grid(row=0,column=1,sticky='ew',padx=(7,5),pady=5)
            var=self._training_group_vars.setdefault(lab,tk.StringVar(value=lab))
            ent=self._entry(row,var);ent.config(width=13)
            ent.grid(row=0,column=2,sticky='ew',padx=5,pady=5,ipady=2)
            tk.Label(
                row,text=f"{int(counts.get(lab,0)):,}".replace(',','.'),
                bg=C['card2'],fg=C['muted'],font=('Sans',8)
            ).grid(row=0,column=3,sticky='e',padx=(5,7),pady=5)

    def _training_output_labels(self):
        out=[]
        for lab in self._training_selected_labels():
            var=self._training_group_vars.get(lab)
            dest=(var.get() if var is not None else lab).strip()
            if not dest:dest=lab
            if dest not in out:out.append(dest)
        return out

    def _training_set_split_mode_all(self,mode):
        mode='N' if str(mode).upper()=='N' else '%'
        if not hasattr(self,'_training_split_mode_var') or self._training_split_mode_var is None:
            self._training_split_mode_var=tk.StringVar(value=mode)
        else:
            self._training_split_mode_var.set(mode)
        for lab in self._training_output_labels():
            if lab not in self._training_split_vars:
                self._training_split_vars[lab]={
                    'mode':tk.StringVar(value=mode),'train':tk.StringVar(value='70'),
                    'valid':tk.StringVar(value='15'),'test':tk.StringVar(value='15')
                }
            else:
                self._training_split_vars[lab]['mode'].set(mode)
        self._training_rebuild_split_table()

    def _training_rebuild_split_table(self):
        host=self._training_split_host
        if host is None:return
        try:
            if not host.winfo_exists():return
        except Exception:return
        for w in host.winfo_children():w.destroy()
        counts=self._training_context_df['manual_label'].fillna('').astype(str).value_counts().to_dict() if (self._training_context_df is not None and not self._training_context_df.empty and 'manual_label' in self._training_context_df.columns) else {}
        totals={}
        for lab in self._training_selected_labels():
            var=self._training_group_vars.get(lab)
            dest=(var.get() if var is not None else lab).strip() or lab
            totals[dest]=totals.get(dest,0)+int(counts.get(lab,0))
        palette=self._training_palette()

        head=tk.Frame(host,bg=C['card2'])
        head.grid(row=0,column=0,sticky='ew',padx=3,pady=(4,2))
        head.columnconfigure(1,weight=1)
        for col,text,anchor in [
            (1,'Classe / Grupo','w'),(2,'Modo','center'),(3,'Train','center'),
            (4,'Valid','center'),(5,'Test','center')
        ]:
            tk.Label(head,text=text,bg=C['card2'],fg='#ffffff',font=('Sans',8,'bold'),anchor=anchor).grid(
                row=0,column=col,sticky='ew',padx=3
            )

        for i,lab in enumerate(self._training_output_labels(),1):
            if lab not in self._training_split_vars:
                default_mode=self._training_split_mode_var.get() if hasattr(self,'_training_split_mode_var') and self._training_split_mode_var is not None else '%'
                self._training_split_vars[lab]={
                    'mode':tk.StringVar(value=default_mode),'train':tk.StringVar(value='70'),
                    'valid':tk.StringVar(value='15'),'test':tk.StringVar(value='15')
                }
            v=self._training_split_vars[lab]
            row=tk.Frame(host,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1)
            row.grid(row=i,column=0,sticky='ew',padx=3,pady=2)
            row.columnconfigure(1,weight=1)
            tk.Frame(row,bg=palette[(i-1)%len(palette)],width=6).grid(row=0,column=0,sticky='nsw')
            tk.Label(
                row,text=lab,bg=C['card2'],fg=C['text'],font=('Sans',8,'bold'),
                anchor='w',justify='left',wraplength=135
            ).grid(row=0,column=1,sticky='ew',padx=(7,4),pady=6)
            cb=ttk.Combobox(row,width=3,state='readonly',values=['%','N'],textvariable=v['mode'],style='Dark.TCombobox')
            cb.grid(row=0,column=2,sticky='ew',padx=3,pady=6)
            for col,key in [(3,'train'),(4,'valid'),(5,'test')]:
                ent=self._entry(row,v[key]);ent.config(width=5,justify='center')
                ent.grid(row=0,column=col,sticky='ew',padx=3,pady=6,ipady=2)

        tk.Label(
            host,
            text='Cada grupo pode usar percentual (%) ou quantidade absoluta (N). '
                 'A barra de rolagem aparece quando houver mais grupos que o espaço disponível.',
            bg=C['card2'],fg=C['muted'],font=('Sans',8),wraplength=350,justify='left'
        ).grid(row=999,column=0,sticky='ew',padx=6,pady=(7,4))

    def _training_grouping_updated(self):
        self._training_rebuild_group_table()
        self._training_rebuild_split_table()
        self._training_clear_preview()

    def _training_split_config(self):
        cfg={}
        for lab in self._training_output_labels():
            v=self._training_split_vars.get(lab)
            if v is None:
                cfg[lab]={'mode':'percent','train':70,'valid':15,'test':15}
                continue
            mode='count' if str(v['mode'].get()).strip().upper()=='N' else 'percent'
            try:
                vals={k:float(str(v[k].get()).replace(',','.')) for k in ('train','valid','test')}
            except Exception:
                raise ValueError(f'{lab}: train/valid/test devem ser números.')
            cfg[lab]={'mode':mode,**vals}
        return cfg

    def _training_build_plan(self):
        if self._training_context_df is None or self._training_context_df.empty:
            self._training_refresh_context(rebuild=False)
        if self._training_context_df is None or self._training_context_df.empty:
            raise ValueError(self._export_text('empty_scope'))
        labels=self._training_selected_labels()
        if not labels:raise ValueError(self._export_text('training_no_classes'))
        group_map={}
        for lab in labels:
            dest=(self._training_group_vars.get(lab).get() if lab in self._training_group_vars else lab).strip()
            if not dest:raise ValueError(f'{lab}: classe de saída vazia.')
            group_map[lab]=dest
        safe={}
        for dest in list(dict.fromkeys(group_map.values())):
            folder=sanitize_filename(dest) or 'unlabeled'
            if folder in safe and safe[folder]!=dest:
                raise ValueError(f'Classes de saída "{safe[folder]}" e "{dest}" resultam na mesma pasta "{folder}".')
            safe[folder]=dest
        cfg=self._training_split_config()
        try:seed=int(str(self.vars['training_seed'].get()).strip())
        except Exception:raise ValueError('Seed deve ser um número inteiro.')
        unit=self.vars['training_unit'].get() or 'image'
        group_col='custom_data_orig' if unit=='group' else None
        frame=self._training_context_df.copy()
        plan,summary=plan_training_split(frame,group_map,cfg,seed=seed,unit=unit,group_column=group_col)
        return plan,summary,group_map,cfg,seed,unit

    def _training_clear_preview(self):
        tree=self._training_preview_tree
        if tree is not None:
            try:
                for x in tree.get_children():tree.delete(x)
            except Exception:pass
        if self._training_summary_label is not None:
            try:self._training_summary_label.config(text='Configure as classes e clique em Pré-visualizar distribuição.')
            except Exception:pass

    def _training_preview(self):
        try:
            plan,summary,group_map,cfg,seed,unit=self._training_build_plan()
        except Exception as e:
            messagebox.showwarning(self._export_text('training_invalid'),str(e));return
        tree=self._training_preview_tree
        if tree is not None:
            for x in tree.get_children():tree.delete(x)
            totals={'available':0,'train':0,'valid':0,'test':0,'unused':0}
            for _,d in summary.iterrows():
                lab=str(d.get('class',''))
                tree.insert('', 'end', values=(
                    lab,int(d.get('available',0)),int(d.get('train',0)),
                    int(d.get('valid',0)),int(d.get('test',0))
                ))
                for k in totals:totals[k]+=int(d.get(k,0))
            tree.insert('', 'end', values=(
                self._export_text('training_total'),totals['available'],totals['train'],
                totals['valid'],totals['test']
            ))
        if self._training_summary_label is not None:
            used=int((plan['split']!='unused').sum()) if 'split' in plan.columns else 0
            txt=f"{len(summary)} classes finais • {used:,} imagens usadas • {len(plan)-used:,} não usadas".replace(',','.')
            if unit=='group':txt+=' • '+self._export_text('training_group_unit_note')
            self._training_summary_label.config(text=txt)

    def _training_browse_output(self):
        initial=self.vars['training_dir'].get() or (str(self.project.project_dir/'exports'/'training') if self.project else str(self.app.project_root))
        p=filedialog.askdirectory(initialdir=initial)
        if p:self.vars['training_dir'].set(p)

    @staticmethod
    def _training_unique_path(path):
        path=Path(path)
        if not path.exists():return path
        for i in range(2,1000000):
            p=path.with_name(f'{path.stem}_{i}{path.suffix}')
            if not p.exists():return p
        return path.with_name(f'{path.stem}_{int(time.time())}{path.suffix}')

    def _training_export_dataset(self):
        if not self.project:return
        try:
            plan,summary,group_map,cfg,seed,unit=self._training_build_plan()
        except Exception as e:
            messagebox.showwarning(self._export_text('training_invalid'),str(e));return
        use=plan[plan['split'].isin(['train','valid','test'])].copy()
        if use.empty:
            messagebox.showwarning(self._export_text('training_title'),self._export_text('empty_scope'));return
        parent=self.vars['training_dir'].get().strip()
        if not parent:
            self._training_browse_output();parent=self.vars['training_dir'].get().strip()
        if not parent:return
        name=sanitize_filename(Path(self.vars['training_name'].get().strip() or 'ilt_dataset').name) or 'ilt_dataset'
        final_dir=Path(parent).expanduser()/name

        if not hasattr(self,'_training_existing_mode') or self._training_existing_mode is None:
            self._training_existing_mode=tk.StringVar(value='merge')
        if not hasattr(self,'_training_report_duplicates_var') or self._training_report_duplicates_var is None:
            self._training_report_duplicates_var=tk.BooleanVar(value=True)
        if not hasattr(self,'_training_duplicate_status_var') or self._training_duplicate_status_var is None:
            self._training_duplicate_status_var=tk.StringVar(value='Duplicadas ignoradas: 0')

        existing_mode=self._training_existing_mode.get() or 'merge'
        keep_names=bool(self.vars['training_keep_names'].get())
        make_manifest=bool(self.vars['training_manifest'].get())
        make_config=bool(self.vars['training_config'].get())
        make_stats=bool(self.vars['training_stats'].get())
        report_duplicates=bool(self._training_report_duplicates_var.get())
        training_scope_snapshot=str(self.vars['training_scope'].get() or 'all')
        selected_batches_snapshot=sorted(map(str,self._training_batch_selection))
        selected_classes_snapshot=list(self._training_selected_labels())
        pixel_size_snapshot=self._segmentation_pixel_size(None,self.vars['pixel_size'].get())

        if final_dir.exists():
            if existing_mode=='replace':
                ok=messagebox.askyesno(
                    self._export_text('training_title'),
                    'A pasta do dataset já existe.\nSubstituir todo o conteúdo dessa pasta?'
                )
                if not ok:return
            else:
                ok=messagebox.askyesno(
                    self._export_text('training_title'),
                    'A pasta do dataset já existe.\n'
                    'Mesclar com o dataset existente? Imagens com o mesmo nome original serão ignoradas.'
                )
                if not ok:return

        def work(progress):
            duplicates=[]
            missing=0
            total=len(use)
            tmp=None
            direct_target=(existing_mode=='merge' and final_dir.exists())
            if direct_target:
                target_root=final_dir
                target_root.mkdir(parents=True,exist_ok=True)
            else:
                tmp=final_dir.with_name(final_dir.name+f'.part-{os.getpid()}-{int(time.time())}')
                if tmp.exists():shutil.rmtree(tmp,ignore_errors=True)
                tmp.mkdir(parents=True,exist_ok=False)
                target_root=tmp

            existing_by_name={}
            if direct_target:
                manifest_path=target_root/'dataset_manifest.csv'
                if manifest_path.exists():
                    try:
                        prev_manifest=pd.read_csv(
                            manifest_path,
                            usecols=lambda c:c in {'image_name','exported_file'},
                            low_memory=False,
                        )
                        if 'image_name' in prev_manifest.columns:
                            for _,r in prev_manifest.iterrows():
                                nm=str(r.get('image_name','') or '').strip()
                                if not nm:continue
                                rel=str(r.get('exported_file','') or '').strip()
                                existing_by_name.setdefault(nm.casefold(),str(target_root/rel) if rel else 'dataset_manifest.csv')
                    except Exception:
                        pass
                for split_name in ('train','valid','test'):
                    split_dir=target_root/split_name
                    if not split_dir.is_dir():continue
                    try:
                        for dirpath,_,filenames in os.walk(split_dir):
                            base=Path(dirpath)
                            for filename in filenames:
                                existing_by_name.setdefault(filename.casefold(),str(base/filename))
                    except Exception:
                        pass

            manifest=[]
            jobs=[]
            reserved_dest=set()
            self.project.reset_image_resolve_stats()
            try:
                # Phase 1: resolve all sources and reserve destination names.
                for pos,row in enumerate(use.itertuples(index=False),1):
                    d=row._asdict()
                    bn=str(d.get('source_batch','') or self._training_batch_from_gid(d.get('gid',''),''))
                    image_name=str(d.get('names',''))
                    src=self.project.resolve_image(bn,image_name)
                    label=str(d.get('export_label','unlabeled') or 'unlabeled')
                    split=str(d.get('split','unused'))
                    dest_dir=target_root/split/(sanitize_filename(label) or 'unlabeled')
                    status='missing';dest_rel=''

                    entry={
                        'source_batch':bn,'image_name':image_name,'gid':str(d.get('gid','')),
                        'manual_label_original':str(d.get('manual_label','')),'export_label':label,'split':split,
                        'group_key':str(d.get('custom_data_orig','')) if unit=='group' else '',
                        'status':status,'exported_file':dest_rel,
                    }
                    manifest_index=len(manifest)

                    if src is not None and src.is_file():
                        duplicate_key=src.name.casefold()
                        existing_path=existing_by_name.get(duplicate_key)
                        if existing_path:
                            entry['status']='duplicate'
                            duplicates.append({
                                'file_name':src.name,
                                'split_requested':split,
                                'class_requested':label,
                                'source_batch':bn,
                                'source_path':str(src),
                                'existing_path':existing_path,
                            })
                        else:
                            if keep_names:
                                candidate=dest_dir/src.name
                            else:
                                suffix=sanitize_filename(Path(bn).stem) or 'batch'
                                candidate=dest_dir/f'{src.stem}__{suffix}{src.suffix}'

                            dest=Path(candidate)
                            base=dest
                            key=os.path.normcase(os.path.abspath(str(dest)))
                            if dest.exists() or key in reserved_dest:
                                for i in range(2,1000000):
                                    dest=base.with_name(f'{base.stem}_{i}{base.suffix}')
                                    key=os.path.normcase(os.path.abspath(str(dest)))
                                    if not dest.exists() and key not in reserved_dest:
                                        break
                            reserved_dest.add(key)
                            existing_by_name[duplicate_key]=str(dest)
                            entry['status']='planned'
                            entry['exported_file']=str(dest.relative_to(target_root))
                            jobs.append(CopyJob(Path(src),dest,manifest_index))
                    else:
                        missing+=1

                    manifest.append(entry)
                    if pos==total or pos%250==0:
                        progress((pos/max(total,1))*25.0,100,f'Preparando {pos:,}/{total:,}'.replace(',','.'))

                # Phase 2: copy with bounded, storage-agnostic concurrency.
                def copy_progress(cur,total_files,label=''):
                    pct=25.0+(float(cur)/max(float(total_files),1.0))*65.0
                    progress(pct,100,f'Copiando {cur:,}/{total_files:,}'.replace(',','.'))

                report=copy_jobs(
                    jobs,
                    progress=copy_progress,
                    workers=recommended_io_workers(),
                    preserve_metadata=True,
                )

                succeeded=report.succeeded_tokens
                failed=report.failed_tokens
                for idx in succeeded:
                    if 0<=int(idx)<len(manifest):
                        manifest[int(idx)]['status']='copied'
                for idx in failed:
                    if 0<=int(idx)<len(manifest):
                        manifest[int(idx)]['status']='error'

                copied=report.copied
                copy_failed=report.failed
                progress(92,100,'Gravando manifestos e estatísticas…')

                if make_manifest:
                    out_manifest=target_root/'dataset_manifest.csv'
                    new_manifest=pd.DataFrame(manifest)
                    if direct_target and out_manifest.exists():
                        try:
                            prev=pd.read_csv(out_manifest,low_memory=False)
                            pd.concat([prev,new_manifest],ignore_index=True).to_csv(out_manifest,index=False)
                        except Exception:
                            new_manifest.to_csv(out_manifest,index=False)
                    else:
                        new_manifest.to_csv(out_manifest,index=False)

                if make_stats:
                    summary.to_csv(target_root/'dataset_split_stats.csv',index=False)

                if make_config:
                    config={
                        'created_at':datetime.now().isoformat(timespec='seconds'),
                        'project_dir':str(self.project.project_dir),'dataset_dir':str(final_dir),
                        'scope':training_scope_snapshot,'selected_batches':selected_batches_snapshot,
                        'selected_classes':selected_classes_snapshot,'group_map':group_map,'split_config':cfg,
                        'seed':seed,'sampling_unit':unit,'group_column':'custom_data_orig' if unit=='group' else None,
                        'pixel_size_um_per_px':pixel_size_snapshot,
                        'keep_original_filenames':keep_names,'planned_images':int(len(use)),
                        'copied_images':int(copied),'missing_images':int(missing),
                        'copy_failures':int(copy_failed),
                        'duplicate_images_skipped':int(len(duplicates)),
                        'existing_dataset_mode':existing_mode,
                        'duplicate_rule':'same original filename anywhere in existing train/valid/test tree',
                        'io_workers':int(report.workers),
                        'copy_elapsed_seconds':float(report.elapsed_seconds),
                        'copy_files_per_second':float(report.files_per_second),
                        'copy_mib_per_second':float(report.mib_per_second),
                    }
                    (target_root/'dataset_config.json').write_text(
                        json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8'
                    )

                if report_duplicates:
                    report_path=target_root/'duplicate_files_report.csv'
                    pd.DataFrame(
                        duplicates,
                        columns=['file_name','split_requested','class_requested','source_batch','source_path','existing_path']
                    ).to_csv(report_path,index=False)

                if tmp is not None:
                    if final_dir.exists():shutil.rmtree(final_dir,ignore_errors=True)
                    os.replace(tmp,final_dir)

                progress(100,100,'Concluído')
                return final_dir,copied,missing,len(duplicates),copy_failed,report,self.project.image_resolve_stats()
            except Exception:
                if tmp is not None:shutil.rmtree(tmp,ignore_errors=True)
                raise

        def done(res):
            path,copied,missing,dups,copy_failed,report,resolver=res
            self._training_progress_var.set(100.0)
            if hasattr(self,'_training_duplicate_status_var') and self._training_duplicate_status_var is not None:
                self._training_duplicate_status_var.set(f'Duplicadas ignoradas: {dups:,}'.replace(',','.'))
            msg=f"{self._export_text('training_done')}: {copied:,} imagens copiadas".replace(',','.')
            if missing:msg+=f" • {missing:,} ausentes".replace(',','.')
            if dups:msg+=f" • {dups:,} duplicadas ignoradas".replace(',','.')
            if copy_failed:msg+=f" • {copy_failed:,} falhas de cópia".replace(',','.')
            self._log(msg,'success' if not copy_failed else 'warn')
            self._log(
                f"Training export I/O: {report.files_per_second:.1f} arquivos/s • "
                f"{report.mib_per_second:.1f} MiB/s • {report.elapsed_seconds:.1f}s • "
                f"{report.workers} worker(s)",
                'info',
            )
            if resolver.get('global_index_builds'):
                self._log(
                    f"Resolução por índice global: {resolver.get('global_index_hit',0)} "
                    f"• ausentes={resolver.get('missing',0)}",
                    'info',
                )
            messagebox.showinfo(self._export_text('training_title'),msg)

        self._training_progress_var.set(0)
        if hasattr(self,'_training_duplicate_status_var') and self._training_duplicate_status_var is not None:
            self._training_duplicate_status_var.set('Duplicadas ignoradas: 0')
        self._run_operation(self.tr('exporting'),work,done,local_progress=self._training_progress_var)

    def _build_training_export(self,host,row=1):
        if not hasattr(self,'_training_split_mode_var') or self._training_split_mode_var is None:
            self._training_split_mode_var=tk.StringVar(value='%')
        if not hasattr(self,'_training_existing_mode') or self._training_existing_mode is None:
            self._training_existing_mode=tk.StringVar(value='merge')
        if not hasattr(self,'_training_report_duplicates_var') or self._training_report_duplicates_var is None:
            self._training_report_duplicates_var=tk.BooleanVar(value=True)
        if not hasattr(self,'_training_duplicate_status_var') or self._training_duplicate_status_var is None:
            self._training_duplicate_status_var=tk.StringVar(value='Duplicadas ignoradas: 0')

        if self.project and not self.vars['training_dir'].get():
            self.vars['training_dir'].set(str(self.project.project_dir/'exports'/'training'))
        self.vars['training_scope'].set(self.vars['training_scope'].get() or 'all')
        if self.vars['training_scope'].get()=='current':
            self.vars['training_scope'].set('all')
        if self.project:
            signature=self._training_context_signature()
            if self._training_context_key!=signature:
                self._training_render_loading(host,row=row)
                self._training_start_context_load(signature)
                return
        else:
            self._training_context_df=pd.DataFrame()
            self._training_context_key=None
        counts=self._training_context_df['manual_label'].fillna('').astype(str).value_counts().to_dict() if (self._training_context_df is not None and not self._training_context_df.empty and 'manual_label' in self._training_context_df.columns) else {}
        for lab in sorted(counts):
            self._training_class_vars.setdefault(lab,tk.BooleanVar(value=True))
            self._training_group_vars.setdefault(lab,tk.StringVar(value=lab))

        # Five-column workflow.  The center columns deliberately receive more
        # width because grouping/renaming and split configuration contain the
        # longest editable values.
        for col,weight in enumerate((3,5,5,4,3)):
            host.columnconfigure(col,weight=weight,minsize=0)

        title=tk.Frame(host,bg=C['bg'])
        title.grid(row=row,column=0,columnspan=5,sticky='ew',padx=4,pady=(2,4))
        tk.Label(title,text='Opção C — Fluxo por etapas',bg=C['bg'],fg='#79bdff',font=('Sans',12,'bold')).pack(anchor='w')
        tk.Label(
            title,text='Selecione as classes, crie agrupamentos, defina a divisão e revise a distribuição antes de exportar.',
            bg=C['bg'],fg=C['muted'],font=('Sans',8)
        ).pack(anchor='w',pady=(1,0))
        cards_row=row+1

        # 1. Selection — intentionally narrower than steps 2 and 3.
        c1,b1=self._card(host,'1. Selecionar classes')
        c1.grid(row=cards_row,column=0,sticky='nsew',padx=(4,3),pady=4)
        tk.Label(
            b1,text='Escolha o escopo e as classes que serão incluídas no dataset de treinamento.',
            bg=C['card'],fg=C['muted'],font=('Sans',8),wraplength=245,justify='left'
        ).pack(anchor='w',pady=(0,6))
        scope_box=tk.Frame(b1,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1)
        scope_box.pack(fill='x',pady=(0,7))
        tk.Label(scope_box,text='Escopo dos dados',bg=C['card2'],fg='#ffffff',font=('Sans',9,'bold')).pack(anchor='w',padx=8,pady=(7,4))
        for val,label in [
            ('all','Todas as imagens do projeto'),
            ('batches','Apenas batch(es) selecionados'),
            ('selected','Apenas imagens selecionadas na galeria')
        ]:
            tk.Radiobutton(
                scope_box,text=label,value=val,variable=self.vars['training_scope'],
                command=self._training_scope_changed,bg=C['card2'],fg=C['text'],
                selectcolor=C['red'],activebackground=C['card2'],activeforeground='white',
                anchor='w',justify='left'
            ).pack(fill='x',padx=8,pady=1)
        self._btn(scope_box,'Selecionar batches...',self._training_select_batches).pack(fill='x',padx=8,pady=(5,6))

        seedrow=tk.Frame(scope_box,bg=C['card2']);seedrow.pack(fill='x',padx=8,pady=(0,4))
        tk.Label(seedrow,text='Seed do sorteio',bg=C['card2'],fg=C['muted'],font=('Sans',8)).pack(side='left')
        seed_entry=self._entry(seedrow,self.vars['training_seed']);seed_entry.config(width=8)
        seed_entry.pack(side='right',ipady=2)

        unitrow=tk.Frame(scope_box,bg=C['card2']);unitrow.pack(fill='x',padx=8,pady=(0,8))
        tk.Label(unitrow,text='Unidade de sorteio',bg=C['card2'],fg=C['muted'],font=('Sans',8)).pack(side='left')
        unit_box=ttk.Combobox(
            unitrow,state='readonly',
            values=[self._export_text('training_unit_image'),self._export_text('training_unit_group')],
            width=16,style='Dark.TCombobox'
        )
        unit_display=self._export_text('training_unit_group') if self.vars['training_unit'].get()=='group' else self._export_text('training_unit_image')
        unit_box.set(unit_display)
        def _unit_pick(_=None):
            self.vars['training_unit'].set('group' if unit_box.get()==self._export_text('training_unit_group') else 'image')
            self._training_clear_preview()
        unit_box.bind('<<ComboboxSelected>>',_unit_pick)
        unit_box.pack(side='right')

        sel_box=tk.Frame(b1,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1)
        sel_box.pack(fill='both',expand=True)
        hdr=tk.Frame(sel_box,bg=C['card2']);hdr.pack(fill='x',padx=7,pady=(6,3))
        tk.Label(hdr,text='Classes incluídas',bg=C['card2'],fg='#ffffff',font=('Sans',9,'bold')).pack(side='left')
        tk.Checkbutton(
            hdr,text='Todas',variable=self.vars['training_all_classes'],command=self._training_toggle_all,
            bg=C['card2'],fg=C['text'],selectcolor=C['red'],activebackground=C['card2'],
            activeforeground='white'
        ).pack(side='right')
        self._training_class_host=self._training_scroll_area(sel_box,430)
        self._training_render_class_list(counts)
        self._training_batch_label=tk.Label(sel_box,text='',bg=C['card2'],fg=C['muted'],font=('Sans',8))
        self._training_batch_label.pack(anchor='w',padx=8,pady=(4,6))
        if self._training_batch_selection:
            self._training_batch_label.config(text=f"{len(self._training_batch_selection)} batch(es) selecionado(s)")

        # 2. Group / rename — wider editable table.
        c2,b2=self._card(host,'2. Agrupar / renomear')
        c2.grid(row=cards_row,column=1,sticky='nsew',padx=3,pady=4)
        tk.Label(
            b2,text='Agrupe classes usando o mesmo nome de saída, ou renomeie cada classe individualmente.',
            bg=C['card'],fg=C['muted'],font=('Sans',8),wraplength=360,justify='left'
        ).pack(anchor='w',pady=(0,6))
        toolbar=tk.Frame(b2,bg=C['card']);toolbar.pack(fill='x',pady=(0,6))
        self._btn(toolbar,'Usar nomes originais',self._training_reset_group_names).pack(side='left')
        self._btn(toolbar,'Atualizar grupos',self._training_grouping_updated,C['purple']).pack(side='right')
        group_box=tk.Frame(b2,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1)
        group_box.pack(fill='both',expand=True)
        self._training_group_host=self._training_scroll_area(group_box,515)
        self._training_rebuild_group_table()

        # 3. Split — wider compact table with per-group % / N.
        c3,b3=self._card(host,'3. Definir divisão')
        c3.grid(row=cards_row,column=2,sticky='nsew',padx=3,pady=4)
        tk.Label(
            b3,text='Configure Train / Validação / Teste para cada grupo final, em percentual (%) ou quantidade absoluta (N).',
            bg=C['card'],fg=C['muted'],font=('Sans',8),wraplength=360,justify='left'
        ).pack(anchor='w',pady=(0,6))
        mode_row=tk.Frame(b3,bg=C['card']);mode_row.pack(fill='x',pady=(0,6))
        tk.Label(mode_row,text='Modo global:',bg=C['card'],fg=C['muted'],font=('Sans',8)).pack(side='left')
        tk.Radiobutton(
            mode_row,text='Percentual (%)',value='%',variable=self._training_split_mode_var,
            command=lambda:self._training_set_split_mode_all('%'),bg=C['card'],fg=C['text'],
            selectcolor=C['red'],activebackground=C['card'],activeforeground='white'
        ).pack(side='left',padx=(7,4))
        tk.Radiobutton(
            mode_row,text='Quantidade (N)',value='N',variable=self._training_split_mode_var,
            command=lambda:self._training_set_split_mode_all('N'),bg=C['card'],fg=C['text'],
            selectcolor=C['red'],activebackground=C['card'],activeforeground='white'
        ).pack(side='left')
        split_box=tk.Frame(b3,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1)
        split_box.pack(fill='both',expand=True)
        self._training_split_host=self._training_scroll_area(split_box,515)
        self._training_rebuild_split_table()

        # 4. Preview in its own column.
        c4,b4=self._card(host,'4. Pré-visualização')
        c4.grid(row=cards_row,column=3,sticky='nsew',padx=3,pady=4)
        tk.Label(
            b4,text='Resumo das imagens por classe ou grupo e por split.',
            bg=C['card'],fg=C['muted'],font=('Sans',8),wraplength=300,justify='left'
        ).pack(anchor='w',pady=(0,6))
        prev_box=tk.Frame(b4,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1)
        prev_box.pack(fill='both',expand=True)
        cols=('class','available','train','valid','test')
        tree=ttk.Treeview(prev_box,columns=cols,show='headings',height=22)
        preview_cols=[
            ('class','Classe / Grupo',105,'w'),
            ('available','Total',52,'center'),
            ('train','Train',48,'center'),
            ('valid','Valid',48,'center'),
            ('test','Test',48,'center')
        ]
        for col,title_txt,width,anchor in preview_cols:
            tree.heading(col,text=title_txt)
            tree.column(col,width=width,anchor=anchor,stretch=(col=='class'))
        sy=ttk.Scrollbar(prev_box,orient='vertical',command=tree.yview)
        tree.configure(yscrollcommand=sy.set)
        tree.pack(side='left',fill='both',expand=True,padx=(7,0),pady=(7,5))
        sy.pack(side='right',fill='y',padx=(0,5),pady=(7,5))
        self._training_preview_tree=tree
        self._training_summary_label=tk.Label(
            b4,text='Configure as classes e clique em Pré-visualizar distribuição.',
            bg=C['card'],fg=C['muted'],font=('Sans',8),wraplength=300,justify='left'
        )
        self._training_summary_label.pack(anchor='w',fill='x',pady=(6,5))
        self._btn(b4,'Pré-visualizar distribuição',self._training_preview,C['blue']).pack(fill='x',ipady=5)

        # 5 + 6. Options/output share the fifth, narrower column.
        c5,b5=self._card(host,'5. Opções adicionais')
        c5.grid(row=cards_row,column=4,sticky='nsew',padx=(3,4),pady=4)
        for key,label in [
            ('training_keep_names','Manter nomes originais'),
            ('training_manifest','Gerar manifesto (CSV)'),
            ('training_config','Salvar configuração (JSON)'),
            ('training_stats','Salvar estatísticas (CSV)')
        ]:
            tk.Checkbutton(
                b5,text=label,variable=self.vars[key],bg=C['card'],fg=C['text'],
                selectcolor=C['red'],activebackground=C['card'],activeforeground='white',
                anchor='w',justify='left'
            ).pack(fill='x',pady=2)

        out_box=tk.Frame(b5,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1)
        out_box.pack(fill='x',pady=(10,0))
        tk.Label(out_box,text='6. Local de exportação',bg=C['card2'],fg='#ffffff',font=('Sans',9,'bold')).pack(anchor='w',padx=8,pady=(7,4))
        tk.Label(out_box,text='Pasta de destino',bg=C['card2'],fg=C['muted'],font=('Sans',8)).pack(anchor='w',padx=8)
        rr=tk.Frame(out_box,bg=C['card2']);rr.pack(fill='x',padx=8,pady=2)
        dir_entry=self._entry(rr,self.vars['training_dir']);dir_entry.config(width=10)
        dir_entry.pack(side='left',fill='x',expand=True,ipady=3)
        self._btn(rr,'Procurar...',self._training_browse_output).pack(side='left',padx=(4,0))
        tk.Label(out_box,text='Nome da pasta',bg=C['card2'],fg=C['muted'],font=('Sans',8)).pack(anchor='w',padx=8,pady=(4,0))
        name_entry=self._entry(out_box,self.vars['training_name']);name_entry.config(width=12)
        name_entry.pack(fill='x',padx=8,pady=(2,5),ipady=3)
        tk.Label(out_box,text='Se a pasta já existir:',bg=C['card2'],fg=C['muted'],font=('Sans',8)).pack(anchor='w',padx=8,pady=(4,2))
        tk.Radiobutton(
            out_box,text='Mesclar com dataset existente',value='merge',variable=self._training_existing_mode,
            bg=C['card2'],fg=C['text'],selectcolor=C['red'],activebackground=C['card2'],
            activeforeground='white',anchor='w',justify='left'
        ).pack(fill='x',padx=8,pady=1)
        tk.Radiobutton(
            out_box,text='Substituir conteúdo existente',value='replace',variable=self._training_existing_mode,
            bg=C['card2'],fg=C['text'],selectcolor=C['red'],activebackground=C['card2'],
            activeforeground='white',anchor='w',justify='left'
        ).pack(fill='x',padx=8,pady=1)
        tk.Checkbutton(
            out_box,text='Gerar relatório de duplicadas',variable=self._training_report_duplicates_var,
            bg=C['card2'],fg=C['text'],selectcolor=C['red'],activebackground=C['card2'],
            activeforeground='white',anchor='w',justify='left'
        ).pack(fill='x',padx=8,pady=(4,2))
        tk.Label(
            out_box,textvariable=self._training_duplicate_status_var,bg=C['card2'],
            fg='#56c0ff',font=('Sans',8,'bold'),wraplength=220,justify='left'
        ).pack(anchor='w',padx=8,pady=(0,6))

        self._btn(b5,'Exportar dataset de treinamento',self._training_export_dataset,C['green']).pack(fill='x',ipady=7,pady=(10,3))
        ttk.Progressbar(
            b5,maximum=100,variable=self._training_progress_var,mode='determinate'
        ).pack(fill='x',pady=(2,4))
