#!/usr/bin/env python3
"""Gallery, thumbnail, relabeling and similarity mixin for the IAT workspace."""
from __future__ import annotations

import math
import heapq
import os
import re
import time

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk

try:
    from PIL import Image, ImageDraw, ImageTk
except Exception:
    Image = ImageDraw = ImageTk = None

from iat_backend import (
    SEGMENTATION_METHODS,
    filter_dataframe,
    order_choices,
    sort_gallery_multi,
)
from iat_widgets import RangeSlider, ScrollFrame
from project_settings import (
    DEFAULT_PIXEL_SIZE_UM_PER_PX,
    load_project_settings,
    parse_pixel_size,
)
from ui_resources import IAT_COLORS as C


class IATGalleryMixin:
    def _clear_gallery_similarity_state(self,clear_reference=True,refresh=False):
        self._gallery_similarity_active=False
        self._gallery_similarity_scores={}
        if clear_reference:
            self._gallery_similarity_ref_gid=None
        status=getattr(self,'_gallery_similarity_status_var',None)
        if status is not None:
            try:status.set('')
            except Exception:pass
        self._update_similarity_reference_ui()
        if refresh and self.current_tab=='gallery':
            self._gallery_update_in_place(reset_page=True,prefetch=True)

    def _gallery_base_source(self):
        """Return Gallery membership before Confidence/Variance ranges and sorting."""
        if self.vars['batch'].get()=='Overview':
            if self._gallery_selection_df is not None:
                base=self._gallery_selection_df
            elif self.selected_gids:
                ids=set(self.selected_gids)
                base=self.current_df[self.current_df['gid'].astype(str).isin(ids)]
            else:
                return self.current_df.iloc[0:0]
        elif self.selected_gids:
            ids=set(self.selected_gids)
            base=self.current_df[self.current_df['gid'].astype(str).isin(ids)]
        else:
            base=self.current_df

        discarded=getattr(self,'_gallery_discarded_gids',set())
        if discarded and not base.empty and 'gid' in base.columns:
            base=base.loc[~base['gid'].astype(str).isin(discarded)]

        if self.vars['hide_relabel'].get() and {'manual_label','correct_label'}.issubset(base.columns):
            base=base.loc[base['manual_label'].astype(str)==base['correct_label'].astype(str)]
        cls=self.vars['class_filter'].get()
        if cls not in ('Todas','All','') and 'manual_label' in base.columns:
            base=base.loc[base['manual_label'].astype(str)==str(cls)]
        return base

    def _gallery_histogram_source(self):
        return self._gallery_base_source()

    def _refresh_gallery_histograms(self):
        if not hasattr(self,'gallery_conf_slider') and not hasattr(self,'gallery_var_slider'):
            return
        base=self._gallery_histogram_source()
        if hasattr(self,'gallery_conf_slider'):
            vals=pd.to_numeric(base['confs1'],errors='coerce').to_numpy(float) if 'confs1' in base.columns else np.asarray([])
            try:self.gallery_conf_slider.set_values(vals)
            except Exception:pass
        if hasattr(self,'gallery_var_slider'):
            vals=pd.to_numeric(base['var_norm'],errors='coerce').to_numpy(float) if 'var_norm' in base.columns else np.asarray([])
            try:self.gallery_var_slider.set_values(vals)
            except Exception:pass

    def _similarity_reference_row(self,base=None):
        gid=str(getattr(self,'_gallery_similarity_ref_gid',None) or '')
        if not gid:
            return None
        base=self._gallery_base_source() if base is None else base
        if base.empty or 'gid' not in base.columns:
            return None
        hit=base.loc[base['gid'].astype(str)==gid]
        return None if hit.empty else hit.iloc[0]

    def _update_similarity_reference_ui(self):
        var=getattr(self,'_gallery_similarity_ref_var',None)
        label=getattr(self,'_gallery_similarity_preview',None)
        row=self._similarity_reference_row() if self.project is not None else None
        if row is None:
            if var is not None:
                try:var.set('—')
                except Exception:pass
            if label is not None:
                try:label.configure(image='',text='—'); label.image=None
                except Exception:pass
            self._similarity_ref_photo=None
            return
        name=str(row.get('names',''))
        if var is not None:
            try:var.set(name)
            except Exception:pass
        if label is not None and Image is not None and ImageTk is not None:
            try:
                p=self.project.resolve_image(str(row.get('csv_basename','')),name)
                if p and p.exists():
                    with Image.open(p) as src:
                        im=src.convert('RGB')
                    im.thumbnail((72,54),Image.Resampling.LANCZOS)
                    self._similarity_ref_photo=ImageTk.PhotoImage(im)
                    label.configure(image=self._similarity_ref_photo,text='')
                    label.image=self._similarity_ref_photo
                else:
                    label.configure(image='',text='—')
            except Exception:
                try:label.configure(image='',text='—')
                except Exception:pass

    def _set_similarity_reference_from_checked(self):
        gids=[str(g) for g in self.checked_gids]
        if len(gids)!=1:
            messagebox.showinfo(self.tr('iat_title'),self.tr('similarity_need_one'))
            return
        base=self._gallery_base_source()
        gid=gids[0]
        if base.empty or 'gid' not in base.columns or gid not in set(base['gid'].astype(str)):
            messagebox.showinfo(self.tr('iat_title'),self.tr('similarity_need_one'))
            return
        self._gallery_similarity_ref_gid=gid
        self._gallery_similarity_active=False
        self._gallery_similarity_scores={}
        self._update_similarity_reference_ui()
        status=getattr(self,'_gallery_similarity_status_var',None)
        if status is not None:
            row=self._similarity_reference_row(base)
            try:status.set(('Referência definida: ' if self.lang()=='pt' else 'Reference set: ')+str(row.get('names','')))
            except Exception:pass

    def _compute_similarity_scores_for_df(self,base,ref_gid,mode,method,progress=None):
        """Return gid -> distance for the complete Gallery selection."""
        if base.empty or 'gid' not in base.columns:
            raise ValueError(self.tr('no_selection'))
        base=base.copy()
        gids=base['gid'].astype(str).to_numpy(object)
        try:ref_pos=int(np.flatnonzero(gids==str(ref_gid))[0])
        except Exception:
            raise ValueError(self.tr('similarity_need_one'))

        if mode=='tsne':
            if not {'x','y'}.issubset(base.columns):
                raise ValueError('A projeção atual não possui coordenadas x/y.')
            x=pd.to_numeric(base['x'],errors='coerce').to_numpy(float)
            y=pd.to_numeric(base['y'],errors='coerce').to_numpy(float)
            if not (math.isfinite(x[ref_pos]) and math.isfinite(y[ref_pos])):
                raise ValueError('A imagem de referência não possui coordenadas válidas na projeção.')
            dist=np.hypot(x-x[ref_pos],y-y[ref_pos])
            dist[~(np.isfinite(x)&np.isfinite(y))]=np.inf
            if progress:progress(1,1,'t-SNE 2D')
            return {str(g):float(d) for g,d in zip(gids,dist)}

        # PCA is calculated on standardized numeric segmentation measurements.
        parts=[]
        grouped=list(base.groupby('csv_basename',sort=False)) if 'csv_basename' in base.columns else []
        total_groups=max(len(grouped),1)
        for j,(bn,ss) in enumerate(grouped,1):
            parts.append(self.project.attach_segmentation(ss,str(bn),method))
            if progress:progress(j,total_groups,str(bn))
        joined=pd.concat(parts,ignore_index=True,sort=False) if parts else base.copy()
        if joined.empty:
            raise ValueError(self.tr('similarity_no_metrics'))

        features=[
            'Area (pxl)','Perimeter (pxl)','ESD (pxl)','circularity','Elongation','Rectangularity','Solidity',
            'Feret_max','Feret_min','Major (pxl)','Minor (pxl)','Eccentricity',
            'Mean Intensity','Median Intensity','Std Intensity','Contrast',
            'Percent_on_border','Number Holes'
        ]
        cols=[]
        arrays=[]
        for col in features:
            if col not in joined.columns:continue
            a=pd.to_numeric(joined[col],errors='coerce').to_numpy(float)
            finite=a[np.isfinite(a)]
            if finite.size>=2 and float(np.nanstd(finite))>1e-12:
                cols.append(col);arrays.append(a)
        if len(cols)<2:
            raise ValueError(self.tr('similarity_no_metrics'))

        X=np.column_stack(arrays)
        finite_mask=np.isfinite(X)
        valid_counts=finite_mask.sum(axis=1)
        required=max(2,min(4,int(math.ceil(len(cols)*0.35))))
        valid_rows=valid_counts>=required
        joined_gids=joined['gid'].astype(str).to_numpy(object)
        ref_idx_arr=np.flatnonzero(joined_gids==str(ref_gid))
        if not len(ref_idx_arr):
            raise ValueError(self.tr('similarity_need_one'))
        ref_idx=int(ref_idx_arr[0])
        if not valid_rows[ref_idx]:
            raise ValueError('A imagem de referência não possui métricas de segmentação suficientes.')

        med=np.nanmedian(np.where(finite_mask,X,np.nan),axis=0)
        if not np.all(np.isfinite(med)):
            raise ValueError(self.tr('similarity_no_metrics'))
        Xi=np.where(finite_mask,X,med)
        fit=Xi[valid_rows]
        if len(fit)<2:
            raise ValueError(self.tr('similarity_no_metrics'))
        mean=fit.mean(axis=0)
        std=fit.std(axis=0)
        std=np.where(std>1e-12,std,1.0)
        Z=(Xi-mean)/std
        Zfit=Z[valid_rows]
        try:
            _,_,vt=np.linalg.svd(Zfit,full_matrices=False)
        except np.linalg.LinAlgError as e:
            raise ValueError('Não foi possível calcular a PCA das métricas.') from e
        k=min(2,vt.shape[0])
        if k<1:
            raise ValueError(self.tr('similarity_no_metrics'))
        coords=Z@vt[:k].T
        delta=coords-coords[ref_idx]
        dist=np.sqrt(np.sum(delta*delta,axis=1))
        dist[~valid_rows]=np.inf
        return {str(g):float(d) for g,d in zip(joined_gids,dist)}

    def _apply_similarity_sort(self):
        if not self.project:return
        ref_gid=str(getattr(self,'_gallery_similarity_ref_gid',None) or '')
        base=self._gallery_base_source().copy()
        if not ref_gid or base.empty or ref_gid not in set(base['gid'].astype(str)):
            messagebox.showinfo(self.tr('iat_title'),self.tr('similarity_need_one'))
            return
        mode=str(self.vars['similarity_mode'].get() or 'pca')
        method=str(self.vars['seg_method'].get())
        def work(progress):
            return self._compute_similarity_scores_for_df(base,ref_gid,mode,method,progress)
        def done(scores):
            self._gallery_similarity_scores=scores
            self._gallery_similarity_active=True
            row=self._similarity_reference_row(base)
            name=str(row.get('names','')) if row is not None else ref_gid
            status=getattr(self,'_gallery_similarity_status_var',None)
            if status is not None:
                crit=self.tr('similarity_pca') if mode=='pca' else self.tr('similarity_tsne')
                status.set(f'{crit} • {name}')
            self.page_idx=0
            if self.current_tab=='gallery':
                self._gallery_update_in_place(reset_page=True,prefetch=True)
        label=self.tr('similarity_pca') if mode=='pca' else self.tr('similarity_tsne')
        self._run_operation(label,work,done)

    def _clear_similarity_sort(self):
        self._gallery_similarity_active=False
        self._gallery_similarity_scores={}
        status=getattr(self,'_gallery_similarity_status_var',None)
        if status is not None:
            try:status.set('')
            except Exception:pass
        self._gallery_update_in_place(reset_page=True,prefetch=True)

    def _gallery_order_keys(self):
        keys=[]
        for key_var,dir_var in [('order','order_dir'),('order2','order2_dir'),('order3','order3_dir')]:
            raw=str(self.vars[key_var].get() or '').strip()
            if not raw:continue
            # Accept values left by older versions, but keep direction separate.
            old_dir=None
            m=re.search(r'_(asc|desc)$',raw)
            if m:
                old_dir=m.group(1); raw=re.sub(r'_(asc|desc)$','',raw)
            if raw=='az':raw='names'
            if raw=='var':raw='var_norm'
            direction=str(old_dir or self.vars[dir_var].get() or 'asc')
            if direction not in ('asc','desc'):direction='asc'
            keys.append(f'{raw}_{direction}')
        return keys

    def _toggle_order_direction(self,varname):
        var=self.vars[varname]
        var.set('desc' if str(var.get())=='asc' else 'asc')
        btn=getattr(self,'_order_direction_buttons',{}).get(varname)
        if btn is not None:
            try:btn.config(text='↑' if var.get()=='asc' else '↓')
            except Exception:pass
        self._gallery_update_in_place(reset_page=True)

    def _gallery_source(self):
        # Start from the complete Gallery membership. Confidence/Variance only
        # filter membership; histogram distributions remain based on this base.
        base=self._gallery_base_source()
        if base.empty:
            return base.copy()

        c,v=self._ranges()
        df=filter_dataframe(base,c,v,False,None)
        order_keys=self._gallery_order_keys()
        if self.project and any(k.startswith('seg_') for k in order_keys):
            method=self.vars['seg_method'].get(); parts=[]
            for bn,ss in df.groupby('csv_basename',sort=False):
                parts.append(self.project.attach_segmentation(ss,str(bn),method))
            if parts:
                df=pd.concat(parts,ignore_index=True,sort=False)

        # Existing priority controls remain useful as deterministic tie-breakers.
        df=sort_gallery_multi(df,order_keys)
        if getattr(self,'_gallery_similarity_active',False):
            scores=getattr(self,'_gallery_similarity_scores',{})
            if scores and 'gid' in df.columns:
                df=df.copy()
                df['__similarity']=df['gid'].astype(str).map(scores).fillna(np.inf)
                df=df.sort_values('__similarity',ascending=True,na_position='last',kind='mergesort')
        return df

    def _gallery_available_orders(self,src):
        extra=[]
        if self.project:
            try:
                csvs=src['csv_basename'].astype(str).unique().tolist() if not src.empty and 'csv_basename' in src.columns else None
                extra=self.project.available_segmentation_columns(self.vars['seg_method'].get(),csvs)
            except Exception: extra=[]
        return order_choices(src,extra_columns=extra)

    def _build_gallery(self):
        host=tk.Frame(self.content,bg=C['bg']);host.pack(fill='both',expand=True);host.columnconfigure(1,weight=1);host.rowconfigure(0,weight=1)
        card,ctrl_host=self._card(host,self.tr('allb_title'));card.grid(row=0,column=0,sticky='ns',padx=(4,7),pady=4);card.config(width=310);card.grid_propagate(False)
        cscroll=ScrollFrame(ctrl_host,C['card']);cscroll.pack(fill='both',expand=True);ctrl=cscroll.inner
        r=tk.Frame(ctrl,bg=C['card']);r.pack(fill='x',pady=(0,6));self._btn(r,self.tr('prev'),lambda:self.change_page(-1)).pack(side='left',fill='x',expand=True);self._btn(r,self.tr('next'),lambda:self.change_page(1)).pack(side='left',fill='x',expand=True,padx=(5,0))
        self.gallery_page_lbl=tk.Label(ctrl,text='',bg=C['card'],fg=C['muted'],font=('Sans',8));self.gallery_page_lbl.pack(anchor='w',pady=(0,4))
        ps=tk.Frame(ctrl,bg=C['card']);ps.pack(fill='x',pady=(1,5))
        tk.Label(ps,text=('Imagens por página (máx. 500)' if self.lang()=='pt' else 'Images per page (max 500)'),
                 bg=C['card'],fg=C['muted'],font=('Sans',8)).pack(side='left')
        self.gallery_page_size_spin=tk.Spinbox(
            ps,from_=1,to=500,increment=10,textvariable=self.vars['page_size'],width=7,
            command=self._gallery_page_size_changed,bg=C['entry'],fg=C['text'],
            insertbackground='white',buttonbackground=C['card2'],relief='flat',
            highlightthickness=1,highlightbackground=C['border']
        )
        self.gallery_page_size_spin.pack(side='right',ipady=2)
        self.gallery_page_size_spin.bind('<Return>',lambda e:self._gallery_page_size_changed())
        self.gallery_page_size_spin.bind('<FocusOut>',lambda e:self._gallery_page_size_changed())

        self._label(ctrl,self.tr('new_label'),bold=True,muted=False).pack(anchor='w',pady=(2,1));self._entry(ctrl,self.vars['new_label']).pack(fill='x',ipady=5)
        for text,var in [(self.tr('save_auto'),self.vars['save_auto']),(self.tr('discard_after'),self.vars['discard_after'])]:
            tk.Checkbutton(ctrl,text=text,variable=var,bg=C['card'],fg=C['text'],selectcolor=C['red'],activebackground=C['card'],activeforeground='white').pack(anchor='w',pady=0)
        tk.Checkbutton(ctrl,text=self.tr('hide_labeled'),variable=self.vars['hide_relabel'],bg=C['card'],fg=C['text'],selectcolor=C['red'],activebackground=C['card'],activeforeground='white',command=lambda:self._gallery_update_in_place(reset_page=True)).pack(anchor='w',pady=0)
        tk.Checkbutton(ctrl,text=self.tr('show_outline'),variable=self.vars['gallery_outline'],bg=C['card'],fg=C['text'],selectcolor=C['red'],activebackground=C['card'],activeforeground='white',command=self._refresh_visible_thumbnails).pack(anchor='w',pady=(2,0))

        self._label(ctrl,self.tr('gallery_segmentation')).pack(anchor='w',pady=(5,1))
        generated=self.project.available_segmentation_methods() if self.project else []
        if generated and self.vars['seg_method'].get() not in generated:self.vars['seg_method'].set(generated[0])
        seg_values=[SEGMENTATION_METHODS.get(m,m) for m in generated]
        seg_rev={SEGMENTATION_METHODS.get(m,m):m for m in generated}
        self.gallery_seg_display=tk.StringVar(value=SEGMENTATION_METHODS.get(self.vars['seg_method'].get(),self.vars['seg_method'].get()) if generated else self.tr('no_segmentation'))
        segcb=ttk.Combobox(ctrl,textvariable=self.gallery_seg_display,values=seg_values or [self.tr('no_segmentation')],state='readonly',style='Dark.TCombobox')
        segcb.pack(fill='x')
        def _gallery_seg_changed(e=None):
            key=seg_rev.get(self.gallery_seg_display.get())
            if key:
                changed=key!=self.vars['seg_method'].get()
                self.vars['seg_method'].set(key)
                if changed and getattr(self,'_gallery_similarity_active',False) and self.vars['similarity_mode'].get()=='pca':
                    self._clear_gallery_similarity_state(clear_reference=False,refresh=False)
            self._gallery_update_in_place(reset_page=True,prefetch=True)
        segcb.bind('<<ComboboxSelected>>',_gallery_seg_changed)

        self._btn(ctrl,self.tr('apply_label'),self.apply_label,C['green']).pack(fill='x',ipady=3,pady=(5,2))
        r=tk.Frame(ctrl,bg=C['card']);r.pack(fill='x')
        self._btn(r,self.tr('invert'),self.invert_checks).pack(side='left',fill='x',expand=True)
        self._btn(r,self.tr('undo'),self.undo_label).pack(side='left',fill='x',expand=True,padx=(5,0))

        # Similarity ordering is intentionally explicit: one checked image becomes
        # the reference, while all images in the Gallery selection are ranked.
        sim=tk.Frame(ctrl,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1)
        sim.pack(fill='x',pady=(8,5))
        tk.Label(sim,text=self.tr('similarity_title'),bg=C['card2'],fg='#ffffff',font=('Sans',8,'bold')).pack(anchor='w',padx=7,pady=(6,3))
        rr=tk.Frame(sim,bg=C['card2']);rr.pack(fill='x',padx=7)
        self._gallery_similarity_preview=tk.Label(rr,text='—',bg=C['log'],fg=C['muted'],width=9,height=4)
        self._gallery_similarity_preview.pack(side='left',padx=(0,6),pady=(0,4))
        self._gallery_similarity_ref_var=tk.StringVar(value='—')
        info=tk.Frame(rr,bg=C['card2']);info.pack(side='left',fill='both',expand=True)
        tk.Label(info,text=self.tr('similarity_reference'),bg=C['card2'],fg=C['muted'],font=('Sans',7)).pack(anchor='w')
        tk.Label(info,textvariable=self._gallery_similarity_ref_var,bg=C['card2'],fg=C['text'],font=('Sans',7),wraplength=165,justify='left').pack(anchor='w')
        self._btn(sim,self.tr('similarity_use_checked'),self._set_similarity_reference_from_checked).pack(fill='x',padx=7,pady=(1,4))

        self._label(sim,self.tr('similarity_metric')).pack(anchor='w',padx=7)
        sim_labels=[self.tr('similarity_pca'),self.tr('similarity_tsne')]
        sim_map={sim_labels[0]:'pca',sim_labels[1]:'tsne'}
        sim_rev={'pca':sim_labels[0],'tsne':sim_labels[1]}
        self._gallery_similarity_display=tk.StringVar(value=sim_rev.get(self.vars['similarity_mode'].get(),sim_labels[0]))
        simcb=ttk.Combobox(sim,textvariable=self._gallery_similarity_display,values=sim_labels,state='readonly',style='Dark.TCombobox')
        simcb.pack(fill='x',padx=7,pady=(1,4))
        def _sim_mode_changed(e=None):
            self.vars['similarity_mode'].set(sim_map.get(self._gallery_similarity_display.get(),'pca'))
            if getattr(self,'_gallery_similarity_active',False):
                self._clear_similarity_sort()
        simcb.bind('<<ComboboxSelected>>',_sim_mode_changed)
        br=tk.Frame(sim,bg=C['card2']);br.pack(fill='x',padx=7,pady=(0,4))
        self._btn(br,self.tr('similarity_apply'),self._apply_similarity_sort,C['blue']).pack(side='left',fill='x',expand=True)
        self._btn(br,self.tr('similarity_clear'),self._clear_similarity_sort).pack(side='left',padx=(4,0))
        self._gallery_similarity_status_var=tk.StringVar(value='')
        tk.Label(sim,textvariable=self._gallery_similarity_status_var,bg=C['card2'],fg=C['cyan'],font=('Sans',7),wraplength=265,justify='left').pack(anchor='w',padx=7,pady=(0,6))
        self._update_similarity_reference_ui()
        if getattr(self,'_gallery_similarity_active',False):
            row=self._similarity_reference_row()
            if row is not None:
                crit=self.tr('similarity_pca') if self.vars['similarity_mode'].get()=='pca' else self.tr('similarity_tsne')
                self._gallery_similarity_status_var.set(f"{crit} • {str(row.get('names',''))}")

        self._label(ctrl,self.tr('filters_order'),bold=True,muted=False).pack(anchor='w',pady=(7,2))
        hist_base=self._gallery_histogram_source()
        conf_vals=pd.to_numeric(hist_base['confs1'],errors='coerce').to_numpy(float) if 'confs1' in hist_base.columns else np.asarray([])
        var_vals=pd.to_numeric(hist_base['var_norm'],errors='coerce').to_numpy(float) if 'var_norm' in hist_base.columns else np.asarray([])
        self._label(ctrl,self.tr('confidence')).pack(anchor='w')
        self.gallery_conf_slider=RangeSlider(ctrl,self.vars['conf_min'],self.vars['conf_max'],values=conf_vals,command=lambda:self._gallery_update_in_place(reset_page=True))
        self.gallery_conf_slider.pack(fill='x',pady=(0,2))
        self._label(ctrl,self.tr('variance')).pack(anchor='w')
        self.gallery_var_slider=RangeSlider(ctrl,self.vars['var_min'],self.vars['var_max'],values=var_vals,command=lambda:self._gallery_update_in_place(reset_page=True))
        self.gallery_var_slider.pack(fill='x',pady=(0,4))

        self._label(ctrl,self.tr('order_priority'),bold=True,muted=False).pack(anchor='w',pady=(3,2))
        src=self._gallery_source(); choices=self._gallery_available_orders(src); values=[x[0] for x in choices];keys={label:key for label,key in choices}; rev={key:label for label,key in choices}
        display_vars=[]
        self._order_direction_buttons={}
        for lab,varname,dirvar in [
            (self.tr('primary'),'order','order_dir'),
            (self.tr('secondary'),'order2','order2_dir'),
            (self.tr('tertiary'),'order3','order3_dir')
        ]:
            row=tk.Frame(ctrl,bg=C['card']);row.pack(fill='x',pady=1)
            tk.Label(row,text=lab,bg=C['card'],fg=C['muted'],font=('Sans',8),width=7,anchor='w').pack(side='left')
            var=self.vars[varname]
            cur_key=str(var.get() or '')
            cur_key=re.sub(r'_(asc|desc)$','',cur_key)
            cur_key={'az':'manual_label','var':'var_norm'}.get(cur_key,cur_key)
            cur=rev.get(cur_key,'—'); dv=tk.StringVar(value=cur); display_vars.append(dv)
            vals=values if varname=='order' else ['—']+values
            cb=ttk.Combobox(row,textvariable=dv,values=vals,state='readonly',style='Dark.TCombobox',width=20);cb.pack(side='left',fill='x',expand=True)
            def _order_changed(e=None,dv=dv,var=var):
                var.set('' if dv.get()=='—' else keys.get(dv.get(),'names'))
                self._gallery_update_in_place(reset_page=True)
            cb.bind('<<ComboboxSelected>>',_order_changed)
            direction=self.vars[dirvar]
            db=self._btn(row,'↑' if direction.get()=='asc' else '↓',lambda d=dirvar:self._toggle_order_direction(d))
            db.config(width=3,font=('Sans',10,'bold'))
            db.pack(side='left',padx=(4,0))
            self._order_direction_buttons[dirvar]=db
        self._order_display_vars=display_vars

        gcard,gbody=self._card(host,self.tr('gallery'));gcard.grid(row=0,column=1,sticky='nsew',padx=(0,4),pady=4)
        info=tk.Frame(gbody,bg=C['card']);info.pack(fill='x');self.gallery_info=tk.Label(info,text='',bg=C['card'],fg=C['green'],font=('Sans',9,'bold'));self.gallery_info.pack(side='right')
        self.gallery_scroll=ScrollFrame(gbody,C['card'],gallery_mode=True);self.gallery_scroll.pack(fill='both',expand=True,pady=(5,0)); self._gallery_card_pool=[]; self._gallery_render_signature=None; self._render_gallery(src)

    def _gallery_refresh(self):
        """Full Gallery rebuild, reserved for structural UI changes."""
        self._gallery_update_token+=1
        self._gallery_prefetch_state=None
        self.show_tab('gallery',rebuild=True)

    def _gallery_page_size_changed(self):
        """Clamp the Gallery page size and update cards without rebuilding the tab."""
        try:
            size=int(float(self.vars['page_size'].get()))
        except Exception:
            size=80
        size=max(1,min(500,size))
        if str(self.vars['page_size'].get())!=str(size):
            self.vars['page_size'].set(str(size))
        self._gallery_update_in_place(reset_page=True,prefetch=True)

    def _gallery_update_in_place(self,reset_page=False,prefetch=True):
        """Show the target page immediately; load its thumbnails independently.

        ``prefetch`` remains in the signature for existing callers. Waiting for
        all images before changing the page made a single slow file stall a page
        of 500 cards. Card creation and thumbnail loading are now progressive.
        """
        if reset_page:
            self.page_idx=0
        if not hasattr(self,'gallery_scroll') or not self.gallery_scroll.winfo_exists():
            self._gallery_refresh()
            return
        src=self._gallery_source()
        self._refresh_gallery_histograms()
        self._patch_gallery_page(src)

    def _finish_gallery_prefetch(self,token,key,im):
        """Accept late events from an older gallery request safely."""
        self._apply_thumbnail_result(key,im)

    def _refresh_visible_thumbnails(self):
        """Refresh visible card images (for example after outline changes)."""
        self._gallery_begin_thumbnail_generation()
        widgets = getattr(self, "_gallery_visible_widgets", {})
        for idx, entry in enumerate(list(widgets.values())):
            if not isinstance(entry, dict):
                continue
            label = entry.get("image")
            if label is None:
                continue
            try:
                label.configure(image=self._gallery_placeholder_photo)
                label.image = self._gallery_placeholder_photo
            except Exception:
                pass
            self._set_thumbnail(
                label,
                str(entry.get("bn", "")),
                str(entry.get("name", "")),
                show_outline=self.vars["gallery_outline"].get(),
                method=self.vars["seg_method"].get(),
                priority=0 if idx < 60 else 2,
            )
        self._schedule_gallery_prefetch()

    def change_page(self,delta):
        src=self._gallery_source()
        try:size=max(1,min(500,int(self.vars['page_size'].get())))
        except Exception:size=80
        pages=max(1,math.ceil(len(src)/size))
        new_idx=max(0,min(pages-1,self.page_idx+delta))
        if new_idx==self.page_idx:
            return
        self.page_idx=new_idx
        # Avoid re-filtering/sorting the same 5,000+ rows for this page click.
        self._refresh_gallery_histograms()
        self._patch_gallery_page(src)

    def _gallery_page_state(self,src):
        """Return normalized pagination state for the current gallery source."""
        self._gallery_current_source=src
        self._gallery_ordered_gids=src['gid'].astype(str).tolist() if not src.empty and 'gid' in src.columns else []
        try:size=max(1,min(500,int(self.vars['page_size'].get())))
        except:size=80
        pages=max(1,math.ceil(len(src)/size))
        self.page_idx=max(0,min(self.page_idx,pages-1))
        page=src.iloc[self.page_idx*size:(self.page_idx+1)*size].copy()
        self._gallery_visible_count=len(page)
        if hasattr(self,'gallery_page_lbl'):
            self.gallery_page_lbl.config(text=f"{self.tr('page')} {self.page_idx+1} / {pages}  •  {len(src):,} imagens".replace(',','.'))
        if hasattr(self,'gallery_info'):
            self.gallery_info.config(text=f"{len(self.checked_gids)} {self.tr('checked')} / {self._gallery_visible_count}")
        return page,pages,size

    def _create_gallery_card(self, inner, rd, idx, cols=5):
        """Create one reusable Gallery card.

        Page changes rebind these widgets instead of destroying/recreating up to
        500 cards, which keeps Tk layout work bounded and makes pagination feel
        immediate.
        """
        gid = str(rd.get("gid"))
        bn = str(rd.get("csv_basename", ""))
        name = str(rd.get("names", ""))
        lab = str(rd.get("manual_label", ""))
        selected = gid in self.checked_gids
        card = tk.Frame(
            inner, bg=C["card2"],
            highlightbackground=C["red"] if selected else C["border"],
            highlightthickness=2 if selected else 1
        )
        card.grid(row=idx // cols, column=idx % cols, sticky="nsew", padx=4, pady=4)
        v = tk.BooleanVar(value=selected)
        chk = tk.Checkbutton(
            card, variable=v, bg=C["card2"], fg="white",
            selectcolor=C["red"], activebackground=C["card2"],
            activeforeground="white"
        )
        chk.pack(anchor="nw")
        if getattr(self, "_gallery_placeholder_photo", None) is None:
            self._gallery_placeholder_photo = tk.PhotoImage(width=105, height=82)
        img_label = tk.Label(
            card, image=self._gallery_placeholder_photo, bg=C["log"],
            width=105, height=82, cursor="hand2"
        )
        img_label.image = self._gallery_placeholder_photo
        img_label.pack(padx=5, pady=(0, 4))
        name_lbl = tk.Label(card, text=name[:22], bg=C["card2"], fg=C["text"], font=("Sans", 7))
        name_lbl.pack(anchor="w", padx=5)
        manual_lbl = tk.Label(card, text=f"manual: {lab}"[:34], bg=C["card2"], fg=C["muted"], font=("Sans", 7))
        manual_lbl.pack(anchor="w", padx=5, pady=(0, 5))
        entry = {
            "var": v, "card": card, "check": chk, "image": img_label,
            "name_label": name_lbl, "manual_label_widget": manual_lbl,
            "gid": gid, "bn": bn, "name": name
        }
        self.gallery_scroll._bind_descendants(card)
        self._reuse_gallery_card(entry, rd, idx, cols)
        return entry


    def _reuse_gallery_card(self, entry, rd, idx, cols=5):
        """Rebind one existing card to a different image without recreating Tk widgets."""
        gid = str(rd.get("gid"))
        bn = str(rd.get("csv_basename", ""))
        name = str(rd.get("names", ""))
        lab = str(rd.get("manual_label", ""))
        selected = gid in self.checked_gids
        card = entry["card"]
        var = entry["var"]
        chk = entry["check"]
        image = entry["image"]
        try:
            card.grid(row=idx // cols, column=idx % cols, sticky="nsew", padx=4, pady=4)
            card.configure(
                bg=C["card2"],
                highlightbackground=C["red"] if selected else C["border"],
                highlightthickness=2 if selected else 1,
            )
            var.set(selected)
            chk.configure(
                variable=var,
                command=lambda g=gid, v=var: self._toggle_check(g, v.get()),
                bg=C["card2"], activebackground=C["card2"]
            )
            chk.bind(
                "<Button-1>",
                lambda e, g=gid, v=var, c=card: self._gallery_check_click(e, g, v, c)
            )
            image.configure(image=self._gallery_placeholder_photo)
            image.image = self._gallery_placeholder_photo
            image.bind("<Button-1>", lambda _e, b=bn, n=name: self.open_full_image(b, n))
            entry["name_label"].configure(text=name[:22])
            entry["manual_label_widget"].configure(text=f"manual: {lab}"[:34])
        except Exception:
            return entry

        entry.update({"gid": gid, "bn": bn, "name": name})
        # First viewport gets priority; the rest of a 500-card page is background work.
        priority = 0 if idx < 60 else 2
        self._set_thumbnail(
            image, bn, name,
            show_outline=self.vars["gallery_outline"].get(),
            method=self.vars["seg_method"].get(),
            priority=priority,
        )
        return entry

    def _gallery_prepare_leave(self):
        """Invalidate Gallery callbacks/jobs without clearing the bounded thumbnail cache."""
        self._gallery_cards_token=getattr(self,'_gallery_cards_token',0)+1
        self._thumb_generation=int(getattr(self,'_thumb_generation',0))+1
        self._thumb_pending=[];self._thumb_pending_keys=set();self._thumb_waiters.clear()
        pending=getattr(self,'_gallery_prefetch_after',None)
        if pending is not None:
            try:self.after_cancel(pending)
            except Exception:pass
            self._gallery_prefetch_after=None
        scroll=getattr(self,'gallery_scroll',None)
        if scroll is not None:
            after_id=getattr(scroll,'_gallery_layout_after',None)
            if after_id is not None:
                try:scroll.after_cancel(after_id)
                except Exception:pass
                try:scroll._gallery_layout_after=None
                except Exception:pass
        self._gallery_prefetch_state=None
        cards=list(getattr(self,'_gallery_card_pool',[]));self._gallery_card_pool=[];self._gallery_visible_widgets={}
        return cards

    def _gallery_begin_thumbnail_generation(self):
        """Cancel queued (not running) requests from older pages."""
        self._thumb_generation = int(getattr(self, "_thumb_generation", 0)) + 1
        self._thumb_pending = []
        self._thumb_pending_keys = set()
        inflight = set(getattr(self, "_thumb_inflight", {}).keys())
        # Keep waiters only for work that is already running; stale queued jobs
        # disappear immediately instead of delaying the next page.
        for key in list(getattr(self, "_thumb_waiters", {}).keys()):
            if key not in inflight:
                self._thumb_waiters.pop(key, None)
        if getattr(self, "_gallery_prefetch_after", None) is not None:
            try:
                self.after_cancel(self._gallery_prefetch_after)
            except Exception:
                pass
            self._gallery_prefetch_after = None
        return self._thumb_generation

    def _queue_thumbnail_request(self, key, bn, name, show_outline, method, priority=1, label=None):
        if label is not None:
            self._thumb_waiters.setdefault(key, []).append(label)
        if key in self._thumb_cache:
            if label is not None:
                photo = self._thumb_cache.get(key)
                try:
                    if label.winfo_exists() and getattr(label, "_iat_thumb_key", None) == key:
                        label.config(image=photo)
                        label.image = photo
                except Exception:
                    pass
            return
        if key in self._thumb_inflight or key in self._thumb_pending_keys:
            return
        self._thumb_seq = int(getattr(self, "_thumb_seq", 0)) + 1
        item = (
            int(priority), self._thumb_seq, int(self._thumb_generation),
            key, bn, name, bool(show_outline), method
        )
        heapq.heappush(self._thumb_pending, item)
        self._thumb_pending_keys.add(key)
        self._thumb_pump()

    def _thumb_pump(self):
        """Keep only a small bounded number of thumbnail decodes in flight."""
        maximum = int(getattr(self, "_thumb_max_inflight", 8))
        while len(self._thumb_inflight) < maximum and self._thumb_pending:
            priority, seq, generation, key, bn, name, show_outline, method = heapq.heappop(self._thumb_pending)
            self._thumb_pending_keys.discard(key)
            if generation != self._thumb_generation:
                if key not in self._thumb_inflight:
                    self._thumb_waiters.pop(key, None)
                continue
            if key in self._thumb_cache:
                self._apply_thumbnail_result(key, None, from_cache=True)
                continue
            if key in self._thumb_inflight:
                continue
            try:
                fut = self._thumb_executor.submit(
                    self._prepare_thumbnail, bn, name, show_outline, method
                )
            except RuntimeError:
                self._thumb_waiters.pop(key, None)
                continue
            self._thumb_inflight[key] = fut
            def ready(f, k=key):
                try:
                    im = f.result()
                except Exception:
                    im = None
                self._event_q.put(("thumb", k, im))
            fut.add_done_callback(ready)

    def _schedule_gallery_prefetch(self):
        """Warm the next page after the visible viewport has been requested."""
        if getattr(self, "_gallery_prefetch_after", None) is not None:
            try:
                self.after_cancel(self._gallery_prefetch_after)
            except Exception:
                pass
        try:
            self._gallery_prefetch_after = self.after(120, self._gallery_prefetch_adjacent_pages)
        except Exception:
            self._gallery_prefetch_after = None

    def _gallery_prefetch_adjacent_pages(self):
        self._gallery_prefetch_after = None
        src = getattr(self, "_gallery_current_source", None)
        if src is None or src.empty:
            return
        try:
            size = max(1, min(500, int(self.vars["page_size"].get())))
        except Exception:
            size = 80
        pages = max(1, math.ceil(len(src) / size))
        candidates = []
        if self.page_idx + 1 < pages:
            candidates.append((self.page_idx + 1, 100))
        if self.page_idx - 1 >= 0:
            candidates.append((self.page_idx - 1, 40))
        show_outline = bool(self.vars["gallery_outline"].get())
        method = self.vars["seg_method"].get()
        for page_idx, limit in candidates:
            part = src.iloc[page_idx * size:(page_idx + 1) * size].head(limit)
            for row in part.itertuples(index=False):
                rd = row._asdict()
                bn = str(rd.get("csv_basename", ""))
                name = str(rd.get("names", ""))
                key = (bn, name, show_outline, method if show_outline else None)
                self._queue_thumbnail_request(
                    key, bn, name, show_outline, method,
                    priority=1, label=None
                )


    def _gallery_fill_cards(self, rows, retained=None, updated_labels=None):
        """Populate/rebind page cards in small Tk slices.

        Cards are pooled and reused across pages. Thumbnail decoding is separately
        bounded, so 500-card pages no longer submit 500 executor jobs at once.
        """
        token = getattr(self, "_gallery_cards_token", 0) + 1
        self._gallery_cards_token = token
        updated_labels = updated_labels or {}
        inner = self.gallery_scroll.inner

        # Keep only live pooled entries.
        pool = []
        for entry in list(getattr(self, "_gallery_card_pool", [])):
            try:
                if entry["card"].winfo_exists():
                    pool.append(entry)
            except Exception:
                pass
        self._gallery_card_pool = pool
        self._gallery_visible_widgets = {}

        def step(offset=0):
            if token != getattr(self, "_gallery_cards_token", None) or self.current_tab != "gallery":
                return
            try:
                if not inner.winfo_exists():
                    return
            except Exception:
                return

            stop = min(offset + 20, len(rows))
            for idx in range(offset, stop):
                rd = dict(rows[idx])
                gid = str(rd.get("gid"))
                if gid in updated_labels:
                    rd["manual_label"] = updated_labels[gid]
                if idx < len(self._gallery_card_pool):
                    entry = self._gallery_card_pool[idx]
                    self._reuse_gallery_card(entry, rd, idx, 5)
                else:
                    entry = self._create_gallery_card(inner, rd, idx, 5)
                    self._gallery_card_pool.append(entry)
                self._gallery_visible_widgets[gid] = entry

            if stop < len(rows):
                self.after(4, lambda: step(stop))
                return

            # Hide surplus pooled cards, but keep them for a later larger page.
            for idx in range(len(rows), len(self._gallery_card_pool)):
                try:
                    self._gallery_card_pool[idx]["card"].grid_remove()
                except Exception:
                    pass
            self._update_gallery_check_widgets()
            self._schedule_gallery_prefetch()

        step()

    def _render_gallery(self, src):
        """Initial render with pooled cards and bounded thumbnail work."""
        page, _, _ = self._gallery_page_state(src)
        inner = self.gallery_scroll.inner
        for c in range(5):
            inner.columnconfigure(c, weight=1)
        self._gallery_begin_thumbnail_generation()
        self._gallery_render_signature = (
            int(self.page_idx),
            tuple(page["gid"].astype(str).tolist()) if "gid" in page.columns else tuple(),
        )
        self._gallery_visible_widgets = {}
        self._gallery_empty_label = None
        # A freshly created Gallery has no useful pool; remove any stale widgets.
        for w in inner.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        self._gallery_card_pool = []
        if page.empty:
            self._gallery_empty_label = tk.Label(
                inner, text=self.tr("no_selection"), bg=C["card"], fg=C["muted"]
            )
            self._gallery_empty_label.pack(pady=50)
        else:
            self._gallery_fill_cards([r._asdict() for r in page.itertuples(index=False)])

    def _patch_gallery_page(self, src, updated_labels=None):
        """Update one page immediately, reusing card widgets across page changes."""
        if not hasattr(self, "gallery_scroll") or not self.gallery_scroll.winfo_exists():
            self._gallery_refresh()
            return
        page, _, _ = self._gallery_page_state(src)
        inner = self.gallery_scroll.inner
        for c in range(5):
            inner.columnconfigure(c, weight=1)
        rows = [r._asdict() for r in page.itertuples(index=False)]
        signature = (
            int(self.page_idx),
            tuple(str(rd.get("gid")) for rd in rows),
        )
        if signature != getattr(self, "_gallery_render_signature", None):
            self._gallery_begin_thumbnail_generation()
            self._gallery_render_signature = signature

        self._gallery_cards_token = getattr(self, "_gallery_cards_token", 0) + 1

        if rows and getattr(self, "_gallery_empty_label", None) is not None:
            try:
                if self._gallery_empty_label.winfo_exists():
                    self._gallery_empty_label.destroy()
            except Exception:
                pass
            self._gallery_empty_label = None

        if not rows:
            self._gallery_visible_widgets = {}
            for entry in getattr(self, "_gallery_card_pool", []):
                try:
                    entry["card"].grid_remove()
                except Exception:
                    pass
            if getattr(self, "_gallery_empty_label", None) is None:
                self._gallery_empty_label = tk.Label(
                    inner, text=self.tr("no_selection"), bg=C["card"], fg=C["muted"]
                )
                self._gallery_empty_label.pack(pady=50)
            return

        self._gallery_fill_cards(rows, updated_labels=updated_labels)

    def _update_gallery_check_widgets(self,gids=None):
        """Update only checkbox/card state for visible gallery items."""
        widgets=getattr(self,'_gallery_visible_widgets',{})
        target_gids=list(widgets.keys()) if gids is None else gids
        for g in target_gids:
            item=widgets.get(str(g))
            if item is None:
                continue
            if isinstance(item,dict):
                var=item.get('var');card=item.get('card')
            else:
                # Backward-safe fallback for a page created before this update.
                try:var,card=item[:2]
                except Exception:continue
            selected=str(g) in self.checked_gids
            try:
                if bool(var.get())!=selected:
                    var.set(selected)
            except Exception:
                pass
            try:
                card.config(
                    highlightbackground=C['red'] if selected else C['border'],
                    highlightthickness=2 if selected else 1,
                )
            except Exception:
                pass
        if hasattr(self,'gallery_info'):
            try:
                self.gallery_info.config(
                    text=f"{len(self.checked_gids)} {self.tr('checked')} / {getattr(self,'_gallery_visible_count',0)}"
                )
            except Exception:
                pass

    def _gallery_check_click(self,event,gid,var,card=None):
        """Checkbox click with desktop-style Shift range selection.

        Only checkbox/card state is changed in-place. The image grid and
        thumbnails are deliberately left untouched, so Shift selection does not
        flash or trigger disk/PIL work.
        """
        gid=str(gid); new_state=not bool(var.get())
        shift=bool(int(getattr(event,'state',0) or 0) & 0x0001)
        ordered=self._gallery_ordered_gids
        if shift and self._gallery_anchor_gid in ordered and gid in ordered:
            a=ordered.index(self._gallery_anchor_gid); b=ordered.index(gid)
            lo,hi=sorted((a,b))
            # Update the logical selection for the complete range, including
            # items on other pages, but touch widgets only on the visible page.
            if new_state:
                self.checked_gids.update(ordered[lo:hi+1])
            else:
                self.checked_gids.difference_update(ordered[lo:hi+1])
            self._gallery_anchor_gid=gid
            self._update_gallery_check_widgets()
        else:
            if new_state:self.checked_gids.add(gid)
            else:self.checked_gids.discard(gid)
            self._gallery_anchor_gid=gid
            self._update_gallery_check_widgets((gid,))
        return 'break'

    def _toggle_check(self,gid,val):
        gid=str(gid)
        if val:self.checked_gids.add(gid)
        else:self.checked_gids.discard(gid)
        self._update_gallery_check_widgets((gid,))

    def _prepare_thumbnail(self,bn,name,show_outline=False,method='otsu'):
        if not self.project or Image is None:return None
        p=self.project.resolve_image(bn,name)
        try:
            if p:
                with Image.open(p) as src:
                    ow,oh=src.size
                    # JPEG can decode directly at a reduced scale (no-op for
                    # PNG/TIFF); the original size is captured first so the
                    # segmentation outline scaling below is unchanged.
                    try:src.draft('RGB',(224,164))
                    except Exception:pass
                    im=src.convert('RGB')
            else:
                im=Image.new('RGB',(128,96),'#050506'); ow,oh=im.size
            im.thumbnail((112,82),Image.Resampling.LANCZOS)
            if show_outline and p:
                ol=self.project.load_outline_cache(bn,method).get(name)
                if ol and ol.get('points'):
                    sx=im.width/max(float(ol.get('width') or ow),1);sy=im.height/max(float(ol.get('height') or oh),1);pts=[(x*sx,y*sy) for x,y in ol['points']]
                    if len(pts)>1: ImageDraw.Draw(im).line(pts+[pts[0]],fill='#ff3b4d',width=1)
            return im.copy()
        except Exception:
            return None

    def _set_thumbnail(self, label, bn, name, show_outline=False, method="otsu", priority=1):
        if not self.project or Image is None or ImageTk is None:
            return
        try:
            if not label.winfo_exists():
                return
        except Exception:
            return
        key = (bn, name, bool(show_outline), method if show_outline else None)
        try:
            label._iat_thumb_key = key
        except Exception:
            pass

        photo = self._thumb_cache.get(key)
        if photo is not None:
            try:
                if getattr(label, "_iat_thumb_key", None) == key:
                    label.config(image=photo)
                    label.image = photo
            except Exception:
                pass
            return

        self._queue_thumbnail_request(
            key, bn, name, show_outline, method,
            priority=priority, label=label
        )

    def _apply_thumbnail_result(self, key, im, from_cache=False):
        """Apply one thumbnail result and immediately release a queue slot."""
        if not from_cache:
            self._thumb_inflight.pop(key, None)
        labels = self._thumb_waiters.pop(key, [])

        photo = self._thumb_cache.get(key)
        if photo is None and im is not None and ImageTk is not None:
            try:
                photo = ImageTk.PhotoImage(im)
            except Exception:
                photo = None
            if photo is not None:
                self._thumb_cache[key] = photo
                while len(self._thumb_cache) > self._thumb_cache_limit:
                    try:
                        self._thumb_cache.pop(next(iter(self._thumb_cache)))
                    except Exception:
                        break

        if photo is not None:
            for label in labels:
                try:
                    if label.winfo_exists() and getattr(label, "_iat_thumb_key", None) == key:
                        label.config(image=photo)
                        label.image = photo
                except Exception:
                    pass

        self._thumb_pump()

    def _nice_scale_bar(self,width_px,pixel_size_um):
        """Choose a conventional physical scale length that fits the image."""
        target_um=max(float(pixel_size_um)*max(float(width_px)*0.42,1.0),float(pixel_size_um))
        # 1-2-5 lengths from 0.1 µm to 100 mm (same rule as the web viewer).
        candidates=[m*10**e for e in range(-1,6) for m in (1,2,5)]
        nice=max([v for v in candidates if v<=target_um] or [candidates[0]])
        bar_px=max(1,int(round(nice/max(float(pixel_size_um),1e-12))))
        label=(f'{nice/1000:g} mm' if nice>=1000 else f'{nice:g} µm').replace('.',',')
        return bar_px,label

    def _format_segmentation_metrics(self,rr,pixel_size_um):
        """Format stored segmentation measurements without changing calculations.

        Area is a foreground-pixel count (displayed as px²) and Feret values are
        lengths in pixels from the current minAreaRect implementation.  Physical
        values are display-time conversions using the configured µm/pixel.
        """
        rows=[]
        def num(key,dec=4):
            try:
                v=float(rr.get(key))
                return v if math.isfinite(v) else None
            except Exception:return None
        version=num('Metrics Version')
        if version is None or version<2:
            rows.append('Métricas v1 (Feret = retângulo mínimo; perímetro em passos de pixel). Gere a segmentação de novo.')
        area=num('Area (pxl)')
        if area is not None:
            rows.append(f'Área: {area:.0f} px²  ({area*pixel_size_um*pixel_size_um:,.1f} µm²)')
        for key,label in [('Perimeter (pxl)','Perímetro (Crofton)'),('ESD (pxl)','ESD'),('Feret_max','Feret máx.'),('Feret_min','Feret mín.'),
                          ('Major (pxl)','Eixo maior'),('Minor (pxl)','Eixo menor')]:
            v=num(key)
            if v is not None:rows.append(f'{label}: {v:.4f} px  ({v*pixel_size_um:.2f} µm)')
        v=num('Biovolume (pxl3)')
        if v is not None:rows.append(f'Biovolume (elipsoide): {v:,.0f} px³  ({v*pixel_size_um**3:,.0f} µm³)')
        for key,label in [('circularity','Circularidade'),('Elongation','Alongamento'),('Rectangularity','Retangularidade'),('Solidity','Solidez'),('Eccentricity','Excentricidade')]:
            v=num(key)
            if v is not None:rows.append(f'{label}: {v:.4f}  (adimensional)')
        v=num('Orientation (deg)')
        if v is not None:rows.append(f'Orientação: {v:.1f}°')
        for key,label in [('Mean Intensity','Intensidade média'),('Median Intensity','Intensidade mediana'),('Std Intensity','Desvio-padrão'),('Contrast','Contraste')]:
            v=num(key)
            if v is not None:rows.append(f'{label}: {v:.4f}  (nível de cinza 0–255)')
        sk=num('Skewness'); ku=num('Kurtosis')
        if sk is not None and ku is not None:rows.append(f'Assimetria: {sk:.3f}  •  Curtose: {ku:.3f}')
        v=num('Percent_on_border')
        if v is not None:rows.append(f'No bordo: {v:.4f}%')
        v=num('Number Holes')
        if v is not None:rows.append(f'Número de orifícios: {v:.0f}')
        if 'Validation State' in rr.index:
            raw=rr.get('Validation State')
            try:valid=bool(float(raw))
            except Exception:valid=bool(raw)
            rows.append(f'Validação: {"sim" if valid else "não"}')
        return rows

    def _segmentation_pixel_size(self, row=None, fallback=None):
        """Return the calibration stored with segmentation when available.

        New segmentation CSVs persist the calibration used at generation time.
        When a project-level calibration exists it is authoritative for current
        physical-unit conversions; the stored segmentation value remains provenance.
        Legacy projects without project settings continue to prefer the stored value.
        """
        candidates=[]
        settings=load_project_settings(self.project.project_dir) if self.project else {}
        explicit_project_px='pixel_size_um_per_px' in settings
        # An explicit project calibration is authoritative.  Segmentation masks
        # and pixel metrics are independent of µm/pixel, so an updated project
        # calibration can safely refresh physical-unit display/export without
        # recomputing the mask.  Legacy projects with no project setting keep
        # using the calibration stored in each segmentation CSV first.
        if explicit_project_px:
            candidates.append(self.vars['pixel_size'].get())
            candidates.append(fallback)
        if row is not None:
            try:candidates.append(row.get('Pixel Size (um/pixel)'))
            except Exception:pass
        if not explicit_project_px:
            candidates.append(fallback)
            candidates.append(self.vars['pixel_size'].get())
        candidates.append(DEFAULT_PIXEL_SIZE_UM_PER_PX)
        for raw in candidates:
            try:
                return parse_pixel_size(raw)
            except Exception:
                pass
        return DEFAULT_PIXEL_SIZE_UM_PER_PX

    def open_full_image(self,bn,name):
        if not self.project or Image is None or ImageTk is None:return
        p=self.project.resolve_image(str(bn),str(name))
        if not p or not p.exists(): self._log(self.tr('no_image'),'warn'); return
        try:
            im=Image.open(p).convert('RGB'); ow,oh=im.size
            method=self.vars['seg_method'].get()
            if self.vars['gallery_outline'].get():
                ol=self.project.load_outline_cache(str(bn),method).get(str(name))
                if ol and ol.get('points'):
                    pts=[(float(x),float(y)) for x,y in ol['points']]
                    if len(pts)>1: ImageDraw.Draw(im).line(pts+[pts[0]],fill='#ff3048',width=1)
            fallback_um=self._segmentation_pixel_size(None,self.vars['pixel_size'].get())
            seg_row=None
            try:
                sdf=self.project.get_segmentation_table(str(bn),method)
                if not sdf.empty and 'Image Name' in sdf.columns:
                    sm=sdf[sdf['Image Name'].astype(str)==str(name)]
                    if not sm.empty:
                        seg_row=sm.iloc[0]
            except Exception:
                seg_row=None
            um=self._segmentation_pixel_size(seg_row,fallback_um)

            # Conventional single-length scale bar in a dedicated footer.  The
            # physical conversion is exactly pixel_count * µm/pixel.
            bar_px,bar_label=self._nice_scale_bar(ow,um)
            footer_h=54; cw=max(ow,bar_px+32); ch=oh+footer_h
            combined=Image.new('RGB',(cw,ch),'black');combined.paste(im,(0,0))
            dr=ImageDraw.Draw(combined); y=oh+18; x0=16; x1=x0+bar_px
            dr.line((x0,y,x1,y),fill='white',width=3)
            dr.line((x0,y-5,x0,y+5),fill='white',width=2);dr.line((x1,y-5,x1,y+5),fill='white',width=2)
            fnt=None
            try:
                from PIL import ImageFont
                # Portable fallback chain: Pillow default is used if no TTF is found.
                for fp in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','C:/Windows/Fonts/arial.ttf'):
                    if os.path.exists(fp):
                        fnt=ImageFont.truetype(fp,11);break
            except Exception: pass
            try:
                bbox=dr.textbbox((0,0),bar_label,font=fnt); tw=bbox[2]-bbox[0]
            except Exception: tw=max(20,len(bar_label)*6)
            dr.text((x0+(bar_px-tw)/2,oh+31),bar_label,fill='white',font=fnt)

            # Current label comes from the live dataframe (therefore it reflects
            # relabeling even when the segmentation CSV predates that edit).
            class_name='—'
            try:
                q=self.current_df
                m=q[(q['csv_basename'].astype(str)==str(bn))&(q['names'].astype(str)==str(name))]
                if not m.empty:class_name=str(m.iloc[0].get('manual_label','—'))
            except Exception:pass

            win=tk.Toplevel(self);win.title(f'{self.tr("full_image")} — {name}');win.configure(bg=C['bg']);win.geometry(f'{min(cw+410,1500)}x{min(ch+90,900)}')
            outer=tk.Frame(win,bg=C['bg']);outer.pack(fill='both',expand=True,padx=8,pady=8);outer.columnconfigure(0,weight=1);outer.rowconfigure(0,weight=1)
            fr=tk.Frame(outer,bg=C['bg']);fr.grid(row=0,column=0,sticky='nsew');cv=tk.Canvas(fr,bg='black',highlightthickness=0);sx=tk.Scrollbar(fr,orient='horizontal',command=cv.xview);sy=tk.Scrollbar(fr,orient='vertical',command=cv.yview);cv.configure(xscrollcommand=sx.set,yscrollcommand=sy.set);sx.pack(side='bottom',fill='x');sy.pack(side='right',fill='y');cv.pack(side='left',fill='both',expand=True)
            photo=ImageTk.PhotoImage(combined);cv.create_image(0,0,image=photo,anchor='nw');cv.configure(scrollregion=(0,0,cw,ch));win._photo=photo
            def popup_wheel(e):
                try:
                    if int(getattr(e,'state',0) or 0)&0x0001:
                        step=-1 if (getattr(e,'delta',0)>0 or getattr(e,'num',None)==4) else 1;cv.xview_scroll(step,'units')
                    else:
                        step=-1 if (getattr(e,'delta',0)>0 or getattr(e,'num',None)==4) else 1;cv.yview_scroll(step,'units')
                    return 'break'
                except Exception:return
            cv.bind('<MouseWheel>',popup_wheel);cv.bind('<Button-4>',popup_wheel);cv.bind('<Button-5>',popup_wheel)

            metrics=tk.Frame(outer,bg=C['card'],highlightbackground=C['border'],highlightthickness=1,width=355);metrics.grid(row=0,column=1,sticky='ns',padx=(8,0));metrics.grid_propagate(False)
            tk.Label(metrics,text=self.tr('metrics'),bg=C['card'],fg=C['text'],font=('Sans',12,'bold')).pack(anchor='w',padx=12,pady=(12,4))
            tk.Label(metrics,text=f'{self.tr("class")}: {class_name}',bg=C['card'],fg=C['green'],font=('Sans',10,'bold'),wraplength=325,justify='left').pack(anchor='w',padx=12,pady=(0,5))
            tk.Label(metrics,text=f'{self.tr("method")}: {SEGMENTATION_METHODS.get(method,method)}',bg=C['card'],fg=C['cyan'],font=('Sans',8,'bold'),wraplength=325,justify='left').pack(anchor='w',padx=12,pady=(0,2))
            tk.Label(metrics,text=f'Pixel size: {um:.2f} µm/pixel  •  Imagem: {ow} × {oh} px',bg=C['card'],fg=C['muted'],font=('Sans',8),wraplength=325,justify='left').pack(anchor='w',padx=12,pady=(0,8))
            metric_text=self._format_segmentation_metrics(seg_row,um) if seg_row is not None else []
            if not metric_text:metric_text=[self.tr('no_segmentation')]
            tk.Label(metrics,text='\n'.join(metric_text),justify='left',anchor='nw',bg=C['card'],fg=C['text'],font=('Monospace',8),wraplength=325).pack(fill='both',expand=True,padx=12,pady=(0,12))
            self._full_image_refs.append(win);win.bind('<Destroy>',lambda e,w=win:self._full_image_refs.remove(w) if w in self._full_image_refs else None)
        except Exception as e:self._log(f'{self.tr("no_image")}: {e}','error')

    def apply_label(self):
        if not self.project:return
        if not self.checked_gids:self._log(self.tr('no_checked'),'warn');return
        lab=self.vars['new_label'].get().strip()
        if not lab:self._log(self.tr('label_empty'),'warn');return
        gids=list(self.checked_gids);save_auto=bool(self.vars['save_auto'].get())
        edited_was_visible=False
        try:
            ids=set(map(str,gids));pre=self.current_df.loc[self.current_df['gid'].astype(str).isin(ids)];edited_was_visible=bool(set(pre['manual_label'].astype(str))&set(self.visible_classes))
        except Exception:pass
        def work(progress):
            t=time.perf_counter();n=self.project.apply_labels(gids,lab);label_s=time.perf_counter()-t;save_s=0.0
            if save_auto:t=time.perf_counter();self.project.save_dirty(progress);save_s=time.perf_counter()-t
            return n,label_s,save_s
        def done(result):
            n,label_s,save_s=result;cb_t=time.perf_counter();self._log(f'{n} imagens → {lab}','success');self._log(f'Diagnóstico rótulo: atualização={label_s:.3f}s, salvamento={save_s:.3f}s.','muted');removed=set(map(str,gids))
            try:
                if self.current_df is not None and not self.current_df.empty and 'gid' in self.current_df.columns:
                    mask=self.current_df['gid'].astype(str).isin(removed)
                    if mask.any():self.current_df.loc[mask,'manual_label']=lab
            except Exception:pass
            try:
                if self._analysis_df is not None and not self._analysis_df.empty and 'gid' in self._analysis_df.columns:
                    mask=self._analysis_df['gid'].astype(str).isin(removed)
                    if mask.any():self._analysis_df.loc[mask,'manual_label']=lab
                    self._analysis_labels_stale=True
            except Exception:pass
            try:
                if self._validation_candidates is not None and not self._validation_candidates.empty and 'gid' in self._validation_candidates.columns:
                    mask=self._validation_candidates['gid'].astype(str).isin(removed)
                    if mask.any():self._validation_candidates.loc[mask,'manual_label']=lab
            except Exception:pass
            try:
                if self._scatter_class_column(self.current_df)=='manual_label':
                    live=set(self.current_df['manual_label'].fillna('').astype(str).tolist());live.discard('');self.visible_classes.intersection_update(live)
                    if edited_was_visible and lab in live:self.visible_classes.add(lab)
            except Exception:pass
            # checked_gids is Gallery state; selected_gids (Projection) is intentionally untouched.
            self.checked_gids.difference_update(removed)
            if self._gallery_selection_df is not None and not self._gallery_selection_df.empty:
                mask=self._gallery_selection_df['gid'].astype(str).isin(removed)
                if mask.any():self._gallery_selection_df.loc[mask,'manual_label']=lab
            if self.vars['discard_after'].get():
                self._gallery_discarded_gids.update(removed)
                if self.allb_gids:self.allb_gids=[g for g in self.allb_gids if str(g) not in removed]
                if self._gallery_selection_df is not None:self._gallery_selection_df=self._gallery_selection_df[~self._gallery_selection_df['gid'].astype(str).isin(removed)].copy()
                if self._gallery_anchor_gid in removed:self._gallery_anchor_gid=None
            self._apply_filters();self.page_idx=0 if self.page_idx<0 else self.page_idx;src=self._gallery_source();self._patch_gallery_page(src,{str(g):lab for g in removed});self._log(f'Diagnóstico rótulo: callback de conclusão={time.perf_counter()-cb_t:.3f}s.','muted')
        self._run_operation(self.tr('saving'),work,done)

    def invert_checks(self):
        src=self._gallery_source();
        try:size=int(self.vars['page_size'].get())
        except:size=80
        page=src.iloc[self.page_idx*size:(self.page_idx+1)*size]
        g=set(page['gid'].astype(str))
        self.checked_gids=(self.checked_gids-g)|(g-self.checked_gids)
        # Inverting checkboxes is also a local state change: never rebuild
        # thumbnails/cards just to flip their selected appearance.
        self._update_gallery_check_widgets(g)

    def undo_label(self):
        if not self.project:return
        n=self.project.undo();self._log(f'Undo: {n} imagens','info');self.change_batch()
