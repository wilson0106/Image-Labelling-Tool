#!/usr/bin/env python3
"""Workspace gráfico do IAT integrado à dashboard Tkinter do ILT."""
from __future__ import annotations

import math
import os
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageDraw, ImageTk
except Exception:
    Image = ImageDraw = ImageTk = None

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    from matplotlib.collections import LineCollection
    from matplotlib.path import Path as MplPath
    from matplotlib.colors import LinearSegmentedColormap, to_rgba
    from matplotlib.widgets import LassoSelector, RectangleSelector
    from matplotlib.ticker import FuncFormatter
except Exception:
    FigureCanvasTkAgg = NavigationToolbar2Tk = Figure = LineCollection = MplPath = LinearSegmentedColormap = to_rgba = LassoSelector = RectangleSelector = FuncFormatter = None

dendrogram = ConvexHull = gaussian_filter = None
_scipy_helpers_checked = False


def _load_scipy_helpers():
    global dendrogram, ConvexHull, gaussian_filter, _scipy_helpers_checked
    if not _scipy_helpers_checked:
        _scipy_helpers_checked = True
        try:
            from scipy.cluster.hierarchy import dendrogram as _dendrogram
            from scipy.spatial import ConvexHull as _ConvexHull
            from scipy.ndimage import gaussian_filter as _gaussian_filter
            dendrogram = _dendrogram
            ConvexHull = _ConvexHull
            gaussian_filter = _gaussian_filter
        except Exception:
            dendrogram = ConvexHull = gaussian_filter = None
    return dendrogram is not None

from iat_backend import (
    IATProject, SEGMENTATION_METHODS, analyze_embedding_space, cluster_groups,
    filter_dataframe, generate_selection_cluster_artifact, get_color, get_class_color,
    reconcile_class_cluster_artifacts,
    read_artifact_names, read_cluster_artifact, run_segmentation, sanitize_filename, validation_capabilities,
)
from platform_utils import open_path
from hw_resources import available_cpus
from ilt_contracts import resolve_scatter_class_column
from project_settings import DEFAULT_PIXEL_SIZE_UM_PER_PX, format_pixel_size, parse_pixel_size, project_pixel_size, save_project_settings
from ui_resources import IAT_COLORS as C, IAT_TEXT as IAT_T
from ui_contrast import enforce_readable_tree, readable_text_on
from segmentation_docs import METRIC_FIELDS_TEXT, METRIC_FORMULAS, method_doc_text

_TQDM_LINE_RE=re.compile(r'\d{1,3}%\|.*\|\s*\d+/\d+')
from iat_widgets import ScrollFrame
from dashboard_widgets import GradientBar, RoundedPanel
from iat_filename_filters import ImageNameFilter, filter_image_names
from iat_temporal_spatial import TemporalSpatialPanel
from iat_export_mixin import IATExportMixin
from iat_training_export_mixin import IATTrainingExportMixin
from iat_gallery_mixin import IATGalleryMixin
from iat_model_cycle_mixin import IATModelCycleMixin


class IATWorkspace(IATModelCycleMixin, IATGalleryMixin, IATTrainingExportMixin, IATExportMixin, tk.Frame):
    def __init__(self,parent,app):
        super().__init__(parent,bg=C['bg'])
        self.app=app; self.lang=lambda: app.lang
        self.project:Optional[IATProject]=None
        self.current_df=pd.DataFrame(); self.filtered_df=pd.DataFrame()
        self.selected_gids:set[str]=set(); self.checked_gids:set[str]=set(); self.allb_gids:List[str]=[]; self.page_idx=0
        self.current_tab='scatter'; self.current_cluster_path:Optional[Path]=None
        self._thumb_cache={}; self._outline_cache={}; self._preview_photo=None
        # Thumbnails are prepared off the Tk thread. Keep only a bounded number
        # of PhotoImages so browsing many pages does not grow memory indefinitely.
        self._thumb_executor=ThreadPoolExecutor(max_workers=max(4,min(8,available_cpus())), thread_name_prefix='iat-thumb')
        self._thumb_waiters={}
        # Bound thumbnail work independently from the executor queue. This avoids
        # submitting hundreds of stale image decodes when the user changes page.
        self._thumb_pending=[]
        self._thumb_pending_keys=set()
        self._thumb_inflight={}
        self._thumb_generation=0
        self._thumb_seq=0
        self._thumb_max_inflight=max(4,min(16,int(os.environ.get('ILT_IAT_THUMB_INFLIGHT','8'))))
        self._thumb_cache_limit=max(600,int(os.environ.get('ILT_IAT_THUMB_CACHE','1600')))
        self._gallery_prefetch_after=None
        self._gallery_card_pool=[]
        self._gallery_render_signature=None
        self._gallery_selection_df=None
        self._gallery_anchor_gid=None; self._gallery_ordered_gids=[]; self._gallery_current_source=None
        self._gallery_visible_widgets={}; self._gallery_visible_count=0; self._gallery_empty_label=None
        self._gallery_discarded_gids:set[str]=set()
        self._gallery_update_token=0
        self._gallery_prefetch_state=None
        self._gallery_similarity_ref_gid=None
        self._gallery_similarity_scores={}
        self._gallery_similarity_active=False
        self._similarity_ref_photo=None
        self._plot_resize_after={}
        self._op_running=False; self._op_thread=None; self._op_stop=False; self._op_start=None
        self._event_q=queue.Queue(); self._op_done_cb=None
        self.log_lines=[]; self.saved_count=0
        self._selection_tool=None; self._scatter_canvas=None; self._scatter_ax=None; self._scatter_points_df=pd.DataFrame()
        self._scatter_artist_gids={}; self._legend_artist_classes={}; self._scatter_pick_cid=None; self._class_scroll_cid=None; self._operation_origin_tab='scatter'
        self._scatter_toolbar=None; self._class_axis_scale_label=None; self._all_classes_btn=None
        self._name_filter_spec=ImageNameFilter(); self._name_filter_cache=None
        self._name_filter_count_text=None
        self._selection_outline=None; self._full_image_refs=[]
        # Embedding-based validation is session-local. Original x/y and labels are
        # never overwritten by a projection or an unsupervised method.
        self._analysis_df=pd.DataFrame(); self._analysis_meta={}; self._analysis_projection_active=False
        self._analysis_projection_method='original'; self._validation_candidates=pd.DataFrame()
        self.visible_classes:set[str]=set(); self._class_listbox=None
        self._active_local_progress=None; self._active_local_status=None
        self._dendro_canvas=None
        # Clustering visualization is prepared off the Tk thread.  The token
        # discards stale results when the user changes cutoff or leaves/reopens
        # the tab, while the debounce avoids recomputing during slider drags.
        self._cluster_view_token=0
        self._cluster_cut_after=None
        self._poll_last_perf=time.perf_counter(); self._last_poll_lag_log=0.0; self._last_nav_block_log=0.0
        self.vars={
            'project':tk.StringVar(),'batch':tk.StringVar(value='Overview'),'conf_min':tk.StringVar(value='0.0'),'conf_max':tk.StringVar(value='1.0'),
            'var_min':tk.StringVar(value='0.0'),'var_max':tk.StringVar(value='1.0'),'hide_relabel':tk.BooleanVar(value=False),
            'class_filter':tk.StringVar(value='Todas'),'show_hulls':tk.BooleanVar(value=False),'show_density':tk.BooleanVar(value=False),
            'name_date_from':tk.StringVar(),'name_date_to':tk.StringVar(),
            'name_time_from':tk.StringVar(),'name_time_to':tk.StringVar(),
            'name_depth_min':tk.StringVar(),'name_depth_max':tk.StringVar(),
            'name_attribute_regex':tk.StringVar(),'name_attribute_value':tk.StringVar(),
            'name_filter_stains':tk.BooleanVar(value=True),
            'page_size':tk.StringVar(value='80'),'new_label':tk.StringVar(),'save_auto':tk.BooleanVar(value=True),'discard_after':tk.BooleanVar(value=True),
            'order':tk.StringVar(value='manual_label'),'order2':tk.StringVar(value=''),'order3':tk.StringVar(value=''),
            'order_dir':tk.StringVar(value='asc'),'order2_dir':tk.StringVar(value='asc'),'order3_dir':tk.StringVar(value='asc'),
            'cluster_class':tk.StringVar(),'cluster_metric':tk.StringVar(value='cosine'),
            'cluster_linkage':tk.StringVar(value='average'),'cluster_cut':tk.DoubleVar(value=0.5),'seg_method':tk.StringVar(value='otsu'),
            'seg_radius':tk.StringVar(value='25'),'pixel_size':tk.StringVar(value=format_pixel_size(DEFAULT_PIXEL_SIZE_UM_PER_PX)),'seg_outline':tk.BooleanVar(value=False),'gallery_outline':tk.BooleanVar(value=False),
            'seg_scope':tk.StringVar(value='selected'),'similarity_mode':tk.StringVar(value='pca'),
            'save_name':tk.StringVar(),'export_dir':tk.StringVar(),'csv_output_dir':tk.StringVar(),
            'export_scope':tk.StringVar(value='current'),'export_all_classes':tk.BooleanVar(value=True),
            'export_class_search':tk.StringVar(),'export_save_metadata':tk.BooleanVar(value=True),
            'export_create_subfolders':tk.BooleanVar(value=True),'export_keep_names':tk.BooleanVar(value=True),
            'export_include_outline':tk.BooleanVar(value=False),'export_only_segmented':tk.BooleanVar(value=False),
            'export_view':tk.StringVar(value='csv'),
            'training_scope':tk.StringVar(value='all'),'training_seed':tk.StringVar(value='42'),
            'training_unit':tk.StringVar(value='image'),'training_all_classes':tk.BooleanVar(value=True),'training_dir':tk.StringVar(),'training_name':tk.StringVar(value='ilt_dataset'),
            'training_keep_names':tk.BooleanVar(value=True),'training_manifest':tk.BooleanVar(value=True),
            'training_config':tk.BooleanVar(value=True),'training_stats':tk.BooleanVar(value=True),
            'class_axis_mode':tk.StringVar(value='N'),'class_axis_scale':tk.DoubleVar(value=100.0),'overview_multi_classes':tk.BooleanVar(value=False),
            'projection_method':tk.StringVar(value='original'),'scatter_color_mode':tk.StringVar(value='manual_label'),
            'analysis_pca_dim':tk.StringVar(value='50'),'analysis_knn_k':tk.StringVar(value='15'),'analysis_min_cluster':tk.StringVar(value='15'),
            'review_threshold':tk.DoubleVar(value=0.55),
            'review_use_conf':tk.BooleanVar(value=True),'review_use_var':tk.BooleanVar(value=True),
            'review_use_knn':tk.BooleanVar(value=True),'review_use_outlier':tk.BooleanVar(value=True),
            'review_use_pred_mismatch':tk.BooleanVar(value=True),'review_use_cluster_mismatch':tk.BooleanVar(value=False),
        }
        self._export_class_vars={}
        self._export_field_vars={}
        self._export_class_host=None
        self._export_class_count_lbl=None
        self._export_seg_method_rev={}
        self._training_batch_selection=set()
        self._training_class_vars={}
        self._training_group_vars={}
        self._training_split_vars={}
        self._training_class_host=None
        self._training_group_host=None
        self._training_split_host=None
        self._training_preview_tree=None
        self._training_context_df=pd.DataFrame()
        self._training_context_key=None
        self._training_context_loading=False
        self._training_batch_label=None
        self._training_summary_label=None
        self._training_progress_var=tk.DoubleVar(value=0)
        self.status_var=tk.StringVar(value=self.tr('ready')); self.operation_var=tk.StringVar(value=''); self.elapsed_var=tk.StringVar(value='00:00:00')
        self.progress_var=tk.DoubleVar(value=0); self.progress_text=tk.StringVar(value='0%')
        self._build()
        # Cross-platform mouse-wheel routing for every ScrollFrame in the IAT.
        # The handler only consumes the wheel when the pointer is actually over
        # a scrollable IAT frame, so Matplotlib/Listbox wheel behavior is kept.
        self.bind_all('<MouseWheel>', self._route_mousewheel, add='+')
        self.bind_all('<Button-4>', self._route_mousewheel, add='+')
        self.bind_all('<Button-5>', self._route_mousewheel, add='+')
        self.app.register_callback(self.refresh_language)
        initial=getattr(app,'last_output_path',None)
        if initial and (Path(initial)/'dataframes').is_dir():
            self.vars['project'].set(str(initial)); self.after(120,self.load_project)
        self.after(100,self._poll_events)
        self.after(500,self._tick)

    def tr(self,key): return IAT_T.get(self.lang(),IAT_T['pt']).get(key,key)
    def refresh_language(self):
        if not self.winfo_exists(): return
        # Rebuild persistent chrome too, so PT/EN affects every visible string.
        self._build_project_bar()
        self.show_tab(self.current_tab,rebuild=True)

    # -------- common UI --------
    def _build(self):
        # IAT uses the full workspace width. Runtime status/logs live in the Logs
        # sub-tab instead of occupying a permanent right-hand column.
        self.columnconfigure(0,weight=1); self.rowconfigure(0,weight=1)
        self.left=tk.Frame(self,bg=C['bg']); self.left.grid(row=0,column=0,sticky='nsew',padx=14,pady=14)
        self.left.rowconfigure(2,weight=1); self.left.columnconfigure(0,weight=1)
        head=tk.Frame(self.left,bg=C['bg']); head.grid(row=0,column=0,sticky='ew',padx=4,pady=(0,8)); self.iat_head=head
        head_row=tk.Frame(head,bg=C['bg']); head_row.pack(anchor='w')
        try:
            iat_icon = getattr(self.app, '_brand_page_icons', {}).get('iat')
        except Exception:
            iat_icon = None
        if iat_icon is not None:
            tk.Label(head_row, image=iat_icon, bg=C['bg']).pack(side='left', padx=(0,8))
        tk.Label(head_row,text=self.tr('iat_title'),bg=C['bg'],fg=C['text'],font=('Sans',17,'bold')).pack(side='left')
        tk.Label(head,text=self.tr('iat_sub'),bg=C['bg'],fg=C['muted'],font=('Sans',9)).pack(anchor='w',pady=(2,0))
        self.project_bar=tk.Frame(self.left,bg=C['card'],highlightbackground=C['border'],highlightthickness=1)
        self.project_bar.grid(row=1,column=0,sticky='ew',padx=4,pady=(0,8)); self._build_project_bar()
        self.content=tk.Frame(self.left,bg=C['bg']); self.content.grid(row=2,column=0,sticky='nsew')
        self.log=None; self.status_badge=None; self.stop_btn=None
        self.show_tab('scatter',rebuild=True)

    def _build_project_bar(self):
        f=self.project_bar
        for w in f.winfo_children(): w.destroy()
        for idx in range(9):
            f.columnconfigure(idx,weight=1 if idx<8 else 0)
        tk.Label(f,text=self.tr('project_folder'),bg=C['card'],fg=C['muted'],font=('Sans',8,'bold')).grid(row=0,column=0,columnspan=4,sticky='w',padx=12,pady=(8,2))
        e=tk.Entry(f,textvariable=self.vars['project'],bg=C['entry'],fg=C['text'],insertbackground='white',relief='flat')
        e.grid(row=1,column=0,columnspan=3,sticky='ew',padx=(12,5),pady=(0,9),ipady=5)
        tk.Button(f,text=self.tr('browse'),command=self.browse_project,bg=C['card2'],fg=C['text'],relief='flat').grid(row=1,column=3,sticky='ew',padx=(0,10),pady=(0,9),ipady=3)
        tk.Label(f,text=self.tr('batch'),bg=C['card'],fg=C['muted'],font=('Sans',8,'bold')).grid(row=0,column=4,columnspan=2,sticky='w',padx=8,pady=(8,2))
        self.batch_combo=ttk.Combobox(f,textvariable=self.vars['batch'],state='readonly',style='Dark.TCombobox')
        self.batch_combo.grid(row=1,column=4,columnspan=2,sticky='ew',padx=8,pady=(0,9),ipady=3); self.batch_combo.bind('<<ComboboxSelected>>',lambda e:self.change_batch())
        tk.Label(f,text=self.tr('mode'),bg=C['card'],fg=C['muted'],font=('Sans',8,'bold')).grid(row=0,column=6,sticky='w',padx=8,pady=(8,2))
        self.mode_lbl=tk.Label(f,text='—',bg=C['card'],fg=C['text'],font=('Sans',9,'bold')); self.mode_lbl.grid(row=1,column=6,sticky='w',padx=8,pady=(0,9))
        tk.Label(f,text=self.tr('pixel_size'),bg=C['card'],fg=C['muted'],font=('Sans',8,'bold')).grid(row=0,column=7,sticky='w',padx=8,pady=(8,2))
        self.project_pixel_entry=tk.Entry(f,textvariable=self.vars['pixel_size'],width=9,bg=C['entry'],fg=C['text'],insertbackground='white',relief='flat')
        self.project_pixel_entry.grid(row=1,column=7,sticky='ew',padx=8,pady=(0,9),ipady=4)
        self.project_pixel_entry.bind('<Return>',lambda e:self._persist_project_pixel_size())
        self.project_pixel_entry.bind('<FocusOut>',lambda e:self._persist_project_pixel_size(silent=True))
        tk.Label(f,text=self.tr('images'),bg=C['card'],fg=C['muted'],font=('Sans',8,'bold')).grid(row=0,column=8,sticky='w',padx=8,pady=(8,2))
        self.image_count_lbl=tk.Label(f,text='—',bg=C['card'],fg=C['cyan'],font=('Sans',10,'bold')); self.image_count_lbl.grid(row=1,column=8,sticky='w',padx=(8,12),pady=(0,9))

    def _build_status(self):
        f=self.right
        for w in f.winfo_children(): w.destroy()
        top=tk.Frame(f,bg=C['panel']); top.pack(fill='x',padx=14,pady=(14,5))
        self.status_title=tk.Label(top,text=self.tr('operation'),bg=C['panel'],fg=C['text'],font=('Sans',12,'bold')); self.status_title.pack(side='left')
        tk.Label(top,textvariable=self.elapsed_var,bg=C['panel'],fg=C['muted'],font=('Monospace',8)).pack(side='right')
        line=tk.Frame(f,bg=C['panel']); line.pack(fill='x',padx=14,pady=4)
        tk.Label(line,text=self.tr('status'),bg=C['panel'],fg=C['muted'],font=('Sans',8)).pack(side='left')
        self.status_badge=tk.Label(line,textvariable=self.status_var,bg='#103a2a',fg=C['green'],font=('Sans',8,'bold'),padx=12,pady=3); self.status_badge.pack(side='left',padx=10)
        tk.Label(f,textvariable=self.operation_var,bg=C['panel'],fg=C['muted'],anchor='w',font=('Sans',8)).pack(fill='x',padx=14,pady=(4,4))
        ph=tk.Frame(f,bg=C['panel']); ph.pack(fill='x',padx=14,pady=(4,2)); tk.Label(ph,text=self.tr('progress'),bg=C['panel'],fg=C['text'],font=('Sans',9,'bold')).pack(side='left'); tk.Label(ph,textvariable=self.progress_text,bg=C['panel'],fg=C['muted'],font=('Sans',8)).pack(side='right')
        self.pb=ttk.Progressbar(f,maximum=100,variable=self.progress_var,style='Blue.Horizontal.TProgressbar'); self.pb.pack(fill='x',padx=14,pady=(0,12))
        hdr=tk.Frame(f,bg=C['panel']); hdr.pack(fill='x',padx=14,pady=(6,4)); tk.Label(hdr,text=self.tr('last_events'),bg=C['panel'],fg=C['text'],font=('Sans',9,'bold')).pack(side='left')
        tk.Button(hdr,text=self.tr('clear'),command=self.clear_log,bg=C['card2'],fg=C['text'],relief='flat').pack(side='right')
        box=tk.Frame(f,bg=C['log'],highlightbackground=C['border'],highlightthickness=1); box.pack(fill='both',expand=True,padx=14,pady=(0,10))
        self.log=tk.Text(box,bg=C['log'],fg='#e8e8ea',insertbackground='white',relief='flat',wrap='word',font=('Monospace',8)); self.log.pack(fill='both',expand=True,padx=4,pady=4)
        for tag,col in [('normal','#e8e8ea'),('muted','#7f93aa'),('info',C['cyan']),('success',C['green']),('warn',C['amber']),('error',C['red']),('cluster',C['purple'])]: self.log.tag_configure(tag,foreground=col)
        self.log.configure(state='disabled')
        bottom=tk.Frame(f,bg=C['panel']); bottom.pack(fill='x',padx=14,pady=(0,14))
        self.stop_btn=tk.Button(bottom,text=self.tr('stop'),command=self.stop_operation,bg='#5b2630',fg='white',relief='flat',state='disabled'); self.stop_btn.pack(side='left',ipadx=10,ipady=5)
        tk.Button(bottom,text=self.tr('open_folder'),command=self.open_project_folder,bg=C['card2'],fg=C['text'],relief='flat').pack(side='right',ipadx=10,ipady=5)
        for old in self.log_lines: self._insert_log(*old,store=False)

    def _card(self,parent,title):
        """Rounded card: returns (outer widget for geometry, content frame)."""
        fr=RoundedPanel(parent,bg=C['card'],border=C['border'],radius=12,pad=1)
        inner=fr.inner
        head=tk.Frame(inner,bg=C['card']); head.pack(fill='x',padx=12,pady=(10,6))
        accent=tk.Canvas(head,width=4,height=16,bg=C['card'],highlightthickness=0,bd=0)
        accent.create_rectangle(0,0,4,8,fill=C['grad_a'],width=0); accent.create_rectangle(0,8,4,16,fill=C['grad_b'],width=0)
        accent.pack(side='left',padx=(0,8))
        tk.Label(head,text=title,bg=C['card'],fg=C['text'],font=('Sans',10,'bold')).pack(side='left')
        body=tk.Frame(inner,bg=C['card']); body.pack(fill='both',expand=True,padx=12,pady=(0,12)); return fr,body
    def _entry(self,parent,var): return tk.Entry(parent,textvariable=var,bg=C['entry'],fg=C['text'],insertbackground='white',relief='flat',highlightthickness=1,highlightbackground=C['border'],font=('Sans',9))
    def _label(self,parent,text,bold=False,muted=True): return tk.Label(parent,text=text,bg=parent.cget('bg'),fg=C['muted'] if muted else C['text'],font=('Sans',8,'bold' if bold else 'normal'))
    def _btn(self,parent,text,cmd,color=None):
        bg=color or C['card2']
        # Text colour follows the background (white on the bright accent
        # colours was below WCAG AA; e.g. 2.1:1 on green, 2.7:1 on purple).
        return tk.Button(parent,text=text,command=cmd,bg=bg,fg=readable_text_on(bg),activebackground=C['border'],activeforeground='white',relief='flat',bd=0,highlightthickness=1,highlightbackground=bg if color else C['border'],padx=10,pady=3,cursor='hand2',font=('Sans',9,'bold' if color else 'normal'))

    def _route_mousewheel(self,event):
        """Scroll the scrollable IAT area currently under the pointer (Linux/Windows)."""
        try:
            widget=self.winfo_containing(event.x_root,event.y_root)
        except Exception:
            return
        target_canvas=None; cur=widget
        while cur is not None:
            training_canvas=getattr(cur,'_training_scroll_canvas',None)
            if training_canvas is not None:
                target_canvas=training_canvas; break
            if isinstance(cur,ScrollFrame):
                target_canvas=cur.canvas; break
            cur=getattr(cur,'master',None)
        if target_canvas is None:
            return
        try:
            if getattr(event,'num',None)==4: steps=-1
            elif getattr(event,'num',None)==5: steps=1
            else:
                delta=int(getattr(event,'delta',0) or 0)
                if delta==0:return
                steps=-max(1,abs(delta)//120) if delta>0 else max(1,abs(delta)//120)
            target_canvas.yview_scroll(int(steps),'units')
            return 'break'
        except Exception:
            return

    def _fit_plot_canvas(self,canvas,figure):
        """Force a Matplotlib figure to the actual Tk widget size."""
        if canvas is None or figure is None:return
        try:
            widget=canvas.get_tk_widget()
            if not widget.winfo_exists():return
            widget.update_idletasks()
            width=int(widget.winfo_width()); height=int(widget.winfo_height())
            if width<120 or height<120:return
            dpi=float(figure.get_dpi() or 100.0)
            figure.set_size_inches(width/dpi,height/dpi,forward=False)
            canvas.draw_idle()
        except Exception:
            pass

    def _bind_plot_autofit(self,canvas,figure):
        """Debounced configure handler plus delayed first fit for new tabs."""
        if canvas is None or figure is None:return
        widget=canvas.get_tk_widget(); key=str(widget)
        def schedule(event=None):
            old=self._plot_resize_after.get(key)
            if old is not None:
                try:self.after_cancel(old)
                except Exception:pass
            try:self._plot_resize_after[key]=self.after(55,lambda:self._fit_plot_canvas(canvas,figure))
            except Exception:pass
        widget.bind('<Configure>',schedule,add='+')
        self.after_idle(schedule)
        self.after(140,schedule)

    def _refit_active_plots(self):
        if self.current_tab=='scatter':
            self._fit_plot_canvas(getattr(self,'_scatter_canvas',None),getattr(self,'scatter_fig',None))
        elif self.current_tab=='clustering':
            self._fit_plot_canvas(getattr(self,'_dendro_canvas',None),getattr(self,'cluster_fig',None))


    # -------- nav/view --------
    def show_tab(self,tab,rebuild=False):
        if self._op_running and tab not in {self.current_tab,'logs',self._operation_origin_tab}:
            now=time.monotonic()
            if now-getattr(self,'_last_nav_block_log',0.0)>0.8:self._last_nav_block_log=now;self._log(f'Navegação para “{tab}” bloqueada: operação ativa em “{self._operation_origin_tab}”.','warn')
            return
        old_tab=self.current_tab
        if old_tab=='clustering' and tab!='clustering':self._cluster_view_token+=1
        active=[w for w in self.content.winfo_children() if not getattr(w,'_iat_retired',False)]
        deferred=(old_tab=='gallery' and tab!='gallery' and bool(active));gallery_cards=[]
        if deferred:
            try:gallery_cards=self._gallery_prepare_leave()
            except Exception as exc:self._log(f'Diagnóstico Galeria: falha ao encerrar callbacks: {exc}','warn')
            for w in active:
                try:w._iat_retired=True;w.pack_forget();w.grid_forget()
                except Exception:pass
        else:
            t=time.perf_counter()
            for w in active:
                try:w.destroy()
                except Exception:pass
            elapsed=time.perf_counter()-t
            if old_tab=='gallery' and elapsed>0.05:self._log(f'Diagnóstico Galeria: destruição síncrona={elapsed:.3f}s.','muted')
        self.current_tab=tab
        if hasattr(self.app,'set_iat_subnav_active'):self.app.set_iat_subnav_active(tab)
        try:
            if tab=='model_cycle':self.iat_head.grid_remove();self.project_bar.grid_remove()
            else:self.iat_head.grid();self.project_bar.grid()
        except Exception:pass
        self.log=None;self.status_badge=None;self.stop_btn=None
        builders={'scatter':self._build_scatter,'gallery':self._build_gallery,'temporal':self._build_temporal,'clustering':self._build_clustering,'model_cycle':self._build_model_cycle,'segmentation':self._build_segmentation,'export':self._build_export,'logs':self._build_logs}
        t=time.perf_counter();builders.get(tab,self._build_scatter)();elapsed=time.perf_counter()-t
        if tab=='clustering' and elapsed>0.03:self._log(f'Diagnóstico Clustering: construção inicial da aba={elapsed:.3f}s.','muted')
        if deferred:self._destroy_retired_gallery_async(active,gallery_cards)
        self.app.after_idle(self.app._apply_font_scale);self.after(90,self._refit_active_plots);self.after(220,self._refit_active_plots)
        self.after_idle(self._enforce_contrast)

    def _enforce_contrast(self):
        try:
            if self.winfo_exists():enforce_readable_tree(self)
        except Exception:
            pass

    # -------- project/data --------
    def _persist_project_pixel_size(self, silent=False):
        """Validate and persist the project-level calibration used by IAT."""
        try:
            value=parse_pixel_size(self.vars['pixel_size'].get())
        except ValueError as exc:
            fallback=project_pixel_size(self.project.project_dir, DEFAULT_PIXEL_SIZE_UM_PER_PX) if self.project else DEFAULT_PIXEL_SIZE_UM_PER_PX
            self.vars['pixel_size'].set(format_pixel_size(fallback))
            if not silent:
                messagebox.showwarning(self.tr('iat_title'),str(exc))
            return False
        self.vars['pixel_size'].set(format_pixel_size(value))
        if self.project:
            try:
                current=project_pixel_size(self.project.project_dir, DEFAULT_PIXEL_SIZE_UM_PER_PX)
                if abs(float(current)-float(value))>1e-12:
                    save_project_settings(
                        self.project.project_dir,
                        pixel_size_um_per_px=value,
                        source='iat',
                        extra={'project_name': self.project.name},
                    )
                    self._log(f'Pixel size: {format_pixel_size(value)} µm/pixel','info')
            except Exception as exc:
                if not silent:
                    messagebox.showwarning(self.tr('iat_title'),str(exc))
                return False
        return True

    def _destroy_retired_gallery_async(self,roots,cards):
        """Destroy hidden Gallery cards in short slices after the next tab is visible."""
        roots=list(roots or []);cards=[e.get('card') if isinstance(e,dict) else e for e in (cards or [])];start=time.perf_counter();state={'idx':0}
        def step():
            deadline=time.perf_counter()+0.008
            while state['idx']<len(cards) and time.perf_counter()<deadline:
                w=cards[state['idx']];state['idx']+=1
                try:
                    if w is not None and w.winfo_exists():w.destroy()
                except Exception:pass
            if state['idx']<len(cards):
                try:self.after(1,step)
                except Exception:pass
                return
            for root in roots:
                try:
                    if root.winfo_exists():root.destroy()
                except Exception:pass
            elapsed=time.perf_counter()-start
            if elapsed>0.05:self._log(f'Diagnóstico Galeria: limpeza diferida de {len(cards)} cards={elapsed:.3f}s.','muted')
        try:self.after_idle(step)
        except Exception:step()

    def browse_project(self):
        p=filedialog.askdirectory(initialdir=self.vars['project'].get() or str(self.app.project_root))
        if p: self.vars['project'].set(p); self.load_project()
    def load_project(self):
        p=Path(self.vars['project'].get()).expanduser()
        if not (p/'dataframes').is_dir(): messagebox.showerror(self.tr('iat_title'),self.tr('invalid_project')); return
        def work(progress):
            pr=IATProject(p); csvs=pr.csv_names(); return pr,csvs
        def done(res):
            self.project,csvs=res
            self.vars['pixel_size'].set(format_pixel_size(project_pixel_size(self.project.project_dir, DEFAULT_PIXEL_SIZE_UM_PER_PX)))
            self.visible_classes.clear()
            self.vars['overview_multi_classes'].set(False)
            self.batch_combo['values']=['Overview']+csvs
            initial_batch=csvs[0] if csvs else 'Overview'
            self.vars['batch'].set(initial_batch)
            self.mode_lbl.config(text='Overview' if initial_batch=='Overview' else 'Batch')
            self._log(f'Projeto carregado: {p}','success')
            self.change_batch()
        self._run_operation(self.tr('loading'),work,done)
    def change_batch(self):
        if not self.project: return
        bn=self.vars['batch'].get() or 'Overview'; self.mode_lbl.config(text='Overview' if bn=='Overview' else 'Batch')
        def work(progress):
            if bn=='Overview': return self.project.load_overview(progress=progress)
            return self.project.load_batch(bn)
        def done(df):
            self.current_df=df
            self._name_filter_cache=None
            if bn=='Overview':
                # Overview deliberately starts visually empty. The user chooses
                # which class density to render; multi-class mode is opt-in.
                self.visible_classes.clear()
                self.vars['overview_multi_classes'].set(False)
            self.selected_gids.clear(); self.checked_gids.clear(); self.allb_gids=[]; self._gallery_selection_df=None; self._gallery_anchor_gid=None; self._gallery_discarded_gids.clear(); self._clear_gallery_similarity_state(clear_reference=True,refresh=False)
            self._analysis_df=pd.DataFrame(); self._analysis_meta={}; self._analysis_projection_active=False; self._analysis_projection_method='original'; self._validation_candidates=pd.DataFrame(); self._analysis_labels_stale=False
            self.vars['projection_method'].set('original'); self.vars['scatter_color_mode'].set('manual_label')
            self.page_idx=0; self._apply_filters(); self._update_header_count(); self.show_tab(self.current_tab,rebuild=True)
            self._log(f'{bn}: {len(df):,} imagens'.replace(',','.'),'info')
        self._run_operation(self.tr('loading'),work,done)
    def _update_header_count(self):
        try:self.image_count_lbl.config(text=f'{len(self.current_df):,}'.replace(',','.'))
        except Exception:pass
    def _ranges(self):
        try:c=(float(self.vars['conf_min'].get()),float(self.vars['conf_max'].get()))
        except:c=(0,1)
        try:v=(float(self.vars['var_min'].get()),float(self.vars['var_max'].get()))
        except:v=(0,1)
        return c,v
    def _apply_filters(self):
        c,v=self._ranges(); cls=self.vars['class_filter'].get(); classes=None if cls in ('Todas','All','') else [cls]
        self.filtered_df=filter_dataframe(self.current_df,c,v,self.vars['hide_relabel'].get(),classes)
    def refresh_current_view(self): self._apply_filters(); self.show_tab(self.current_tab,rebuild=True)

    # -------- scatter --------
    def _analysis_plot_source(self):
        if not self._analysis_projection_active or self._analysis_df is None or self._analysis_df.empty:
            return self.current_df
        if self.current_df is None or self.current_df.empty or 'gid' not in self.current_df.columns:
            return self._analysis_df
        metrics=[c for c in self._analysis_df.columns if c.startswith('analysis_') or c.startswith('knn_') or c.startswith('hdbscan_') or c in {
            'review_priority','manual_prediction_mismatch','source_prediction_mismatch'
        }]
        metrics=[c for c in metrics if c in self._analysis_df.columns]
        base=self.current_df.loc[self.current_df['gid'].astype(str).isin(set(self._analysis_df['gid'].astype(str)))].copy()
        addon=self._analysis_df[['gid']+metrics].drop_duplicates('gid')
        base=base.merge(addon,on='gid',how='left',suffixes=('','__analysis'))
        if {'analysis_x','analysis_y'}.issubset(base.columns):
            base['x']=pd.to_numeric(base['analysis_x'],errors='coerce')
            base['y']=pd.to_numeric(base['analysis_y'],errors='coerce')
            base=base.loc[np.isfinite(base['x'].to_numpy(float)) & np.isfinite(base['y'].to_numpy(float))]
        return base

    def _name_filtered_source(self, df):
        """Optional *visual-only* subset; never writes or recomputes project data."""
        if df is None or df.empty or not self._name_filter_spec.enabled:
            if self._name_filter_count_text is not None:
                self._name_filter_count_text.set("Sem filtro ativo" if self.lang()=='pt' else "No active filter")
            return df
        if 'names' not in df.columns:
            if self._name_filter_count_text is not None:
                self._name_filter_count_text.set("Coluna names indisponível" if self.lang()=='pt' else "names column unavailable")
            return df.iloc[:0]
        key=(id(df),len(df),self._name_filter_spec)
        if self._name_filter_cache is None or self._name_filter_cache[0]!=key:
            mask=filter_image_names(df['names'],self._name_filter_spec)
            self._name_filter_cache=(key,mask)
        mask=self._name_filter_cache[1]
        out=df.loc[mask]
        if self._name_filter_count_text is not None:
            n=f'{len(out):,}'.replace(',','.')
            total=f'{len(df):,}'.replace(',','.')
            self._name_filter_count_text.set(
                (f'{n} / {total} imagens no filtro' if self.lang()=='pt' else f'{n} / {total} images matched')
            )
        return out

    def _apply_name_filter(self):
        spec=ImageNameFilter(
            date_from=self.vars['name_date_from'].get().strip(),
            date_to=self.vars['name_date_to'].get().strip(),
            time_from=self.vars['name_time_from'].get().strip(),
            time_to=self.vars['name_time_to'].get().strip(),
            depth_min=self.vars['name_depth_min'].get().strip(),
            depth_max=self.vars['name_depth_max'].get().strip(),
            attribute_regex=self.vars['name_attribute_regex'].get().strip(),
            attribute_value=self.vars['name_attribute_value'].get().strip(),
        )
        try:
            spec.validate()
        except (ValueError, re.error) as exc:
            messagebox.showerror(
                ('Filtro do nome da imagem' if self.lang()=='pt' else 'Image name filter'),
                str(exc),parent=self,
            )
            return
        self._name_filter_spec=spec
        self._name_filter_cache=None
        self._draw_scatter()

    def _clear_name_filter(self):
        for name in ('name_date_from','name_date_to','name_time_from','name_time_to',
                     'name_depth_min','name_depth_max',
                     'name_attribute_regex','name_attribute_value'):
            self.vars[name].set('')
        self._name_filter_spec=ImageNameFilter()
        self._name_filter_cache=None
        self._draw_scatter()

    def _scatter_filtered(self):
        df=self._analysis_plot_source()
        df=self._name_filtered_source(df)

        if df is None or df.empty:
            return pd.DataFrame()

        if (
            self.vars['hide_relabel'].get()
            and {'manual_label','correct_label'}.issubset(df.columns)
        ):
            df=df[
                df['manual_label'].astype(str)
                ==df['correct_label'].astype(str)
            ]

        class_col=self._scatter_class_column(df)

        if class_col in df.columns:
            class_values=(
                df[class_col]
                .fillna('')
                .astype(str)
                .str.strip()
            )

            all_labels=set(
                class_values[
                    class_values.str.len()>0
                ].unique().tolist()
            )
        else:
            class_values=pd.Series(
                '',
                index=df.index,
                dtype='object',
            )
            all_labels=set()

        is_overview=(
            self.vars['batch'].get()
            =='Overview'
        )

        # In Overview an empty set intentionally means "render no class".
        # Batch mode keeps the previous behaviour of showing all classes.
        if (
            not self.visible_classes
            and not is_overview
        ):
            self.visible_classes=set(
                all_labels
            )

        visible=(
            self.visible_classes
            & all_labels
        )

        if visible:
            df=df[
                class_values.isin(
                    visible
                )
            ]
        else:
            df=df.iloc[0:0]

        return df

    def _audit_mode_active(self, df=None):
        df=self.current_df if df is None else df
        if df is None or 'audit_source_labels' not in df.columns:return False
        s=df['audit_source_labels']
        try:
            if getattr(s,'dtype',None) == bool:return bool(s.any())
            return bool(s.fillna('').astype(str).str.strip().str.lower().isin({'1','true','t','yes','y'}).any())
        except Exception:return False

    def _projection_display_maps(self):
        if self.lang()=='pt':
            pairs=[
                ('Original t-SNE','original'),('PCA 2D','pca'),('t-SNE recalculado','tsne'),
                ('UMAP','umap'),('UMAP supervisionado*','umap_supervised'),('PaCMAP','pacmap'),
            ]
        else:
            pairs=[
                ('Original t-SNE','original'),('PCA 2D','pca'),('Recomputed t-SNE','tsne'),
                ('UMAP','umap'),('Supervised UMAP*','umap_supervised'),('PaCMAP','pacmap'),
            ]
        return pairs,{label:key for label,key in pairs},{key:label for label,key in pairs}

    def _color_display_maps(self):
        pt=self.lang()=='pt'
        candidates=[
            (('Rótulo manual' if pt else 'Manual label'),'manual_label'),
            (('Classe prevista' if pt else 'Predicted class'),'prediction'),
        ]
        if self.current_df is not None and 'source_label' in self.current_df.columns:
            candidates.append((('Rótulo de origem' if pt else 'Source label'),'source_label'))
        analysis_cols=set(self._analysis_df.columns) if self._analysis_df is not None else set()
        if 'hdbscan_cluster' in analysis_cols:candidates.append(('Cluster HDBSCAN','hdbscan_cluster'))
        if 'hdbscan_outlier' in analysis_cols:candidates.append((('Outlier HDBSCAN' if pt else 'HDBSCAN outlier'),'hdbscan_outlier'))
        if 'knn_agreement' in analysis_cols:candidates.append((('Concordância kNN' if pt else 'kNN agreement'),'knn_agreement'))
        if 'review_priority' in analysis_cols:candidates.append((('Prioridade de revisão' if pt else 'Review priority'),'review_priority'))
        if self.current_df is not None and 'confs1' in self.current_df.columns:candidates.append((('Confiança' if pt else 'Confidence'),'confs1'))
        if self.current_df is not None and 'var_norm' in self.current_df.columns:candidates.append((('Variância' if pt else 'Variance'),'var_norm'))
        if 'source_prediction_mismatch' in analysis_cols:
            candidates.append((('Origem ≠ previsão' if pt else 'Source ≠ prediction'),'source_prediction_mismatch'))
        elif 'manual_prediction_mismatch' in analysis_cols:
            candidates.append((('Manual ≠ previsão' if pt else 'Manual ≠ prediction'),'manual_prediction_mismatch'))
        return candidates,{label:key for label,key in candidates},{key:label for label,key in candidates}

    # Shared label-source contract: class filtering follows the active categorical field.
    def _scatter_class_column(self, df=None):
        """Return the class-filter column associated with the Scatter color mode.

        The semantic mapping is centralized in ``ilt_contracts`` so Gallery,
        Scatter and future tools cannot silently reinterpret prediction/source
        provenance as the current human label.
        """
        if df is None:
            df=self.current_df
        if df is None:
            return 'manual_label'

        mode=str(self.vars['scatter_color_mode'].get() or 'manual_label')
        col=resolve_scatter_class_column(mode, df.columns)
        return col or 'manual_label'


    def _scatter_class_labels(self, df=None):
        """Return clean/sorted categories for the current Scatter class field."""

        if df is None:
            try:
                df=self._analysis_plot_source()
            except Exception:
                df=self.current_df

        if df is None or df.empty:
            return []

        col=self._scatter_class_column(df)

        if col not in df.columns:
            return []

        values=(
            df[col]
            .fillna('')
            .astype(str)
            .str.strip()
        )

        values=values[
            values.str.len()>0
        ]

        return sorted(
            values.unique().tolist()
        )


    def _refresh_scatter_class_selector(self, reset=False):
        """Synchronize Listbox membership with the active Scatter color field."""

        labels=self._scatter_class_labels()

        is_overview=(
            self.vars['batch'].get()
            =='Overview'
        )

        label_set=set(labels)

        if reset:
            self.visible_classes.clear()

            # Batch mode historically starts with every class visible.
            # Overview deliberately remains empty until the user chooses.
            if not is_overview:
                self.visible_classes=set(labels)

        else:
            self.visible_classes.intersection_update(
                label_set
            )

            # Preserve historic Batch behaviour: an uninitialized empty
            # selector means show every class.
            if (
                not is_overview
                and not self.visible_classes
                and labels
            ):
                self.visible_classes=set(labels)

        lb=self._class_listbox

        if lb is None:
            return labels

        try:
            if not lb.winfo_exists():
                return labels
        except Exception:
            return labels

        try:
            lb.delete(
                0,
                'end',
            )

            for label in labels:
                lb.insert(
                    'end',
                    label,
                )

            self._sync_class_listbox()

        except Exception:
            pass

        return labels


    def _analysis_scope_rows(self):
        if self.current_df is None or self.current_df.empty:
            raise ValueError('Nenhum dado carregado.' if self.lang()=='pt' else 'No data loaded.')
        if self.selected_gids:
            gids=set(map(str,self.selected_gids))
            rows=self.current_df.loc[self.current_df['gid'].astype(str).isin(gids)].copy()
            if len(rows)<2:
                raise ValueError('Selecione pelo menos duas imagens.' if self.lang()=='pt' else 'Select at least two images.')
            return rows,'selection'
        if self.vars['batch'].get()=='Overview':
            msg=('No Overview, selecione pontos primeiro. A análise estrutural não é executada '
                 'automaticamente sobre milhões de imagens.'
                 if self.lang()=='pt' else
                 'In Overview, select points first. Structural analysis is not run automatically over millions of images.')
            raise ValueError(msg)
        return self.current_df.copy(),'batch'

    def _merge_analysis_metrics_into_current(self, analysis):
        if analysis is None or analysis.empty or self.current_df is None or self.current_df.empty:
            return
        metrics=[c for c in analysis.columns if c.startswith('knn_') or c.startswith('hdbscan_') or c in {
            'review_priority','manual_prediction_mismatch','source_prediction_mismatch','analysis_x','analysis_y'
        }]
        lookup=analysis.set_index(analysis['gid'].astype(str))
        gids=self.current_df['gid'].astype(str)
        text_cols={'knn_dominant_label','hdbscan_dominant_label'}
        for col in metrics:
            mapped=gids.map(lookup[col].to_dict())
            # Only replace analyzed rows; preserve a dtype compatible with the
            # diagnostic field so future pandas versions do not reject updates.
            if col not in self.current_df.columns:
                self.current_df[col]=pd.Series(
                    ['']*len(self.current_df) if col in text_cols else [np.nan]*len(self.current_df),
                    index=self.current_df.index,
                    dtype='object' if col in text_cols else 'float64',
                )
            mask=mapped.notna()
            if mask.any():
                if col in text_cols:
                    self.current_df.loc[mask,col]=mapped.loc[mask].astype(str).to_numpy()
                else:
                    self.current_df.loc[mask,col]=pd.to_numeric(mapped.loc[mask],errors='coerce').to_numpy()

    def _analysis_parameters(self):
        try:pca_dim=max(2,int(float(self.vars['analysis_pca_dim'].get())))
        except Exception:pca_dim=50
        try:knn_k=max(1,int(float(self.vars['analysis_knn_k'].get())))
        except Exception:knn_k=15
        try:min_cluster=max(2,int(float(self.vars['analysis_min_cluster'].get())))
        except Exception:min_cluster=15
        return pca_dim,knn_k,min_cluster

    def _run_projection_analysis(self):
        if not self.project:return
        method=str(self.vars['projection_method'].get() or 'original')
        if method=='original':
            self._restore_original_projection()
            return
        try:
            rows,scope=self._analysis_scope_rows()
            pca_dim,knn_k,min_cluster=self._analysis_parameters()
        except Exception as e:
            messagebox.showinfo(self.tr('iat_title'),str(e));return
        label_col='source_label' if ('source_label' in rows.columns and self._audit_mode_active(rows)) else 'manual_label'
        def work(progress):
            return analyze_embedding_space(
                self.project,rows,projection=method,pca_dim=pca_dim,knn_k=knn_k,
                min_cluster_size=min_cluster,label_col=label_col,
                supervised_label_col=label_col,progress=progress,
            )
        def done(res):
            analysis,meta=res
            self._analysis_df=analysis
            self._analysis_meta=meta
            self._analysis_projection_active=True
            self._analysis_projection_method=method
            self._analysis_labels_stale=False
            self._merge_analysis_metrics_into_current(analysis)
            self._recompute_review_priority(refresh=False)
            trust=meta.get('trustworthiness')
            clusters=meta.get('hdbscan_clusters',0)
            noise=100.0*float(meta.get('hdbscan_noise_fraction',0.0) or 0.0)
            note=f"{method.upper()} • N={len(analysis):,} • HDBSCAN={clusters} • noise={noise:.1f}%".replace(',','.')
            if trust is not None:note+=f" • trust={trust:.3f}"
            self._log(note,'success')
            if self.current_tab=='scatter':self.show_tab('scatter',rebuild=True)
        self._run_operation(('Análise estrutural' if self.lang()=='pt' else 'Structural analysis'),work,done)

    def _restore_original_projection(self):
        self._analysis_projection_active=False
        self._analysis_projection_method='original'
        self.vars['projection_method'].set('original')
        if self.current_tab=='scatter':self.show_tab('scatter',rebuild=True)

    def _recompute_review_priority(self, refresh=True):
        if self._analysis_df is None or self._analysis_df.empty:return
        df=self._analysis_df
        comps=[]
        if self.vars['review_use_conf'].get() and 'confs1' in df.columns:
            a=1.0-pd.to_numeric(df['confs1'],errors='coerce').to_numpy(float);comps.append(np.clip(a,0,1))
        if self.vars['review_use_var'].get() and 'var_norm' in df.columns:
            a=pd.to_numeric(df['var_norm'],errors='coerce').to_numpy(float);comps.append(np.clip(a,0,1))
        if self.vars['review_use_knn'].get() and 'knn_agreement' in df.columns:
            a=1.0-pd.to_numeric(df['knn_agreement'],errors='coerce').to_numpy(float);comps.append(np.clip(a,0,1))
        if self.vars['review_use_outlier'].get() and 'hdbscan_outlier' in df.columns:
            a=pd.to_numeric(df['hdbscan_outlier'],errors='coerce').to_numpy(float);comps.append(np.clip(a,0,1))
        if self.vars['review_use_pred_mismatch'].get():
            col='source_prediction_mismatch' if self._audit_mode_active(df) and 'source_prediction_mismatch' in df.columns else 'manual_prediction_mismatch'
            if col in df.columns:comps.append(pd.to_numeric(df[col],errors='coerce').fillna(0).to_numpy(float))
        # Deliberately opt-in: heterogeneous valid classes can occupy several clusters.
        if self.vars['review_use_cluster_mismatch'].get() and 'hdbscan_label_mismatch' in df.columns:
            comps.append(pd.to_numeric(df['hdbscan_label_mismatch'],errors='coerce').fillna(0).to_numpy(float))
        if comps:
            M=np.column_stack(comps);finite=np.isfinite(M);den=finite.sum(axis=1)
            score=np.divide(np.nansum(np.where(finite,M,0.0),axis=1),den,out=np.zeros(len(df)),where=den>0)
        else:
            score=np.zeros(len(df))
        self._analysis_df=self._analysis_df.copy()
        self._analysis_df['review_priority']=np.clip(score,0,1)
        self._merge_analysis_metrics_into_current(self._analysis_df)
        try:thr=float(self.vars['review_threshold'].get())
        except Exception:thr=.55
        self._validation_candidates=self._analysis_df.loc[self._analysis_df['review_priority']>=thr].sort_values('review_priority',ascending=False,kind='mergesort').copy()
        if refresh and self.current_tab=='scatter':
            self._draw_scatter()

    def _build_scatter(self):
        host=tk.Frame(self.content,bg=C['bg']); host.pack(fill='both',expand=True)
        host.columnconfigure(1,weight=1); host.rowconfigure(0,weight=1)
        card,ctrl_host=self._card(host,self.tr('scatter')); card.grid(row=0,column=0,sticky='ns',padx=(4,7),pady=4); card.config(width=310); card.grid_propagate(False)
        cscroll=ScrollFrame(ctrl_host,C['card']); cscroll.pack(fill='both',expand=True); ctrl=cscroll.inner
        stale_count=0
        if self.current_df is not None and 'projection_stale' in self.current_df.columns:
            stale_series=self.current_df['projection_stale']
            if stale_series.dtype==bool:
                stale_count=int(stale_series.fillna(False).sum())
            else:
                stale_count=int(stale_series.fillna(False).astype(str).str.lower().isin({'1','true','t','yes','y'}).sum())
        if stale_count:
            tk.Label(
                ctrl,
                text=(f'⚠ Projeção x/y desatualizada para {stale_count:,} imagem(ns). As features pertencem a um modelo mais novo; recalcule a projeção antes de interpretar distâncias.'
                      if self.lang()=='pt' else
                      f'⚠ x/y projection is stale for {stale_count:,} image(s). Features belong to a newer model; recompute the projection before interpreting distances.'),
                bg=C['card'],fg=C['amber'],wraplength=260,justify='left',font=('Sans',7,'bold')
            ).pack(anchor='w',fill='x',pady=(0,6))

        # Projection and coloring are independent. The original t-SNE remains the
        # default and requires no recomputation.
        self._label(ctrl,('Projeção' if self.lang()=='pt' else 'Projection'),bold=True,muted=False).pack(anchor='w',pady=(0,3))
        pairs,p_to_key,key_to_p=self._projection_display_maps()
        self._projection_display_var=tk.StringVar(value=key_to_p.get(self.vars['projection_method'].get(),pairs[0][0]))
        pcb=ttk.Combobox(ctrl,textvariable=self._projection_display_var,values=[x[0] for x in pairs],state='readonly',style='Dark.TCombobox')
        pcb.pack(fill='x')
        def _projection_choice(e=None):
            self.vars['projection_method'].set(p_to_key.get(self._projection_display_var.get(),'original'))
        pcb.bind('<<ComboboxSelected>>',_projection_choice)
        prow=tk.Frame(ctrl,bg=C['card']);prow.pack(fill='x',pady=(4,2))
        self._btn(prow,('Calcular' if self.lang()=='pt' else 'Compute'),self._run_projection_analysis,C['blue']).pack(side='left',fill='x',expand=True)
        self._btn(prow,('Original' if self.lang()=='pt' else 'Original'),self._restore_original_projection).pack(side='left',fill='x',expand=True,padx=(5,0))
        tk.Label(
            ctrl,
            text=('* Supervisionado usa rótulos com peso moderado; mantenha o UMAP não supervisionado para procurar erros.'
                  if self.lang()=='pt' else
                  '* Supervised mode uses labels with moderate weight; use unsupervised UMAP when looking for label errors.'),
            bg=C['card'],fg=C['muted'],wraplength=260,justify='left',font=('Sans',7)
        ).pack(anchor='w',pady=(1,5))

        self._label(ctrl,('Colorir por' if self.lang()=='pt' else 'Color by'),bold=True,muted=False).pack(anchor='w',pady=(3,2))
        cpairs,c_to_key,key_to_c=self._color_display_maps()
        self._color_display_var=tk.StringVar(value=key_to_c.get(self.vars['scatter_color_mode'].get(),cpairs[0][0]))
        ccb=ttk.Combobox(ctrl,textvariable=self._color_display_var,values=[x[0] for x in cpairs],state='readonly',style='Dark.TCombobox')
        ccb.pack(fill='x')
        def _color_choice(e=None):
            self.vars['scatter_color_mode'].set(
                c_to_key.get(
                    self._color_display_var.get(),
                    'manual_label',
                )
            )

            # The class selector, abundance chart and class-selection action
            # must follow the categorical field represented by "Colorir por".
            self._refresh_scatter_class_selector(
                reset=True
            )

            self._draw_scatter()
        ccb.bind('<<ComboboxSelected>>',_color_choice)

        # Visual subset from information encoded in image filenames. The t-SNE
        # coordinates and stored labels remain untouched.
        filter_card=tk.Frame(ctrl,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1)
        filter_card.pack(fill='x',pady=(11,5))
        fbody=tk.Frame(filter_card,bg=C['card2'])
        fbody.pack(fill='x',padx=8,pady=8)
        tk.Label(fbody,text=('FILTRO POR NOME DA IMAGEM' if self.lang()=='pt' else 'FILTER BY IMAGE NAME'),
                 bg=C['card2'],fg=C['cyan'],font=('Sans',8,'bold')).pack(anchor='w',pady=(0,5))
        def input_pair(label, a, b, hint):
            tk.Label(fbody,text=label,bg=C['card2'],fg=C['text'],
                     font=('Sans',8,'bold')).pack(anchor='w')
            row=tk.Frame(fbody,bg=C['card2']); row.pack(fill='x',pady=(2,5))
            for v in (a,b):
                ent=self._entry(row,self.vars[v]); ent.configure(width=11)
                ent.pack(side='left',fill='x',expand=True,padx=(0,4))
            tk.Label(fbody,text=hint,bg=C['card2'],fg=C['muted'],
                     font=('Sans',7)).pack(anchor='w',pady=(0,3))
        input_pair(('Data • de / até' if self.lang()=='pt' else 'Date • from / to'),
                   'name_date_from','name_date_to','AAAA-MM-DD / YYYY-MM-DD')
        input_pair(('Hora • de / até' if self.lang()=='pt' else 'Time • from / to'),
                   'name_time_from','name_time_to','HH:MM ou HH:MM:SS')
        input_pair(('Profundidade (mm) • mínima / máxima' if self.lang()=='pt'
                    else 'Depth (mm) • min / max'),
                   'name_depth_min','name_depth_max',
                   ('Ex.: _+0001537_ = 1537 mm; imagens sem profundidade são excluídas com o filtro ativo.'
                    if self.lang()=='pt' else
                    'E.g. _+0001537_ = 1537 mm; files without depth are excluded when this filter is active.'))
        tk.Label(fbody,text=('Padrão do atributo (regex, opcional)' if self.lang()=='pt' else 'Attribute regex (optional)'),
                 bg=C['card2'],fg=C['text'],font=('Sans',8,'bold')).pack(anchor='w',pady=(2,2))
        self._entry(fbody,self.vars['name_attribute_regex']).pack(fill='x')
        tk.Label(fbody,text=('Valor do atributo / texto no nome' if self.lang()=='pt' else 'Attribute value / filename text'),
                 bg=C['card2'],fg=C['text'],font=('Sans',8,'bold')).pack(anchor='w',pady=(5,2))
        self._entry(fbody,self.vars['name_attribute_value']).pack(fill='x')
        tk.Label(fbody,text=r'Ex.: _(?P<valor>estacao[A-Z])_  |  valor: estacaoB',
                 bg=C['card2'],fg=C['muted'],font=('Sans',7),wraplength=270,justify='left').pack(anchor='w',pady=(3,6))
        buttons=tk.Frame(fbody,bg=C['card2']); buttons.pack(fill='x')
        self._btn(buttons,('Aplicar filtro' if self.lang()=='pt' else 'Apply filter'),
                  self._apply_name_filter,C['blue']).pack(side='left',fill='x',expand=True)
        self._btn(buttons,('Limpar' if self.lang()=='pt' else 'Clear'),
                  self._clear_name_filter).pack(side='left',padx=(5,0))
        self._name_filter_count_text=tk.StringVar(value=('Sem filtro ativo' if self.lang()=='pt' else 'No active filter'))
        tk.Label(fbody,textvariable=self._name_filter_count_text,bg=C['card2'],fg=C['cyan'],
                 font=('Sans',8,'bold')).pack(anchor='w',pady=(6,4))
        tk.Checkbutton(fbody,text=('Manchas por classe (estilo Overview)' if self.lang()=='pt' else 'Class density (Overview style)'),
                       variable=self.vars['name_filter_stains'],command=self._draw_scatter,
                       bg=C['card2'],fg=C['text'],selectcolor=C['blue2'],
                       activebackground=C['card2'],activeforeground='white',
                       font=('Sans',8)).pack(anchor='w',pady=(2,0))
        tk.Checkbutton(fbody,text=('Mostrar contorno das manchas' if self.lang()=='pt' else 'Show stain outlines'),
                       variable=self.vars['show_hulls'],command=self._draw_scatter,
                       bg=C['card2'],fg=C['text'],selectcolor=C['blue2'],
                       activebackground=C['card2'],activeforeground='white',
                       font=('Sans',8)).pack(anchor='w')
        tk.Label(fbody,text=('Contornos desligados por padrão; classes com menos de 20 pontos aparecem apenas como pontos.'
                             if self.lang()=='pt' else
                             'Outlines off by default; classes with fewer than 20 images appear as points.'),
                 bg=C['card2'],fg=C['muted'],font=('Sans',7),
                 wraplength=270,justify='left').pack(anchor='w',pady=(2,0))

        tk.Checkbutton(ctrl,text=self.tr('hide_relabel'),variable=self.vars['hide_relabel'],command=self._draw_scatter,
                       bg=C['card'],fg=C['text'],selectcolor=C['red'],activebackground=C['card'],activeforeground='white').pack(anchor='w',pady=3)
        self._label(ctrl,self.tr('class_filter'),bold=True,muted=False).pack(anchor='w',pady=(8,2))
        labels=self._scatter_class_labels()
        is_overview=self.vars['batch'].get()=='Overview'
        if not self.visible_classes and not is_overview:
            self.visible_classes=set(labels)
        list_mode='multiple' if (not is_overview or self.vars['overview_multi_classes'].get()) else 'browse'
        lbwrap=tk.Frame(ctrl,bg=C['entry'],highlightbackground=C['border'],highlightthickness=1); lbwrap.pack(fill='x')
        self._class_listbox=tk.Listbox(lbwrap,selectmode=list_mode,exportselection=False,height=min(9,max(4,len(labels))),
                                       bg=C['entry'],fg=C['text'],selectbackground=C['blue2'],selectforeground='white',
                                       relief='flat',highlightthickness=0,font=('Sans',8))
        sb=tk.Scrollbar(lbwrap,orient='vertical',command=self._class_listbox.yview); self._class_listbox.configure(yscrollcommand=sb.set)
        self._class_listbox.pack(side='left',fill='both',expand=True); sb.pack(side='right',fill='y')
        for i,lab in enumerate(labels):
            self._class_listbox.insert('end',lab)
            if lab in self.visible_classes: self._class_listbox.selection_set(i)
        self._class_listbox.bind('<<ListboxSelect>>',self._on_class_multiselect)
        if is_overview:
            tk.Checkbutton(
                ctrl,text=self.tr('overview_multi_classes'),
                variable=self.vars['overview_multi_classes'],
                command=self._on_overview_multi_toggle,
                bg=C['card'],fg=C['text'],selectcolor=C['blue2'],
                activebackground=C['card'],activeforeground='white'
            ).pack(anchor='w',pady=(5,1))
            tk.Label(ctrl,text=self.tr('overview_multi_tip'),wraplength=260,justify='left',
                     bg=C['card'],fg=C['muted'],font=('Sans',7)).pack(anchor='w',pady=(0,3))
        br=tk.Frame(ctrl,bg=C['card']); br.pack(fill='x',pady=(4,0))
        self._all_classes_btn=self._btn(
            br,
            self.tr('all_classes'),
            lambda:self._set_visible_classes(
                self._scatter_class_labels()
            ),
        )
        self._all_classes_btn.pack(side='left',fill='x',expand=True)
        if is_overview and not self.vars['overview_multi_classes'].get():
            self._all_classes_btn.config(state='disabled',cursor='arrow')
        self._btn(br,self.tr('no_classes'),lambda:self._set_visible_classes([])).pack(side='left',fill='x',expand=True,padx=(5,0))
        tk.Label(ctrl,text=self.tr('class_click_tip'),wraplength=260,justify='left',bg=C['card'],fg=C['muted'],font=('Sans',7)).pack(anchor='w',pady=(4,0))
        self._btn(ctrl,self.tr('select_class_points'),self.select_current_class,C['red']).pack(fill='x',pady=(5,2),ipady=3)
        if is_overview:
            tk.Label(ctrl,text=self.tr('overview_info'),wraplength=260,justify='left',bg=C['card'],fg=C['muted'],font=('Sans',8)).pack(anchor='w',pady=(8,8))
        self._label(ctrl,self.tr('selection_tools'),bold=True,muted=False).pack(anchor='w',pady=(8,5))
        r=tk.Frame(ctrl,bg=C['card']); r.pack(fill='x')
        self._btn(r,self.tr('rectangle'),lambda:self._activate_selector('rect')).pack(side='left',fill='x',expand=True)
        self._btn(r,self.tr('lasso'),lambda:self._activate_selector('lasso')).pack(side='left',fill='x',expand=True,padx=(5,0))
        self._btn(ctrl,self.tr('clear_sel'),self.clear_selection).pack(fill='x',pady=5,ipady=3)
        self._btn(ctrl,self.tr('build_sel'),self.build_selection,C['purple']).pack(fill='x',pady=(0,8),ipady=5)
        self.scatter_selected_lbl=tk.Label(ctrl,text=f"{self.tr('selected')}: {len(self.selected_gids):,}".replace(',','.'),bg=C['card'],fg=C['red'],font=('Sans',10,'bold'))
        self.scatter_selected_lbl.pack(anchor='w',pady=4)
        tk.Label(ctrl,text=self.tr('scatter_tip'),wraplength=255,justify='left',bg=C['card'],fg=C['muted'],font=('Sans',8)).pack(anchor='w',pady=(8,0))

        pcard,pbody=self._card(host,self.tr('scatter')); pcard.grid(row=0,column=1,sticky='nsew',padx=(0,4),pady=4)
        if Figure is None:
            tk.Label(pbody,text='matplotlib indisponível',bg=C['card'],fg=C['red']).pack(expand=True); return

        # Controls for the abundance chart belong only to that chart.  Keeping
        # them compact avoids stealing horizontal space from the projection.
        top=tk.Frame(pbody,bg=C['card']); top.pack(fill='x',pady=(0,4))
        barctrl=tk.Frame(top,bg=C['card']); barctrl.pack(side='right',padx=(0,8))
        tk.Label(barctrl,text=self.tr('class_axis_mode')+':',bg=C['card'],fg=C['muted'],font=('Sans',8,'bold')).pack(side='left')
        mode=ttk.Combobox(barctrl,textvariable=self.vars['class_axis_mode'],values=['N','%'],state='readonly',width=4,style='Dark.TCombobox')
        mode.pack(side='left',padx=(4,8)); mode.bind('<<ComboboxSelected>>',lambda e:self._refresh_class_abundance())
        tk.Label(barctrl,text=self.tr('axis_scale')+':',bg=C['card'],fg=C['muted'],font=('Sans',8,'bold')).pack(side='left')
        sc=ttk.Scale(barctrl,from_=45,to=220,variable=self.vars['class_axis_scale'],orient='horizontal',length=95,command=self._on_class_axis_scale)
        sc.pack(side='left',padx=(5,3))
        self._class_axis_scale_label=tk.Label(barctrl,text=f"{self.vars['class_axis_scale'].get():.0f}%",bg=C['card'],fg=C['text'],font=('Sans',8,'bold'),width=5)
        self._class_axis_scale_label.pack(side='left')
        self._btn(barctrl,self.tr('axis_auto'),lambda:self._set_class_axis_scale(100.0)).pack(side='left',padx=(3,0))

        self.scatter_fig=Figure(figsize=(10.5,7),dpi=100,facecolor=C['card'],constrained_layout=True)
        gs=self.scatter_fig.add_gridspec(1,2,width_ratios=[4.35,1.8],wspace=.10)
        self._scatter_ax=self.scatter_fig.add_subplot(gs[0,0]); self._class_ax=self.scatter_fig.add_subplot(gs[0,1])
        self._scatter_ax.set_facecolor('white'); self._class_ax.set_facecolor('white')
        plotwrap=tk.Frame(pbody,bg=C['card']); plotwrap.pack(fill='both',expand=True)
        self._scatter_canvas=FigureCanvasTkAgg(self.scatter_fig,master=plotwrap)

        # Keep the projection controls deliberately minimal: only zoom in/out.
        # Selection itself is handled by the Rectangle/Lasso tools at left.
        tbwrap=tk.Frame(pbody,bg=C['card']); tbwrap.pack(fill='x',pady=(0,3),before=plotwrap)
        self._btn(tbwrap,self.tr('zoom_in'),lambda:self._scatter_zoom(0.80)).pack(side='left',padx=(0,4))
        self._btn(tbwrap,self.tr('zoom_out'),lambda:self._scatter_zoom(1.25)).pack(side='left')
        widget=self._scatter_canvas.get_tk_widget(); widget.configure(bg=C['card'],highlightthickness=0); widget.pack(fill='both',expand=True)
        self._bind_plot_autofit(self._scatter_canvas,self.scatter_fig)

        self._scatter_artist_gids={}; self._legend_artist_classes={}
        self._scatter_pick_cid=self._scatter_canvas.mpl_connect('pick_event',self._on_pick_point)
        self._class_scroll_cid=self._scatter_canvas.mpl_connect('scroll_event',self._on_class_axis_scroll)
        self._draw_scatter()

    def _scatter_zoom(self,factor: float):
        """Zoom the projection around the current viewport centre."""
        ax=getattr(self,'_scatter_ax',None)
        canvas=getattr(self,'_scatter_canvas',None)
        if ax is None or canvas is None:return
        try:
            x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
            cx=(x0+x1)/2.0; cy=(y0+y1)/2.0
            hx=max(abs(x1-x0)*0.5*float(factor),1e-12)
            hy=max(abs(y1-y0)*0.5*float(factor),1e-12)
            ax.set_xlim(cx-hx,cx+hx); ax.set_ylim(cy-hy,cy+hy)
            canvas.draw_idle()
        except Exception:
            pass

    def _overview_multi_enabled(self):
        return self.vars['batch'].get()!='Overview' or bool(self.vars['overview_multi_classes'].get())

    def _on_overview_multi_toggle(self):
        if self.vars['batch'].get()!='Overview':
            return
        multi=bool(self.vars['overview_multi_classes'].get())
        if not multi and len(self.visible_classes)>1:
            # Keep one deterministic class when returning to single-class mode.
            keep=None
            if self._class_listbox is not None:
                sel=self._class_listbox.curselection()
                if sel:
                    keep=self._class_listbox.get(sel[-1])
            if keep is None:
                keep=sorted(self.visible_classes)[0]
            self.visible_classes={keep}
        if self._class_listbox is not None:
            self._class_listbox.configure(selectmode='multiple' if multi else 'browse')
        if self._all_classes_btn is not None and self._all_classes_btn.winfo_exists():
            self._all_classes_btn.config(state='normal' if multi else 'disabled',
                                         cursor='hand2' if multi else 'arrow')
        self._sync_class_listbox()
        self._draw_scatter()

    def _on_class_multiselect(self,event=None):
        if self._class_listbox is None: return
        selected=[self._class_listbox.get(i) for i in self._class_listbox.curselection()]
        if self.vars['batch'].get()=='Overview' and not self.vars['overview_multi_classes'].get():
            self.visible_classes={selected[-1]} if selected else set()
        else:
            self.visible_classes=set(selected)
        self._sync_class_listbox()
        self._draw_scatter()

    def _set_visible_classes(self,classes):
        classes=list(classes)
        if self.vars['batch'].get()=='Overview' and not self.vars['overview_multi_classes'].get() and len(classes)>1:
            classes=classes[:1]
        self.visible_classes=set(classes)
        self._sync_class_listbox()
        self._draw_scatter()

    def _draw_overview_class(self, ax, s, color):
        _load_scipy_helpers()
        pts=s[['x','y']].apply(pd.to_numeric,errors='coerce').dropna().to_numpy(float)
        # A hull/density is misleading for rare classes. Show actual points
        # instead, with no boundary even if the boundary option is enabled.
        if len(pts)<20 or ConvexHull is None or gaussian_filter is None:
            if len(pts):
                ax.scatter(pts[:,0],pts[:,1],s=14,color=color,alpha=.86,linewidths=0,zorder=3)
            return
        try:
            h=ConvexHull(pts); poly=pts[h.vertices]
        except Exception:
            ax.scatter(pts[:,0],pts[:,1],s=14,color=color,alpha=.86,linewidths=0,zorder=3)
            return
        # Faithful to the R IAT heat-hull idea, but rendered as a smooth RGBA
        # density stain instead of contour bands. No internal contour lines.
        if MplPath is None or to_rgba is None:
            ax.scatter(pts[:,0],pts[:,1],s=14,color=color,alpha=.86,linewidths=0,zorder=3)
            return
        if gaussian_filter is not None and len(pts)>=20 and MplPath is not None and to_rgba is not None:
            xmin,xmax=float(np.min(pts[:,0])),float(np.max(pts[:,0])); ymin,ymax=float(np.min(pts[:,1])),float(np.max(pts[:,1]))
            if xmax>xmin and ymax>ymin:
                H,xe,ye=np.histogram2d(pts[:,0],pts[:,1],bins=70,range=[[xmin,xmax],[ymin,ymax]])
                H=gaussian_filter(H,sigma=2.1)
                xc=(xe[:-1]+xe[1:])/2; yc=(ye[:-1]+ye[1:])/2
                X,Y=np.meshgrid(xc,yc,indexing='xy')
                mask=MplPath(poly).contains_points(np.c_[X.ravel(),Y.ravel()]).reshape(X.shape)
                Z=H.T
                zmax=float(np.nanmax(Z)) if Z.size else 0.0
                if zmax>0:
                    norm=np.clip((Z/zmax-0.05)/0.95,0,1)
                    rgba=np.zeros((Z.shape[0],Z.shape[1],4),dtype=float)
                    base=to_rgba(color)
                    rgba[...,0]=base[0]; rgba[...,1]=base[1]; rgba[...,2]=base[2]
                    # Keep the stain translucent, but dark enough to read on the
                    # white overview background without becoming visually heavy.
                    rgba[...,3]=np.where(mask,0.035+0.30*np.power(norm,1.28),0.0)
                    rgba[...,3]=np.where(norm>0,rgba[...,3],0.0)
                    ax.imshow(rgba,origin='lower',extent=[xmin,xmax,ymin,ymax],interpolation='bilinear',aspect='auto',zorder=1)
        # Optional boundary, disabled by default and never drawn for rare classes.
        if self.vars['show_hulls'].get():
            ax.plot(np.r_[poly[:,0],poly[0,0]],np.r_[poly[:,1],poly[0,1]],color=color,alpha=.68,lw=.55,zorder=2)

    def _set_class_axis_scale(self, value):
        try:value=float(value)
        except Exception:value=100.0
        value=max(45.0,min(220.0,value)); self.vars['class_axis_scale'].set(value)
        if self._class_axis_scale_label is not None and self._class_axis_scale_label.winfo_exists():
            self._class_axis_scale_label.config(text=f'{value:.0f}%')
        self._refresh_class_abundance()

    def _on_class_axis_scale(self, value=None):
        try:v=float(value if value is not None else self.vars['class_axis_scale'].get())
        except Exception:v=100.0
        if self._class_axis_scale_label is not None and self._class_axis_scale_label.winfo_exists():
            self._class_axis_scale_label.config(text=f'{v:.0f}%')
        self._refresh_class_abundance()

    def _on_class_axis_scroll(self, event):
        if self._class_ax is None or getattr(event,'inaxes',None) is not self._class_ax:return
        cur=float(self.vars['class_axis_scale'].get() or 100.0)
        # wheel up = tighter X range; wheel down = wider X range
        cur*=0.90 if getattr(event,'button',None)=='up' else 1.10
        self._set_class_axis_scale(cur)

    def _refresh_class_abundance(self):
        if self._class_ax is None or self._scatter_canvas is None:return
        self._draw_class_abundance(self._name_filtered_source(self._analysis_plot_source())
                                   if self._name_filter_spec.enabled else self.current_df); self._scatter_canvas.draw_idle()

    def _sync_class_listbox(self):
        if self._class_listbox is None:return
        self._class_listbox.selection_clear(0,'end')
        for i in range(self._class_listbox.size()):
            if self._class_listbox.get(i) in self.visible_classes:self._class_listbox.selection_set(i)

    def _draw_class_abundance(self, source_df):
        ax=self._class_ax
        ax.clear()
        ax.set_facecolor('white')
        self._legend_artist_classes={}

        if source_df is None or source_df.empty:
            ax.axis('off')
            return

        class_col=self._scatter_class_column(
            source_df
        )

        if class_col not in source_df.columns:
            ax.axis('off')
            return

        class_values=(
            source_df[class_col]
            .fillna('')
            .astype(str)
            .str.strip()
        )

        valid=(
            class_values.str.len()>0
        )

        if not bool(valid.any()):
            ax.axis('off')
            return

        counts=(
            class_values.loc[valid]
            .value_counts()
            .sort_values(
                ascending=False
            )
        )

        labs=counts.index.tolist()
        raw=counts.values.astype(float)

        # ``counts`` already contains every non-empty class represented in
        # source_df.  Reuse it instead of scanning/converting the full column a
        # second time just to rebuild the same sorted class list.
        full_labs=sorted(str(lab) for lab in labs)

        named_class_color = class_col in {'manual_label', 'source_label', 'preds1', 'pred'}
        cmap={
            lab:(get_class_color(lab) if named_class_color else get_color(i))
            for i,lab in enumerate(full_labs)
        }
        mode=self.vars['class_axis_mode'].get() or 'N'
        if mode=='%':
            total=max(float(raw.sum()),1.0); vals=raw/total*100.0
        else: vals=raw
        y=np.arange(len(labs)); colors=[cmap.get(l,get_color(0)) for l in labs]
        alphas=[.90 if l in self.visible_classes else .16 for l in labs]
        bars=ax.barh(y,vals,color=colors,height=.72,edgecolor='#64748b',linewidth=.25)
        for bar,lab,a in zip(bars,labs,alphas):
            bar.set_alpha(a); bar.set_picker(True); self._legend_artist_classes[bar]=lab
        # Y labels sit outside the white plotting rectangle, over the dark
        # application card. Use high-contrast light text and a readable size.
        label_size=max(7,9-min(len(labs)//14,2))
        ax.set_yticks(y); texts=ax.set_yticklabels(labs,fontsize=label_size,color='#f8fafc'); ax.invert_yaxis()
        for t,lab in zip(texts,labs):
            t.set_fontweight('semibold'); t.set_picker(True); self._legend_artist_classes[t]=lab
        ax.xaxis.tick_top(); ax.xaxis.set_label_position('top')
        xlabel=(self.tr('axis_percent')+' (%)') if mode=='%' else self.tr('class_abundance')
        ax.set_xlabel(xlabel,color='#475569',fontsize=8,labelpad=5)
        if mode=='%' and FuncFormatter is not None: ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}%'))
        elif mode!='%' and FuncFormatter is not None: ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'.replace(',','.')))
        scale=max(45.0,min(220.0,float(self.vars['class_axis_scale'].get() or 100.0)))/100.0
        vmax=float(np.nanmax(vals)) if len(vals) else 1.0
        ax.set_xlim(0,max(vmax*1.08*scale,1.0))
        ax.tick_params(axis='x',colors='#475569',labelsize=7)
        ax.tick_params(axis='y',length=0,pad=4,colors='#f8fafc',labelsize=label_size)
        for side,spine in ax.spines.items():
            spine.set_color('#cbd5e1'); spine.set_linewidth(.5); spine.set_visible(side=='top')
        ax.grid(axis='x',color='#cbd5e1',alpha=.55,lw=.45)

    def _draw_scatter(self):
        if not hasattr(self,'_scatter_ax') or self._scatter_ax is None: return
        df=self._scatter_filtered(); self._scatter_points_df=df.reset_index(drop=True)
        ax=self._scatter_ax
        # Remove colorbar axes left by a previous continuous-color rendering.
        try:
            for extra_ax in list(self.scatter_fig.axes):
                if extra_ax not in {self._scatter_ax,self._class_ax}:
                    self.scatter_fig.delaxes(extra_ax)
        except Exception:pass
        ax.clear(); ax.set_facecolor('white'); self._scatter_artist_gids={}
        if df.empty:
            if self.vars['batch'].get()=='Overview' and not self.visible_classes and self.current_df is not None and not self.current_df.empty:
                empty_text=self.tr('select_class_prompt')
            else:
                empty_text='Sem dados' if self.lang()=='pt' else 'No data'
            ax.text(.5,.5,empty_text,color='#64748b',ha='center',va='center',transform=ax.transAxes)
        else:
            mode=str(self.vars['scatter_color_mode'].get() or 'manual_label')
            pred_col='preds1' if 'preds1' in df.columns else ('pred' if 'pred' in df.columns else None)
            col_map={
                'manual_label':'manual_label','prediction':pred_col,'source_label':'source_label',
                'hdbscan_cluster':'hdbscan_cluster','hdbscan_outlier':'hdbscan_outlier',
                'knn_agreement':'knn_agreement','review_priority':'review_priority',
                'confs1':'confs1','var_norm':'var_norm',
                'source_prediction_mismatch':'source_prediction_mismatch',
                'manual_prediction_mismatch':'manual_prediction_mismatch',
            }
            col=col_map.get(mode,'manual_label')
            if col not in df.columns or col is None:
                col='manual_label'; mode='manual_label'

            continuous=col in {'hdbscan_outlier','knn_agreement','review_priority','confs1','var_norm'}
            original_overview=(self.vars['batch'].get()=='Overview' and not self._analysis_projection_active and
                               mode=='manual_label' and (not self._name_filter_spec.enabled or
                                                         bool(self.vars['name_filter_stains'].get())))
            filter_stains=(self._name_filter_spec.enabled and
                           bool(self.vars['name_filter_stains'].get()) and not continuous and
                           col in {'manual_label','source_label','preds1','pred'})
            if original_overview or filter_stains:
                # Retain the established t-SNE coordinates: filtering does not
                # calculate a second projection or change any saved labels.
                cats=df[col].fillna('').astype(str)
                grouped=cats.groupby(cats,sort=False).groups
                for lab in sorted(grouped):
                    self._draw_overview_class(ax,df.loc[grouped[lab]],get_class_color(lab))
            elif continuous:
                vals=pd.to_numeric(df[col],errors='coerce').to_numpy(float)
                finite=np.isfinite(vals)
                if finite.any():
                    plotvals=np.where(finite,vals,np.nanmedian(vals[finite]))
                    artist=ax.scatter(df['x'],df['y'],s=12,c=plotvals,cmap='viridis',alpha=.84,linewidths=0,picker=5,zorder=3)
                    self._scatter_artist_gids[artist]=df['gid'].astype(str).tolist()
                    try:
                        cb=self.scatter_fig.colorbar(artist,ax=ax,fraction=.035,pad=.02)
                        cb.ax.tick_params(labelsize=7,colors='#334155')
                        cb.set_label(col,fontsize=8,color='#334155')
                    except Exception:pass
                else:
                    artist=ax.scatter(df['x'],df['y'],s=10,c='#64748b',alpha=.75,linewidths=0,picker=5,zorder=3)
                    self._scatter_artist_gids[artist]=df['gid'].astype(str).tolist()
            else:
                cats=df[col].fillna('').astype(str)
                values=sorted(cats.unique().tolist())
                color_map={lab:(get_class_color(lab) if col in {'manual_label', 'source_label', 'preds1', 'pred'} else get_color(i)) for i,lab in enumerate(values)}
                grouped=cats.groupby(cats,sort=False).groups
                for lab in values:
                    ss=df.loc[grouped[lab]]
                    label=('Noise' if col=='hdbscan_cluster' and lab=='-1' else lab)
                    artist=ax.scatter(ss['x'],ss['y'],s=11,c=color_map.get(lab,get_color(0)),alpha=.84,label=label,linewidths=0,picker=5,zorder=3)
                    self._scatter_artist_gids[artist]=ss['gid'].astype(str).tolist()
                if len(values)<=18:
                    try:ax.legend(loc='best',fontsize=6,framealpha=.70)
                    except Exception:pass
            try:
                proj_name='Original t-SNE' if not self._analysis_projection_active else self._analysis_projection_method.upper()
                ax.set_title(f'{proj_name} • {col}',fontsize=9,color='#334155')
            except Exception:pass
            # On white, a thin red outline is visible without obscuring points.
            if self._selection_outline is not None and len(self._selection_outline)>=2:
                poly=np.asarray(self._selection_outline,float)
                ax.plot(poly[:,0],poly[:,1],color='#dc2626',lw=.8,alpha=.95,zorder=10)
        for spine in ax.spines.values(): spine.set_color('#94a3b8'); spine.set_linewidth(.6)
        ax.tick_params(colors='#334155',labelsize=8); ax.set_xlabel('x',color='#334155'); ax.set_ylabel('y',color='#334155')
        ax.grid(color='#cbd5e1',alpha=.48,lw=.45); ax.margins(.03)
        self._draw_class_abundance(self._name_filtered_source(self._analysis_plot_source())
                                   if self._name_filter_spec.enabled else self.current_df)
        self._scatter_canvas.draw_idle()
        if hasattr(self,'scatter_selected_lbl'):
            self.scatter_selected_lbl.config(text=f"{self.tr('selected')}: {len(self.selected_gids):,}".replace(',','.'))

    def _on_pick_point(self,event):
        artist=getattr(event,'artist',None)
        if artist in self._legend_artist_classes:
            lab=self._legend_artist_classes[artist]
            dbl=bool(getattr(getattr(event,'mouseevent',None),'dblclick',False))
            if self.vars['batch'].get()=='Overview' and not self.vars['overview_multi_classes'].get():
                # Default Overview behavior: one class at a time.
                self.visible_classes={lab}
            elif dbl:
                self.visible_classes={lab}
            else:
                if lab in self.visible_classes:self.visible_classes.discard(lab)
                else:self.visible_classes.add(lab)
            self._sync_class_listbox(); self._draw_scatter(); return
        gids=self._scatter_artist_gids.get(artist,[])
        if not gids: return
        inds=list(getattr(event,'ind',[]) or [])
        if not inds: return
        for i in inds:
            if 0<=i<len(gids):
                gid=gids[i]
                if gid in self.selected_gids:self.selected_gids.discard(gid)
                else:self.selected_gids.add(gid)
        self._draw_scatter()

    def select_current_class(self):
        df=self.current_df
        if df is None or df.empty:
            return
        if self.vars['batch'].get()=='Overview' and not self.visible_classes:
            # A blank Overview must never turn this button into an accidental
            # "select the whole project" action on multi-million-image datasets.
            self._log(self.tr('select_class_prompt'),'warn')
            return
        class_col=self._scatter_class_column(
            df
        )

        if class_col not in df.columns:
            return

        class_values=(
            df[class_col]
            .fillna('')
            .astype(str)
            .str.strip()
        )

        available=set(
            class_values[
                class_values.str.len()>0
            ].unique().tolist()
        )

        classes=(
            self.visible_classes
            or available
        )

        self.selected_gids=set(
            df.loc[
                class_values.isin(classes),
                'gid',
            ].astype(str)
        )
        self._selection_outline=None; self._draw_scatter()

    def _activate_selector(self,kind):
        if self._scatter_ax is None or self._scatter_canvas is None:return
        if self._selection_tool:
            try:self._selection_tool.set_active(False);self._selection_tool.disconnect_events()
            except Exception:pass
        props=dict(color='#dc2626',linewidth=.9,alpha=.9)
        if kind=='lasso' and LassoSelector:
            self._selection_tool=LassoSelector(self._scatter_ax,onselect=self._on_lasso,props=props)
        elif RectangleSelector:
            self._selection_tool=RectangleSelector(self._scatter_ax,self._on_rect,useblit=True,button=[1],interactive=False,props=dict(edgecolor='#dc2626',facecolor='none',linewidth=.9))

    def _on_lasso(self,verts):
        if MplPath is None or self._scatter_points_df.empty:return
        pts=self._scatter_points_df[['x','y']].to_numpy(float); mask=MplPath(verts).contains_points(pts)
        self.selected_gids=set(self._scatter_points_df.loc[mask,'gid'].astype(str)); self._selection_outline=list(verts)+([verts[0]] if verts else []); self._draw_scatter()

    def _on_rect(self,eclick,erelease):
        if self._scatter_points_df.empty or eclick.xdata is None or erelease.xdata is None or eclick.ydata is None or erelease.ydata is None:return
        x0,x1=sorted([eclick.xdata,erelease.xdata]); y0,y1=sorted([eclick.ydata,erelease.ydata]); df=self._scatter_points_df
        mask=df['x'].between(x0,x1)&df['y'].between(y0,y1); self.selected_gids=set(df.loc[mask,'gid'].astype(str)); self._selection_outline=[(x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)]; self._draw_scatter()

    def clear_selection(self):
        self.selected_gids.clear(); self.checked_gids.clear(); self.allb_gids=[]; self._gallery_selection_df=None; self._gallery_anchor_gid=None; self._gallery_discarded_gids.clear(); self._clear_gallery_similarity_state(clear_reference=True,refresh=False); self._selection_outline=None
        if self.current_tab=='scatter': self._draw_scatter()

    def build_selection(self):
        if not self.selected_gids: self._log(self.tr('no_selection'),'warn');return
        selected=set(self.selected_gids)
        self._gallery_discarded_gids.clear()
        self._clear_gallery_similarity_state(clear_reference=True,refresh=False)
        self.allb_gids=list(selected); self.page_idx=0
        # Materialize the selected subset once. Re-filtering the complete
        # Overview for every Gallery page becomes expensive when the project
        # contains millions of points.
        if not self.current_df.empty and 'gid' in self.current_df.columns:
            self._gallery_selection_df=self.current_df.loc[
                self.current_df['gid'].astype(str).isin(selected)
            ].copy()
        else:
            self._gallery_selection_df=None
        self._log(f'Build selection: {len(self.allb_gids):,} imagens'.replace(',','.'),'success'); self.show_tab('gallery')

    # -------- isolated temporal / spatial workspace --------
    def _build_temporal(self):
        view=TemporalSpatialPanel(self.content,self)
        view.pack(fill='both',expand=True)

    # -------- gallery --------


    # -------- clustering --------
    def _active_cluster_gids(self):
        return set(self.selected_gids) if self.selected_gids else set(self.checked_gids)

    def _cluster_selection_rows(self, gids=None, project=None, timing=None, cancel_check=None):
        """Resolve exactly the requested clustering scope without touching Tk state.

        Canonical ``<batch>::<id>`` identifiers read only their referenced batches.
        Overview is a compatibility fallback for unresolved/legacy identifiers.
        No identifier is intentionally removed and duplicate rows are suppressed.
        """
        project = project or self.project
        if project is None:
            return pd.DataFrame()
        requested = [str(g) for g in (gids if gids is not None else self._active_cluster_gids()) if str(g)]
        if not requested:
            return pd.DataFrame()
        requested = list(dict.fromkeys(requested))
        unresolved = set(requested)
        parts, seen = [], set()
        timing = timing if timing is not None else {}
        timing.setdefault('batch_reads', 0)
        timing.setdefault('batch_read_s', 0.0)
        timing.setdefault('overview_s', 0.0)
        timing.setdefault('overview_used', False)

        by_batch = {}
        for gid in requested:
            if '::' in gid:
                bn = gid.split('::', 1)[0]
                by_batch.setdefault(bn, set()).add(gid)

        for bn, bgids in by_batch.items():
            if cancel_check is not None and cancel_check():
                timing['cancelled'] = True
                return pd.DataFrame()
            t0 = time.perf_counter()
            try:
                df = project.load_batch(bn)
            except Exception:
                timing['batch_read_s'] += time.perf_counter() - t0
                continue
            timing['batch_read_s'] += time.perf_counter() - t0
            timing['batch_reads'] += 1
            if df is None or df.empty or 'gid' not in df.columns:
                continue
            mask = df['gid'].astype(str).isin(bgids)
            if not mask.any():
                continue
            part = df.loc[mask].copy()
            part = part[~part['gid'].astype(str).isin(seen)].copy()
            if part.empty:
                continue
            ids = part['gid'].astype(str).tolist()
            seen.update(ids)
            unresolved.difference_update(ids)
            parts.append(part)

        if unresolved:
            if cancel_check is not None and cancel_check():
                timing['cancelled'] = True
                return pd.DataFrame()
            t0 = time.perf_counter()
            try:
                ov = project.load_overview()
            except Exception:
                ov = pd.DataFrame()
            timing['overview_s'] += time.perf_counter() - t0
            timing['overview_used'] = True
            if ov is not None and not ov.empty and 'gid' in ov.columns:
                mask = ov['gid'].astype(str).isin(unresolved)
                if mask.any():
                    part = ov.loc[mask].copy()
                    part = part[~part['gid'].astype(str).isin(seen)].copy()
                    if not part.empty:
                        ids = part['gid'].astype(str).tolist()
                        seen.update(ids)
                        unresolved.difference_update(ids)
                        parts.append(part)

        timing['missing_gids'] = len(unresolved)
        timing['requested_gids'] = len(requested)
        if not parts:
            timing['resolved_gids'] = 0
            return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True, sort=False, copy=False)
        out = out.drop_duplicates(subset=['gid'], keep='first')
        order = {gid: idx for idx, gid in enumerate(requested)}
        out['_iat_scope_order'] = out['gid'].astype(str).map(order)
        out = out.sort_values('_iat_scope_order', kind='stable').drop(columns=['_iat_scope_order']).reset_index(drop=True)
        timing['resolved_gids'] = len(out)
        return out

    def _cluster_artifact_matches_rows(self,path,rows):
        """Return identifier mode ('gid'/'names') when artifact matches this scope."""
        if path is None or rows is None or rows.empty:
            return None
        try:
            p=Path(path)
            if not p.exists():return None
            ids=[str(x) for x in read_artifact_names(p)]
        except Exception:
            return None
        if len(ids)!=len(rows):
            return None
        aid=set(ids)
        if 'gid' in rows.columns:
            gids=rows['gid'].astype(str).tolist()
            if len(set(gids))==len(gids) and aid==set(gids):
                return 'gid'
        if 'names' in rows.columns:
            names=rows['names'].astype(str).tolist()
            # Class artifacts historically use image names. Only accept that
            # representation when names are unique, otherwise gid is required.
            if len(set(names))==len(names) and aid==set(names):
                return 'names'
        return None

    def _cluster_artifact_for_rows(self, rows, project=None, current_path=None):
        """Find an existing artifact matching the exact prepared scope."""
        project = project or self.project
        if rows is None or rows.empty or project is None:
            return None, None
        candidates = []
        current_path = self.current_cluster_path if current_path is None else current_path
        if current_path:
            candidates.append(Path(current_path))
        selection_path = project.clusters_dir / '_selection.npz'
        if selection_path.exists():
            candidates.append(selection_path)
        if 'manual_label' in rows.columns:
            classes = rows['manual_label'].dropna().astype(str).unique().tolist()
            if len(classes) == 1:
                candidates.append(project.clusters_dir / f'{sanitize_filename(classes[0])}.npz')
        seen = set()
        for p in candidates:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            mode = self._cluster_artifact_matches_rows(p, rows)
            if mode:
                return p, mode
        return None, None

    def _build_clustering(self):
        """Render controls immediately; selection/artifact reads happen in background."""
        t0 = time.perf_counter()
        host=tk.Frame(self.content,bg=C['bg']);host.pack(fill='both',expand=True);host.columnconfigure(1,weight=1);host.rowconfigure(0,weight=1)
        card,ctrl_host=self._card(host,self.tr('cluster_config'));card.grid(row=0,column=0,sticky='ns',padx=(4,7),pady=4);card.config(width=340);card.grid_propagate(False)
        cscroll=ScrollFrame(ctrl_host,C['card']);cscroll.pack(fill='both',expand=True);ctrl=cscroll.inner
        preparing='Preparando seleção…' if self.lang()=='pt' else 'Preparing selection…'
        self._label(ctrl,self.tr('cluster_sel_count'),bold=True,muted=False).pack(anchor='w')
        self.cluster_scope_count_lbl=tk.Label(ctrl,text='—',bg=C['card'],fg=C['cyan'],font=('Sans',16,'bold'));self.cluster_scope_count_lbl.pack(anchor='w')
        self._label(ctrl,self.tr('cluster_sel_classes')).pack(anchor='w',pady=(4,1))
        self.cluster_scope_classes_lbl=tk.Label(ctrl,text=preparing,wraplength=285,justify='left',bg=C['card'],fg=C['muted'],font=('Sans',8));self.cluster_scope_classes_lbl.pack(anchor='w')
        r=tk.Frame(ctrl,bg=C['card']);r.pack(fill='x',pady=8)
        l=tk.Frame(r,bg=C['card']);l.pack(side='left',fill='x',expand=True);self._label(l,self.tr('metric')).pack(anchor='w');ttk.Combobox(l,textvariable=self.vars['cluster_metric'],values=['cosine','euclidean'],state='readonly',style='Dark.TCombobox',width=10).pack(fill='x')
        l=tk.Frame(r,bg=C['card']);l.pack(side='left',fill='x',expand=True,padx=(6,0));self._label(l,self.tr('linkage')).pack(anchor='w');ttk.Combobox(l,textvariable=self.vars['cluster_linkage'],values=['average','complete','single','ward'],state='readonly',style='Dark.TCombobox',width=10).pack(fill='x')
        self._label(ctrl,self.tr('cut')).pack(anchor='w',pady=(6,2));scale=tk.Scale(ctrl,variable=self.vars['cluster_cut'],from_=0,to=1,resolution=.01,orient='horizontal',bg=C['card'],fg=C['text'],troughcolor=C['border'],highlightthickness=0,command=lambda v:self._schedule_cluster_view());scale.pack(fill='x')
        tk.Label(ctrl,text=self.tr('use_selected_scope'),wraplength=290,justify='left',bg=C['card'],fg=C['muted'],font=('Sans',8)).pack(anchor='w',pady=(5,8))
        self.cluster_progress_vars={'generate':tk.DoubleVar(value=0),'dirty':tk.DoubleVar(value=0),'relabel':tk.DoubleVar(value=0)}
        self.cluster_status_vars={'generate':tk.StringVar(value=self.tr('cluster_ready')),'dirty':tk.StringVar(value=self.tr('cluster_ready')),'relabel':tk.StringVar(value=self.tr('cluster_ready'))}
        def action(text,cmd,key,color):
            self._btn(ctrl,text,cmd,color).pack(fill='x',ipady=5,pady=(3,1))
            ttk.Progressbar(ctrl,maximum=100,variable=self.cluster_progress_vars[key],style='Blue.Horizontal.TProgressbar').pack(fill='x',pady=(0,1))
            tk.Label(ctrl,textvariable=self.cluster_status_vars[key],bg=C['card'],fg=C['muted'],font=('Sans',7),wraplength=285,justify='left').pack(anchor='w',pady=(0,3))
        action(self.tr('generate_cluster'),self.generate_cluster,'generate',C['purple'])
        action(self.tr('update_dirty'),self.update_dirty_clusters,'dirty',C['card2'])
        action(self.tr('relabel_group'),self.relabel_by_group,'relabel',C['amber'])
        self._label(ctrl,self.tr('cluster_desc_title'),bold=True,muted=False).pack(anchor='w',pady=(7,3))
        for key in ('cluster_desc_generate','cluster_desc_dirty','cluster_desc_relabel'):
            tk.Label(ctrl,text=self.tr(key),wraplength=290,justify='left',bg=C['card'],fg=C['muted'],font=('Sans',7)).pack(anchor='w',pady=1)
        pcard,pbody=self._card(host,self.tr('cluster_view'));pcard.grid(row=0,column=1,sticky='nsew',padx=(0,4),pady=4);pbody.rowconfigure(0,weight=1);pbody.columnconfigure(0,weight=1)
        if Figure is not None:
            self.cluster_fig=Figure(figsize=(7,5),dpi=100,facecolor=C['card']);self.cluster_ax=self.cluster_fig.add_subplot(111);self.cluster_fig.subplots_adjust(left=.012,right=.988,bottom=.015,top=.94);self._dendro_axes_style(self.cluster_ax);self._dendro_canvas=FigureCanvasTkAgg(self.cluster_fig,master=pbody);self._dendro_canvas.get_tk_widget().pack(fill='both',expand=True);self._dendro_canvas.get_tk_widget().configure(height=455);self._bind_plot_autofit(self._dendro_canvas,self.cluster_fig)
        self.cluster_stats=tk.Label(pbody,text=preparing,justify='left',anchor='w',bg=C['card'],fg=C['muted'],font=('Monospace',8),wraplength=900);self.cluster_stats.pack(fill='x',pady=(6,0))
        elapsed=time.perf_counter()-t0
        if elapsed>0.03:self._log(f'Diagnóstico Clustering: construção inicial={elapsed:.3f}s.','muted')
        self.after_idle(self.load_cluster_view)

    def generate_cluster(self):
        if not self.project:return
        rows=self._cluster_selection_rows()
        if len(rows)<2:
            if hasattr(self,'cluster_status_vars'):self.cluster_status_vars['generate'].set(self.tr('no_selection'))
            self._log(self.tr('no_selection'),'warn');return
        metric=self.vars['cluster_metric'].get(); method=self.vars['cluster_linkage'].get()
        if method=='ward' and metric!='euclidean':
            metric='euclidean'; self.vars['cluster_metric'].set(metric); self._log('Ward requer Euclidean; métrica ajustada automaticamente.','warn')
        def work(progress):
            self.project.save_dirty(); return generate_selection_cluster_artifact(self.project,rows,metric=metric,method=method,progress=progress)
        def done(path):self.current_cluster_path=path;self._log(f'{self.tr("cluster_done")}: {path.name}','success');self.load_cluster_view()
        self._run_operation(self.tr('cluster_running'),work,done,local_progress=self.cluster_progress_vars.get('generate'),local_status=self.cluster_status_vars.get('generate'))

    def update_dirty_clusters(self):
        if not self.project:return
        classes=sorted(self.project.dirty_classes)
        metric=self.vars['cluster_metric'].get(); method=self.vars['cluster_linkage'].get()
        if method=='ward' and metric!='euclidean': metric='euclidean'; self.vars['cluster_metric'].set(metric)

        def work(progress):
            # Reconcile instead of blindly clustering every dirty label. Old
            # classes may legitimately have zero images after a full relabel.
            # discover_stale also repairs a project reopened after the in-memory
            # dirty-class list was lost.
            return reconcile_class_cluster_artifacts(
                self.project, class_labels=classes, metric=metric, method=method,
                progress=progress, discover_stale=True
            )

        def done(res):
            ngen=len(res.get('generated',[])); nrm=len(res.get('removed',[]))
            nsmall=len(res.get('skipped_small',[])); errs=res.get('errors',[])
            if errs:
                sample='; '.join(f'{c}: {e}' for c,e in errs[:3])
                msg=(f'{ngen} clusters atualizados; {nrm} obsoletos removidos; '
                     f'{len(errs)} falha(s): {sample}')
                if hasattr(self,'cluster_status_vars'):self.cluster_status_vars['dirty'].set(msg)
                self._log(msg,'warn')
            else:
                msg=(f'{ngen} clusters atualizados; {nrm} obsoletos removidos'
                     + (f'; {nsmall} classes com <2 imagens' if nsmall else ''))
                if hasattr(self,'cluster_status_vars'):self.cluster_status_vars['dirty'].set(msg)
                self._log(msg,'success')
            self.load_cluster_view()

        self._run_operation(self.tr('cluster_running'),work,done,
                            local_progress=self.cluster_progress_vars.get('dirty'),
                            local_status=self.cluster_status_vars.get('dirty'))

    def _selection_cluster_path(self):
        if not self.project:return None
        p=self.project.clusters_dir/'_selection.npz'; return p if p.exists() else None

    def _schedule_cluster_view(self,delay_ms=180):
        """Debounce dendrogram refreshes so dragging cutoff stays responsive."""
        pending=getattr(self,'_cluster_cut_after',None)
        if pending is not None:
            try:self.after_cancel(pending)
            except Exception:pass
        try:
            # Only the cut changed: reuse the already resolved selection/artifact.
            self._cluster_cut_after=self.after(int(delay_ms),lambda:self.load_cluster_view(reuse_scope=True))
        except Exception:
            self._cluster_cut_after=None

    _cluster_artifact_cache={}

    @staticmethod
    def _cached_cluster_artifact(path):
        """Read a cluster artifact once per (path, mtime); dragging the cut reuses it."""
        try:mtime=os.stat(path).st_mtime_ns
        except OSError:mtime=None
        key=(str(path),mtime)
        cache=IATWorkspace._cluster_artifact_cache
        art=cache.get(key)
        if art is None:
            art=read_cluster_artifact(path)
            if mtime is not None:
                if len(cache)>=4:cache.pop(next(iter(cache)))
                cache[key]=art
        return art

    @staticmethod
    def _dendro_axes_style(ax):
        """Dendrogram axes carry no meaningful tick numbers: hide them."""
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)

    @staticmethod
    def _prepare_cluster_view_payload(path,cut):
        """Load and reduce a dendrogram outside Tk's main thread.

        All images remain in the clustering calculation.  For large selections
        only the *drawing* is truncated, because thousands of individual leaves
        are not readable and can stall Matplotlib/Tk.
        """
        _load_scipy_helpers()
        if dendrogram is None:
            raise RuntimeError('SciPy indisponível para o dendrograma.')
        art=IATWorkspace._cached_cluster_artifact(path)
        Z=art['linkage'];N=len(art['names'])
        dist_max=float(art.get('dist_max',1.0))
        kwargs={
            'no_plot':True,
            'color_threshold':float(cut)*dist_max,
            'above_threshold_color':'#7f93aa',
        }
        truncated=N>=1000
        if truncated:
            kwargs.update(truncate_mode='lastp',p=min(120,max(N,2)),show_leaf_counts=True)
        tree=dendrogram(Z,**kwargs)
        groups=cluster_groups(art,float(cut))
        return {
            'icoord':tree.get('icoord',[]),
            'dcoord':tree.get('dcoord',[]),
            'color_list':tree.get('color_list',[]),
            'ivl':tree.get('ivl',[]),
            'truncated':truncated,
            'n':N,
            'dist_max':dist_max,
            'groups':int(len(np.unique(groups))),
            'mode':art.get('mode',''),
        }

    def _apply_cluster_view_status(self,token,text):
        if token!=getattr(self,'_cluster_view_token',None) or self.current_tab!='clustering':return
        try:self.cluster_scope_classes_lbl.config(text=text);self.cluster_stats.config(text=text)
        except Exception:pass

    def _log_cluster_timings(self,timings,render_t0=None):
        timings=dict(timings or {})
        if render_t0 is not None:timings['render_dispatch_s']=time.perf_counter()-render_t0
        parts=[]
        for key,label in (('scope_s','seleção'),('batch_read_s','batches'),('overview_s','overview'),('artifact_lookup_s','artifact'),('dendrogram_prepare_s','dendrograma'),('render_dispatch_s','render'),('total_s','total')):
            if key in timings:parts.append(f'{label}={1000*float(timings[key]):.0f}ms')
        if timings.get('overview_used'):parts.append('overview=usado')
        if 'batch_reads' in timings:parts.append(f"batch_reads={int(timings['batch_reads'])}")
        if timings.get('missing_gids'):parts.append(f"gids_ausentes={int(timings['missing_gids'])}")
        if parts:self._log('Diagnóstico Clustering: '+', '.join(parts),'muted')

    def _apply_cluster_view_result(self,token,payload,error=None,timings=None):
        if token!=getattr(self,'_cluster_view_token',None) or self.current_tab!='clustering':return
        if not hasattr(self,'cluster_ax') or self.cluster_ax is None or self._dendro_canvas is None:return
        try:
            if not self._dendro_canvas.get_tk_widget().winfo_exists():return
        except Exception:return
        render_t0=time.perf_counter();ax=self.cluster_ax;ax.clear();ax.set_facecolor(C['log']);self._dendro_axes_style(ax)
        timings=timings or (payload.get('timings',{}) if isinstance(payload,dict) else {})
        if error is not None:
            ax.text(.5,.5,str(error),color=C['red'],ha='center',va='center',transform=ax.transAxes)
            try:self.cluster_stats.config(text='')
            except Exception:pass
            self._dendro_canvas.draw_idle();self._log(f'Falha ao carregar visualização do clustering: {error}','error');return
        n=int(payload.get('n',0) or 0);classes=int(payload.get('classes',0) or 0)
        try:self.cluster_scope_count_lbl.config(text=f'{n:,}'.replace(',','.'));self.cluster_scope_classes_lbl.config(text=f'{classes} classe(s)')
        except Exception:pass
        if payload.get('missing'):
            msg=('Nenhum clustering salvo para esta seleção.\nClique em “Gerar / atualizar cluster”.' if payload.get('has_selection') and self.lang()=='pt' else ('No clustering saved for this selection.\nClick “Generate / update cluster”.' if payload.get('has_selection') else self.tr('no_artifact')))
            ax.text(.5,.5,msg,ha='center',va='center',color=C['muted'],transform=ax.transAxes);self.cluster_stats.config(text='');self._dendro_canvas.draw_idle();self._log_cluster_timings(timings,render_t0);return
        segments=[list(zip(xs,ys)) for xs,ys in zip(payload['icoord'],payload['dcoord'])]
        if segments:
            colors=payload.get('color_list') or [C['cyan']]*len(segments)
            try:rgba=[to_rgba(c) for c in colors]
            except Exception:rgba=[to_rgba(C['cyan'])]*len(segments)
            ax.add_collection(LineCollection(segments,colors=rgba,linewidths=1.0))
            xs=[x for seg in segments for x,_ in seg];ys=[y for seg in segments for _,y in seg]
            if xs and ys:ax.set_xlim(min(xs),max(xs));ymax=max(ys);ax.set_ylim(0,ymax*1.03 if ymax>0 else 1)
        ax.axhline(float(payload['cut'])*float(payload['dist_max']),color=C['amber'],lw=1,ls='--')
        ax.set_title(('Dendrograma resumido' if payload.get('truncated') else 'Dendrograma') if self.lang()=='pt' else ('Truncated dendrogram' if payload.get('truncated') else 'Dendrogram'),color=C['text'],fontsize=9)
        self._dendro_axes_style(ax)
        scope_label='gid' if payload.get('id_mode')=='gid' else 'name';mode_note=' • visual reduzido' if payload.get('truncated') and self.lang()=='pt' else (' • truncated view' if payload.get('truncated') else '')
        try:self.cluster_stats.config(text=(f"Escopo: seleção\nN imagens: {payload['n']:,}\nClasses: {payload['classes']}\ndist_max: {payload['dist_max']:.6g}\n{self.tr('groups')}: {payload['groups']}\nModo: {payload['mode']}\nIDs: {scope_label}\nArtifact: {payload['path']}{mode_note}").replace(',','.'))
        except Exception:pass
        self.current_cluster_path=Path(payload['path'])
        self._dendro_canvas.draw_idle();self._log_cluster_timings(timings,render_t0)

    def load_cluster_view(self,reuse_scope=False):
        """Prepare scope, locate artifact and reduce dendrogram off the Tk thread."""
        self._cluster_cut_after=None
        if not hasattr(self,'cluster_ax') or self.cluster_ax is None:return
        self._cluster_view_token+=1
        token=self._cluster_view_token
        cut=float(self.vars['cluster_cut'].get())
        gids=tuple(sorted(str(g) for g in self._active_cluster_gids()))
        project=self.project
        lang=self.lang()
        current_path=Path(self.current_cluster_path) if self.current_cluster_path else None
        cached=getattr(self,'_cluster_scope_cache',None)
        scope_key=(gids,id(project),str(current_path) if current_path else '')
        if not (reuse_scope and cached and cached[0]==scope_key):
            cached=None
            # Full reload: show a placeholder.  A cut-only change keeps the
            # current drawing on screen until the new one is ready (no flicker).
            ax=self.cluster_ax;ax.clear();ax.set_facecolor(C['log']);self._dendro_axes_style(ax)
            loading='Preparando seleção…' if self.lang()=='pt' else 'Preparing selection…'
            ax.text(.5,.5,loading,ha='center',va='center',color=C['muted'],transform=ax.transAxes)
            try:self.cluster_scope_count_lbl.config(text='—');self.cluster_scope_classes_lbl.config(text=loading);self.cluster_stats.config(text=loading)
            except Exception:pass
            self._dendro_canvas.draw_idle()

        def target():
            total_t0=time.perf_counter();timings={}
            try:
                if cached is not None:
                    _,path,classes,id_mode=cached
                    if Path(path).exists():
                        t=time.perf_counter();payload=self._prepare_cluster_view_payload(path,cut);timings['dendrogram_prepare_s']=time.perf_counter()-t
                        timings['total_s']=time.perf_counter()-total_t0
                        payload.update({'cut':cut,'classes':classes,'id_mode':id_mode,'path':path,'timings':timings})
                        self._event_q.put(('cluster_view_done',token,payload))
                        return
                self._event_q.put(('cluster_view_status',token,'Preparando seleção…' if lang=='pt' else 'Preparing selection…'))
                t=time.perf_counter();rows=self._cluster_selection_rows(gids=gids,project=project,timing=timings,cancel_check=lambda: token!=getattr(self,'_cluster_view_token',token));timings['scope_s']=time.perf_counter()-t
                if token!=getattr(self,'_cluster_view_token',token):return
                self._event_q.put(('cluster_view_status',token,'Localizando clustering existente…' if lang=='pt' else 'Locating saved clustering…'))
                t=time.perf_counter();p,id_mode=self._cluster_artifact_for_rows(rows,project=project,current_path=current_path);timings['artifact_lookup_s']=time.perf_counter()-t
                if token!=getattr(self,'_cluster_view_token',token):return
                classes=(rows['manual_label'].astype(str).nunique() if rows is not None and not rows.empty and 'manual_label' in rows.columns else 0)
                if not p:
                    timings['total_s']=time.perf_counter()-total_t0
                    self._event_q.put(('cluster_view_done',token,{'missing':True,'has_selection':rows is not None and len(rows)>=2,'n':0 if rows is None else len(rows),'classes':classes,'timings':timings}))
                    return
                self._event_q.put(('cluster_view_status',token,'Carregando visualização…' if lang=='pt' else 'Loading visualization…'))
                path=str(p);t=time.perf_counter();payload=self._prepare_cluster_view_payload(path,cut);timings['dendrogram_prepare_s']=time.perf_counter()-t
                self._cluster_scope_cache=(scope_key,path,classes,id_mode)
                timings['total_s']=time.perf_counter()-total_t0
                payload.update({'cut':cut,'classes':classes,'id_mode':id_mode,'path':path,'timings':timings})
                self._event_q.put(('cluster_view_done',token,payload))
            except Exception as exc:
                timings['total_s']=time.perf_counter()-total_t0
                self._event_q.put(('cluster_view_error',token,exc,timings))
        threading.Thread(target=target,daemon=True,name='iat-cluster-view').start()

    def relabel_by_group(self):
        if not self.project:return
        rows=self._cluster_selection_rows()
        p,id_mode=self._cluster_artifact_for_rows(rows)
        if not p:
            if hasattr(self,'cluster_status_vars'):self.cluster_status_vars['relabel'].set(self.tr('no_artifact'))
            self._log(self.tr('no_artifact'),'warn');return
        cluster_cut=float(self.vars['cluster_cut'].get())
        save_auto=bool(self.vars['save_auto'].get())
        def work(progress):
            art=read_cluster_artifact(p)
            groups=cluster_groups(art,cluster_cut)
            raw_ids=[str(n) for n in art['names']]
            if id_mode=='gid':
                gid_group={rid:int(g) for rid,g in zip(raw_ids,groups)}
            else:
                # Historic per-class artifacts store image names. The helper
                # only accepts this mode when names are unique in the scope.
                name_to_gid=dict(zip(rows['names'].astype(str),rows['gid'].astype(str)))
                gid_group={name_to_gid[rid]:int(g) for rid,g in zip(raw_ids,groups) if rid in name_to_gid}
            sub=rows[rows['gid'].astype(str).isin(gid_group)].copy()
            by={}
            for row in sub.itertuples(index=False):
                g=gid_group.get(str(row.gid))
                if g is not None:by.setdefault(f'{row.manual_label}__g{g}',[]).append(str(row.gid))
            total=0;items=list(by.items())
            for i,(newlab,gg) in enumerate(items,1):
                total+=self.project.apply_labels(gg,newlab);progress(i,len(items),newlab)
            if save_auto:self.project.save_dirty()
            return total
        def done(total):self._log(f'{total} imagens relabeladas por grupo','success');self.change_batch()
        self._run_operation(self.tr('saving'),work,done,
                            local_progress=self.cluster_progress_vars.get('relabel'),
                            local_status=self.cluster_status_vars.get('relabel'))


    # -------- segmentation --------
    def _build_segmentation(self):
        host=tk.Frame(self.content,bg=C['bg'])
        host.pack(fill='both',expand=True)
        host.columnconfigure(1,weight=1)
        host.rowconfigure(0,weight=1)

        # The left column is scrollable so the added scope controls remain fully
        # accessible on Windows machines with larger DPI/font scaling.
        leftcol=tk.Frame(host,bg=C['bg'],width=315)
        leftcol.grid(row=0,column=0,sticky='ns',padx=(4,7),pady=4)
        leftcol.grid_propagate(False)
        lscroll=ScrollFrame(leftcol,C['bg'])
        lscroll.pack(fill='both',expand=True)

        scope_card,scope=self._card(lscroll.inner,self.tr('seg_scope_title'))
        scope_card.pack(fill='x',pady=(0,7))
        self.seg_scope_buttons={}
        self.seg_scope_count_labels={}
        self.seg_scope_texts={
            'selected':self.tr('seg_scope_selected'),
            'checked':self.tr('seg_scope_checked'),
            'batch':self.tr('seg_scope_batch'),
            'project':self.tr('seg_scope_project'),
        }
        for key in ('selected','checked','batch','project'):
            row=tk.Frame(scope,bg=C['card'])
            row.pack(fill='x',pady=1)
            rb=tk.Radiobutton(
                row,text=self.seg_scope_texts[key],variable=self.vars['seg_scope'],value=key,
                command=self._refresh_seg_scope_ui,bg=C['card'],fg=C['text'],
                selectcolor=C['blue2'],activebackground=C['card'],activeforeground='white',
                anchor='w',justify='left',font=('Sans',8)
            )
            rb.pack(side='left',fill='x',expand=True)
            cnt=tk.Label(row,text='',bg=C['card'],fg=C['cyan'],font=('Sans',7,'bold'))
            cnt.pack(side='right',padx=(5,0))
            self.seg_scope_buttons[key]=rb
            self.seg_scope_count_labels[key]=cnt

        self.seg_scope_summary_var=tk.StringVar(value='')
        tk.Label(
            scope,textvariable=self.seg_scope_summary_var,bg=C['card'],fg=C['muted'],
            font=('Sans',7),wraplength=260,justify='left'
        ).pack(anchor='w',pady=(5,0))

        # Method parameters.
        card,ctrl=self._card(lscroll.inner,self.tr('seg_params'))
        card.pack(fill='x')

        self._label(ctrl,self.tr('method')).pack(anchor='w')
        display=list(SEGMENTATION_METHODS.values())
        rev={v:k for k,v in SEGMENTATION_METHODS.items()}
        self._seg_method_rev=rev
        self.seg_display=tk.StringVar(value=SEGMENTATION_METHODS[self.vars['seg_method'].get()])
        cb=ttk.Combobox(ctrl,textvariable=self.seg_display,values=display,state='readonly',style='Dark.TCombobox')
        cb.pack(fill='x')
        cb.bind('<<ComboboxSelected>>',self._seg_method_changed)

        self.seg_radius_label=self._label(ctrl,self.tr('radius'))
        self.seg_radius_label.pack(anchor='w',pady=(8,2))
        self.seg_radius_entry=self._entry(ctrl,self.vars['seg_radius'])
        self.seg_radius_entry.configure(disabledbackground=C['panel'],disabledforeground=C['muted'])
        self.seg_radius_entry.pack(fill='x',ipady=4)
        tk.Label(
            ctrl,
            text=('Usado apenas em Rolling ball, Black-hat + Otsu e Adaptive / Sauvola.'
                  if self.lang()=='pt' else
                  'Used only by Rolling ball, Black-hat + Otsu and Adaptive / Sauvola.'),
            wraplength=255,justify='left',bg=C['card'],fg=C['muted'],font=('Sans',7)
        ).pack(anchor='w',pady=(2,6))

        scale_box=tk.Frame(ctrl,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1)
        scale_box.pack(fill='x',pady=(4,8))
        tk.Label(
            scale_box,
            text=('Escala física das métricas' if self.lang()=='pt' else 'Physical metric scale'),
            bg=C['card2'],fg='#ffffff',font=('Sans',8,'bold')
        ).pack(anchor='w',padx=8,pady=(7,2))
        scale_row=tk.Frame(scale_box,bg=C['card2'])
        scale_row.pack(fill='x',padx=8,pady=(0,2))
        self.seg_pixel_entry=self._entry(scale_row,self.vars['pixel_size'])
        self.seg_pixel_entry.config(width=9)
        self.seg_pixel_entry.pack(side='left',ipady=3)
        tk.Label(scale_row,text='µm/pixel',bg=C['card2'],fg=C['text'],font=('Sans',8)).pack(side='left',padx=(5,0))
        tk.Label(
            scale_box,
            text=('Não altera o limiar/máscara; é usada para µm, µm² e barra de escala.'
                  if self.lang()=='pt' else
                  'Does not change threshold/mask; used for µm, µm² and the scale bar.'),
            wraplength=250,justify='left',bg=C['card2'],fg=C['muted'],font=('Sans',7)
        ).pack(anchor='w',padx=8,pady=(0,7))
        self.seg_pixel_entry.bind('<Return>',lambda e:(self._persist_project_pixel_size(),self._seg_refresh_preview()))
        self.seg_pixel_entry.bind('<FocusOut>',lambda e:(self._persist_project_pixel_size(silent=True),self._seg_refresh_preview()))

        tk.Checkbutton(
            ctrl,text=self.tr('show_outline'),variable=self.vars['seg_outline'],
            command=self._seg_refresh_preview,bg=C['card'],fg=C['text'],
            selectcolor=C['red'],activebackground=C['card'],activeforeground='white'
        ).pack(anchor='w',pady=6)
        self._btn(ctrl,self.tr('build_seg'),self.build_segmentation,C['purple']).pack(fill='x',ipady=5)

        # Preview + metrics.
        center=tk.Frame(host,bg=C['bg'])
        center.grid(row=0,column=1,sticky='nsew',padx=(0,4),pady=4)
        center.columnconfigure(0,weight=1)
        center.columnconfigure(1,weight=1)
        center.rowconfigure(0,weight=1)

        pc,pb=self._card(center,self.tr('preview'))
        pc.grid(row=0,column=0,sticky='nsew',padx=(0,4),pady=(0,4))
        self.seg_preview=tk.Label(pb,bg=C['log'])
        self.seg_preview.pack(fill='both',expand=True)

        mc,mb=self._card(center,self.tr('metrics'))
        mc.grid(row=0,column=1,sticky='nsew',padx=(4,0),pady=(0,4))
        self.seg_metrics=tk.Label(
            mb,text='—',justify='left',anchor='nw',bg=C['card'],fg=C['text'],
            font=('Monospace',8),wraplength=420
        )
        self.seg_metrics.pack(fill='both',expand=True)

        # Method documentation: equation actually implemented + reference.
        dc,db=self._card(center,'Método: equação e referência' if self.lang()=='pt' else 'Method: equation and reference')
        dc.grid(row=1,column=0,sticky='nsew',padx=(0,4),pady=(4,4))
        self.seg_doc_title=tk.Label(db,text='',bg=C['card'],fg=C['text'],font=('Sans',10,'bold'),anchor='w')
        self.seg_doc_title.pack(fill='x')
        self.seg_doc_eq=tk.Label(db,text='',bg=C['card2'],fg=C['text'],font=('DejaVu Sans Mono',9),
                                 justify='left',anchor='nw',padx=10,pady=8,wraplength=620)
        self.seg_doc_eq.pack(fill='x',pady=(6,6))
        self.seg_doc_ref=tk.Label(db,text='',bg=C['card'],fg=C['muted'],font=('Sans',8),
                                  justify='left',anchor='nw',wraplength=620)
        self.seg_doc_ref.pack(fill='x')
        fc,fb=self._card(center,'Métricas: definições' if self.lang()=='pt' else 'Metrics: definitions')
        fc.grid(row=1,column=1,sticky='nsew',padx=(4,0),pady=(4,4))
        formulas=tk.Label(fb,text=METRIC_FORMULAS['en' if self.lang()=='en' else 'pt'],bg=C['card2'],fg=C['text'],
                 font=('DejaVu Sans Mono',9),justify='left',anchor='nw',padx=10,pady=8,wraplength=620)
        formulas.pack(fill='x')
        # Wrap long equations to the real card width instead of a fixed width.
        for lbl in (self.seg_doc_eq,self.seg_doc_ref,formulas):
            lbl.bind('<Configure>',lambda e,w=lbl:w.configure(wraplength=max(200,e.width-24)) if abs(int(float(w.cget('wraplength') or 0))-(e.width-24))>8 else None,add='+')
        self._seg_update_method_doc()

        oc,ob=self._card(center,self.tr('gallery_order_metrics'))
        oc.grid(row=2,column=0,columnspan=2,sticky='nsew',pady=(4,0))
        fields_text=(METRIC_FIELDS_TEXT['en' if self.lang()=='en' else 'pt'])
        tk.Label(
            ob,text=self.tr('available_fields')+':\n'+fields_text,
            wraplength=760,justify='left',bg=C['card'],fg=C['muted'],font=('Sans',8)
        ).pack(anchor='w')

        self._refresh_seg_scope_ui()
        self._update_seg_param_state()
        self._seg_refresh_preview()

    def _refresh_seg_scope_ui(self):
        """Refresh scope counts/states without rebuilding the Segmentation tab."""
        bn=str(self.vars['batch'].get() or 'Overview')
        selected_n=len(self.selected_gids)
        checked_n=len(self.checked_gids)
        batch_n=len(self.current_df) if bn!='Overview' else None
        project_n=len(self.current_df) if bn=='Overview' else None
        counts={'selected':selected_n,'checked':checked_n,'batch':batch_n,'project':project_n}
        enabled={
            'selected':selected_n>0,
            'checked':checked_n>0,
            'batch':bn!='Overview',
            'project':bool(self.project and self.project.csv_names()),
        }
        for key,btn in getattr(self,'seg_scope_buttons',{}).items():
            try:btn.configure(state='normal' if enabled.get(key,False) else 'disabled')
            except Exception:pass
        for key,lbl in getattr(self,'seg_scope_count_labels',{}).items():
            n=counts.get(key)
            if n is None:
                text=('todos os batches' if key=='project' and bn!='Overview' and self.lang()=='pt'
                      else 'all batches' if key=='project' and bn!='Overview'
                      else '—')
            else:
                text=f'{n:,}'.replace(',','.')
            try:lbl.configure(text=text)
            except Exception:pass

        key=str(self.vars['seg_scope'].get() or 'selected')
        desc=getattr(self,'seg_scope_texts',{}).get(key,key)
        n=counts.get(key)
        if n is None:
            amount=('todos os batches' if key=='project' and bn!='Overview' and self.lang()=='pt'
                    else 'all batches' if key=='project' and bn!='Overview'
                    else 'indisponível' if self.lang()=='pt' else 'unavailable')
        else:
            amount=f'{n:,}'.replace(',','.')
        if hasattr(self,'seg_scope_summary_var'):
            self.seg_scope_summary_var.set(f"{self.tr('seg_scope_confirm')}: {desc} • {amount}")

    def _seg_method_changed(self,event=None):
        rev=getattr(self,'_seg_method_rev',{})
        key=rev.get(self.seg_display.get(),self.vars['seg_method'].get())
        self.vars['seg_method'].set(key)
        self._update_seg_param_state()
        self._seg_update_method_doc()
        self._seg_refresh_preview()

    def _seg_update_method_doc(self):
        try:
            method=self.vars['seg_method'].get()
            eq,ref=method_doc_text(method,self.lang())
            self.seg_doc_title.config(text=SEGMENTATION_METHODS.get(method,method))
            self.seg_doc_eq.config(text=eq)
            self.seg_doc_ref.config(text=ref)
        except Exception:
            pass

    def _update_seg_param_state(self):
        """Enable only controls that are genuine inputs for the selected method."""
        method=self.vars['seg_method'].get()
        radius_methods={'rolling_ball','black_hat_otsu','adaptive_sauvola'}
        enabled=method in radius_methods
        entry=getattr(self,'seg_radius_entry',None)
        label=getattr(self,'seg_radius_label',None)
        if entry is not None:
            try:entry.configure(state='normal' if enabled else 'disabled')
            except Exception:pass
        if label is not None:
            try:label.configure(fg=C['text'] if enabled else C['muted'])
            except Exception:pass
    def _current_preview_row(self):
        if not self.project:return None
        gids=list(self.checked_gids or self.selected_gids)
        df=self.current_df
        if gids:
            s=df[df['gid'].astype(str).isin(gids)];
            if not s.empty:return s.iloc[0]
        if not df.empty:return df.iloc[0]
        return None
    def _seg_refresh_preview(self):
        if not hasattr(self,'seg_preview') or not self.project:return
        row=self._current_preview_row()
        if row is None:return
        bn=str(row.get('csv_basename',''))
        name=str(row.get('names',''))
        p=self.project.resolve_image(bn,name)
        method=self.vars['seg_method'].get()
        cache=self.project.load_outline_cache(bn,method)
        ol=cache.get(name)
        try:
            im=Image.open(p).convert('RGB') if p and Image else None
            if im:
                ow,oh=im.size
                im.thumbnail((330,240),Image.Resampling.LANCZOS)
                if self.vars['seg_outline'].get() and ol and ol.get('points'):
                    sx=im.width/max(float(ol.get('width') or ow),1)
                    sy=im.height/max(float(ol.get('height') or oh),1)
                    pts=[(x*sx,y*sy) for x,y in ol['points']]
                    if len(pts)>1:
                        ImageDraw.Draw(im).line(pts+[pts[0]],fill='#ff4f66',width=1)
                self._preview_photo=ImageTk.PhotoImage(im)
                self.seg_preview.config(image=self._preview_photo)
        except Exception:
            pass

        fallback_um=self._segmentation_pixel_size(None,self.vars['pixel_size'].get())
        seg_row=None
        segp=self.project.segmentation_csv_path(bn,method)
        if segp.exists():
            try:
                sdf=self.project.get_segmentation_table(bn,method)
                m=sdf[sdf['Image Name'].astype(str)==name] if not sdf.empty and 'Image Name' in sdf.columns else pd.DataFrame()
                if not m.empty:
                    seg_row=m.iloc[0]
            except Exception:
                seg_row=None
        um=self._segmentation_pixel_size(seg_row,fallback_um)

        metric_lines=[
            f"{self.tr('method')}: {SEGMENTATION_METHODS.get(method,method)}",
            f"Pixel size: {um:g} µm/pixel",
            ''
        ]
        if seg_row is not None:
            metric_lines.extend(self._format_segmentation_metrics(seg_row,um))
        else:
            metric_lines.append(self.tr('no_segmentation'))
        try:self.seg_metrics.config(text='\n'.join(metric_lines))
        except Exception:pass
    def build_segmentation(self):
        if not self.project:return
        bn=str(self.vars['batch'].get() or 'Overview')
        method=str(self.vars['seg_method'].get())
        scope=str(self.vars['seg_scope'].get() or 'selected')
        try:
            radius=int(float(self.vars['seg_radius'].get()))
            px=parse_pixel_size(self.vars['pixel_size'].get())
            if not math.isfinite(px) or px<=0:raise ValueError
        except Exception:
            messagebox.showerror(self.tr('iat_title'),
                                 'Verifique o raio e o pixel size.' if self.lang()=='pt' else 'Check radius and pixel size.')
            return

        tasks=[]
        scope_count=0
        if scope in ('selected','checked'):
            gids=set(self.selected_gids if scope=='selected' else self.checked_gids)
            if not gids:
                messagebox.showinfo(
                    self.tr('iat_title'),
                    self.tr('no_selection') if scope=='selected' else self.tr('no_checked')
                )
                return
            by_batch={}
            for gid in map(str,gids):
                if '::' not in gid:continue
                csvn,_=gid.split('::',1)
                if csvn:
                    by_batch.setdefault(csvn,[]).append(gid)
            valid_csv=set(self.project.csv_names())
            tasks=[(csvn,gg) for csvn,gg in by_batch.items() if csvn in valid_csv and gg]
            scope_count=sum(len(gg) for _,gg in tasks)
            if not tasks:
                messagebox.showinfo(self.tr('iat_title'),self.tr('no_selection'))
                return
        elif scope=='batch':
            if bn=='Overview':
                messagebox.showinfo(
                    self.tr('iat_title'),
                    'Selecione um batch específico antes de usar este escopo.'
                    if self.lang()=='pt' else
                    'Select a specific batch before using this scope.'
                )
                return
            tasks=[(bn,None)]
            scope_count=len(self.current_df)
        elif scope=='project':
            csvs=self.project.csv_names()
            if not csvs:
                return
            tasks=[(csvn,None) for csvn in csvs]
            scope_count=len(self.current_df) if bn=='Overview' else 0
            detail=(f'{scope_count:,} imagens em {len(csvs)} batches'.replace(',','.')
                    if scope_count else f'{len(csvs)} batches')
            question=(
                f'Você está prestes a gerar/atualizar a segmentação para o projeto inteiro ({detail}).\n\n'
                'A operação pode demorar bastante. Deseja continuar?'
                if self.lang()=='pt' else
                f'You are about to generate/update segmentation for the entire project ({detail}).\n\n'
                'This operation may take a long time. Continue?'
            )
            if not messagebox.askyesno(self.tr('iat_title'),question):
                return
        else:
            return

        # Snapshot the plan before entering the worker; do not read Tk variables
        # from the background thread.
        plan=[(str(csvn),None if gids is None else tuple(map(str,gids))) for csvn,gids in tasks]
        def work(progress):
            outs=[]
            total_tasks=max(len(plan),1)
            for j,(csvn,gids) in enumerate(plan):
                def local(i,total,name,j=j,csvn=csvn):
                    progress(j+i/max(total,1),total_tasks,f'{csvn}: {name}')
                outs.append(
                    run_segmentation(
                        self.project,csvn,method,radius,px,local,
                        include_gids=gids
                    )
                )
            return outs

        def done(outs):
            # Keep ordinary thumbnails cached; only invalidate variants that
            # contain an outline from this segmentation method.
            self._thumb_cache={
                k:v for k,v in self._thumb_cache.items()
                if not (len(k)>=4 and bool(k[2]) and k[3]==method)
            }
            stats={'rows':0,'located':0,'read':0,'segmented':0,'contours':0}
            for _,_,s in outs:
                for k in stats:stats[k]+=s.get(k,0)
            self._log(
                f"{self.tr('seg_done')}: processadas={stats['rows']} "
                f"segmentadas={stats['segmented']} contornos={stats['contours']}",
                'success'
            )
            self._seg_refresh_preview()
            self._refresh_seg_scope_ui()

        self._run_operation(self.tr('seg_running'),work,done)


    # -------- export --------


    # -------- training-dataset export (IAT > Export only) --------
    def _refresh_state_cards(self):
        # Kept for compatibility with older callbacks; the Export tab no longer
        # contains the "Estado das alterações" panel.
        if not hasattr(self,'state_frame'):return
        try:
            if not self.state_frame.winfo_exists():return
        except Exception:return
        for w in self.state_frame.winfo_children():w.destroy()
        vals=[(self.tr('dirty_batches'),len(self.project.dirty_batches) if self.project else 0,C['amber']),(self.tr('dirty_classes'),len(self.project.dirty_classes) if self.project else 0,C['red']),(self.tr('csv_saved'),self.saved_count,C['green']),(self.tr('selected_images'),len(self.checked_gids or self.selected_gids),C['cyan'])]
        for i,(lab,val,col) in enumerate(vals):
            f=tk.Frame(self.state_frame,bg=C['card2'],highlightbackground=C['border'],highlightthickness=1);f.grid(row=0,column=i,sticky='nsew',padx=5,pady=5);self.state_frame.columnconfigure(i,weight=1);tk.Label(f,text=lab,bg=C['card2'],fg=C['muted'],font=('Sans',8)).pack(anchor='w',padx=10,pady=(8,2));tk.Label(f,text=str(val),bg=C['card2'],fg=col,font=('Sans',17,'bold')).pack(anchor='w',padx=10,pady=(0,8))


    # -------- logs tab --------
    def _build_logs(self):
        host=tk.Frame(self.content,bg=C['bg']); host.pack(fill='both',expand=True)
        host.rowconfigure(1,weight=1); host.columnconfigure(0,weight=1)

        c,b=self._card(host,self.tr('operation')); c.grid(row=0,column=0,sticky='ew',padx=4,pady=(4,6))
        top=tk.Frame(b,bg=C['card']); top.pack(fill='x')
        tk.Label(top,text=self.tr('status'),bg=C['card'],fg=C['muted'],font=('Sans',8)).pack(side='left')
        self.status_badge=tk.Label(top,textvariable=self.status_var,bg='#103a2a',fg=C['green'],font=('Sans',8,'bold'),padx=12,pady=3)
        self.status_badge.pack(side='left',padx=10)
        tk.Label(top,textvariable=self.elapsed_var,bg=C['card'],fg=C['muted'],font=('Monospace',9)).pack(side='right')
        tk.Label(b,textvariable=self.operation_var,bg=C['card'],fg=C['text'],anchor='w',font=('Sans',9,'bold')).pack(fill='x',pady=(8,5))
        ph=tk.Frame(b,bg=C['card']); ph.pack(fill='x')
        tk.Label(ph,text=self.tr('progress'),bg=C['card'],fg=C['muted'],font=('Sans',8)).pack(side='left')
        tk.Label(ph,textvariable=self.progress_text,bg=C['card'],fg=C['muted'],font=('Sans',8)).pack(side='right')
        ttk.Progressbar(b,maximum=100,variable=self.progress_var,style='Blue.Horizontal.TProgressbar').pack(fill='x',pady=(3,8))
        actions=tk.Frame(b,bg=C['card']); actions.pack(fill='x')
        self.stop_btn=self._btn(actions,self.tr('stop'),self.stop_operation,'#5b2630'); self.stop_btn.pack(side='left',ipadx=8,ipady=3)
        self.stop_btn.configure(state='normal' if self._op_running else 'disabled')
        self._btn(actions,self.tr('open_folder'),self.open_project_folder).pack(side='right',ipadx=8,ipady=3)

        c,b=self._card(host,self.tr('iat_log')); c.grid(row=1,column=0,sticky='nsew',padx=4,pady=(0,4))
        top=tk.Frame(b,bg=C['card']); top.pack(fill='x')
        self._btn(top,self.tr('clear'),self.clear_log).pack(side='right')
        self.log=tk.Text(b,bg=C['log'],fg=C['text'],relief='flat',font=('Monospace',9),wrap='word')
        self.log.pack(fill='both',expand=True,pady=(6,0))
        tags={'normal':'#e8e8ea','muted':'#7f93aa','info':C['cyan'],'success':C['green'],'warn':C['amber'],'error':C['red'],'cluster':C['purple']}
        for tag,col in tags.items(): self.log.tag_configure(tag,foreground=col)
        for msg,tag,stamp in self.log_lines:
            self.log.insert('end',f'[{stamp}] ','muted'); self.log.insert('end',msg+'\n',tag if tag in tags else 'normal')
        self.log.configure(state='disabled'); self.log.see('end')

    # -------- operation/log infra --------
    def _run_operation(self,label,worker,done=None,local_progress=None,local_status=None):
        if self._op_running:
            self._log('Outra operação já está em andamento.','warn'); return
        self._op_running=True; self._op_stop=False; self._op_start=time.time(); self._op_done_cb=done
        self._active_local_progress=local_progress; self._active_local_status=local_status
        if local_progress is not None: local_progress.set(0)
        if local_status is not None: local_status.set(self.tr('cluster_running_short'))
        if self.current_tab!='logs': self._operation_origin_tab=self.current_tab
        for name in ('nav_train','nav_class','nav_iat'):
            try: getattr(self.app,name).configure(state='disabled')
            except Exception: pass
        self.operation_var.set(label); self.status_var.set(self.tr('running'))
        if self.status_badge is not None:
            try:self.status_badge.config(bg='#173b2a',fg=C['green'])
            except Exception:pass
        self.progress_var.set(0); self.progress_text.set('0%')
        if self.stop_btn is not None:
            try:self.stop_btn.config(state='normal')
            except Exception:pass
        self._log(label,'info')
        def progress(cur,total,text=''):
            if self._op_stop: raise InterruptedError('Interrompido')
            try: pct=max(0,min(100,float(cur)/max(float(total),1)*100))
            except Exception: pct=0
            self._event_q.put(('progress',pct,text))
        def target():
            try:
                result=worker(progress); self._event_q.put(('done',result))
            except InterruptedError as e:
                self._event_q.put(('stopped',e))
            except Exception as e:
                self._event_q.put(('error',e))
        self._op_thread=threading.Thread(target=target,daemon=True); self._op_thread.start()

    def _poll_events(self):
        """Process a bounded queue slice and always re-arm the Tk monitor."""
        now=time.perf_counter();previous=getattr(self,'_poll_last_perf',None);self._poll_last_perf=now
        if previous is not None:
            gap=now-previous;last_log=getattr(self,'_last_poll_lag_log',0.0)
            if gap>0.75 and now-last_log>2.0:self._last_poll_lag_log=now;self._log(f'Diagnóstico Tk: event loop sem polling por {gap:.2f}s.','warn')
        deadline=time.monotonic()+0.012;processed=0
        try:
            while processed<60 and time.monotonic()<deadline:
                try:event=self._event_q.get_nowait()
                except queue.Empty:break
                processed+=1;kind=event[0] if event else 'unknown'
                try:
                    if kind=='progress':self._set_progress(event[1],event[2])
                    elif kind=='thumb':self._apply_thumbnail_result(event[1],event[2])
                    elif kind=='gallery_prefetch_thumb':self._finish_gallery_prefetch(event[1],event[2],event[3])
                    elif kind=='cycle_training_ready':self._cycle_launch_training_process(event[1])
                    elif kind=='cycle_training_done':self._cycle_training_finished(event[1])
                    elif kind=='cycle_apply_progress':self._cycle_apply_progress_event(event[1])
                    elif kind=='cycle_training_progress':self._cycle_training_progress_event(event[1])
                    elif kind=='cycle_cluster_progress':self._cycle_cluster_progress_event(event[1])
                    elif kind=='cycle_cluster_done':self._cycle_cluster_finished(event[1])
                    elif kind=='cluster_view_status':self._apply_cluster_view_status(event[1],event[2])
                    elif kind=='cluster_view_done':self._apply_cluster_view_result(event[1],event[2])
                    elif kind=='cluster_view_error':self._apply_cluster_view_result(event[1],None,event[2],event[3] if len(event)>3 else None)
                    elif kind=='cycle_log':
                        line=str(event[1]);self._cycle_training_log_event(line)
                        # tqdm refreshes (≈10/s) would flood the log; epoch
                        # summaries and all other messages are still logged.
                        if not _TQDM_LINE_RE.search(line):self._log(line,'info')
                    elif kind=='done':cb=self._op_done_cb;self._op_done_cb=None;self._finish_operation(True,event[1],cb,None)
                    elif kind=='stopped':self._op_done_cb=None;self._finish_operation(False,None,None,event[1],stopped=True)
                    elif kind=='error':self._op_done_cb=None;self._finish_operation(False,None,None,event[1])
                    else:self._log(f'Evento IAT desconhecido ignorado: {kind}','warn')
                except Exception as exc:
                    self._log(f'Erro recuperável ao processar evento IAT {kind}: {exc}','error')
                    if kind in {'done','stopped','error'} and self._op_running:
                        self._op_done_cb=None
                        try:self._finish_operation(False,None,None,exc)
                        except Exception as finish_exc:self._op_running=False;self._log(f'Falha ao finalizar operação após erro de evento: {finish_exc}','error')
        except Exception as exc:self._log(f'Falha recuperável no monitor de eventos IAT: {exc}','error')
        finally:
            try:exists=self.winfo_exists()
            except Exception:exists=False
            if exists:
                try:self.after(15 if not self._event_q.empty() else 100,self._poll_events)
                except Exception as exc:
                    try:self.log_lines.append((f'Falha ao reagendar monitor IAT: {exc}','error',datetime.now().strftime('%H:%M:%S')))
                    except Exception:pass

    def _set_progress(self,pct,text=''):
        self.progress_var.set(pct);self.progress_text.set(f'{pct:.0f}%')
        if self._active_local_progress is not None:
            try:self._active_local_progress.set(pct)
            except Exception:pass
        if text:
            self.operation_var.set(text)
            if self._active_local_status is not None:
                try:self._active_local_status.set(text)
                except Exception:pass
    def _finish_operation(self,ok,result,done,error,stopped=False):
        self._op_running=False
        if self.stop_btn is not None:
            try:self.stop_btn.config(state='disabled')
            except Exception:pass
        for name in ('nav_train','nav_class','nav_iat'):
            try: getattr(self.app,name).configure(state='normal')
            except Exception: pass
        if ok:
            self.progress_var.set(100);self.progress_text.set('100%');self.status_var.set(self.tr('completed'))
            if self._active_local_progress is not None:
                try:self._active_local_progress.set(100)
                except Exception:pass
            if self._active_local_status is not None:
                try:self._active_local_status.set(self.tr('cluster_completed'))
                except Exception:pass
            if self.status_badge is not None:
                try:self.status_badge.config(bg='#103a2a',fg=C['green'])
                except Exception:pass
            if done:
                try:done(result)
                except Exception as e:self._log(str(e),'error')
        else:
            self.status_var.set(self.tr('failed') if not stopped else self.tr('stop'))
            if self._active_local_status is not None:
                try:self._active_local_status.set((self.tr('cluster_failed')+': '+str(error))[:120])
                except Exception:pass
            if self.status_badge is not None:
                try:self.status_badge.config(bg='#49222b',fg=C['red'])
                except Exception:pass
            self._log(str(error),'error')
        self._active_local_progress=None; self._active_local_status=None
    def stop_operation(self):self._op_stop=True;self._log('Solicitação de interrupção enviada.','warn')
    def _tick(self):
        if self._op_running and self._op_start:
            s=int(time.time()-self._op_start);self.elapsed_var.set(f'{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}')
        self.after(500,self._tick)
    def _log(self,msg,tag='normal'):
        stamp=datetime.now().strftime('%H:%M:%S');self.log_lines.append((str(msg),tag,stamp));self.log_lines=self.log_lines[-1000:];self._insert_log(str(msg),tag,stamp,store=False)
    def _insert_log(self,msg,tag='normal',stamp=None,store=False):
        if self.log is None:return
        try:
            if not self.log.winfo_exists(): return
            stamp=stamp or datetime.now().strftime('%H:%M:%S');self.log.configure(state='normal');self.log.insert('end',f'[{stamp}] ', 'muted');self.log.insert('end',str(msg)+'\n',tag if tag in self.log.tag_names() else 'normal')
            lines=int(self.log.index('end-1c').split('.')[0])
            if lines>2000:self.log.delete('1.0',f'{lines-2000}.0')
            self.log.see('end');self.log.configure(state='disabled')
        except Exception:
            return
    def clear_log(self):
        self.log_lines.clear()
        if self.log is not None:
            try:self.log.configure(state='normal');self.log.delete('1.0','end');self.log.configure(state='disabled')
            except Exception:pass
    def open_project_folder(self):
        if self.project and self.project.project_dir.exists():
            try:open_path(self.project.project_dir)
            except Exception:pass
