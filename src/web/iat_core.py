#!/usr/bin/env python3
"""Headless IAT workspace for the web interface.

``HeadlessIAT`` *is* the classic ``IATWorkspace`` (with all its mixins:
gallery, export, training export, integrated model cycle) minus the Tk
widgets.  Every data operation — project loading, filters, structural
analysis, selection, relabel/undo/save, clustering, segmentation, exports,
the integrated cycle — runs the original methods unchanged.  The web layer
only reads the resulting state and renders it.

The original ``__init__`` is reused by source transformation: only the lines
that build Tk widgets are removed, so the initial state and defaults can
never drift from the classic interface.
"""
from __future__ import annotations

import inspect
import itertools
import textwrap
import time
from datetime import datetime

import iat_export_mixin as _exp_mod
import iat_gallery_mixin as _gal_mod
import iat_model_cycle_mixin as _cyc_mod
import iat_temporal_spatial as _tsp_mod
import iat_training_export_mixin as _tex_mod
import iat_workspace as _ws_mod
from iat_workspace import IATWorkspace

from .controller import get_controller
from .headless import StubWidget, install_module_shims

install_module_shims(_ws_mod, _gal_mod, _exp_mod, _tex_mod, _cyc_mod, _tsp_mod)

_WIDGET_ONLY_LINES = (
    "super().__init__(parent,bg=C['bg'])",
    'self._build()',
    "self.bind_all('<MouseWheel>', self._route_mousewheel, add='+')",
    "self.bind_all('<Button-4>', self._route_mousewheel, add='+')",
    "self.bind_all('<Button-5>', self._route_mousewheel, add='+')",
    'self.app.register_callback(self.refresh_language)',
)


def _headless_init_source():
    src = textwrap.dedent(inspect.getsource(IATWorkspace.__init__))
    for line in _WIDGET_ONLY_LINES:
        if src.count(line) != 1:
            raise RuntimeError(f'IATWorkspace.__init__ changed; cannot strip: {line}')
        src = src.replace(line, 'pass')
    src = src.replace('def __init__(self,parent,app):', 'def _iat_headless_init(self,parent,app):', 1)
    namespace = {}
    exec(compile(src, _ws_mod.__file__, 'exec'), _ws_mod.__dict__, namespace)
    return namespace['_iat_headless_init']


_HEADLESS_INIT = _headless_init_source()


class AppAdapter:
    """What ``IATWorkspace`` expects from its parent application."""

    def __init__(self, ctl):
        self._ctl = ctl

    def __getattr__(self, name):
        return getattr(self._ctl, name)

    def register_callback(self, *_a):
        return None

    def after_idle(self, func, *args):
        return self._ctl.after(0, func, *args)

    def _apply_font_scale(self):
        return None

    def set_iat_subnav_active(self, _tab):
        return None


class HeadlessIAT(IATWorkspace):
    def __init__(self, ctl):  # noqa: D401 - deliberately not calling tk.Frame
        self._ctl = ctl
        self._log_seq = itertools.count(1)
        self.log_seq = 0
        self.log_epoch = 0
        self._seq_lines = []
        self.view_version = 0
        self.scatter_version = 0
        self.gallery_version = 0
        self.cluster_view = None
        self.cluster_view_version = 0
        # Widgets referenced by the reused code paths.
        for name in ('batch_combo', 'mode_lbl', 'image_count_lbl', 'status_badge', 'stop_btn', 'content',
                     'iat_head', 'project_bar', 'right', 'scatter_selected_lbl'):
            setattr(self, name, StubWidget())
        _HEADLESS_INIT(self, None, AppAdapter(ctl))
        self.log = None
        # The "Diagnóstico Tk: event loop sem polling" message measures Tk's
        # main loop, which does not exist here; disable it (it would only
        # report web rendering time, not a stalled interface).
        self._last_poll_lag_log = float('inf')
        self._scatter_cache = {}

    # --- Tk event-loop stand-ins -------------------------------------
    def after(self, ms, func=None, *args):
        return self._ctl.after(ms, func, *args)

    def after_idle(self, func=None, *args):
        return self._ctl.after(0, func, *args)

    def after_cancel(self, job):
        self._ctl.after_cancel(job)

    def winfo_exists(self):
        return True

    def update_idletasks(self):
        return None

    # --- presentation hooks ------------------------------------------
    def show_tab(self, tab, rebuild=False):
        if self._op_running and tab not in {self.current_tab, 'logs', self._operation_origin_tab}:
            return
        if self.current_tab == 'clustering' and tab != 'clustering':
            self._cluster_view_token += 1
        self.current_tab = tab
        self.view_version += 1

    def _draw_scatter(self):
        self.scatter_version += 1

    def _refresh_state_cards(self):
        return None

    def _enforce_contrast(self):
        return None

    def _refit_active_plots(self):
        return None

    def _insert_log(self, msg, tag='normal', stamp=None, store=False):
        return None

    # Scatter: the original filtering methods, memoized while the data and the
    # display options are unchanged (Overview can hold millions of rows; the
    # classic UI also recomputed these only when something changed).
    def scatter_base_key(self):
        df = self.current_df
        return (id(df), 0 if df is None else len(df), str(self.vars['scatter_color_mode'].get()),
                self.gallery_version, self.view_version, bool(self._analysis_projection_active),
                id(self._analysis_df), self._name_filter_spec, bool(self.vars['hide_relabel'].get()),
                str(self.vars['batch'].get()))

    def _scatter_memo(self, name, extra, fn):
        base = self.scatter_base_key()
        if self._scatter_cache.get('__base__') != base:
            self._scatter_cache = {'__base__': base}
        key = (name, extra)
        if key not in self._scatter_cache:
            self._scatter_cache[key] = fn()
        return self._scatter_cache[key]

    def _scatter_filtered(self):
        visible = frozenset(self.visible_classes)
        out = self._scatter_memo('filtered', visible, lambda: IATWorkspace._scatter_filtered(self))
        return out

    def _scatter_class_labels(self, df=None):
        if df is not None:
            return IATWorkspace._scatter_class_labels(self, df)
        return list(self._scatter_memo('labels', None, lambda: IATWorkspace._scatter_class_labels(self)))

    # Gallery: the original methods compute membership/order/pagination; the
    # web page re-renders the visible page when ``gallery_version`` changes.
    def _patch_gallery_page(self, src, updated_labels=None):
        self._gallery_page_state(src)
        self.gallery_version += 1

    def _update_gallery_check_widgets(self, gids=None):
        self.gallery_version += 1

    def _gallery_update_in_place(self, reset_page=False, prefetch=True):
        if reset_page:
            self.page_idx = 0
        self.gallery_version += 1

    def _refresh_gallery_histograms(self):
        return None

    def _refresh_visible_thumbnails(self):
        self.gallery_version += 1

    def _schedule_gallery_prefetch(self, *a, **k):
        return None

    def _gallery_begin_thumbnail_generation(self, *a, **k):
        return None

    # Segmentation preview is rendered by the web page from the same sources.
    def _seg_refresh_preview(self):
        self.seg_version = getattr(self, 'seg_version', 0) + 1

    def _seg_update_method_doc(self):
        self.seg_version = getattr(self, 'seg_version', 0) + 1

    # Clustering: the original background loader posts its result here.
    @staticmethod
    def _dendro_axes_style(ax):
        return None

    def _apply_cluster_view_status(self, token, text):
        if token != getattr(self, '_cluster_view_token', None):
            return
        self.cluster_view = {'status': text}
        self.cluster_view_version += 1

    def _apply_cluster_view_result(self, token, payload, error=None, timings=None):
        if token != getattr(self, '_cluster_view_token', None):
            return
        if error is not None:
            self.cluster_view = {'error': True, 'message': str(error)}
            self._log(f'Falha ao carregar visualização do clustering: {error}', 'error')
        else:
            self.cluster_view = dict(payload or {})
            if payload and payload.get('path'):
                from pathlib import Path as _P
                self.current_cluster_path = _P(payload['path'])
            timings = timings or (payload.get('timings', {}) if isinstance(payload, dict) else {})
            self._log_cluster_timings(timings)
        self.cluster_view_version += 1

    # Integrated cycle: same data steps as the classic widgets, no drawing.
    def prepare_cycle(self):
        from .headless import HeadlessBoolVar
        self._cycle_state_init()
        for name in ('_cycle_train_btn', '_cycle_apply_btn', '_cycle_eval_btn', '_cycle_model_combo',
                     '_cycle_target_manual_host'):
            if getattr(self, name, None) is None:
                setattr(self, name, StubWidget())
        key = (str(self.project.project_dir), tuple(self.project.csv_names())) if self.project else None
        if getattr(self, '_cycle_batches_key', None) != key:
            self._cycle_batches_key = key
            self._cycle_populate_batches()
            self._cycle_refresh_saved_models()
            self._cycle_target_mode_changed()
        self._HeadlessBoolVar = HeadlessBoolVar

    def _cycle_populate_batches(self):
        from .headless import HeadlessBoolVar
        self._cycle_batch_vars = {}
        self._cycle_batch_meta = {}
        if not self.project:
            return
        for name in self.project.csv_names():
            self._cycle_batch_vars[name] = HeadlessBoolVar(value=False)
            try:
                df = self.project.load_batch(name)
                total = len(df)
                validated = int(df['manual_label'].fillna('').astype(str).str.strip().ne('').sum()) \
                    if 'manual_label' in df.columns else 0
            except Exception:
                total = validated = 0
            self._cycle_batch_meta[name] = {'total': total, 'validated': validated,
                                            'pct': (100.0 * validated / total) if total else 0.0}
        self._cycle_batch_selection_changed()

    def _cycle_batch_selection_changed(self):
        """Same as the classic method, but totals come from the batch metadata
        already read (the classic version re-read every selected CSV only to
        print a label that the web page computes itself)."""
        self._cycle_reference_batches = list(self._cycle_selected_batches())
        self._cycle_refresh_batch_visuals()
        if getattr(self, '_cycle_target_mode', None) is not None and self._cycle_target_mode.get() == 'manual':
            self._cycle_target_mode_changed()

    def _cycle_load_train_config(self):
        """Classic config + the pixel size typed in the cycle (metadata of the
        model, like in train.py; it does not change the network input)."""
        from iat_model_cycle_mixin import IATModelCycleMixin
        cfg = IATModelCycleMixin._cycle_load_train_config(self)
        var = getattr(self, '_cycle_pixel_size', None)
        if var is not None and str(var.get()).strip():
            from project_settings import parse_pixel_size
            cfg['pixelSize'] = parse_pixel_size(var.get())
        return cfg

    def _cycle_render_classes(self, counts):
        from .headless import HeadlessBoolVar, HeadlessVar
        self._cycle_class_vars = {}
        for cls in sorted(counts, key=lambda x: (-counts[x], x.lower())):
            n = int(counts[cls])
            if self._cycle_distribution_mode.get() == 'percent':
                tv, vv, sv = '70', '15', '15'
            else:
                tv, vv = str(int(n * .70)), str(int(n * .15))
                sv = str(n - int(n * .70) - int(n * .15))
            self._cycle_class_vars[cls] = {
                'enabled': HeadlessBoolVar(value=True), 'final': HeadlessVar(value=cls),
                'train': HeadlessVar(value=tv), 'valid': HeadlessVar(value=vv), 'test': HeadlessVar(value=sv),
                'unused': HeadlessVar(value='0'), 'available': n,
            }
        self.web_version = getattr(self, 'web_version', 0) + 1

    def _cycle_target_mode_changed(self):
        from .headless import HeadlessBoolVar
        self._cycle_target_vars = {}
        if self.project and self._cycle_target_mode.get() == 'manual':
            refs = set(self._cycle_reference_batches_for_version())
            for name in self.project.csv_names():
                if name not in refs:
                    self._cycle_target_vars[name] = HeadlessBoolVar(value=True)
        self.web_version = getattr(self, 'web_version', 0) + 1

    def _cycle_redraw_training_chart(self):
        self.web_version = getattr(self, 'web_version', 0) + 1

    def _cycle_refresh_batch_visuals(self, *a, **k):
        self.web_version = getattr(self, 'web_version', 0) + 1

    _cycle_refresh_class_visuals = _cycle_refresh_target_visuals = _cycle_refresh_batch_visuals
    _cycle_filter_batch_rows = _cycle_filter_class_rows = _cycle_refresh_batch_visuals

    def _cycle_open_evaluation(self):
        if self._cycle_trained_version and self._cycle_model_path:
            self.cycle_eval_request = (str(self._cycle_model_path.parent), str(self._cycle_trained_version))

    def prepare_clustering(self):
        """Attributes the original clustering methods expect from the tab."""
        from .headless import HeadlessDoubleVar, HeadlessVar
        if not hasattr(self, 'cluster_progress_vars'):
            self.cluster_progress_vars = {k: HeadlessDoubleVar(value=0) for k in ('generate', 'dirty', 'relabel')}
            self.cluster_status_vars = {k: HeadlessVar(value=self.tr('cluster_ready'))
                                        for k in ('generate', 'dirty', 'relabel')}
        for name in ('cluster_ax', '_dendro_canvas', 'cluster_scope_count_lbl', 'cluster_scope_classes_lbl',
                     'cluster_stats'):
            if getattr(self, name, None) is None:
                setattr(self, name, StubWidget())

    def _log(self, msg, tag='normal'):
        stamp = datetime.now().strftime('%H:%M:%S')
        self.log_lines.append((str(msg), tag, stamp))
        self.log_lines = self.log_lines[-1000:]
        seq = next(self._log_seq)
        self.log_seq = seq
        self._seq_lines.append((seq, stamp, tag, str(msg)))
        self._seq_lines = self._seq_lines[-2000:]

    def clear_log(self):
        self.log_lines.clear()
        self._seq_lines = []
        self.log_epoch += 1

    def log_since(self, seq, epoch):
        reset = epoch != self.log_epoch
        if reset:
            seq = 0
        lines = [item for item in self._seq_lines if item[0] > seq]
        return self.log_epoch, reset, lines[-1500:]

    # --- read model ----------------------------------------------------
    def op_state(self):
        return {
            'running': bool(self._op_running),
            'status': self.status_var.get(),
            'operation': self.operation_var.get(),
            'elapsed': self.elapsed_var.get(),
            'pct': float(self.progress_var.get() or 0),
            'pct_text': self.progress_text.get(),
            'project': self.project.name if self.project else '',
            'batch': self.vars['batch'].get(),
            'count': len(self.current_df) if self.current_df is not None else 0,
            'selected': len(self.selected_gids),
            'view': self.view_version,
            'scatter': self.scatter_version,
            'gallery': self.gallery_version,
            'cluster': self.cluster_view_version,
            'seg': getattr(self, 'seg_version', 0),
            'web': getattr(self, 'web_version', 0),
            'tab': self.current_tab,
            'last_seq': self.log_seq,
            'log_epoch': self.log_epoch,
        }


_IAT = None


def get_iat():
    global _IAT
    if _IAT is None:
        ctl = get_controller()
        with ctl.lock:
            if _IAT is None:
                _IAT = HeadlessIAT(ctl)
    return _IAT


def wait_idle(iat, timeout=0.0):
    """Small helper for callbacks that want an operation's result quickly."""
    end = time.monotonic() + timeout
    while iat._op_running and time.monotonic() < end:
        time.sleep(0.02)
