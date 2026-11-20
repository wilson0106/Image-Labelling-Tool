#!/usr/bin/env python3
"""Headless dashboard controller for the web interface.

Reuses — without copying — the orchestration code of the classic interface:

* ``TrainingDatasetUIMixin``: dataset detection, per-class split allocation,
  grouping/renaming, temporary dataset view, split manifest;
* ``ILTProcessMixin``: validation of the forms, worker command lines,
  subprocess launch, log/progress parsing, classification monitor,
  per-class clustering hand-off, stop/finalize;
* selected ``ILTApp`` methods: defaults, form variables, training config,
  model/labels pair detection, config save/load, folder pickers.

Only the presentation hooks (log widget, stage bar, charts, result window)
are re-implemented here, as plain data that the web layer renders.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dashboard_execution_mixin as _exec_mod  # noqa: E402
import ilt_gui as _gui_mod  # noqa: E402
import training_dataset_ui_mixin as _ds_mod  # noqa: E402
from dashboard_execution_mixin import ILTProcessMixin  # noqa: E402
from ilt_gui import ILTApp  # noqa: E402
from platform_utils import application_dir, open_path  # noqa: E402
from training_dataset import DATASET_MODE_INVALID, SPLITS  # noqa: E402
from training_dataset_ui_mixin import TrainingDatasetUIMixin  # noqa: E402
from ui_resources import APP_TEXT, COLORS  # noqa: E402

from .headless import (  # noqa: E402
    NOTIFY, FiledialogShim, HeadlessVar, Scheduler, SeriesStore, StubTree,
    StubWidget, install_module_shims,
)

install_module_shims(_exec_mod, _gui_mod, _ds_mod)

TRAIN_STAGES = ['train_stage_prepare', 'train_stage_epochs', 'train_stage_test', 'train_stage_save']
CLASS_STAGES = ['stage_loading', 'stage_features', 'stage_tsne', 'stage_results', 'stage_clustering']
CLASS_METRICS = ['current_batch', 'batches', 'images_processed', 'total_images']


class WebController(TrainingDatasetUIMixin, ILTProcessMixin):
    # --- methods reused verbatim from the classic application -------------
    tr = ILTApp.tr
    _load_train_defaults = ILTApp._load_train_defaults
    _init_vars = ILTApp._init_vars
    _train_config = ILTApp._train_config
    detect_model_pair = ILTApp.detect_model_pair
    _class_pixel_size_is_default = ILTApp._class_pixel_size_is_default
    _refresh_model_status_language = ILTApp._refresh_model_status_language
    _refresh_dynamic_texts = ILTApp._refresh_dynamic_texts
    pick_dir = ILTApp.pick_dir
    pick_train_dataset = ILTApp.pick_train_dataset
    _recipe_var_text = staticmethod(ILTApp._recipe_var_text)
    apply_training_recipe = ILTApp.apply_training_recipe
    _train_recipe_config = ILTApp._train_recipe_config
    pick_model_folder = ILTApp.pick_model_folder
    save_train_config = ILTApp.save_train_config
    load_train_config = ILTApp.load_train_config
    _ansi_re = ILTApp._ansi_re

    _LOG_KEEP = 6000

    def __init__(self):
        self._sched = Scheduler()
        self.lock = self._sched.lock
        from . import headless
        headless.DIALOG_LOCK = self.lock
        self.results_shown_seq = 0
        with self.lock:
            self._init_state()

    # ------------------------------------------------------------------
    def _init_state(self):
        self.src_dir = SRC_DIR
        self.project_root = application_dir()
        self.input_dir = self.project_root / 'input'
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.project_root / 'output'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lang = 'pt'

        # Same attribute set as ILTApp.__init__ uses for the reused mixins.
        self.proc = None
        import queue
        self.proc_queue = queue.Queue()
        self.proc_task = None
        self.start_time = None
        self.last_output_path = None
        self.temp_train_json = None
        self.temp_train_dataset = None
        self._train_dataset_refresh_job = None
        self._train_class_dataset_key = None
        self._train_dataset_mode = DATASET_MODE_INVALID
        self._train_class_counts = {}
        self._train_class_totals = {}
        self._train_class_map_vars = {}
        self._train_class_use_vars = {}
        self._train_class_allocations = {}
        self._train_class_percentages = {}
        self._train_class_split_vars = {}
        self._train_class_unused_vars = {}
        self._train_class_allocation_errors = {}
        self._train_allocation_sync = False
        self._train_distribution_mode_current = 'percent'
        self._train_distribution_default_cache = {
            'percent': {'train': '70', 'valid': '15', 'test': '15'},
            'number': {'train': '0', 'valid': '0', 'test': '0'},
        }
        self.last_split_manifest_path = None
        self.class_monitor_info = {}
        self._stopping = False
        self.run_class_clustering_after = True
        self.clustering_total = 0
        self.clustering_seen = 0

        self._init_vars()

        # Presentation stand-ins (read by the web layer).
        self._train_class_rows_host = object()
        self._train_model_tree = StubTree()
        self._train_model_summary_var = HeadlessVar(value=self.tr('dataset_classes_wait'))
        self._train_model_summary_label = StubWidget(fg=COLORS['muted'])
        self._train_dataset_detection_var = HeadlessVar(value=self.tr('dataset_detection_wait'))
        self._train_dataset_detection_label = StubWidget(fg=COLORS['muted'])
        self._train_distribution_unit_var = HeadlessVar(value='%')
        self.model_status_var = HeadlessVar(value=self.tr('detected_wait'))
        self.model_status_label = StubWidget(fg=COLORS['muted'])
        self.run_status_var = HeadlessVar(value=self.tr('ready'))
        self.run_status_label = StubWidget(fg=COLORS['green'])
        self.system_status_var = HeadlessVar(value=self.tr('ready'))
        self.elapsed_var = HeadlessVar(value='00:00:00')
        self.progress_pct_var = HeadlessVar(value='0%')
        self.live_progress_var = HeadlessVar(value='')
        self.progress = StubWidget(value=0)
        self.stop_button = StubWidget(state='disabled')
        self.start_button = StubWidget(state='normal')
        self.nav_train = StubWidget(state='normal')
        self.nav_class = StubWidget(state='normal')
        self.nav_iat = StubWidget(state='normal')
        self.loss_chart = SeriesStore(['train_loss', 'valid_loss'])
        self.validation_chart = SeriesStore(['valid_acc', 'valid_macro'])
        self.class_metric_vars = {key: HeadlessVar(value='—') for key in CLASS_METRICS}
        self._current_train_epoch = 0
        self.expected_epochs = 1

        self.rows_version = 0
        self._log = deque(maxlen=self._LOG_KEEP)
        self._log_seq = 0
        self.log_epoch = 0
        self.results_request = None
        self.results_seq = 0
        self.set_stage_task('training')

        self._refresh_training_dataset_classes()
        self.after(100, self._poll_process_queue)
        self.after(500, self._tick)

    # --- Tk event-loop stand-ins --------------------------------------
    def after(self, ms, func=None, *args):
        return self._sched.after(ms, func, *args)

    def after_cancel(self, job):
        self._sched.after_cancel(job)

    # --- presentation hooks used by the mixins --------------------------
    def _render_training_class_rows(self):
        self.rows_version += 1

    def _configure_log_tags(self):
        return None

    def _append_log(self, text):
        text = self._ansi_re.sub('', str(text)).replace('\r', '')
        for line in text.splitlines(True):
            clean = line.rstrip('\n')
            if not clean.strip():
                continue
            tag, icon = self._log_style(clean)
            self._log_seq += 1
            self._log.append((self._log_seq, time.strftime('%H:%M:%S'), tag, icon + clean))

    def clear_log(self):
        self._log.clear()
        self.log_epoch += 1
        self.live_progress_var.set('')

    def set_stage_task(self, task):
        self.stage_task = task
        self.stage_keys = CLASS_STAGES if task == 'classification' else TRAIN_STAGES
        self.stage_states = ['pending'] * len(self.stage_keys)
        self.stage_index = -1

    def _set_stage(self, index):
        self.stage_index = index
        self.stage_states = [
            'done' if i < index else 'active' if i == index else 'pending'
            for i in range(len(self.stage_keys))
        ]

    def _complete_stages(self):
        self.stage_states = ['done'] * len(self.stage_keys)
        self.stage_index = len(self.stage_keys)

    def _skip_last_stage(self):
        if self.stage_keys:
            self.stage_states = ['done'] * (len(self.stage_keys) - 1) + ['skipped']

    def open_training_results(self):
        output = Path(
            getattr(self, '_last_training_output', None)
            or self.v['train_output'].get().strip()
        ).expanduser()
        project = str(
            getattr(self, '_last_training_project', None)
            or self.v['train_project'].get().strip()
        ).strip()
        if (not project or not (output / f'matrix_{project}.csv').is_file()
                or not (output / f'labels_{project}.json').is_file()):
            NOTIFY.push('info', self.tr('training_results'),
                        self.tr('training_results_not_found').format(project=project or '?'))
            return
        self.results_seq += 1
        self.results_request = (self.results_seq, str(output), project)

    def open_old_log(self):
        p = FiledialogShim.askopenfilename(
            title=self.tr('select_log'), initialdir=str(self.project_root / 'output'),
            filetypes=[('Log', '*.log'), ('Text', '*.txt'), ('All', '*')])
        if not p:
            return None
        try:
            return Path(p).name, Path(p).read_text(errors='replace')
        except Exception as exc:
            NOTIFY.push('error', self.tr('error'), str(exc))
            return None

    def open_output(self):
        p = self.last_output_path
        if p and Path(p).exists():
            try:
                open_path(p)
            except Exception:
                pass

    def open_folder(self, path):
        try:
            if path and Path(path).expanduser().exists():
                open_path(Path(path).expanduser())
        except Exception:
            pass

    # --- actions called by the web layer (always under ``lock``) --------
    def set_language(self, lang):
        self.lang = 'en' if lang == 'en' else 'pt'
        self._refresh_dynamic_texts()
        self._refresh_train_dataset_detection_text()
        self._refresh_training_model_summary()
        if self._train_distribution_unit_var is not None:
            self._train_distribution_unit_var.set(
                '%' if self._train_distribution_mode_var.get() == 'percent' else self.tr('images_unit'))

    def set_var(self, key, value):
        var = self.v.get(key)
        if var is None:
            return
        if isinstance(var.get(), bool):
            value = bool(value)
        else:
            value = '' if value is None else str(value)
        if var.get() != value:
            var.set(value)

    def refresh_dataset_now(self):
        if self._train_dataset_refresh_job is not None:
            self.after_cancel(self._train_dataset_refresh_job)
            self._train_dataset_refresh_job = None
        self._refresh_training_dataset_classes()

    def set_class_field(self, class_name, field, value):
        if field == 'use':
            var = self._train_class_use_vars.get(class_name)
            if var is not None and bool(var.get()) != bool(value):
                var.set(bool(value))
        elif field == 'target':
            var = self._train_class_map_vars.get(class_name)
            if var is not None and var.get() != str(value or ''):
                var.set(str(value or ''))
        elif field in SPLITS:
            var = self._train_class_split_vars.get(class_name, {}).get(field)
            if var is not None and var.get() != str(value or ''):
                var.set(str(value or ''))

    def set_distribution_mode(self, mode):
        if mode in ('percent', 'number') and self._train_distribution_mode_var.get() != mode:
            self._train_distribution_mode_var.set(mode)
            self.rows_version += 1

    def set_distribution_default(self, split, value):
        var = self._train_distribution_default_vars.get(split)
        if var is not None:
            var.set(str(value or ''))

    def apply_distribution_all(self):
        self._apply_training_distribution_to_all()
        self.rows_version += 1

    def reset_class_names(self):
        self._reset_training_class_names()
        self.rows_version += 1

    def start(self, task, existing=None):
        if self.proc is not None:
            return
        self.set_stage_task(task)
        if task == 'training':
            self.start_training(existing_decision=existing)
        else:
            self.start_classification()

    def stop(self):
        self.stop_process()

    # --- read models for the web layer ---------------------------------
    def form_values(self):
        return {key: var.get() for key, var in self.v.items()}

    def class_rows(self):
        rows = []
        for name in self._train_class_counts:
            splits = self._train_class_split_vars.get(name, {})
            rows.append({
                'name': name,
                'total': int(self._train_class_totals.get(name, 0) or 0),
                'use': bool(self._train_class_use_vars[name].get()) if name in self._train_class_use_vars else True,
                'target': self._train_class_map_vars[name].get() if name in self._train_class_map_vars else name,
                'splits': {s: (splits[s].get() if s in splits else '') for s in SPLITS},
                'unused': self._train_class_unused_vars[name].get() if name in self._train_class_unused_vars else '',
                'error': name in self._train_class_allocation_errors,
            })
        return rows

    def dataset_state(self):
        return {
            'mode': self._train_dataset_mode,
            'detection': self._train_dataset_detection_var.get(),
            'detection_color': self._train_dataset_detection_label.state.get('fg', COLORS['muted']),
            'summary': self._train_model_summary_var.get(),
            'summary_color': self._train_model_summary_label.state.get('fg', COLORS['muted']),
            'preview': list(self._train_model_tree.rows),
            'distribution_mode': self._train_distribution_mode_var.get(),
            'defaults': {s: self._train_distribution_default_vars[s].get() for s in SPLITS},
            'unit': self._train_distribution_unit_var.get(),
            'rows_version': self.rows_version,
        }

    def model_state(self):
        return {
            'status': self.model_status_var.get(),
            'color': self.model_status_label.state.get('fg', COLORS['muted']),
            'model_file': self.v['model_file'].get(),
            'labels_file': self.v['labels_file'].get(),
        }

    def log_since(self, seq, epoch):
        """Return (epoch, reset, lines) newer than *seq* for the live log."""
        reset = epoch != self.log_epoch
        if reset:
            seq = 0
        lines = [item for item in self._log if item[0] > seq]
        if len(lines) > 1500:  # the client never needs more than one screenful burst
            lines = lines[-1500:]
            reset = True
        return self.log_epoch, reset, lines

    def exec_state(self):
        running = self.proc is not None
        return {
            'running': running,
            'task': self.proc_task or self.stage_task,
            'status': self.run_status_var.get(),
            'status_color': self.run_status_label.state.get('fg', COLORS['green']),
            'system': self.system_status_var.get(),
            'elapsed': self.elapsed_var.get(),
            'pct': float(self.progress['value'] or 0),
            'live': self.live_progress_var.get(),
            'stages': [[self.tr(k), s] for k, s in zip(self.stage_keys, self.stage_states)],
            'metrics': {k: [self.tr(k), v.get()] for k, v in self.class_metric_vars.items()},
            'charts_version': self.loss_chart.version * 100003 + self.validation_chart.version,
            'last_seq': self._log_seq,
            'log_epoch': self.log_epoch,
        }

    def chart_data(self):
        return {'loss': self.loss_chart.as_lists(), 'valid': self.validation_chart.as_lists()}


_CONTROLLER = None
_CONTROLLER_LOCK = threading.Lock()


def get_controller():
    global _CONTROLLER
    with _CONTROLLER_LOCK:
        if _CONTROLLER is None:
            _CONTROLLER = WebController()
        return _CONTROLLER


def text(lang, key):
    return APP_TEXT.get(lang, APP_TEXT['pt']).get(key, key)
