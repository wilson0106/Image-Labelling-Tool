#!/usr/bin/env python3
"""Process orchestration mixin for the ILT dashboard.

Training/classification subprocess management is kept separate from Tk page
construction.  The mixin operates on the same ``ILTApp`` state and does not
change the scientific pipeline.
"""
from __future__ import annotations

import csv
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox

from platform_utils import open_path, worker_command
from project_settings import format_pixel_size, parse_pixel_size
from ui_resources import COLORS
from training_dataset import DATASET_MODE_INVALID


class ILTProcessMixin:
    def training_existing_model_state(self):
        """Return ``(state, model_path)`` for the configured output/project.

        *state* is ``'none'`` (no model), ``'partial'`` (``model_<p>_last_epoch``
        exists, training can be resumed) or ``'complete'``.
        """
        output = Path(self.v['train_output'].get().strip() or '.').expanduser()
        project = self.v['train_project'].get().strip()
        if not project:
            return 'none', None
        model = output / f'model_{project}'
        if not model.exists():
            return 'none', model
        partial = (output / f'model_{project}_last_epoch').exists()
        return ('partial' if partial else 'complete'), model

    def training_conflict(self):
        """``(state, model_path)`` when starting needs a user decision, else None."""
        state, model = self.training_existing_model_state()
        if state == 'none':
            return None
        if state == 'partial' and self.v['readModel'].get() == 'T':
            return None  # resumes automatically, as before
        return state, model

    def start_training(self, existing_decision=None):
        dataset_text = self.v['train_dataset'].get().strip()
        output_text = self.v['train_output'].get().strip()
        project = self.v['train_project'].get().strip()
        if not dataset_text or not output_text or not project:
            messagebox.showerror(self.tr('error'), self.tr('fill_required')); return
        dataset = Path(dataset_text).expanduser()
        output = Path(output_text).expanduser()
        if not dataset.is_dir():
            messagebox.showerror(self.tr('error'), self.tr('dataset_missing')); return
        try:
            output.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            messagebox.showerror(self.tr('error'), str(e)); return
        try:
            cfg = self._train_config()
        except ValueError as e:
            messagebox.showerror(self.tr('error'), str(e)); return

        # An existing model_<project> used to make train.py wait on a console
        # prompt (input()) that the dashboard could never answer.  The decision
        # is taken here and passed in the configuration instead.
        decision = str(existing_decision or '').strip().lower()
        conflict = self.training_conflict()
        if conflict is not None and not decision:
            state, model_path = conflict
            msg = self.tr('model_exists_msg').format(path=model_path)
            if state == 'partial':
                msg += '\n' + self.tr('model_exists_partial_hint')
            if not messagebox.askyesno(self.tr('model_exists_title'), msg):
                return
            decision = 'overwrite'
        if decision == 'overwrite':
            cfg['readModel'] = 'F'
            self.v['readModel'].set('F')
        elif decision == 'resume':
            cfg['readModel'] = 'T'
            self.v['readModel'].set('T')
        if decision:
            cfg['existingModel'] = decision

        # Refresh immediately before launch so the visual plan, CSV audit, and
        # dataset actually passed to train.py describe the same source files.
        self._refresh_training_dataset_classes()
        if self._train_dataset_mode == DATASET_MODE_INVALID:
            messagebox.showerror(
                self.tr('error'), self.tr('train_structure_missing')
            )
            return

        mapping, summary, errors = self._training_class_plan()
        if not mapping:
            messagebox.showerror(
                self.tr('error'), self.tr('model_classes_empty')
            )
            return
        if errors:
            messagebox.showerror(self.tr('error'), errors[0])
            return

        try:
            assignments = self._build_training_assignments(dataset, cfg, mapping)
        except Exception as e:
            messagebox.showerror(self.tr('error'), str(e))
            return
        if not assignments:
            messagebox.showerror(
                self.tr('error'), self.tr('model_classes_empty')
            )
            return

        dataset_for_training = dataset
        if self._training_plan_changes_dataset(mapping):
            try:
                dataset_for_training = self._prepare_training_dataset(
                    dataset, cfg, mapping, summary, assignments
                )
            except Exception as e:
                self._cleanup_temp_train_dataset()
                messagebox.showerror(self.tr('error'), str(e))
                return
        else:
            self._cleanup_temp_train_dataset()

        try:
            manifest = self._write_training_split_manifest(
                assignments, output, project
            )
        except Exception as e:
            self._cleanup_temp_train_dataset()
            messagebox.showerror(self.tr('error'), str(e))
            return

        temp_dir=Path(tempfile.gettempdir())/'ILT_Data_Extraction'
        temp_dir.mkdir(parents=True,exist_ok=True)
        temp=temp_dir/'.ilt_gui_train_params.json'
        temp.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
        self.temp_train_json=temp
        cmd = worker_command(
            'training', str(dataset_for_training), str(output),
            project, str(temp), src_dir=self.src_dir
        )
        self._reset_execution('training')
        self.last_output_path = output
        self._last_training_output = output
        self._last_training_project = project
        self.expected_epochs = cfg['numEpochs']
        self._launch(cmd, 'training')
        if self.proc is None:
            self._cleanup_temp_train_dataset()
        else:
            try:
                self._append_log(
                    self.tr('split_manifest_created').format(path=manifest) + '\n'
                )
            except Exception:
                pass

    def start_classification(self):
        inp = Path(self.v['class_input'].get()).expanduser()
        parent = Path(self.v['class_output_parent'].get()).expanduser()
        project = self.v['class_project'].get().strip()
        if not inp.is_dir():
            messagebox.showerror(self.tr('error'), self.tr('dataset_missing')); return
        if not parent.is_dir():
            messagebox.showerror(self.tr('error'), self.tr('output_missing')); return
        if not project:
            messagebox.showerror(self.tr('error'), self.tr('fill_required')); return
        model, labels = self.detect_model_pair(self.v['model_folder'].get()) or ('','')
        if not model or not labels:
            messagebox.showerror(self.tr('error'), self.tr('model_pair_missing')); return
        final = parent / project
        if final.exists():
            messagebox.showerror(self.tr('error'), self.tr('output_exists')); return
        try:
            batch = int(self.v['max_batch'].get())
            if batch < 1: raise ValueError
            pixel_size = parse_pixel_size(self.v['class_pixel_size'].get())
        except ValueError:
            messagebox.showerror(self.tr('error'), self.tr('invalid_input')); return
        self.v['class_pixel_size'].set(format_pixel_size(pixel_size))
        env = os.environ.copy()
        env['ILT_BATCH_MAX_SIZE'] = str(batch)
        env['ILT_PIXEL_SIZE_UM_PER_PX'] = format_pixel_size(pixel_size)
        env['ILT_IMG_MODE'] = self.v['img_mode'].get()
        env['ILT_AUDIT_SOURCE_LABELS'] = '1' if self.v['audit_source_labels'].get() else '0'
        cmd = worker_command('classification', str(inp), str(final), model, labels, src_dir=self.src_dir)
        self._reset_execution('classification')
        self.last_output_path = final
        self.class_monitor_info = {
            'output': final, 'total_batches': 0, 'batch_sizes': {}, 'total_images': 0,
            'batch_max': batch, 'batch_csv_seen_sig': None, 'batch_csv_stable_ticks': 0,
            'batch_csv_loaded_sig': None,
        }
        self.run_class_clustering_after = bool(self.v['run_class_clustering'].get())
        self.clustering_total = 0
        self.clustering_seen = 0
        self._launch(cmd, 'classification', env=env)

    def _reset_execution(self, task):
        self.clear_log()
        if hasattr(self, 'live_progress_var'):
            self.live_progress_var.set('')
        self.progress['value'] = 0
        self.progress_pct_var.set('0%')
        self._last_result = None
        if task == 'training':
            self._set_stage(0)
            self._current_train_epoch = 0
            self.loss_chart.reset()
            self.validation_chart.reset()
        else:
            self._set_stage(0)
            for v in self.class_metric_vars.values(): v.set('—')

    def _launch(self, cmd, task, env=None, cwd=None, preserve_timer=False, announce=True):
        if self.proc is not None:
            return
        env = env or os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        # Workers never get a console: stdin is closed and the worker is told
        # so, hence no prompt can block the run (train.py existing-model check).
        env['ILT_NONINTERACTIVE'] = '1'
        try:
            self.proc = subprocess.Popen(cmd, cwd=str(cwd or self.src_dir), stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1, env=env)
        except Exception as e:
            self.proc = None
            messagebox.showerror(self.tr('error'), str(e)); return
        self.proc_task = task
        if not preserve_timer or self.start_time is None:
            self.start_time = time.time()
        self._stopping = False
        self.run_status_var.set(self.tr('running'))
        self.run_status_label.configure(fg=COLORS['green'])
        self.system_status_var.set(self.tr('running'))
        self.stop_button.configure(state='normal')
        self.start_button.configure(state='disabled')
        self.nav_train.configure(state='disabled')
        self.nav_class.configure(state='disabled')
        self.nav_iat.configure(state='disabled')
        if announce:
            self._append_log(self._translate_dynamic('Process started.') + '\n')
        proc = self.proc
        threading.Thread(target=self._reader_thread, args=(proc,), daemon=True).start()

    def _reader_thread(self, proc):
        try:
            for line in iter(proc.stdout.readline, ''):
                if not line:
                    break
                self.proc_queue.put(('line', line))
            rc = proc.wait()
            self.proc_queue.put(('done', rc))
        except Exception as e:
            self.proc_queue.put(('error', str(e)))

    # Upper bound of subprocess lines handled per Tk tick.  A burst of output
    # (e.g. fast tqdm updates) is spread over several ticks instead of freezing
    # the interface; nothing is dropped.
    _POLL_MAX_LINES = 400
    # Keep the live log bounded so multi-day trainings do not slow Tk down.
    _LOG_MAX_LINES = 6000

    def _poll_process_queue(self):
        handled = 0
        try:
            while handled < self._POLL_MAX_LINES:
                handled += 1
                kind, payload = self.proc_queue.get_nowait()
                if kind == 'line':
                    if self._is_tqdm_line(payload):
                        self._update_live_progress(payload)
                    else:
                        self._append_log(self._translate_dynamic(payload))
                    if self.proc_task == 'training':
                        self._parse_training_line(payload)
                    elif self.proc_task == 'classification':
                        self._parse_classification_line(payload)
                    elif self.proc_task == 'clustering':
                        self._parse_clustering_line(payload)
                elif kind == 'done':
                    self._process_done(payload)
                elif kind == 'error':
                    self._append_log(payload + '\n')
        except queue.Empty:
            pass
        self.after(20 if handled >= self._POLL_MAX_LINES else 100, self._poll_process_queue)

    def _clean_terminal_line(self, line):
        return self._ansi_re.sub('', str(line)).replace('\r', '').strip('\n')

    def _is_tqdm_line(self, line):
        clean = self._clean_terminal_line(line).strip()
        return bool(re.search(r'\b\d{1,3}%\|.*?\|\s*\d+/\d+\s*\[', clean))

    def _update_live_progress(self, line):
        if not hasattr(self, 'live_progress_var'):
            return
        clean = self._clean_terminal_line(line).strip()
        if self.proc_task == 'training' and getattr(self, 'stage_index', -1) < 1:
            self._set_stage(1)
        m = re.search(r'(\d{1,3})%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]', clean)
        if not m:
            self.live_progress_var.set(clean)
            return
        pct, cur, total, details = m.groups()
        detail_parts = [x.strip() for x in details.split(',') if x.strip()]
        speed = detail_parts[-1] if detail_parts else ''
        text = f"{self.tr('internal_progress')}: {cur} / {total}  •  {pct}%"
        if speed:
            text += f"  •  {speed}"
        self.live_progress_var.set(text)

    def _translate_dynamic(self, line):
        if self.lang == 'en':
            replacements = [
                ('--class ausente -> rodando para', '--class not provided -> running for'),
                ('Falhou na classe', 'Failed for class'),
                ('linhas', 'rows'),
                ('centroides', 'centroids'),
                ('não disponível -> pulando PCA', 'not available -> skipping PCA'),
                ('ajustado: pedido=', 'adjusted: requested='),
                (' -> usando=', ' -> using='),
            ]
            out = line
            for a, b in replacements:
                out = out.replace(a, b)
            return out
        replacements = [
            ('Process started.', 'Processo iniciado.'),
            ('Creating batches...', 'Criando batches...'),
            ('Done creating batches!', 'Criação dos batches concluída!'),
            ('Creating image symlinks', 'Preparando imagens'),
            ('Computing base features...', 'Extraindo features do batch base...'),
            ('Computing base projections...', 'Preparando referência interna do batch base...'),
            ('Computing all features/projections...', 'Processando features e resultados de todos os batches...'),
            ('Classification batches:', 'Batches da classificação:'),
            ('Labeling predictions...', 'Aplicando rótulos às predições...'),
            ('Total time:', 'Tempo total:'),
            ('Num Classes:', 'Número de classes:'),
            ('Weights:', 'Modelo:'),
            ('Labels:', 'Rótulos:'),
            ('Finished', 'Finalizado'),
        ]
        out = line
        for a,b in replacements:
            out = out.replace(a,b)
        return out

    def _configure_log_tags(self):
        if not hasattr(self, 'log_text'):
            return
        self.log_text.tag_configure('normal', foreground='#e8e8ea', spacing1=1, spacing3=1)
        self.log_text.tag_configure('muted', foreground='#7f93aa', spacing1=1, spacing3=1)
        self.log_text.tag_configure('progress', foreground=COLORS['cyan'], spacing1=1, spacing3=1)
        self.log_text.tag_configure('epoch', foreground='#79a9ff', spacing1=2, spacing3=1)
        self.log_text.tag_configure('validation', foreground='#6ee7a8', spacing1=1, spacing3=1)
        self.log_text.tag_configure('success', foreground=COLORS['green'], spacing1=1, spacing3=1)
        self.log_text.tag_configure('warning', foreground=COLORS['amber'], spacing1=1, spacing3=1)
        self.log_text.tag_configure('error', foreground=COLORS['red'], spacing1=1, spacing3=1)
        self.log_text.tag_configure('path', foreground=COLORS['purple'], spacing1=1, spacing3=1)

    def _log_style(self, line):
        low = line.lower().strip()
        if any(k in low for k in ('traceback', 'error', 'erro', 'failed', 'falhou', 'exception')):
            return 'error', '✕ '
        if any(k in low for k in ('warning', 'aviso', 'not found', 'não encontrado', 'deprecated')):
            return 'warning', '⚠ '
        if any(k in low for k in ('done', 'conclu', 'finished', 'finalizado', 'saved', 'salv', 'best model', 'melhor modelo')):
            return 'success', '✓ '
        if low.startswith('valid acc:') or low.startswith('validação'):
            return 'validation', '◆ '
        if low.startswith('train acc:'):
            return 'progress', '◆ '
        if re.search(r'\b(epoch|época)\s+\d+', low):
            return 'epoch', '● '
        if any(k in low for k in ('batch', 'comput', 'extraindo', 'creating', 'criando', 'preparando', 'labeling', '[clustering]', '[pca]', '[centroid]', '[full]', '[index]', '[embed]')):
            return 'progress', '• '
        path_like = bool(re.search(r'(^|\s)(/[^\s]+|[A-Za-z]:[\\/][^\s]+)', line))
        if path_like or any(k in low for k in ('weights:', 'modelo:', 'labels:', 'rótulos:', 'output:', 'saída:')):
            return 'path', ''
        if not line.strip():
            return 'muted', ''
        return 'normal', ''

    def _append_log(self, text):
        if not hasattr(self, 'log_text'):
            return
        self.log_text.configure(state='normal')
        text = self._ansi_re.sub('', str(text)).replace('\r', '')
        for line in text.splitlines(True):
            clean = line.rstrip('\n')
            tag, icon = self._log_style(clean)
            if clean.strip():
                stamp = time.strftime('%H:%M:%S')
                self.log_text.insert('end', f'[{stamp}] ', 'muted')
                self.log_text.insert('end', icon + clean + '\n', tag)
        try:
            lines = int(self.log_text.index('end-1c').split('.')[0])
            if lines > self._LOG_MAX_LINES:
                self.log_text.delete('1.0', f'{lines - self._LOG_MAX_LINES}.0')
        except Exception:
            pass
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def clear_log(self):
        if hasattr(self, 'log_text'):
            self.log_text.configure(state='normal')
            self.log_text.delete('1.0','end')
            self.log_text.configure(state='disabled')
        if hasattr(self, 'live_progress_var'):
            self.live_progress_var.set('')

    def _set_classification_plan(self,total_images,total_batches=None):
        """Update classification metrics without scanning an 8M-row batches.csv."""
        try:total_images=max(0,int(total_images))
        except Exception:return
        if total_batches is None:
            try:
                max_batch=max(1,int(self.class_monitor_info.get('batch_max') or self.v['max_batch'].get()))
                total_batches=(total_images+max_batch-1)//max_batch if total_images else 0
            except Exception:
                total_batches=0
        try:total_batches=max(0,int(total_batches))
        except Exception:total_batches=0
        info=self.class_monitor_info
        info['total_images']=total_images; info['total_batches']=total_batches
        if total_batches:
            per=(total_images+total_batches-1)//total_batches
            remaining=total_images; sizes={}
            for i in range(1,total_batches+1):
                b=f'batch_{i:04d}'; n=min(per,remaining)
                sizes[b]=max(0,n); remaining-=n
            info['batch_sizes']=sizes
            info['batch_order']=list(sizes.keys())
        if hasattr(self,'class_metric_vars'):
            self.class_metric_vars['batches'].set(f"0 / {total_batches}" if total_batches else '—')
            self.class_metric_vars['total_images'].set(f'{total_images:,}'.replace(',','.'))

    def _parse_classification_line(self,line):
        clean=self._clean_terminal_line(line).strip()
        # This line is emitted immediately after the recursive scan, so the GUI
        # can show the correct totals while batches.csv is still being written.
        # batch.py prints "Image scan complete: N images ..." (older builds
        # printed "PNG scan complete"); accept both so the totals come from the
        # log instead of parsing a multi-million-row batches.csv on the Tk thread.
        m=re.search(r'(?:PNG|Image) scan complete:\s*(\d+)\s+images',clean,flags=re.I)
        if m:
            self._set_classification_plan(int(m.group(1)))
            return
        m=re.search(r'Done creating batches:\s*(\d+)\s+(?:PNGs?|images?)\s+in\s+(\d+)\s+batches',clean,flags=re.I)
        if m:
            self._set_classification_plan(int(m.group(1)),int(m.group(2)))
            return
        m=re.search(r'Classification batches:\s*(\d+)',clean,flags=re.I)
        if m:
            total=int(m.group(1)); info=self.class_monitor_info
            if info.get('total_images'):
                self._set_classification_plan(info['total_images'],total)
            else:
                info['total_batches']=total

    def _parse_training_line(self, line):
        if self.proc_task == 'training' and self.stage_index < 1 and (self._is_tqdm_line(line) or 'Epoch' in line):
            self._set_stage(1)
        m = re.search(r'Epoch\s+(\d+)', line)
        if m:
            epoch = int(m.group(1))
            self._current_train_epoch = epoch
            pct = min(100, 100 * epoch / max(1, self.expected_epochs))
            self.progress['value'] = pct
            self.progress_pct_var.set(f'{pct:.0f}%')

        epoch = max(1, getattr(self, '_current_train_epoch', 1))
        m = re.search(r'train acc:\s*([0-9.]+),\s*loss:\s*([0-9.]+).*macro:\s*([0-9.]+)', line)
        if m:
            train_loss = float(m.group(2))
            self.loss_chart.add_point('train_loss', epoch, train_loss)

        if re.search(r'^test acc:', line.strip(), flags=re.I):
            self._set_stage(2)
            self.progress['value'] = max(float(self.progress['value']), 97)
            self.progress_pct_var.set(f"{float(self.progress['value']):.0f}%")
        if 'Finished' in line or 'Finalizado' in line:
            self._set_stage(3)
            self.progress['value'] = max(float(self.progress['value']), 99)
            self.progress_pct_var.set('99%')

        m = re.search(r'valid acc:\s*([0-9.]+),\s*loss:\s*([0-9.]+).*macro:\s*([0-9.]+)', line)
        if m:
            acc, loss, macro = map(float, m.groups())
            self.loss_chart.add_point('valid_loss', epoch, loss)
            self.validation_chart.add_point('valid_acc', epoch, acc * 100.0)
            self.validation_chart.add_point('valid_macro', epoch, macro * 100.0)

    def _parse_clustering_line(self, line):
        clean = self._clean_terminal_line(line).strip()
        m = re.search(r'rodando para\s+(\d+)\s+classes', clean, flags=re.I)
        if not m:
            m = re.search(r'running for\s+(\d+)\s+classes', clean, flags=re.I)
        if m:
            self.clustering_total = int(m.group(1))
            self.clustering_seen = 0
            return
        m = re.search(r'\[clustering\]\s+class=(.+)$', clean, flags=re.I)
        if m:
            self.clustering_seen += 1
            total = max(self.clustering_total, self.clustering_seen)
            pct_local = 100.0 * self.clustering_seen / max(1, total)
            label = m.group(1).strip()
            self.live_progress_var.set(f"{self.tr('cluster_progress')}: {self.clustering_seen} / {total}  •  {pct_local:.0f}%  •  {label}")
            overall = 90.0 + 9.0 * self.clustering_seen / max(1, total)
            self.progress['value'] = min(99, overall)
            self.progress_pct_var.set(f'{min(99, overall):.0f}%')
        m = re.search(r'\[done\].*total=(\d+)', clean, flags=re.I)
        if m:
            total = int(m.group(1))
            self.live_progress_var.set(f"{self.tr('cluster_progress')}: {total} / {total}  •  100%")
            self.progress['value'] = 99
            self.progress_pct_var.set('99%')

    def _start_class_clustering(self):
        out = Path(self.class_monitor_info.get('output', ''))
        project = self.v['class_project'].get().strip() or out.name
        script = self.src_dir / 'class_clustering.py'
        if not getattr(sys,'frozen',False) and not script.is_file():
            self._append_log(f"class_clustering.py not found: {script}\n")
            self._finalize_process(1, finished_task='clustering')
            return
        self._set_stage(4)
        self.progress['value'] = 90
        self.progress_pct_var.set('90%')
        self.live_progress_var.set(self.tr('stage_clustering'))
        self._append_log('────────────────────────────────────────\n')
        if self.lang == 'pt':
            self._append_log('Classificação concluída. Iniciando clustering por classe...\n')
        else:
            self._append_log('Classification completed. Starting per-class clustering...\n')
        cmd = worker_command('clustering', project, '--project-dir', str(out), src_dir=self.src_dir)
        self._launch(cmd, 'clustering', preserve_timer=True, announce=False)

    def _tick(self):
        if self.proc is not None and self.start_time:
            sec = int(time.time()-self.start_time)
            self.elapsed_var.set(time.strftime('%H:%M:%S', time.gmtime(sec)))
            if self.proc_task == 'classification':
                self._monitor_classification()
        self.after(500, self._tick)

    def _monitor_classification(self):
        info = self.class_monitor_info
        out = Path(info.get('output',''))
        if not out.exists():
            self._set_stage(0); return
        batches_csv = out/'batches.csv'

        # Normal path: totals come from the classifier log and batch sizes are
        # derived mathematically, avoiding a synchronous scan of millions of CSV
        # rows on Tk's UI thread. Fallback parsing is only used for older main.py
        # versions that do not emit the expected progress lines, and only after
        # the file has stopped changing.
        if batches_csv.exists() and not info.get('batch_sizes'):
            try:
                st=batches_csv.stat(); sig=(st.st_size,st.st_mtime_ns)
                if sig==info.get('batch_csv_seen_sig'):
                    info['batch_csv_stable_ticks']=info.get('batch_csv_stable_ticks',0)+1
                else:
                    info['batch_csv_seen_sig']=sig; info['batch_csv_stable_ticks']=0
                if info.get('batch_csv_stable_ticks',0)>=2 and sig!=info.get('batch_csv_loaded_sig'):
                    sizes={}; total=0
                    with batches_csv.open(newline='') as f:
                        for row in csv.DictReader(f):
                            b=row.get('batch','')
                            if b:
                                sizes[b]=sizes.get(b,0)+1; total+=1
                    if sizes:
                        info['batch_sizes']=sizes; info['batch_order']=sorted(sizes.keys())
                        info['total_batches']=len(sizes); info['total_images']=total
                        info['batch_csv_loaded_sig']=sig
            except Exception:
                pass
        total_batches = info.get('total_batches',0)
        total_images = info.get('total_images',0)
        if total_batches:
            self.class_metric_vars['batches'].set(f"0 / {total_batches}")
            self.class_metric_vars['total_images'].set(f'{total_images:,}'.replace(',','.'))
        df_dir = out/'dataframes'
        emb_dir = out/'embeddings'
        df_files = list(df_dir.glob('batch_*.csv')) if df_dir.exists() else []
        emb_files = list(emb_dir.glob('batch_*.npz')) if emb_dir.exists() else []
        valid_batch_ids = set(info.get('batch_sizes', {}).keys())
        batch_order = info.get('batch_order') or sorted(valid_batch_ids)
        done_ids = set()
        for p in df_files:
            m = re.match(r'(batch_\d{4})_', p.name)
            if m:
                batch_id = m.group(1)
                if not valid_batch_ids or batch_id in valid_batch_ids:
                    done_ids.add(batch_id)
        if valid_batch_ids:
            done_ids &= valid_batch_ids
        done = min(len(done_ids), total_batches) if total_batches else len(done_ids)
        processed = sum(info.get('batch_sizes',{}).get(b,0) for b in done_ids)
        if total_batches:
            self.class_metric_vars['batches'].set(f'{done} / {total_batches}')
            self.class_metric_vars['images_processed'].set(f'{processed:,}'.replace(',','.'))
            next_batch = next((b for b in batch_order if b not in done_ids), None)
            self.class_metric_vars['current_batch'].set(next_batch or '—')
        if not batches_csv.exists():
            idx, pct = 0, 2
        elif total_batches and done == 0 and len(emb_files) == 0:
            idx, pct = 1, 10
        elif total_batches and done < total_batches:
            idx = 2
            pct = 15 + 65*(done/total_batches)
        elif total_batches and done >= total_batches and not (out/'complete.csv').exists():
            idx, pct = 3, 85
        elif (out/'complete.csv').exists():
            idx, pct = 3, (90 if self.run_class_clustering_after else 100)
        else:
            idx, pct = 3, 85
        self._set_stage(idx)
        self.progress['value'] = pct
        self.progress_pct_var.set(f'{pct:.0f}%')

    def _process_done(self, rc):
        finished_task = self.proc_task
        was_stopping = self._stopping
        self.proc = None

        if rc == 0 and not was_stopping and finished_task == 'classification' and self.run_class_clustering_after:
            self._start_class_clustering()
            return

        self._finalize_process(rc, finished_task=finished_task, was_stopping=was_stopping)

    def _finalize_process(self, rc, finished_task=None, was_stopping=False):
        self.stop_button.configure(state='disabled')
        self.start_button.configure(state='normal')
        self.nav_train.configure(state='normal')
        self.nav_class.configure(state='normal')
        self.nav_iat.configure(state='normal')
        self.system_status_var.set(self.tr('ready'))

        if rc == 0 and not was_stopping:
            self._last_result = 'completed'
            self.run_status_var.set(self.tr('completed'))
            self.run_status_label.configure(fg=COLORS['green'])
            self.progress['value'] = 100
            self.progress_pct_var.set('100%')
            if finished_task == 'classification' and not self.run_class_clustering_after:
                self._skip_last_stage()
            else:
                self._complete_stages()
            if hasattr(self, 'live_progress_var'):
                self.live_progress_var.set('')
        elif was_stopping:
            self._last_result = 'stopped'
            self.run_status_var.set(self.tr('stopped'))
            self.run_status_label.configure(fg=COLORS['amber'])
        else:
            self._last_result = 'failed'
            self.run_status_var.set(self.tr('failed'))
            self.run_status_label.configure(fg=COLORS['red'])
            if finished_task == 'clustering':
                if self.lang == 'pt':
                    self._append_log('A classificação foi concluída, mas o clustering por classe falhou.\n')
                else:
                    self._append_log('Classification completed, but per-class clustering failed.\n')
        if rc == 0 and not was_stopping and finished_task == 'training':
            # train.py has already persisted matrix/labels/evaluation artifacts.
            # Open a non-modal scientific summary automatically; the same view
            # can be reopened later from the Training page.
            try:
                self.after(160, self.open_training_results)
            except Exception:
                pass
        if self.temp_train_json and self.temp_train_json.exists():
            try: self.temp_train_json.unlink()
            except Exception: pass
        if finished_task == 'training':
            self._cleanup_temp_train_dataset()
        self.proc_task = None

    def stop_process(self):
        if self.proc is None: return
        if not messagebox.askyesno(self.tr('stop'), self.tr('confirm_stop')):
            return
        self._stopping = True
        try:
            self.proc.terminate()
        except Exception:
            pass

    def open_training_results(self):
        """Open the latest test-set evaluation for the configured training run."""
        output = Path(
            getattr(self, '_last_training_output', None)
            or self.v.get('train_output').get().strip()
        ).expanduser()
        project = str(
            getattr(self, '_last_training_project', None)
            or self.v.get('train_project').get().strip()
        ).strip()
        if not output or not project:
            messagebox.showinfo(self.tr('training_results'), self.tr('training_results_not_found').format(project=project or '?'))
            return
        matrix_path = output / f'matrix_{project}.csv'
        labels_path = output / f'labels_{project}.json'
        if not matrix_path.is_file() or not labels_path.is_file():
            messagebox.showinfo(
                self.tr('training_results'),
                self.tr('training_results_not_found').format(project=project)
            )
            return
        try:
            from training_results import open_training_results as _open_results
            win = _open_results(
                self, output, project, lang=self.lang,
                open_folder_callback=lambda: open_path(output),
            )
            self._training_results_window = win
        except Exception as exc:
            messagebox.showerror(self.tr('error'), str(exc))

    def open_output(self):
        p = self.last_output_path
        if not p or not Path(p).exists():
            return
        try:
            open_path(p)
        except Exception:
            pass

    def open_old_log(self):
        p = filedialog.askopenfilename(title=self.tr('select_log'), initialdir=str(self.project_root/'output'), filetypes=[('Log','*.log'),('Text','*.txt'),('All','*')])
        if not p: return
        try:
            text = Path(p).read_text(errors='replace')
        except Exception as e:
            messagebox.showerror(self.tr('error'), str(e)); return
        win = tk.Toplevel(self); win.title(Path(p).name); win.geometry('900x650'); win.configure(bg=COLORS['panel'])
        t = tk.Text(win, bg=COLORS['log_bg'], fg='#e8e8ea', font=('Monospace',9), wrap='word', padx=10, pady=8, bd=0)
        t.pack(fill='both', expand=True, padx=10, pady=10); t.insert('1.0', text); t.configure(state='disabled')
