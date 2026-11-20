#!/usr/bin/env python3
"""Training dataset editor/controller mixin for the ILT dashboard.

This module owns the Tk-facing state transitions for dataset layout detection,
per-class Train/Valid/Test allocation, grouping/renaming, live model preview,
temporary dataset preparation, and split-manifest generation.  Pure filesystem
and allocation rules remain in :mod:`training_dataset`.
"""
from __future__ import annotations

import re
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from ui_resources import COLORS
from training_dataset import (
    DATASET_MODE_CLASSES,
    DATASET_MODE_INVALID,
    DATASET_MODE_SPLIT,
    SPLITS as TRAINING_SPLITS,
    allocate_from_percentages,
    allocation_changes_dataset as training_allocation_changes_dataset,
    build_training_assignments,
    build_training_class_plan,
    cleanup_training_dataset_view,
    create_training_dataset_view,
    inspect_training_dataset,
    mapping_changes_dataset as training_mapping_changes_dataset,
    normalize_count_allocation,
    write_training_split_csv,
)


class TrainingDatasetUIMixin:
    def _schedule_train_dataset_refresh(self, *_args):
        """Debounce scans while a dataset path or split-folder name is edited."""
        try:
            if self._train_dataset_refresh_job is not None:
                self.after_cancel(self._train_dataset_refresh_job)
            self._train_dataset_refresh_job = self.after(
                250, self._refresh_training_dataset_classes
            )
        except Exception:
            self._train_dataset_refresh_job = None

    def _scan_training_dataset(self):
        split_folders = {
            'train': self.v['trainFolder'].get().strip(),
            'valid': self.v['validFolder'].get().strip(),
            'test': self.v['testFolder'].get().strip(),
        }
        return inspect_training_dataset(
            self.v['train_dataset'].get().strip(),
            split_folders,
        )

    def _refresh_train_dataset_detection_text(self):
        var = self._train_dataset_detection_var
        if var is None:
            return
        dataset_text = self.v['train_dataset'].get().strip()
        if not dataset_text:
            key = 'dataset_detection_wait'
            color = COLORS['muted']
        elif self._train_dataset_mode == DATASET_MODE_SPLIT:
            key = 'dataset_detection_split'
            color = COLORS['green']
        elif self._train_dataset_mode == DATASET_MODE_CLASSES:
            key = 'dataset_detection_classes'
            color = COLORS['cyan']
        else:
            key = 'dataset_detection_invalid'
            color = COLORS['amber']
        var.set(self.tr(key))
        try:
            self._train_dataset_detection_label.configure(fg=color)
        except Exception:
            pass

    def _format_split_percent(self, value):
        value = float(value)
        text = f'{value:.2f}'.rstrip('0').rstrip('.')
        return text or '0'

    def _percentages_from_allocation(self, class_name):
        total = int(self._train_class_totals.get(class_name, 0) or 0)
        allocation = self._train_class_allocations.get(
            class_name, {'train': 0, 'valid': 0, 'test': 0}
        )
        if total <= 0:
            return {split: 0.0 for split in TRAINING_SPLITS}
        return {
            split: 100.0 * int(allocation.get(split, 0) or 0) / total
            for split in TRAINING_SPLITS
        }

    def _sync_class_split_vars(self, class_name):
        vars_by_split = self._train_class_split_vars.get(class_name)
        if not vars_by_split:
            return
        mode = self._train_distribution_mode_var.get()
        self._train_allocation_sync = True
        try:
            if mode == 'percent':
                values = self._train_class_percentages.get(
                    class_name, self._percentages_from_allocation(class_name)
                )
                for split in TRAINING_SPLITS:
                    vars_by_split[split].set(
                        self._format_split_percent(values.get(split, 0.0))
                    )
            else:
                values = self._train_class_allocations.get(
                    class_name, {split: 0 for split in TRAINING_SPLITS}
                )
                for split in TRAINING_SPLITS:
                    vars_by_split[split].set(
                        str(int(values.get(split, 0) or 0))
                    )
        finally:
            self._train_allocation_sync = False
        self._refresh_training_unused_value(class_name)

    def _sync_all_class_split_vars(self):
        for class_name in self._train_class_split_vars:
            self._sync_class_split_vars(class_name)

    def _on_train_distribution_mode_changed(self, *_args):
        new_mode = self._train_distribution_mode_var.get()
        if new_mode not in ('percent', 'number'):
            return

        old_mode = getattr(self, '_train_distribution_mode_current', 'percent')
        if hasattr(self, '_train_distribution_default_vars'):
            self._train_distribution_default_cache[old_mode] = {
                split: self._train_distribution_default_vars[split].get()
                for split in TRAINING_SPLITS
            }
        self._train_distribution_mode_current = new_mode

        cache = self._train_distribution_default_cache[new_mode]
        self._train_allocation_sync = True
        try:
            for split in TRAINING_SPLITS:
                self._train_distribution_default_vars[split].set(cache[split])
        finally:
            self._train_allocation_sync = False

        if self._train_distribution_unit_var is not None:
            self._train_distribution_unit_var.set(
                '%' if new_mode == 'percent' else self.tr('images_unit')
            )
        self._sync_all_class_split_vars()
        self._refresh_training_model_summary()

    def _initialize_training_class_allocation(self, class_name):
        total = int(self._train_class_totals.get(class_name, 0) or 0)
        if self._train_dataset_mode == DATASET_MODE_SPLIT:
            allocation = {
                split: int(
                    self._train_class_counts.get(class_name, {}).get(split, 0)
                    or 0
                )
                for split in TRAINING_SPLITS
            }
            percentages = (
                {
                    split: 100.0 * allocation[split] / total
                    for split in TRAINING_SPLITS
                }
                if total > 0
                else {split: 0.0 for split in TRAINING_SPLITS}
            )
        else:
            percentages = {'train': 70.0, 'valid': 15.0, 'test': 15.0}
            allocation = allocate_from_percentages(total, percentages)

        self._train_class_allocations[class_name] = allocation
        self._train_class_percentages[class_name] = percentages
        self._train_class_allocation_errors.pop(class_name, None)

    def _parse_training_class_allocation(self, class_name):
        if self._train_allocation_sync:
            return True
        vars_by_split = self._train_class_split_vars.get(class_name)
        if not vars_by_split:
            return True

        total = int(self._train_class_totals.get(class_name, 0) or 0)
        mode = self._train_distribution_mode_var.get()
        try:
            if mode == 'percent':
                percentages = {}
                for split in TRAINING_SPLITS:
                    raw = vars_by_split[split].get().strip().replace(',', '.')
                    if raw == '':
                        raise ValueError('split_allocation_invalid')
                    value = float(raw)
                    if value < 0:
                        raise ValueError('split_allocation_invalid')
                    percentages[split] = value
                allocation = allocate_from_percentages(total, percentages)
                self._train_class_percentages[class_name] = percentages
                self._train_class_allocations[class_name] = allocation
            else:
                requested = {}
                for split in TRAINING_SPLITS:
                    raw = vars_by_split[split].get().strip()
                    if raw == '':
                        raise ValueError('split_allocation_invalid')
                    requested[split] = int(raw)
                allocation = normalize_count_allocation(total, requested)
                self._train_class_allocations[class_name] = allocation
                self._train_class_percentages[class_name] = (
                    {
                        split: 100.0 * allocation[split] / total
                        for split in TRAINING_SPLITS
                    }
                    if total > 0
                    else {split: 0.0 for split in TRAINING_SPLITS}
                )
        except (TypeError, ValueError) as exc:
            key = str(exc) if str(exc) in {
                'split_allocation_invalid',
                'split_allocation_exceeds_total',
                'split_percentage_exceeds_100',
            } else 'split_allocation_invalid'
            self._train_class_allocation_errors[class_name] = key
            self._refresh_training_unused_value(class_name, invalid=True)
            return False

        self._train_class_allocation_errors.pop(class_name, None)
        self._refresh_training_unused_value(class_name)
        return True

    def _on_train_split_value_changed(self, class_name):
        if self._train_allocation_sync:
            return
        self._parse_training_class_allocation(class_name)
        self._refresh_training_model_summary()

    def _refresh_training_unused_value(self, class_name, invalid=False):
        var = self._train_class_unused_vars.get(class_name)
        if var is None:
            return
        if invalid or class_name in self._train_class_allocation_errors:
            var.set('—')
            return
        total = int(self._train_class_totals.get(class_name, 0) or 0)
        used = sum(
            int(self._train_class_allocations.get(class_name, {}).get(split, 0) or 0)
            for split in TRAINING_SPLITS
        )
        var.set(f'{max(0, total - used):,}'.replace(',', '.'))

    def _apply_training_distribution_to_all(self):
        mode = self._train_distribution_mode_var.get()
        try:
            if mode == 'percent':
                percentages = {}
                for split in TRAINING_SPLITS:
                    raw = self._train_distribution_default_vars[split].get()
                    value = float(raw.strip().replace(',', '.'))
                    if value < 0:
                        raise ValueError('split_allocation_invalid')
                    percentages[split] = value
                if sum(percentages.values()) > 100.0000001:
                    raise ValueError('split_percentage_exceeds_100')
                for class_name in self._train_class_totals:
                    total = self._train_class_totals[class_name]
                    self._train_class_percentages[class_name] = dict(percentages)
                    self._train_class_allocations[class_name] = (
                        allocate_from_percentages(total, percentages)
                    )
                    self._train_class_allocation_errors.pop(class_name, None)
            else:
                requested = {
                    split: int(
                        self._train_distribution_default_vars[split].get().strip()
                    )
                    for split in TRAINING_SPLITS
                }
                if any(value < 0 for value in requested.values()):
                    raise ValueError('split_allocation_invalid')
                requested_total = sum(requested.values())
                too_small = [
                    class_name
                    for class_name, total in self._train_class_totals.items()
                    if bool(self._train_class_use_vars.get(class_name).get())
                    and requested_total > total
                ]
                if too_small:
                    raise ValueError('distribution_default_exceeds_some_classes')
                for class_name, total in self._train_class_totals.items():
                    allocation = normalize_count_allocation(total, requested)
                    self._train_class_allocations[class_name] = allocation
                    self._train_class_percentages[class_name] = (
                        {
                            split: 100.0 * allocation[split] / total
                            for split in TRAINING_SPLITS
                        }
                        if total > 0
                        else {split: 0.0 for split in TRAINING_SPLITS}
                    )
                    self._train_class_allocation_errors.pop(class_name, None)
        except (TypeError, ValueError) as exc:
            key = str(exc) if str(exc) in {
                'split_allocation_invalid',
                'split_percentage_exceeds_100',
                'distribution_default_exceeds_some_classes',
            } else 'split_allocation_invalid'
            messagebox.showerror(self.tr('error'), self.tr(key))
            return

        self._sync_all_class_split_vars()
        self._refresh_training_model_summary()

    def _render_training_class_rows(self):
        host = self._train_class_rows_host
        if host is None:
            return
        try:
            alive = bool(host.winfo_exists())
        except Exception:
            alive = False
        if not alive:
            return

        for child in host.winfo_children():
            child.destroy()

        if not self.v['train_dataset'].get().strip():
            tk.Label(
                host, text=self.tr('dataset_classes_wait'), bg=COLORS['card2'],
                fg=COLORS['muted'], font=('Sans', 8), anchor='w', justify='left'
            ).pack(fill='x', padx=9, pady=12)
            return

        if not self._train_class_counts:
            text_key = (
                'dataset_structure_invalid'
                if self._train_dataset_mode == DATASET_MODE_INVALID
                else 'dataset_classes_empty'
            )
            tk.Label(
                host, text=self.tr(text_key), bg=COLORS['card2'],
                fg=COLORS['amber'], font=('Sans', 8), anchor='w', justify='left'
            ).pack(fill='x', padx=9, pady=12)
            return

        palette = (
            COLORS['red'], COLORS['amber'], COLORS['green'],
            COLORS['cyan'], COLORS['blue'], COLORS['purple']
        )
        for idx, class_name in enumerate(sorted(self._train_class_counts)):
            row_bg = COLORS['card2'] if idx % 2 == 0 else '#19191d'
            row = tk.Frame(host, bg=row_bg)
            row.pack(fill='x', padx=2, pady=1)

            use_var = self._train_class_use_vars[class_name]
            cb = tk.Checkbutton(
                row, variable=use_var, bg=row_bg, fg=COLORS['text'],
                activebackground=row_bg, activeforeground=COLORS['text'],
                selectcolor=COLORS['entry'], highlightthickness=0, bd=0,
                width=3, anchor='center', cursor='hand2'
            )
            cb.pack(side='left', padx=(2, 0))

            original = tk.Frame(row, bg=row_bg)
            original.pack(side='left', fill='x', expand=True)
            tk.Label(
                original, text='●', bg=row_bg,
                fg=palette[idx % len(palette)], font=('Sans', 8, 'bold')
            ).pack(side='left')
            tk.Label(
                original, text=class_name, bg=row_bg, fg=COLORS['text'],
                width=15, anchor='w', font=('Sans', 8)
            ).pack(side='left', fill='x', expand=True, padx=(3, 2))

            total = int(self._train_class_totals.get(class_name, 0) or 0)
            tk.Label(
                row, text=f'{total:,}'.replace(',', '.'),
                bg=row_bg, fg=COLORS['text'], width=7, anchor='center',
                font=('Monospace', 8, 'bold')
            ).pack(side='left')

            for split in TRAINING_SPLITS:
                entry = tk.Entry(
                    row,
                    textvariable=self._train_class_split_vars[class_name][split],
                    bg=COLORS['entry'], fg=COLORS['text'],
                    insertbackground=COLORS['text'], justify='center',
                    relief='flat', highlightbackground=COLORS['border'],
                    highlightthickness=1, width=7, font=('Monospace', 8)
                )
                entry.pack(side='left', padx=1, pady=3, ipady=3)

            unused_var = self._train_class_unused_vars[class_name]
            tk.Label(
                row, textvariable=unused_var, bg=row_bg, fg=COLORS['muted'],
                width=7, anchor='center', font=('Monospace', 8)
            ).pack(side='left')

            final_entry = tk.Entry(
                row, textvariable=self._train_class_map_vars[class_name],
                bg=COLORS['entry'], fg=COLORS['text'],
                insertbackground=COLORS['text'], relief='flat',
                highlightbackground=COLORS['border'], highlightthickness=1,
                width=17, font=('Sans', 8)
            )
            final_entry.pack(
                side='left', fill='x', expand=True,
                padx=(4, 3), pady=3, ipady=3
            )

    def _refresh_training_dataset_classes(self):
        self._train_dataset_refresh_job = None
        inspection = self._scan_training_dataset()
        dataset_key = (
            inspection.dataset_key,
            inspection.mode,
            self.v['trainFolder'].get().strip(),
            self.v['validFolder'].get().strip(),
            self.v['testFolder'].get().strip(),
        )
        new_dataset = dataset_key != self._train_class_dataset_key

        if new_dataset:
            self._train_class_map_vars = {}
            self._train_class_use_vars = {}
            self._train_class_allocations = {}
            self._train_class_percentages = {}
            self._train_class_split_vars = {}
            self._train_class_unused_vars = {}
            self._train_class_allocation_errors = {}
            self._train_class_dataset_key = dataset_key

        self._train_dataset_mode = inspection.mode
        self._train_class_counts = inspection.counts
        self._train_class_totals = inspection.totals

        if new_dataset and inspection.mode in (
            DATASET_MODE_SPLIT, DATASET_MODE_CLASSES
        ):
            desired_mode = (
                'number'
                if inspection.mode == DATASET_MODE_SPLIT
                else 'percent'
            )
            if self._train_distribution_mode_var.get() != desired_mode:
                self._train_distribution_mode_var.set(desired_mode)

        for class_name in self._train_class_counts:
            if class_name not in self._train_class_map_vars:
                target_var = tk.StringVar(value=class_name)
                target_var.trace_add(
                    'write', lambda *_a: self._refresh_training_model_summary()
                )
                self._train_class_map_vars[class_name] = target_var
            if class_name not in self._train_class_use_vars:
                use_var = tk.BooleanVar(value=True)
                use_var.trace_add(
                    'write', lambda *_a: self._refresh_training_model_summary()
                )
                self._train_class_use_vars[class_name] = use_var

            if class_name not in self._train_class_allocations:
                self._initialize_training_class_allocation(class_name)
            else:
                total = int(self._train_class_totals.get(class_name, 0) or 0)
                if self._train_distribution_mode_var.get() == 'percent':
                    percentages = self._train_class_percentages.get(
                        class_name, {'train': 70.0, 'valid': 15.0, 'test': 15.0}
                    )
                    try:
                        self._train_class_allocations[class_name] = (
                            allocate_from_percentages(total, percentages)
                        )
                    except ValueError:
                        self._initialize_training_class_allocation(class_name)
                elif sum(self._train_class_allocations[class_name].values()) > total:
                    self._initialize_training_class_allocation(class_name)

            if class_name not in self._train_class_split_vars:
                split_vars = {}
                for split in TRAINING_SPLITS:
                    var = tk.StringVar()
                    var.trace_add(
                        'write',
                        lambda *_a, c=class_name: self._on_train_split_value_changed(c)
                    )
                    split_vars[split] = var
                self._train_class_split_vars[class_name] = split_vars
                self._train_class_unused_vars[class_name] = tk.StringVar()
            self._sync_class_split_vars(class_name)

        stale_names = set(self._train_class_map_vars) - set(self._train_class_counts)
        for stale in stale_names:
            self._train_class_map_vars.pop(stale, None)
            self._train_class_use_vars.pop(stale, None)
            self._train_class_allocations.pop(stale, None)
            self._train_class_percentages.pop(stale, None)
            self._train_class_split_vars.pop(stale, None)
            self._train_class_unused_vars.pop(stale, None)
            self._train_class_allocation_errors.pop(stale, None)

        self._refresh_train_dataset_detection_text()
        self._render_training_class_rows()
        self._refresh_training_model_summary()

    def _training_class_plan(self):
        enabled = {
            original: bool(var.get())
            for original, var in self._train_class_use_vars.items()
        }
        allocation_errors = []
        for original in self._train_class_counts:
            if enabled.get(original, True):
                self._parse_training_class_allocation(original)
                key = self._train_class_allocation_errors.get(original)
                if key:
                    allocation_errors.append(key)

        target_names = {
            original: var.get()
            for original, var in self._train_class_map_vars.items()
        }
        mapping, summary, error_keys = build_training_class_plan(
            self._train_class_counts,
            target_names,
            enabled,
            allocations=self._train_class_allocations,
            available_totals=self._train_class_totals,
        )
        keys = list(dict.fromkeys(allocation_errors + error_keys))
        return mapping, summary, [self.tr(key) for key in keys]

    def _refresh_training_model_summary(self):
        mapping, summary, errors = self._training_class_plan()
        tree = self._train_model_tree
        source_counts = {}
        for original, final_class in mapping.items():
            source_counts[final_class] = source_counts.get(final_class, 0) + 1

        if tree is not None:
            try:
                alive = bool(tree.winfo_exists())
            except Exception:
                alive = False
            if alive:
                for item in tree.get_children():
                    tree.delete(item)
                for class_name in sorted(summary):
                    rec = summary[class_name]
                    total = rec['train'] + rec['valid'] + rec['test']
                    tree.insert(
                        '', 'end',
                        values=(
                            class_name,
                            f'{total:,}'.replace(',', '.'),
                            f"{rec['train']:,}".replace(',', '.'),
                            f"{rec['valid']:,}".replace(',', '.'),
                            f"{rec['test']:,}".replace(',', '.'),
                            source_counts.get(class_name, 0),
                        )
                    )

        if self._train_model_summary_var is None:
            return
        if errors:
            self._train_model_summary_var.set(errors[0])
            if hasattr(self, '_train_model_summary_label'):
                self._train_model_summary_label.configure(fg=COLORS['amber'])
        elif not mapping:
            key = (
                'dataset_classes_wait'
                if not self.v['train_dataset'].get().strip()
                else 'model_classes_empty'
            )
            self._train_model_summary_var.set(self.tr(key))
            if hasattr(self, '_train_model_summary_label'):
                self._train_model_summary_label.configure(fg=COLORS['muted'])
        else:
            train_n = sum(rec['train'] for rec in summary.values())
            valid_n = sum(rec['valid'] for rec in summary.values())
            test_n = sum(rec['test'] for rec in summary.values())
            selected_total = train_n + valid_n + test_n
            available_total = sum(
                int(self._train_class_totals.get(original, 0) or 0)
                for original in mapping
            )
            unused_total = max(0, available_total - selected_total)
            original_total = len(self._train_class_counts)
            used_original = len(mapping)
            if self.lang == 'pt':
                msg = (
                    f"Originais: {original_total} • Usadas: {used_original} • "
                    f"Classes finais: {len(summary)} • Disponíveis: {available_total:,} • "
                    f"Selecionadas: {selected_total:,} • Não usadas: {unused_total:,} • "
                    f"Treino: {train_n:,} • Valid: {valid_n:,} • Teste: {test_n:,}"
                )
            else:
                msg = (
                    f"Original: {original_total} • Used: {used_original} • "
                    f"Final classes: {len(summary)} • Available: {available_total:,} • "
                    f"Selected: {selected_total:,} • Unused: {unused_total:,} • "
                    f"Train: {train_n:,} • Valid: {valid_n:,} • Test: {test_n:,}"
                )
            self._train_model_summary_var.set(msg.replace(',', '.'))
            if hasattr(self, '_train_model_summary_label'):
                self._train_model_summary_label.configure(fg=COLORS['green'])

    def _reset_training_class_names(self):
        for class_name, var in self._train_class_map_vars.items():
            var.set(class_name)
        for var in self._train_class_use_vars.values():
            var.set(True)
        self._refresh_training_model_summary()

    def _training_plan_changes_dataset(self, mapping):
        enabled = {
            original: bool(var.get())
            for original, var in self._train_class_use_vars.items()
        }
        return (
            training_mapping_changes_dataset(mapping, self._train_class_counts)
            or training_allocation_changes_dataset(
                self._train_dataset_mode,
                self._train_class_counts,
                self._train_class_allocations,
                enabled,
            )
        )

    def _cleanup_temp_train_dataset(self):
        temp_root = self.temp_train_dataset
        self.temp_train_dataset = None
        cleanup_training_dataset_view(temp_root)

    def _build_training_assignments(self, dataset, cfg, mapping):
        split_folders = {
            'train': cfg['trainFolder'],
            'valid': cfg['validFolder'],
            'test': cfg['testFolder'],
        }
        return build_training_assignments(
            dataset,
            split_folders,
            mapping,
            self._train_class_allocations,
            dataset_mode=self._train_dataset_mode,
            seed=cfg['seed'],
        )

    def _prepare_training_dataset(
        self, dataset, cfg, mapping, summary, assignments
    ):
        """Build a non-destructive temporary train/valid/test view."""
        self._cleanup_temp_train_dataset()
        split_folders = {
            'train': cfg['trainFolder'],
            'valid': cfg['validFolder'],
            'test': cfg['testFolder'],
        }
        temp_root = create_training_dataset_view(
            dataset,
            split_folders,
            mapping,
            summary,
            allocations=self._train_class_allocations,
            dataset_mode=self._train_dataset_mode,
            seed=cfg['seed'],
            assignments=assignments,
        )
        self.temp_train_dataset = temp_root
        return temp_root

    def _write_training_split_manifest(self, assignments, output, project):
        safe_project = re.sub(r'[^A-Za-z0-9._-]+', '_', project).strip('._')
        safe_project = safe_project or 'model'
        path = Path(output) / f'dataset_split_{safe_project}.csv'
        self.last_split_manifest_path = write_training_split_csv(assignments, path)
        return self.last_split_manifest_path

