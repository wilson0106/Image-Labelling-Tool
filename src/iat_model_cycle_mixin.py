#!/usr/bin/env python3
"""Integrated train → classify → cluster workflow shown only inside IAT."""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

from iat_backend import infer_batch_folder
from platform_utils import worker_command
from project_model_cycle import (
    allocate_counts,
    apply_model_in_place,
    batch_id_from_csv,
    build_reference_assignments,
    materialize_reference_dataset,
    next_model_version,
    recompute_projection_for_version,
    save_cycle_metadata,
    summarize_assignments,
    write_cycle_manifest,
)
from training_dataset import cleanup_training_dataset_view
from ui_resources import IAT_COLORS as C
from iat_widgets import ScrollFrame


class IATModelCycleMixin:
    """UI/state for the project-local iterative model cycle."""

    def _cycle_text(self, pt, en):
        return pt if self.lang() == "pt" else en

    def _cycle_state_init(self):
        if getattr(self, "_cycle_initialized", False):
            return
        self._cycle_initialized = True
        self._cycle_batch_vars = {}
        self._cycle_class_vars = {}
        self._cycle_target_vars = {}
        self._cycle_reference_batches = []
        self._cycle_trained_version = None
        self._cycle_model_path = None
        self._cycle_labels_path = None
        self._cycle_temp_dataset = None
        self._cycle_train_proc = None
        self._cycle_stage_progress = {
            "reference": tk.DoubleVar(value=0),
            "training": tk.DoubleVar(value=0),
            "classification": tk.DoubleVar(value=0),
            "cluster": tk.DoubleVar(value=0),
        }
        self._cycle_status = tk.StringVar(value=self._cycle_text("Pronto.", "Ready."))
        self._cycle_distribution_mode = tk.StringVar(value="percent")
        self._cycle_default_train = tk.StringVar(value="70")
        self._cycle_default_valid = tk.StringVar(value="15")
        self._cycle_default_test = tk.StringVar(value="15")
        self._cycle_epochs = tk.StringVar(value="50")
        self._cycle_batch_size = tk.StringVar(value="32")
        self._cycle_lr = tk.StringVar(value="1e-5")
        self._cycle_patience = tk.StringVar(value="10")
        self._cycle_seed = tk.StringVar(value="0")
        self._cycle_version = tk.StringVar(value="")
        self._cycle_apply_model = tk.StringVar(value="")
        self._cycle_target_mode = tk.StringVar(value="pending")
        self._cycle_update_reference_features = tk.BooleanVar(value=True)
        self._cycle_auto_cluster = tk.BooleanVar(value=True)
        # UI-only state for the integrated-cycle dashboard. These variables do
        # not change scientific behavior; they drive filtering and live status.
        self._cycle_batch_search = tk.StringVar(value="")
        self._cycle_class_search = tk.StringVar(value="")
        self._cycle_train_detail = tk.StringVar(value="—")
        self._cycle_train_percent_text = tk.StringVar(value="0%")
        self._cycle_train_history = {"train": [], "valid": []}
        self._cycle_train_epoch = 0
        self._cycle_train_elapsed = ""

    def _cycle_section(self, parent, title, number=None, subtitle=None):
        """Create one visually consistent card for the integrated cycle only."""
        frame = tk.Frame(
            parent, bg=C["card"], highlightbackground=C["border"],
            highlightthickness=1
        )
        head = tk.Frame(frame, bg=C["card"])
        head.pack(fill="x", padx=12, pady=(10, 6))
        if number is not None:
            badge = tk.Label(
                head, text=str(number), bg=C["purple"], fg="white",
                font=("Sans", 10, "bold"), width=2, pady=3
            )
            badge.pack(side="left", padx=(0, 8))
        title_box = tk.Frame(head, bg=C["card"])
        title_box.pack(side="left", fill="x", expand=True)
        tk.Label(
            title_box, text=title, bg=C["card"], fg="#cf83ff",
            font=("Sans", 10, "bold")
        ).pack(anchor="w")
        if subtitle:
            tk.Label(
                title_box, text=subtitle, bg=C["card"], fg=C["muted"],
                font=("Sans", 8), justify="left", wraplength=900
            ).pack(anchor="w", pady=(2, 0))
        body = tk.Frame(frame, bg=C["card"])
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        return frame, body

    def _cycle_progress_line(self, parent, label, key, column=0):
        """Compact stage tile with its own progress bar and percentage."""
        tile = tk.Frame(
            parent, bg=C["card2"], highlightbackground=C["border"],
            highlightthickness=1
        )
        tile.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 4, 0))
        parent.columnconfigure(column, weight=1)
        top = tk.Frame(tile, bg=C["card2"])
        top.pack(fill="x", padx=9, pady=(7, 4))
        tk.Label(
            top, text=label, bg=C["card2"], fg=C["text"],
            font=("Sans", 8, "bold"), anchor="w"
        ).pack(side="left", fill="x", expand=True)
        pct = tk.Label(top, text="0%", bg=C["card2"], fg=C["cyan"], font=("Sans", 8, "bold"))
        pct.pack(side="right")
        ttk.Progressbar(
            tile, maximum=100, variable=self._cycle_stage_progress[key],
            style="Blue.Horizontal.TProgressbar"
        ).pack(fill="x", padx=9, pady=(0, 8))
        var = self._cycle_stage_progress[key]
        def update(*_):
            try:
                pct.configure(text=f"{float(var.get()):.0f}%")
            except Exception:
                return
        var.trace_add("write", update)


    def _cycle_row_palette(self, selected, disabled=False):
        if disabled:
            return "#101012", "#617386", C["border"]
        if selected:
            return "#123a5e", "#ffffff", C["blue"]
        return C["card2"], C["text"], C["border"]

    def _cycle_refresh_batch_visuals(self):
        for name, item in getattr(self, "_cycle_batch_row_widgets", {}).items():
            selected = bool(self._cycle_batch_vars.get(name).get()) if name in self._cycle_batch_vars else False
            bg, fg, accent = self._cycle_row_palette(selected)
            try:
                item["row"].configure(bg=bg, highlightbackground=accent)
                item["indicator"].configure(
                    text="✓" if selected else "",
                    bg=C["blue2"] if selected else "#1e1e23",
                    fg="white" if selected else C["muted"],
                )
                for widget in item["labels"]:
                    widget.configure(bg=bg, fg=fg)
            except Exception:
                pass

    def _cycle_toggle_batch(self, name):
        var = self._cycle_batch_vars.get(name)
        if var is None:
            return
        var.set(not bool(var.get()))
        self._cycle_batch_selection_changed()

    def _cycle_refresh_class_visuals(self):
        for cls, state in getattr(self, "_cycle_class_vars", {}).items():
            selected = bool(state["enabled"].get())
            bg, fg, accent = self._cycle_row_palette(selected)
            row = state.get("_row")
            toggle = state.get("_toggle")
            try:
                if row is not None:
                    row.configure(bg=bg, highlightbackground=accent)
                if toggle is not None:
                    toggle.configure(
                        text="✓" if selected else "",
                        bg=C["blue2"] if selected else "#1e1e23",
                        fg="white",
                        activebackground=C["blue2"] if selected else C["card2"],
                    )
                for widget in state.get("_labels", []):
                    widget.configure(bg=bg, fg=fg)
                swatch = state.get("_swatch")
                if swatch is not None:
                    swatch.configure(bg=bg)
            except Exception:
                pass

    def _cycle_toggle_class(self, cls):
        state = self._cycle_class_vars.get(cls)
        if not state:
            return
        state["enabled"].set(not bool(state["enabled"].get()))
        self._cycle_refresh_summary()

    def _cycle_set_all_targets(self, value):
        for var in getattr(self, "_cycle_target_vars", {}).values():
            var.set(bool(value))
        self._cycle_refresh_target_visuals()

    def _cycle_refresh_target_visuals(self):
        for name, item in getattr(self, "_cycle_target_row_widgets", {}).items():
            selected = bool(self._cycle_target_vars.get(name).get()) if name in self._cycle_target_vars else False
            bg, fg, accent = self._cycle_row_palette(selected)
            try:
                item["row"].configure(bg=bg, highlightbackground=accent)
                item["indicator"].configure(
                    text="✓" if selected else "",
                    bg=C["blue2"] if selected else "#1e1e23",
                    fg="white",
                )
                for widget in item["labels"]:
                    widget.configure(bg=bg, fg=fg)
            except Exception:
                pass

    def _cycle_toggle_target(self, name):
        var = self._cycle_target_vars.get(name)
        if var is None:
            return
        var.set(not bool(var.get()))
        self._cycle_refresh_target_visuals()



    def _cycle_make_step_strip(self, parent):
        """Reference-style five-step strip used only by the integrated cycle."""
        strip = tk.Frame(parent, bg=C["bg"])
        labels = [
            self._cycle_text("Seleção de batches", "Batch selection"),
            self._cycle_text("Classes e dataset", "Classes & dataset"),
            self._cycle_text("Treinamento", "Training"),
            self._cycle_text("Classificação (com clustering)", "Classification (with clustering)"),
            self._cycle_text("Revisão e iteração", "Review & iteration"),
        ]
        progress_keys = ["reference", "reference", "training", "classification", "cluster"]
        for idx, (label, pkey) in enumerate(zip(labels, progress_keys), 1):
            active = idx <= 2
            tile_bg = "#0c67d8" if active else "#151518"
            border = "#1f83ef" if active else C["border"]
            tile = tk.Frame(strip, bg=tile_bg, highlightbackground=border, highlightthickness=1)
            tile.grid(row=0, column=idx-1, sticky="nsew", padx=(0 if idx == 1 else 2, 0))
            strip.columnconfigure(idx-1, weight=1)
            top = tk.Frame(tile, bg=tile_bg)
            top.pack(fill="both", expand=True, padx=8, pady=(6, 4))
            badge_bg = "#1889ff" if active else "#393941"
            badge = tk.Label(
                top, text=str(idx), width=2, bg=badge_bg, fg="white",
                font=("Sans", 9, "bold"), pady=2
            )
            badge.pack(side="left", padx=(0, 7))
            tk.Label(
                top, text=label, bg=tile_bg, fg="white" if active else "#ededef",
                font=("Sans", 8, "bold"), anchor="w"
            ).pack(side="left", fill="x", expand=True)
            # A thin progress indicator preserves the requested "loading per
            # stage" without turning the top strip into a second control panel.
            if idx < 5:
                ttk.Progressbar(
                    tile, maximum=100, variable=self._cycle_stage_progress[pkey],
                    style="Blue.Horizontal.TProgressbar"
                ).pack(fill="x", padx=8, pady=(0, 5))
        return strip

    def _cycle_help_panel(self, parent, title, items):
        """Compact right-column help card matching the approved mockup."""
        frame = tk.Frame(
            parent, bg="#18181b", highlightbackground="#17608f",
            highlightthickness=1
        )
        tk.Label(
            frame, text="ⓘ  " + title, bg="#18181b", fg="#7dc8ff",
            font=("Sans", 9, "bold"), anchor="w"
        ).pack(fill="x", padx=10, pady=(10, 6))
        for heading, body in items:
            row = tk.Frame(frame, bg="#1c1c20", highlightbackground=C["border"], highlightthickness=1)
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(
                row, text="›", bg="#1c1c20", fg="#c9c9ce",
                font=("Sans", 11, "bold")
            ).pack(side="left", padx=(8, 4), pady=6)
            label = tk.Label(
                row, text=heading, bg="#1c1c20", fg=C["text"],
                font=("Sans", 8, "bold"), anchor="w"
            )
            label.pack(side="left", fill="x", expand=True, pady=6)

            def show_help(_e=None, h=heading, b=body):
                messagebox.showinfo(h, b)
            for widget in (row, label):
                widget.bind("<Button-1>", show_help)
                try:
                    widget.configure(cursor="hand2")
                except Exception:
                    pass
        return frame

    def _cycle_select_validated_batches(self):
        """Select only batches whose manual_label coverage is 100%."""
        meta = getattr(self, "_cycle_batch_meta", {})
        for name, var in self._cycle_batch_vars.items():
            info = meta.get(name, {})
            total = int(info.get("total", 0) or 0)
            validated = int(info.get("validated", 0) or 0)
            var.set(total > 0 and validated >= total)
        self._cycle_batch_selection_changed()

    def _cycle_filter_batch_rows(self, *_):
        query = (self._cycle_batch_search.get() or "").strip().lower()
        for name, item in getattr(self, "_cycle_batch_row_widgets", {}).items():
            row = item.get("row")
            if row is None:
                continue
            try:
                row.pack_forget()
            except Exception:
                continue
        for name, item in getattr(self, "_cycle_batch_row_widgets", {}).items():
            if query and query not in batch_id_from_csv(name).lower():
                continue
            item["row"].pack(fill="x", pady=1, padx=1)

    def _cycle_filter_class_rows(self, *_):
        query = (self._cycle_class_search.get() or "").strip().lower()
        for cls, state in getattr(self, "_cycle_class_vars", {}).items():
            row = state.get("_row")
            if row is None:
                continue
            try:
                row.pack_forget()
            except Exception:
                continue
        for cls, state in getattr(self, "_cycle_class_vars", {}).items():
            final = state["final"].get().strip().lower()
            if query and query not in cls.lower() and query not in final:
                continue
            state["_row"].pack(fill="x", pady=1, padx=1)

    def _cycle_training_log_event(self, line):
        """Parse already-emitted train.py text for UI-only live metrics."""
        text = str(line or "")
        m = re.search(r"Epoch\s+(\d+),\s*time:\s*([0-9.]+)\s*min", text, re.I)
        if m:
            self._cycle_train_epoch = int(m.group(1))
            self._cycle_train_elapsed = m.group(2) + " min"
            self._cycle_update_training_detail()
            return
        m = re.search(r"train acc:\s*([0-9.]+).*?loss:\s*([0-9.]+)", text, re.I)
        if m:
            self._cycle_train_history.setdefault("train", []).append(float(m.group(1)))
            self._cycle_update_training_detail(train_acc=float(m.group(1)), train_loss=float(m.group(2)))
            self._cycle_redraw_training_chart()
            return
        m = re.search(r"valid acc:\s*([0-9.]+).*?loss:\s*([0-9.]+)", text, re.I)
        if m:
            self._cycle_train_history.setdefault("valid", []).append(float(m.group(1)))
            self._cycle_update_training_detail(valid_acc=float(m.group(1)), valid_loss=float(m.group(2)))
            self._cycle_redraw_training_chart()

    def _cycle_update_training_detail(self, train_acc=None, train_loss=None, valid_acc=None, valid_loss=None):
        try:
            current = getattr(self, "_cycle_train_metrics", {})
            if train_acc is not None:
                current["train_acc"] = train_acc
            if train_loss is not None:
                current["train_loss"] = train_loss
            if valid_acc is not None:
                current["valid_acc"] = valid_acc
            if valid_loss is not None:
                current["valid_loss"] = valid_loss
            self._cycle_train_metrics = current
            epochs = max(1, int(float(self._cycle_epochs.get() or 1)))
            epoch = int(getattr(self, "_cycle_train_epoch", 0) or 0)
            lines = [
                f"{self._cycle_text('Época','Epoch')}: {epoch} / {epochs}",
                f"{self._cycle_text('Tempo decorrido','Elapsed')}: {getattr(self, '_cycle_train_elapsed', '') or '—'}",
            ]
            if "train_loss" in current:
                lines.append(f"Loss ({self._cycle_text('treino','train')}): {current['train_loss']:.4f}")
            if "valid_acc" in current:
                lines.append(f"{self._cycle_text('Acurácia (validação)','Accuracy (validation)')}: {100*current['valid_acc']:.2f}%")
            self._cycle_train_detail.set("\n".join(lines))
        except Exception:
            pass

    def _cycle_redraw_training_chart(self):
        canvas = getattr(self, "_cycle_train_chart", None)
        if canvas is None:
            return
        try:
            canvas.delete("all")
            w = max(240, int(canvas.winfo_width() or 240))
            h = max(95, int(canvas.winfo_height() or 95))
            left, top, right, bottom = 28, 8, w-8, h-20
            canvas.create_line(left, bottom, right, bottom, fill="#38536e")
            canvas.create_line(left, top, left, bottom, fill="#38536e")
            for frac in (0.25, 0.5, 0.75):
                y = bottom - frac*(bottom-top)
                canvas.create_line(left, y, right, y, fill="#26262c")
            for key, color in (("train", "#57b8ff"), ("valid", "#c168ff")):
                vals = list(self._cycle_train_history.get(key, []))
                if len(vals) < 2:
                    continue
                pts = []
                denom = max(1, len(vals)-1)
                for i, value in enumerate(vals):
                    x = left + (right-left) * i / denom
                    y = bottom - max(0.0, min(1.0, float(value))) * (bottom-top)
                    pts.extend((x, y))
                canvas.create_line(*pts, fill=color, width=2, smooth=True)
            canvas.create_text(left, h-8, text="0", fill=C["muted"], anchor="w", font=("Sans", 7))
            canvas.create_text(right, h-8, text=str(max(len(self._cycle_train_history.get("train", [])),
                                                         len(self._cycle_train_history.get("valid", [])), 1)),
                               fill=C["muted"], anchor="e", font=("Sans", 7))
        except Exception:
            pass


    def _build_model_cycle(self):
        self._cycle_state_init()
        # This page deliberately mirrors the approved 1536×1024 mockup:
        # 3-column information architecture, five-step strip, summary/help on
        # the right, and wide operational cards in the center.
        host = tk.Frame(self.content, bg=C["bg"])
        host.pack(fill="both", expand=True)
        sf = ScrollFrame(host, C["bg"])
        sf.pack(fill="both", expand=True)
        page = sf.inner
        page.columnconfigure(0, weight=34, minsize=330)
        page.columnconfigure(1, weight=50, minsize=500)
        page.columnconfigure(2, weight=16, minsize=185)

        tk.Label(
            page,
            text=self._cycle_text("IAT — Treino + Classificação (Integrado)", "IAT — Training + Classification (Integrated)"),
            bg=C["bg"], fg=C["text"], font=("Sans", 17, "bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(5, 0))
        tk.Label(
            page,
            text=self._cycle_text(
                "Treine um novo modelo com batches já validados e aplique automaticamente aos batches restantes do projeto.",
                "Train a new model from validated batches and automatically apply it to the remaining project batches."
            ),
            bg=C["bg"], fg="#c6c6cc", font=("Sans", 9)
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=6, pady=(1, 8))

        strip = self._cycle_make_step_strip(page)
        strip.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 8))

        # ------------------------------------------------------------------
        # 1. Reference batches
        # ------------------------------------------------------------------
        bcard, bbody = self._cycle_section(
            page,
            self._cycle_text("Seleção de batches para treinamento", "Training batch selection"),
            1,
            self._cycle_text(
                "Escolha quais batches já validados serão usados como referência.",
                "Choose which validated batches will be used as reference."
            )
        )
        bcard.grid(row=3, column=0, sticky="nsew", padx=(6, 4), pady=4)

        btools = tk.Frame(bbody, bg=C["card"])
        btools.pack(fill="x", pady=(0, 6))
        bsearch = tk.Entry(
            btools, textvariable=self._cycle_batch_search, bg=C["entry"], fg=C["text"],
            insertbackground="white", relief="flat", font=("Sans", 8)
        )
        bsearch.pack(side="left", fill="x", expand=True, ipady=5)
        bsearch.bind("<KeyRelease>", self._cycle_filter_batch_rows)
        tk.Button(
            btools, text=self._cycle_text("Selecionar validados", "Select validated"),
            command=self._cycle_select_validated_batches, bg="#25252a", fg=C["text"],
            activebackground=C["blue2"], activeforeground="white", relief="flat",
            padx=9, pady=5, font=("Sans", 8, "bold")
        ).pack(side="right", padx=(6, 0))

        bh = tk.Frame(bbody, bg="#131315")
        bh.pack(fill="x")
        for text, width, anchor in [
            ("", 3, "center"),
            (self._cycle_text("Batch", "Batch"), 14, "w"),
            (self._cycle_text("Total", "Total"), 7, "e"),
            (self._cycle_text("Validados", "Validated"), 8, "e"),
            ("% ", 6, "e"),
            (self._cycle_text("Status", "Status"), 9, "center"),
        ]:
            tk.Label(
                bh, text=text, width=width, bg="#131315", fg="#c9c9ce",
                font=("Sans", 7, "bold"), anchor=anchor
            ).pack(side="left", padx=1, pady=5)

        self._cycle_batch_list = tk.Frame(bbody, bg=C["card"], height=135)
        self._cycle_batch_list.pack(fill="both", expand=True)
        self._cycle_batch_list.pack_propagate(False)

        bfoot = tk.Frame(bbody, bg=C["card"])
        bfoot.pack(fill="x", pady=(7, 0))
        tk.Button(
            bfoot, text=self._cycle_text("Selecionar todos", "Select all"),
            command=lambda: self._cycle_set_all_batches(True),
            bg="#25252a", fg=C["text"], relief="flat", padx=9, pady=4
        ).pack(side="left")
        tk.Button(
            bfoot, text=self._cycle_text("Limpar seleção", "Clear selection"),
            command=lambda: self._cycle_set_all_batches(False),
            bg="#25252a", fg=C["text"], relief="flat", padx=9, pady=4
        ).pack(side="left", padx=5)
        self._cycle_batch_summary = tk.Label(
            bfoot, text="", bg=C["card"], fg=C["cyan"], font=("Sans", 8, "bold")
        )
        self._cycle_batch_summary.pack(side="right")

        # ------------------------------------------------------------------
        # 2. Classes and dataset definition
        # ------------------------------------------------------------------
        ccard, cbody = self._cycle_section(
            page,
            self._cycle_text("Classes e definição do dataset de treinamento", "Classes and training dataset definition"),
            2,
            self._cycle_text(
                "Selecione as classes, renomeie se necessário e defina a quantidade ou percentual para treino, validação e teste.",
                "Select classes, rename when needed and define count or percentage for train, validation and test."
            )
        )
        ccard.grid(row=3, column=1, sticky="nsew", padx=4, pady=4)

        ctools = tk.Frame(cbody, bg=C["card"])
        ctools.pack(fill="x", pady=(0, 5))
        csearch = tk.Entry(
            ctools, textvariable=self._cycle_class_search, bg=C["entry"], fg=C["text"],
            insertbackground="white", relief="flat", font=("Sans", 8), width=24
        )
        csearch.pack(side="left", fill="x", expand=True, ipady=5)
        csearch.bind("<KeyRelease>", self._cycle_filter_class_rows)
        tk.Button(
            ctools, text=self._cycle_text("Ler classes", "Load classes"),
            command=self._cycle_scan_classes, bg=C["purple"], fg="white",
            relief="flat", padx=12, pady=5, font=("Sans", 8, "bold")
        ).pack(side="right", padx=(6, 0))

        dist = tk.Frame(cbody, bg=C["card"])
        dist.pack(fill="x", pady=(0, 6))
        tk.Label(
            dist, text=self._cycle_text("Distribuição por classe:", "Distribution per class:"),
            bg=C["card"], fg=C["text"], font=("Sans", 8, "bold")
        ).pack(side="left", padx=(0, 5))
        for text, value in [("Número de imagens (N)", "count"), ("Percentual (%)", "percent")]:
            tk.Radiobutton(
                dist, text=self._cycle_text(text, text), variable=self._cycle_distribution_mode,
                value=value, command=self._cycle_distribution_changed, indicatoron=True,
                bg=C["card"], fg=C["text"], selectcolor=C["blue2"],
                activebackground=C["card"], activeforeground="white", font=("Sans", 8)
            ).pack(side="left", padx=(0, 8))
        tk.Label(dist, text=self._cycle_text("Aplicar padrão:", "Apply default:"),
                 bg=C["card"], fg=C["muted"], font=("Sans", 8)).pack(side="left", padx=(4, 3))
        for var in (self._cycle_default_train, self._cycle_default_valid, self._cycle_default_test):
            tk.Entry(
                dist, textvariable=var, width=5, bg=C["entry"], fg=C["text"],
                insertbackground="white", relief="flat", justify="center"
            ).pack(side="left", padx=2, ipady=3)
        tk.Button(
            dist, text=self._cycle_text("Aplicar", "Apply"),
            command=self._cycle_apply_defaults, bg=C["blue2"], fg="white",
            relief="flat", padx=9, pady=4
        ).pack(side="right")

        hdr = tk.Frame(cbody, bg="#131315")
        hdr.pack(fill="x")
        # Keep the header on the exact same grid used by the class rows below.
        # This avoids the visual drift caused by mixing Label/Button/Entry widths
        # with pack(), especially across Windows/Linux font metrics.
        class_columns = [
            (0, 42, 0),   # Usar
            (1, 22, 0),   # color swatch
            (2, 135, 1),  # original class
            (3, 135, 1),  # model name
            (4, 72, 0),   # available
            (5, 58, 0),   # train
            (6, 62, 0),   # valid
            (7, 58, 0),   # test
            (8, 68, 0),   # unused
        ]
        for col, minsize, weight in class_columns:
            hdr.columnconfigure(col, minsize=minsize, weight=weight)
        headers = [
            (0, self._cycle_text("Usar", "Use"), "center"),
            (1, "", "center"),
            (2, self._cycle_text("Classe original", "Original class"), "w"),
            (3, self._cycle_text("Nome no modelo", "Model name"), "w"),
            (4, self._cycle_text("Disponível", "Available"), "e"),
            (5, self._cycle_text("Treino", "Train"), "center"),
            (6, self._cycle_text("Validação", "Validation"), "center"),
            (7, self._cycle_text("Teste", "Test"), "center"),
            (8, self._cycle_text("Não usado", "Unused"), "e"),
        ]
        for col, text, anchor in headers:
            tk.Label(
                hdr, text=text, bg="#131315", fg="#c9c9ce",
                font=("Sans", 7, "bold"), anchor=anchor
            ).grid(row=0, column=col, sticky="ew", padx=2, pady=5)
        self._cycle_class_scroll = ScrollFrame(cbody, C["card"])
        self._cycle_class_scroll.pack(fill="both", expand=True)
        self._cycle_class_scroll.configure(height=135)
        self._cycle_class_scroll.pack_propagate(False)
        self._cycle_class_rows_host = self._cycle_class_scroll.inner
        self._cycle_class_empty = tk.Label(
            self._cycle_class_rows_host,
            text=self._cycle_text(
                "Selecione batches e clique em “Ler classes”.",
                "Select batches and click “Load classes”."
            ),
            bg=C["card"], fg=C["muted"], font=("Sans", 8)
        )
        self._cycle_class_empty.pack(anchor="w", pady=10)

        # ------------------------------------------------------------------
        # Right summary sidebar, same visual hierarchy as the approved mockup.
        # ------------------------------------------------------------------
        scard = tk.Frame(page, bg="#18181b", highlightbackground="#17608f", highlightthickness=1)
        scard.grid(row=3, column=2, sticky="nsew", padx=(4, 6), pady=4)
        tk.Label(scard, text=self._cycle_text("Resumo da seleção", "Selection summary"),
                 bg="#18181b", fg="#7dc8ff", font=("Sans", 9, "bold"),
                 anchor="w").pack(fill="x", padx=10, pady=(10, 6))
        self._cycle_summary_metrics = {}
        for key, label in [
            ("batches", self._cycle_text("Batches (treino)", "Batches (train)")),
            ("images", self._cycle_text("Imagens (treino)", "Images (train)")),
            ("classes", self._cycle_text("Classes selecionadas", "Selected classes")),
            ("finals", self._cycle_text("Classes finais", "Final classes")),
            ("dataset", self._cycle_text("Imagens no dataset", "Images in dataset")),
        ]:
            row = tk.Frame(scard, bg="#18181b")
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=label, bg="#18181b", fg="#e6e6e8",
                     font=("Sans", 8), anchor="w").pack(side="left")
            var = tk.StringVar(value="0")
            self._cycle_summary_metrics[key] = var
            tk.Label(row, textvariable=var, bg="#18181b", fg="white",
                     font=("Sans", 8, "bold"), anchor="e").pack(side="right")
        tk.Frame(scard, bg=C["border"], height=1).pack(fill="x", padx=9, pady=8)
        self._cycle_summary_split_vars = {}
        for key, label, color in [
            ("train", self._cycle_text("Treino", "Train"), "#18d39a"),
            ("valid", self._cycle_text("Validação", "Validation"), "#3d9cff"),
            ("test", self._cycle_text("Teste", "Test"), "#a861ff"),
            ("unused", self._cycle_text("Não utilizadas", "Unused"), "#7890a8"),
        ]:
            row = tk.Frame(scard, bg="#18181b")
            row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text="■", bg="#18181b", fg=color, font=("Sans", 10)).pack(side="left")
            tk.Label(row, text=label, bg="#18181b", fg="#e6e6e8",
                     font=("Sans", 8)).pack(side="left", padx=4)
            var = tk.StringVar(value="0")
            self._cycle_summary_split_vars[key] = var
            tk.Label(row, textvariable=var, bg="#18181b", fg="white",
                     font=("Sans", 8, "bold")).pack(side="right")
        # Compatibility target for the existing summary refresher.
        self._cycle_summary_label = tk.Label(scard, text="", bg="#18181b", fg="#18181b",
                                             font=("Sans", 1))
        self._cycle_summary_label.pack_forget()

        # ------------------------------------------------------------------
        # 3. Training, spanning the two main columns.
        # ------------------------------------------------------------------
        tcard, tbody = self._cycle_section(
            page,
            self._cycle_text("Treinamento do novo modelo", "New model training"),
            3,
            self._cycle_text(
                "Treina um novo modelo usando apenas os batches e classes selecionados.",
                "Trains a new model using only the selected batches and classes."
            )
        )
        tcard.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=(6, 4), pady=4)
        tbody.columnconfigure(0, weight=42)
        tbody.columnconfigure(1, weight=58)

        param_box = tk.Frame(tbody, bg="#161619", highlightbackground=C["border"], highlightthickness=1)
        param_box.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        tk.Label(param_box, text=self._cycle_text("Parâmetros do treinamento", "Training parameters"),
                 bg="#161619", fg="#7dc8ff", font=("Sans", 9, "bold")).pack(anchor="w", padx=9, pady=(8, 5))

        top_params = tk.Frame(param_box, bg=C["entry"])
        top_params.pack(fill="x", padx=9)
        tk.Label(
            top_params,
            text=self._cycle_text(
                "Modelo base: ConvNeXt-Tiny    |    Dispositivo: Automático (CUDA se disponível)",
                "Base model: ConvNeXt-Tiny    |    Device: Automatic (CUDA when available)"
            ),
            bg=C["entry"], fg=C["text"], font=("Sans", 8), anchor="w", padx=7, pady=5
        ).pack(fill="x")

        fields_row = tk.Frame(param_box, bg="#161619")
        fields_row.pack(fill="x", padx=9, pady=(7, 0))
        for label, var, width in [
            (self._cycle_text("Épocas", "Epochs"), self._cycle_epochs, 7),
            (self._cycle_text("Tamanho do batch", "Batch size"), self._cycle_batch_size, 7),
            (self._cycle_text("Taxa de aprendizado", "Learning rate"), self._cycle_lr, 9),
            ("Patience", self._cycle_patience, 7),
            ("Seed", self._cycle_seed, 7),
            (self._cycle_text("Versão", "Version"), self._cycle_version, 8),
        ]:
            col = tk.Frame(fields_row, bg="#161619")
            col.pack(side="left", padx=(0, 7))
            tk.Label(col, text=label, bg="#161619", fg=C["muted"], font=("Sans", 7)).pack(anchor="w")
            tk.Entry(col, textvariable=var, width=width, bg=C["entry"], fg=C["text"],
                     insertbackground="white", relief="flat").pack(ipady=4)

        invariant = tk.Frame(param_box, bg="#161619")
        invariant.pack(fill="x", padx=9, pady=(8, 8))
        tk.Label(invariant, text="✓", width=2, bg=C["blue2"], fg="white",
                 font=("Sans", 8, "bold")).pack(side="left")
        tk.Label(
            invariant,
            text=self._cycle_text(
                "Manter manual_label dos batches de treino (não alterar)",
                "Keep manual_label from training batches (do not change)"
            ),
            bg="#161619", fg=C["text"], font=("Sans", 8)
        ).pack(side="left", padx=5)

        progress_box = tk.Frame(tbody, bg="#161619", highlightbackground=C["border"], highlightthickness=1)
        progress_box.grid(row=0, column=1, sticky="nsew")
        tk.Label(progress_box, text=self._cycle_text("Progresso do treinamento", "Training progress"),
                 bg="#161619", fg="#7dc8ff", font=("Sans", 9, "bold")).pack(anchor="w", padx=9, pady=(8, 5))
        prog_line = tk.Frame(progress_box, bg="#161619")
        prog_line.pack(fill="x", padx=9)
        ttk.Progressbar(
            prog_line, maximum=100, variable=self._cycle_stage_progress["training"],
            style="Blue.Horizontal.TProgressbar"
        ).pack(side="left", fill="x", expand=True)
        tk.Label(prog_line, textvariable=self._cycle_train_percent_text,
                 bg="#161619", fg="white", font=("Sans", 8, "bold")).pack(side="right", padx=(8, 0))
        pcontent = tk.Frame(progress_box, bg="#161619")
        pcontent.pack(fill="both", expand=True, padx=9, pady=7)
        tk.Label(
            pcontent, textvariable=self._cycle_train_detail, bg="#161619",
            fg="#e6e6e8", justify="left", anchor="nw", font=("Monospace", 8),
            width=26
        ).pack(side="left", fill="y")
        self._cycle_train_chart = tk.Canvas(
            pcontent, bg="#0f0f12", highlightthickness=0, height=38
        )
        self._cycle_train_chart.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._cycle_train_chart.bind("<Configure>", lambda _e: self._cycle_redraw_training_chart())
        actions = tk.Frame(progress_box, bg="#161619")
        actions.pack(fill="x", padx=9, pady=(0, 8))
        self._cycle_train_btn = tk.Button(
            actions, text=self._cycle_text("⚙  Iniciar treinamento", "⚙  Start training"),
            command=self._cycle_start_training, bg=C["purple"], fg="white",
            relief="flat", font=("Sans", 9, "bold"), padx=13, pady=6
        )
        self._cycle_train_btn.pack(side="left", fill="x", expand=True)
        self._cycle_eval_btn = tk.Button(
            actions, text=self._cycle_text("Ver avaliação do modelo", "View model evaluation"),
            command=self._cycle_open_evaluation, bg="#25252a", fg=C["text"],
            relief="flat", state="disabled", padx=10, pady=6
        )
        self._cycle_eval_btn.pack(side="right", padx=(7, 0))

        training_help = self._cycle_help_panel(
            page, self._cycle_text("O que é cada campo?", "What does each field mean?"),
            [
                (self._cycle_text("Batches de referência", "Reference batches"),
                 self._cycle_text("Somente os batches selecionados entram no novo treinamento.", "Only selected batches enter the new training.")),
                (self._cycle_text("Renomear classes", "Rename classes"),
                 self._cycle_text("Duas classes com o mesmo nome final são agrupadas no novo modelo.", "Two classes with the same final name are grouped in the new model.")),
                (self._cycle_text("Distribuição do dataset", "Dataset split"),
                 self._cycle_text("Train/Valid/Test podem ser definidos por N ou percentual, individualmente por classe.", "Train/Valid/Test can be set by count or percentage per class.")),
                (self._cycle_text("Modelo e parâmetros", "Model and parameters"),
                 self._cycle_text("Mantém o pipeline científico atual; esta tela apenas orquestra o ciclo.", "Keeps the current scientific pipeline; this page only orchestrates the cycle.")),
                (self._cycle_text("Manter manual label", "Keep manual label"),
                 self._cycle_text("Rótulos manuais dos batches de referência nunca são sobrescritos.", "Manual labels from reference batches are never overwritten.")),
            ]
        )
        training_help.grid(row=4, column=2, sticky="nsew", padx=(4, 6), pady=4)

        # ------------------------------------------------------------------
        # 4. Apply model to project.
        # ------------------------------------------------------------------
        acard, abody = self._cycle_section(
            page,
            self._cycle_text("Aplicação do novo modelo ao projeto", "Apply new model to project"),
            4,
            self._cycle_text(
                "Classifique automaticamente os batches escolhidos e aplique o clustering nas classes.",
                "Automatically classify chosen batches and apply clustering to the classes."
            )
        )
        acard.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=(6, 4), pady=4)
        abody.columnconfigure(0, weight=42)
        abody.columnconfigure(1, weight=30)
        abody.columnconfigure(2, weight=28)

        left = tk.Frame(abody, bg="#161619")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(left, text=self._cycle_text("Escopo da nova classificação", "New classification scope"),
                 bg="#161619", fg=C["text"], font=("Sans", 9, "bold")).pack(anchor="w", padx=8, pady=(8, 4))
        for val, pt, en in [
            ("pending", "Todos os batches pendentes para o novo modelo (recomendado)",
             "All batches pending for the new model (recommended)"),
            ("manual", "Selecionar batches manualmente", "Select batches manually"),
        ]:
            tk.Radiobutton(
                left, text=self._cycle_text(pt, en), variable=self._cycle_target_mode,
                value=val, command=self._cycle_target_mode_changed, indicatoron=True,
                anchor="w", justify="left", bg="#161619", fg=C["text"],
                selectcolor=C["blue2"], activebackground="#161619",
                activeforeground="white", font=("Sans", 8), wraplength=390
            ).pack(fill="x", padx=8, pady=2)

        self._cycle_target_manual_host = tk.Frame(left, bg="#161619", height=82)
        self._cycle_target_manual_host.pack(fill="both", expand=True, padx=8, pady=(4, 7))
        self._cycle_target_manual_host.pack_propagate(False)

        middle = tk.Frame(abody, bg="#161619")
        middle.grid(row=0, column=1, sticky="nsew", padx=3)
        tk.Label(middle, text=self._cycle_text("Opções", "Options"),
                 bg="#161619", fg=C["text"], font=("Sans", 9, "bold")).pack(anchor="w", padx=8, pady=(8, 5))

        model_row = tk.Frame(middle, bg="#161619")
        model_row.pack(fill="x", padx=8, pady=(0, 6))
        tk.Label(
            model_row, text=self._cycle_text("Modelo a aplicar", "Model to apply"),
            bg="#161619", fg=C["muted"], font=("Sans", 7)
        ).pack(anchor="w")
        self._cycle_model_combo = ttk.Combobox(
            model_row, textvariable=self._cycle_apply_model, state="readonly",
            style="Dark.TCombobox"
        )
        self._cycle_model_combo.pack(fill="x", pady=(2, 0))
        self._cycle_model_combo.bind("<<ComboboxSelected>>", lambda _e: self._cycle_select_saved_model())

        for text, var in [
            (
                self._cycle_text(
                    "Aplicar clustering automaticamente após a classificação",
                    "Automatically run clustering after classification"
                ),
                self._cycle_auto_cluster,
            ),
            (
                self._cycle_text(
                    "Atualizar também as features dos batches usados no treinamento",
                    "Also refresh features for batches used in training"
                ),
                self._cycle_update_reference_features,
            ),
        ]:
            tk.Checkbutton(
                middle, text=text, variable=var, indicatoron=True, anchor="w",
                justify="left", wraplength=290, bg="#161619", fg=C["text"],
                selectcolor=C["blue2"], activebackground="#161619",
                activeforeground="white", font=("Sans", 8)
            ).pack(fill="x", padx=8, pady=4)
        vr = tk.Frame(middle, bg="#161619")
        vr.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(vr, text=self._cycle_text("Nome da versão", "Version name"),
                 bg="#161619", fg=C["muted"], font=("Sans", 7)).pack(side="left")
        tk.Entry(vr, textvariable=self._cycle_version, width=9, bg=C["entry"], fg=C["text"],
                 insertbackground="white", relief="flat").pack(side="right", ipady=3)

        right = tk.Frame(abody, bg="#161619")
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        tk.Label(right, text=self._cycle_text("Progresso da classificação", "Classification progress"),
                 bg="#161619", fg=C["text"], font=("Sans", 9, "bold")).pack(anchor="w", padx=8, pady=(8, 5))
        cprog = tk.Frame(right, bg="#161619")
        cprog.pack(fill="x", padx=8)
        ttk.Progressbar(
            cprog, maximum=100, variable=self._cycle_stage_progress["classification"],
            style="Blue.Horizontal.TProgressbar"
        ).pack(side="left", fill="x", expand=True)
        tk.Label(cprog, text=" ", bg="#161619", fg=C["text"]).pack(side="right")
        self._cycle_apply_detail = tk.StringVar(value="—")
        tk.Label(
            right, textvariable=self._cycle_apply_detail, bg="#161619",
            fg="#e6e6e8", justify="left", anchor="nw", font=("Monospace", 8),
            wraplength=250
        ).pack(fill="both", expand=True, padx=8, pady=6)
        crow = tk.Frame(right, bg="#161619")
        crow.pack(fill="x", padx=8, pady=(0, 5))
        tk.Label(crow, text=self._cycle_text("Clustering", "Clustering"),
                 bg="#161619", fg=C["muted"], font=("Sans", 7), width=10, anchor="w").pack(side="left")
        ttk.Progressbar(
            crow, maximum=100, variable=self._cycle_stage_progress["cluster"],
            style="Blue.Horizontal.TProgressbar"
        ).pack(side="left", fill="x", expand=True)
        self._cycle_apply_btn = tk.Button(
            right, text=self._cycle_text("▶  Classificar e aplicar", "▶  Classify and apply"),
            command=self._cycle_start_apply, bg="#12c77a", fg="#07140d",
            relief="flat", font=("Sans", 9, "bold"), state="disabled", pady=7
        )
        self._cycle_apply_btn.pack(fill="x", padx=8, pady=(6, 8))

        # Live processing log for this sub-tab: every message emitted by the
        # integrated cycle (train.py output, classification, clustering) is
        # shown here, so the user does not need to switch to the Logs tab.
        how = tk.Frame(page, bg=C["card"], highlightbackground=C["border"], highlightthickness=1)
        how.grid(row=5, column=2, sticky="nsew", padx=(4, 6), pady=4)
        head = tk.Frame(how, bg=C["card"])
        head.pack(fill="x", padx=10, pady=(9, 2))
        tk.Label(head, text="●  " + self._cycle_text("Log em tempo real", "Live log"),
                 bg=C["card"], fg=C["text"], font=("Sans", 9, "bold"),
                 anchor="w").pack(side="left")
        tk.Label(how, text=self._cycle_text(
                     "Treino, classificação e clustering desta subaba em tempo real.",
                     "Training, classification and clustering of this tab in real time."),
                 bg=C["card"], fg=C["muted"], font=("Sans", 7), anchor="w",
                 justify="left", wraplength=240).pack(fill="x", padx=10)
        log_box = tk.Frame(how, bg=C["log"])
        log_box.pack(fill="both", expand=True, padx=8, pady=(5, 4))
        mini = tk.Text(log_box, bg=C["log"], fg=C["text"], relief="flat", bd=0,
                       font=("Monospace", 8), wrap="word", width=34, height=12,
                       padx=6, pady=4, highlightthickness=0)
        mini_scroll = ttk.Scrollbar(log_box, orient="vertical", command=mini.yview)
        mini.configure(yscrollcommand=mini_scroll.set)
        mini_scroll.pack(side="right", fill="y")
        mini.pack(side="left", fill="both", expand=True)
        for tag, col in {"normal": C["text"], "muted": C["muted"], "info": C["cyan"],
                         "success": C["green"], "warn": C["amber"], "error": C["red"],
                         "cluster": C["purple"]}.items():
            mini.tag_configure(tag, foreground=col)
        for msg, tag, stamp in self.log_lines[-150:]:
            mini.insert("end", f"[{stamp}] ", "muted")
            mini.insert("end", msg + "\n", tag if tag in mini.tag_names() else "normal")
        mini.configure(state="disabled")
        mini.see("end")
        self.log = mini
        tk.Button(how, text=self._cycle_text("Limpar log", "Clear log"), command=self.clear_log,
                  bg=C["card2"], fg=C["text"], activebackground=C["border"],
                  activeforeground="white", relief="flat", bd=0, padx=8, pady=1,
                  font=("Sans", 7), cursor="hand2").pack(anchor="e", padx=8, pady=(0, 8))

        footer = tk.Frame(page, bg="#062f2b", highlightbackground="#12d58a", highlightthickness=1)
        footer.grid(row=6, column=0, columnspan=3, sticky="ew", padx=6, pady=(2, 0))
        tk.Label(
            footer, text="↻  " + self._cycle_text("Ciclo de iteração", "Iteration cycle"),
            bg="#062f2b", fg="#14e59a", font=("Sans", 9, "bold")
        ).pack(side="left", padx=10, pady=2)
        tk.Label(
            footer,
            text=self._cycle_text(
                "Após a classificação, revise os resultados na Galeria & Rotulagem. Ajuste classes se necessário e repita o ciclo.",
                "After classification, review results in Gallery & Labeling. Adjust classes if needed and repeat the cycle."
            ),
            bg="#062f2b", fg="#b8d7d0", font=("Sans", 8)
        ).pack(side="left", padx=8)

        self._cycle_populate_batches()
        self._cycle_version.set(next_model_version(self.project.project_dir) if self.project else "v001")
        self._cycle_refresh_saved_models(prefer=self._cycle_trained_version)
        self._cycle_target_mode_changed()
        self._cycle_refresh_summary()
        self._cycle_update_training_detail()

    def _cycle_populate_batches(self):
        if not hasattr(self, "_cycle_batch_list"):
            return
        for w in self._cycle_batch_list.winfo_children():
            w.destroy()
        self._cycle_batch_vars = {}
        self._cycle_batch_row_widgets = {}
        self._cycle_batch_meta = {}
        if not self.project:
            tk.Label(
                self._cycle_batch_list,
                text=self._cycle_text("Abra um projeto primeiro.", "Open a project first."),
                bg=C["card"], fg=C["muted"], font=("Sans", 8)
            ).pack(anchor="w", padx=6, pady=6)
            return

        sf = ScrollFrame(self._cycle_batch_list, C["card"])
        sf.pack(fill="both", expand=True)
        inner = sf.inner

        names = self.project.csv_names()
        for idx, name in enumerate(names):
            var = tk.BooleanVar(value=False)
            self._cycle_batch_vars[name] = var
            row_bg = C["card2"]
            row = tk.Frame(
                inner, bg=row_bg, cursor="hand2",
                highlightbackground=C["border"], highlightthickness=1
            )
            row.pack(fill="x", pady=1, padx=1)
            indicator = tk.Label(
                row, text="", width=3, bg="#1e1e23", fg="white",
                font=("Sans", 9, "bold"), cursor="hand2"
            )
            indicator.pack(side="left", padx=(4, 4), pady=4)

            batch_id = batch_id_from_csv(name)
            try:
                df = self.project.load_batch(name)
                total = len(df)
                if "manual_label" in df.columns:
                    labels = df["manual_label"].fillna("").astype(str).str.strip()
                    validated = int(labels.ne("").sum())
                else:
                    validated = 0
            except Exception:
                total = 0
                validated = 0
            pct = (100.0 * validated / total) if total else 0.0
            status = self._cycle_text("Validado", "Validated") if total and validated >= total else self._cycle_text("Pendente", "Pending")
            status_fg = C["green"] if total and validated >= total else C["amber"]
            self._cycle_batch_meta[name] = {
                "total": total, "validated": validated, "pct": pct,
            }

            batch_lbl = tk.Label(
                row, text=batch_id, bg=row_bg, fg=C["text"],
                font=("Monospace", 8), width=14, anchor="w", cursor="hand2"
            )
            batch_lbl.pack(side="left", padx=1)
            total_lbl = tk.Label(
                row, text=f"{total:,}".replace(",", "."), bg=row_bg, fg=C["muted"],
                font=("Sans", 8), width=7, anchor="e", cursor="hand2"
            )
            total_lbl.pack(side="left", padx=1)
            valid_lbl = tk.Label(
                row, text=f"{validated:,}".replace(",", "."), bg=row_bg, fg=C["muted"],
                font=("Sans", 8), width=8, anchor="e", cursor="hand2"
            )
            valid_lbl.pack(side="left", padx=1)
            pct_lbl = tk.Label(
                row, text=f"{pct:.0f}%", bg=row_bg, fg=C["muted"],
                font=("Sans", 8), width=6, anchor="e", cursor="hand2"
            )
            pct_lbl.pack(side="left", padx=1)
            status_lbl = tk.Label(
                row, text=status, bg=row_bg, fg=status_fg,
                font=("Sans", 8, "bold"), width=9, anchor="center", cursor="hand2"
            )
            status_lbl.pack(side="left", padx=1)

            self._cycle_batch_row_widgets[name] = {
                "row": row, "indicator": indicator,
                "labels": [batch_lbl, total_lbl, valid_lbl, pct_lbl, status_lbl],
            }
            for widget in (row, indicator, batch_lbl, total_lbl, valid_lbl, pct_lbl, status_lbl):
                widget.bind("<Button-1>", lambda _e, bn=name: self._cycle_toggle_batch(bn))

        self._cycle_filter_batch_rows()
        self._cycle_batch_selection_changed()

    def _cycle_set_all_batches(self, value):
        for var in self._cycle_batch_vars.values(): var.set(bool(value))
        self._cycle_batch_selection_changed()

    def _cycle_selected_batches(self):
        return [k for k,v in self._cycle_batch_vars.items() if bool(v.get())]

    def _cycle_batch_selection_changed(self):
        selected = self._cycle_selected_batches()
        total = 0
        if self.project:
            for name in selected:
                try:
                    total += len(self.project.load_batch(name))
                except Exception:
                    pass
        if hasattr(self, "_cycle_batch_summary"):
            self._cycle_batch_summary.configure(
                text=self._cycle_text(
                    f"{len(selected)} selecionados • {total:,} imagens",
                    f"{len(selected)} selected • {total:,} images",
                ).replace(",", ".")
            )
        self._cycle_reference_batches = list(selected)
        self._cycle_refresh_batch_visuals()
        self._cycle_refresh_summary()
        # If manual target selection is active, keep reference batches out of it.
        if getattr(self, "_cycle_target_mode", None) is not None and self._cycle_target_mode.get() == "manual":
            self._cycle_target_mode_changed()

    def _cycle_scan_classes(self):
        if not self.project:
            messagebox.showwarning(self._cycle_text("Aviso","Warning"),self._cycle_text("Abra um projeto primeiro.","Open a project first."));return
        batches=self._cycle_selected_batches()
        if not batches:
            messagebox.showwarning(self._cycle_text("Aviso","Warning"),self._cycle_text("Selecione ao menos um batch.","Select at least one batch."));return
        self._cycle_stage_progress["reference"].set(5)
        def work(progress):
            counts={}
            total=len(batches)
            for i,bn in enumerate(batches,1):
                df=self.project.load_batch(bn)
                vc=df["manual_label"].fillna("").astype(str).str.strip().value_counts()
                for cls,n in vc.items():
                    if cls:counts[cls]=counts.get(cls,0)+int(n)
                progress(i,total,bn)
            return counts
        def done(counts):
            self._cycle_stage_progress["reference"].set(100)
            self._cycle_render_classes(counts)
            self._cycle_status.set(self._cycle_text("Referência carregada.","Reference loaded."))
        self._run_operation(self._cycle_text("Lendo classes dos batches de referência...","Reading reference-batch classes..."),
                            work,done,local_progress=self._cycle_stage_progress["reference"])

    def _cycle_render_classes(self, counts):
        host = self._cycle_class_rows_host
        for w in host.winfo_children():
            w.destroy()
        self._cycle_class_vars = {}
        palette = ["#ff6b6b", "#4bd37b", "#55a7ff", "#f2c14e", "#a56cff", "#f78fb3",
                   "#63d4c7", "#ff9f43", "#7b8cff", "#8ed081", "#e07a5f", "#9d79bc"]

        for index, cls in enumerate(sorted(counts, key=lambda x: (-counts[x], x.lower()))):
            n = int(counts[cls])
            enabled = tk.BooleanVar(value=True)
            final = tk.StringVar(value=cls)
            if self._cycle_distribution_mode.get() == "percent":
                tv, vv, sv = "70", "15", "15"
            else:
                tv = str(int(n * .70))
                vv = str(int(n * .15))
                sv = str(n - int(n * .70) - int(n * .15))
            tr = tk.StringVar(value=tv)
            va = tk.StringVar(value=vv)
            te = tk.StringVar(value=sv)
            unused = tk.StringVar(value="0")

            row = tk.Frame(
                host, bg=C["card2"], highlightbackground=C["border"],
                highlightthickness=1
            )
            row.pack(fill="x", pady=1, padx=1)

            for col, minsize, weight in [
                (0, 42, 0), (1, 22, 0), (2, 135, 1), (3, 135, 1),
                (4, 72, 0), (5, 58, 0), (6, 62, 0), (7, 58, 0), (8, 68, 0),
            ]:
                row.columnconfigure(col, minsize=minsize, weight=weight)

            toggle = tk.Button(
                row, text="✓", bg=C["blue2"], fg="white",
                activebackground=C["blue2"], activeforeground="white",
                relief="flat", command=lambda c=cls: self._cycle_toggle_class(c),
                font=("Sans", 8, "bold")
            )
            toggle.grid(row=0, column=0, sticky="ew", padx=3, pady=3)

            swatch = tk.Label(row, text="■", bg=C["card2"],
                              fg=palette[index % len(palette)], font=("Sans", 10))
            swatch.grid(row=0, column=1, sticky="ew", padx=1)

            cls_lbl = tk.Label(
                row, text=cls, bg=C["card2"], fg=C["text"],
                anchor="w", font=("Sans", 8), cursor="hand2"
            )
            cls_lbl.grid(row=0, column=2, sticky="ew", padx=2)
            cls_lbl.bind("<Button-1>", lambda _e, c=cls: self._cycle_toggle_class(c))
            swatch.bind("<Button-1>", lambda _e, c=cls: self._cycle_toggle_class(c))

            ent = tk.Entry(
                row, textvariable=final, bg=C["entry"], fg=C["text"],
                insertbackground="white", relief="flat", font=("Sans", 8)
            )
            ent.grid(row=0, column=3, sticky="ew", padx=2, pady=3, ipady=2)

            avail = tk.Label(
                row, text=f"{n:,}".replace(",", "."), bg=C["card2"], fg=C["cyan"],
                anchor="e", font=("Sans", 8), cursor="hand2"
            )
            avail.grid(row=0, column=4, sticky="ew", padx=2)
            avail.bind("<Button-1>", lambda _e, c=cls: self._cycle_toggle_class(c))

            for col_index, var in zip((5, 6, 7), (tr, va, te)):
                e = tk.Entry(
                    row, textvariable=var, bg=C["entry"], fg=C["text"],
                    insertbackground="white", relief="flat", font=("Sans", 8),
                    justify="center"
                )
                e.grid(row=0, column=col_index, sticky="ew", padx=2, pady=3, ipady=2)
                var.trace_add("write", lambda *_args: self._cycle_refresh_summary())

            unused_lbl = tk.Label(
                row, textvariable=unused, bg=C["card2"], fg=C["muted"],
                anchor="e", font=("Sans", 8)
            )
            unused_lbl.grid(row=0, column=8, sticky="ew", padx=2)

            self._cycle_class_vars[cls] = {
                "enabled": enabled, "final": final, "train": tr, "valid": va,
                "test": te, "unused": unused, "available": n,
                "_row": row, "_toggle": toggle,
                "_labels": [cls_lbl, avail, unused_lbl], "_swatch": swatch,
            }
            final.trace_add("write", lambda *_args: (self._cycle_refresh_summary(), self._cycle_filter_class_rows()))

        self._cycle_filter_class_rows()
        self._cycle_refresh_summary()

    def _cycle_distribution_changed(self):
        # Keep existing values; only their interpretation changes until Apply is clicked.
        self._cycle_refresh_summary()

    def _cycle_apply_defaults(self):
        mode=self._cycle_distribution_mode.get()
        for cls,s in self._cycle_class_vars.items():
            n=s["available"]
            if mode=="percent":
                vals=(self._cycle_default_train.get(),self._cycle_default_valid.get(),self._cycle_default_test.get())
            else:
                try:
                    ps=[float(self._cycle_default_train.get()),float(self._cycle_default_valid.get()),float(self._cycle_default_test.get())]
                    if sum(ps)<=100:
                        vals=(str(int(n*ps[0]/100)),str(int(n*ps[1]/100)),str(int(n*ps[2]/100)))
                    else:
                        vals=tuple(str(int(x)) for x in ps)
                except Exception:
                    vals=("0","0","0")
            s["train"].set(vals[0]);s["valid"].set(vals[1]);s["test"].set(vals[2])
        self._cycle_refresh_summary()

    def _cycle_class_plan(self):
        plan={}
        mode=self._cycle_distribution_mode.get()
        for cls,s in self._cycle_class_vars.items():
            if not s["enabled"].get():continue
            final=s["final"].get().strip()
            if not final: raise ValueError(f"Classe final vazia: {cls}")
            try:
                train=float(s["train"].get().replace(",","."))
                valid=float(s["valid"].get().replace(",","."))
                test=float(s["test"].get().replace(",","."))
                counts=allocate_counts(s["available"],train,valid,test,mode)
            except Exception as e:
                raise ValueError(f"{cls}: {e}") from e
            plan[cls]={"enabled":True,"final":final,"train":train,"valid":valid,"test":test,"mode":mode,"counts":counts}
        return plan

    def _cycle_refresh_summary(self):
        if not hasattr(self, "_cycle_summary_label"):
            return
        batches = self._cycle_selected_batches() if hasattr(self, "_cycle_batch_vars") else []
        batch_images = 0
        for name in batches:
            info = getattr(self, "_cycle_batch_meta", {}).get(name)
            if info:
                batch_images += int(info.get("total", 0) or 0)
            elif self.project:
                try:
                    batch_images += len(self.project.load_batch(name))
                except Exception:
                    pass

        available = sum(
            s["available"] for s in self._cycle_class_vars.values() if s["enabled"].get()
        ) if self._cycle_class_vars else 0
        finals = set()
        totals = {"train": 0, "valid": 0, "test": 0}
        unused = 0
        enabled_count = 0
        for cls, s in self._cycle_class_vars.items():
            if not s["enabled"].get():
                s["unused"].set(str(s["available"]))
                unused += s["available"]
                continue
            enabled_count += 1
            finals.add(s["final"].get().strip() or cls)
            try:
                c = allocate_counts(
                    s["available"],
                    float(s["train"].get().replace(",", ".")),
                    float(s["valid"].get().replace(",", ".")),
                    float(s["test"].get().replace(",", ".")),
                    self._cycle_distribution_mode.get(),
                )
                for key in totals:
                    totals[key] += c[key]
                u = s["available"] - sum(c.values())
                s["unused"].set(str(u))
                unused += u
            except Exception:
                s["unused"].set("!")
                unused += s["available"]

        # Keep the legacy single-label summary updated for compatibility, but
        # drive the approved right-hand summary with dedicated variables.
        text = self._cycle_text(
            f"Batches: {len(batches)} • Classes: {enabled_count} → {len([x for x in finals if x])} finais • "
            f"Disponíveis: {available:,} • Train: {totals['train']:,} • Valid: {totals['valid']:,} • "
            f"Test: {totals['test']:,} • Não usadas: {unused:,}",
            f"Batches: {len(batches)} • Classes: {enabled_count} → {len([x for x in finals if x])} final • "
            f"Available: {available:,} • Train: {totals['train']:,} • Valid: {totals['valid']:,} • "
            f"Test: {totals['test']:,} • Unused: {unused:,}",
        ).replace(",", ".")
        try:
            self._cycle_summary_label.configure(text=text)
        except Exception:
            pass

        metrics = getattr(self, "_cycle_summary_metrics", {})
        values = {
            "batches": str(len(batches)),
            "images": f"{batch_images:,}".replace(",", "."),
            "classes": str(enabled_count),
            "finals": str(len([x for x in finals if x])),
            "dataset": f"{available:,}".replace(",", "."),
        }
        for key, value in values.items():
            try:
                metrics[key].set(value)
            except Exception:
                pass

        split_vars = getattr(self, "_cycle_summary_split_vars", {})
        selected_total = max(1, totals["train"] + totals["valid"] + totals["test"] + unused)
        for key, count in [
            ("train", totals["train"]),
            ("valid", totals["valid"]),
            ("test", totals["test"]),
            ("unused", unused),
        ]:
            pct = 100.0 * count / selected_total
            value = f"{count:,} ({pct:.0f}%)".replace(",", ".")
            try:
                split_vars[key].set(value)
            except Exception:
                pass

        self._cycle_refresh_class_visuals()

    def _cycle_load_train_config(self):
        candidates=[
            Path(getattr(self.app,"src_dir",Path(__file__).resolve().parent)).parent/"input"/"train_params.json",
            Path(__file__).resolve().parent.parent/"input"/"train_params.json",
        ]
        cfg={}
        for p in candidates:
            if p.is_file():
                try:cfg=json.loads(p.read_text(encoding="utf-8"));break
                except Exception:pass
        cfg.update({
            "lr":float(self._cycle_lr.get().replace(",",".")),
            "numEpochs":int(self._cycle_epochs.get()),
            "batchSize":int(self._cycle_batch_size.get()),
            "patience":int(self._cycle_patience.get()),
            "seed":int(self._cycle_seed.get()),
            "readModel":"F",
            "trainFolder":"train","validFolder":"valid","testFolder":"test",
        })
        cfg.setdefault("lossName","focal");cfg.setdefault("focalGamma",2);cfg.setdefault("validMetric","valid_macro")
        cfg.setdefault("numWorkers",-1)
        # Calibration metadata of the new model = the project's pixel size (the
        # value the IAT uses), not the 12.87 µm/pixel program default.
        from project_settings import DEFAULT_PIXEL_SIZE_UM_PER_PX,parse_pixel_size
        try:cfg["pixelSize"]=parse_pixel_size(self.vars["pixel_size"].get())
        except Exception:cfg.setdefault("pixelSize",DEFAULT_PIXEL_SIZE_UM_PER_PX)
        return cfg

    def _cycle_start_training(self):
        if not self.project:return
        batches=self._cycle_selected_batches()
        if not batches:
            messagebox.showerror(self._cycle_text("Erro","Error"),self._cycle_text("Selecione batches de referência.","Select reference batches."));return
        try:plan=self._cycle_class_plan()
        except Exception as e:messagebox.showerror(self._cycle_text("Erro","Error"),str(e));return
        if not plan:
            messagebox.showerror(self._cycle_text("Erro","Error"),self._cycle_text("Nenhuma classe selecionada.","No class selected."));return
        version=self._cycle_version.get().strip() or next_model_version(self.project.project_dir)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+",version):
            messagebox.showerror(self._cycle_text("Erro","Error"),self._cycle_text("Nome de versão inválido.","Invalid version name."));return
        model_dir=Path(self.project.project_dir)/"models"/version
        if model_dir.exists() and any(model_dir.iterdir()):
            messagebox.showerror(self._cycle_text("Erro","Error"),self._cycle_text("Essa versão já existe. Use outro nome.","That version already exists. Use another name."));return
        try:cfg=self._cycle_load_train_config()
        except Exception as e:messagebox.showerror(self._cycle_text("Erro","Error"),str(e));return

        def work(progress):
            self._cycle_stage_progress["reference"].set(0)
            assignments=build_reference_assignments(self.project,batches,plan,seed=cfg["seed"],progress=progress)
            if not assignments:raise RuntimeError(self._cycle_text("Nenhuma imagem válida encontrada.","No valid images found."))
            summary=summarize_assignments(assignments)
            for cls,counts in summary.items():
                if counts.get("train",0)<=0:
                    raise RuntimeError(f"{cls}: sem imagens em Train.")
            temp=materialize_reference_dataset(assignments)
            self._cycle_temp_dataset=temp
            model_dir.mkdir(parents=True,exist_ok=True)
            write_cycle_manifest(assignments,model_dir,version)
            cfg["numClasses"]=len(summary)
            params=model_dir/f"training_config_{version}.json"
            params.write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
            save_cycle_metadata(Path(self.project.project_dir),version,{
                "version":version,"status":"training","reference_batches":batches,
                "class_plan":plan,"model_dir":str(model_dir)
            })
            self._event_q.put(("cycle_training_ready",(temp,model_dir,params,version,cfg,batches,plan)))
            return None

        self._cycle_status.set(self._cycle_text("Preparando dataset de referência...","Preparing reference dataset..."))
        self._run_operation(self._cycle_text("Preparando referência...","Preparing reference..."),work,
                            local_progress=self._cycle_stage_progress["reference"])

    def _cycle_launch_training_process(self, payload):
        temp,model_dir,params,version,cfg,batches,plan=payload
        self._cycle_stage_progress["reference"].set(100)
        self._cycle_stage_progress["training"].set(0)
        self._cycle_train_history = {"train": [], "valid": []}
        self._cycle_train_metrics = {}
        self._cycle_train_epoch = 0
        self._cycle_train_elapsed = ""
        try:
            self._cycle_train_percent_text.set("0%")
            self._cycle_train_detail.set("—")
            self._cycle_redraw_training_chart()
        except Exception:
            pass
        cmd=worker_command("training",str(temp),str(model_dir),version,str(params),src_dir=getattr(self.app,"src_dir",Path(__file__).resolve().parent))
        self._cycle_status.set(self._cycle_text(f"Treinando {version}... ",f"Training {version}..."))
        self._cycle_train_btn.configure(state="disabled")
        def run():
            ok=False;lines=[]
            try:
                # No console for the worker: stdin closed and ILT_NONINTERACTIVE
                # set, so train.py can never block on a prompt.
                proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                      text=True,bufsize=1,env=dict(os.environ,ILT_NONINTERACTIVE="1",PYTHONUNBUFFERED="1"))
                self._cycle_train_proc=proc
                epochs=max(1,int(cfg.get("numEpochs",1)))
                for line in proc.stdout or []:
                    lines.append(line)
                    m=re.search(r"Epoch\s+(\d+)",line)
                    if m:
                        pct=min(98,100*int(m.group(1))/epochs)
                        self._event_q.put(("cycle_training_progress",pct))
                    if line.strip():
                        self._event_q.put(("cycle_log",line.rstrip()))
                rc=proc.wait();ok=(rc==0)
            except Exception as e:
                lines.append(str(e));rc=1
            self._event_q.put(("cycle_training_done",(ok,rc,model_dir,version,batches,plan,"\n".join(lines[-30:]))))
        threading.Thread(target=run,daemon=True).start()

    def _cycle_widget_exists(self, attribute):
        """A rebuilt IAT tab can leave references to destroyed Tk widgets."""
        widget = getattr(self, attribute, None)
        if widget is None:
            return False
        try:
            return bool(widget.winfo_exists())
        except (tk.TclError, RuntimeError):
            return False

    def _cycle_set_button_state(self, attribute, state):
        """Update a cycle button only while its Tk instance is alive."""
        if not self._cycle_widget_exists(attribute):
            return
        try:
            getattr(self, attribute).configure(state=state)
        except (tk.TclError, RuntimeError):
            # The UI may be rebuilt between processing queued worker events.
            pass

    def _cycle_training_finished(self, payload):
        ok,rc,model_dir,version,batches,plan,tail=payload
        self._cycle_set_button_state("_cycle_train_btn", "normal")
        cleanup_training_dataset_view(self._cycle_temp_dataset);self._cycle_temp_dataset=None
        if not ok:
            self._cycle_stage_progress["training"].set(0)
            self._cycle_status.set(self._cycle_text("Falha no treinamento.","Training failed."))
            messagebox.showerror(self._cycle_text("Treinamento","Training"),tail or f"exit={rc}")
            return
        model=model_dir/f"model_{version}"
        labels=model_dir/f"labels_{version}.json"
        if not model.exists() or not labels.exists():
            self._cycle_status.set(self._cycle_text("Treinamento terminou, mas os artefatos não foram encontrados.","Training finished but artifacts were not found."))
            return
        self._cycle_stage_progress["training"].set(100)
        self._cycle_trained_version=version;self._cycle_model_path=model;self._cycle_labels_path=labels
        self._cycle_status.set(self._cycle_text(f"{version} treinado e pronto para aplicar.",f"{version} trained and ready to apply."))
        try:
            save_cycle_metadata(Path(self.project.project_dir),version,{
                "version":version,"status":"trained","reference_batches":batches,
                "model_path":str(model),"labels_path":str(labels),"class_plan":plan
            })
        except Exception:pass
        # Training completion and metadata must not depend on the active tab.
        # show_tab() destroys the old buttons whenever the user navigates.
        self._cycle_set_button_state("_cycle_apply_btn", "normal")
        self._cycle_set_button_state("_cycle_eval_btn", "normal")
        if (getattr(self, "current_tab", None) == "model_cycle"
                and self._cycle_widget_exists("_cycle_model_combo")
                and self._cycle_widget_exists("_cycle_target_manual_host")):
            self._cycle_refresh_saved_models(prefer=version)
            self._cycle_target_mode_changed()

    def _cycle_open_evaluation(self):
        if not self._cycle_trained_version or not self._cycle_model_path:
            return
        try:
            from training_results import open_training_results
            open_training_results(
                self,
                self._cycle_model_path.parent,
                self._cycle_trained_version,
                lang=self.lang(),
                open_folder_callback=lambda: None,
            )
        except Exception as exc:
            messagebox.showerror(self._cycle_text("Avaliação","Evaluation"),str(exc))


    def _cycle_saved_model_entries(self):
        """Return saved model artifacts keyed by version for the current project."""
        entries = {}
        if not self.project:
            return entries
        models_dir = Path(self.project.project_dir) / "models"
        if not models_dir.is_dir():
            return entries
        for model_dir in sorted((p for p in models_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            version = model_dir.name
            model = model_dir / f"model_{version}"
            labels = model_dir / f"labels_{version}.json"
            if model.is_file() and labels.is_file():
                entries[version] = (model, labels)
        return entries

    def _cycle_reference_batches_for_version(self, version=None):
        """Return persisted reference batches for a model version when available."""
        if not self.project:
            return self._cycle_selected_batches()
        version = str(version or getattr(self, "_cycle_trained_version", None) or "").strip()
        if not version:
            return self._cycle_selected_batches()
        project_dir = Path(self.project.project_dir)
        candidates = [
            project_dir / "model_history" / f"{version}.json",
            project_dir / "active_model.json",
        ]
        for path in candidates:
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if path.name == "active_model.json" and str(payload.get("active_model") or "") != version:
                continue
            batches = payload.get("reference_batches")
            if isinstance(batches, list):
                return [str(x) for x in batches]
        return self._cycle_selected_batches()

    def _cycle_refresh_saved_models(self, prefer=None):
        """Populate the reusable model selector without requiring a new training run."""
        if not hasattr(self, "_cycle_model_combo"):
            return
        entries = self._cycle_saved_model_entries()
        versions = list(entries)
        self._cycle_model_combo["values"] = versions
        if not versions:
            self._cycle_apply_model.set("")
            return

        selected = str(prefer or "").strip()
        if selected not in entries:
            active_path = Path(self.project.project_dir) / "active_model.json" if self.project else None
            if active_path and active_path.is_file():
                try:
                    active = json.loads(active_path.read_text(encoding="utf-8"))
                    selected = str(active.get("active_model") or "")
                except Exception:
                    selected = ""
        if selected not in entries:
            selected = versions[-1]
        self._cycle_apply_model.set(selected)
        self._cycle_select_saved_model(refresh_targets=False)

    def _cycle_select_saved_model(self, refresh_targets=True):
        """Load paths/state for an already trained project-local model version."""
        version = str(self._cycle_apply_model.get() or "").strip()
        entries = self._cycle_saved_model_entries()
        if version not in entries:
            return
        model, labels = entries[version]
        self._cycle_trained_version = version
        self._cycle_model_path = model
        self._cycle_labels_path = labels
        try:
            self._cycle_apply_btn.configure(state="normal")
            self._cycle_eval_btn.configure(state="normal")
        except Exception:
            pass
        self._cycle_status.set(
            self._cycle_text(
                f"{version} carregado e pronto para aplicar.",
                f"{version} loaded and ready to apply.",
            )
        )
        if refresh_targets:
            self._cycle_target_mode_changed()

    def _cycle_target_mode_changed(self):
        if not hasattr(self, "_cycle_target_manual_host"):
            return
        for w in self._cycle_target_manual_host.winfo_children():
            w.destroy()
        self._cycle_target_vars = {}
        self._cycle_target_row_widgets = {}

        if not self.project:
            return

        if self._cycle_target_mode.get() != "manual":
            try:
                self._cycle_target_manual_host.configure(height=24)
            except Exception:
                pass
            tk.Label(
                self._cycle_target_manual_host,
                text=self._cycle_text(
                    "Modo automático: serão usados todos os batches que não fazem parte da referência.",
                    "Automatic mode: all batches outside the reference set will be used."
                ),
                bg="#161619", fg=C["muted"], font=("Sans", 8),
                justify="left", wraplength=760
            ).pack(anchor="w", padx=4, pady=6)
            return

        try:
            self._cycle_target_manual_host.configure(height=115)
        except Exception:
            pass
        refs = set(self._cycle_reference_batches_for_version())
        available = [name for name in self.project.csv_names() if name not in refs]

        actions = tk.Frame(self._cycle_target_manual_host, bg="#161619")
        actions.pack(fill="x", pady=(0, 5))
        tk.Button(
            actions, text=self._cycle_text("Selecionar todos", "Select all"),
            command=lambda: self._cycle_set_all_targets(True),
            bg=C["card2"], fg=C["text"], relief="flat", padx=9, pady=3
        ).pack(side="left")
        tk.Button(
            actions, text=self._cycle_text("Desmarcar todos", "Clear all"),
            command=lambda: self._cycle_set_all_targets(False),
            bg=C["card2"], fg=C["text"], relief="flat", padx=9, pady=3
        ).pack(side="left", padx=5)

        list_host = tk.Frame(self._cycle_target_manual_host, bg="#161619", height=70)
        list_host.pack(fill="both", expand=True)
        list_host.pack_propagate(False)
        sf = ScrollFrame(list_host, "#161619")
        sf.pack(fill="both", expand=True)
        inner = sf.inner

        for name in available:
            var = tk.BooleanVar(value=True)
            self._cycle_target_vars[name] = var
            row = tk.Frame(
                inner, bg="#123a5e", cursor="hand2",
                highlightbackground=C["blue"], highlightthickness=1
            )
            row.pack(fill="x", padx=2, pady=2)
            indicator = tk.Label(
                row, text="✓", width=3, bg=C["blue2"], fg="white",
                font=("Sans", 9, "bold"), cursor="hand2"
            )
            indicator.pack(side="left", padx=(4, 6), pady=4)
            name_lbl = tk.Label(
                row, text=batch_id_from_csv(name), bg="#123a5e", fg="white",
                font=("Monospace", 8), anchor="w", cursor="hand2"
            )
            name_lbl.pack(side="left", fill="x", expand=True, padx=2)
            self._cycle_target_row_widgets[name] = {
                "row": row, "indicator": indicator, "labels": [name_lbl]
            }
            for widget in (row, indicator, name_lbl):
                widget.bind("<Button-1>", lambda _e, bn=name: self._cycle_toggle_target(bn))

        if not available:
            tk.Label(
                inner,
                text=self._cycle_text("Nenhum batch disponível fora da referência.", "No batch available outside the reference set."),
                bg=C["card"], fg=C["muted"], font=("Sans", 8)
            ).pack(anchor="w", padx=5, pady=5)

    def _cycle_target_batches(self):
        refs=set(self._cycle_reference_batches_for_version())
        if not self.project:return []
        if self._cycle_target_mode.get()=="manual":
            return [n for n,v in self._cycle_target_vars.items() if v.get() and n not in refs]
        return [n for n in self.project.csv_names() if n not in refs]

    def _cycle_start_apply(self):
        if not self.project or not self._cycle_model_path or not self._cycle_labels_path:
            messagebox.showwarning(self._cycle_text("Aviso","Warning"),self._cycle_text("Treine um modelo primeiro.","Train a model first."));return
        targets=self._cycle_target_batches();refs=self._cycle_reference_batches_for_version()
        if not targets and not self._cycle_update_reference_features.get():
            messagebox.showwarning(self._cycle_text("Aviso","Warning"),self._cycle_text("Nenhum batch selecionado para classificação.","No batch selected for classification."));return
        apply_batches=list(targets)
        preserve=[]
        if self._cycle_update_reference_features.get():
            for b in refs:
                if b not in apply_batches:apply_batches.append(b)
            preserve=list(refs)
        version=self._cycle_trained_version
        auto_cluster=bool(self._cycle_auto_cluster.get())
        self._cycle_stage_progress["classification"].set(0);self._cycle_stage_progress["cluster"].set(0)
        self._cycle_apply_btn.configure(state="disabled")
        self._cycle_status.set(self._cycle_text("Aplicando novo modelo ao projeto...","Applying new model to project..."))

        def work(progress):
            def cb(i,total,bn,done,total_images):
                pct=100*i/max(1,total)
                self._event_q.put(("cycle_apply_progress",(pct,bn,i,total,done,total_images)))
            result=apply_model_in_place(
                self.project,apply_batches,self._cycle_model_path,self._cycle_labels_path,version,
                preserve_manual_batches=preserve,progress=cb
            )
            self._event_q.put(("cycle_log",self._cycle_text(
                f"Recalculando projeção t-SNE para {version}...",
                f"Recomputing t-SNE projection for {version}..."
            )))
            projection=recompute_projection_for_version(self.project,version)
            result["projection"]=projection
            return result

        def done(result):
            self._cycle_stage_progress["classification"].set(100)
            if auto_cluster:
                self._cycle_run_clustering(version,result,targets,refs)
            else:
                self._cycle_apply_btn.configure(state="normal")
                self._cycle_status.set(self._cycle_text("Classificação concluída.","Classification completed."))
                self._cycle_finalize_version(version,result,clustered=False)
                self._cycle_reload_project_after_apply()
        self._run_operation(self._cycle_text("Classificando batches do projeto...","Classifying project batches..."),
                            work,done,local_progress=self._cycle_stage_progress["classification"])

    def _cycle_run_clustering(self,version,result,targets,refs):
        self._cycle_stage_progress["cluster"].set(3)
        self._cycle_status.set(self._cycle_text("Classificação concluída. Aplicando clustering...","Classification complete. Running clustering..."))
        cmd=worker_command("clustering",self.project.name,"--project-dir",str(self.project.project_dir),
                           "--feature-version",str(version),
                           src_dir=getattr(self.app,"src_dir",Path(__file__).resolve().parent))
        def run():
            ok=False;tail=[]
            try:
                proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                      text=True,bufsize=1,env=dict(os.environ,ILT_NONINTERACTIVE="1",PYTHONUNBUFFERED="1"))
                seen=0;total=0
                for line in proc.stdout or []:
                    tail.append(line.rstrip())
                    m=re.search(r"rodando para\s+(\d+)\s+classes",line,re.I)
                    if m:total=int(m.group(1))
                    if "[clustering] class=" in line:
                        seen+=1;pct=min(98,100*seen/max(1,total or seen))
                        self._event_q.put(("cycle_cluster_progress",pct))
                    if line.strip():self._event_q.put(("cycle_log",line.rstrip()))
                rc=proc.wait();ok=(rc==0)
            except Exception as e:
                tail.append(str(e));rc=1
            self._event_q.put(("cycle_cluster_done",(ok,rc,version,result,"\n".join(tail[-30:]))))
        threading.Thread(target=run,daemon=True).start()

    def _cycle_cluster_finished(self,payload):
        ok,rc,version,result,tail=payload
        self._cycle_set_button_state("_cycle_apply_btn", "normal")
        if ok:
            self._cycle_stage_progress["cluster"].set(100)
            self._cycle_status.set(self._cycle_text("Classificação e clustering concluídos.","Classification and clustering completed."))
            self._cycle_finalize_version(version,result,clustered=True)
            self._cycle_reload_project_after_apply()
        else:
            self._cycle_status.set(self._cycle_text("Classificação concluída; clustering falhou.","Classification completed; clustering failed."))
            self._cycle_finalize_version(version,result,clustered=False)
            messagebox.showwarning(self._cycle_text("Clustering","Clustering"),tail or f"exit={rc}")
            self._cycle_reload_project_after_apply()

    def _cycle_finalize_version(self,version,result,clustered):
        project_dir=Path(self.project.project_dir)
        manifest_path=Path(str(result.get("manifest",""))) if result.get("manifest") else None
        if manifest_path is not None:
            try:
                manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"Não foi possível validar o manifesto da aplicação: {exc}") from exc
            if manifest.get("status")!="complete":
                raise RuntimeError(
                    f"A versão {version} não pode ser ativada: manifesto de aplicação está em "
                    f"status={manifest.get('status')!r}."
                )
        active={
            "active_model":version,
            "model_path":str(self._cycle_model_path),
            "labels_path":str(self._cycle_labels_path),
            "clustered":bool(clustered),
            "updated_images":int(result.get("images",0)),
            "application_manifest":str(manifest_path) if manifest_path is not None else None,
            "reference_batches":self._cycle_reference_batches_for_version(version),
            "target_batches":self._cycle_target_batches(),
            "updated_at":time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            (project_dir/"model_history").mkdir(parents=True,exist_ok=True)
            (project_dir/"active_model.json").write_text(json.dumps(active,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
            save_cycle_metadata(project_dir,version,{**active,"status":"active"})
        except Exception:pass

    def _cycle_reload_project_after_apply(self):
        """Reload project state immediately while preserving the user's current batch."""
        if not self.project:
            return
        try:
            from iat_backend import IATProject
            project_dir = Path(self.project.project_dir)
            current_batch = str(self.vars["batch"].get() or "")
            fresh = IATProject(project_dir)
            csvs = fresh.csv_names()
            self.project = fresh
            self.visible_classes.clear()
            self.vars["overview_multi_classes"].set(False)
            self.batch_combo["values"] = ["Overview"] + csvs
            if current_batch == "Overview":
                selected = "Overview"
            elif current_batch in csvs:
                selected = current_batch
            else:
                selected = csvs[0] if csvs else "Overview"
            self.vars["batch"].set(selected)
            self.mode_lbl.config(text="Overview" if selected == "Overview" else "Batch")
            self._cycle_refresh_saved_models(prefer=self._cycle_trained_version)
            # change_batch reloads the dataframe from disk, clears scatter/gallery
            # selections/caches and rebuilds the active tab.  Returning to Scatter
            # therefore uses the freshly written labels and x/y coordinates.
            self.change_batch()
        except Exception as exc:
            self._log(f"Falha ao atualizar projeto após aplicação do modelo: {exc}", "error")

    def _cycle_training_progress_event(self,pct):
        value = float(pct)
        self._cycle_stage_progress["training"].set(value)
        try:
            self._cycle_train_percent_text.set(f"{value:.0f}%")
        except Exception:
            pass
        try:
            epochs = max(1, int(float(self._cycle_epochs.get() or 1)))
            self._cycle_train_epoch = max(self._cycle_train_epoch, int(round(epochs * value / 100.0)))
            self._cycle_update_training_detail()
        except Exception:
            pass

    def _cycle_cluster_progress_event(self,pct):
        self._cycle_stage_progress["cluster"].set(float(pct))

    def _cycle_apply_progress_event(self,payload):
        pct,bn,i,total,done,total_images=payload
        self._cycle_stage_progress["classification"].set(pct)
        self._cycle_apply_detail.set(
            f"{self._cycle_text('Batch','Batch')}: {batch_id_from_csv(bn)}\n"
            f"{self._cycle_text('Batches','Batches')}: {i} / {total}\n"
            f"{self._cycle_text('Imagens','Images')}: {done:,} / {total_images:,}".replace(",",".")
        )
