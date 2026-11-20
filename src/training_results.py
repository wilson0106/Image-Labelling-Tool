#!/usr/bin/env python3
"""Scientific, didactic post-training evaluation window for ILT."""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import numpy as np

from ui_resources import COLORS
from training_evaluation import load_training_evaluation

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - surfaced to the caller with a clear message
    FigureCanvasTkAgg = NavigationToolbar2Tk = Figure = None


TEXT = {
    "pt": {
        "title": "Resultados do treinamento",
        "subtitle": "Avaliação no conjunto de teste — matriz de confusão e métricas por classe.",
        "summary": "Resumo",
        "matrix": "Matriz de confusão",
        "classes": "Métricas por classe",
        "accuracy": "Acurácia",
        "macro_f1": "F1-macro",
        "macro_recall": "Recall macro",
        "macro_precision": "Precision macro",
        "bias": "Bias abs. médio",
        "samples": "Imagens de teste",
        "classes_n": "Classes",
        "count": "Contagens",
        "normalized": "% por classe real",
        "true_class": "Classe real",
        "pred_class": "Classe predita",
        "matrix_help": (
            "Linhas = classe real; colunas = classe predita. "
            "Na visualização normalizada, cada linha soma 100%."
        ),
        "bias_help": (
            "Bias de prevalência por classe = prevalência predita − prevalência observada, "
            "em pontos percentuais (p.p.). Valor ideal = 0. "
            "O bias geral exibido é a média do valor absoluto entre as classes."
        ),
        "macro_help": (
            "Precision macro, Recall macro e F1-macro são médias não ponderadas entre as classes, "
            "portanto cada classe tem o mesmo peso independentemente de sua abundância."
        ),
        "support": "N real",
        "predicted": "N predito",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "bias_pp": "Bias (p.p.)",
        "class": "Classe",
        "ideal_zero": "ideal = 0",
        "open_folder": "Abrir pasta",
        "test_loss": "Loss de teste",
    },
    "en": {
        "title": "Training results",
        "subtitle": "Test-set evaluation — confusion matrix and per-class metrics.",
        "summary": "Summary",
        "matrix": "Confusion matrix",
        "classes": "Per-class metrics",
        "accuracy": "Accuracy",
        "macro_f1": "Macro F1",
        "macro_recall": "Macro recall",
        "macro_precision": "Macro precision",
        "bias": "Mean abs. bias",
        "samples": "Test images",
        "classes_n": "Classes",
        "count": "Counts",
        "normalized": "% by true class",
        "true_class": "True class",
        "pred_class": "Predicted class",
        "matrix_help": (
            "Rows = true class; columns = predicted class. "
            "In normalized view, every row sums to 100%."
        ),
        "bias_help": (
            "Per-class prevalence bias = predicted prevalence − observed prevalence, "
            "in percentage points (p.p.). Ideal value = 0. "
            "The overall bias shown is the mean absolute value across classes."
        ),
        "macro_help": (
            "Macro precision, macro recall and macro F1 are unweighted means across classes, "
            "so every class has equal weight regardless of abundance."
        ),
        "support": "True N",
        "predicted": "Predicted N",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "bias_pp": "Bias (p.p.)",
        "class": "Class",
        "ideal_zero": "ideal = 0",
        "open_folder": "Open folder",
        "test_loss": "Test loss",
    },
}


class TrainingResultsWindow(tk.Toplevel):
    """Non-modal results viewer that can be reopened after training."""

    def __init__(self, parent, output_dir, project_name, *, lang="pt", open_folder_callback=None):
        super().__init__(parent)
        self.lang = "en" if str(lang).lower().startswith("en") else "pt"
        self.t = TEXT[self.lang]
        self.output_dir = Path(output_dir)
        self.project_name = str(project_name)
        self.open_folder_callback = open_folder_callback

        self.matrix, self.class_names, self.metrics = load_training_evaluation(
            self.output_dir, self.project_name
        )

        self.title(f"ILT — {self.t['title']} — {self.project_name}")
        self.configure(bg=COLORS["bg"])
        self.geometry("1220x820")
        self.minsize(900, 640)
        self.transient(parent)

        self._matrix_mode = tk.StringVar(value="normalized")
        self._build()
        self.after_idle(self._draw_matrix)

    def _build(self):
        header = tk.Frame(self, bg=COLORS["bg"])
        header.pack(fill="x", padx=18, pady=(16, 8))
        tk.Label(
            header, text=self.t["title"], bg=COLORS["bg"], fg=COLORS["text"],
            font=("Sans", 17, "bold")
        ).pack(anchor="w")
        tk.Label(
            header,
            text=f"{self.t['subtitle']}  •  {self.project_name}",
            bg=COLORS["bg"], fg=COLORS["muted"], font=("Sans", 9)
        ).pack(anchor="w", pady=(3, 0))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.summary_tab = tk.Frame(self.notebook, bg=COLORS["panel"])
        self.matrix_tab = tk.Frame(self.notebook, bg=COLORS["panel"])
        self.class_tab = tk.Frame(self.notebook, bg=COLORS["panel"])
        self.notebook.add(self.summary_tab, text=self.t["summary"])
        self.notebook.add(self.matrix_tab, text=self.t["matrix"])
        self.notebook.add(self.class_tab, text=self.t["classes"])

        self._build_summary()
        self._build_matrix_tab()
        self._build_class_table()

        # The confusion matrix is the primary post-training result.  Open the
        # viewer directly on it and redraw whenever its tab becomes visible,
        # which also avoids blank canvases on some Tk/Matplotlib backends when
        # a figure was initially created while the tab was hidden.
        self.notebook.select(self.matrix_tab)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_results_tab_changed)

    def _on_results_tab_changed(self, _event=None):
        try:
            if self.notebook.select() == str(self.matrix_tab):
                self.after_idle(self._draw_matrix)
        except Exception:
            pass

    def _metric_card(self, parent, title, value, subtitle=""):
        card = tk.Frame(
            parent, bg=COLORS["card"], highlightbackground=COLORS["border"],
            highlightthickness=1
        )
        tk.Label(
            card, text=title, bg=COLORS["card"], fg=COLORS["muted"],
            font=("Sans", 8, "bold")
        ).pack(anchor="w", padx=12, pady=(10, 2))
        tk.Label(
            card, text=value, bg=COLORS["card"], fg=COLORS["cyan"],
            font=("Sans", 20, "bold")
        ).pack(anchor="w", padx=12)
        if subtitle:
            tk.Label(
                card, text=subtitle, bg=COLORS["card"], fg=COLORS["muted"],
                font=("Sans", 7)
            ).pack(anchor="w", padx=12, pady=(1, 9))
        else:
            tk.Frame(card, height=9, bg=COLORS["card"]).pack()
        return card

    def _build_summary(self):
        host = self.summary_tab
        cards = tk.Frame(host, bg=COLORS["panel"])
        cards.pack(fill="x", padx=16, pady=(16, 10))
        for col in range(3):
            cards.columnconfigure(col, weight=1)

        values = [
            (self.t["accuracy"], f"{100*self.metrics['accuracy']:.2f}%", ""),
            (self.t["macro_f1"], f"{self.metrics['macro_f1']:.4f}", ""),
            (self.t["macro_recall"], f"{self.metrics['macro_recall']:.4f}", ""),
            (self.t["macro_precision"], f"{self.metrics['macro_precision']:.4f}", ""),
            (self.t["bias"], f"{self.metrics['mean_abs_bias_pp']:.2f} p.p.", self.t["ideal_zero"]),
            (self.t["samples"], f"{self.metrics['n_samples']:,}".replace(",", "."), f"{self.t['classes_n']}: {self.metrics['n_classes']}"),
        ]
        for idx, (title, value, subtitle) in enumerate(values):
            card = self._metric_card(cards, title, value, subtitle)
            card.grid(row=idx // 3, column=idx % 3, sticky="nsew", padx=5, pady=5)

        if "test_loss" in self.metrics:
            loss = tk.Frame(host, bg=COLORS["card2"])
            loss.pack(fill="x", padx=21, pady=(0, 8))
            tk.Label(
                loss, text=f"{self.t['test_loss']}: {float(self.metrics['test_loss']):.6f}",
                bg=COLORS["card2"], fg=COLORS["text"], font=("Monospace", 9, "bold")
            ).pack(anchor="w", padx=12, pady=8)

        expl = tk.Frame(
            host, bg=COLORS["card"], highlightbackground=COLORS["border"],
            highlightthickness=1
        )
        expl.pack(fill="both", expand=True, padx=21, pady=(4, 16))
        tk.Label(
            expl, text=self.t["macro_help"], bg=COLORS["card"], fg=COLORS["text"],
            justify="left", anchor="w", wraplength=1040, font=("Sans", 9)
        ).pack(fill="x", padx=14, pady=(14, 8))
        tk.Label(
            expl, text=self.t["bias_help"], bg=COLORS["card"], fg=COLORS["muted"],
            justify="left", anchor="w", wraplength=1040, font=("Sans", 9)
        ).pack(fill="x", padx=14, pady=(0, 14))

    def _build_matrix_tab(self):
        controls = tk.Frame(self.matrix_tab, bg=COLORS["panel"])
        controls.pack(fill="x", padx=14, pady=(12, 4))
        tk.Label(
            controls, text=self.t["matrix_help"], bg=COLORS["panel"],
            fg=COLORS["muted"], font=("Sans", 8)
        ).pack(side="left")
        for value, label in (("count", self.t["count"]), ("normalized", self.t["normalized"])):
            tk.Radiobutton(
                controls, text=label, value=value, variable=self._matrix_mode,
                command=self._draw_matrix, bg=COLORS["panel"], fg=COLORS["text"],
                activebackground=COLORS["panel"], activeforeground=COLORS["text"],
                selectcolor=COLORS["entry"], highlightthickness=0, bd=0
            ).pack(side="right", padx=5)

        plot_host = tk.Frame(
            self.matrix_tab, bg=COLORS["card"],
            highlightbackground=COLORS["border"], highlightthickness=1
        )
        plot_host.pack(fill="both", expand=True, padx=14, pady=(4, 12))

        if Figure is None:
            raise RuntimeError("Matplotlib/Tk backend is unavailable.")

        self.figure = Figure(figsize=(10.5, 6.8), dpi=100, facecolor=COLORS["card"])
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_host)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_host, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")

    def _draw_matrix(self):
        if not hasattr(self, "ax"):
            return
        ax = self.ax
        ax.clear()
        arr = self.matrix.astype(float)
        mode = self._matrix_mode.get()
        if mode == "normalized":
            row_sum = arr.sum(axis=1, keepdims=True)
            data = np.divide(arr, row_sum, out=np.zeros_like(arr), where=row_sum != 0) * 100.0
            vmax = 100.0
        else:
            data = arr
            vmax = max(1.0, float(data.max()) if data.size else 1.0)

        im = ax.imshow(data, interpolation="nearest", aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        n = len(self.class_names)
        ticks = np.arange(n)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        fs = 8 if n <= 12 else 7 if n <= 20 else 6
        ax.set_xticklabels(self.class_names, rotation=55, ha="right", fontsize=fs)
        ax.set_yticklabels(self.class_names, fontsize=fs)
        ax.set_xlabel(self.t["pred_class"], fontsize=9)
        ax.set_ylabel(self.t["true_class"], fontsize=9)
        ax.set_title(self.t["matrix"], fontsize=11, fontweight="bold")

        # Cell annotations remain legible for small/medium matrices. Larger
        # matrices are best inspected with zoom/pan from the Matplotlib toolbar.
        if n <= 20:
            threshold = vmax * 0.52
            for i in range(n):
                for j in range(n):
                    value = data[i, j]
                    label = f"{value:.1f}%" if mode == "normalized" else str(int(round(value)))
                    ax.text(
                        j, i, label, ha="center", va="center",
                        fontsize=6.5 if n > 12 else 7.5,
                        color="black" if value > threshold else "white",
                    )

        # Remove a prior colorbar before creating a new one.
        if getattr(self, "_colorbar", None) is not None:
            try:
                self._colorbar.remove()
            except Exception:
                pass
        self._colorbar = self.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        self._colorbar.ax.tick_params(labelsize=7)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _build_class_table(self):
        host = self.class_tab
        tk.Label(
            host, text=self.t["bias_help"], bg=COLORS["panel"], fg=COLORS["muted"],
            justify="left", anchor="w", wraplength=1080, font=("Sans", 8)
        ).pack(fill="x", padx=15, pady=(12, 7))

        box = tk.Frame(
            host, bg=COLORS["card"], highlightbackground=COLORS["border"],
            highlightthickness=1
        )
        box.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        cols = ("class", "support", "predicted", "precision", "recall", "f1", "bias")
        tree = ttk.Treeview(box, columns=cols, show="headings")
        headings = {
            "class": self.t["class"],
            "support": self.t["support"],
            "predicted": self.t["predicted"],
            "precision": self.t["precision"],
            "recall": self.t["recall"],
            "f1": self.t["f1"],
            "bias": self.t["bias_pp"],
        }
        widths = {"class": 260, "support": 90, "predicted": 90, "precision": 110, "recall": 110, "f1": 110, "bias": 110}
        for col in cols:
            tree.heading(col, text=headings[col])
            tree.column(col, width=widths[col], anchor="w" if col == "class" else "center", stretch=(col == "class"))

        sy = ttk.Scrollbar(box, orient="vertical", command=tree.yview)
        sx = ttk.Scrollbar(box, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        tree.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        box.rowconfigure(0, weight=1)
        box.columnconfigure(0, weight=1)

        for row in self.metrics["per_class"]:
            tree.insert(
                "", "end",
                values=(
                    row["class"],
                    row["support"],
                    row["predicted"],
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1']:.4f}",
                    f"{row['bias_pp']:+.2f}",
                )
            )
        self.metrics_tree = tree


def open_training_results(parent, output_dir, project_name, *, lang="pt", open_folder_callback=None):
    """Open and return a new results window."""
    return TrainingResultsWindow(
        parent, output_dir, project_name,
        lang=lang, open_folder_callback=open_folder_callback
    )
