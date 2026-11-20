#!/usr/bin/env python3
"""Evaluation helpers for ILT neural-network training.

The module is intentionally independent from Tk so metrics can be tested,
reused by ``train.py`` and rendered by the dashboard without duplicating
scientific calculations.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def ordered_class_names(labels: Mapping[str, int] | Sequence[str]) -> list[str]:
    """Return class names in the confusion-matrix index order."""
    if isinstance(labels, Mapping):
        pairs = sorted(((str(name), int(idx)) for name, idx in labels.items()), key=lambda item: item[1])
        indices = [idx for _, idx in pairs]
        if indices != list(range(len(pairs))):
            raise ValueError("Class indices must be contiguous and start at zero.")
        return [name for name, _ in pairs]
    return [str(name) for name in labels]


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    return out


def evaluate_confusion_matrix(
    matrix: Iterable[Iterable[int]] | np.ndarray,
    class_names: Sequence[str],
) -> dict:
    """Compute standard multiclass metrics plus prevalence bias.

    Matrix convention: rows are observed/true classes and columns are predicted
    classes.

    Per-class prevalence bias is:
        100 * (predicted_prevalence - observed_prevalence)
    in percentage points. The global bias summary is the mean absolute value of
    those class biases. Both are ideal at zero.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        if arr.size == 1:
            arr = arr.reshape(1, 1)
        else:
            raise ValueError("Confusion matrix must be square.")
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Confusion matrix must be square.")

    names = [str(name) for name in class_names]
    if len(names) != arr.shape[0]:
        raise ValueError(
            f"Confusion matrix has {arr.shape[0]} classes but {len(names)} class names were provided."
        )
    if np.any(arr < 0):
        raise ValueError("Confusion matrix cannot contain negative counts.")

    support = arr.sum(axis=1)
    predicted = arr.sum(axis=0)
    tp = np.diag(arr)
    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    total = float(arr.sum())
    accuracy = float(tp.sum() / total) if total else 0.0
    observed_prev = support / total if total else np.zeros_like(support, dtype=float)
    predicted_prev = predicted / total if total else np.zeros_like(predicted, dtype=float)
    bias_pp = 100.0 * (predicted_prev - observed_prev)

    per_class = []
    for idx, name in enumerate(names):
        per_class.append(
            {
                "index": idx,
                "class": name,
                "support": int(round(float(support[idx]))),
                "predicted": int(round(float(predicted[idx]))),
                "true_positive": int(round(float(tp[idx]))),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "observed_prevalence": float(observed_prev[idx]),
                "predicted_prevalence": float(predicted_prev[idx]),
                "bias_pp": float(bias_pp[idx]),
            }
        )

    return {
        "n_samples": int(round(total)),
        "n_classes": len(names),
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision)) if len(precision) else 0.0,
        "macro_recall": float(np.mean(recall)) if len(recall) else 0.0,
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "mean_abs_bias_pp": float(np.mean(np.abs(bias_pp))) if len(bias_pp) else 0.0,
        "max_abs_bias_pp": float(np.max(np.abs(bias_pp))) if len(bias_pp) else 0.0,
        "per_class": per_class,
        "definitions": {
            "precision": "TP / predicted positives for the class",
            "recall": "TP / observed positives for the class",
            "f1": "harmonic mean of class precision and recall",
            "macro_f1": "unweighted mean of class F1 values",
            "macro_precision": "unweighted mean of class precision values",
            "macro_recall": "unweighted mean of class recall values",
            "bias_pp": "100 × (predicted prevalence − observed prevalence), in percentage points; ideal = 0",
            "mean_abs_bias_pp": "mean absolute class prevalence bias, in percentage points; ideal = 0",
        },
    }


def evaluation_paths(output_dir: str | Path, project_name: str) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "matrix": root / f"matrix_{project_name}.csv",
        "labels": root / f"labels_{project_name}.json",
        "summary": root / f"evaluation_{project_name}.json",
        "metrics": root / f"metrics_{project_name}.csv",
    }


def load_training_evaluation(output_dir: str | Path, project_name: str) -> tuple[np.ndarray, list[str], dict]:
    """Load matrix/labels produced by training and recompute metrics deterministically."""
    paths = evaluation_paths(output_dir, project_name)
    if not paths["matrix"].is_file():
        raise FileNotFoundError(paths["matrix"])
    if not paths["labels"].is_file():
        raise FileNotFoundError(paths["labels"])

    matrix = np.loadtxt(paths["matrix"], delimiter=",")
    matrix = np.asarray(matrix)
    if matrix.ndim == 0:
        matrix = matrix.reshape(1, 1)
    elif matrix.ndim == 1:
        # np.loadtxt returns a vector for a single row. A valid single-class
        # matrix contains one value; larger vectors are not square matrices.
        if matrix.size == 1:
            matrix = matrix.reshape(1, 1)
        else:
            n = int(round(np.sqrt(matrix.size)))
            if n * n == matrix.size:
                matrix = matrix.reshape(n, n)
            else:
                raise ValueError("Stored confusion matrix is not square.")

    with paths["labels"].open(encoding="utf-8") as handle:
        labels = json.load(handle)
    names = ordered_class_names(labels)
    metrics = evaluate_confusion_matrix(matrix, names)

    if paths["summary"].is_file():
        try:
            with paths["summary"].open(encoding="utf-8") as handle:
                persisted = json.load(handle)
            for key in ("test_loss", "configured_macro_divisor", "created_by"):
                if key in persisted:
                    metrics[key] = persisted[key]
        except Exception:
            pass
    return matrix.astype(int), names, metrics


def save_evaluation_artifacts(
    matrix: Iterable[Iterable[int]] | np.ndarray,
    labels: Mapping[str, int] | Sequence[str],
    output_dir: str | Path,
    project_name: str,
    *,
    test_loss: float | None = None,
    configured_macro_divisor: int | float | None = None,
) -> dict[str, Path]:
    """Persist a machine-readable summary and a per-class CSV beside the model."""
    names = ordered_class_names(labels)
    metrics = evaluate_confusion_matrix(matrix, names)
    if test_loss is not None:
        metrics["test_loss"] = float(test_loss)
    if configured_macro_divisor is not None:
        metrics["configured_macro_divisor"] = configured_macro_divisor
    metrics["created_by"] = "ILT training evaluation"

    paths = evaluation_paths(output_dir, project_name)
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)

    tmp_json = paths["summary"].with_suffix(paths["summary"].suffix + ".tmp")
    tmp_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_json.replace(paths["summary"])

    fieldnames = [
        "class", "support", "predicted", "true_positive",
        "precision", "recall", "f1",
        "observed_prevalence", "predicted_prevalence", "bias_pp",
    ]
    tmp_csv = paths["metrics"].with_suffix(paths["metrics"].suffix + ".tmp")
    with tmp_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics["per_class"]:
            writer.writerow({key: row[key] for key in fieldnames})
    tmp_csv.replace(paths["metrics"])
    return paths
