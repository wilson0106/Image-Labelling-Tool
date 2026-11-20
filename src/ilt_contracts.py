#!/usr/bin/env python3
"""Shared stability contracts for ILT Data Extraction & IAT.

This module centralizes rules that must stay identical across training,
classification, projections, clustering and IAT. It intentionally contains no
scientific preprocessing logic.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

ILT_VERSION = "1.1.0"

IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
})

# Semantic dataframe columns. Keeping these names in one place prevents UI
# components from accidentally treating prediction/source provenance as the
# current human label.
MANUAL_LABEL_COLUMN = "manual_label"
SOURCE_LABEL_COLUMN = "source_label"
PREDICTION_COLUMNS = ("preds1", "pred")
GID_COLUMN = "gid"


def is_image_filename(value) -> bool:
    """Return True when *value* has a supported image suffix, case-insensitively."""
    return Path(str(value)).suffix.lower() in IMAGE_EXTENSIONS


def sanitize_filename(value: str) -> str:
    """Map a class/selection label to the historical ILT artifact filename form."""
    text = str(value).replace("/", "_").replace("\\", "_").strip()
    return re.sub(r"\s+", "_", text)


def ensure_unique_sanitized_labels(labels: Iterable[str]) -> dict[str, str]:
    """Return label -> sanitized label, rejecting artifact filename collisions.

    Two biologically distinct labels must never silently overwrite the same
    ``clusters/<label>.npz`` artifact.
    """
    out: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for raw in labels:
        label = str(raw)
        safe = sanitize_filename(label)
        key = safe.casefold()
        previous = reverse.get(key)
        if previous is not None and previous != label:
            raise ValueError(
                "Class labels collide after filename sanitization: "
                f"{previous!r} and {label!r} -> {safe!r}."
            )
        reverse[key] = label
        out[label] = safe
    return out


def resolve_prediction_column(columns: Iterable[str]) -> Optional[str]:
    """Return the canonical prediction column available in *columns*."""
    cols = set(columns)
    for name in PREDICTION_COLUMNS:
        if name in cols:
            return name
    return None


def resolve_scatter_class_column(color_mode: str, columns: Iterable[str]) -> Optional[str]:
    """Resolve the categorical class dimension associated with Scatter coloring.

    Categorical color modes follow the field they represent. Continuous
    diagnostic modes (confidence, variance, kNN, outlier, review priority, etc.)
    intentionally keep ``manual_label`` as the class-filter dimension.
    """
    cols = set(columns)
    mode = str(color_mode or MANUAL_LABEL_COLUMN)
    pred_col = resolve_prediction_column(cols)

    requested = {
        MANUAL_LABEL_COLUMN: MANUAL_LABEL_COLUMN,
        "prediction": pred_col,
        SOURCE_LABEL_COLUMN: SOURCE_LABEL_COLUMN,
        "hdbscan_cluster": "hdbscan_cluster",
    }.get(mode, MANUAL_LABEL_COLUMN)

    if requested and requested in cols:
        return requested

    # Defensive compatibility for malformed/legacy frames.
    for candidate in (MANUAL_LABEL_COLUMN, *PREDICTION_COLUMNS, SOURCE_LABEL_COLUMN):
        if candidate in cols:
            return candidate
    return None


def validate_labels(labels, source="labels"):
    """Validate and normalize the class-name -> contiguous integer-index contract."""
    if not isinstance(labels, dict) or not labels:
        raise ValueError(f"{source}: label map must be a non-empty JSON object.")

    normalized = {}
    used = set()
    for raw_name, raw_idx in labels.items():
        name = str(raw_name)
        if not name:
            raise ValueError(f"{source}: empty class name.")
        if isinstance(raw_idx, bool):
            raise ValueError(f"{source}: class {name!r} has boolean index {raw_idx!r}.")
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source}: class {name!r} has invalid index {raw_idx!r}."
            ) from exc
        if isinstance(raw_idx, float) and not raw_idx.is_integer():
            raise ValueError(
                f"{source}: class {name!r} has non-integer index {raw_idx!r}."
            )
        if idx < 0:
            raise ValueError(f"{source}: class {name!r} has negative index {idx}.")
        if idx in used:
            raise ValueError(f"{source}: duplicate class index {idx}.")
        normalized[name] = idx
        used.add(idx)

    actual = sorted(used)
    expected = list(range(len(normalized)))
    if actual != expected:
        raise ValueError(
            f"{source}: class indices must be contiguous from 0 to "
            f"{len(normalized)-1}; found {actual}."
        )
    return normalized


def safe_perplexity(n_samples, maximum=30.0):
    """Return a t-SNE perplexity guaranteed to be positive and strictly < N."""
    n = int(n_samples)
    if n < 2:
        raise ValueError(f"t-SNE requires at least 2 samples; received {n}.")
    return float(min(
        float(maximum),
        max(1.0, (n - 1) / 3.0),
        n - 1e-3,
    ))


def validate_embedding_matrix(array, source="embeddings") -> np.ndarray:
    """Validate the numeric NxD embedding contract and return a NumPy view."""
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(
            f"{source}: expected a 2D embedding matrix, received shape {arr.shape}."
        )
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(
            f"{source}: embedding matrix must be non-empty, received {arr.shape}."
        )
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{source}: embedding matrix must be numeric.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{source}: embedding matrix contains NaN or infinity.")
    return arr


def _temporary_path(path: Path) -> Path:
    """Create a unique empty temporary file in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    return Path(name)


def _cleanup_temp(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _fsync_path(path: Path) -> None:
    """Best-effort flush of file contents before the atomic rename."""
    try:
        with path.open("rb") as fh:
            os.fsync(fh.fileno())
    except (OSError, AttributeError):
        pass


def atomic_write_csv(df, path, **kwargs):
    """Write a complete CSV then atomically replace the destination."""
    path = Path(path)
    tmp = _temporary_path(path)
    try:
        df.to_csv(tmp, **kwargs)
        _fsync_path(tmp)
        os.replace(tmp, path)
    finally:
        _cleanup_temp(tmp)


def atomic_savez_compressed(path, **arrays):
    """Write a complete compressed NPZ then atomically replace the destination."""
    path = Path(path)
    tmp = _temporary_path(path)
    try:
        # A file handle avoids NumPy silently appending ".npz" to the temp name.
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **arrays)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    finally:
        _cleanup_temp(tmp)


def atomic_write_json(payload, path, **kwargs):
    """Write JSON completely before replacing the destination."""
    path = Path(path)
    tmp = _temporary_path(path)
    opts = {"ensure_ascii": False, "indent": 2}
    opts.update(kwargs)
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(payload, **opts) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    finally:
        _cleanup_temp(tmp)


def atomic_write_text(text, path, encoding="utf-8"):
    """Write text completely before replacing the destination."""
    path = Path(path)
    tmp = _temporary_path(path)
    try:
        with tmp.open("w", encoding=encoding, newline="\n") as fh:
            fh.write(str(text))
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    finally:
        _cleanup_temp(tmp)
