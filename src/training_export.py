#!/usr/bin/env python3
"""Pure helpers for IAT training-dataset export.

This module is intentionally independent from Tkinter.  It only plans a
reproducible train/valid/test split; the IAT Export tab owns file I/O and UI.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


SPLITS = ("train", "valid", "test")


def _stable_seed(seed: int, label: str) -> int:
    raw = f"{int(seed)}::{label}".encode("utf-8", "surrogatepass")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little", signed=False)


def allocation_targets(available: int, mode: str, values: Mapping[str, float]) -> Dict[str, int]:
    """Return requested image counts for a class.

    Percent mode accepts a total <= 100; the remainder is intentionally unused.
    Count mode accepts a total <= the number of available images.
    """
    available = max(0, int(available))
    mode = str(mode or "percent").lower()
    vals = {s: float(values.get(s, 0) or 0) for s in SPLITS}
    if any(v < 0 for v in vals.values()):
        raise ValueError("Train/valid/test values cannot be negative.")

    if mode in {"percent", "%", "pct"}:
        total_pct = sum(vals.values())
        if total_pct > 100.0000001:
            raise ValueError("Train + valid + test percentages cannot exceed 100%.")
        raw = {s: available * vals[s] / 100.0 for s in SPLITS}
        target_total = min(available, int(round(available * total_pct / 100.0)))
        out = {s: int(np.floor(raw[s])) for s in SPLITS}
        left = target_total - sum(out.values())
        # Largest-remainder apportionment: deterministic and exact on the target total.
        order = sorted(SPLITS, key=lambda s: (-(raw[s] - out[s]), SPLITS.index(s)))
        for s in order[:max(0, left)]:
            out[s] += 1
        return out

    if mode in {"count", "n", "number"}:
        out = {s: int(round(vals[s])) for s in SPLITS}
        if sum(out.values()) > available:
            raise ValueError(
                f"Requested {sum(out.values())} images, but only {available} are available."
            )
        return out

    raise ValueError(f"Unknown split mode: {mode}")


def _assign_image_rows(class_df: pd.DataFrame, targets: Mapping[str, int], seed: int, label: str):
    if class_df.empty:
        return pd.Series(dtype="object")
    # Sort first so the same data + seed remains reproducible even if CSV row order changes.
    key = "gid" if "gid" in class_df.columns else class_df.columns[0]
    ordered = class_df.sort_values(key, kind="stable")
    rng = np.random.default_rng(_stable_seed(seed, label))
    perm = rng.permutation(len(ordered))
    idx = ordered.index.to_numpy()[perm]
    assigned = pd.Series("unused", index=class_df.index, dtype="object")
    pos = 0
    for split in SPLITS:
        n = int(targets.get(split, 0))
        if n:
            assigned.loc[idx[pos:pos+n]] = split
        pos += n
    return assigned


def _assign_group_rows(
    class_df: pd.DataFrame,
    targets: Mapping[str, int],
    seed: int,
    label: str,
    group_col: str,
    mode: str,
):
    assigned = pd.Series("unused", index=class_df.index, dtype="object")
    if class_df.empty:
        return assigned

    keys = class_df[group_col].fillna("").astype(str)
    # Blank grouping IDs are treated as independent images, not as one giant group.
    keys = keys.where(keys.str.strip().ne(""), "__row__" + class_df.index.astype(str))
    groups = [(k, np.asarray(v, dtype=object)) for k, v in class_df.groupby(keys, sort=True).groups.items()]
    rng = np.random.default_rng(_stable_seed(seed, label))
    if groups:
        order = rng.permutation(len(groups))
        groups = [groups[int(i)] for i in order]

    actual = {s: 0 for s in SPLITS}
    target_total = sum(int(targets.get(s, 0)) for s in SPLITS)
    used_total = 0
    count_mode = str(mode).lower() in {"count", "n", "number"}

    for _, idx in groups:
        size = len(idx)
        if target_total <= used_total:
            break
        deficits = {s: int(targets.get(s, 0)) - actual[s] for s in SPLITS}
        positive = [s for s in SPLITS if deficits[s] > 0]
        if not positive:
            break

        if count_mode:
            # Counts are maxima when grouping is enabled; never exceed a requested N.
            fits = [s for s in positive if size <= deficits[s]]
            if not fits:
                continue
            split = max(fits, key=lambda s: (deficits[s], -SPLITS.index(s)))
        else:
            # Percentage targets are approximate for indivisible groups.  If 100% is
            # requested this assigns every group while staying as close as possible.
            split = max(positive, key=lambda s: (deficits[s], -SPLITS.index(s)))

        assigned.loc[idx] = split
        actual[split] += size
        used_total += size

    return assigned


def plan_training_split(
    frame: pd.DataFrame,
    group_map: Mapping[str, str],
    split_config: Mapping[str, Mapping[str, object]],
    seed: int = 42,
    unit: str = "image",
    group_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reproducible split plan.

    Required input columns: ``manual_label`` and preferably ``gid``.
    ``group_map`` maps original manual labels to output labels.  Labels absent
    from the map are excluded.
    """
    if frame is None or frame.empty:
        cols = list(frame.columns) if isinstance(frame, pd.DataFrame) else []
        return pd.DataFrame(columns=cols + ["export_label", "split"]), pd.DataFrame(
            columns=["class", "available", "train", "valid", "test", "unused"]
        )
    if "manual_label" not in frame.columns:
        raise ValueError("manual_label is required for training export.")

    df = frame.copy()
    original = df["manual_label"].fillna("").astype(str)
    df["export_label"] = original.map({str(k): str(v) for k, v in group_map.items()})
    df = df[df["export_label"].notna()].copy()
    df["export_label"] = df["export_label"].astype(str).str.strip()
    df = df[df["export_label"].ne("")].copy()
    if df.empty:
        return df.assign(split=pd.Series(dtype="object")), pd.DataFrame(
            columns=["class", "available", "train", "valid", "test", "unused"]
        )

    df["split"] = "unused"
    summary = []
    unit = str(unit or "image").lower()

    for label in sorted(df["export_label"].unique()):
        mask = df["export_label"].eq(label)
        sub = df.loc[mask]
        cfg = split_config.get(label)
        if cfg is None:
            cfg = {"mode": "percent", "train": 70, "valid": 15, "test": 15}
        mode = str(cfg.get("mode", "percent")).lower()
        targets = allocation_targets(
            len(sub), mode, {s: float(cfg.get(s, 0) or 0) for s in SPLITS}
        )
        if unit == "group" and group_column and group_column in sub.columns:
            assigned = _assign_group_rows(sub, targets, int(seed), label, group_column, mode)
        else:
            assigned = _assign_image_rows(sub, targets, int(seed), label)
        df.loc[sub.index, "split"] = assigned

        counts = assigned.value_counts().to_dict()
        summary.append({
            "class": label,
            "available": int(len(sub)),
            "train": int(counts.get("train", 0)),
            "valid": int(counts.get("valid", 0)),
            "test": int(counts.get("test", 0)),
            "unused": int(counts.get("unused", 0)),
            "target_train": int(targets["train"]),
            "target_valid": int(targets["valid"]),
            "target_test": int(targets["test"]),
            "mode": mode,
        })

    summary_df = pd.DataFrame(summary)
    return df, summary_df
