#!/usr/bin/env python3
"""Training-dataset inspection, allocation, and non-destructive preparation.

The dashboard accepts two source layouts:

* ``split``: ``train/valid/test`` folders containing class folders.
* ``classes``: class folders directly below the selected dataset root.

This module owns the filesystem/business rules and deliberately has no Tkinter
dependency.  It lets the GUI edit a split plan without touching source files;
the final train/valid/test tree is materialized only when training starts.
"""
from __future__ import annotations

import csv
import hashlib
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ilt_contracts import is_image_filename

SPLITS = ("train", "valid", "test")
DATASET_MODE_SPLIT = "split"
DATASET_MODE_CLASSES = "classes"
DATASET_MODE_INVALID = "invalid"

_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


@dataclass(frozen=True)
class TrainingDatasetInspection:
    """Snapshot used by the dashboard while the user edits a training plan."""

    mode: str
    dataset_key: str | None
    counts: dict[str, dict[str, int]]
    totals: dict[str, int]


@dataclass(frozen=True)
class TrainingAssignment:
    """One source image selected for one destination split."""

    source: Path
    image_name: str
    original_class: str
    final_class: str
    split: str


def is_valid_training_class_name(value: object) -> bool:
    """Return whether *value* is safe as a class-folder name on this platform."""
    name = str(value or "").strip()
    if not name or name in (".", ".."):
        return False
    if "/" in name or "\\" in name or "\x00" in name:
        return False
    if os.name != "nt":
        return True
    if any(ch in name for ch in '<>:"|?*'):
        return False
    if any(ord(ch) < 32 for ch in name):
        return False
    if name.endswith((".", " ")):
        return False
    device_stem = name.split(".", 1)[0].upper()
    return device_stem not in _WINDOWS_RESERVED_NAMES


def _image_entries(folder: Path) -> list[Path]:
    """Return supported direct-child images in deterministic filename order."""
    try:
        with os.scandir(folder) as entries:
            result = sorted(
                (
                    Path(entry.path)
                    for entry in entries
                    if entry.is_file() and is_image_filename(entry.name)
                ),
                key=lambda path: path.name,
            )
    except OSError:
        return []
    return result


def _class_dirs(folder: Path) -> list[Path]:
    """Return direct-child class directories in deterministic name order."""
    try:
        with os.scandir(folder) as entries:
            return sorted(
                (Path(entry.path) for entry in entries if entry.is_dir()),
                key=lambda path: path.name,
            )
    except OSError:
        return []


def list_split_image_files(split_dir: str | os.PathLike[str]) -> list[str]:
    """Return supported images exactly one class directory below *split_dir*."""
    split_path = Path(os.path.abspath(os.fspath(split_dir)))
    if not split_path.is_dir():
        raise FileNotFoundError(f"Split folder not found: {split_path}")

    files: list[str] = []
    for class_dir in _class_dirs(split_path):
        files.extend(str(path) for path in _image_entries(class_dir))
    return files


def inspect_training_dataset(
    dataset_value: str | os.PathLike[str] | None,
    split_folders: Mapping[str, str],
) -> TrainingDatasetInspection:
    """Detect source layout and return class counts/totals.

    A complete configured ``train/valid/test`` structure wins detection.  If it
    is not present, direct child folders containing supported images are treated
    as taxonomic classes.  Mixed/partial split structures are intentionally
    rejected instead of guessed.
    """
    dataset_text = str(dataset_value or "").strip()
    if not dataset_text:
        return TrainingDatasetInspection(DATASET_MODE_INVALID, None, {}, {})

    dataset = Path(dataset_text).expanduser()
    if not dataset.is_dir():
        return TrainingDatasetInspection(
            DATASET_MODE_INVALID, str(dataset), {}, {}
        )

    configured = {
        split: str(split_folders.get(split, "") or "").strip()
        for split in SPLITS
    }
    split_paths = {
        split: dataset / name
        for split, name in configured.items()
        if name
    }
    all_configured = len(split_paths) == len(SPLITS)
    existing_splits = {
        split: path.is_dir() for split, path in split_paths.items()
    }

    if all_configured and all(existing_splits.values()):
        counts: dict[str, dict[str, int]] = {}
        for split in SPLITS:
            for class_dir in _class_dirs(split_paths[split]):
                record = counts.setdefault(
                    class_dir.name, {"train": 0, "valid": 0, "test": 0}
                )
                record[split] = len(_image_entries(class_dir))
        counts = dict(sorted(counts.items()))
        totals = {
            class_name: sum(int(record.get(split, 0) or 0) for split in SPLITS)
            for class_name, record in counts.items()
        }
        key = str(dataset.resolve())
        return TrainingDatasetInspection(DATASET_MODE_SPLIT, key, counts, totals)

    # A partial split layout is ambiguous: do not accidentally treat "train" as
    # a taxonomic class if only one/two split folders exist.
    if any(existing_splits.values()):
        key = str(dataset.resolve())
        return TrainingDatasetInspection(DATASET_MODE_INVALID, key, {}, {})

    counts = {}
    totals = {}
    for class_dir in _class_dirs(dataset):
        images = _image_entries(class_dir)
        if not images:
            continue
        counts[class_dir.name] = {"train": 0, "valid": 0, "test": 0}
        totals[class_dir.name] = len(images)

    if counts:
        key = str(dataset.resolve())
        return TrainingDatasetInspection(
            DATASET_MODE_CLASSES,
            key,
            dict(sorted(counts.items())),
            dict(sorted(totals.items())),
        )

    key = f"{dataset.resolve()}|{DATASET_MODE_INVALID}"
    return TrainingDatasetInspection(DATASET_MODE_INVALID, key, {}, {})


def scan_training_dataset(
    dataset_value: str | os.PathLike[str] | None,
    split_folders: Mapping[str, str],
) -> tuple[dict[str, dict[str, int]], str | None]:
    """Backward-compatible count API retained for external callers/tests."""
    inspection = inspect_training_dataset(dataset_value, split_folders)
    return inspection.counts, inspection.dataset_key


def allocate_from_percentages(
    total: int,
    percentages: Mapping[str, float],
) -> dict[str, int]:
    """Convert split percentages to deterministic integer counts.

    The largest-remainder method makes 100% use every available image while
    still supporting totals below 100% (leftover images remain unused).
    """
    total = max(0, int(total))
    values = {split: float(percentages.get(split, 0.0) or 0.0) for split in SPLITS}
    if any(value < 0 for value in values.values()):
        raise ValueError("split_allocation_invalid")
    pct_sum = sum(values.values())
    if pct_sum > 100.0000001:
        raise ValueError("split_percentage_exceeds_100")

    exact = {split: total * values[split] / 100.0 for split in SPLITS}
    base = {split: int(exact[split]) for split in SPLITS}
    target_used = min(total, int(round(total * pct_sum / 100.0)))
    remaining = max(0, target_used - sum(base.values()))
    order = sorted(
        SPLITS,
        key=lambda split: (-(exact[split] - base[split]), SPLITS.index(split)),
    )
    for split in order[:remaining]:
        base[split] += 1
    return base


def normalize_count_allocation(
    total: int,
    allocation: Mapping[str, int],
) -> dict[str, int]:
    """Validate and normalize explicit per-split image counts."""
    total = max(0, int(total))
    result: dict[str, int] = {}
    for split in SPLITS:
        raw = allocation.get(split, 0)
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("split_allocation_invalid") from exc
        if value < 0:
            raise ValueError("split_allocation_invalid")
        result[split] = value
    if sum(result.values()) > total:
        raise ValueError("split_allocation_exceeds_total")
    return result


def build_training_class_plan(
    counts: Mapping[str, Mapping[str, int]],
    target_names: Mapping[str, str] | None = None,
    enabled: Mapping[str, bool] | None = None,
    *,
    allocations: Mapping[str, Mapping[str, int]] | None = None,
    available_totals: Mapping[str, int] | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, int]], list[str]]:
    """Build original->final mapping and aggregate final split counts."""
    target_names = target_names or {}
    enabled = enabled or {}
    allocations = allocations or {}
    available_totals = available_totals or {}
    mapping: dict[str, str] = {}
    summary: dict[str, dict[str, int]] = {}
    errors: list[str] = []

    for original in sorted(counts):
        if original in enabled and not bool(enabled[original]):
            continue

        target = str(target_names.get(original, original) or "").strip()
        if not is_valid_training_class_name(target):
            errors.append("class_name_invalid")
            continue

        original_counts = counts[original]
        total = int(
            available_totals.get(
                original,
                sum(int(original_counts.get(split, 0) or 0) for split in SPLITS),
            )
            or 0
        )
        if original in allocations:
            try:
                source = normalize_count_allocation(total, allocations[original])
            except ValueError as exc:
                errors.append(str(exc))
                continue
        else:
            source = {
                split: int(original_counts.get(split, 0) or 0)
                for split in SPLITS
            }

        mapping[original] = target
        record = summary.setdefault(
            target, {"train": 0, "valid": 0, "test": 0}
        )
        for split in SPLITS:
            record[split] += int(source.get(split, 0) or 0)

    if mapping:
        if any(record["train"] <= 0 for record in summary.values()):
            errors.append("class_without_train")
        if any(
            sum(record[split] for record in summary.values()) <= 0
            for split in SPLITS
        ):
            errors.append("split_empty_after_mapping")

    return mapping, summary, list(dict.fromkeys(errors))


def mapping_changes_dataset(
    mapping: Mapping[str, str],
    original_classes: Mapping[str, object],
) -> bool:
    """Return whether filtering/renaming requires a temporary dataset view."""
    if set(mapping) != set(original_classes):
        return True
    return any(mapping.get(original) != original for original in original_classes)


def allocation_changes_dataset(
    mode: str,
    counts: Mapping[str, Mapping[str, int]],
    allocations: Mapping[str, Mapping[str, int]],
    enabled: Mapping[str, bool] | None = None,
) -> bool:
    """Return whether the selected split plan differs from the source layout."""
    if mode != DATASET_MODE_SPLIT:
        return True
    enabled = enabled or {}
    for original, record in counts.items():
        if original in enabled and not bool(enabled[original]):
            continue
        desired = allocations.get(original)
        if desired is None:
            continue
        if any(
            int(desired.get(split, 0) or 0)
            != int(record.get(split, 0) or 0)
            for split in SPLITS
        ):
            return True
    return False


def cleanup_training_dataset_view(path: str | os.PathLike[str] | None) -> None:
    """Best-effort cleanup of a temporary training view."""
    if not path:
        return
    try:
        shutil.rmtree(Path(path), ignore_errors=True)
    except Exception:
        pass


def _link_or_copy(source: Path, target: Path) -> None:
    """Create the cheapest safe reference available, then fall back to copy."""
    source_abs = source.resolve()
    if os.name == "nt":
        strategies = (
            lambda: os.link(str(source_abs), str(target)),
            lambda: os.symlink(str(source_abs), str(target)),
        )
    else:
        strategies = (
            lambda: os.symlink(str(source_abs), str(target)),
            lambda: os.link(str(source_abs), str(target)),
        )

    for create_link in strategies:
        try:
            create_link()
            return
        except OSError:
            pass
    shutil.copy2(source_abs, target)


def _stable_random(seed: int, class_name: str) -> random.Random:
    payload = f"{int(seed)}\0{class_name}".encode("utf-8", errors="surrogatepass")
    digest = hashlib.sha256(payload).digest()
    return random.Random(int.from_bytes(digest[:8], "big", signed=False))


def _source_images_by_class(
    dataset: Path,
    split_folders: Mapping[str, str],
    mode: str,
    originals: Sequence[str],
) -> dict[str, dict[str, list[Path]]]:
    """Collect direct source images, retaining original split provenance."""
    result: dict[str, dict[str, list[Path]]] = {}
    for original in originals:
        per_split = {split: [] for split in SPLITS}
        if mode == DATASET_MODE_SPLIT:
            for split in SPLITS:
                folder_name = str(split_folders.get(split, "") or "").strip()
                if folder_name:
                    per_split[split] = _image_entries(
                        dataset / folder_name / original
                    )
        elif mode == DATASET_MODE_CLASSES:
            # ``train`` is only a provenance bucket here; the allocation phase
            # will reshuffle it into the requested destinations.
            per_split["train"] = _image_entries(dataset / original)
        else:
            raise ValueError("train_structure_missing")
        result[original] = per_split
    return result


def build_training_assignments(
    dataset: str | os.PathLike[str],
    split_folders: Mapping[str, str],
    mapping: Mapping[str, str],
    allocations: Mapping[str, Mapping[str, int]],
    *,
    dataset_mode: str,
    seed: int = 0,
) -> list[TrainingAssignment]:
    """Select source images and assign them reproducibly to destination splits.

    Existing assignments are preserved exactly when a pre-split dataset keeps
    the same per-class counts.  As soon as counts change (or the source is a
    class-only dataset), images are pooled per original class and shuffled with
    a stable seed before slicing train/valid/test.
    """
    dataset_path = Path(dataset)
    source_map = _source_images_by_class(
        dataset_path, split_folders, dataset_mode, sorted(mapping)
    )
    assignments: list[TrainingAssignment] = []

    for original in sorted(mapping):
        final_class = mapping[original]
        per_split = source_map[original]
        actual = {split: len(per_split[split]) for split in SPLITS}
        total = sum(actual.values())
        desired = normalize_count_allocation(total, allocations[original])

        preserve_existing = (
            dataset_mode == DATASET_MODE_SPLIT
            and all(desired[split] == actual[split] for split in SPLITS)
        )
        if preserve_existing:
            chosen = {
                split: list(per_split[split])
                for split in SPLITS
            }
        else:
            pool = [
                path
                for split in SPLITS
                for path in per_split[split]
            ]
            pool = sorted(
                pool,
                key=lambda path: (
                    path.parent.parent.name,
                    path.parent.name,
                    path.name,
                    str(path),
                ),
            )
            _stable_random(seed, original).shuffle(pool)
            chosen = {}
            offset = 0
            for split in SPLITS:
                amount = desired[split]
                chosen[split] = pool[offset:offset + amount]
                offset += amount

        for split in SPLITS:
            for source in chosen[split]:
                assignments.append(
                    TrainingAssignment(
                        source=source,
                        image_name=source.name,
                        original_class=original,
                        final_class=final_class,
                        split=split,
                    )
                )

    return assignments


def write_training_split_csv(
    assignments: Sequence[TrainingAssignment],
    path: str | os.PathLike[str],
) -> Path:
    """Write the requested 3-column image/class/destination audit CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("imagem", "classe_taxonomica", "destino"))
            for item in assignments:
                writer.writerow(
                    (item.image_name, item.original_class, item.split)
                )
        os.replace(tmp, output)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return output


def create_training_dataset_view(
    dataset: str | os.PathLike[str],
    split_folders: Mapping[str, str],
    mapping: Mapping[str, str],
    summary: Mapping[str, Mapping[str, int]],
    *,
    allocations: Mapping[str, Mapping[str, int]] | None = None,
    dataset_mode: str = DATASET_MODE_SPLIT,
    seed: int = 0,
    assignments: Sequence[TrainingAssignment] | None = None,
    temp_parent: str | os.PathLike[str] | None = None,
) -> Path:
    """Create the filtered/grouped/re-split temporary view used by train.py."""
    dataset_path = Path(dataset)
    if allocations is None:
        # Legacy behavior: retain each original split count.
        inspection = inspect_training_dataset(dataset_path, split_folders)
        allocations = {
            original: {
                split: int(inspection.counts.get(original, {}).get(split, 0) or 0)
                for split in SPLITS
            }
            for original in mapping
        }
        dataset_mode = inspection.mode

    if assignments is None:
        assignments = build_training_assignments(
            dataset_path,
            split_folders,
            mapping,
            allocations,
            dataset_mode=dataset_mode,
            seed=seed,
        )

    if temp_parent is None:
        base = Path(tempfile.gettempdir()) / "ILT"
    else:
        base = Path(temp_parent)
    base.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="train_", dir=base))

    try:
        for split in SPLITS:
            folder_name = str(split_folders[split])
            out_split = temp_root / folder_name
            out_split.mkdir(parents=True, exist_ok=True)
            for final_class in sorted(summary):
                (out_split / final_class).mkdir(parents=True, exist_ok=True)

        counters = {
            (split, final_class): 0
            for split in SPLITS
            for final_class in summary
        }
        for item in assignments:
            folder_name = str(split_folders[item.split])
            key = (item.split, item.final_class)
            counters[key] += 1
            suffix = item.source.suffix.lower()
            target = (
                temp_root
                / folder_name
                / item.final_class
                / f"image_{counters[key]:08d}{suffix}"
            )
            _link_or_copy(item.source, target)

        return temp_root
    except Exception:
        cleanup_training_dataset_view(temp_root)
        raise
