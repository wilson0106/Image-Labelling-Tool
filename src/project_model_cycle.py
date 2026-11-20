#!/usr/bin/env python3
"""Project-local model iteration services for the integrated IAT cycle.

This module deliberately does not modify the standalone training/classification
pipelines.  It reuses their scientific primitives while keeping the workflow
inside an existing IAT project.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ilt_contracts import atomic_savez_compressed, atomic_write_csv, atomic_write_json
from training_dataset import TrainingAssignment, cleanup_training_dataset_view, create_training_dataset_view, write_training_split_csv


SPLITS = ("train", "valid", "test")


def batch_id_from_csv(csv_name: str) -> str:
    stem = Path(csv_name).stem
    if stem.startswith("batch_"):
        parts = stem.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2])
    return stem


def next_model_version(project_dir: str | os.PathLike[str]) -> str:
    models = Path(project_dir) / "models"
    models.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in models.glob("v[0-9][0-9][0-9]"):
        try:
            nums.append(int(p.name[1:]))
        except Exception:
            pass
    return f"v{(max(nums) + 1 if nums else 1):03d}"


def stable_shuffle(items: Sequence, seed: int, salt: str) -> list:
    out = list(items)
    digest = hashlib.sha256(str(salt).encode("utf-8")).digest()
    salt_int = int.from_bytes(digest[:8], "little", signed=False)
    random.Random(int(seed) ^ salt_int).shuffle(out)
    return out


def allocate_counts(total: int, train: float, valid: float, test: float, mode: str) -> dict[str, int]:
    total = max(0, int(total))
    if str(mode).lower().startswith("p"):
        vals = [max(0.0, float(train)), max(0.0, float(valid)), max(0.0, float(test))]
        if sum(vals) > 100.000001:
            raise ValueError("Train + valid + test não pode ultrapassar 100%.")
        raw = [total * v / 100.0 for v in vals]
        counts = [int(x) for x in raw]
        remainder = min(total - sum(counts), int(round(sum(raw))) - sum(counts))
        order = sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)
        for i in order[:max(0, remainder)]:
            counts[i] += 1
    else:
        counts = [int(float(train)), int(float(valid)), int(float(test))]
        if min(counts) < 0 or sum(counts) > total:
            raise ValueError("A soma Train + Valid + Test excede as imagens disponíveis.")
    return dict(zip(SPLITS, counts))


def build_reference_assignments(
    project,
    batch_names: Sequence[str],
    class_plan: Mapping[str, Mapping],
    *,
    seed: int = 0,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[TrainingAssignment]:
    """Build deterministic train/valid/test assignments from current manual labels."""
    grouped: dict[str, list[Path]] = {}
    image_names: dict[str, dict[str, str]] = {}
    batch_lookup: dict[str, str] = {}
    total_batches = max(1, len(batch_names))
    for bi, csv_name in enumerate(batch_names, 1):
        df = project.load_batch(csv_name)
        batch_id = batch_id_from_csv(csv_name)
        for row in df.itertuples(index=False):
            cls = str(getattr(row, "manual_label", "") or "").strip()
            if not cls or cls not in class_plan or not class_plan[cls].get("enabled", True):
                continue
            name = str(getattr(row, "names", "") or "")
            if not name:
                continue
            p = project.resolve_image(csv_name, name)
            if p is None or not Path(p).is_file():
                continue
            grouped.setdefault(cls, []).append(Path(p))
            image_names.setdefault(cls, {})[str(Path(p))] = name
            batch_lookup[str(Path(p))] = csv_name
        if progress:
            progress(bi, total_batches, csv_name)

    assignments: list[TrainingAssignment] = []
    for cls in sorted(grouped):
        spec = class_plan[cls]
        final = str(spec.get("final") or cls).strip()
        files = stable_shuffle(grouped[cls], seed, cls)
        counts = allocate_counts(
            len(files), spec.get("train", 0), spec.get("valid", 0),
            spec.get("test", 0), spec.get("mode", "percent")
        )
        pos = 0
        for split in SPLITS:
            n = counts[split]
            for p in files[pos:pos+n]:
                assignments.append(TrainingAssignment(
                    source=Path(p),
                    image_name=image_names[cls].get(str(p), Path(p).name),
                    original_class=cls,
                    final_class=final,
                    split=split,
                ))
            pos += n
    return assignments


def summarize_assignments(assignments: Sequence[TrainingAssignment]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for a in assignments:
        row = out.setdefault(a.final_class, {"train": 0, "valid": 0, "test": 0})
        row[a.split] += 1
    return out


def materialize_reference_dataset(
    assignments: Sequence[TrainingAssignment],
    *,
    temp_parent: str | os.PathLike[str] | None = None,
) -> Path:
    summary = summarize_assignments(assignments)
    if not summary:
        raise ValueError("Nenhuma imagem foi selecionada para o treinamento.")
    return create_training_dataset_view(
        ".",
        {"train": "train", "valid": "valid", "test": "test"},
        {},
        summary,
        assignments=assignments,
        temp_parent=temp_parent,
    )


def write_cycle_manifest(assignments: Sequence[TrainingAssignment], model_dir: Path, version: str) -> Path:
    return write_training_split_csv(assignments, model_dir / f"dataset_split_{version}.csv")


def save_cycle_metadata(project_dir: Path, version: str, payload: Mapping) -> Path:
    history = project_dir / "model_history"
    history.mkdir(parents=True, exist_ok=True)
    return atomic_write_json(dict(payload), history / f"{version}.json", ensure_ascii=False, indent=2)


def load_labels(path: str | os.PathLike[str]) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items()}


def _sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_copy_file(src: Path, dest: Path) -> None:
    """Copy a file completely before atomically replacing the destination."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def application_manifest_path(project_dir: str | os.PathLike[str], version: str) -> Path:
    return Path(project_dir) / "model_history" / str(version) / "application_manifest.json"


def rollback_model_application(project, version: str, batch_names: Sequence[str] | None = None) -> dict:
    """Restore dataframe snapshots for a partially applied model version.

    Embeddings are versioned and immutable from the perspective of older CSVs,
    so restoring the CSV snapshots restores their prior embedding references too.
    """
    project_dir = Path(project.project_dir)
    backup_dir = project_dir / "model_history" / str(version) / "dataframes_before"
    if not backup_dir.is_dir():
        return {"restored": [], "missing": []}

    wanted = set(map(str, batch_names or []))
    restored, missing = [], []
    candidates = sorted(backup_dir.glob("*.csv"))
    if wanted:
        candidates = [p for p in candidates if p.name in wanted]
        missing = sorted(wanted - {p.name for p in candidates})
    for backup in candidates:
        dest = project.dataframes_dir / backup.name
        _atomic_copy_file(backup, dest)
        restored.append(backup.name)

    try:
        project.batch_cache.clear()
    except Exception:
        pass
    try:
        project._overview_cache = None
        project._image_global_index = None
    except Exception:
        pass
    return {"restored": restored, "missing": missing}





def _prepare_model_application_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Make project columns assignment-safe across pandas 2.x/3.x.

    Pandas 3 uses stricter string dtypes for CSV columns.  Reclassifying a
    project writes numeric predictions/confidences/embedding indices and may
    replace categorical/string labels.  Normalizing only the columns touched by
    model application avoids invalid-value errors without changing unrelated
    dataframe columns.
    """
    numeric_columns = ("pred", "confs1", "confs2", "confs3", "embed_idx")
    text_columns = (
        "preds1", "preds2", "preds3",
        "embed_file", "model_version", "feature_model_version",
        "projection_model_version", "manual_label", "correct_label",
    )
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in text_columns:
        if column in df.columns:
            # object accepts ordinary Python strings even when the CSV was
            # loaded as pandas' strict StringDtype/category dtype.
            df[column] = df[column].astype("object")
    if "projection_stale" in df.columns:
        # Existing projects may persist this as a textual column.  Object dtype
        # safely accepts booleans while preserving legacy/missing values.
        df["projection_stale"] = df["projection_stale"].astype("object")
    return df


def apply_model_in_place(
    project,
    batch_names: Sequence[str],
    model_path: str | os.PathLike[str],
    labels_path: str | os.PathLike[str],
    version: str,
    *,
    preserve_manual_batches: Iterable[str] = (),
    progress: Callable[[int, int, str, int, int], None] | None = None,
) -> dict:
    """Recompute predictions/features in existing batches with safe versioning.

    Invariants:
    * existing ``manual_label``/``correct_label`` are preserved for reference batches;
    * new embeddings are stored under ``embeddings/<version>/`` and never overwrite
      embeddings referenced by previous model versions;
    * every batch must match all expected image names exactly before its CSV is
      published;
    * CSV snapshots allow rollback if any later batch fails;
    * projections are explicitly marked stale because x/y are not recomputed here.
    """
    from features import compute_features, get_model

    batch_names = [str(x) for x in batch_names]
    if len(set(batch_names)) != len(batch_names):
        raise ValueError("Duplicate batch names were supplied to model application.")

    labels = load_labels(labels_path)
    model = get_model(load=True, num_classes=len(labels), pretrained=False)
    preserve = set(map(str, preserve_manual_batches))
    project_dir = Path(project.project_dir)
    history_dir = project_dir / "model_history" / str(version)
    backup_dir = history_dir / "dataframes_before"
    version_embeddings_dir = project.embeddings_dir / str(version)
    backup_dir.mkdir(parents=True, exist_ok=True)
    version_embeddings_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = application_manifest_path(project_dir, version)
    manifest = {
        "format_version": 2,
        "version": str(version),
        "status": "applying",
        "model_path": str(model_path),
        "labels_path": str(labels_path),
        "model_sha256": _sha256_file(model_path),
        "labels_sha256": _sha256_file(labels_path),
        "requested_batches": list(batch_names),
        "preserve_manual_batches": sorted(preserve),
        "batches": {
            name: {"status": "pending", "expected": None, "matched": None, "missing": [], "extra": []}
            for name in batch_names
        },
    }
    atomic_write_json(manifest, manifest_path)

    total_images = 0
    batch_sizes = {}
    for csv_name in batch_names:
        try:
            n = len(project.load_batch(csv_name))
        except Exception:
            n = 0
        batch_sizes[csv_name] = n
        total_images += n

    done_images = 0
    changed = 0
    results = []
    committed: list[str] = []

    try:
        for idx, csv_name in enumerate(batch_names, 1):
            batch_id = batch_id_from_csv(csv_name)
            manifest["batches"][csv_name]["status"] = "processing"
            atomic_write_json(manifest, manifest_path)

            features, names, predictions, confs = compute_features(
                str(project.images_dir), batch_id, model, str(model_path), labels
            )
            name_list = [str(n) for n in names]
            inferred_index = pd.Index(name_list, dtype="object")
            if inferred_index.has_duplicates:
                dupes = inferred_index[inferred_index.duplicated(keep=False)].drop_duplicates().tolist()[:10]
                raise RuntimeError(f"{csv_name}: duplicate image names returned by inference: {dupes}")

            disk_path = project.dataframes_dir / csv_name
            backup = backup_dir / csv_name
            if disk_path.exists() and not backup.exists():
                _atomic_copy_file(disk_path, backup)

            df = project.load_batch(csv_name, force=True).copy()
            if "names" not in df.columns:
                raise RuntimeError(f"{csv_name}: dataframe has no 'names' column.")
            df_names = df["names"].astype(str)
            if df_names.duplicated().any():
                dupes = df_names[df_names.duplicated(keep=False)].drop_duplicates().tolist()[:10]
                raise RuntimeError(f"{csv_name}: duplicate image names in dataframe: {dupes}")

            lookup = {name: i for i, name in enumerate(name_list)}
            expected_names = set(df_names.tolist())
            inferred_names = set(lookup)
            missing = sorted(expected_names - inferred_names)
            extra = sorted(inferred_names - expected_names)
            matched_count = len(expected_names & inferred_names)
            batch_audit = manifest["batches"][csv_name]
            batch_audit.update(
                expected=len(expected_names),
                matched=matched_count,
                missing=missing[:50],
                extra=extra[:50],
            )
            if missing or extra or matched_count != len(df):
                batch_audit["status"] = "failed_matching"
                atomic_write_json(manifest, manifest_path)
                raise RuntimeError(
                    f"{csv_name}: model application did not match the dataframe exactly "
                    f"(expected={len(df)}, matched={matched_count}, missing={len(missing)}, extra={len(extra)}). "
                    "No partial batch update was published."
                )

            matched = df_names.map(lookup)
            mask = matched.notna()
            pos = matched.astype(int)

            old_manual = df["manual_label"].copy() if "manual_label" in df.columns else None
            old_correct = df["correct_label"].copy() if "correct_label" in df.columns else None

            # Preserve which model produced the existing x/y coordinates before
            # replacing the feature-model version.
            if "projection_model_version" not in df.columns:
                if "feature_model_version" in df.columns:
                    prior_projection_version = df["feature_model_version"].fillna("legacy").astype(str)
                elif "model_version" in df.columns:
                    prior_projection_version = df["model_version"].fillna("legacy").astype(str)
                else:
                    prior_projection_version = pd.Series(["legacy"] * len(df), index=df.index, dtype="object")
                df["projection_model_version"] = prior_projection_version
            if "projection_stale" not in df.columns:
                df["projection_stale"] = False

            # CSV dtype inference differs across pandas versions.  Normalize
            # only the fields this operation mutates before assigning new
            # numeric/string values.
            df = _prepare_model_application_dtypes(df)

            embed_rel = Path("embeddings") / str(version) / f"{batch_id}.npz"
            atomic_savez_compressed(project_dir / embed_rel, feats=features)

            df.loc[mask, "pred"] = [int(predictions[i]) for i in pos]
            for rank in (1, 2, 3):
                df.loc[mask, f"preds{rank}"] = [confs[i][f"top{rank}"][0] for i in pos]
                df.loc[mask, f"confs{rank}"] = [float(confs[i][f"top{rank}"][1]) for i in pos]
            df.loc[mask, "embed_file"] = str(embed_rel)
            df.loc[mask, "embed_idx"] = pos.to_numpy()
            df.loc[mask, "model_version"] = version
            df.loc[mask, "feature_model_version"] = version
            df.loc[mask, "projection_stale"] = (
                df.loc[mask, "projection_model_version"].fillna("legacy").astype(str) != str(version)
            )

            if csv_name not in preserve:
                df.loc[mask, "manual_label"] = df.loc[mask, "preds1"].astype(str)
                if "correct_label" in df.columns:
                    df.loc[mask, "correct_label"] = df.loc[mask, "preds1"].astype(str)
            else:
                if old_manual is not None:
                    df["manual_label"] = old_manual
                if old_correct is not None:
                    df["correct_label"] = old_correct

            atomic_write_csv(df, disk_path, index=False)
            committed.append(csv_name)
            try:
                project.batch_cache.pop(csv_name, None)
                project._overview_cache = None
                project._image_global_index = None
            except Exception:
                pass

            done_images += int(mask.sum())
            changed += int(mask.sum())
            result_row = {
                "batch": csv_name,
                "images": int(mask.sum()),
                "expected": len(df),
                "matched": matched_count,
                "embed_file": str(embed_rel),
            }
            results.append(result_row)
            batch_audit.update(
                status="complete",
                embed_file=str(embed_rel),
                images=int(mask.sum()),
            )
            atomic_write_json(manifest, manifest_path)
            if progress:
                progress(idx, len(batch_names), csv_name, done_images, total_images)

    except BaseException as exc:
        rollback = rollback_model_application(project, version, committed)
        manifest["status"] = "rolled_back_after_failure"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["rollback"] = rollback
        atomic_write_json(manifest, manifest_path)
        raise

    manifest["status"] = "complete"
    manifest["updated_images"] = int(changed)
    manifest["total_images"] = int(total_images)
    atomic_write_json(manifest, manifest_path)
    return {
        "batches": results,
        "images": changed,
        "total_images": total_images,
        "manifest": str(manifest_path),
        "version": str(version),
    }



def _projection_reference_path(project_dir: str | os.PathLike[str], version: str) -> Path:
    return Path(project_dir) / "model_history" / str(version) / "projection_reference.json"


def recompute_projection_for_version(
    project,
    version: str,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Recompute a common 2-D openTSNE projection for all batches using *version*.

    The same reference batch is persisted per model version so later applications
    of that model to additional batches remain in the same projection frame.
    Only rows whose ``feature_model_version`` matches *version* are updated.
    """
    from projections import opentsne_fit, opentsne_transform

    version = str(version)
    eligible: list[tuple[str, Path, np.ndarray]] = []
    for csv_name in project.csv_names():
        df = project.load_batch(csv_name, force=True)
        if "feature_model_version" not in df.columns or "embed_file" not in df.columns or "embed_idx" not in df.columns:
            continue
        mask = df["feature_model_version"].fillna("").astype(str).eq(version)
        if not bool(mask.any()):
            continue
        rels = df.loc[mask, "embed_file"].dropna().astype(str).unique().tolist()
        if len(rels) != 1:
            raise RuntimeError(
                f"{csv_name}: expected exactly one embedding file for model {version}; got {len(rels)}."
            )
        embed_path = Path(project.project_dir) / rels[0]
        if not embed_path.is_file():
            raise FileNotFoundError(f"{csv_name}: embedding file not found: {embed_path}")
        indices = pd.to_numeric(df.loc[mask, "embed_idx"], errors="coerce")
        if indices.isna().any():
            raise RuntimeError(f"{csv_name}: invalid embed_idx values for model {version}.")
        eligible.append((csv_name, embed_path, indices.astype(int).to_numpy()))

    if not eligible:
        raise RuntimeError(f"No batches with feature_model_version={version!r} were found.")

    eligible.sort(key=lambda item: item[0])
    ref_path = _projection_reference_path(project.project_dir, version)
    reference_batch = None
    if ref_path.is_file():
        try:
            payload = json.loads(ref_path.read_text(encoding="utf-8"))
            candidate = str(payload.get("reference_batch") or "")
            if any(name == candidate for name, _, _ in eligible):
                reference_batch = candidate
        except Exception:
            reference_batch = None
    if not reference_batch:
        reference_batch = eligible[0][0]
        atomic_write_json(
            {"version": version, "reference_batch": reference_batch},
            ref_path,
            ensure_ascii=False,
            indent=2,
        )

    by_name = {name: (embed_path, indices) for name, embed_path, indices in eligible}
    ref_embed_path, _ = by_name[reference_batch]
    with np.load(ref_embed_path, allow_pickle=False) as npz:
        ref_features = np.asarray(npz["feats"])
    if len(ref_features) < 2:
        raise RuntimeError(f"{reference_batch}: not enough features to fit t-SNE.")

    base_tsne = opentsne_fit(ref_features)
    base_projection = np.asarray(base_tsne, dtype=np.float64)
    if base_projection.ndim != 2 or base_projection.shape[1] < 2:
        raise RuntimeError(f"Invalid reference projection shape: {base_projection.shape}")

    # Compute every projection before publishing any CSV update.
    projected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    total = len(eligible)
    for i, (csv_name, embed_path, indices) in enumerate(eligible, 1):
        if csv_name == reference_batch:
            full_projection = base_projection
        else:
            with np.load(embed_path, allow_pickle=False) as npz:
                features = np.asarray(npz["feats"])
            full_projection = np.asarray(opentsne_transform(features, base_tsne), dtype=np.float64)
        if indices.size and (indices.min() < 0 or indices.max() >= len(full_projection)):
            raise RuntimeError(f"{csv_name}: embed_idx outside projection bounds.")
        projected[csv_name] = (
            np.asarray(full_projection[indices, 0], dtype=np.float64),
            np.asarray(full_projection[indices, 1], dtype=np.float64),
        )
        if progress:
            progress(i, total, csv_name)

    updated_images = 0
    for csv_name, _, _ in eligible:
        df = project.load_batch(csv_name, force=True).copy()
        mask = df["feature_model_version"].fillna("").astype(str).eq(version)
        xvals, yvals = projected[csv_name]
        if int(mask.sum()) != len(xvals):
            raise RuntimeError(f"{csv_name}: projection row count changed during update.")
        df.loc[mask, "x"] = xvals
        df.loc[mask, "y"] = yvals
        if "projection_model_version" not in df.columns:
            df["projection_model_version"] = None
        df["projection_model_version"] = df["projection_model_version"].astype("object")
        df.loc[mask, "projection_model_version"] = version
        if "projection_stale" not in df.columns:
            df["projection_stale"] = False
        df["projection_stale"] = df["projection_stale"].astype("object")
        df.loc[mask, "projection_stale"] = False
        atomic_write_csv(df, project.dataframes_dir / csv_name, index=False)
        updated_images += int(mask.sum())
        try:
            project.batch_cache.pop(csv_name, None)
            project._overview_cache = None
        except Exception:
            pass

    return {
        "version": version,
        "reference_batch": reference_batch,
        "batches": [name for name, _, _ in eligible],
        "images": updated_images,
    }

def run_class_clustering(project_dir: str | os.PathLike[str], project_name: str) -> None:
    """Run the existing class-clustering engine against the updated project."""
    from class_clustering import main as clustering_main
    clustering_main([str(project_name), "--project-dir", str(project_dir)])
