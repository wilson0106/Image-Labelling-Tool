#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
class_clustering.py

Gera artefatos de clustering por classe em:
  main/assets/<project>/clusters/<class>.npz

Compatível com o data_explorer.py:
  - names (N,)  -> nomes reais das imagens
  - linkage (N-1,4)
  - dist_max (float)

Fonte dos embeddings:
  - CSVs em main/assets/<project>/dataframes/*.csv
  - colunas: names, embed_file, embed_idx, e uma coluna de classe:
      preferencialmente manual_label; se for boolean/ruim, tenta preds1, depois pred.
  - embeddings em main/assets/<project>/embeddings/*.npz
      chaves comuns: feats / embeddings / vectors, e (opcional) names

Para classes grandes: usa centroides (MiniBatchKMeans) e "expande" o linkage para N folhas
sem custo O(N^2) de distâncias completas.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from hw_resources import memory_budget_bytes
from ilt_contracts import (
    ensure_unique_sanitized_labels,
    sanitize_filename as _contract_sanitize_filename,
)
from ilt_clustering import (
    apply_pca_if_needed as _shared_apply_pca_if_needed,
    centroid_expanded_linkage as _shared_centroid_expanded_linkage,
    full_linkage as _shared_full_linkage,
    hierarchical_linkage,
    l2_normalize as _shared_l2_normalize,
    save_cluster_artifact as _shared_save_cluster_artifact,
)

# ======================================================================================
# Utils
# ======================================================================================

def _now() -> float:
    return time.perf_counter()


def _log(msg: str):
    print(msg, flush=True)


def _sanitize_filename(x: str) -> str:
    """Backward-compatible wrapper around the shared artifact-name contract."""
    return _contract_sanitize_filename(x)


def _looks_booleanish(s: pd.Series) -> bool:
    # True/False em bool, ou strings "True"/"False", ou 0/1 com baixa diversidade
    if s.dtype == bool:
        return True
    vals = s.dropna().astype(str).str.lower().unique()
    if len(vals) == 0:
        return True
    if set(vals).issubset({"true", "false", "0", "1"}):
        # se só tem 1 valor e é "true", costuma ser bug de pipeline
        return True
    return False


def _pick_label_column(df: pd.DataFrame) -> str:
    """Choose the class field without silently overriding human labels.

    Modern ILT projects always use ``manual_label`` as the authoritative class
    membership for per-class clustering.  Prediction columns are accepted only
    for legacy projects where ``manual_label`` is genuinely absent.
    """
    if "manual_label" in df.columns:
        if _looks_booleanish(df["manual_label"]):
            raise RuntimeError(
                "manual_label existe, mas parece booleano/inválido. "
                "Recusando fallback silencioso para a previsão."
            )
        return "manual_label"

    for candidate in ("preds1", "pred"):
        if candidate in df.columns:
            return candidate

    raise RuntimeError(
        "Não encontrei coluna de classe. Projetos modernos exigem manual_label; "
        "preds1/pred são aceitos apenas para projetos legados."
    )


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # ``eps`` is retained in the public helper signature for compatibility.
    return _shared_l2_normalize(X)


def _resolve_embed_path(project_dir: Path, embed_file_value: str) -> Path:
    """
    embed_file no CSV pode vir:
      - só o nome (batch_0001.npz)
      - relativo (embeddings/batch_0001.npz)
      - absoluto
    """
    p = Path(str(embed_file_value))
    if p.is_absolute() and p.exists():
        return p
    # relativo ao projeto
    cand1 = project_dir / p
    if cand1.exists():
        return cand1
    # relativo à pasta embeddings
    cand2 = project_dir / "embeddings" / p.name
    if cand2.exists():
        return cand2
    # tenta também caso embed_file já venha com "embeddings/..."
    if "embeddings" in p.parts:
        tail = Path(*p.parts[p.parts.index("embeddings")+1:])
        cand3 = project_dir / "embeddings" / tail
        if cand3.exists():
            return cand3
    raise FileNotFoundError(f"embed_file não encontrado: '{embed_file_value}' (projeto={project_dir})")


def _load_feats_from_npz(npz_path: Path) -> np.ndarray:
    """
    Suporta npz com chaves:
      - feats (preferido)
      - embeddings
      - vectors
      - X
    """
    z = np.load(npz_path, allow_pickle=False)
    keys = list(z.keys())

    for k in ("feats", "embeddings", "vectors", "X"):
        if k in keys:
            arr = z[k]
            break
    else:
        # pega o primeiro array 2D
        arr = None
        for k in keys:
            a = z[k]
            if isinstance(a, np.ndarray) and a.ndim == 2:
                arr = a
                break
        if arr is None:
            raise KeyError(f"{npz_path.name}: não achei array 2D de embeddings (keys={keys})")

    if arr.ndim != 2:
        raise ValueError(f"{npz_path.name}: embeddings não são 2D (shape={arr.shape})")

    return arr


# ======================================================================================
# Core: build X per class
# ======================================================================================

def _read_index_table(
    dataframes_dir: Path,
    verbose: bool = False,
    feature_version: str | None = None,
) -> pd.DataFrame:
    """Read the minimal clustering index and enforce one feature space.

    Legacy projects without ``feature_model_version`` are accepted as one
    implicit feature space.  Projects that contain multiple explicit versions,
    or a mix of explicit and legacy rows, are rejected unless the caller
    requests one exact ``feature_version``.
    """
    csvs = sorted([p for p in dataframes_dir.glob("*.csv") if p.is_file()])
    if not csvs:
        raise FileNotFoundError(f"Sem CSVs em {dataframes_dir}")

    df0 = pd.read_csv(csvs[0], nrows=50)
    label_col = _pick_label_column(df0)
    base_cols = ["names", "embed_file", "embed_idx", label_col]
    missing = [c for c in base_cols if c not in df0.columns]
    if missing:
        raise RuntimeError(
            f"CSV não tem colunas necessárias {missing}. Colunas disponíveis: {list(df0.columns)}"
        )

    parts = []
    for p in csvs:
        t0 = _now()
        header = pd.read_csv(p, nrows=0)
        if label_col not in header.columns:
            raise RuntimeError(f"{p.name}: coluna de classe esperada {label_col!r} não existe.")
        wanted = list(base_cols)
        if "feature_model_version" in header.columns:
            wanted.append("feature_model_version")
        df = pd.read_csv(
            p,
            usecols=wanted,
            dtype={"names": "string", "embed_file": "string"},
            low_memory=False,
        )
        if "feature_model_version" not in df.columns:
            df["feature_model_version"] = ""
        df["feature_model_version"] = (
            df["feature_model_version"].fillna("").astype("string").str.strip()
        )
        df["embed_idx"] = pd.to_numeric(df["embed_idx"], errors="coerce").fillna(-1).astype(np.int32)
        df[label_col] = df[label_col].astype("string")
        df = df[
            (df["embed_idx"] >= 0)
            & df["names"].notna()
            & df["embed_file"].notna()
            & df[label_col].notna()
        ]
        parts.append(df)
        if verbose:
            _log(f"[index] {p.name}: {len(df)} linhas (t={_now()-t0:.2f}s)")

    out = pd.concat(parts, ignore_index=True, copy=False)
    version_series = out["feature_model_version"].fillna("").astype(str).str.strip()
    explicit_versions = sorted(v for v in version_series.unique().tolist() if v)
    has_legacy = bool((version_series == "").any())

    if feature_version:
        wanted_version = str(feature_version).strip()
        keep = version_series == wanted_version
        if not bool(keep.any()):
            raise RuntimeError(
                f"Nenhuma linha possui feature_model_version={wanted_version!r}; "
                "clustering abortado para evitar usar embeddings de outro modelo."
            )
        out = out.loc[keep].copy().reset_index(drop=True)
        version_series = out["feature_model_version"].astype(str)
        out.attrs["feature_model_version"] = wanted_version
    else:
        if len(explicit_versions) > 1 or (explicit_versions and has_legacy):
            detail = explicit_versions + (["<legacy>"] if has_legacy else [])
            raise RuntimeError(
                "Embeddings de versões de modelo diferentes foram encontrados: "
                f"{detail}. Execute clustering para uma única feature_model_version "
                "ou atualize as features dos batches antes de combinar o projeto."
            )
        out.attrs["feature_model_version"] = explicit_versions[0] if explicit_versions else "legacy"

    out["embed_file"] = out["embed_file"].astype("category")
    out[label_col] = out[label_col].astype("category")
    out["feature_model_version"] = out["feature_model_version"].astype("category")
    out.attrs["label_col"] = label_col
    del parts
    return out


def _get_cached_feats(
    embed_cache: "OrderedDict[str, np.ndarray]",
    path: Path,
    max_items: int,
    verbose: bool = False,
    max_bytes: int | None = None,
) -> np.ndarray:
    """LRU cache for compressed embedding batches.

    A single batch of 8k x 768 float32 is ~24 MiB. Classes are processed one
    after another and every class usually spans every batch, so a cache that
    is smaller than the project forces each ``.npz`` to be decompressed again
    for every class. The cache is bounded by item count and, optionally, by a
    RAM budget in bytes (the most recently used batch is always kept).
    """
    key = str(path)
    arr = embed_cache.get(key)
    if arr is not None:
        embed_cache.move_to_end(key)
        return arr
    t0 = _now()
    arr = _load_feats_from_npz(path)
    embed_cache[key] = arr
    embed_cache.move_to_end(key)
    while len(embed_cache) > max(1, int(max_items)):
        embed_cache.popitem(last=False)
    if max_bytes is not None:
        total = sum(int(a.nbytes) for a in embed_cache.values())
        while len(embed_cache) > 1 and total > int(max_bytes):
            _, old = embed_cache.popitem(last=False)
            total -= int(old.nbytes)
    if verbose:
        _log(f"[embed] load {path.name} shape={arr.shape} (t={_now()-t0:.2f}s)")
    return arr


def _gather_class_matrix(
    df_index: pd.DataFrame,
    project_dir: Path,
    class_label: str,
    embed_cache: "OrderedDict[str, np.ndarray]",
    embed_cache_max: int = 4,
    verbose: bool = False,
    embed_cache_bytes: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return embeddings and image names for one class.

    The embedding cache is bounded and invalid indices are tracked while each
    batch is already in memory, avoiding the old second validation pass that
    could force every embedding batch to remain resident.
    """
    label_col = df_index.attrs["label_col"]
    dfc = df_index[df_index[label_col] == str(class_label)]
    if dfc.empty:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=object)

    names = dfc["names"].astype(str).to_numpy(dtype=object)
    embed_files = dfc["embed_file"].astype(str).to_numpy(dtype=object)
    embed_idxs = dfc["embed_idx"].to_numpy(dtype=np.int32, copy=False)

    file_to_pos: Dict[str, List[int]] = {}
    for pos, ef in enumerate(embed_files):
        file_to_pos.setdefault(ef, []).append(pos)

    first_file = next(iter(file_to_pos.keys()))
    first_path = _resolve_embed_path(project_dir, first_file)
    first_feats = _get_cached_feats(embed_cache, first_path, embed_cache_max, verbose, embed_cache_bytes)
    if first_feats.ndim != 2:
        raise ValueError(f"Embedding inválido em {first_path.name}: esperado array 2D, recebido shape {first_feats.shape}.")
    D = int(first_feats.shape[1])

    N = len(names)
    X = np.empty((N, D), dtype=np.float32)
    valid_mask = np.zeros(N, dtype=bool)
    bad = 0

    for ef, positions in file_to_pos.items():
        ep = _resolve_embed_path(project_dir, ef)
        feats = _get_cached_feats(embed_cache, ep, embed_cache_max, verbose, embed_cache_bytes)
        if feats.ndim != 2:
            raise ValueError(f"Embedding inválido em {ep.name}: esperado array 2D, recebido shape {feats.shape}.")
        if feats.shape[1] != D:
            raise ValueError(f"Dimensão de embedding incompatível em {ep.name}: {feats.shape[1]} != {D} ({first_path.name}).")
        pos_arr = np.asarray(positions, dtype=np.int64)
        idxs = embed_idxs[pos_arr].astype(np.int64, copy=False)
        ok = (idxs >= 0) & (idxs < feats.shape[0])
        if not np.all(ok):
            bad += int((~ok).sum())
        if not np.any(ok):
            continue
        pos_ok = pos_arr[ok]
        idx_ok = idxs[ok]
        X[pos_ok, :] = feats[idx_ok, :].astype(np.float32, copy=False)
        valid_mask[pos_ok] = True

    if bad:
        _log(f"[WARN] classe='{class_label}': {bad} linha(s) com embed_idx inválido ignorada(s).")
    if not np.all(valid_mask):
        X = X[valid_mask]
        names = names[valid_mask]

    return X, names


# ======================================================================================
# Clustering: full vs centroid-expanded linkage
# ======================================================================================

def _apply_pca_if_needed(X: np.ndarray, pca_dim: int, verbose: bool = False) -> np.ndarray:
    """Compatibility wrapper; numerical PCA logic lives in ilt_clustering.py."""
    Xr, _ = _shared_apply_pca_if_needed(
        X,
        pca_dim,
        log=_log if verbose else None,
    )
    return Xr


def _centroid_expanded_linkage(
    X: np.ndarray,
    method: str,
    metric: str,
    max_centroids: int,
    verbose: bool = False,
) -> np.ndarray:
    """Compatibility wrapper around the single shared clustering engine."""
    return _shared_centroid_expanded_linkage(
        X,
        method=method,
        metric=metric,
        max_centroids=max_centroids,
        log=_log if verbose else None,
    )


def _full_linkage(X: np.ndarray, method: str, metric: str, verbose: bool = False) -> np.ndarray:
    """Compatibility wrapper around the single shared clustering engine."""
    return _shared_full_linkage(
        X,
        method=method,
        metric=metric,
        log=_log if verbose else None,
    )


# ======================================================================================
# Main
# ======================================================================================

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("project", help="nome do projeto em main/assets/<project>/")
    ap.add_argument("--project-dir", default=None,
                    help="caminho direto para a pasta do projeto. Se omitido, usa main/assets/<project> a partir do cwd")
    ap.add_argument("--class", dest="one_class", default=None, help="processa apenas esta classe")
    ap.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"], help="métrica do linkage")
    ap.add_argument("--linkage", dest="method", default="average",
                    choices=["average", "complete", "single", "ward"],
                    help="método do linkage (ward só faz sentido com euclidean)")
    ap.add_argument("--pca-dim", type=int, default=64, help="dimensão PCA (0 desliga). default=64")
    ap.add_argument("--max-full", type=int, default=1500, help="até N<=max_full usa linkage completo. default=1500")
    ap.add_argument("--max-centroids", type=int, default=1500, help="limite de centroides no fallback. default=1500")
    ap.add_argument("--min-samples", type=int, default=2, help="mínimo de amostras por classe. default=2")
    ap.add_argument("--verbose", action="store_true", help="logs detalhados")
    ap.add_argument("--feature-version", default=None,
                    help="usa somente embeddings desta feature_model_version; sem este filtro, versões mistas abortam")
    args = ap.parse_args(argv)

    project = args.project
    method = args.method
    metric = args.metric

    if method == "ward" and metric != "euclidean":
        raise SystemExit("ward requer metric=euclidean (caso contrário, distâncias ficam inválidas).")

    if args.project_dir:
        project_dir = Path(args.project_dir).expanduser().resolve()
    else:
        cwd = Path(os.getcwd())
        project_dir = cwd / "main" / "assets" / project
    dataframes_dir = project_dir / "dataframes"
    clusters_dir = project_dir / "clusters"

    if not dataframes_dir.is_dir():
        raise SystemExit(f"[ERROR] dataframes não encontrado: {dataframes_dir}")

    clusters_dir.mkdir(parents=True, exist_ok=True)

    _log(f"[clustering] project={project}")
    _log(f"[clustering] metric={metric} linkage={method} pca_dim={args.pca_dim} max_full={args.max_full} max_centroids={args.max_centroids}")

    t_all = _now()
    df_index = _read_index_table(dataframes_dir, verbose=args.verbose, feature_version=args.feature_version)
    label_col = df_index.attrs["label_col"]
    _log(f"[clustering] label_col='{label_col}' | rows={len(df_index)} | feature_version={df_index.attrs.get('feature_model_version')}")

    classes = sorted(
        x for x in df_index[label_col].dropna().astype(str).str.strip().unique().tolist()
        if x
    )
    if args.one_class is not None:
        classes = [args.one_class]
        _log(f"[clustering] --class={args.one_class}")
    else:
        _log(f"[clustering] --class ausente -> rodando para {len(classes)} classes.")

    # Reject filename collisions before any class artifact can overwrite another.
    safe_names = ensure_unique_sanitized_labels(classes)

    # ILT_CLUSTER_EMBED_CACHE=<n> keeps the historic fixed item limit.  By
    # default the cache may hold every batch, bounded by a RAM budget, so each
    # compressed .npz is decompressed once instead of once per class.
    env_cache = str(os.environ.get("ILT_CLUSTER_EMBED_CACHE", "")).strip()
    embed_cache_bytes = None
    if env_cache:
        embed_cache_max = max(1, int(env_cache))
        _log(f"[clustering] embedding cache: até {embed_cache_max} batch(es)")
    else:
        embed_cache_max = max(1, int(df_index["embed_file"].nunique()))
        embed_cache_bytes = memory_budget_bytes()
        _log(
            f"[clustering] embedding cache: até {embed_cache_max} batch(es) "
            f"/ {embed_cache_bytes / 1024 ** 3:.1f} GiB"
        )
    embed_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    ok = 0
    fail = 0

    for cls in classes:
        t0 = _now()
        _log(f"\n[clustering] class={cls}")

        try:
            X, names = _gather_class_matrix(
                df_index, project_dir, cls, embed_cache, embed_cache_max,
                verbose=args.verbose, embed_cache_bytes=embed_cache_bytes,
            )
            N = int(X.shape[0])
            if N < args.min_samples:
                _log(f"[skip] N={N} < min_samples={args.min_samples}")
                continue

            _log(f"[clustering] selected rows: {N}")

            result = hierarchical_linkage(
                X,
                method=method,
                metric=metric,
                pca_dim=args.pca_dim,
                max_full=args.max_full,
                max_centroids=args.max_centroids,
                log=_log if args.verbose else None,
            )

            out_path = clusters_dir / f"{safe_names[cls]}.npz"
            _shared_save_cluster_artifact(
                out_path,
                names,
                result,
                metric=metric,
                method=method,
                pca_dim=args.pca_dim,
                max_full=args.max_full,
            )

            _log(
                f"[OK] saved: {out_path} | mode={result.mode} | "
                f"dist_max={result.dist_max:.6g} | t={_now()-t0:.2f}s"
            )
            ok += 1

        except Exception as e:
            fail += 1
            _log(f"[ERROR] Falhou na classe '{cls}': {e} (t={_now()-t0:.2f}s)")
            continue

    _log(f"\n[done] ok={ok} fail={fail} total={len(classes)} | t_total={_now()-t_all:.2f}s")
    if fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

