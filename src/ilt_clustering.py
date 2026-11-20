#!/usr/bin/env python3
"""Single numerical clustering engine shared by the ILT CLI and IAT.

The project historically had two large implementations of hierarchical
clustering: one in ``class_clustering.py`` and one in ``iat_backend.py``.  This
module is the authoritative implementation used by both paths so fixes cannot
silently diverge again.

It does not decide which rows/classes should be clustered.  Callers remain
responsible for selecting rows and gathering embeddings.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ilt_contracts import atomic_savez_compressed, validate_embedding_matrix

try:
    from scipy.cluster.hierarchy import fcluster, linkage
except Exception:  # pragma: no cover - reported clearly at runtime
    fcluster = linkage = None

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover - reported clearly at runtime
    MiniBatchKMeans = PCA = None


ProgressFn = Optional[Callable[[int, int, str], None]]
LogFn = Optional[Callable[[str], None]]


@dataclass(frozen=True)
class HierarchicalClusteringResult:
    linkage: np.ndarray
    mode: str
    dist_max: float
    pca_dim_used: int
    n_samples: int
    n_features_in: int


def _emit(log: LogFn, message: str) -> None:
    if log is not None:
        log(str(message))


def validate_clustering_request(method: str, metric: str) -> tuple[str, str]:
    """Validate linkage/metric values used by every ILT clustering entry point."""
    method = str(method).strip().lower()
    metric = str(metric).strip().lower()
    if method not in {"average", "complete", "single", "ward"}:
        raise ValueError(f"Unsupported linkage method: {method!r}.")
    if metric not in {"cosine", "euclidean"}:
        raise ValueError(f"Unsupported clustering metric: {metric!r}.")
    if method == "ward" and metric != "euclidean":
        raise ValueError("ward requer metric=euclidean")
    return method, metric


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Return float32 row-wise L2-normalized embeddings."""
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def apply_pca_if_needed(
    X: np.ndarray,
    pca_dim: int,
    *,
    log: LogFn = None,
) -> tuple[np.ndarray, int]:
    """Apply the historical randomized PCA rule and return (matrix, used_dim)."""
    X = validate_embedding_matrix(X, source="clustering embeddings")
    N, D = X.shape
    requested = int(pca_dim)
    if requested <= 0 or PCA is None:
        if requested > 0 and PCA is None:
            _emit(log, "[PCA] sklearn não disponível -> pulando PCA.")
        return X, int(D)

    ncomp = int(min(requested, N, D))
    if ncomp < 2 or ncomp >= D:
        return X, int(D)

    t0 = time.perf_counter()
    pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=0)
    Xr = pca.fit_transform(X).astype(np.float32, copy=False)
    _emit(
        log,
        f"[PCA] ajustado: pedido={requested} -> usando={ncomp} "
        f"(N={N}, D={D}) (t={time.perf_counter()-t0:.2f}s)",
    )
    return Xr, int(ncomp)


def full_linkage(
    X: np.ndarray,
    *,
    method: str,
    metric: str,
    log: LogFn = None,
) -> np.ndarray:
    """Run the exact SciPy hierarchical linkage path."""
    if linkage is None:
        raise RuntimeError("scipy não está disponível (linkage).")
    method, metric = validate_clustering_request(method, metric)
    X = validate_embedding_matrix(X, source="clustering embeddings")
    if len(X) < 2:
        return np.zeros((0, 4), dtype=np.float64)

    t0 = time.perf_counter()
    Z = linkage(X, method=method, metric=metric)
    Z = np.asarray(Z, dtype=np.float64)
    _emit(log, f"[full] linkage done: Z={Z.shape} (t={time.perf_counter()-t0:.2f}s)")
    return Z


def centroid_expanded_linkage(
    X: np.ndarray,
    *,
    method: str,
    metric: str,
    max_centroids: int,
    progress: ProgressFn = None,
    log: LogFn = None,
) -> np.ndarray:
    """Hierarchical clustering on occupied MiniBatchKMeans centroids.

    The centroid hierarchy is expanded back to a valid SciPy linkage with all N
    original leaves.  Empty requested centroids are removed before linkage; this
    is the v6.5 regression fix that prevents writing past the N-1 linkage rows.
    """
    if linkage is None:
        raise RuntimeError("scipy não está disponível (linkage).")
    if MiniBatchKMeans is None:
        raise RuntimeError("scikit-learn não está disponível (MiniBatchKMeans).")
    method, metric = validate_clustering_request(method, metric)
    X = validate_embedding_matrix(X, source="clustering embeddings")

    N, _ = X.shape
    if N < 2:
        return np.zeros((0, 4), dtype=np.float64)

    K_requested = int(min(
        max(1, int(max_centroids)),
        max(2, int(math.sqrt(N) * 20)),
        N,
    ))
    _emit(log, f"[centroid] N={N} -> K={K_requested} centroides")
    if progress:
        progress(0, 3, f"K-means: {K_requested} centroides")

    t0 = time.perf_counter()
    km = MiniBatchKMeans(
        n_clusters=K_requested,
        random_state=0,
        batch_size=2048,
        max_iter=200,
        n_init=3,
        reassignment_ratio=0.01,
    )
    labels_raw = km.fit_predict(X)
    centers_all = l2_normalize(km.cluster_centers_.astype(np.float32, copy=False))

    occupied = np.unique(labels_raw).astype(np.int64)
    if occupied.size == 0:
        raise RuntimeError("MiniBatchKMeans não produziu nenhum centróide ocupado.")

    labels = np.searchsorted(occupied, labels_raw).astype(np.int64)
    centers = centers_all[occupied]
    K = int(len(occupied))

    _emit(
        log,
        f"[centroid] kmeans done: ocupados={K}/{K_requested} "
        f"(t={time.perf_counter()-t0:.2f}s)",
    )
    if progress and K != K_requested:
        progress(1, 3, f"Centroides ocupados: {K}/{K_requested}")

    t1 = time.perf_counter()
    Zc = (
        linkage(centers, method=method, metric=metric)
        if K > 1
        else np.zeros((0, 4), dtype=np.float64)
    )
    _emit(
        log,
        f"[centroid] linkage(centroids) done: Zc={Zc.shape} "
        f"(t={time.perf_counter()-t1:.2f}s)",
    )

    order = np.argsort(labels, kind="mergesort")
    labels_sorted = labels[order]
    Z = np.zeros((N - 1, 4), dtype=np.float64)
    sizes = np.ones(2 * N + 4, dtype=np.int64)
    centroid_leaf = np.full(K, -1, dtype=np.int64)

    next_cluster = N
    row = 0
    start = 0

    for k in range(K):
        end = start
        while end < N and labels_sorted[end] == k:
            end += 1
        members = order[start:end]

        if len(members) <= 0:
            raise RuntimeError(f"Centróide compacto vazio inesperado: {k}")

        if len(members) == 1:
            centroid_leaf[k] = int(members[0])
        else:
            cur = int(members[0])
            cur_size = 1
            for leaf in members[1:]:
                Z[row] = [cur, int(leaf), 0.0, cur_size + 1]
                sizes[next_cluster] = cur_size + 1
                cur = next_cluster
                cur_size += 1
                next_cluster += 1
                row += 1
            centroid_leaf[k] = cur
        start = end

    # There are exactly N-K internal merges before the centroid hierarchy.
    if row != N - K:
        raise RuntimeError(f"Merges internos inconsistentes: {row} != {N-K}")

    base = next_cluster
    for r in range(len(Zc)):
        a, b, dist = int(Zc[r, 0]), int(Zc[r, 1]), float(Zc[r, 2])
        ag = int(centroid_leaf[a]) if a < K else int(base + (a - K))
        bg = int(centroid_leaf[b]) if b < K else int(base + (b - K))
        newg = N + row
        size_new = int(sizes[ag] + sizes[bg])
        Z[row] = [ag, bg, dist, size_new]
        sizes[newg] = size_new
        row += 1

    if row != N - 1:
        raise RuntimeError(f"Linkage expandido inconsistente: {row} != {N-1}")
    return Z


def hierarchical_linkage(
    X: np.ndarray,
    *,
    method: str = "average",
    metric: str = "cosine",
    pca_dim: int = 64,
    max_full: int = 1500,
    max_centroids: int = 1500,
    progress: ProgressFn = None,
    log: LogFn = None,
) -> HierarchicalClusteringResult:
    """Run the single ILT hierarchical clustering pipeline.

    Scientific behavior intentionally matches v6.7:
    L2 normalize -> optional PCA -> L2 normalize after PCA -> exact linkage for
    small N or centroid-expanded linkage for large N.
    """
    method, metric = validate_clustering_request(method, metric)
    X = validate_embedding_matrix(X, source="clustering embeddings")
    N, D = X.shape
    if N < 2:
        raise ValueError(f"N={N} < min_samples=2")

    Xn = l2_normalize(X)
    Xr, used_dim = apply_pca_if_needed(Xn, int(pca_dim), log=log)
    if Xr is not Xn:
        Xr = l2_normalize(Xr)

    threshold = max(1, int(max_full))
    if N <= threshold:
        Z = full_linkage(Xr, method=method, metric=metric, log=log)
        mode = "full"
    else:
        Z = centroid_expanded_linkage(
            Xr,
            method=method,
            metric=metric,
            max_centroids=max_centroids,
            progress=progress,
            log=log,
        )
        mode = "centroid"

    if Z.shape != (N - 1, 4):
        raise RuntimeError(
            f"Linkage final inválido: shape={Z.shape}, esperado={(N-1, 4)}."
        )
    dist_max = float(np.max(Z[:, 2])) if Z.size else 1.0

    return HierarchicalClusteringResult(
        linkage=np.asarray(Z, dtype=np.float64),
        mode=mode,
        dist_max=dist_max,
        pca_dim_used=int(used_dim),
        n_samples=int(N),
        n_features_in=int(D),
    )


def save_cluster_artifact(
    path,
    names,
    result: HierarchicalClusteringResult,
    *,
    metric: str,
    method: str,
    pca_dim: int,
    max_full: int,
) -> Path:
    """Save the historical ILT cluster artifact schema atomically."""
    path = Path(path)
    atomic_savez_compressed(
        path,
        names=np.asarray(names, dtype=str),
        linkage=np.asarray(result.linkage, dtype=np.float32),
        dist_max=np.float32(result.dist_max),
        mode=np.array(result.mode),
        metric=np.array(str(metric)),
        method=np.array(str(method)),
        pca_dim=np.int32(pca_dim),
        max_full=np.int32(max_full),
    )
    return path


def read_cluster_artifact(path) -> dict:
    """Read current or historical cluster artifacts.

    ``allow_pickle=True`` is retained only for compatibility with historical
    artifacts whose ``names`` array was written as dtype=object. New artifacts
    always write strings.
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        def scalar(key, default=None):
            if key not in z.files:
                return default
            arr = np.asarray(z[key])
            return arr.item() if arr.size == 1 else arr

        return {
            "path": path,
            "names": np.asarray(z["names"]).astype(str),
            "linkage": np.asarray(z["linkage"], dtype=float),
            "dist_max": float(scalar("dist_max", 1.0)),
            "mode": str(scalar("mode", "full")),
            "metric": str(scalar("metric", "cosine")),
            "method": str(scalar("method", "average")),
            "pca_dim": int(scalar("pca_dim", 64)),
        }


def read_artifact_names(path) -> np.ndarray:
    """Read only artifact identifiers without loading a potentially large linkage."""
    with np.load(Path(path), allow_pickle=True) as z:
        if "names" not in z.files:
            raise ValueError(f"Artifact without names: {path}")
        return np.asarray(z["names"]).astype(str)


def cluster_groups(artifact: dict, cut01: float) -> np.ndarray:
    """Return 1-based flat group ids from an ILT clustering artifact."""
    names = np.asarray(artifact.get("names", []))
    N = len(names)
    Z = np.asarray(artifact.get("linkage", []), dtype=float)
    if N < 2 or Z.size == 0:
        return np.ones(N, dtype=int)
    if fcluster is None:
        raise RuntimeError("scipy não está disponível (fcluster).")
    h = float(cut01) * max(float(artifact.get("dist_max", 1.0)), 1e-12)
    return fcluster(Z, t=h, criterion="distance").astype(int)
