#!/usr/bin/env python3
"""Flat groups of a hierarchical-clustering artifact at a given cut (no UI).

Used by the IAT › Clustering page to show, for the current cut, every group
with its size, label composition and the images that best represent it.

Representatives are the members whose L2-normalised embedding has the highest
cosine similarity to the group's mean direction (the spherical centroid), i.e.
the most typical images of the group (a medoid-like choice for the cosine
geometry used by the clustering; Dhillon & Modha 2001, *Concept decompositions
for large sparse text data using clustering*, Machine Learning 42:143-175).
The flat cut is SciPy's ``fcluster(criterion='distance')`` (Müllner 2011,
arXiv:1109.2378), the same function used by "Relabel por grupo".
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

MAX_EMBEDDINGS = 60000  # above this the representatives are a seeded random sample


def _id_maps(rows: pd.DataFrame):
    gid = rows['gid'].astype(str).to_numpy(object)
    names = rows['names'].astype(str).to_numpy(object) if 'names' in rows.columns else gid
    labels = rows['manual_label'].fillna('').astype(str).to_numpy(object) if 'manual_label' in rows.columns \
        else np.array([''] * len(rows), dtype=object)
    batches = rows['csv_basename'].astype(str).to_numpy(object) if 'csv_basename' in rows.columns \
        else np.array([''] * len(rows), dtype=object)
    return gid, names, labels, batches


def embedding_index(project, rows: pd.DataFrame):
    """``(gid array, L2-normalised float32 matrix)`` for *rows*, or ``None``."""
    if rows is None or rows.empty or len(rows) > MAX_EMBEDDINGS:
        return None
    try:
        from iat_backend import _l2, gather_embeddings
        X, gids = gather_embeddings(project, rows, id_col='gid')
    except Exception:
        return None
    if len(X) == 0:
        return None
    return np.asarray(gids, dtype=object).astype(str), _l2(np.asarray(X, dtype=np.float32))


def cluster_group_summary(rows: pd.DataFrame, artifact: dict, cut01: float, id_mode: str,
                          embeddings=None, max_groups: int = 24, n_repr: int = 4) -> dict:
    """Groups at *cut01* (fraction of ``dist_max``), largest first.

    Each group: ``id`` (fcluster label), ``n``, ``composition`` [(label, frac)],
    ``dominant``, ``dominant_frac``, ``gids`` (all members) and ``repr``
    [(csv_basename, image name)] for up to *n_repr* representative images.
    """
    from iat_backend import cluster_groups
    names = [str(x) for x in artifact.get('names', [])]
    if not names:
        return {'groups': [], 'n_groups': 0, 'n': 0, 'cut': float(cut01)}
    flat = np.asarray(cluster_groups(artifact, float(cut01)), dtype=int)
    gid_arr, name_arr, label_arr, batch_arr = _id_maps(rows)
    pos_by_gid = {g: i for i, g in enumerate(gid_arr)}
    if id_mode == 'gid':
        row_pos = [pos_by_gid.get(n) for n in names]
    else:
        pos_by_name = {n: i for i, n in enumerate(name_arr)}
        row_pos = [pos_by_name.get(n) for n in names]
    emb_pos = {}
    X = None
    if embeddings is not None:
        e_gids, X = embeddings
        emb_pos = {g: i for i, g in enumerate(e_gids)}
    rng = np.random.default_rng(0)
    members = {}
    for k, p in zip(flat.tolist(), row_pos):
        if p is not None:
            members.setdefault(int(k), []).append(int(p))
    groups = []
    for k, idx in members.items():
        idx = np.asarray(idx, dtype=int)
        labs = Counter(label_arr[idx].tolist())
        n = int(len(idx))
        comp = [(lab or '—', cnt / n) for lab, cnt in labs.most_common(4)]
        rep_idx = idx[:n_repr]
        if X is not None:
            e_idx = np.array([emb_pos.get(gid_arr[i], -1) for i in idx], dtype=int)
            ok = e_idx >= 0
            if ok.sum() >= 1:
                V = X[e_idx[ok]]
                c = V.mean(axis=0)
                c /= max(float(np.linalg.norm(c)), 1e-12)
                order = np.argsort(-(V @ c), kind='stable')[:n_repr]
                rep_idx = idx[ok][order]
        elif n > n_repr:
            rep_idx = np.sort(rng.choice(idx, n_repr, replace=False))
        groups.append({
            'id': k, 'n': n, 'composition': comp, 'dominant': comp[0][0], 'dominant_frac': comp[0][1],
            'gids': [str(g) for g in gid_arr[idx]],
            'repr': [(str(batch_arr[i]), str(name_arr[i])) for i in rep_idx],
        })
    groups.sort(key=lambda g: (-g['n'], g['id']))
    return {'groups': groups[:max_groups], 'all_groups': groups, 'n_groups': len(groups),
            'n': int(sum(g['n'] for g in groups)), 'cut': float(cut01)}
