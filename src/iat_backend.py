#!/usr/bin/env python3
"""Backend Python do IAT integrado ao ILT Data Extraction.

Porta as operações centrais do IAT/Shiny em R para Python, mantendo a estrutura
em disco gerada pela classificação: dataframes/, images/, embeddings/, clusters/.
"""
from __future__ import annotations

import json
import hashlib
import colorsys
from collections import OrderedDict
import math
import os
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ilt_contracts import (
    IMAGE_EXTENSIONS, atomic_write_csv, atomic_write_text, safe_perplexity,
    ensure_unique_sanitized_labels,
    sanitize_filename as _contract_sanitize_filename,
)
from project_settings import DEFAULT_PIXEL_SIZE_UM_PER_PX
from hw_resources import available_cpus, cpu_jobs
from concurrent.futures import ThreadPoolExecutor
# Heavy/optional scientific dependencies are loaded on first use.  This keeps
# opening the GUI and loading ordinary project CSVs independent from clustering,
# segmentation and assisted-validation libraries.
cv2 = None
_cv2_checked = False

SklearnHDBSCAN = None
PCA = TSNE = trustworthiness = silhouette_score = NearestNeighbors = None
_sklearn_checked = False

hdbscan_external = None
_hdbscan_checked = False
umap_module = None
_umap_checked = False
pacmap_module = None
_pacmap_checked = False


def _load_cv2():
    global cv2, _cv2_checked
    if not _cv2_checked:
        _cv2_checked = True
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except Exception:
            cv2 = None
    return cv2


def _load_sklearn():
    global SklearnHDBSCAN, PCA, TSNE, trustworthiness, silhouette_score, NearestNeighbors, _sklearn_checked
    if not _sklearn_checked:
        _sklearn_checked = True
        try:
            from sklearn.cluster import HDBSCAN as _SklearnHDBSCAN
        except Exception:
            _SklearnHDBSCAN = None
        try:
            from sklearn.decomposition import PCA as _PCA
            from sklearn.manifold import TSNE as _TSNE, trustworthiness as _trustworthiness
            from sklearn.metrics import silhouette_score as _silhouette_score
            from sklearn.neighbors import NearestNeighbors as _NearestNeighbors
        except Exception:
            _PCA = _TSNE = _trustworthiness = _silhouette_score = _NearestNeighbors = None
        SklearnHDBSCAN = _SklearnHDBSCAN
        PCA = _PCA
        TSNE = _TSNE
        trustworthiness = _trustworthiness
        silhouette_score = _silhouette_score
        NearestNeighbors = _NearestNeighbors
    return PCA is not None


def _load_hdbscan_external():
    global hdbscan_external, _hdbscan_checked
    if not _hdbscan_checked:
        _hdbscan_checked = True
        try:
            import hdbscan as _hdbscan_external
            hdbscan_external = _hdbscan_external
        except Exception:
            hdbscan_external = None
    return hdbscan_external


def _load_umap():
    global umap_module, _umap_checked
    if not _umap_checked:
        _umap_checked = True
        try:
            import umap as _umap_module
            umap_module = _umap_module
        except Exception:
            umap_module = None
    return umap_module


def _load_pacmap():
    global pacmap_module, _pacmap_checked
    if not _pacmap_checked:
        _pacmap_checked = True
        try:
            import pacmap as _pacmap_module
            pacmap_module = _pacmap_module
        except Exception:
            pacmap_module = None
    return pacmap_module


# Clustering itself imports SciPy/scikit-learn.  Keep the public wrappers but
# defer importing that engine until a clustering operation is requested.
def _shared_cluster_groups(*args, **kwargs):
    from ilt_clustering import cluster_groups as _fn
    return _fn(*args, **kwargs)


def hierarchical_linkage(*args, **kwargs):
    from ilt_clustering import hierarchical_linkage as _fn
    return _fn(*args, **kwargs)


def read_artifact_names(*args, **kwargs):
    from ilt_clustering import read_artifact_names as _fn
    return _fn(*args, **kwargs)


def _shared_read_cluster_artifact(*args, **kwargs):
    from ilt_clustering import read_cluster_artifact as _fn
    return _fn(*args, **kwargs)


def _shared_save_cluster_artifact(*args, **kwargs):
    from ilt_clustering import save_cluster_artifact as _fn
    return _fn(*args, **kwargs)

IMAGE_EXTS = IMAGE_EXTENSIONS
# Version of the segmentation measurements written by segmentation_metrics().
# 2 = Crofton perimeter, true Feret diameters (rotating calipers), consistent
# pixel-set geometry (hull/solidity/rectangularity), ESD, moment ellipse,
# biovolume and intensity statistics.  Files without the column are version 1.
SEGMENTATION_METRICS_VERSION = 2
# Per-image measurements, in the order they are written to the CSV.
SEGMENTATION_METRIC_KEYS = (
    'Area (pxl)', 'Image Width (pxl)', 'Image Height (pxl)', 'Image Size (pxl)',
    'Perimeter (pxl)', 'ESD (pxl)', 'circularity', 'Elongation', 'Rectangularity', 'Solidity',
    'Feret_max', 'Feret_min', 'MinRect_w (pxl)', 'MinRect_h (pxl)', 'Hull Area (pxl)',
    'Major (pxl)', 'Minor (pxl)', 'Eccentricity', 'Orientation (deg)', 'Biovolume (pxl3)',
    'Mean Intensity', 'Median Intensity', 'Std Intensity', 'Min Intensity', 'Max Intensity',
    'Skewness', 'Kurtosis', 'Contrast', 'Percent_on_border', 'Number Holes',
    'Validation State', 'Metrics Version',
)
SEGMENTATION_COLUMNS = {
    'Image Name', 'Image State (T/F)', 'SegmentationMethod', 'Pixel Size (um/pixel)',
    *SEGMENTATION_METRIC_KEYS,
    'Event Indexer', 'Class Name', 'ClassName_previuos', 'Background Image',
}
VALIDATION_TRANSIENT_COLUMNS = {
    'analysis_x', 'analysis_y',
    'knn_agreement', 'knn_disagreement', 'knn_dominant_label',
    'hdbscan_cluster', 'hdbscan_probability', 'hdbscan_outlier',
    'hdbscan_dominant_label', 'hdbscan_label_agreement', 'hdbscan_label_mismatch',
    'review_priority', 'manual_prediction_mismatch', 'source_prediction_mismatch',
}

SEGMENTATION_METHODS = {
    'otsu': 'Otsu com auto-polaridade',
    'rolling_ball': 'Rolling ball + Otsu',
    'fixed_243': 'Fixed 243 estilo ZooProcess',
    'black_hat_otsu': 'Black-hat + Otsu',
    'adaptive_sauvola': 'Adaptive / Sauvola',
    'percentile_dark': 'Percentile dark-object',
    'watershed_otsu': 'Watershed pós-Otsu',
    'kmeans_intensity': 'K-means de intensidade',
    'otsu_lcc': 'Otsu + maior componente/filtro de área',
}


def get_color(i: int) -> str:
    """Mesma regra determinística de cor usada no csv_io.R."""
    i = int(i or 0)
    r = (201 * i + 100) % 256
    g = (91 * i + 149) % 256
    b = (53 * i + 237) % 256
    return f'#{r:02x}{g:02x}{b:02x}'


def get_class_color(label: str) -> str:
    """Stable class-name color across batches, sessions and operating systems.

    This is a display-only color. It intentionally does not change the
    numerical ``colors`` column in existing CSVs or the legacy get_color().
    """
    name = str(label).strip()
    if not name:
        return get_color(0)
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:4], "big") / 2**32
    saturation = 0.66 + (digest[4] / 255.0) * 0.14
    value = 0.77 + (digest[5] / 255.0) * 0.12
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def sanitize_filename(x: str) -> str:
    """Backward-compatible wrapper around the shared artifact-name contract."""
    return _contract_sanitize_filename(x)


def infer_batch_folder(csv_basename: str) -> Optional[str]:
    stem = Path(csv_basename).stem
    m = re.match(r'^(batch_\d+)', stem)
    return m.group(1) if m else None


def make_gid(csv_basename: str, custom_data: str) -> str:
    return f'{csv_basename}::{custom_data}'


def split_gid(gid: str) -> Tuple[str, str]:
    if '::' not in str(gid):
        return '', str(gid)
    return tuple(str(gid).split('::', 1))  # type: ignore[return-value]


def ensure_required_columns(df: pd.DataFrame, csv_basename: str = '', copy: bool = True) -> pd.DataFrame:
    """Equivalente ao ensure_required_columns() do csv_io.R."""
    if copy:
        df = df.copy()
    if 'names' not in df.columns:
        raise ValueError("CSV não possui coluna 'names' (obrigatória).")

    if 'x' not in df.columns or 'y' not in df.columns:
        df['x'] = np.arange(1, len(df) + 1, dtype=float)
        df['y'] = 0.0
    else:
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        keep = np.isfinite(df['x'].to_numpy(float)) & np.isfinite(df['y'].to_numpy(float))
        df = df.loc[keep].copy()

    if 'manual_label' not in df.columns:
        if 'correct_label' in df.columns:
            df['manual_label'] = df['correct_label'].astype(str)
        elif 'preds1' in df.columns:
            df['manual_label'] = df['preds1'].astype(str)
        elif 'pred' in df.columns:
            df['manual_label'] = df['pred'].astype(str)
        else:
            df['manual_label'] = 'unlabeled'
    if 'correct_label' not in df.columns:
        df['correct_label'] = df['manual_label'].astype(str)

    # Keep the label inherited from the source folder independent from both
    # model predictions and later human relabeling. Older projects only have
    # ``klass``; derive source_label lazily so they remain compatible.
    if 'klass' in df.columns:
        raw = df['klass'].fillna('').astype(str).str.rstrip('/\\')
        derived_source = raw.str.replace(r'^.*[\\/]', '', regex=True)
        if 'source_label' not in df.columns:
            df['source_label'] = derived_source
        else:
            src = df['source_label'].fillna('').astype(str)
            blank = src.str.strip().eq('')
            src.loc[blank] = derived_source.loc[blank]
            df['source_label'] = src
    if 'source_label' in df.columns:
        df['source_label'] = df['source_label'].fillna('').astype(str)

    if 'custom_data' not in df.columns:
        df['custom_data'] = df['names'].astype(str)
    else:
        s = df['custom_data'].astype('string')
        bad = s.isna() | (s.str.len() == 0)
        s.loc[bad] = df.loc[bad, 'names'].astype(str)
        df['custom_data'] = s.astype(str)

    if 'custom_data_orig' not in df.columns:
        df['custom_data_orig'] = df['custom_data'].astype(str)
    else:
        s = df['custom_data_orig'].astype('string')
        bad = s.isna() | (s.str.len() == 0)
        s.loc[bad] = df.loc[bad, 'custom_data'].astype(str)
        df['custom_data_orig'] = s.astype(str)

    if csv_basename and 'csv_basename' not in df.columns:
        df['csv_basename'] = csv_basename

    if 'colors' not in df.columns:
        labels = sorted(df['manual_label'].astype(str).dropna().unique().tolist())
        cmap = {lab: i for i, lab in enumerate(labels)}
        df['colors'] = df['manual_label'].astype(str).map(cmap).fillna(0).astype(int)
    else:
        df['colors'] = pd.to_numeric(df['colors'], errors='coerce').fillna(0).astype(int)

    if 'confs1' not in df.columns:
        df['confs1'] = np.nan
    else:
        df['confs1'] = pd.to_numeric(df['confs1'], errors='coerce')
    if 'var_norm' not in df.columns:
        df['var_norm'] = np.nan
    else:
        df['var_norm'] = pd.to_numeric(df['var_norm'], errors='coerce')
    # ``var_norm`` = prediction uncertainty from the top-1 − top-2 softmax
    # margin, u = 1 − (p1 − p2) in [0, 1] (margin sampling; Settles 2009).
    # New projects store it at classification time (projections.py); older
    # projects get it here from confs1/confs2 so the Gallery filter, histogram
    # and review priority always have values.
    if {'confs1', 'confs2'}.issubset(df.columns):
        need = df['var_norm'].isna()
        if bool(need.any()):
            p1 = pd.to_numeric(df['confs1'], errors='coerce')
            p2 = pd.to_numeric(df['confs2'], errors='coerce')
            df.loc[need, 'var_norm'] = (1.0 - (p1 - p2)).clip(0.0, 1.0)[need]
    if 'batch' not in df.columns and csv_basename:
        df['batch'] = Path(csv_basename).stem

    if csv_basename:
        df['gid'] = str(csv_basename) + '::' + df['custom_data'].astype(str)
    elif 'gid' not in df.columns:
        df['gid'] = df['custom_data'].astype(str)
    return df.reset_index(drop=True)


def strip_segmentation_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Segmentation joins and assisted-validation diagnostics are derived data.
    # Keep them in memory for the current IAT session, but never let an autosave
    # of manual labels silently persist stale analytical results into the source CSV.
    transient = SEGMENTATION_COLUMNS | VALIDATION_TRANSIENT_COLUMNS
    return df.drop(columns=[c for c in transient if c in df.columns], errors='ignore')


def save_with_backup(df: pd.DataFrame, out_path: Path) -> Path:
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = out_path.parent / 'backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        shutil.copy2(out_path, backup_dir / f'{out_path.name}.{ts}.bak')
    atomic_write_csv(strip_segmentation_columns(df), out_path, index=False)
    return out_path


def order_choices(df: pd.DataFrame, extra_columns: Optional[Sequence[str]] = None) -> List[Tuple[str, str]]:
    """Return one entry per sortable field.

    Sort direction is intentionally not encoded in the visible choice.  The UI
    keeps direction in a separate asc/desc toggle, which avoids duplicating
    every segmentation feature in the combobox.
    """
    cols = set(df.columns)
    cols.update(map(str, extra_columns or []))
    base = [
        # A-Z means alphabetical order of the CURRENT HUMAN/MANUAL label.
        # Image filename remains available as a separate criterion.
        ('A-Z', 'manual_label'),
        ('Image name', 'names'),
        ('Confidence', 'confs1'), ('Variance', 'var_norm'),
        ('Review priority', 'review_priority'), ('kNN agreement', 'knn_agreement'),
        ('HDBSCAN outlier', 'hdbscan_outlier'), ('HDBSCAN probability', 'hdbscan_probability'),
    ]
    base = [(label,key) for label,key in base if key in cols]
    extras = [
        ('Area', 'area', 'area'), ('Perimeter', 'perimeter', 'perimeter'),
        ('Width', 'width', 'width'), ('Height', 'height', 'height'),
        ('Bbox Area', 'bbox_area', 'bbox_area'), ('Bbox Ratio', 'bbox_ratio', 'bbox_ratio'),
        ('Bbox Fill', 'bbox_fill', 'bbox_fill'), ('Otsu', 'otsu', 'otsu'),
        ('Seg Area', 'seg_area', 'Area (pxl)'),
        ('Seg Perimeter', 'seg_perimeter', 'Perimeter (pxl)'),
        ('Seg ESD', 'seg_esd', 'ESD (pxl)'),
        ('Seg Contrast', 'seg_contrast', 'Contrast'),
        ('Seg Circularity', 'seg_circularity', 'circularity'),
        ('Seg Elongation', 'seg_elongation', 'Elongation'),
        ('Seg Rectangularity', 'seg_rectangularity', 'Rectangularity'),
        ('Seg Solidity', 'seg_solidity', 'Solidity'),
        ('Seg Feret max', 'seg_feret_max', 'Feret_max'),
        ('Seg Feret min', 'seg_feret_min', 'Feret_min'),
        ('Seg Major axis', 'seg_major', 'Major (pxl)'),
        ('Seg Minor axis', 'seg_minor', 'Minor (pxl)'),
        ('Seg Eccentricity', 'seg_eccentricity', 'Eccentricity'),
        ('Seg Biovolume', 'seg_biovolume', 'Biovolume (pxl3)'),
        ('Seg Mean Intensity', 'seg_mean_intensity', 'Mean Intensity'),
        ('Seg Median Intensity', 'seg_median_intensity', 'Median Intensity'),
        ('Seg Std Intensity', 'seg_std_intensity', 'Std Intensity'),
        ('Seg Percent on border', 'seg_border', 'Percent_on_border'),
        ('Seg Number holes', 'seg_holes', 'Number Holes'),
    ]
    return base + [(label, key) for label, key, col in extras if col in cols]


def _order_key_spec(key: str):
    key = key or ''
    # Backwards compatibility with keys produced by older UI versions.
    if key == 'az': return ('manual_label', True, 'text')
    if key == 'confs1_desc': return ('confs1', False, 'num')
    if key == 'var_desc': return ('var_norm', False, 'num')
    aliases = {
        'names': 'names', 'manual_label': 'manual_label', 'confs1': 'confs1', 'var': 'var_norm', 'var_norm': 'var_norm',
        'review_priority': 'review_priority', 'knn_agreement': 'knn_agreement',
        'hdbscan_outlier': 'hdbscan_outlier', 'hdbscan_probability': 'hdbscan_probability',
        'seg_area': 'Area (pxl)', 'seg_perimeter': 'Perimeter (pxl)', 'seg_esd': 'ESD (pxl)',
        'seg_contrast': 'Contrast', 'seg_circularity': 'circularity',
        'seg_elongation': 'Elongation', 'seg_rectangularity': 'Rectangularity', 'seg_solidity': 'Solidity',
        'seg_feret_max': 'Feret_max', 'seg_feret_min': 'Feret_min', 'seg_major': 'Major (pxl)',
        'seg_minor': 'Minor (pxl)', 'seg_eccentricity': 'Eccentricity', 'seg_biovolume': 'Biovolume (pxl3)',
        'seg_mean_intensity': 'Mean Intensity', 'seg_median_intensity': 'Median Intensity',
        'seg_std_intensity': 'Std Intensity', 'seg_border': 'Percent_on_border', 'seg_holes': 'Number Holes'
    }
    if key.endswith('_desc') or key.endswith('_asc'):
        ascending = key.endswith('_asc')
        raw = re.sub(r'_(desc|asc)$', '', key)
        col = aliases.get(raw, raw)
        kind = 'text' if col in {'names', 'manual_label'} else 'num'
        return (col, ascending, kind)
    if key in aliases:
        col = aliases[key]
        return (col, True, 'text' if col in {'names', 'manual_label'} else 'num')
    return (None, True, 'text')


def sort_gallery_multi(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Stable lexicographic ordering by up to several gallery criteria."""
    if df.empty: return df
    keys = [str(k) for k in keys if k]
    if not keys: return df
    out = df.copy(); by=[]; asc=[]; tmp=[]
    seen=set()
    for i,key in enumerate(keys):
        col,ascending,kind = _order_key_spec(key)
        if not col or col not in out.columns or col in seen: continue
        seen.add(col)
        t=f'__sort_{i}'
        if kind=='text': out[t]=out[col].astype(str).str.lower()
        else: out[t]=pd.to_numeric(out[col],errors='coerce')
        by.append(t); asc.append(ascending); tmp.append(t)
    if not by: return out
    out=out.sort_values(by=by,ascending=asc,na_position='last',kind='mergesort')
    return out.drop(columns=tmp)

def sort_gallery(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return sort_gallery_multi(df, [key])

def _dir_has_images(path: Path) -> bool:
    return path.is_dir() and any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in path.iterdir())


def ensure_samples_dir(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        if path.name.lower() in {'samples', 'sample'} and path.parent.exists():
            return path.parent
        return path
    if _dir_has_images(path):
        return path
    if path.name.lower() in {'samples', 'sample'} and _dir_has_images(path.parent):
        return path.parent
    for sub in ('samples', 'Samples', 'SAMPLES', 'sample', 'Sample'):
        p = path / sub
        if _dir_has_images(p):
            return p
    dirs = [p for p in path.iterdir() if p.is_dir()]
    if len(dirs) == 1:
        return dirs[0]
    return path


class IATProject:
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.dataframes_dir = self.project_dir / 'dataframes'
        self.images_dir = self.project_dir / 'images'
        self.embeddings_dir = self.project_dir / 'embeddings'
        self.clusters_dir = self.project_dir / 'clusters'
        self.segmentation_dir = self.project_dir / 'segmentation'
        self.segmentation_cache_dir = self.project_dir / 'segmentation_cache'
        if not self.dataframes_dir.is_dir():
            raise FileNotFoundError(f'dataframes não encontrado: {self.dataframes_dir}')
        self.batch_cache: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        self._batch_cache_max = max(2, int(os.environ.get('ILT_IAT_BATCH_CACHE', '8')))
        self.dirty_batches: set[str] = set()
        self.dirty_classes: set[str] = set()
        self.undo_stack: List[dict] = []
        self._overview_cache: Optional[pd.DataFrame] = None
        self._image_path_cache: "OrderedDict[Tuple[str, str], Optional[Path]]" = OrderedDict()
        self._image_path_cache_max = max(1000, int(os.environ.get('ILT_IAT_PATH_CACHE', '50000')))
        self._image_dir_cache: Dict[str, Tuple[Path, ...]] = {}
        self._image_global_index: Optional[Dict[str, List[Path]]] = None
        self._image_index_lock = threading.RLock()
        self._image_resolve_stats = {
            'cache_hit': 0, 'direct_hit': 0, 'global_index_hit': 0,
            'missing': 0, 'global_index_builds': 0,
        }
        # Segmentation files can be large, so cache only a bounded number of
        # batch/method pairs. This avoids unbounded RAM growth while browsing.
        self._seg_table_cache: "OrderedDict[Tuple[str, str], Tuple[float, pd.DataFrame]]" = OrderedDict()
        self._outline_cache: "OrderedDict[Tuple[str, str], Tuple[float, dict]]" = OrderedDict()
        self._seg_cache_max = max(2, int(os.environ.get('ILT_IAT_SEG_CACHE', '12')))

    @property
    def name(self) -> str:
        return self.project_dir.name

    def csv_names(self) -> List[str]:
        return sorted(p.name for p in self.dataframes_dir.glob('*.csv') if p.is_file())

    def _trim_batch_cache(self, protected: Optional[str] = None) -> None:
        """Trim only clean batches and never evict the batch currently being loaded/edited.

        A relabel operation may span more batches than the configured LRU size.
        Unsaved batches are therefore pinned.  If every possible victim is dirty
        (or is the protected current batch), the cache is deliberately allowed to
        grow temporarily rather than losing an in-memory edit.
        """
        while len(self.batch_cache) > self._batch_cache_max:
            victim = None
            for key in list(self.batch_cache.keys()):
                if key == protected or key in self.dirty_batches:
                    continue
                victim = key
                break
            if victim is None:
                break
            self.batch_cache.pop(victim, None)

    def _cache_batch(self, csv_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """LRU cache for batches without ever evicting the just-loaded batch."""
        self.batch_cache[csv_name]=df
        try:self.batch_cache.move_to_end(csv_name)
        except Exception:pass
        self._trim_batch_cache(protected=csv_name)
        return df

    @staticmethod
    def _bounded_cache_put(cache, key, value, max_items: int):
        cache[key]=value
        try:cache.move_to_end(key)
        except Exception:pass
        while len(cache)>max_items:
            try:cache.popitem(last=False)
            except TypeError:
                cache.pop(next(iter(cache)))
            except Exception:
                break
    def load_batch(self, csv_name: str, segmentation_method: Optional[str] = None, force: bool = False) -> pd.DataFrame:
        if not force and csv_name in self.batch_cache:
            df=self.batch_cache[csv_name]
            try:self.batch_cache.move_to_end(csv_name)
            except Exception:pass
        else:
            path=self.dataframes_dir/csv_name
            if not path.exists():
                raise FileNotFoundError(path)
            df=ensure_required_columns(pd.read_csv(path,low_memory=False),csv_name,copy=False)
            self._cache_batch(csv_name,df)
        if segmentation_method:
            return self.attach_segmentation(df,csv_name,segmentation_method)
        return df
    def load_overview(self, segmentation_method: Optional[str] = None, force: bool = False,
                      progress: Optional[Callable[[int, int, str], None]] = None) -> pd.DataFrame:
        if self._overview_cache is not None and not force and not segmentation_method:
            return self._overview_cache

        # Do not call load_batch() for every file here: that used to retain every
        # batch *and* the concatenated Overview simultaneously. Read untouched
        # batches directly from disk, while reusing dirty/cached batches so live
        # relabeling is preserved.
        parts=[]
        names=self.csv_names()
        for i,bn in enumerate(names,1):
            if bn in self.batch_cache:
                df=self.batch_cache[bn]
                try:self.batch_cache.move_to_end(bn)
                except Exception:pass
            else:
                path=self.dataframes_dir/bn
                df=ensure_required_columns(pd.read_csv(path,low_memory=False),bn,copy=False)
            if segmentation_method:
                df=self.attach_segmentation(df,bn,segmentation_method)
            parts.append(df)
            if progress:
                progress(i,len(names),bn)

        out=pd.concat(parts,ignore_index=True,sort=False,copy=False) if parts else pd.DataFrame()
        if not segmentation_method:
            self._overview_cache=out
        return out
    def refresh_overview_labels(self, gids: Sequence[str], labels: Sequence[str], colors: Optional[Sequence[int]] = None) -> None:
        if self._overview_cache is None or self._overview_cache.empty:
            return
        pos = {str(g): i for i, g in enumerate(gids)}
        mask = self._overview_cache['gid'].astype(str).isin(pos)
        if not mask.any():
            return
        idxs = self._overview_cache.index[mask]
        for idx in idxs:
            p = pos[str(self._overview_cache.at[idx, 'gid'])]
            self._overview_cache.at[idx, 'manual_label'] = str(labels[p])
            if colors is not None:
                self._overview_cache.at[idx, 'colors'] = int(colors[p])

    def apply_labels(self, gids: Sequence[str], new_label: str) -> int:
        gids = [str(g) for g in gids if str(g)]
        if not gids:
            return 0
        by_batch: Dict[str, List[str]] = {}
        for gid in gids:
            bn, cid = split_gid(gid)
            if bn:
                by_batch.setdefault(bn, []).append(cid)
        mods = {}
        changed = 0
        new_colors = []
        new_labels = []
        changed_gids = []
        for bn, cids in by_batch.items():
            df = self.load_batch(bn)
            mask = df['custom_data'].astype(str).isin(set(cids))
            if not mask.any():
                continue
            rows = df.index[mask]
            old_label = df.loc[rows, 'manual_label'].astype(str).tolist()
            old_color = df.loc[rows, 'colors'].astype(int).tolist()
            old_gids = df.loc[rows, 'gid'].astype(str).tolist()
            # deterministic color id: preserve existing map if label exists, else next id
            existing = df.loc[df['manual_label'].astype(str) == new_label, 'colors']
            new_color = int(existing.iloc[0]) if len(existing) else int(pd.to_numeric(df['colors'], errors='coerce').max() + 1 if len(df) else 0)
            df.loc[rows, 'manual_label'] = new_label
            df.loc[rows, 'colors'] = new_color
            mods[bn] = {'custom_data': df.loc[rows, 'custom_data'].astype(str).tolist(), 'old_label': old_label,
                        'old_color': old_color, 'gid': old_gids}
            # Pin the exact edited dataframe in the cache before declaring it
            # dirty.  This is a second line of defence for very large Overview
            # relabel operations spanning many CSV batches.
            self.batch_cache[bn] = df
            try:self.batch_cache.move_to_end(bn)
            except Exception:pass
            self.dirty_batches.add(bn)
            self.dirty_classes.update([x for x in old_label if x])
            self.dirty_classes.add(new_label)
            changed += len(rows)
            new_labels.extend([new_label] * len(rows)); new_colors.extend([new_color] * len(rows)); changed_gids.extend(old_gids)
        if mods:
            self.undo_stack.append({'mods': mods})
            self.undo_stack = self.undo_stack[-20:]
            self.refresh_overview_labels(changed_gids, new_labels, new_colors)
        return changed

    def undo(self) -> int:
        if not self.undo_stack:
            return 0
        snap = self.undo_stack.pop()
        changed = 0
        gids_all, labels_all, colors_all = [], [], []
        for bn, m in snap.get('mods', {}).items():
            df = self.load_batch(bn)
            pos = {str(cid): i for i, cid in enumerate(m['custom_data'])}
            mask = df['custom_data'].astype(str).isin(pos)
            for idx in df.index[mask]:
                p = pos[str(df.at[idx, 'custom_data'])]
                current = str(df.at[idx, 'manual_label'])
                old_label = str(m['old_label'][p]); old_color = int(m['old_color'][p])
                df.at[idx, 'manual_label'] = old_label; df.at[idx, 'colors'] = old_color
                gid = str(df.at[idx, 'gid'])
                gids_all.append(gid); labels_all.append(old_label); colors_all.append(old_color)
                self.dirty_classes.update([current, old_label]); changed += 1
            self.dirty_batches.add(bn)
        self.refresh_overview_labels(gids_all, labels_all, colors_all)
        return changed

    def save_batch(self, csv_name: str, out_name: Optional[str] = None) -> Path:
        if csv_name not in self.batch_cache:
            self.load_batch(csv_name)
        out_name = out_name or csv_name
        if not out_name.lower().endswith('.csv'):
            out_name += '.csv'
        path = save_with_backup(self.batch_cache[csv_name], self.dataframes_dir / out_name)
        self.dirty_batches.discard(csv_name)
        if out_name != csv_name:
            self.batch_cache[out_name] = self.batch_cache.pop(csv_name)
        self._trim_batch_cache(protected=out_name)
        # Keep the live Overview dataframe attached when overwriting the same
        # batch. apply_labels()/undo() already mirror label/color changes into
        # _overview_cache before the save. Invalidating it here detached the
        # workspace's current_df after the first autosave, so later relabels were
        # written correctly to disk but the visible Overview stayed stale and
        # could make the original class appear to be "reconstructed".
        #
        # Saving under a different filename really changes Overview membership,
        # so that case still requires a rebuild.
        if out_name != csv_name:
            self._overview_cache = None
        return path

    def save_dirty(self, progress: Optional[Callable[[int, int, str], None]] = None) -> List[Path]:
        names = sorted(self.dirty_batches)
        out = []
        for i, bn in enumerate(names, 1):
            if bn not in self.batch_cache:
                # This state should not occur with the protected LRU logic above.
                # For compatibility with a session that may already have reached
                # the old failure state, recover the authoritative manual labels
                # from the live Overview instead of silently skipping the save.
                if self._overview_cache is None or self._overview_cache.empty:
                    raise RuntimeError(
                        f'Batch sujo ausente do cache: {bn}. '
                        'O salvamento foi interrompido para não perder relabels.'
                    )
                ov = self._overview_cache
                if 'csv_basename' not in ov.columns:
                    raise RuntimeError(
                        f'Não foi possível recuperar {bn}: Overview sem csv_basename.'
                    )
                live = ov.loc[ov['csv_basename'].astype(str) == str(bn)].copy()
                if live.empty:
                    raise RuntimeError(
                        f'Não foi possível recuperar {bn}: linhas não encontradas no Overview.'
                    )
                disk = self.load_batch(bn, force=True)
                if 'gid' not in disk.columns or 'gid' not in live.columns:
                    raise RuntimeError(
                        f'Não foi possível recuperar {bn}: gid indisponível.'
                    )
                live = live.drop_duplicates('gid').set_index(live['gid'].astype(str))
                gids = disk['gid'].astype(str)
                mask = gids.isin(live.index)
                if not mask.any():
                    raise RuntimeError(
                        f'Não foi possível recuperar {bn}: gids não coincidem.'
                    )
                for col in ('manual_label', 'colors'):
                    if col in live.columns and col in disk.columns:
                        mapped = gids.loc[mask].map(live[col])
                        disk.loc[mask, col] = mapped.to_numpy()
                self.batch_cache[bn] = disk
                try:self.batch_cache.move_to_end(bn)
                except Exception:pass
            out.append(self.save_batch(bn))
            if progress:
                progress(i, len(names), bn)
        return out

    def _candidate_image_dirs(self, csv_name: str) -> List[Path]:
        key = str(csv_name)
        with self._image_index_lock:
            cached = self._image_dir_cache.get(key)
        if cached is not None:
            return list(cached)

        stem = Path(csv_name).stem
        batch = infer_batch_folder(csv_name)
        cands = []
        if batch:
            cands += [self.images_dir / batch / 'samples', self.images_dir / batch]
        cands += [self.images_dir / stem / 'samples', self.images_dir / stem, self.images_dir]
        out = []
        for c in cands:
            c = ensure_samples_dir(c)
            if c.is_dir() and c not in out:
                out.append(c)
        with self._image_index_lock:
            self._image_dir_cache[key] = tuple(out)
        return out

    def reset_image_resolve_stats(self) -> None:
        with self._image_index_lock:
            for key in self._image_resolve_stats:
                self._image_resolve_stats[key] = 0

    def image_resolve_stats(self) -> Dict[str, int]:
        with self._image_index_lock:
            return dict(self._image_resolve_stats)

    def _build_global_image_index(self) -> Dict[str, List[Path]]:
        """Index project images once for unusual layouts.

        Older code executed ``rglob(image_name)`` for every unresolved image.
        With large projects that turns a handful of path mismatches into tens of
        thousands of full-tree scans.  This index makes the expensive fallback a
        one-time operation while retaining support for legacy layouts.
        """
        with self._image_index_lock:
            if self._image_global_index is not None:
                return self._image_global_index

            index: Dict[str, List[Path]] = {}
            if self.images_dir.is_dir():
                for dirpath, dirnames, filenames in os.walk(self.images_dir, followlinks=False):
                    base = Path(dirpath)
                    for filename in filenames:
                        if Path(filename).suffix.lower() not in IMAGE_EXTS:
                            continue
                        index.setdefault(filename, []).append(base / filename)
            self._image_global_index = index
            self._image_resolve_stats['global_index_builds'] += 1
            return index

    def _choose_global_image_match(self, csv_name: str, image_name: str, matches: Sequence[Path]) -> Optional[Path]:
        if not matches:
            return None
        batch = infer_batch_folder(csv_name)
        stem = Path(csv_name).stem
        preferred = [x for x in (batch, stem) if x]
        for marker in preferred:
            marker = str(marker)
            for path in matches:
                if marker in path.parts:
                    return path
        return matches[0]

    def resolve_image(self, csv_name: str, image_name: str) -> Optional[Path]:
        key = (str(csv_name), str(image_name))
        with self._image_index_lock:
            if key in self._image_path_cache:
                val = self._image_path_cache[key]
                try:
                    self._image_path_cache.move_to_end(key)
                except Exception:
                    pass
                self._image_resolve_stats['cache_hit'] += 1
                return val

        found = None
        for directory in self._candidate_image_dirs(csv_name):
            candidate = directory / str(image_name)
            if candidate.is_file():
                found = candidate
                with self._image_index_lock:
                    self._image_resolve_stats['direct_hit'] += 1
                break

        if found is None and self.images_dir.is_dir():
            index = self._build_global_image_index()
            found = self._choose_global_image_match(csv_name, str(image_name), index.get(str(image_name), ()))
            with self._image_index_lock:
                if found is not None:
                    self._image_resolve_stats['global_index_hit'] += 1
                else:
                    self._image_resolve_stats['missing'] += 1

        with self._image_index_lock:
            self._bounded_cache_put(self._image_path_cache, key, found, self._image_path_cache_max)
        return found
    def export_by_label(self, out_dir: Path, progress: Optional[Callable[[int, int, str], None]] = None) -> int:
        out_dir = Path(out_dir).expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True)
        csvs = self.csv_names()
        total = 0
        tables = []
        for bn in csvs:
            df = self.load_batch(bn)
            sub = df[['names', 'manual_label']].copy()
            sub = sub[sub['names'].notna() & sub['manual_label'].notna() & (sub['manual_label'].astype(str) != '')]
            tables.append((bn, sub)); total += len(sub)
        done = copied = 0
        for bn, sub in tables:
            for row in sub.itertuples(index=False):
                done += 1
                src = self.resolve_image(bn, str(row.names))
                if src and src.exists():
                    dest = out_dir / str(row.manual_label)
                    dest.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(src, dest / src.name); copied += 1
                    except Exception:
                        pass
                if progress and (done % 10 == 0 or done == total):
                    progress(done, total, str(row.manual_label))
        return copied

    # ---- segmentation ----
    def segmentation_csv_path(self, csv_name: str, method: str) -> Path:
        return self.segmentation_dir / f'{Path(csv_name).stem}__{method}.csv'

    def segmentation_cache_path(self, csv_name: str, method: str) -> Path:
        return self.segmentation_cache_dir / f'{Path(csv_name).stem}__{method}.json'

    def get_segmentation_table(self, csv_name: str, method: str) -> pd.DataFrame:
        p = self.segmentation_csv_path(csv_name, method)
        if not p.exists():
            return pd.DataFrame()
        key = (str(csv_name), str(method))
        try:
            mtime = p.stat().st_mtime
        except OSError:
            return pd.DataFrame()
        cached = self._seg_table_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            seg = pd.read_csv(p, low_memory=False)
        except Exception:
            return pd.DataFrame()
        self._bounded_cache_put(self._seg_table_cache,key,(mtime,seg),self._seg_cache_max)
        return seg

    def available_segmentation_methods(self) -> List[str]:
        """Return segmentation method keys already present on disk.

        This lets the Gallery use existing segmentation artifacts immediately,
        without requiring the user to visit the Segmentation tab first.
        """
        methods = set()
        if self.segmentation_dir.is_dir():
            for p in self.segmentation_dir.glob('*.csv'):
                stem = p.stem
                if '__' in stem:
                    methods.add(stem.rsplit('__', 1)[1])
        if self.segmentation_cache_dir.is_dir():
            for p in self.segmentation_cache_dir.glob('*.json'):
                stem = p.stem
                if '__' in stem:
                    methods.add(stem.rsplit('__', 1)[1])
        return sorted(methods)

    def available_segmentation_columns(self, method: str, csv_names: Optional[Sequence[str]] = None) -> List[str]:
        """Return union of already-generated segmentation columns without joining them."""
        names = list(csv_names or self.csv_names())
        cols = set()
        for bn in names:
            p = self.segmentation_csv_path(str(bn), method)
            if not p.exists():
                continue
            try:
                cached = self._seg_table_cache.get((str(bn), str(method)))
                if cached and cached[0] == p.stat().st_mtime:
                    cols.update(cached[1].columns)
                else:
                    cols.update(pd.read_csv(p, nrows=0).columns)
            except Exception:
                pass
        cols.discard('Image Name')
        return sorted(cols)

    def attach_segmentation(self, df: pd.DataFrame, csv_name: str, method: str) -> pd.DataFrame:
        seg = self.get_segmentation_table(csv_name, method)
        if seg.empty or 'Image Name' not in seg.columns:
            return df.copy()
        cols = [c for c in seg.columns if c != 'Image Name']
        return df.merge(seg[['Image Name'] + cols], how='left', left_on='names', right_on='Image Name')

    def load_outline_cache(self, csv_name: str, method: str) -> dict:
        p = self.segmentation_cache_path(csv_name, method)
        if not p.exists():
            return {}
        key = (str(csv_name), str(method))
        try:
            mtime = p.stat().st_mtime
        except OSError:
            return {}
        cached = self._outline_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            data = json.loads(p.read_text())
        except Exception:
            data = {}
        self._bounded_cache_put(self._outline_cache,key,(mtime,data),self._seg_cache_max)
        return data

    def invalidate_segmentation_cache(self, csv_name: Optional[str] = None, method: Optional[str] = None) -> None:
        def keep(key):
            bn, mt = key
            return ((csv_name is not None and bn != str(csv_name)) or
                    (method is not None and mt != str(method)))
        if csv_name is None and method is None:
            self._seg_table_cache.clear(); self._outline_cache.clear(); return
        self._seg_table_cache = OrderedDict((k,v) for k,v in self._seg_table_cache.items() if keep(k))
        self._outline_cache = OrderedDict((k,v) for k,v in self._outline_cache.items() if keep(k))


def filter_dataframe(df: pd.DataFrame, conf_range: Tuple[float, float] = (0, 1), var_range: Tuple[float, float] = (0, 1),
                     hide_relabel: bool = False, classes: Optional[Sequence[str]] = None) -> pd.DataFrame:
    out = df
    if 'confs1' in out.columns and out['confs1'].notna().any():
        v = pd.to_numeric(out['confs1'], errors='coerce')
        out = out[(v.isna()) | ((v >= conf_range[0]) & (v <= conf_range[1]))]
    if 'var_norm' in out.columns and out['var_norm'].notna().any():
        v = pd.to_numeric(out['var_norm'], errors='coerce')
        out = out[(v.isna()) | ((v >= var_range[0]) & (v <= var_range[1]))]
    if hide_relabel and {'manual_label', 'correct_label'}.issubset(out.columns):
        out = out[out['manual_label'].astype(str) == out['correct_label'].astype(str)]
    if classes:
        out = out[out['manual_label'].astype(str).isin(set(map(str, classes)))]
    return out


# ---------------- Clustering helpers ----------------
def load_feats(npz_path: Path) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=False)
    for key in ('feats', 'embeddings', 'vectors', 'X'):
        if key in z and np.asarray(z[key]).ndim == 2:
            return np.asarray(z[key])
    for key in z.files:
        a = np.asarray(z[key])
        if a.ndim == 2:
            return a
    raise KeyError(f'{npz_path}: não achei array 2D de embeddings')


def resolve_embed(project: IATProject, value: str) -> Path:
    p = Path(str(value))
    if p.is_absolute() and p.exists(): return p
    cand = project.project_dir / p
    if cand.exists(): return cand
    cand = project.embeddings_dir / p.name
    if cand.exists(): return cand
    raise FileNotFoundError(f'embed_file não encontrado: {value}')


def gather_embeddings(project: IATProject, rows: pd.DataFrame,
                      progress: Optional[Callable[[int, int, str], None]] = None,
                      id_col: str = 'names') -> Tuple[np.ndarray, np.ndarray]:
    """Gather only selected embedding rows without retaining every batch in RAM."""
    need = {'names', 'embed_file', 'embed_idx'}
    if not need.issubset(rows.columns):
        raise ValueError(f'CSV não tem colunas necessárias: {sorted(need - set(rows.columns))}')
    rows = rows.copy()
    rows['embed_idx'] = pd.to_numeric(rows['embed_idx'], errors='coerce').fillna(-1).astype(int)
    rows = rows[rows['embed_idx'] >= 0]
    if id_col not in rows.columns:
        id_col = 'names'
    names = rows[id_col].astype(str).to_numpy(object)
    files = rows['embed_file'].astype(str).to_numpy(object)
    idxs = rows['embed_idx'].to_numpy(int)
    positions: Dict[str, List[int]] = {}
    for pos, ef in enumerate(files):
        positions.setdefault(str(ef), []).append(pos)
    if not positions:
        return np.empty((0, 0), np.float32), np.array([], object)

    first_value=next(iter(positions))
    first_path=resolve_embed(project,first_value)
    first_feats=load_feats(first_path)
    if first_feats.ndim != 2:
        raise ValueError(f'Embedding inválido em {first_path.name}: esperado array 2D, recebido shape {first_feats.shape}.')
    D=first_feats.shape[1]
    X=np.empty((len(names),D),dtype=np.float32)
    valid=np.ones(len(names),bool)

    for j,(ef,poslist) in enumerate(positions.items(),1):
        ep=resolve_embed(project,ef)
        feats=first_feats if str(ep)==str(first_path) else load_feats(ep)
        if feats.ndim != 2:
            raise ValueError(f'Embedding inválido em {ep.name}: esperado array 2D, recebido shape {feats.shape}.')
        if feats.shape[1] != D:
            raise ValueError(
                f'Dimensão de embedding incompatível em {ep.name}: {feats.shape[1]} != {D} ({first_path.name}).'
            )
        for pos in poslist:
            idx=int(idxs[pos])
            if idx<0 or idx>=feats.shape[0]:
                valid[pos]=False
            else:
                X[pos]=feats[idx].astype(np.float32,copy=False)
        if progress:progress(j,len(positions),ep.name)
        # Do not cache the complete compressed batch set. X already stores only
        # the selected rows; releasing each source batch keeps peak RAM bounded.
        if feats is not first_feats:
            del feats
    return X[valid],names[valid]


def _l2(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True); n = np.maximum(n, eps); return X / n


def validation_capabilities() -> Dict[str, bool]:
    """Report optional analysis capabilities, importing them only when requested."""
    _load_sklearn()
    _load_umap()
    _load_pacmap()
    if SklearnHDBSCAN is None:
        _load_hdbscan_external()
    return {
        'pca': PCA is not None,
        'tsne': TSNE is not None,
        'umap': umap_module is not None,
        'pacmap': pacmap_module is not None,
        'hdbscan': SklearnHDBSCAN is not None or hdbscan_external is not None,
        'knn': NearestNeighbors is not None,
    }


def _safe_numeric(values, default=np.nan) -> np.ndarray:
    s = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    if default is not np.nan:
        s[~np.isfinite(s)] = float(default)
    return s


def _torch_cuda():
    """``torch`` with a usable CUDA device, else ``None`` (imported lazily so
    opening a project never loads PyTorch)."""
    if str(os.environ.get('ILT_IAT_GPU', '1')).strip().lower() in {'0', 'false', 'no', 'off'}:
        return None
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


# Below this many rows scikit-learn on the CPU is instantaneous; the GPU path
# only pays off for larger selections (CUDA context + transfers).
GPU_MIN_ROWS = 3000


def _gpu_pca(X: np.ndarray, ncomp: int):
    """Centred randomized PCA on the GPU (``torch.pca_lowrank``); ``None`` when
    CUDA is unavailable or the input is small.  Same model as scikit-learn's
    randomized solver (Halko, Martinsson & Tropp 2011, SIAM Rev. 53:217-288),
    so only signs/orientation of components may differ."""
    torch = _torch_cuda()
    if torch is None or len(X) < GPU_MIN_ROWS:
        return None
    try:
        with torch.no_grad():
            torch.manual_seed(0)
            xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).cuda()
            xc = xt - xt.mean(dim=0, keepdim=True)
            q = int(min(ncomp + 10, min(xc.shape)))
            _u, _s, v = torch.pca_lowrank(xc, q=q, center=False, niter=4)
            out = (xc @ v[:, :ncomp]).cpu().numpy().astype(np.float32)
        del xt, xc, v
        torch.cuda.empty_cache()
        return out
    except Exception:
        return None


def _gpu_knn_indices(X: np.ndarray, k_plus: int):
    """Indices of the ``k_plus`` nearest rows (Euclidean, self included) for
    every row, computed on the GPU in memory-bounded blocks with the
    ‖a−b‖² = ‖a‖² − 2a·b + ‖b‖² expansion and ``topk``.  ``None`` when CUDA is
    unavailable or the input is small."""
    torch = _torch_cuda()
    n = len(X)
    if torch is None or n < GPU_MIN_ROWS:
        return None
    try:
        with torch.no_grad():
            xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).cuda()
            sq = (xt * xt).sum(dim=1)
            free, _total = torch.cuda.mem_get_info()
            budget = max(64 << 20, int(free * 0.35))
            chunk = int(max(128, min(n, budget // (n * 4 * 3))))
            out = np.empty((n, k_plus), dtype=np.int64)
            for start in range(0, n, chunk):
                q = xt[start:start + chunk]
                d = sq[start:start + chunk, None] - 2.0 * (q @ xt.T) + sq[None, :]
                out[start:start + chunk] = torch.topk(d, k_plus, dim=1, largest=False).indices.cpu().numpy()
                del q, d
        del xt, sq
        torch.cuda.empty_cache()
        return out
    except Exception:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None


def _pca_for_validation(X: np.ndarray, requested_dim: int = 50) -> np.ndarray:
    """Reduce embeddings for statistical validation while preserving >2D structure."""
    _load_sklearn()
    X = np.asarray(X, dtype=np.float32)
    if len(X) < 2:
        return X
    if PCA is None:
        return X
    max_comp = min(X.shape[1], max(1, X.shape[0] - 1))
    ncomp = min(max(2, int(requested_dim)), max_comp)
    if ncomp >= X.shape[1]:
        return X
    gpu = _gpu_pca(X, ncomp)
    if gpu is not None:
        return gpu
    solver = 'randomized' if ncomp < min(X.shape) else 'auto'
    return PCA(n_components=ncomp, svd_solver=solver, random_state=0).fit_transform(X).astype(np.float32)


def _fit_hdbscan(X: np.ndarray, min_cluster_size: int = 15):
    """Fit HDBSCAN using sklearn when available, otherwise the hdbscan package."""
    _load_sklearn()
    if SklearnHDBSCAN is None:
        _load_hdbscan_external()
    n = len(X)
    if n < 3:
        return np.full(n, -1, dtype=int), np.zeros(n, dtype=float)
    mcs = max(2, min(int(min_cluster_size), n))
    model = None
    if SklearnHDBSCAN is not None:
        try:
            model = SklearnHDBSCAN(
                min_cluster_size=mcs,
                metric='euclidean',
                cluster_selection_method='eom',
                allow_single_cluster=False,
                copy=True,
            )
        except TypeError:
            model = SklearnHDBSCAN(
                min_cluster_size=mcs,
                metric='euclidean',
                cluster_selection_method='eom',
                allow_single_cluster=False,
            )
    elif hdbscan_external is not None:
        model = hdbscan_external.HDBSCAN(
            min_cluster_size=mcs,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=False,
        )
    else:
        raise RuntimeError('HDBSCAN indisponível. Instale scikit-learn recente ou o pacote hdbscan.')

    labels = np.asarray(model.fit_predict(X), dtype=int)
    probs = getattr(model, 'probabilities_', None)
    if probs is None:
        probs = np.where(labels >= 0, 1.0, 0.0)
    probs = np.asarray(probs, dtype=float)
    probs[~np.isfinite(probs)] = 0.0
    probs = np.clip(probs, 0.0, 1.0)
    return labels, probs


def _knn_label_agreement(X: np.ndarray, labels: Sequence[str], k: int = 15):
    _load_sklearn()
    n = len(X)
    if n < 2 or NearestNeighbors is None:
        return np.full(n, np.nan), np.array([''] * n, dtype=object)
    kk = max(1, min(int(k), n - 1))
    idx = _gpu_knn_indices(np.asarray(X, dtype=np.float32), min(kk + 1, n))
    if idx is None:
        nn = NearestNeighbors(n_neighbors=min(kk + 1, n), metric='euclidean', n_jobs=cpu_jobs())
        nn.fit(X)
        idx = nn.kneighbors(X, n_neighbors=min(kk + 1, n), return_distance=False)
    labs = np.asarray([str(x) for x in labels], dtype=object)
    # Vectorised equivalent of the former per-point loop: drop the point
    # itself, keep the first kk neighbours, then count labels per row.  Label
    # codes follow np.unique's sorted order, so ties resolve to the same
    # (lexicographically smallest) label as before.
    vals, codes = np.unique(labs, return_inverse=True)
    codes = codes.reshape(-1).astype(np.int64, copy=False)
    n_labels = len(vals)
    not_self = idx != np.arange(n)[:, None]
    order = np.argsort(~not_self, axis=1, kind='stable')[:, :kk]
    neigh = np.take_along_axis(idx, order, axis=1)
    neigh_codes = codes[neigh]
    agreement = (neigh_codes == codes[:, None]).mean(axis=1).astype(float)
    dominant_codes = np.empty(n, dtype=np.int64)
    chunk = max(1, 4_000_000 // max(1, n_labels))
    for start in range(0, n, chunk):
        part = neigh_codes[start:start + chunk]
        rows = np.repeat(np.arange(len(part)), part.shape[1])
        counts = np.zeros((len(part), n_labels), dtype=np.int32)
        np.add.at(counts, (rows, part.reshape(-1)), 1)
        dominant_codes[start:start + chunk] = counts.argmax(axis=1)
    dominant = np.asarray([str(v) for v in vals[dominant_codes]], dtype=object)
    return agreement, dominant


def _cluster_label_diagnostics(cluster_labels: np.ndarray, point_labels: Sequence[str]):
    labs = np.asarray([str(x) for x in point_labels], dtype=object)
    dominant = np.array([''] * len(labs), dtype=object)
    agreement = np.full(len(labs), np.nan, dtype=float)
    mismatch = np.zeros(len(labs), dtype=bool)
    for cid in sorted(set(np.asarray(cluster_labels, dtype=int).tolist())):
        if cid < 0:
            continue
        idx = np.flatnonzero(cluster_labels == cid)
        if len(idx) == 0:
            continue
        vals, counts = np.unique(labs[idx], return_counts=True)
        best = int(np.argmax(counts))
        dom = str(vals[best])
        frac = float(counts[best] / len(idx))
        dominant[idx] = dom
        agreement[idx] = frac
        # Only flag a cluster-label discrepancy when the cluster has a clear
        # majority. This avoids treating naturally multimodal classes as errors.
        if frac >= 0.70:
            mismatch[idx] = labs[idx] != dom
    return dominant, agreement, mismatch


# Selections at least this large run UMAP/PaCMAP on all cores (no fixed seed).
PARALLEL_PROJECTION_MIN_N = 20000


def _project_validation_space(X: np.ndarray, method: str, labels: Optional[Sequence[str]] = None) -> np.ndarray:
    method = str(method or 'pca').lower()
    n = len(X)
    if n == 0:
        return np.empty((0, 2), dtype=float)
    if n == 1:
        return np.array([[0.0, 0.0]], dtype=float)

    if method == 'pca':
        _load_sklearn()
        if PCA is None:
            raise RuntimeError('PCA requer scikit-learn.')
        ncomp = min(2, X.shape[1], n)
        y = PCA(n_components=ncomp, random_state=0).fit_transform(X)
        if y.shape[1] == 1:
            y = np.column_stack([y[:, 0], np.zeros(n)])
        return np.asarray(y[:, :2], dtype=float)

    if method == 'tsne':
        _load_sklearn()
        if TSNE is None:
            raise RuntimeError('t-SNE requer scikit-learn.')
        perplexity = safe_perplexity(n)
        return np.asarray(TSNE(
            n_components=2, perplexity=perplexity, learning_rate='auto',
            init='pca', random_state=0, metric='euclidean', n_jobs=cpu_jobs(),
        ).fit_transform(X), dtype=float)

    if method in {'umap', 'umap_supervised'}:
        _load_umap()
        if umap_module is None:
            raise RuntimeError('UMAP indisponível. Instale umap-learn.')
        nn = max(2, min(30, n - 1))
        # umap-learn runs single-threaded whenever random_state is set.  Large
        # selections use every core (results then vary slightly between runs);
        # small ones keep the deterministic seed.  ILT_UMAP_DETERMINISTIC=1
        # forces the seed for any size.
        parallel = n >= PARALLEL_PROJECTION_MIN_N and str(
            os.environ.get('ILT_UMAP_DETERMINISTIC', '0')).strip().lower() not in {'1', 'true', 'yes'}
        seed_kw = {'n_jobs': -1} if parallel else {'random_state': 0}
        reducer = umap_module.UMAP(
            n_components=2, n_neighbors=nn, min_dist=0.10,
            metric='euclidean', **seed_kw,
            # modest supervision: enough to expose class structure without
            # forcing every possibly-wrong label to become geometric truth.
            target_weight=0.35 if method == 'umap_supervised' else 0.5,
        )
        if method == 'umap_supervised':
            if labels is None:
                raise ValueError('UMAP supervisionado precisa de rótulos.')
            y = pd.Categorical([str(x) for x in labels]).codes
            return np.asarray(reducer.fit_transform(X, y=y), dtype=float)
        return np.asarray(reducer.fit_transform(X), dtype=float)

    if method == 'pacmap':
        _load_pacmap()
        if pacmap_module is None:
            raise RuntimeError('PaCMAP indisponível. Instale pacmap.')
        pac_kw = {'random_state': 0} if n < PARALLEL_PROJECTION_MIN_N else {}
        try:
            model = pacmap_module.PaCMAP(
                n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, **pac_kw,
            )
        except TypeError:
            model = pacmap_module.PaCMAP(
                n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0,
            )
        try:
            y = model.fit_transform(X, init='pca')
        except TypeError:
            y = model.fit_transform(X)
        return np.asarray(y, dtype=float)

    raise ValueError(f'Método de projeção desconhecido: {method}')


def _audit_mode_active(df: pd.DataFrame) -> bool:
    if df is None or 'audit_source_labels' not in df.columns:
        return False
    s=df['audit_source_labels']
    if getattr(s,'dtype',None) == bool:
        return bool(s.any())
    vals=s.fillna('').astype(str).str.strip().str.lower()
    return bool(vals.isin({'1','true','t','yes','y'}).any())


def analyze_embedding_space(
    project: IATProject,
    rows: pd.DataFrame,
    projection: str = 'umap',
    pca_dim: int = 50,
    knn_k: int = 15,
    min_cluster_size: int = 15,
    label_col: str = 'manual_label',
    supervised_label_col: str = 'manual_label',
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Compute projection + independent validation evidence from neural embeddings.

    The routine never changes labels. It returns one row per valid embedding with
    diagnostics that the IAT can use to prioritize human review.
    """
    if rows is None or rows.empty:
        raise ValueError('Nenhuma imagem para analisar.')
    rows = rows.copy()
    if 'gid' not in rows.columns:
        raise ValueError("A análise requer a coluna 'gid'.")
    rows = rows.drop_duplicates('gid', keep='first')
    if len(rows) < 2:
        raise ValueError('Selecione pelo menos duas imagens.')

    if progress:
        progress(0, 6, 'Embeddings')
    X, gids = gather_embeddings(project, rows, progress=None, id_col='gid')
    if len(X) < 2:
        raise ValueError('Menos de dois embeddings válidos.')
    X = _l2(np.asarray(X, dtype=np.float32))

    row_index = rows.set_index(rows['gid'].astype(str), drop=False)
    valid_rows = row_index.loc[[str(g) for g in gids]].copy().reset_index(drop=True)
    if progress:
        progress(1, 6, f'{len(valid_rows)} embeddings')

    Xstat = _pca_for_validation(X, pca_dim)
    if progress:
        progress(2, 6, f'PCA {Xstat.shape[1]}D')

    label_name = label_col if label_col in valid_rows.columns else 'manual_label'
    labels = valid_rows[label_name].fillna('').astype(str).to_numpy(object)
    knn_agreement, knn_dominant = _knn_label_agreement(Xstat, labels, knn_k)
    if progress:
        progress(3, 6, 'kNN')

    cluster_labels, cluster_prob = _fit_hdbscan(Xstat, min_cluster_size)
    cluster_dom, cluster_agree, cluster_mismatch = _cluster_label_diagnostics(cluster_labels, labels)
    outlier = np.clip(1.0 - cluster_prob, 0.0, 1.0)
    outlier[cluster_labels < 0] = np.maximum(outlier[cluster_labels < 0], 0.90)
    if progress:
        progress(4, 6, 'HDBSCAN')

    sup_col = supervised_label_col if supervised_label_col in valid_rows.columns else label_name
    sup_labels = valid_rows[sup_col].fillna('').astype(str).to_numpy(object)
    coords = _project_validation_space(Xstat, projection, sup_labels)
    if progress:
        progress(5, 6, projection.upper())

    out = valid_rows.copy()
    out['analysis_x'] = coords[:, 0]
    out['analysis_y'] = coords[:, 1]
    out['knn_agreement'] = knn_agreement
    out['knn_disagreement'] = 1.0 - knn_agreement
    out['knn_dominant_label'] = knn_dominant
    out['hdbscan_cluster'] = cluster_labels
    out['hdbscan_probability'] = cluster_prob
    out['hdbscan_outlier'] = outlier
    out['hdbscan_dominant_label'] = cluster_dom
    out['hdbscan_label_agreement'] = cluster_agree
    out['hdbscan_label_mismatch'] = cluster_mismatch.astype(int)

    pred_col = 'preds1' if 'preds1' in out.columns else ('pred' if 'pred' in out.columns else None)
    if pred_col:
        pred = out[pred_col].fillna('').astype(str)
        out['manual_prediction_mismatch'] = (
            out['manual_label'].fillna('').astype(str) != pred
        ).astype(int)
        if 'source_label' in out.columns:
            source = out['source_label'].fillna('').astype(str)
            valid_source = source.str.len() > 0
            out['source_prediction_mismatch'] = ((source != pred) & valid_source).astype(int)
        else:
            out['source_prediction_mismatch'] = 0
    else:
        out['manual_prediction_mismatch'] = 0
        out['source_prediction_mismatch'] = 0

    conf = pd.to_numeric(out.get('confs1', np.nan), errors='coerce').to_numpy(float)
    low_conf = np.clip(1.0 - conf, 0.0, 1.0)
    var = pd.to_numeric(out.get('var_norm', np.nan), errors='coerce').to_numpy(float)
    var = np.clip(var, 0.0, 1.0)

    # Conservative default priority: structural signals + model uncertainty.
    # Cluster/label mismatch is intentionally NOT included by default because a
    # valid class may contain several morphological subgroups.
    audit_active=_audit_mode_active(out)
    pred_disagreement = (
        np.asarray(out['source_prediction_mismatch'], dtype=float)
        if audit_active and 'source_prediction_mismatch' in out.columns
        else np.asarray(out['manual_prediction_mismatch'], dtype=float)
    )
    comps = np.column_stack([
        low_conf,
        var,
        np.asarray(out['knn_disagreement'], dtype=float),
        np.asarray(out['hdbscan_outlier'], dtype=float),
        pred_disagreement,
    ])
    finite = np.isfinite(comps)
    denom = finite.sum(axis=1)
    score = np.divide(
        np.nansum(np.where(finite, comps, 0.0), axis=1),
        denom,
        out=np.zeros(len(out), dtype=float),
        where=denom > 0,
    )
    out['review_priority'] = np.clip(score, 0.0, 1.0)

    _load_sklearn()
    meta = {
        'n': int(len(out)),
        'projection': str(projection),
        'pca_dim_used': int(Xstat.shape[1]),
        'knn_k': int(max(1, min(int(knn_k), len(out) - 1))),
        'hdbscan_clusters': int(len(set(cluster_labels.tolist()) - {-1})),
        'hdbscan_noise_fraction': float(np.mean(cluster_labels < 0)),
        'trustworthiness': None,
        'silhouette_hdbscan': None,
        'audit_source_labels': bool(audit_active),
    }

    # Metrics are calculated on a deterministic subset to avoid quadratic work
    # on very large selections.
    sample_n = min(len(out), 2000)
    if sample_n >= 6 and trustworthiness is not None:
        rng = np.random.default_rng(0)
        sel = np.arange(len(out)) if sample_n == len(out) else np.sort(rng.choice(len(out), sample_n, replace=False))
        try:
            n_neighbors = max(2, min(15, (sample_n - 1) // 2))
            meta['trustworthiness'] = float(trustworthiness(Xstat[sel], coords[sel], n_neighbors=n_neighbors))
        except Exception:
            pass
        if silhouette_score is not None:
            c = cluster_labels[sel]
            keep = c >= 0
            distinct = set(c[keep].tolist())
            if keep.sum() >= 5 and 1 < len(distinct) < int(keep.sum()):
                try:
                    meta['silhouette_hdbscan'] = float(silhouette_score(Xstat[sel][keep], c[keep]))
                except Exception:
                    pass

    if progress:
        progress(6, 6, 'Concluído')
    return out, meta


def generate_cluster_artifact(project: IATProject, class_label: str, rows: Optional[pd.DataFrame] = None,
                              metric: str = 'cosine', method: str = 'average', pca_dim: int = 64,
                              max_full: int = 1500, max_centroids: int = 1500,
                              progress: Optional[Callable[[int, int, str], None]] = None,
                              selection_name: Optional[str] = None,
                              filter_class: bool = True,
                              id_col: str = 'names',
                              output_name: Optional[str] = None) -> Path:
    """Generate one hierarchical-clustering artifact using the shared engine.

    Normal class artifacts keep the historic behavior. When ``filter_class`` is
    False the rows are clustered exactly as supplied; this is used by the IAT
    selection workflow so an arbitrary multi-class scatter selection can be
    clustered as one set. ``id_col='gid'`` keeps identifiers unique across
    batches.
    """
    if rows is None:
        df = project.load_overview()
        rows = df[df['manual_label'].astype(str) == str(class_label)]
    elif filter_class:
        rows = rows[rows['manual_label'].astype(str) == str(class_label)]
    else:
        rows = rows.copy()

    if len(rows) < 2:
        raise ValueError(f'N={len(rows)} < min_samples=2')

    X, names = gather_embeddings(project, rows, progress=progress, id_col=id_col)
    if len(X) < 2:
        raise ValueError(f'N embeddings válidos={len(X)} < 2')

    result = hierarchical_linkage(
        X,
        method=method,
        metric=metric,
        pca_dim=pca_dim,
        max_full=max_full,
        max_centroids=max_centroids,
        progress=progress,
    )

    project.clusters_dir.mkdir(parents=True, exist_ok=True)
    if output_name:
        out = project.clusters_dir / output_name
    elif selection_name:
        out_dir = project.clusters_dir / 'selections'
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f'{sanitize_filename(class_label)}__{sanitize_filename(selection_name)}.npz'
    else:
        out = project.clusters_dir / f'{sanitize_filename(class_label)}.npz'

    return _shared_save_cluster_artifact(
        out,
        names,
        result,
        metric=metric,
        method=method,
        pca_dim=pca_dim,
        max_full=max_full,
    )

def generate_selection_cluster_artifact(project: IATProject, rows: pd.DataFrame,
                                        metric: str = 'cosine', method: str = 'average', pca_dim: int = 64,
                                        max_full: int = 1500, max_centroids: int = 1500,
                                        progress: Optional[Callable[[int, int, str], None]] = None) -> Path:
    """Cluster the current selected points as one (possibly multi-class) set."""
    if 'gid' not in rows.columns:
        raise ValueError('A seleção precisa conter a coluna gid.')
    return generate_cluster_artifact(
        project, '_selection', rows=rows, metric=metric, method=method, pca_dim=pca_dim,
        max_full=max_full, max_centroids=max_centroids, progress=progress,
        filter_class=False, id_col='gid', output_name='_selection.npz'
    )



def _class_artifact_matches_rows(path: Path, rows: pd.DataFrame) -> bool:
    """Best-effort membership check for historic per-class artifacts.

    Per-class artifacts store image names rather than gids. We compare the full
    multiplicity of names so a class whose membership changed after relabeling
    is detected as stale. Rows without a usable embedding are ignored because
    generate_cluster_artifact() ignores them too.
    """
    try:
        if not Path(path).exists() or rows is None:
            return False
        use = rows.copy()
        if 'embed_idx' in use.columns:
            idx = pd.to_numeric(use['embed_idx'], errors='coerce').fillna(-1).astype(int)
            use = use.loc[idx >= 0]
        expected = use['names'].astype(str).value_counts(dropna=False).sort_index()
        # Read only identifiers. The shared reader preserves historical
        # dtype=object compatibility without loading the linkage matrix.
        got_names = read_artifact_names(path)
        got = pd.Series(got_names).value_counts(dropna=False).sort_index()
        return expected.equals(got)
    except Exception:
        return False


def reconcile_class_cluster_artifacts(project: IATProject,
                                      class_labels: Optional[Sequence[str]] = None,
                                      metric: str = 'cosine', method: str = 'average',
                                      pca_dim: int = 64, max_full: int = 1500,
                                      max_centroids: int = 1500,
                                      progress: Optional[Callable[[int, int, str], None]] = None,
                                      discover_stale: bool = True) -> Dict[str, object]:
    """Reconcile per-class cluster artifacts with current ``manual_label`` values.

    This may create, replace or remove only top-level per-class clusters/*.npz.
    CSVs, embeddings, predictions and manual labels are never changed here.

    ``discover_stale=True`` also repairs a project reopened after relabeling:
    obsolete artifacts are removed and current classes with missing/mismatched
    artifacts are rebuilt even if the in-memory dirty-class registry was lost.
    """
    project.save_dirty()
    ov = project.load_overview()
    if 'manual_label' not in ov.columns:
        raise ValueError('Overview não contém manual_label.')

    labels_s = ov['manual_label'].fillna('').astype(str).str.strip()
    current_labels = sorted(x for x in labels_s.unique().tolist() if x)
    # Keep only row-index groups in memory.  Materializing one dataframe copy
    # per class is unnecessarily expensive on multi-million-row projects.
    grouped_indices = labels_s.groupby(labels_s, sort=False).groups
    def rows_for(lab: str) -> Optional[pd.DataFrame]:
        idx = grouped_indices.get(lab)
        return None if idx is None else ov.loc[idx].copy()
    safe_labels = ensure_unique_sanitized_labels(current_labels)
    expected_paths = {
        lab: project.clusters_dir / f'{safe_labels[lab]}.npz'
        for lab in current_labels
    }
    expected_resolved = {str(p.resolve()).lower(): lab for lab, p in expected_paths.items()}

    project.clusters_dir.mkdir(parents=True, exist_ok=True)
    removed: List[str] = []
    generated: List[Path] = []
    skipped_small: List[str] = []
    errors: List[Tuple[str, str]] = []
    handled: set[str] = set()
    stale_current: set[str] = set()

    # Remove only top-level historic per-class artifacts. Selection artifacts
    # (_selection.npz) and clusters/selections/* are deliberately preserved.
    if discover_stale:
        for fp in project.clusters_dir.glob('*.npz'):
            if fp.name.startswith('_'):
                continue
            key = str(fp.resolve()).lower()
            if key not in expected_resolved:
                try:
                    with np.load(fp, allow_pickle=True) as z:
                        looks_like_class_artifact = 'names' in z.files and 'linkage' in z.files
                    if not looks_like_class_artifact:
                        continue
                    fp.unlink()
                    removed.append(fp.name)
                except Exception as exc:
                    errors.append((fp.stem, f'Não foi possível verificar/remover artifact obsoleto {fp.name}: {exc}'))

        # Detect current classes whose artifact is missing or whose membership
        # no longer matches the current manual labels.
        for lab in current_labels:
            rows = rows_for(lab)
            fp = expected_paths[lab]
            if len(rows) < 2:
                if fp.exists():
                    try:
                        fp.unlink()
                        removed.append(fp.name)
                    except Exception as exc:
                        errors.append((lab, f'Não foi possível remover artifact inválido (<2 imagens): {exc}'))
                skipped_small.append(lab)
                handled.add(lab)
                continue
            if not fp.exists() or not _class_artifact_matches_rows(fp, rows):
                stale_current.add(lab)

    requested = set(str(x).strip() for x in (class_labels or []) if str(x).strip())
    targets = sorted(requested | stale_current)
    total = max(1, len(targets))

    for i, lab in enumerate(targets, 1):
        rows = rows_for(lab)
        fp = project.clusters_dir / f'{sanitize_filename(lab)}.npz'

        if rows is None or len(rows) == 0:
            # A fully relabeled old class is extinct. Its .npz is obsolete; trying
            # to cluster it is what previously produced N=0 < min_samples=2.
            try:
                if fp.exists():
                    fp.unlink()
                    if fp.name not in removed:
                        removed.append(fp.name)
                handled.add(lab)
            except Exception as exc:
                errors.append((lab, f'Falha ao remover artifact de classe sem imagens: {exc}'))
            if progress:
                progress(i, total, f'{lab}: removida (0 imagens)')
            continue

        if len(rows) < 2:
            try:
                if fp.exists():
                    fp.unlink()
                    if fp.name not in removed:
                        removed.append(fp.name)
                skipped_small.append(lab)
                handled.add(lab)
            except Exception as exc:
                errors.append((lab, f'Falha ao remover artifact de classe com <2 imagens: {exc}'))
            if progress:
                progress(i, total, f'{lab}: sem clustering (<2 imagens)')
            continue

        try:
            out = generate_cluster_artifact(
                project, lab, rows=rows, metric=metric, method=method,
                pca_dim=pca_dim, max_full=max_full, max_centroids=max_centroids,
                progress=None, filter_class=True, id_col='names'
            )
            generated.append(out)
            handled.add(lab)
        except Exception as exc:
            errors.append((lab, str(exc)))

        if progress:
            progress(i, total, lab)

    # Clear only classes whose state was actually reconciled. Any failed class
    # remains dirty and can be retried.
    try:
        project.dirty_classes.difference_update(handled)
    except Exception:
        pass

    return {
        'generated': generated,
        'removed': removed,
        'skipped_small': sorted(set(skipped_small)),
        'errors': errors,
        'handled': sorted(handled),
        'current_labels': current_labels,
        'targets': targets,
    }


def read_cluster_artifact(path: Path) -> dict:
    """Backward-compatible IAT wrapper around the shared artifact reader."""
    return _shared_read_cluster_artifact(path)


def cluster_groups(artifact: dict, cut01: float) -> np.ndarray:
    """Backward-compatible IAT wrapper around the shared flat-cut helper."""
    return _shared_cluster_groups(artifact, cut01)


# ---------------- Segmentation ----------------
def _largest_component(mask: np.ndarray, min_area: int = 2) -> np.ndarray:
    _load_cv2()
    if cv2 is None: return mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1: return np.zeros_like(mask, np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]; k = 1 + int(np.argmax(areas))
    if stats[k, cv2.CC_STAT_AREA] < min_area: return np.zeros_like(mask, np.uint8)
    return (lab == k).astype(np.uint8)


def _auto_polarity(gray: np.ndarray, mask_dark: np.ndarray, mask_light: Optional[np.ndarray] = None) -> np.ndarray:
    h,w = gray.shape; total=h*w
    candidates=[mask_dark]
    if mask_light is not None: candidates.append(mask_light)
    else: candidates.append((1-mask_dark).astype(np.uint8))
    scored=[]
    for m in candidates:
        m=_largest_component(m)
        area=int(m.sum())
        frac=area/max(total,1)
        # prefer plausible foreground that does not occupy almost whole image
        border=np.concatenate([m[0,:],m[-1,:],m[:,0],m[:,-1]]) if area else np.array([1])
        border_frac=float(border.mean())
        score=abs(frac-0.18)+0.7*border_frac
        if frac < 0.001 or frac > 0.95: score += 10
        scored.append((score,m))
    return min(scored,key=lambda t:t[0])[1]


def _threshold_mask(gray: np.ndarray, method: str, radius: int = 25) -> np.ndarray:
    _load_cv2()
    if cv2 is None: raise RuntimeError('opencv-python não está disponível')
    g=gray.astype(np.uint8)
    if method in ('otsu','otsu_lcc'):
        _, dark=cv2.threshold(g,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        _, light=cv2.threshold(g,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask=_auto_polarity(g,dark,light)
    elif method=='rolling_ball':
        k=max(3,int(radius)*2+1); bg=cv2.GaussianBlur(g,(k if k%2 else k+1,k if k%2 else k+1),0)
        darkness=cv2.subtract(bg,g); _,mask=cv2.threshold(darkness,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU); mask=_largest_component(mask)
    elif method=='fixed_243':
        dark=(g<243).astype(np.uint8); light=(g>12).astype(np.uint8); mask=_auto_polarity(g,dark,light)
    elif method=='black_hat_otsu':
        k=max(3,min(101,int(radius)*2+1)); kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)); bh=cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,kernel)
        _,mask=cv2.threshold(bh,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU); mask=_largest_component(mask)
    elif method=='adaptive_sauvola':
        r=max(3,int(radius)); win=2*r+1
        gf=g.astype(np.float32)/255.0; mean=cv2.boxFilter(gf,-1,(win,win),normalize=True,borderType=cv2.BORDER_REFLECT)
        mean2=cv2.boxFilter(gf*gf,-1,(win,win),normalize=True,borderType=cv2.BORDER_REFLECT); std=np.sqrt(np.maximum(mean2-mean*mean,0))
        k=0.34; R=0.5; thr=mean*(1+k*(std/R-1)); dark=(gf<thr).astype(np.uint8); mask=_auto_polarity(g,dark)
    elif method=='percentile_dark':
        thr=np.percentile(g,35); dark=(g<=thr).astype(np.uint8); mask=_auto_polarity(g,dark)
    elif method=='kmeans_intensity':
        vals=g.reshape(-1,1).astype(np.float32); criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,50,0.2)
        _,labs,cent=cv2.kmeans(vals,3,None,criteria,3,cv2.KMEANS_PP_CENTERS); cent=cent.ravel(); target=int(np.argmin(cent)); dark=(labs.ravel().reshape(g.shape)==target).astype(np.uint8); mask=_auto_polarity(g,dark)
    elif method=='watershed_otsu':
        _,base=cv2.threshold(g,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU); base=_auto_polarity(g,base)
        # For one-region output, watershed-like separation followed by largest region.
        dist=cv2.distanceTransform(base,cv2.DIST_L2,5); _,sure=cv2.threshold(dist,0.45*dist.max() if dist.max()>0 else 0,1,0); mask=_largest_component(sure.astype(np.uint8))
        if mask.sum()<2: mask=_largest_component(base)
    else:
        _,dark=cv2.threshold(g,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU); mask=_auto_polarity(g,dark)
    # speckle cleanup + one object, matching the R postprocessing intention
    mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    return _largest_component(mask)


def segmentation_threshold(gray: np.ndarray, method: str):
    """Grey-level threshold applied by *method* to the whole image, or ``None``.

    Global methods: Otsu t* (otsu, otsu_lcc and the first step of
    watershed_otsu; Otsu 1979), the fixed level 243 (fixed_243) and the 35th
    percentile (percentile_dark).  Rolling ball, black-hat, Sauvola and k-means
    have no single grey level (local or derived thresholds).
    Returns ``(value_or_None, description)``.
    """
    _load_cv2()
    g = np.asarray(gray, dtype=np.uint8)
    if method in ('otsu', 'otsu_lcc', 'watershed_otsu') and cv2 is not None:
        t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(t), 'Otsu t*'
    if method == 'fixed_243':
        return 243.0, 'limiar fixo 243 (12 se a polaridade for invertida)'
    if method == 'percentile_dark':
        return float(np.percentile(g, 35)), 'percentil 35'
    return None, 'limiar local/derivado: não há um único nível de cinza'


def _holes(mask: np.ndarray) -> int:
    _load_cv2()
    if cv2 is None or mask.sum()==0: return 0
    contours,hier=cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hier is None: return 0
    return int(sum(1 for h in hier[0] if h[3] >= 0))


def empty_segmentation_metrics() -> dict:
    """Measurement record for an image that could not be segmented."""
    out={key:np.nan for key in SEGMENTATION_METRIC_KEYS}
    out['Validation State']=False
    out['Metrics Version']=SEGMENTATION_METRICS_VERSION
    return out


# Crofton perimeter, 4 directions (0°, 45°, 90°, 135°).  Each 2×2 pixel
# configuration (bit 1 = top-left, 2 = bottom-left, 4 = top-right,
# 8 = bottom-right) contributes a fixed length; the estimate is unbiased for
# isotropically oriented shapes and has ≈1 % error on digital discs, against
# ≈5–10 % for the pixel-step length of the traced contour (cv2.arcLength).
# Legland, Kiêu & Devaux 2007, Image Anal. Stereol. 26:83-92;
# Lehmann & Legland 2012, Efficient N-D discrete perimeter estimation with
# Crofton formula (HAL hal-00673187); same table as scikit-image
# ``perimeter_crofton(directions=4)``.
_CROFTON_4 = np.array([
    0.0, math.pi/4*(1+1/math.sqrt(2)), math.pi/(4*math.sqrt(2)), math.pi/(2*math.sqrt(2)), 0.0,
    math.pi/4*(1+1/math.sqrt(2)), 0.0, math.pi/(4*math.sqrt(2)), math.pi/4, math.pi/2,
    math.pi/(4*math.sqrt(2)), math.pi/(4*math.sqrt(2)), math.pi/4, math.pi/2, 0.0, 0.0,
], dtype=float)


def crofton_perimeter(mask: np.ndarray) -> float:
    """Perimeter (pixels) of a binary mask by the 4-direction Crofton estimator."""
    from scipy import ndimage as ndi
    img=np.pad(np.asarray(mask,dtype=np.uint8)>0,1).astype(np.uint8)
    codes=ndi.convolve(img,np.array([[0,0,0],[0,1,4],[0,2,8]],dtype=np.uint8),mode='constant',cval=0)
    hist=np.bincount(codes.ravel(),minlength=16)[:16]
    return float(np.dot(_CROFTON_4,hist))


def pixel_corner_hull(contour_pts: np.ndarray) -> np.ndarray:
    """Convex hull (N×2, float) of the pixel *squares* of a boundary contour.

    Contour points are pixel centres; using the four corners of every boundary
    pixel gives the hull of the whole pixel set, so Feret diameters, the
    minimum-area rectangle and the hull area are measured on the same geometry
    as the pixel-count area (a single pixel is a 1×1 square, not a point).
    """
    c=np.asarray(contour_pts,dtype=np.float64).reshape(-1,2)
    corners=np.concatenate([c+[-0.5,-0.5],c+[0.5,-0.5],c+[-0.5,0.5],c+[0.5,0.5]]).astype(np.float32)
    hull=cv2.convexHull(corners)
    return hull[:,0,:].astype(np.float64)


def feret_diameters(hull: np.ndarray) -> Tuple[float,float]:
    """(Feret_max, Feret_min) of a convex polygon.

    Feret_max is the largest vertex-to-vertex distance; Feret_min is the
    smallest width over all directions, attained with one hull edge flush
    against the caliper (rotating calipers: Shamos 1978; Toussaint 1983,
    *Solving geometric problems with the rotating calipers*, Proc. IEEE
    MELECON).  Both are O(h²) in the number of hull vertices.
    """
    pts=np.asarray(hull,dtype=np.float64).reshape(-1,2)
    if len(pts)<2:
        return 0.0,0.0
    diff=pts[:,None,:]-pts[None,:,:]
    fmax=float(np.sqrt((diff*diff).sum(-1)).max())
    edges=np.roll(pts,-1,axis=0)-pts
    lengths=np.hypot(edges[:,0],edges[:,1])
    ok=lengths>1e-9
    if not np.any(ok):
        return fmax,0.0
    normals=np.c_[-edges[ok,1],edges[ok,0]]/lengths[ok,None]
    rel=pts[None,:,:]-pts[ok][:,None,:]
    widths=np.abs((rel*normals[:,None,:]).sum(-1)).max(axis=1)
    return fmax,float(widths.min())


def moment_ellipse(mask: np.ndarray) -> Tuple[float,float,float,float]:
    """(major, minor, eccentricity, orientation°) of the ellipse with the same
    normalized second central moments as the pixel set (Hu 1962; the usual
    "Major/Minor" of ImageJ, ZooProcess and EcoTaxa).  Every pixel is a unit
    square (adds 1/12 to each variance), so a single pixel is not a point."""
    m=cv2.moments(np.asarray(mask,dtype=np.uint8),binaryImage=True)
    m00=float(m['m00'])
    if m00<=0:
        return np.nan,np.nan,np.nan,np.nan
    mu20=m['mu20']/m00+1.0/12.0; mu02=m['mu02']/m00+1.0/12.0; mu11=m['mu11']/m00
    common=math.sqrt(4.0*mu11*mu11+(mu20-mu02)**2)
    l1=max((mu20+mu02+common)/2.0,0.0); l2=max((mu20+mu02-common)/2.0,0.0)
    major=4.0*math.sqrt(l1); minor=4.0*math.sqrt(l2)
    ecc=math.sqrt(max(0.0,1.0-(minor/major)**2)) if major>0 else np.nan
    theta=0.5*math.degrees(math.atan2(2.0*mu11,mu20-mu02))%180.0
    return major,minor,ecc,theta


def segmentation_metrics(gray: np.ndarray, method: str='otsu', radius: int=25, pixel_size_um: float=0.0) -> Tuple[dict, list]:
    """Segment one grey image and measure the object (all values in pixels).

    Definitions (see segmentation_docs.METRIC_FORMULAS and docs/METRICAS_E_REFERENCIAS.md):
    area A = number of foreground pixels; perimeter P = Crofton estimator
    (4 directions); circularity = 4πA/P²; ESD = 2√(A/π); Feret_max/min =
    true Feret diameters of the pixel-set hull; Elongation = Feret_max/Feret_min;
    MinRect = sides of the minimum-area enclosing rectangle; Rectangularity =
    A/(MinRect_w·MinRect_h); Solidity = A/hull area (same pixel geometry, so it
    is ≤ 1); Major/Minor/Eccentricity/Orientation from the moment ellipse;
    Biovolume = π·Major·Minor²/6 (ellipsoid, Vandromme et al. 2012);
    intensity statistics over the object pixels (0–255, Fisher kurtosis).
    Physical units are display-time conversions by the µm/pixel calibration.
    """
    _load_cv2()
    if cv2 is None: raise RuntimeError('opencv-python não está disponível')
    mask=_threshold_mask(gray,method,radius)
    h,w=gray.shape; area=int(mask.sum()); image_size=int(h*w)
    base=empty_segmentation_metrics()
    base.update({'Area (pxl)':area,'Image Width (pxl)':w,'Image Height (pxl)':h,'Image Size (pxl)':image_size,
                 'Validation State':bool(area>0),'Number Holes':0})
    if area<=0: return base,[]
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cnt=max(contours,key=cv2.contourArea)
    per=crofton_perimeter(mask)
    base['Perimeter (pxl)']=per
    base['circularity']=float(min(4*math.pi*area/(per*per),1.0)) if per>0 else np.nan
    base['ESD (pxl)']=float(2.0*math.sqrt(area/math.pi))
    hull=pixel_corner_hull(cnt[:,0,:])
    fmax,fmin=feret_diameters(hull)
    base['Feret_max']=fmax; base['Feret_min']=fmin
    base['Elongation']=fmax/fmin if fmin>1e-12 else np.nan
    rect=cv2.minAreaRect(hull.astype(np.float32)); rw,rh=float(rect[1][0]),float(rect[1][1])
    base['MinRect_w (pxl)']=max(rw,rh); base['MinRect_h (pxl)']=min(rw,rh)
    base['Rectangularity']=float(min(area/(rw*rh),1.0)) if rw*rh>1e-12 else np.nan
    hull_area=float(cv2.contourArea(hull.astype(np.float32)))
    base['Hull Area (pxl)']=hull_area
    base['Solidity']=float(min(area/hull_area,1.0)) if hull_area>1e-12 else np.nan
    major,minor,ecc,theta=moment_ellipse(mask)
    base['Major (pxl)']=major; base['Minor (pxl)']=minor; base['Eccentricity']=ecc; base['Orientation (deg)']=theta
    base['Biovolume (pxl3)']=float(math.pi*major*minor*minor/6.0) if np.isfinite(major) and np.isfinite(minor) else np.nan
    vals=gray[mask.astype(bool)].astype(float)
    mu=float(np.mean(vals)); sd=float(np.std(vals))
    base['Mean Intensity']=mu; base['Median Intensity']=float(np.median(vals)); base['Std Intensity']=sd
    base['Min Intensity']=float(np.min(vals)); base['Max Intensity']=float(np.max(vals))
    base['Contrast']=float(np.max(vals)-np.min(vals))
    if sd>1e-12:
        z=(vals-mu)/sd
        base['Skewness']=float(np.mean(z**3)); base['Kurtosis']=float(np.mean(z**4)-3.0)
    else:
        base['Skewness']=0.0; base['Kurtosis']=0.0
    border=np.zeros_like(mask,bool); border[[0,-1],:]=True; border[:,[0,-1]]=True; base['Percent_on_border']=float(mask[border].sum()/max(area,1)*100.0)
    base['Number Holes']=_holes(mask)
    # outline in pixel coordinates, simplify to <=220 pts
    pts=cnt[:,0,:].astype(float)
    if len(pts)>220: pts=pts[np.linspace(0,len(pts)-1,220).astype(int)]
    outline=[[float(x),float(y)] for x,y in pts]
    return base,outline


def segmentation_workers(n_images: int) -> int:
    """Threads for batch segmentation: ``ILT_SEG_WORKERS`` or CPUs − 1 (max 8)."""
    raw=str(os.environ.get('ILT_SEG_WORKERS','')).strip()
    try:
        workers=int(raw) if raw else max(1,min(8,available_cpus()-1))
    except ValueError:
        workers=max(1,min(8,available_cpus()-1))
    return max(1,min(workers,max(1,int(n_images))))


def _segment_one(task,method,radius,pixel_size_um):
    """Decode and measure one image (worker thread). Returns
    ``(located, read_ok, metrics, outline, height)``."""
    _pos,_raw_index,_rd,_name,img_path=task
    metrics=empty_segmentation_metrics(); outline=[]; height=None
    found=bool(img_path and img_path.exists())
    if not found:
        return False,False,metrics,outline,height
    g=cv2.imread(str(img_path),cv2.IMREAD_GRAYSCALE) if cv2 is not None else None
    if g is None:
        return True,False,metrics,outline,height
    try:
        metrics,outline=segmentation_metrics(g,method,radius,pixel_size_um)
        metrics['Image Width (pxl)']=g.shape[1]; metrics['Image Height (pxl)']=g.shape[0]; metrics['Image Size (pxl)']=g.size
        height=int(g.shape[0])
    except Exception:
        metrics=empty_segmentation_metrics(); outline=[]
    return True,True,metrics,outline,height


def run_segmentation(project: IATProject, csv_name: str, method: str='otsu', radius: int=25, pixel_size_um: float=DEFAULT_PIXEL_SIZE_UM_PER_PX,
                     progress: Optional[Callable[[int,int,str],None]]=None,
                     include_gids: Optional[Sequence[str]]=None) -> Tuple[Path,Path,dict]:
    """Generate or update segmentation for one batch.

    When *include_gids* is provided, only those rows are recomputed and the
    existing segmentation CSV/cache for every other image is preserved. This is
    what allows the GUI to segment a Scatter selection or Gallery marks without
    destroying prior results for the remainder of the batch.
    """
    _load_cv2()
    method=method if method in SEGMENTATION_METHODS else 'otsu'
    src=project.dataframes_dir/csv_name
    raw=pd.read_csv(src,low_memory=False)
    if 'names' not in raw.columns:
        raise ValueError(f'{csv_name}: coluna names ausente')
    # Build the same stable gid used by the IAT without calling
    # ensure_required_columns(): segmentation must not drop images merely because
    # their projection x/y is missing or non-finite.
    if 'custom_data' in raw.columns:
        custom=raw['custom_data'].astype('string')
        bad=custom.isna() | (custom.str.len()==0)
        custom.loc[bad]=raw.loc[bad,'names'].astype(str)
        custom=custom.astype(str)
    else:
        custom=raw['names'].astype(str)
    raw['__iat_gid']=str(csv_name)+'::'+custom

    subset_mode=include_gids is not None
    if subset_mode:
        wanted=set(map(str,include_gids or []))
        work=raw.loc[raw['__iat_gid'].astype(str).isin(wanted)].copy()
    else:
        work=raw.copy()

    rows=[]; outlines={}; located=read_ok=segmented=0
    total=len(work)
    # Paths are resolved in this thread (shared cache); decoding + measuring
    # runs in a thread pool: OpenCV releases the GIL, so the speed-up is close
    # to the number of cores.  ex.map keeps the row order.
    tasks=[]
    for pos,(raw_index,row) in enumerate(work.iterrows(),1):
        rd=row.to_dict(); name=str(rd.get('names',''))
        tasks.append((pos,raw_index,rd,name,project.resolve_image(csv_name,name)))
    workers=segmentation_workers(total)
    executor=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='ilt-seg') if workers>1 else None
    try:
        results=(executor.map(_segment_one,tasks,(method,)*total,(radius,)*total,(pixel_size_um,)*total,chunksize=8)
                 if executor is not None else
                 (_segment_one(t,method,radius,pixel_size_um) for t in tasks))
        for (pos,raw_index,rd,name,img_path),(found,read,metrics,outline,height) in zip(tasks,results):
            located+=int(found); read_ok+=int(read)
            if metrics['Validation State']: segmented+=1
            class_name=str(rd.get('manual_label',rd.get('preds1',rd.get('pred',''))))
            prev=str(rd.get('correct_label',class_name))
            try:event_index=int(raw_index)
            except Exception:event_index=pos-1
            rec={'Image Name':name,'Image State (T/F)':bool(found),'SegmentationMethod':method,
                 'Pixel Size (um/pixel)':float(pixel_size_um),**metrics,
                 'Event Indexer':event_index,'Class Name':class_name,'ClassName_previuos':prev,'Background Image':''}
            rows.append(rec)
            if outline:
                outlines[name]={'points':outline,'width':metrics.get('Image Width (pxl)'),'height':height}
            if progress and (pos==1 or pos%20==0 or pos==total):
                progress(pos,max(total,1),name)
    except BaseException:
        if executor is not None:
            executor.shutdown(wait=False,cancel_futures=True)  # "Stop" must not wait for the queue
        raise
    else:
        if executor is not None:
            executor.shutdown(wait=True)

    project.segmentation_dir.mkdir(parents=True,exist_ok=True)
    project.segmentation_cache_dir.mkdir(parents=True,exist_ok=True)
    out_csv=project.segmentation_csv_path(csv_name,method)
    out_cache=project.segmentation_cache_path(csv_name,method)
    new_df=pd.DataFrame(rows)

    if subset_mode:
        processed_names=set(work['names'].astype(str).tolist())
        # Preserve every segmentation row that was outside the requested scope.
        if out_csv.exists():
            try:
                old_df=pd.read_csv(out_csv,low_memory=False)
            except Exception:
                old_df=pd.DataFrame()
        else:
            old_df=pd.DataFrame()
        if not old_df.empty and 'Image Name' in old_df.columns:
            old_df=old_df.loc[~old_df['Image Name'].astype(str).isin(processed_names)].copy()
            combined=pd.concat([old_df,new_df],ignore_index=True,sort=False)
        else:
            combined=new_df
        if not combined.empty and 'Event Indexer' in combined.columns:
            try:
                combined['__event_sort']=pd.to_numeric(combined['Event Indexer'],errors='coerce')
                combined=combined.sort_values('__event_sort',kind='mergesort',na_position='last').drop(columns='__event_sort')
            except Exception:
                pass
        atomic_write_csv(combined,out_csv,index=False)

        # Preserve outlines outside the subset; selected images are replaced,
        # including removal of a stale outline if the new segmentation failed.
        if out_cache.exists():
            try:merged_outlines=json.loads(out_cache.read_text())
            except Exception:merged_outlines={}
        else:
            merged_outlines={}
        for name in processed_names:
            merged_outlines.pop(name,None)
        merged_outlines.update(outlines)
        atomic_write_text(json.dumps(merged_outlines,separators=(',',':')),out_cache)
    else:
        atomic_write_csv(new_df,out_csv,index=False)
        atomic_write_text(json.dumps(outlines,separators=(',',':')),out_cache)

    project.invalidate_segmentation_cache(csv_name, method)
    stats={'rows':total,'located':located,'read':read_ok,'segmented':segmented,'contours':len(outlines)}
    return out_csv,out_cache,stats
