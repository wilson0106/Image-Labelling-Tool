#!/usr/bin/env python3
"""EcoTaxa import table (TSV) from IAT rows with segmentation (no UI).

EcoTaxa (Picheral et al. 2017, https://ecotaxa.obs-vlfr.fr) imports a folder
or zip containing images and a tab-separated file whose name starts with
``ecotaxa_``; the first header line has the column names and the second line
the type of each column, ``[t]`` (text) or ``[f]`` (float).  Column names use
the ``img_``, ``object_``, ``sample_``, ``acq_`` and ``process_`` prefixes.
Morphometric values are written in pixels, as ZooProcess does, with the
calibration in ``process_pixel`` (µm per pixel); EcoTaxa converts them.
Measurement definitions: docs/METRICAS_E_REFERENCIAS.md, section 3.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ilt_contracts import ILT_VERSION

# (IAT segmentation column, EcoTaxa column) — values in pixels.
MEASUREMENTS = [
    ('Area (pxl)', 'object_area'),
    ('Perimeter (pxl)', 'object_perim.'),
    ('Major (pxl)', 'object_major'),
    ('Minor (pxl)', 'object_minor'),
    ('Orientation (deg)', 'object_angle'),
    ('circularity', 'object_circ.'),
    ('Feret_max', 'object_feret'),
    ('Feret_min', 'object_feretmin'),
    ('ESD (pxl)', 'object_esd'),
    ('Elongation', 'object_elongation_feret'),
    ('Eccentricity', 'object_eccentricity'),
    ('Solidity', 'object_solidity'),
    ('Rectangularity', 'object_rectangularity'),
    ('Hull Area (pxl)', 'object_convex_area'),
    ('Biovolume (pxl3)', 'object_biovolume_px3'),
    ('Mean Intensity', 'object_mean'),
    ('Median Intensity', 'object_median'),
    ('Std Intensity', 'object_stddev'),
    ('Min Intensity', 'object_min'),
    ('Max Intensity', 'object_max'),
    ('Skewness', 'object_skew'),
    ('Kurtosis', 'object_kurt'),
    ('Contrast', 'object_range'),
    ('Percent_on_border', 'object_border_pct'),
    ('Number Holes', 'object_holes'),
]

TEXT_COLUMNS = ['img_file_name', 'object_id', 'object_date', 'object_time', 'object_annotation_category',
                'object_annotation_status', 'object_pred_category', 'sample_id', 'acq_id', 'process_id',
                'process_software', 'process_segmentation_method']
FLOAT_COLUMNS = ['img_rank', 'object_depth_min', 'object_depth_max', *[e for _c, e in MEASUREMENTS],
                 'object_pred_score', 'object_uncertainty_margin', 'process_pixel', 'process_metrics_version']
COLUMNS = ['img_file_name', 'img_rank', 'object_id', 'object_date', 'object_time', 'object_depth_min',
           'object_depth_max', 'object_annotation_category', 'object_annotation_status',
           *[e for _c, e in MEASUREMENTS], 'object_pred_category', 'object_pred_score', 'object_uncertainty_margin',
           'sample_id', 'acq_id', 'process_id', 'process_pixel', 'process_software', 'process_segmentation_method',
           'process_metrics_version']
TYPES = ['[f]' if c in FLOAT_COLUMNS else '[t]' for c in COLUMNS]

STATUS_MODES = {
    'auto': 'Automático: “validated” se o rótulo foi alterado no IAT (ou veio da pasta de origem no modo '
            'auditoria); senão “predicted”',
    'predicted': 'Todas como “predicted” (previsão ainda não revisada)',
    'validated': 'Todas como “validated” (todas foram revisadas por um especialista)',
}


def _clean_text(series):
    return series.fillna('').astype(str).str.replace(r'[\t\r\n]+', ' ', regex=True)


def _truthy(series):
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna('').astype(str).str.strip().str.lower().isin({'1', 'true', 't', 'yes', 'y'})


def ecotaxa_frame(df: pd.DataFrame, pixel_um: float, method: str, status_mode: str = 'auto') -> pd.DataFrame:
    """One EcoTaxa row per image of *df* (rows with segmentation columns attached)."""
    from iat_temporal_analysis import parse_metadata
    out = pd.DataFrame(index=df.index, columns=COLUMNS)
    if df.empty:
        return out.iloc[0:0]
    names = df['names'].astype(str)
    out['img_file_name'] = names
    out['img_rank'] = 0
    out['object_id'] = names.str.replace(r'\.[^./\\]+$', '', regex=True)
    meta = parse_metadata(pd.DataFrame({'names': names.to_numpy()}, index=df.index))
    out['object_date'] = meta['day'].dt.strftime('%Y%m%d').fillna('')
    has_time = meta['minute'].notna()
    out['object_time'] = meta['timestamp'].dt.strftime('%H%M%S').where(has_time, '').fillna('')
    depth_m = pd.to_numeric(meta['depth'], errors='coerce') / 1000.0
    out['object_depth_min'] = depth_m
    out['object_depth_max'] = depth_m
    manual = _clean_text(df['manual_label']) if 'manual_label' in df.columns else pd.Series('', index=df.index)
    out['object_annotation_category'] = manual
    if status_mode == 'validated':
        status = pd.Series('validated', index=df.index)
    elif status_mode == 'predicted':
        status = pd.Series('predicted', index=df.index)
    else:
        initial = _clean_text(df['correct_label']) if 'correct_label' in df.columns else manual
        changed = manual != initial
        audit = _truthy(df['audit_source_labels']) if 'audit_source_labels' in df.columns \
            else pd.Series(False, index=df.index)
        status = pd.Series(np.where(changed | audit, 'validated', 'predicted'), index=df.index)
    out['object_annotation_status'] = status.where(manual.str.len() > 0, '')
    for src, dst in MEASUREMENTS:
        out[dst] = pd.to_numeric(df[src], errors='coerce') if src in df.columns else np.nan
    pred_col = 'preds1' if 'preds1' in df.columns else ('pred' if 'pred' in df.columns else None)
    out['object_pred_category'] = _clean_text(df[pred_col]) if pred_col else ''
    out['object_pred_score'] = pd.to_numeric(df['confs1'], errors='coerce') if 'confs1' in df.columns else np.nan
    out['object_uncertainty_margin'] = pd.to_numeric(df['var_norm'], errors='coerce') if 'var_norm' in df.columns \
        else np.nan
    batch = df['csv_basename'].astype(str) if 'csv_basename' in df.columns else pd.Series('', index=df.index)
    sample = batch.str.replace(r'\.csv$', '', regex=True)
    out['sample_id'] = sample
    out['acq_id'] = sample
    out['process_id'] = f'ilt_{method}'
    out['process_pixel'] = float(pixel_um)
    out['process_software'] = f'ILT {ILT_VERSION}'
    out['process_segmentation_method'] = method
    out['process_metrics_version'] = pd.to_numeric(df['Metrics Version'], errors='coerce') \
        if 'Metrics Version' in df.columns else np.nan
    for col in FLOAT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors='coerce').astype(float)
    for col in TEXT_COLUMNS:
        out[col] = _clean_text(out[col])
    return out[COLUMNS]


def write_header(handle):
    handle.write('\t'.join(COLUMNS) + '\n')
    handle.write('\t'.join(TYPES) + '\n')


def append_rows(handle, frame: pd.DataFrame):
    if frame.empty:
        return 0
    frame.to_csv(handle, sep='\t', header=False, index=False, na_rep='', lineterminator='\n',
                 float_format='%.6g')
    return len(frame)


def ecotaxa_filename(name: str) -> str:
    """EcoTaxa only reads TSV files whose name starts with ``ecotaxa_``."""
    stem = Path(str(name or '').strip() or 'ecotaxa_export').name
    for ext in ('.tsv', '.txt', '.csv'):
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
    stem = stem.replace(os.sep, '_')
    if not stem.lower().startswith('ecotaxa_'):
        stem = 'ecotaxa_' + stem
    return stem + '.tsv'
