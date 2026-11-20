#!/usr/bin/env python3
"""Plotly figures for the IAT (presentation only).

Data selection always comes from the reused ``IATWorkspace`` methods; this
module only turns their output into WebGL figures (``scattergl``), which the
browser renders on the GPU.
"""
from __future__ import annotations

import base64
import io
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from iat_backend import get_class_color, get_color

from .theme import STATUS, plot_layout

NAMED_CLASS_COLS = {'manual_label', 'source_label', 'preds1', 'pred'}
CONTINUOUS = {'hdbscan_outlier', 'knn_agreement', 'review_priority', 'confs1', 'var_norm'}
MAX_POINTS_DRAWN = 400_000  # display only; selections always use every point
MAX_SELECTION_MARKERS = 20_000  # ring markers drawn for the selection overlay
_STAIN_CACHE = {}


def _rgba(color):
    from matplotlib.colors import to_rgba
    return to_rgba(color)


def _png_data_uri(rgba):
    from PIL import Image
    img = Image.fromarray(np.clip(rgba * 255, 0, 255).astype(np.uint8), 'RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=False, compress_level=3)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def class_stain(pts, color, show_hull):
    """Same density-stain recipe as the classic Overview (no internal contours)."""
    from matplotlib.path import Path as MplPath
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import ConvexHull
    out = {'image': None, 'hull': None, 'points': None}
    if len(pts) < 20:
        out['points'] = pts
        return out
    try:
        hull = ConvexHull(pts)
        poly = pts[hull.vertices]
    except Exception:
        out['points'] = pts
        return out
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    if xmax > xmin and ymax > ymin:
        H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=70, range=[[xmin, xmax], [ymin, ymax]])
        H = gaussian_filter(H, sigma=2.1)
        xc = (xe[:-1] + xe[1:]) / 2
        yc = (ye[:-1] + ye[1:]) / 2
        X, Y = np.meshgrid(xc, yc, indexing='xy')
        mask = MplPath(poly).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
        Z = H.T
        zmax = float(np.nanmax(Z)) if Z.size else 0.0
        if zmax > 0:
            norm = np.clip((Z / zmax - 0.05) / 0.95, 0, 1)
            rgba = np.zeros((Z.shape[0], Z.shape[1], 4), dtype=float)
            base = _rgba(color)
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = base[0], base[1], base[2]
            # Slightly stronger alpha than on the classic white background so
            # the stain reads on the dark HUD surface.
            rgba[..., 3] = np.where(mask, 0.06 + 0.42 * np.power(norm, 1.28), 0.0)
            rgba[..., 3] = np.where(norm > 0, rgba[..., 3], 0.0)
            out['image'] = (_png_data_uri(rgba[::-1]), xmin, xmax, ymin, ymax)
    if show_hull:
        out['hull'] = np.vstack([poly, poly[:1]])
    return out


def scatter_figure(iat, lang, theme):
    """Projection figure for the current Scatter state of ``iat``."""
    df = iat._scatter_filtered()
    iat._scatter_points_df = df.reset_index(drop=True)
    layout = plot_layout(theme, margin=dict(l=8, r=8, t=30, b=8), dragmode='lasso', uirevision='scatter',
                         xaxis=dict(showticklabels=False, showgrid=True, zeroline=False),
                         yaxis=dict(showticklabels=False, showgrid=True, zeroline=False, scaleanchor='x'),
                         legend=dict(orientation='v', x=1.0, y=1.0, xanchor='right', bgcolor='rgba(5,8,10,.6)',
                                     font=dict(size=10, color='#d6dce0')))
    traces = []
    images = []
    if df.empty:
        if iat.vars['batch'].get() == 'Overview' and not iat.visible_classes and not iat.current_df.empty:
            msg = iat.tr('select_class_prompt')
        else:
            msg = 'Sem dados' if lang == 'pt' else 'No data'
        layout['annotations'] = [dict(text=msg, x=.5, y=.5, xref='paper', yref='paper', showarrow=False,
                                      font=dict(color=STATUS['muted'], size=14))]
        return {'data': [], 'layout': layout}, 0, False
    mode = str(iat.vars['scatter_color_mode'].get() or 'manual_label')
    pred_col = 'preds1' if 'preds1' in df.columns else ('pred' if 'pred' in df.columns else None)
    col_map = {'manual_label': 'manual_label', 'prediction': pred_col, 'source_label': 'source_label'}
    col = col_map.get(mode, mode)
    if col is None or col not in df.columns:
        col, mode = 'manual_label', 'manual_label'
    continuous = col in CONTINUOUS
    original_overview = (iat.vars['batch'].get() == 'Overview' and not iat._analysis_projection_active
                         and mode == 'manual_label'
                         and (not iat._name_filter_spec.enabled or bool(iat.vars['name_filter_stains'].get())))
    filter_stains = (iat._name_filter_spec.enabled and bool(iat.vars['name_filter_stains'].get())
                     and not continuous and col in NAMED_CLASS_COLS)
    xs = pd.to_numeric(df['x'], errors='coerce').to_numpy(float)
    ys = pd.to_numeric(df['y'], errors='coerce').to_numpy(float)
    gids = df['gid'].astype(str).to_numpy(object)
    sampled = len(df) > MAX_POINTS_DRAWN
    take = slice(None, None, max(1, math.ceil(len(df) / MAX_POINTS_DRAWN))) if sampled else slice(None)
    if original_overview or filter_stains:
        cats = df[col].fillna('').astype(str)
        show_hull = bool(iat.vars['show_hulls'].get())
        base_key = iat.scatter_base_key() if hasattr(iat, 'scatter_base_key') else None
        for lab, idx in sorted(cats.groupby(cats, sort=False).groups.items()):
            color = get_class_color(lab)
            ck = (base_key, lab, len(idx), show_hull)
            st = _STAIN_CACHE.get(ck) if base_key is not None else None
            if st is None:
                pts = np.c_[pd.to_numeric(df.loc[idx, 'x'], errors='coerce'), pd.to_numeric(df.loc[idx, 'y'], errors='coerce')]
                pts = pts[np.isfinite(pts).all(axis=1)]
                st = class_stain(pts, color, show_hull)
                if base_key is not None:
                    if len(_STAIN_CACHE) > 400:
                        _STAIN_CACHE.clear()
                    _STAIN_CACHE[ck] = st
            if st['image']:
                uri, x0, x1, y0, y1 = st['image']
                images.append(dict(source=uri, xref='x', yref='y', x=x0, y=y1, sizex=x1 - x0, sizey=y1 - y0,
                                   sizing='stretch', layer='below', opacity=1))
                # invisible anchor trace keeps the axes range and enables hover by class
                traces.append(dict(type='scattergl', mode='markers', x=[x0, x1], y=[y0, y1], name=lab,
                                   marker=dict(size=1, color=color, opacity=0), hoverinfo='skip', showlegend=True))
            if st['points'] is not None:
                traces.append(dict(type='scattergl', mode='markers', x=st['points'][:, 0], y=st['points'][:, 1],
                                   name=lab, marker=dict(size=6, color=color, opacity=.9), hoverinfo='name'))
            if st['hull'] is not None:
                traces.append(dict(type='scattergl', mode='lines', x=st['hull'][:, 0], y=st['hull'][:, 1],
                                   line=dict(color=color, width=1), opacity=.7, hoverinfo='skip', showlegend=False))
    elif continuous:
        vals = pd.to_numeric(df[col], errors='coerce').to_numpy(float)
        finite = np.isfinite(vals)
        plotvals = np.where(finite, vals, np.nanmedian(vals[finite]) if finite.any() else 0.0)
        # hoverinfo 'none': the thumbnail tooltip (dcc.Tooltip, iat_scatter) replaces the text label.
        traces.append(dict(type='scattergl', mode='markers', x=xs[take], y=ys[take], customdata=gids[take],
                           marker=dict(size=4, color=plotvals[take], colorscale='Viridis', opacity=.85,
                                       colorbar=dict(title=dict(text=col, font=dict(size=10)), thickness=10,
                                                     tickfont=dict(size=9, color=STATUS['muted']))),
                           hoverinfo='none', name=col, showlegend=False))
    else:
        cats = df[col].fillna('').astype(str).to_numpy(object)
        values = sorted(set(cats.tolist()))
        named = col in NAMED_CLASS_COLS
        for i, lab in enumerate(values):
            m = cats == lab
            if sampled:
                m = m & (np.arange(len(m)) % max(1, math.ceil(len(df) / MAX_POINTS_DRAWN)) == 0)
            label = 'Noise' if col == 'hdbscan_cluster' and lab == '-1' else lab
            traces.append(dict(type='scattergl', mode='markers', x=xs[m], y=ys[m], customdata=gids[m], name=label,
                               marker=dict(size=4, color=get_class_color(lab) if named else get_color(i), opacity=.85),
                               hoverinfo='none', showlegend=len(values) <= 18))
    # Selection overlay (red ring) and lasso/rectangle outline.
    if iat.selected_gids:
        sel = pd.Index(gids).isin(iat.selected_gids)
        if sel.any():
            s_idx = np.flatnonzero(sel)
            if len(s_idx) > MAX_SELECTION_MARKERS:
                s_idx = s_idx[::math.ceil(len(s_idx) / MAX_SELECTION_MARKERS)]
            traces.append(dict(type='scattergl', mode='markers', x=xs[s_idx], y=ys[s_idx], customdata=gids[s_idx],
                               name=iat.tr('selected'), showlegend=False, hoverinfo='skip',
                               marker=dict(size=7, color='rgba(0,0,0,0)', line=dict(color='#ff4d5e', width=1.2))))
    if iat._selection_outline is not None and len(iat._selection_outline) >= 2:
        poly = np.asarray(iat._selection_outline, float)
        traces.append(dict(type='scatter', mode='lines', x=poly[:, 0], y=poly[:, 1], hoverinfo='skip',
                           line=dict(color='#ff4d5e', width=1.2), showlegend=False))
    proj = 'Original t-SNE' if not iat._analysis_projection_active else str(iat._analysis_projection_method).upper()
    layout['title'] = dict(text=f'{proj} • {col}', font=dict(size=12, color=STATUS['muted']), x=.01, y=.99)
    layout['images'] = images
    return {'data': traces, 'layout': layout}, len(df), sampled


def _class_counts(iat):
    def compute():
        source = (iat._name_filtered_source(iat._analysis_plot_source())
                  if iat._name_filter_spec.enabled else iat.current_df)
        if source is None or source.empty:
            return None, None
        col = iat._scatter_class_column(source)
        if col not in source.columns:
            return None, None
        vals = source[col].fillna('').astype(str).str.strip()
        vals = vals[vals.str.len() > 0]
        if vals.empty:
            return col, None
        return col, vals.value_counts().sort_values(ascending=False)
    if hasattr(iat, '_scatter_memo'):
        return iat._scatter_memo('counts', None, compute)
    return compute()


def class_abundance_figure(iat, lang, theme):
    """Same counts/colours/scale rule as the classic class panel."""
    layout = plot_layout(theme, margin=dict(l=8, r=40, t=26, b=8), bargap=.28, clickmode='event',
                         yaxis=dict(autorange='reversed', automargin=True, showgrid=False,
                                    tickfont=dict(size=10, color='#e8ecef')),
                         xaxis=dict(side='top', tickfont=dict(size=9, color=STATUS['muted'])), hovermode='closest')
    col, counts = _class_counts(iat)
    if counts is None:
        return {'data': [], 'layout': layout}
    labs = counts.index.tolist()
    raw = counts.values.astype(float)
    full = sorted(labs)
    named = col in NAMED_CLASS_COLS
    cmap = {lab: (get_class_color(lab) if named else get_color(i)) for i, lab in enumerate(full)}
    pct = (iat.vars['class_axis_mode'].get() or 'N') == '%'
    x = raw / max(raw.sum(), 1.0) * 100.0 if pct else raw
    scale = max(45.0, min(220.0, float(iat.vars['class_axis_scale'].get() or 100.0))) / 100.0
    vmax = float(np.nanmax(x)) if len(x) else 1.0
    opac = [.95 if lab in iat.visible_classes else .22 for lab in labs]
    layout['xaxis']['range'] = [0, max(vmax * 1.08 * scale, 1.0)]
    layout['xaxis']['ticksuffix'] = '%' if pct else ''
    layout['height'] = max(260, 22 * len(labs) + 60)
    text = [f'{v:.1f}%' if pct else f'{int(v):,}'.replace(',', '.') for v in x]
    return {'data': [dict(type='bar', orientation='h', y=labs, x=x, marker=dict(color=[cmap[l] for l in labs], opacity=opac),
                          customdata=labs, text=text, textposition='outside', cliponaxis=False,
                          textfont=dict(size=9, color=STATUS['muted']),
                          hovertemplate='%{y}: %{text}<extra></extra>')], 'layout': layout}


def dendrogram_figure(payload, theme, lang):
    layout = plot_layout(theme, margin=dict(l=6, r=6, t=30, b=6), hovermode='closest', showlegend=False,
                         xaxis=dict(visible=False), yaxis=dict(visible=False))
    if not payload or payload.get('missing') or payload.get('error'):
        msg = payload.get('message') if payload else ''
        layout['hovermode'] = False
        layout['annotations'] = [dict(text=msg or '', x=.5, y=.5, xref='paper', yref='paper', showarrow=False,
                                      font=dict(color=STATUS['bad'] if payload and payload.get('error') else STATUS['muted'], size=13))]
        return {'data': [], 'layout': layout}
    by_color = {}
    node_x, node_y, node_c = [], [], []
    for xs, ys, c in zip(payload['icoord'], payload['dcoord'], payload.get('color_list') or []):
        d = by_color.setdefault(c, ([], []))
        d[0].extend(list(xs) + [None])
        d[1].extend(list(ys) + [None])
        # Merge node = top of the ∩ link; clicking it sets the cut just below.
        node_x.append((xs[1] + xs[2]) / 2.0)
        node_y.append(ys[1])
        node_c.append(c)
    traces = [dict(type='scattergl', mode='lines', x=x, y=y, line=dict(color=c, width=1.1), hoverinfo='skip')
              for c, (x, y) in by_color.items()]
    hint = ('altura %{y:.4g}<br>clique: cortar logo abaixo deste nó<extra></extra>' if lang == 'pt'
            else 'height %{y:.4g}<br>click: cut just below this node<extra></extra>')
    traces.append(dict(type='scatter', mode='markers', x=node_x, y=node_y, hovertemplate=hint,
                       marker=dict(size=7, color=node_c, opacity=.55, line=dict(width=0)), name='nodes'))
    cut_y = float(payload['cut']) * float(payload['dist_max'])
    layout['shapes'] = [dict(type='line', xref='paper', x0=0, x1=1, y0=cut_y, y1=cut_y,
                             line=dict(color=STATUS['warn'], width=1.2, dash='dash'))]
    title = (('Dendrograma resumido' if payload.get('truncated') else 'Dendrograma') if lang == 'pt'
             else ('Truncated dendrogram' if payload.get('truncated') else 'Dendrogram'))
    layout['title'] = dict(text=title, font=dict(size=12, color=STATUS['muted']), x=.01)
    return {'data': traces, 'layout': layout}


def histogram_figure(values, lo, hi, theme, color='#3b82f6', height=90):
    vals = np.asarray(values, float)
    vals = vals[np.isfinite(vals)]
    layout = plot_layout(theme, margin=dict(l=4, r=4, t=4, b=18), height=height, bargap=.05, showlegend=False,
                         xaxis=dict(range=[0, 1], tickfont=dict(size=9, color=STATUS['muted'])),
                         yaxis=dict(visible=False))
    layout['shapes'] = [dict(type='rect', xref='x', yref='paper', x0=lo, x1=hi, y0=0, y1=1,
                             fillcolor='rgba(34,211,238,.10)', line=dict(width=0))]
    if not len(vals):
        layout['annotations'] = [dict(text='Sem valores neste projeto/seleção', x=.5, y=.5, xref='paper', yref='paper',
                                      showarrow=False, font=dict(size=11, color=STATUS['muted']))]
        return {'data': [], 'layout': layout}
    return {'data': [dict(type='histogram', x=vals, xbins=dict(start=0, end=1, size=.025),
                          marker=dict(color=color), hoverinfo='skip')], 'layout': layout}
