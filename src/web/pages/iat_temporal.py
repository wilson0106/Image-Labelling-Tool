#!/usr/bin/env python3
"""IAT › Temporal and spatial analysis (read-only view of a fixed projection frame).

Data loading, filters, intervals, palette, density recipe, figure export and
GIF export use the original modules (``iat_temporal_analysis``,
``iat_temporal_figure``) and the original panel methods; only the on-screen
charts are Plotly (WebGL).  The projection can be the project t-SNE (CSV x/y)
or one computed from the embeddings (PCA, t-SNE, UMAP, supervised UMAP,
PaCMAP) — ``iat_temporal_analysis.temporal_projection``.
"""
from __future__ import annotations

import io
import queue
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

import iat_temporal_spatial as _tsp
from iat_temporal_analysis import (PROJECTION_AXES, PROJECTION_METHODS, diversity_indices, filter_fixed,
                                   frame_class_counts, intervals, load_points, temporal_projection)
from iat_temporal_figure import (DENSITY_ALPHAS, DENSITY_MIN_POINTS, _savefig_kwargs, class_density, density_levels,
                                 density_reference, export_style, make_class_palette, render_figure,
                                 save_animated_gif, save_figure, save_frame_images)

from iat_display_groups import count_renamed, empty_groups, load_groups, map_names

from .. import components as C
from ..headless import FiledialogShim, HeadlessBoolVar, HeadlessVar, NOTIFY
from . import iat_groups as _groups
from . import iat_origin as _origin
from ..theme import GRAPH_CONFIG, STATUS, hex_alpha, plot_layout
from .execution import act, dyn

AXES = ['Data', 'Hora', 'Profundidade', 'Atributo']
PERIODS = ['Horário', 'Diário', 'Semanal', 'Mensal', 'Trimestral', 'Anual', 'Lunar (astronômico)']
MODES = ['Individual', 'Comparar', 'Animação']
SPEEDS = {'0.5 s': 500, '1 s': 1000, '2 s': 2000, '3 s': 3000}
SERIES_MODES = ['Contagem', 'Proporção (%)', 'Diversidade']
AXIS_TIPS = {
    'Data': 'Escolha Horário, Diário, Semanal, Lunar, Mensal, Trimestral ou Anual. Mês = mês real, não 30 dias.',
    'Hora': 'Cada quadro mostra uma faixa horária. Passo em minutos.',
    'Profundidade': 'Cada quadro mostra uma faixa de profundidade. Passo em mm.',
    'Atributo': 'Cada quadro mostra um valor do atributo (ex.: estacaoA, estacaoB).',
}
PROJECTION_TIPS = {
    'original': 'Coordenadas x/y gravadas na classificação (openTSNE, referencial comum a todos os batches).',
    'pca': 'Projeção linear dos embeddings (L2 → PCA). Distâncias globais preservadas; separação local menor.',
    'tsne': 't-SNE recalculado sobre as imagens carregadas (openTSNE, cosseno). Preserva vizinhanças locais.',
    'umap': 'UMAP não supervisionado (n_neighbors 30, min_dist 0,1). Indicado para procurar erros de rótulo.',
    'umap_supervised': ('UMAP supervisionado pelos rótulos manuais atuais (peso 0,35): aproxima imagens da mesma '
                        'classe e mostra a estrutura dada a anotação. Não use para procurar erros de rótulo.'),
    'pacmap': 'PaCMAP: equilíbrio entre estrutura local e global.',
}
FILTERS = (
    ('Data', [('De (AAAA-MM-DD)', 'date_from'), ('Até (AAAA-MM-DD)', 'date_to')], 'Escolha um período de coleta.'),
    ('Hora', [('De (HH:MM)', 'time_from'), ('Até (HH:MM)', 'time_to')], 'Ex.: 10:00 até 14:00, em todos os dias.'),
    ('Profundidade', [('Mínima (mm)', 'depth_min'), ('Máxima (mm)', 'depth_max')], 'Ex.: 1000 até 2000 mm.'),
    ('Atributo', [('Texto presente no nome da imagem', 'attribute_value')], 'Ex.: estacaoB. Sem expressão regular.'),
)


def _bump_iat():
    from .. import iat_core
    iat = iat_core._IAT
    if iat is not None:
        iat.web_version = getattr(iat, 'web_version', 0) + 1


class TemporalState:
    """State of the temporal panel; reuses the classic panel's frame logic."""
    _settings = _tsp.TemporalSpatialPanel._settings
    _display_frames = _tsp.TemporalSpatialPanel._display_frames
    _display_panels = _tsp.TemporalSpatialPanel._display_panels
    frame_from_label = _tsp.TemporalSpatialPanel.frame_from_label

    def __init__(self, ctl):
        self._ctl = ctl
        self.data = pd.DataFrame()
        self.view = pd.DataFrame()
        self.frames = []
        self.current = 0
        self.bounds = None
        self.palette = {}
        self.timer = None
        self.version = 0
        self.selected_classes = None
        self.all_classes = []
        self.vars = {name: HeadlessVar(value=val) for name, val in {
            'axis': 'Data', 'mode': 'Individual', 'lower': '', 'upper': '', 'step': '1', 'axis_regex': '',
            'date_from': '', 'date_to': '', 'time_from': '', 'time_to': '', 'depth_min': '', 'depth_max': '',
            'attribute_regex': '', 'attribute_value': '', 'version': '', 'frame_a': '', 'frame_b': '', 'frame_c': '',
            'period': 'Diário', 'speed': '1 s', 'lunar_lat': '', 'lunar_lon': '', 'lunar_tz': '',
            'caption': 'Carregue a projeção do projeto para começar.', 'frame_info': '—',
            'projection': 'original', 'series_mode': 'Contagem'}.items()}
        self.points = HeadlessBoolVar(value=True)
        self.density = HeadlessBoolVar(value=True)
        self.density_common = HeadlessBoolVar(value=True)
        self.density_ref = {}
        self.contours = HeadlessBoolVar(value=False)
        self.compare_three = HeadlessBoolVar(value=False)
        self.export_status = HeadlessVar(value='')
        # Projection frame in use and every frame already computed (aligned to data rows).
        self.projection = 'original'
        self.coords = {}
        self.projection_note = ''
        self.load_token = 0
        # Counts per interval × class and diversity indices (series chart, CSV).
        self.counts = pd.DataFrame()
        self.diversity = pd.DataFrame()
        # Display names per column (iat_display_groups): only charts/exports change.
        self.group_aliases = empty_groups()
        # Background exports (GIF, one image per interval).
        self._export_busy = False
        self._export_queue = None
        self._export_kind = ''
        self._export_target = ''

    # Tk stand-ins used by the reused classic methods.
    def after(self, ms, func=None, *a):
        return self._ctl.after(ms, func, *a)

    def after_cancel(self, job):
        self._ctl.after_cancel(job)

    def winfo_exists(self):
        return True

    def pause(self):
        self.timer = None

    def axis_titles(self):
        return PROJECTION_AXES.get(self.projection, ('t-SNE 1', 't-SNE 2'))

    def set_data(self, frame):
        self.data = frame
        self.load_token += 1
        self.coords = {'original': (frame['x'].to_numpy(float).copy(), frame['y'].to_numpy(float).copy())}
        self.projection = 'original'
        self.vars['projection'].set('original')
        self.projection_note = ''

    def use_coords(self, method):
        x, y = self.coords[method]
        self.data = self.data.assign(x=x, y=y)
        self.projection = method

    def apply(self):
        """Same steps as the classic ``apply`` (filters, palette, intervals, bounds)."""
        if self.data.empty:
            self.vars['caption'].set('Clique em Carregar projeção primeiro.')
            return
        keys = ['date_from', 'date_to', 'time_from', 'time_to', 'depth_min', 'depth_max', 'attribute_value']
        params = {k: self.vars[k].get().strip() for k in keys}
        selected = list(self.selected_classes if self.selected_classes is not None else self.all_classes)
        self.view = filter_fixed(self.data, self.vars['axis'].get(), params, selected)
        if not self.view.empty:
            # Images without an embedding have no coordinates in a computed projection.
            finite = np.isfinite(self.view['x'].to_numpy(float)) & np.isfinite(self.view['y'].to_numpy(float))
            if not finite.all():
                self.view = self.view.loc[finite]
        self._group_view()
        if not self.view.empty:
            self.palette = make_class_palette(self.view['classe'].unique())
        use_lunar = self.vars['axis'].get() == 'Data' and self.vars['period'].get() == 'Lunar (astronômico)'
        site = (self.vars['lunar_lat'].get(), self.vars['lunar_lon'].get(), self.vars['lunar_tz'].get()) if use_lunar else None
        self.frames = intervals(self.view, self.vars['axis'].get(), self.vars['lower'].get(), self.vars['upper'].get(),
                                self.vars['step'].get(), self.vars['axis_regex'].get(),
                                period=self.vars['period'].get() if self.vars['axis'].get() == 'Data' else None,
                                lunar_site=site)
        self.current = 0
        labels = [lab for lab, _ in self.frames]
        self.vars['frame_a'].set(labels[0] if labels else '')
        self.vars['frame_b'].set(labels[-1] if labels else '')
        self.vars['frame_c'].set(labels[len(labels) // 2] if labels else '')
        self.bounds = None
        if not self.view.empty:
            x, y = self.view['x'], self.view['y']
            dx, dy = max(x.max() - x.min(), 1) * .06, max(y.max() - y.min(), 1) * .06
            self.bounds = (x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy)
        # One reference peak per class for every frame (common density scale).
        self.density_ref = density_reference(self.view, self.frames, self.bounds)
        self.counts = frame_class_counts(self.view, self.frames)
        self.diversity = diversity_indices(self.counts)
        self.vars['caption'].set(f'{len(self.view):,} imagens • {len(self.frames)} intervalos • '
                                 f'{PROJECTION_METHODS.get(self.projection, self.projection)} • dados intactos'
                                 .replace(',', '.'))
        self.version += 1

    def _group_view(self):
        """Display names of the label column (groups of classes); the original
        label stays in ``classe_orig`` for the Origem → Previsão panel."""
        if self.view is None or self.view.empty:
            return
        orig = self.view['classe_orig'] if 'classe_orig' in self.view.columns else self.view['classe']
        mapping = (self.group_aliases or {}).get('revisado') or {}
        if mapping:
            self.view = self.view.assign(classe_orig=orig, classe=map_names(orig, mapping))
        elif 'classe_orig' in self.view.columns:
            self.view = self.view.assign(classe=orig).drop(columns=['classe_orig'])

    def regroup(self):
        """Apply new display names keeping the intervals and the current frame."""
        if self.view is not None and not self.view.empty:
            self._group_view()
            self.palette = make_class_palette(self.view['classe'].unique())
            self.density_ref = density_reference(self.view, self.frames, self.bounds)
            self.counts = frame_class_counts(self.view, self.frames)
            self.diversity = diversity_indices(self.counts)
        self.version += 1

    def axis_changed(self):
        val = self.vars['axis'].get()
        self.vars['step'].set({'Data': '1', 'Hora': '60', 'Profundidade': '500', 'Atributo': '1'}[val])
        for v in ('lower', 'upper'):
            self.vars[v].set('')
        self.version += 1

    # ---- background exports (GIF and one image per interval) ----
    def start_background_export(self, kind, target, job):
        if self._export_busy:
            raise RuntimeError('Já há uma exportação em andamento.')
        self._export_busy = True
        self._export_kind = kind
        self._export_target = str(target)
        q = queue.Queue()
        self._export_queue = q

        def run():
            try:
                q.put(('done', job(lambda a, b: q.put(('progress', a, b)))))
            except Exception as exc:  # reported to the user by the poller
                q.put(('error', f'{type(exc).__name__}: {exc}'))
        threading.Thread(target=run, daemon=True, name='ilt-temporal-export').start()
        self.export_status.set(f'{kind}: iniciando…')
        self._ctl.after(200, self._poll_background_export)

    def _poll_background_export(self):
        q = self._export_queue
        finished = False
        while q is not None:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item[0] == 'progress':
                self.export_status.set(f'{self._export_kind}: {item[1]}/{item[2]} quadros')
            elif item[0] == 'done':
                self.export_status.set(f'{self._export_kind} salvo: {self._export_target}')
                NOTIFY.push('info', f'{self._export_kind} salvo', f'{item[1]}\n{self._export_target}')
                finished = True
            else:
                self.export_status.set(f'Erro ao exportar ({self._export_kind}).')
                NOTIFY.push('error', f'Exportar {self._export_kind}', item[1])
                finished = True
        if finished:
            self._export_busy = False
            self._export_queue = None
            _bump_iat()
        else:
            self._ctl.after(250, self._poll_background_export)


_STATE = None


def state(ctl):
    global _STATE
    if _STATE is None:
        _STATE = TemporalState(ctl)
    return _STATE


def _panel_traces(rows, bounds, settings, palette, xref, yref):
    """Same recipe as the classic ``_plot`` (``iat_temporal_figure``): 55-bin
    histogram, Gaussian σ=1.4, bands at 17/38/64 % of the reference peak (common
    to all frames by default) with opacities 0.22/0.37/0.56, optional outlines,
    points (≤4000 per class).  Density bands are composited into one
    transparent image per panel (no background fill)."""
    from scipy.ndimage import zoom
    from matplotlib.colors import to_rgba
    traces, image = [], None
    if rows.empty or not bounds:
        return traces, image
    xmin, xmax, ymin, ymax = bounds
    up = 4
    canvas = None
    a0, a1, a2 = DENSITY_ALPHAS
    for name, group in rows.groupby('classe', sort=True):
        color = palette.get(name, '#FFFFFF')
        xy = group[['x', 'y']].to_numpy(dtype=float)
        n = len(xy)
        dens = class_density(xy, bounds) if (n >= DENSITY_MIN_POINTS and settings.get('density', True)) else None
        if dens is not None:
            smooth, xedge, yedge = dens
            peak = float(smooth.max())
            bands = density_levels(peak, settings, name)
            if bands is not None:
                l0, l1, l2 = bands
                z = zoom(smooth.T, up, order=1)
                a = np.where(z >= l2, a2, np.where(z >= l1, a1, np.where(z >= l0, a0, 0.0)))
                r, g, b, _ = to_rgba(color)
                if canvas is None:
                    canvas = np.zeros(z.shape + (4,), float)
                rgb = np.array([r, g, b])
                canvas[..., :3] = rgb * a[..., None] + canvas[..., :3] * (1 - a[..., None])
                canvas[..., 3] = a + canvas[..., 3] * (1 - a)
                if settings.get('contours', False):
                    try:
                        import contourpy
                        xx = (xedge[:-1] + xedge[1:]) * .5
                        yy = (yedge[:-1] + yedge[1:]) * .5
                        gen = contourpy.contour_generator(xx, yy, smooth.T)
                        xs, ys = [], []
                        for level in (l0, l1, l2):
                            for line in gen.lines(level):
                                xs += list(line[:, 0]) + [None]
                                ys += list(line[:, 1]) + [None]
                        traces.append(dict(type='scatter', mode='lines', x=xs, y=ys, xaxis=xref, yaxis=yref,
                                           line=dict(color=color, width=.9), opacity=.92, hoverinfo='skip',
                                           showlegend=False))
                    except Exception:
                        pass
        if settings.get('points', True) or n < DENSITY_MIN_POINTS:
            if n > 4000:
                xy = xy[np.linspace(0, n - 1, 4000, dtype=int)]
            traces.append(dict(type='scattergl', mode='markers', x=xy[:, 0], y=xy[:, 1], xaxis=xref, yaxis=yref,
                               marker=dict(size=5, color=color, opacity=.83), name=str(name), legendgroup=str(name),
                               hovertemplate=f'{name}<extra></extra>', showlegend=False))
    if canvas is not None:
        from ..iat_plots import _png_data_uri
        image = dict(source=_png_data_uri(canvas[::-1]), xref=xref, yref=yref, x=xmin, y=ymax,
                     sizex=xmax - xmin, sizey=ymax - ymin, sizing='stretch', layer='below', opacity=1)
    return traces, image


def figure(st, theme):
    panels = st._display_panels() if st.frames else []
    n = max(1, len(panels))
    xlabel, ylabel = st.axis_titles()
    layout = plot_layout(theme, margin=dict(l=46, r=10, t=44, b=40), showlegend=True,
                         legend=dict(orientation='v', x=1.0, xanchor='left', y=1, yanchor='top',
                                     font=dict(size=11, color='#e8ecef'), bgcolor='rgba(5,8,10,.55)'))
    data, images, annotations = [], [], []
    right = 0.84
    gap = 0.045
    width = (right - gap * (n - 1)) / n
    settings = st._settings()
    for i in range(n):
        ax = '' if i == 0 else str(i + 1)
        x0 = i * (width + gap)
        layout[f'xaxis{ax}'] = dict(domain=[x0, x0 + width], range=[st.bounds[0], st.bounds[1]] if st.bounds else None,
                                    gridcolor='rgba(255,255,255,.08)', zeroline=False, title=dict(text=xlabel),
                                    tickfont=dict(size=10, color=STATUS['muted']), showline=True,
                                    linecolor='rgba(255,255,255,.18)', mirror=True)
        layout[f'yaxis{ax}'] = dict(domain=[0, 1], range=[st.bounds[2], st.bounds[3]] if st.bounds else None,
                                    anchor=f'x{ax}', gridcolor='rgba(255,255,255,.08)', zeroline=False,
                                    title=dict(text=ylabel if i == 0 else ''), showline=True,
                                    linecolor='rgba(255,255,255,.18)', mirror=True,
                                    tickfont=dict(size=10, color=STATUS['muted']))
        if i < len(panels):
            title, rows = panels[i]
            traces, image = _panel_traces(rows, st.bounds, settings, st.palette, f'x{ax}', f'y{ax}')
            data += traces
            if image:
                images.append(image)
            annotations.append(dict(text=f'<b>{title}</b>', x=x0 + width / 2, y=1.04, xref='paper', yref='paper',
                                    showarrow=False, yanchor='bottom', font=dict(size=13, color='#ffffff')))
            if rows.empty:
                annotations.append(dict(text='Sem imagens nesta faixa', x=x0 + width / 2, y=.5, xref='paper',
                                        yref='paper', showarrow=False, font=dict(color=STATUS['muted'])))
    shown = sorted({c for _t, rows in panels for c in rows['classe'].unique()}) if panels else []
    for cl in shown:
        data.append(dict(type='scattergl', mode='markers', x=[None], y=[None], name=str(cl),
                         marker=dict(size=9, color=st.palette.get(cl, '#fff')), showlegend=True))
    layout['annotations'] = annotations
    layout['images'] = images
    return {'data': data, 'layout': layout}


def _short(label, n=16):
    label = str(label)
    return label if len(label) <= n else label[:n - 1] + '…'


def series_figure(st, theme):
    """Evolution across the intervals (item 42): images per class, composition
    (%) or diversity (S, Hill N1 and N2).  One y axis per view."""
    counts = st.counts
    layout = plot_layout(theme, margin=dict(l=56, r=12, t=8, b=64), showlegend=True, hovermode='x unified',
                         legend=dict(orientation='h', x=0, y=1.02, yanchor='bottom',
                                     font=dict(size=10, color='#d6dce0'), bgcolor='rgba(0,0,0,0)'))
    if counts is None or counts.empty:
        layout['annotations'] = [dict(text='Gere os intervalos para ver a evolução por classe.', x=.5, y=.5,
                                      xref='paper', yref='paper', showarrow=False,
                                      font=dict(color=STATUS['muted'], size=12))]
        return {'data': [], 'layout': layout}
    labels = [str(i) for i in counts.index]
    x = list(range(1, len(labels) + 1))
    step = max(1, int(np.ceil(len(labels) / 14)))
    layout['xaxis'] = dict(layout['xaxis'], tickvals=x[::step], ticktext=[_short(l) for l in labels][::step],
                           tickangle=-30, title=dict(text='intervalo'), range=[0.5, len(labels) + 0.5])
    mode = st.vars['series_mode'].get()
    data = []
    if mode == 'Diversidade':
        div = st.diversity
        for col, name, color, dash in (('S', 'Riqueza S', '#e8ecef', 'dot'),
                                       ('N1', 'Hill N1 = exp(H′)', '#22d3ee', 'solid'),
                                       ('N2', 'Hill N2 = 1/Σp²', '#a78bfa', 'solid')):
            data.append(dict(type='scatter', mode='lines+markers', x=x, y=div[col].astype(float).tolist(), name=name,
                             customdata=labels, line=dict(color=color, width=2, dash=dash), marker=dict(size=5),
                             hovertemplate=f'{name}: %{{y:.2f}}<extra></extra>'))
        layout['yaxis'] = dict(layout['yaxis'], title=dict(text='nº (efetivo) de classes'), rangemode='tozero')
    elif mode == 'Proporção (%)':
        totals = counts.sum(axis=1).replace(0, np.nan)
        pct = counts.div(totals, axis=0).fillna(0.0) * 100.0
        for cl in counts.columns:
            color = st.palette.get(cl, '#9aa5ad')
            data.append(dict(type='scatter', mode='lines', x=x, y=pct[cl].round(3).tolist(), name=str(cl),
                             stackgroup='one', line=dict(width=.6, color=color), fillcolor=hex_alpha(color, .62),
                             hovertemplate=f'{cl}: %{{y:.1f}}%<extra></extra>'))
        layout['yaxis'] = dict(layout['yaxis'], title=dict(text='% das imagens do intervalo'), range=[0, 100],
                               ticksuffix='%')
    else:
        for cl in counts.columns:
            color = st.palette.get(cl, '#9aa5ad')
            data.append(dict(type='scatter', mode='lines+markers', x=x, y=counts[cl].astype(int).tolist(),
                             name=str(cl), line=dict(color=color, width=2), marker=dict(size=5),
                             hovertemplate=f'{cl}: %{{y:,}}<extra></extra>'))
        layout['yaxis'] = dict(layout['yaxis'], title=dict(text='imagens'), rangemode='tozero')
    # Hover header: the full interval label.
    data.append(dict(type='scatter', mode='markers', x=x, y=[None] * len(x), customdata=labels, showlegend=False,
                     hovertemplate='<b>%{customdata}</b><extra></extra>', marker=dict(opacity=0)))
    shapes, notes = [], []
    if st.vars['mode'].get() == 'Comparar':
        marks = [('A', st.vars['frame_a'].get()), ('B', st.vars['frame_b'].get())]
        if st.compare_three.get():
            marks.append(('C', st.vars['frame_c'].get()))
        for letter, lab in marks:
            if lab in labels:
                xi = labels.index(lab) + 1
                shapes.append(dict(type='line', xref='x', yref='paper', x0=xi, x1=xi, y0=0, y1=1,
                                   line=dict(color=STATUS['warn'], width=1.4, dash='dash')))
                notes.append(dict(text=letter, x=xi, y=1, xref='x', yref='paper', showarrow=False, yanchor='bottom',
                                  font=dict(color=STATUS['warn'], size=12)))
    elif st.frames:
        xi = st.current + 1
        shapes.append(dict(type='line', xref='x', yref='paper', x0=xi, x1=xi, y0=0, y1=1,
                           line=dict(color='#ffffff', width=1.2, dash='dot')))
    layout['shapes'] = shapes
    layout['annotations'] = notes
    return {'data': data, 'layout': layout}


def stats_text(st):
    panels = st._display_panels() if st.frames else []
    lines = []
    div = st.diversity if st.diversity is not None else pd.DataFrame()
    for title, rows in panels:
        total = len(rows)
        lines.extend((str(title), f'{total:,} imagens'.replace(',', '.')))
        if str(title) in div.index:
            d = div.loc[str(title)]
            fmt = lambda v, p: '—' if pd.isna(v) else f'{v:.{p}f}'
            lines.append(f"S = {int(d['S'])} • H′ = {fmt(d['H'], 3)} • N1 = {fmt(d['N1'], 2)} • "
                         f"N2 = {fmt(d['N2'], 2)} • J = {fmt(d['J'], 3)}")
        lines.append('')
        for cl, n in rows['classe'].value_counts().items():
            lines.append(f'{cl}\n  {n:,}  ({100 * n / total if total else 0:.1f}%)'.replace(',', '.'))
        lines.append('')
    return '\n'.join(lines) or 'Nenhum dado nos filtros.'


def tv(key):
    return {'type': 'tv', 'key': key}


def t_input(st, key, **kw):
    return dcc.Input(id=tv(key), value=st.vars[key].get(), debounce=True, className='inp', **kw)


def render(iat, lang, theme):
    from .iat import iact
    st = state(iat._ctl)
    axis = st.vars['axis'].get()
    mode = st.vars['mode'].get()
    labels = [lab for lab, _ in st.frames]
    selected = st.selected_classes if st.selected_classes is not None else st.all_classes

    def line(title, key):
        return html.Div([html.Label(title, className='lbl'), t_input(st, key)])

    chosen = st.vars['projection'].get()
    proj_opts = [{'label': lbl + ('  ✓' if k in st.coords and k != 'original' else ''), 'value': k}
                 for k, lbl in PROJECTION_METHODS.items()]
    projection_box = [
        html.Label('Projeção (coordenadas)', className='lbl'),
        dcc.Dropdown(id=tv('projection'), options=proj_opts, value=chosen, clearable=False, className='dd'),
        html.Div([C.btn('Usar / calcular projeção', iact('tp_project'), 'sm pri'),
                  html.Span(f'Em uso: {PROJECTION_METHODS.get(st.projection, st.projection)}', className='hint')],
                 className='row'),
        C.hint(PROJECTION_TIPS.get(chosen, '') + ('' if chosen == 'original' or chosen in st.coords else
                                                  ' Calculada a partir dos embeddings de todas as imagens carregadas; '
                                                  'fica em cache na pasta projections/ do projeto.')),
        html.Div(st.projection_note, className='accent-text small') if st.projection_note else None,
    ]
    axis_box = [
        html.Label('Eixo de evolução', className='lbl'),
        C.seg(tv('axis'), [(a, a) for a in AXES], axis),
        C.hint(AXIS_TIPS[axis]),
    ]
    if axis == 'Atributo':
        axis_box += [line('Prefixo do atributo no nome (ex.: estacao)', 'axis_regex'),
                     C.hint('Ex.: estacaoA, estacaoB. Não precisa escrever regex.')]
    else:
        unit = {'Data': 'AAAA-MM-DD [HH:MM]', 'Hora': 'HH:MM', 'Profundidade': 'mm'}[axis]
        axis_box += [line(f'Intervalo inicial ({unit})', 'lower'), line(f'Intervalo final ({unit})', 'upper')]
        if axis == 'Data':
            axis_box += [html.Label('Unidade do intervalo (calendário)', className='lbl'),
                         dcc.Dropdown(id=tv('period'), options=PERIODS, value=st.vars['period'].get(), clearable=False,
                                      className='dd')]
            if st.vars['period'].get() == 'Lunar (astronômico)':
                axis_box += [html.Div([
                    html.Div('Local da coleta • obrigatório no ciclo lunar', className='kicker'),
                    line('Latitude (graus; Sul é negativo)', 'lunar_lat'),
                    line('Longitude (graus; Oeste é negativo)', 'lunar_lon'),
                    line('Fuso do horário no nome (UTC; ex.: -3)', 'lunar_tz'),
                    C.hint('Ciclos reais Lua nova → Lua nova. Informe local e fuso UTC das imagens.'),
                ], className='subcard')]
        step_label = {'Data': 'Passo — quantidade de unidades', 'Hora': 'Passo (minutos)',
                      'Profundidade': 'Passo (mm)'}[axis]
        axis_box += [line(step_label, 'step')]
    filters = []
    for kind, fields, tip in FILTERS:
        if kind == axis:
            continue
        filters.append(html.Div([html.B(kind, className='small'), *[line(t_, k) for t_, k in fields], C.hint(tip)]))
    controls = C.hud('Análise Temporal e Espacial', [
        C.btn('⟳ Carregar projeção', iact('tp_load'), 'pri'),
        *projection_box,
        *axis_box,
        html.Label('Filtros adicionais — opcionais', className='lbl'),
        C.hint('Mantêm o mesmo recorte em todos os quadros. Deixe vazio para incluir todos os valores.'),
        *filters,
        html.Label('Classes de interesse', className='lbl'),
        html.Div(dcc.Checklist(id='tp-classes', options=st.all_classes, value=list(selected), className='clist',
                               inputClassName='chk', labelClassName='clist-item'), className='clist-box'),
        html.Div([C.btn('Todas', iact('tp_all'), 'sm'), C.btn('Nenhuma', iact('tp_none'), 'sm')], className='row'),
        html.Label('Grupos e nomes nos gráficos', className='lbl'),
        html.Div([C.btn('Agrupar / renomear classes…', iact('tp_groups'), 'sm'),
                  html.Span(f'{count_renamed(st.group_aliases)} nome(s) alterado(s)'
                            if count_renamed(st.group_aliases) else 'nomes originais', className='hint')],
                 className='row'),
        C.hint('Une ou renomeia classes em todos os gráficos e exportações desta aba; os rótulos das imagens não '
               'mudam. As classes de interesse acima usam os nomes originais.'),
        html.Label('Gráfico', className='lbl'),
        C.switch(tv('points'), 'Pontos individuais', st.points.get()),
        C.switch(tv('density'), 'Manchas (≥ 20 imagens)', st.density.get()),
        C.switch(tv('density_common'), 'Escala de densidade comum a todos os quadros', st.density_common.get()),
        C.hint('Ligada: as faixas de cada classe são relativas ao quadro mais denso, então quadros podem ser comparados. '
               'Desligada: cada quadro usa o próprio pico (um quadro com 20 imagens parece tão denso quanto um com 2.000).'),
        C.switch(tv('contours'), 'Contornos das manchas', st.contours.get()),
        C.btn('Aplicar e gerar intervalos', iact('tp_apply'), 'pri'),
    ], class_name='ctrl-col')

    info = f'{st.current + 1}/{len(st.frames)} • {st.frames[st.current][0]}' if st.frames else '—'
    compare = None
    if mode == 'Comparar' and labels:
        def fsel(letter, key):
            return html.Div([html.B(f'{letter}:', className='small'),
                             dcc.Dropdown(id=tv(key), options=labels, value=st.vars[key].get(), clearable=False,
                                          className='dd grow')], className='row nowrap')
        three = st.compare_three.get()
        compare = html.Div([
            html.Div([fsel('A', 'frame_a'), fsel('B', 'frame_b'), fsel('C', 'frame_c') if three else None],
                     className='g', style={'gridTemplateColumns': f'repeat({3 if three else 2}, minmax(0,1fr))'}),
            C.switch(tv('compare_three'), 'Incluir terceiro intervalo (C)', three),
        ], className='stack')
    transport = html.Div([
        C.btn('◀', iact('tp_prev'), 'sm icon'),
        C.btn('■' if st.timer else '▶', iact('tp_play'), 'sm icon pri'),
        C.btn('▶|', iact('tp_next'), 'sm icon'),
        dcc.Slider(id='tp-slider', min=0, max=max(len(st.frames) - 1, 0), step=1, value=st.current,
                   marks=None, tooltip={'always_visible': False}, className='grow slider'),
        html.Span('Velocidade', className='hint'),
        dcc.Dropdown(id=tv('speed'), options=list(SPEEDS), value=st.vars['speed'].get(), clearable=False,
                     className='dd', style={'width': '90px'}),
    ], className='row nowrap') if st.frames else None
    plot = C.hud(f'Projeção: {PROJECTION_METHODS.get(st.projection, st.projection)} • mesmo referencial e eixos '
                 'para todos os quadros', [
        html.Div([C.seg(tv('mode'), [(m, m) for m in MODES], mode),
                  html.Span(info, id='tp-info', className='hint push')], className='row'),
        C.hint('Compare somente batches com o mesmo referencial de projeção e versão de rede.'),
        compare,
        dcc.Graph(id='tp-graph', figure=figure(st, theme), config=GRAPH_CONFIG,
                  style={'height': '62vh', 'minHeight': '420px'}),
        transport,
        dcc.Interval(id='tp-timer', interval=SPEEDS.get(st.vars['speed'].get(), 1000), disabled=not st.timer),
        html.Div([C.btn('Salvar figura…', iact('tp_export_fig'), 'pri'),
                  C.btn('Exportar animação (GIF)…', iact('tp_export_gif'), disabled=st._export_busy),
                  C.btn('Exportar contagens (CSV)', iact('tp_export_counts')),
                  html.Span(st.export_status.get(), className='hint')], className='row'),
        html.Div(st.vars['caption'].get(), className='accent-text small'),
    ], meta=f'{len(st.frames)} intervalos' if st.frames else None)
    series = C.hud('Evolução entre intervalos', [
        html.Div([C.seg(tv('series_mode'), [(m, m) for m in SERIES_MODES], st.vars['series_mode'].get()),
                  C.hint('Clique em um ponto para ir ao intervalo.', class_name='push')], className='row'),
        dcc.Graph(id='tp-series', figure=series_figure(st, theme), config=dict(GRAPH_CONFIG, displayModeBar=False),
                  style={'height': '300px'}),
        C.hint('Contagens de imagens (não abundância normalizada por esforço). Diversidade: S = classes presentes; '
               'N1 = exp(H′) e N2 = 1/Σp² são números de Hill (nº efetivo de classes); J = H′/ln S na tabela e no CSV.'),
    ]) if st.frames else None
    summary = C.hud('Resumo por classe', html.Pre(stats_text(st), className='pre-block'), variant='alt')
    origin = _origin.panel(st, iat, theme)  # "Origem → Previsão" (alluvial / Sankey)
    return html.Div([controls, html.Div([plot, series, origin, summary], className='stack')], className='g iat-2col')


# ---------------- export dialog ----------------
# Text sizes, colours and fonts map 1:1 to iat_temporal_figure.export_style();
# sizes are typed in cm (the renderer works in inches).
_FONT_WANTED = ['DejaVu Sans', 'DejaVu Serif', 'Liberation Sans', 'Liberation Serif', 'Arial', 'Helvetica',
                'Times New Roman', 'Calibri', 'Cambria', 'Georgia', 'Verdana', 'Noto Sans', 'Noto Serif', 'Ubuntu',
                'Open Sans', 'Roboto', 'Lato', 'Carlito', 'Caladea', 'DejaVu Sans Mono']
_FONTS = None


def font_options():
    """Font families matplotlib can really use here (no silent fallback)."""
    global _FONTS
    if _FONTS is None:
        try:
            from matplotlib import font_manager as fm
            have = {f.name for f in fm.fontManager.ttflist}
        except Exception:
            have = set()
        _FONTS = [f for f in _FONT_WANTED if f in have] or ['DejaVu Sans']
        if 'DejaVu Sans' not in _FONTS:
            _FONTS.insert(0, 'DejaVu Sans')
    return _FONTS


SIZE_PRESETS = {
    'default': ('Padrão do ILT (pelo nº de painéis)', None, None),
    'auto': ('Automático — menor tamanho em que tudo cabe', 10.0, 10.0),
    'slide169': ('Apresentação 16:9 (33,9 × 19,1 cm)', 33.87, 19.05),
    'a4l': ('A4 paisagem (29,7 × 21,0 cm)', 29.7, 21.0),
    'a3l': ('A3 paisagem (42,0 × 29,7 cm)', 42.0, 29.7),
    'poster': ('Pôster (60 × 40 cm)', 60.0, 40.0),
    'custom': ('Personalizado', None, None),
}
THEME_PRESETS = {
    'light': ('Claro — fundo branco, texto escuro', '#ffffff', '#131b2a'),
    'dark': ('Escuro — fundo preto, texto claro', '#0b0b0d', '#f4f7f8'),
    'custom': ('Personalizado', None, None),
}
COLOR_KEYS = ('title_color', 'axis_color', 'tick_color', 'legend_color', 'legend_title_color')
FAMILY_KEYS = ('title_family', 'axis_family', 'tick_family', 'legend_family')
SIZE_KEYS = ('title_size', 'axis_size', 'tick_size', 'legend_size', 'legend_title_size')
FIG_FORMATS = [('png', 'PNG (raster)'), ('tiff', 'TIFF LZW (raster, revistas)'), ('jpg', 'JPEG (raster)'),
               ('svg', 'SVG (vetorial)'), ('pdf', 'PDF (vetorial)')]


def _fx(key):
    return {'type': 'tpx', 'key': key}


def _default_size_in(st):
    n = max(1, len(st._display_panels()))
    return {1: 16.5, 2: 20.0, 3: 25.0}.get(n, 16.5), 8.5


def _export_folder(iat):
    if iat is not None and getattr(iat, 'project', None) is not None:
        return str(Path(iat.project.project_dir) / 'exports' / 'temporal')
    return str(Path(iat._ctl.project_root) / 'output') if iat is not None else str(Path.home())


def _default_form(st, gif, folder):
    base = export_style()
    w_in, h_in = _default_size_in(st)
    xl, yl = st.axis_titles()
    mode = {'Individual': 'quadro', 'Comparar': 'comparacao', 'Animação': 'animacao'}.get(st.vars['mode'].get(), 'fig')
    form = {
        'title_template': base['title_template'], 'panel_title_a': '', 'panel_title_b': '', 'panel_title_c': '',
        'xlabel': xl, 'ylabel': yl, 'legend_title': base['legend_title'],
        'font_all': base['title_family'], **{k: base[k] for k in FAMILY_KEYS},
        **{k: base[k] for k in SIZE_KEYS},
        'theme_preset': 'light', 'background': base['background'].lower(),
        **{k: base[k].lower() for k in COLOR_KEYS},
        'size_preset': 'default', 'width_cm': round(w_in * 2.54, 1), 'height_cm': round(h_in * 2.54, 1),
        'auto_expand': ['on'], 'trim': ['on'],
        'fmt': 'gif' if gif else 'png', 'dpi': 100 if gif else 300, 'transparent': [],
        'scope': 'shown', 'folder': folder,
        'filename': f'temporal_{mode}_{datetime.now():%Y%m%d_%H%M}',
        'gif_duration': SPEEDS.get(st.vars['speed'].get(), 1000) / 1000.0,
    }
    return form


def _on(value):
    return bool(value) and (value is True or 'on' in (value if isinstance(value, (list, tuple)) else [value]))


def _num(value, name, lo, hi):
    try:
        out = float(str(value).replace(',', '.'))
    except (TypeError, ValueError):
        raise ValueError(f'{name}: informe um número.') from None
    if not lo <= out <= hi:
        raise ValueError(f'{name}: use um valor entre {lo:g} e {hi:g}.')
    return out


def _style_from(form, aliases, gif):
    """Validate the editor form and build the render_figure style (+ output options)."""
    from matplotlib.colors import is_color_like
    style = export_style()
    for key in ('title_template', 'panel_title_a', 'panel_title_b', 'panel_title_c', 'xlabel', 'ylabel', 'legend_title'):
        style[key] = str(form.get(key) or '').strip()
    if not style['title_template']:
        style['title_template'] = '{intervalo}'
    fonts = font_options()
    for key in FAMILY_KEYS:
        fam = str(form.get(key) or '')
        style[key] = fam if fam in fonts else 'DejaVu Sans'
    labels = {'title_size': 'Tamanho do título', 'axis_size': 'Tamanho dos títulos dos eixos',
              'tick_size': 'Tamanho dos números', 'legend_size': 'Tamanho da legenda',
              'legend_title_size': 'Tamanho do título da legenda'}
    for key in SIZE_KEYS:
        style[key] = _num(form.get(key), labels[key], 6, 36)
    for key in ('background',) + COLOR_KEYS:
        val = str(form.get(key) or '').strip()
        if not is_color_like(val):
            raise ValueError(f'Cor inválida em {key}.')
        style[key] = val.upper()
    style['figure_width'] = _num(form.get('width_cm'), 'Largura (cm)', 10, 137) / 2.54
    style['figure_height'] = _num(form.get('height_cm'), 'Altura (cm)', 10, 137) / 2.54
    style['auto_expand'] = _on(form.get('auto_expand'))
    style['trim'] = _on(form.get('trim'))
    style['dpi'] = _num(form.get('dpi'), 'Resolução (dpi)', 30, 300 if gif else 1200)
    style['transparent'] = _on(form.get('transparent')) and not gif
    clean = {}
    for name, new in (aliases or {}).items():
        new = str(new or '').strip()
        if not new:
            raise ValueError(f'Informe um nome para a classe {name}.')
        if new != name:
            clean[name] = new
    style['legend_names'] = clean
    scope = 'frames' if (not gif and form.get('scope') == 'frames') else 'shown'
    if gif or scope == 'frames':
        # Every frame is titled by its own interval.
        style['panel_title_a'] = style['panel_title_b'] = style['panel_title_c'] = ''
    out = {
        'fmt': 'gif' if gif else (str(form.get('fmt') or 'png') if str(form.get('fmt')) in dict(FIG_FORMATS) else 'png'),
        'scope': scope,
        'folder': str(form.get('folder') or '').strip(),
        'filename': str(form.get('filename') or '').strip(),
        'gif_duration': _num(form.get('gif_duration'), 'Duração de cada quadro (s)', 0.1, 10) if gif else None,
    }
    return style, out


def _preview_panels(st, gif, scope):
    if (gif or scope == 'frames') and st.frames:
        label, mask = st.frames[st.current]
        return [(label, st.view.loc[mask.fillna(False)])]
    return st._display_panels()


_PREVIEWS = {}


def _render_preview(st, style, panels):
    """Same renderer and size as the file; low dpi so the page stays fast."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(style['figure_width'], style['figure_height']), dpi=100, facecolor=style['background'])
    render_figure(fig, panels, st.bounds, st._settings(), st.palette, style=style)
    w_in, h_in = fig.get_size_inches()
    pdpi = max(30.0, min(110.0, 1150.0 / max(w_in, 1e-6)))
    kw = _savefig_kwargs('preview.png', dict(style, dpi=pdpi, transparent=False))
    buf = io.BytesIO()
    fig.savefig(buf, format='png', **kw)
    from PIL import Image
    buf.seek(0)
    with Image.open(buf) as im:
        px_w, px_h = im.size
    token = uuid.uuid4().hex
    _PREVIEWS.clear()
    _PREVIEWS[token] = buf.getvalue()
    out_w, out_h = px_w / pdpi, px_h / pdpi
    return f'/tp-preview/{token}.png', (w_in, h_in), (out_w, out_h)


def _status_text(style, out, requested, final):
    w, h = final
    dpi = style['dpi']
    msg = (f"Arquivo: {w * 2.54:.1f} × {h * 2.54:.1f} cm ({w:.2f} × {h:.2f} pol) • "
           f"{round(w * dpi):,} × {round(h * dpi):,} px a {dpi:g} dpi").replace(',', '.')
    rw, rh = style['figure_width'], style['figure_height']
    if requested[0] > rw + 0.01 or requested[1] > rh + 0.01:
        msg += ' • ampliada automaticamente para caber títulos e legenda'
    if out['fmt'] in ('svg', 'pdf'):
        msg += f' • vetorial (nuvens de pontos rasterizadas a {dpi:g} dpi)'
    if out['scope'] == 'frames':
        msg += ' • prévia do intervalo atual; um arquivo por intervalo, todos do mesmo tamanho'
    return msg


def _row(label, control, hint=None):
    return html.Div([html.Label(label, className='hint'), control,
                     html.Div(hint, className='hint small') if hint else None], className='xf-row')


def _text_in(form, key, placeholder=''):
    return dcc.Input(id=_fx(key), value=form.get(key, ''), debounce=0.7, className='inp sm', placeholder=placeholder)


def _num_in(form, key, lo, hi, step):
    return dcc.Input(id=_fx(key), type='number', min=lo, max=hi, step=step, value=form.get(key),
                     debounce=0.7, className='inp sm')


def _color_in(form, key):
    return dcc.Input(id=_fx(key), type='color', value=form.get(key, '#000000'), debounce=0.5, className='color-in')


def _dd(form, key, options):
    return dcc.Dropdown(id=_fx(key), options=[{'label': lbl, 'value': val} for val, lbl in options],
                        value=form.get(key), clearable=False, className='dd')


def _sw(form, key, label):
    return dcc.Checklist(id=_fx(key), value=form.get(key) or [], options=[{'label': label, 'value': 'on'}],
                         className='sw', inputClassName='sw-in', labelClassName='sw-lb')


def export_modal(st, gif, form, aliases=None, message=''):
    classes = sorted({c for _t, rows in _preview_panels(st, gif, form.get('scope'))
                      for c in rows['classe'].unique()} | (set(st.view['classe'].unique()) if gif and not st.view.empty
                                                            else set()))
    aliases = aliases or {c: c for c in classes}
    preview, status = html.Div('—', className='hint'), message
    try:
        style, out = _style_from(form, aliases, gif)
        url, requested, final = _render_preview(st, style, _preview_panels(st, gif, out['scope']))
        preview = html.A(html.Img(id='tpx-img', src=url, className='xf-img'), href=url, target='_blank',
                         title='Abrir a prévia em outra aba')
        status = status or _status_text(style, out, requested, final)
    except (ValueError, RuntimeError) as exc:
        preview = html.Img(id='tpx-img', src='', className='xf-img')
        status = str(exc)
    fonts = [(f, f) for f in font_options()]
    fmt_opts = [('gif', 'GIF animado')] if gif else FIG_FORMATS
    file_box = html.Div([
        html.Div('Arquivo', className='kicker'),
        _row('Formato', _dd(form, 'fmt', fmt_opts)),
        _row('Resolução (dpi)', _num_in(form, 'dpi', 30, 300 if gif else 1200, 10),
             'GIF: 60–120 dpi deixa o arquivo leve. Revistas: 300 dpi (fotos) a 600 dpi (gráficos).'),
        None if gif else _row('O que salvar', _dd(form, 'scope', [
            ('shown', 'O que está na tela (quadro atual ou comparação)'),
            ('frames', 'Todos os intervalos — um arquivo por intervalo')])),
        _row('Duração de cada quadro (s)', _num_in(form, 'gif_duration', 0.1, 10, 0.1)) if gif else None,
        None if gif else _sw(form, 'transparent', 'Fundo transparente (PNG, TIFF, SVG, PDF)'),
        _sw(form, 'trim', 'Recortar margens em branco'),
        _row('Pasta', html.Div([_text_in(form, 'folder'),
                                C.btn('…', 'tpx-browse', 'sm icon', title='Escolher pasta')], className='browse')),
        _row('Nome do arquivo (sem extensão)', _text_in(form, 'filename'),
             'Se já existir, um sufixo _2, _3… é acrescentado; nada é sobrescrito.'),
    ], className='subcard')
    size_box = html.Div([
        html.Div('Tamanho', className='kicker'),
        _row('Predefinição', _dd(form, 'size_preset', [(k, v[0]) for k, v in SIZE_PRESETS.items()])),
        html.Div([_row('Largura (cm)', _num_in(form, 'width_cm', 10, 137, 0.1)),
                  _row('Altura (cm)', _num_in(form, 'height_cm', 10, 137, 0.1))], className='g g-2'),
        _sw(form, 'auto_expand', 'Ampliar automaticamente se textos ou legenda não couberem'),
        C.hint('As letras nunca são reduzidas: se não couberem, a figura cresce (ligado) ou um aviso aparece '
               '(desligado).'),
    ], className='subcard')
    text_box = html.Div([
        html.Div('Textos', className='kicker'),
        _row('Título (use {intervalo} e {painel})', _text_in(form, 'title_template')),
        None if gif else html.Div([_row('Painel A', _text_in(form, 'panel_title_a', 'título padrão')),
                                   _row('Painel B', _text_in(form, 'panel_title_b', 'título padrão')),
                                   _row('Painel C', _text_in(form, 'panel_title_c', 'título padrão'))],
                                  className='g g-3'),
        html.Div([_row('Eixo horizontal', _text_in(form, 'xlabel')),
                  _row('Eixo vertical', _text_in(form, 'ylabel'))], className='g g-2'),
        _row('Título da legenda', _text_in(form, 'legend_title')),
    ], className='subcard')
    font_box = html.Div([
        html.Div('Fontes e tamanhos (pt)', className='kicker'),
        _row('Fonte de todos os textos', _dd(form, 'font_all', fonts),
             'Aplica a mesma família a título, eixos, números e legenda; ajuste abaixo se quiser diferenciar.'),
        html.Div([
            _row('Título', _dd(form, 'title_family', fonts)), _row('pt', _num_in(form, 'title_size', 6, 36, 0.5)),
            _row('Eixos', _dd(form, 'axis_family', fonts)), _row('pt', _num_in(form, 'axis_size', 6, 36, 0.5)),
            _row('Números', _dd(form, 'tick_family', fonts)), _row('pt', _num_in(form, 'tick_size', 6, 36, 0.5)),
            _row('Legenda', _dd(form, 'legend_family', fonts)), _row('pt', _num_in(form, 'legend_size', 6, 36, 0.5)),
            html.Div(), _row('Título da legenda (pt)', _num_in(form, 'legend_title_size', 6, 36, 0.5)),
        ], className='xf-fonts'),
    ], className='subcard')
    color_box = html.Div([
        html.Div('Cores', className='kicker'),
        _row('Tema', _dd(form, 'theme_preset', [(k, v[0]) for k, v in THEME_PRESETS.items()])),
        html.Div([
            _row('Fundo', _color_in(form, 'background')), _row('Título', _color_in(form, 'title_color')),
            _row('Eixos', _color_in(form, 'axis_color')), _row('Números', _color_in(form, 'tick_color')),
            _row('Legenda', _color_in(form, 'legend_color')),
            _row('Título da legenda', _color_in(form, 'legend_title_color')),
        ], className='xf-colors'),
        C.hint('As cores das classes são as mesmas da tela.'),
    ], className='subcard')
    alias_box = html.Div([
        html.Div('Nomes das classes na legenda', className='kicker'),
        C.hint('Só na figura exportada; não altera rótulos nem cores.'),
        html.Div([_row(c, dcc.Input(id={'type': 'tpa', 'key': c}, value=aliases.get(c, c), debounce=0.7,
                                    className='inp sm')) for c in classes], className='g g-2'),
    ], className='subcard') if classes else None
    body = html.Div([
        dcc.Store(id='tpx-gif', data=bool(gif)),
        html.Div([
            html.Div([html.Div(preview, className='xf-preview'),
                      html.Div(status, id='tpx-status', className='hint'),
                      C.hint('Clique na prévia para abri-la em outra aba. A prévia usa o mesmo desenho do arquivo; '
                             'só a resolução é menor.')], className='stack xf-left'),
            html.Div([file_box, size_box, text_box, font_box, color_box, alias_box], className='stack xf-right'),
        ], className='xf-grid'),
    ])
    title = 'Exportar animação (GIF)' if gif else 'Salvar figura'
    return C.modal_card(title, body, act('modal_close'), wide=True,
                        actions=[C.btn('Restaurar padrão', 'tpx-reset'),
                                 C.btn('Fechar', act('modal_close')),
                                 C.btn('Salvar', 'tpx-save', 'pri')])


def _unique_path(path):
    path = Path(path)
    if not path.exists():
        return path
    for i in range(2, 10000):
        cand = path.with_name(f'{path.stem}_{i}{path.suffix}')
        if not cand.exists():
            return cand
    return path.with_name(f'{path.stem}_{uuid.uuid4().hex[:6]}{path.suffix}')


def _safe_stem(text):
    stem = re.sub(r'[\\/:*?"<>|]+', '_', str(text or '').strip()).strip(' .')
    return stem[:120]


def _save_export(st, iat, form, aliases, gif):
    """Write the file(s); returns a message.  Raises ValueError for invalid input."""
    style, out = _style_from(form, aliases, gif)
    if not st.frames:
        raise ValueError('Primeiro carregue a projeção e gere os intervalos.')
    folder = Path(out['folder'] or _export_folder(iat)).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(out['filename']) or f'temporal_{datetime.now():%Y%m%d_%H%M%S}'
    settings = st._settings()
    palette = dict(st.palette)
    bounds = st.bounds
    if gif:
        path = _unique_path(folder / f'{stem}.gif')
        view = st.view
        frames = [(label, mask.copy()) for label, mask in st.frames]
        duration = int(round(out['gif_duration'] * 1000))
        st.start_background_export(
            'GIF', path, lambda progress: f"{save_animated_gif(str(path), view, frames, bounds, settings, palette, duration_ms=duration, style=style, progress=progress)} quadros")
        return f'Exportando GIF em segundo plano: {path}'
    if out['scope'] == 'frames':
        target = _unique_path(folder / stem)
        view = st.view
        frames = [(label, mask.copy()) for label, mask in st.frames]
        fmt = out['fmt']
        st.start_background_export(
            'Imagens por intervalo', target,
            lambda progress: f"{len(save_frame_images(target, stem, fmt, view, frames, bounds, settings, palette, style, progress))} arquivos")
        return f'Exportando um arquivo por intervalo em segundo plano: {target}'
    path = _unique_path(folder / f"{stem}.{out['fmt']}")
    save_figure(str(path), st._display_panels(), bounds, settings, palette, style)
    if iat is not None:
        iat._log(f'Figura salva: {path}', 'success')
    NOTIFY.push('info', 'Figura salva', str(path))
    return str(path)


def register(app, ctl, get_iat, respond):
    from flask import Response, abort

    @app.server.route('/tp-preview/<token>.png')
    def _tp_preview(token):
        data = _PREVIEWS.get(token)
        if data is None:
            abort(404)
        return Response(data, mimetype='image/png', headers={'Cache-Control': 'no-store'})

    def bump_and_respond(lang, theme):
        iat = get_iat()
        before = None  # force re-render of the temporal body
        iat.web_version = getattr(iat, 'web_version', 0) + 1
        return respond(iat, before, lang, theme)

    @app.callback(Output(dyn(ALL), 'children', allow_duplicate=True),
                  Input(tv(ALL), 'value'), State('lang-store', 'data'), State('theme-store', 'data'),
                  prevent_initial_call=True)
    def tp_var(_v, lang, theme):
        trig = ctx.triggered_id
        if not isinstance(trig, dict):
            raise PreventUpdate
        key, value = trig['key'], ctx.triggered[0]['value']
        st = state(ctl)
        with ctl.lock:
            if key in ('points', 'density', 'density_common', 'contours', 'compare_three'):
                getattr(st, key).set(bool(value))
            else:
                if st.vars[key].get() == (value or ''):
                    raise PreventUpdate
                st.vars[key].set(value or '')
                if key == 'axis':
                    st.axis_changed()
                if key == 'mode':
                    st.pause()
            if key in ('lower', 'upper', 'step', 'axis_regex', 'period', 'lunar_lat', 'lunar_lon', 'lunar_tz',
                       'date_from', 'date_to', 'time_from', 'time_to', 'depth_min', 'depth_max', 'attribute_value') \
                    and key not in ('axis', 'period'):
                raise PreventUpdate  # applied with "Aplicar e gerar intervalos", like the classic panel
            return bump_and_respond(lang, theme)

    @app.callback(Output(dyn(ALL), 'children', allow_duplicate=True),
                  Input('tp-classes', 'value'), prevent_initial_call=True)
    def tp_classes(value):
        st = state(ctl)
        with ctl.lock:
            st.selected_classes = list(value or [])
        raise PreventUpdate

    @app.callback(Output('tp-graph', 'figure', allow_duplicate=True), Output('tp-slider', 'value'),
                  Output('tp-series', 'figure', allow_duplicate=True), Output('tp-info', 'children'),
                  Input('tp-slider', 'value'), Input('tp-timer', 'n_intervals'), Input('tp-series', 'clickData'),
                  State('theme-store', 'data'), prevent_initial_call=True)
    def tp_frame(value, _n, click, theme):
        from ..theme import sanitize
        st = state(ctl)
        with ctl.lock:
            if not st.frames:
                raise PreventUpdate
            if ctx.triggered_id == 'tp-timer':
                if not st.timer or st.vars['mode'].get() != 'Animação':
                    raise PreventUpdate
                st.current = (st.current + 1) % len(st.frames)
            elif ctx.triggered_id == 'tp-series':
                pts = (click or {}).get('points') or []
                if not pts or st.vars['mode'].get() == 'Comparar':
                    raise PreventUpdate
                idx = int(round(float(pts[0].get('x', 1)))) - 1
                idx = max(0, min(len(st.frames) - 1, idx))
                if idx == st.current:
                    raise PreventUpdate
                st.current = idx
            else:
                idx = max(0, min(len(st.frames) - 1, int(value or 0)))
                if idx == st.current:
                    raise PreventUpdate
                st.current = idx
            theme = sanitize(theme or {})
            info = f'{st.current + 1}/{len(st.frames)} • {st.frames[st.current][0]}'
            return figure(st, theme), st.current, series_figure(st, theme), info

    # ---- export editor ----
    @app.callback(Output('tpx-img', 'src'), Output('tpx-status', 'children'),
                  Input({'type': 'tpx', 'key': ALL}, 'value'), Input({'type': 'tpa', 'key': ALL}, 'value'),
                  State('tpx-gif', 'data'), prevent_initial_call=True)
    def tp_export_preview(values, aliases, gif):
        st = state(ctl)
        form = {o['id']['key']: v for o, v in zip(ctx.inputs_list[0], values)}
        als = {o['id']['key']: v for o, v in zip(ctx.inputs_list[1], aliases)}
        with ctl.lock:
            sig = repr((sorted(form.items(), key=lambda kv: kv[0]), sorted(als.items()), bool(gif)))
            if getattr(st, '_export_sig', None) == sig:
                raise PreventUpdate  # re-inserted inputs, nothing changed
            st._export_sig = sig
            try:
                style, out = _style_from(form, als, bool(gif))
                url, requested, final = _render_preview(st, style, _preview_panels(st, bool(gif), out['scope']))
                return url, _status_text(style, out, requested, final)
            except (ValueError, RuntimeError) as exc:
                return no_update, f'⚠ {exc}'

    @app.callback(Output({'type': 'tpx', 'key': 'width_cm'}, 'value'),
                  Output({'type': 'tpx', 'key': 'height_cm'}, 'value'),
                  Output({'type': 'tpx', 'key': 'auto_expand'}, 'value'),
                  Input({'type': 'tpx', 'key': 'size_preset'}, 'value'), State('tpx-gif', 'data'),
                  prevent_initial_call=True)
    def tp_size_preset(preset, _gif):
        if preset not in SIZE_PRESETS or preset == 'custom':
            raise PreventUpdate
        st = state(ctl)
        if preset == 'default':
            w_in, h_in = _default_size_in(st)
            return round(w_in * 2.54, 1), round(h_in * 2.54, 1), ['on']
        _label, w, h = SIZE_PRESETS[preset]
        return w, h, ['on'] if preset == 'auto' else no_update

    @app.callback(*[Output({'type': 'tpx', 'key': k}, 'value') for k in ('background',) + COLOR_KEYS],
                  Input({'type': 'tpx', 'key': 'theme_preset'}, 'value'), prevent_initial_call=True)
    def tp_theme_preset(preset):
        if preset not in THEME_PRESETS or preset == 'custom':
            raise PreventUpdate
        _label, bg, ink = THEME_PRESETS[preset]
        return (bg,) + (ink,) * len(COLOR_KEYS)

    @app.callback(*[Output({'type': 'tpx', 'key': k}, 'value') for k in FAMILY_KEYS],
                  Input({'type': 'tpx', 'key': 'font_all'}, 'value'), prevent_initial_call=True)
    def tp_font_all(family):
        if family not in font_options():
            raise PreventUpdate
        return (family,) * len(FAMILY_KEYS)

    @app.callback(Output({'type': 'tpx', 'key': 'folder'}, 'value'),
                  Input('tpx-browse', 'n_clicks'), State({'type': 'tpx', 'key': 'folder'}, 'value'),
                  prevent_initial_call=True)
    def tp_browse(n, current):
        if not n:
            raise PreventUpdate
        chosen = FiledialogShim.askdirectory(initialdir=current or str(Path.home()), title='Pasta da exportação')
        if not chosen:
            raise PreventUpdate
        return chosen

    @app.callback(Output('modal-host', 'children', allow_duplicate=True),
                  Output('tpx-status', 'children', allow_duplicate=True),
                  Input('tpx-save', 'n_clicks'), State({'type': 'tpx', 'key': ALL}, 'value'),
                  State({'type': 'tpa', 'key': ALL}, 'value'), State('tpx-gif', 'data'), prevent_initial_call=True)
    def tp_export_save(n, values, aliases, gif):
        if not n:
            raise PreventUpdate
        form = {o['id']['key']: v for o, v in zip(ctx.states_list[0], values)}
        als = {o['id']['key']: v for o, v in zip(ctx.states_list[1], aliases)}
        st = state(ctl)
        iat = get_iat()
        with ctl.lock:
            try:
                msg = _save_export(st, iat, form, als, bool(gif))
            except (ValueError, RuntimeError, OSError) as exc:
                return no_update, f'⚠ {exc}'
            _bump_iat()
        return [], no_update

    @app.callback(Output('modal-host', 'children', allow_duplicate=True),
                  Input('tpx-reset', 'n_clicks'), State('tpx-gif', 'data'), prevent_initial_call=True)
    def tp_export_reset(n, gif):
        if not n:
            raise PreventUpdate
        st = state(ctl)
        iat = get_iat()
        with ctl.lock:
            st._export_sig = None
            return export_modal(st, bool(gif), _default_form(st, bool(gif), _export_folder(iat)))

    _origin.register(app, ctl, get_iat)
    _groups.register(app, ctl, get_iat)


def _tp_load(iat, lang):
    st = state(iat._ctl)
    if iat.project is None:
        NOTIFY.push('info', 'Análise Temporal', 'Carregue um projeto no IAT.')
        return
    project = iat.project
    st.vars['caption'].set('Carregando CSVs do projeto...')

    def done(frame):
        st.set_data(frame)
        st.group_aliases = load_groups(project.project_dir)  # saved display names of this project
        st.palette = make_class_palette(frame['classe'].unique())
        st.all_classes = sorted(frame['classe'].unique())
        if st.selected_classes is None:
            st.selected_classes = list(st.all_classes)
        else:
            st.selected_classes = [c for c in st.selected_classes if c in st.all_classes] or list(st.all_classes)
        versions = sorted(frame['versao'].unique())
        note = ' • atenção: coordenadas de versões diferentes podem não ser comparáveis' if len(versions) > 1 else ''
        st.vars['caption'].set(f'{len(frame):,} imagens • usando resultados atuais{note}'.replace(',', '.'))
        try:
            st.apply()
        except (ValueError, TypeError) as exc:
            NOTIFY.push('error', 'Análise temporal', str(exc))
        iat.web_version = getattr(iat, 'web_version', 0) + 1
    iat._run_operation('Carregando projeção temporal',
                       lambda progress: load_points(project, progress, with_origin=True), done)


def _tp_project(iat, lang):
    """Use the chosen projection; compute it in the background the first time."""
    st = state(iat._ctl)
    if st.data.empty or iat.project is None:
        NOTIFY.push('info', 'Análise Temporal', 'Clique em Carregar projeção primeiro.')
        return
    method = st.vars['projection'].get() or 'original'
    if method in st.coords:
        st.use_coords(method)
        st.projection_note = '' if method == 'original' else st.projection_note
        try:
            st.apply()
        except (ValueError, TypeError) as exc:
            NOTIFY.push('error', 'Análise temporal', str(exc))
        iat.web_version = getattr(iat, 'web_version', 0) + 1
        return
    project, data, token = iat.project, st.data, st.load_token

    def done(res):
        if token != st.load_token:
            return  # project reloaded meanwhile
        coords, note = res
        st.coords[method] = (coords[:, 0].copy(), coords[:, 1].copy())
        st.use_coords(method)
        st.projection_note = note
        try:
            st.apply()
        except (ValueError, TypeError) as exc:
            NOTIFY.push('error', 'Análise temporal', str(exc))
        iat.web_version = getattr(iat, 'web_version', 0) + 1
    iat._run_operation(f'Projeção: {PROJECTION_METHODS.get(method, method)}',
                       lambda progress: temporal_projection(project, data, method, progress), done)


def _tp_apply(iat, lang):
    st = state(iat._ctl)
    try:
        st.apply()
    except (ValueError, TypeError) as exc:
        NOTIFY.push('error', 'Análise temporal', str(exc))
    iat.web_version = getattr(iat, 'web_version', 0) + 1


def _tp_step(delta):
    def run(iat, lang):
        st = state(iat._ctl)
        if st.frames:
            st.current = (st.current + delta) % len(st.frames)
        iat.web_version = getattr(iat, 'web_version', 0) + 1
    return run


def _tp_play(iat, lang):
    st = state(iat._ctl)
    if st.timer:
        st.timer = None
    elif st.frames:
        st.vars['mode'].set('Animação')
        st.timer = 'web'
    iat.web_version = getattr(iat, 'web_version', 0) + 1


def _tp_classes(all_):
    def run(iat, lang):
        st = state(iat._ctl)
        st.selected_classes = list(st.all_classes) if all_ else []
        iat.web_version = getattr(iat, 'web_version', 0) + 1
    return run


def _tp_export(gif):
    def run(iat, lang):
        st = state(iat._ctl)
        if not st.frames:
            NOTIFY.push('info', 'Exportar', 'Primeiro carregue a projeção e gere os intervalos.')
            return
        if st._export_busy:
            NOTIFY.push('info', 'Exportar', 'Aguarde a exportação em andamento terminar.')
            return
        st._export_sig = None
        return {'modal': export_modal(st, gif, _default_form(st, gif, _export_folder(iat)))}
    return run


def _tp_export_counts(iat, lang):
    """Images per class and diversity per interval → CSV in <projeto>/exports/temporal/."""
    from ilt_contracts import atomic_write_csv
    st = state(iat._ctl)
    if st.counts is None or st.counts.empty:
        NOTIFY.push('info', 'Exportar contagens', 'Primeiro carregue a projeção e gere os intervalos.')
        return
    folder = Path(_export_folder(iat))
    folder.mkdir(parents=True, exist_ok=True)
    path = _unique_path(folder / f'temporal_contagens_{datetime.now():%Y%m%d_%H%M%S}.csv')
    div = st.diversity.rename(columns={'N': 'div_N_imagens', 'S': 'div_S_riqueza', 'H': 'div_H_shannon_nats',
                                       'N1': 'div_N1_hill1', 'N2': 'div_N2_hill2', 'J': 'div_J_pielou'})
    table = st.counts.join(div).reset_index()
    table.insert(1, 'ordem', range(1, len(table) + 1))
    table.insert(2, 'eixo', st.vars['axis'].get())
    table.insert(3, 'projecao', st.projection)
    atomic_write_csv(table, path, index=False)
    iat._log(f'Contagens por intervalo salvas: {path}', 'success')
    NOTIFY.push('info', 'Contagens exportadas', str(path))


ACTIONS = {
    'tp_load': _tp_load, 'tp_project': _tp_project, 'tp_apply': _tp_apply, 'tp_prev': _tp_step(-1),
    'tp_next': _tp_step(1), 'tp_play': _tp_play,
    'tp_all': _tp_classes(True), 'tp_none': _tp_classes(False),
    'tp_export_fig': _tp_export(False), 'tp_export_gif': _tp_export(True),
    'tp_export_counts': _tp_export_counts,
}
ACTIONS.update(_origin.ACTIONS)
ACTIONS.update(_groups.ACTIONS)
