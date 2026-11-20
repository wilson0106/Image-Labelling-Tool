#!/usr/bin/env python3
"""Visual theme of the web interface (tokens, presets, persistence).

Pure data + JSON persistence; no processing code depends on this module.
The theme is stored in ``input/ui_theme.json`` next to ``train_params.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_THEME = {
    'accent': '#22d3ee',
    'accent2': '#3b82f6',
    'bg': '#050607',
    'ink': '#f4f7f8',
    'glass': 0.55,
    'radius': 14,
    'font_display': 'Chakra Petch',
    'font_body': 'IBM Plex Sans',
    'font_scale': 1.0,
    'gpu_monitor': True,
}

PRESETS = [
    ('Ciano HUD', '#22d3ee', '#3b82f6'),
    ('Azul neon', '#3b82f6', '#a855f7'),
    ('Esmeralda', '#34d399', '#22d3ee'),
    ('Âmbar', '#fbbf24', '#f97316'),
    ('Magenta', '#f472b6', '#a855f7'),
    ('Grafite', '#d4d4d8', '#71717a'),
]

FONT_CHOICES = ['Chakra Petch', 'IBM Plex Sans', 'Lato', 'DejaVu Sans', 'Liberation Sans']

# Fixed class colours, validated for colour-vision deficiency (adjacent pairs,
# OKLab ΔE ≥ 11) and ≥ 3:1 contrast on the dark surface.  They never change
# with the interface theme so a class keeps its identity everywhere.
CLASS_PALETTE = ['#00A1C4', '#D97230', '#0F74C5', '#C5547C', '#AF8F00', '#8673D9', '#3EAB5E', '#BE67B7']

STATUS = {'ok': '#34d399', 'warn': '#fbbf24', 'bad': '#f87171', 'muted': '#a3adb5', 'dim': '#6b7680'}


def theme_path(project_root):
    return Path(project_root) / 'input' / 'ui_theme.json'


def _valid_hex(value):
    v = str(value or '')
    return len(v) == 7 and v.startswith('#') and all(c in '0123456789abcdefABCDEF' for c in v[1:])


def sanitize(theme):
    out = dict(DEFAULT_THEME)
    for key, default in DEFAULT_THEME.items():
        if key not in (theme or {}):
            continue
        value = theme[key]
        if isinstance(default, bool):
            out[key] = bool(value)
        elif isinstance(default, (int, float)) and not isinstance(default, bool):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if key == 'glass':
                value = min(0.95, max(0.15, value))
            elif key == 'radius':
                value = int(min(24, max(0, value)))
            elif key == 'font_scale':
                value = min(1.4, max(0.8, round(value, 2)))
            out[key] = value
        elif key in ('accent', 'accent2', 'bg', 'ink'):
            if _valid_hex(value):
                out[key] = value.lower()
        else:
            out[key] = str(value)
    return out


def load_theme(project_root):
    path = theme_path(project_root)
    try:
        return sanitize(json.loads(path.read_text(encoding='utf-8')))
    except Exception:
        return dict(DEFAULT_THEME)


def save_theme(project_root, theme):
    path = theme_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = sanitize(theme)
    path.write_text(json.dumps(clean, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return clean


def contrast_ratio(a, b):
    def lum(h):
        n = int(h[1:], 16)
        vals = []
        for v in ((n >> 16) & 255, (n >> 8) & 255, n & 255):
            v /= 255
            vals.append(v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4)
        return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]
    la, lb = lum(a), lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def hex_alpha(color, alpha):
    n = int(color[1:], 16)
    return f'rgba({(n >> 16) & 255},{(n >> 8) & 255},{n & 255},{alpha})'


def plot_layout(theme=None, **extra):
    """Base Plotly layout matching the HUD theme (transparent background)."""
    theme = theme or DEFAULT_THEME
    ink, muted = theme['ink'], STATUS['muted']
    axis = dict(gridcolor='rgba(255,255,255,.06)', zeroline=False, linecolor='rgba(255,255,255,.12)',
                tickfont=dict(color=muted, size=11, family='JetBrains Mono, DejaVu Sans Mono, monospace'),
                title=dict(font=dict(color=muted, size=11)))
    layout = dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=f"{theme['font_body']}, DejaVu Sans, sans-serif", color=ink, size=12),
        margin=dict(l=48, r=14, t=8, b=36),
        xaxis=dict(axis), yaxis=dict(axis),
        legend=dict(orientation='h', y=1.12, x=0, font=dict(color=muted, size=11), bgcolor='rgba(0,0,0,0)'),
        hoverlabel=dict(bgcolor='#0b1216', bordercolor=theme['accent'], font=dict(color=ink)),
        hovermode='closest',
    )
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(layout.get(key), dict):
            merged = dict(layout[key])
            merged.update(value)
            layout[key] = merged
        else:
            layout[key] = value
    return layout


GRAPH_CONFIG = {
    'displaylogo': False, 'responsive': True,
    'modeBarButtonsToRemove': ['select2d', 'autoScale2d', 'toggleSpikelines',
                               'hoverCompareCartesian', 'hoverClosestCartesian'],
    'toImageButtonOptions': {'format': 'png', 'scale': 2},
}
