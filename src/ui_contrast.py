#!/usr/bin/env python3
"""WCAG contrast helpers for the Tk interface.

The dark theme uses bright accent colours (green, blue, purple, amber) both as
text colours on dark panels and as button/badge backgrounds.  White text on
those bright backgrounds falls below the WCAG AA minimum (4.5:1), e.g. white on
the primary green is only 2.1:1.  These helpers pick a readable text colour for
any background and can repair a whole widget tree after it is built.

Only *light* text on a background that is too bright is changed; dark-on-dark
"muted"/"disabled" styling is intentionally left alone.
"""
from __future__ import annotations

from functools import lru_cache

DARK_INK = '#06101b'
LIGHT_INK = '#ffffff'
AA_NORMAL = 4.5

_NAMED = {
    'white': '#ffffff', 'black': '#000000',
}


def _parse_hex(color: str):
    c = _NAMED.get(str(color).strip().lower(), str(color).strip())
    if not c.startswith('#'):
        return None
    c = c[1:]
    if len(c) == 3:
        c = ''.join(ch * 2 for ch in c)
    if len(c) == 12:  # Tk 16-bit-per-channel form #rrrrggggbbbb
        c = c[0:2] + c[4:6] + c[8:10]
    if len(c) != 6:
        return None
    try:
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def _luminance_rgb(rgb) -> float:
    def channel(v):
        v = v / 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


@lru_cache(maxsize=1024)
def relative_luminance(color: str) -> float | None:
    rgb = _parse_hex(color)
    return None if rgb is None else _luminance_rgb(rgb)


def contrast_ratio(fg: str, bg: str) -> float | None:
    a, b = relative_luminance(fg), relative_luminance(bg)
    if a is None or b is None:
        return None
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


@lru_cache(maxsize=1024)
def readable_text_on(bg: str, light: str = LIGHT_INK, dark: str = DARK_INK) -> str:
    """Return whichever of *light*/*dark* text has the higher contrast on *bg*."""
    lc, dc = contrast_ratio(light, bg), contrast_ratio(dark, bg)
    if lc is None or dc is None:
        return light
    return light if lc >= dc else dark


# ----------------------------------------------------------------------------
# Widget-tree repair
# ----------------------------------------------------------------------------

_TEXT_WIDGET_CLASSES = {'Label', 'Button', 'Checkbutton', 'Radiobutton', 'Message'}


def _widget_hex(widget, color) -> str | None:
    if not color:
        return None
    parsed = _parse_hex(color)
    if parsed is not None:
        return '#%02x%02x%02x' % parsed
    try:
        r, g, b = widget.winfo_rgb(color)  # 16-bit per channel
    except Exception:
        return None
    return '#%02x%02x%02x' % (r >> 8, g >> 8, b >> 8)


def _fix_pair(widget, fg_opt: str, bg_opt: str, min_ratio: float) -> None:
    try:
        fg = _widget_hex(widget, widget.cget(fg_opt))
        bg = _widget_hex(widget, widget.cget(bg_opt))
    except Exception:
        return
    if not fg or not bg:
        return
    lum = relative_luminance(fg)
    ratio = contrast_ratio(fg, bg)
    if lum is None or ratio is None or ratio >= min_ratio or lum < 0.6:
        return
    better = readable_text_on(bg)
    better_ratio = contrast_ratio(better, bg) or 0.0
    if better != fg and better_ratio > ratio + 0.5:
        try:
            widget.configure(**{fg_opt: better})
        except Exception:
            pass


def fix_widget_contrast(widget, min_ratio: float = AA_NORMAL) -> None:
    try:
        cls = widget.winfo_class()
    except Exception:
        return
    if cls not in _TEXT_WIDGET_CLASSES:
        return
    _fix_pair(widget, 'foreground', 'background', min_ratio)
    if cls in ('Button', 'Checkbutton', 'Radiobutton'):
        _fix_pair(widget, 'activeforeground', 'activebackground', min_ratio)


def enforce_readable_tree(root, min_ratio: float = AA_NORMAL) -> None:
    """Repair light-on-bright text in *root* and all its descendants (iterative)."""
    stack = [root]
    while stack:
        widget = stack.pop()
        fix_widget_contrast(widget, min_ratio)
        try:
            stack.extend(widget.winfo_children())
        except Exception:
            continue
