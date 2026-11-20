#!/usr/bin/env python3
"""Reusable Tk widgets for the ILT dashboard.

The widgets in this module are presentation-only. Application orchestration
stays in ``ilt_gui.py`` so the controller remains explicit while common visual
components are isolated and reusable.
"""

import tkinter as tk
import weakref
from tkinter import ttk

from ui_resources import COLORS


class RoundedPanel(tk.Canvas):
    """Canvas-backed panel with rounded corners, usable with pack/grid."""
    def __init__(self, parent, bg, border=None, radius=14, pad=1, **kwargs):
        parent_bg = kwargs.pop('canvas_bg', None)
        if parent_bg is None:
            try:
                parent_bg = parent.cget('bg')
            except Exception:
                parent_bg = COLORS['bg']
        super().__init__(parent, bg=parent_bg, highlightthickness=0, bd=0, **kwargs)
        self.panel_bg = bg
        self.border = border or bg
        self.radius = radius
        self.pad = max(1, pad)
        self.inner = tk.Frame(self, bg=bg, bd=0, highlightthickness=0)
        self._window = self.create_window((self.pad + 1, self.pad + 1), window=self.inner, anchor='nw')
        self._redraw_job = None
        self.bind('<Configure>', self._redraw)
        self.inner.bind('<Configure>', self._sync_height)

    @staticmethod
    def _points(x1, y1, x2, y2, r):
        r = max(2, min(r, (x2-x1)/2, (y2-y1)/2))
        return [
            x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
            x2, y2-r, x2, y2, x2-r, y2, x1+r, y2,
            x1, y2, x1, y2-r, x1, y1+r, x1, y1,
        ]

    def _sync_height(self, _event=None):
        req = self.inner.winfo_reqheight() + 2 * (self.pad + 1)
        try:
            current = int(float(self.cget('height')))
        except Exception:
            current = 0
        if req > 4 and abs(current - req) > 2:
            self.configure(height=req)
        # Coalesce: a page build fires dozens of inner <Configure> events;
        # one idle redraw is enough.
        if self._redraw_job is None:
            try:
                self._redraw_job = self.after_idle(self._idle_redraw)
            except Exception:
                self._redraw_job = None

    def _idle_redraw(self):
        self._redraw_job = None
        try:
            if self.winfo_exists():
                self._redraw()
        except Exception:
            pass

    def _redraw(self, _event=None):
        w = max(4, self.winfo_width())
        h = max(4, self.winfo_height())
        self.delete('rounded_bg')
        pts = self._points(1, 1, w-1, h-1, self.radius)
        self.create_polygon(pts, smooth=True, splinesteps=24, fill=self.panel_bg,
                            outline=self.border, width=1, tags='rounded_bg')
        self.tag_lower('rounded_bg')
        inset = self.pad + 1
        self.coords(self._window, inset, inset)
        self.itemconfigure(self._window, width=max(1, w - 2*inset), height=max(1, h - 2*inset))


class ScrollableFrame(tk.Frame):
    def __init__(self, parent, bg):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.scroll = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self.inner.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor='nw')
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfigure(self.window, width=e.width))
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.scroll.pack(side='right', fill='y')
        # Tk reports the wheel differently on Windows/macOS (<MouseWheel>)
        # and X11/Linux (<Button-4>/<Button-5>).  A single application-wide
        # dispatcher is bound once; previously every page rebuild appended a
        # new bind_all handler that kept running for destroyed frames.
        ScrollableFrame._instances.add(self)
        root = self._root()
        if root not in ScrollableFrame._bound_roots:
            ScrollableFrame._bound_roots.add(root)
            for seq in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
                self.canvas.bind_all(seq, ScrollableFrame._dispatch_wheel, add='+')

    _instances = weakref.WeakSet()
    _bound_roots = weakref.WeakSet()

    @staticmethod
    def _dispatch_wheel(event):
        """Route a wheel event to the innermost ScrollableFrame under the pointer."""
        live = ScrollableFrame._instances
        cur = None
        # event.widget can be a plain string for some ttk popups; fall back to
        # any live frame to resolve the widget under the pointer.
        for probe in (event.widget, *list(live)[:1]):
            try:
                cur = probe.winfo_containing(event.x_root, event.y_root)
                break
            except Exception:
                continue
        while cur is not None:
            if cur in live:
                return cur._wheel(event)
            cur = getattr(cur, 'master', None)
        return None

    def _wheel(self, event):
        try:
            if getattr(event, 'num', None) == 4:
                steps = -1
            elif getattr(event, 'num', None) == 5:
                steps = 1
            else:
                delta = int(getattr(event, 'delta', 0) or 0)
                if delta == 0:
                    return
                steps = -max(1, abs(delta) // 120) if delta > 0 else max(1, abs(delta) // 120)
            self.canvas.yview_scroll(int(steps), 'units')
            return 'break'
        except Exception:
            return


class TrainingPlotCard(RoundedPanel):
    """Compact live line chart used by the training execution panel."""
    def __init__(self, parent, app, title_key, series, percent=False):
        super().__init__(parent, bg=COLORS['card2'], border=COLORS['border'], radius=14, pad=1)
        self.app = app
        self.title_key = title_key
        self.series = series
        self.percent = percent
        self.history = {key: [] for key, _, _ in series}
        self.current = {key: None for key, _, _ in series}

        head = tk.Frame(self.inner, bg=COLORS['card2'])
        head.pack(fill='x', padx=10, pady=(8, 2))
        self.title_var = tk.StringVar(value=app.tr(title_key))
        tk.Label(head, textvariable=self.title_var, bg=COLORS['card2'], fg=COLORS['text'],
                 font=('Sans', 9, 'bold')).pack(side='left')

        self.legend = tk.Frame(self.inner, bg=COLORS['card2'])
        self.legend.pack(fill='x', padx=10, pady=(1, 2))
        self.legend_vars = {}
        for key, label_key, color in self.series:
            cell = tk.Frame(self.legend, bg=COLORS['card2'])
            cell.pack(side='left', padx=(0, 12))
            tk.Label(cell, text='●', bg=COLORS['card2'], fg=color, font=('Sans', 8, 'bold')).pack(side='left')
            var = tk.StringVar(value='—')
            self.legend_vars[key] = (var, label_key)
            tk.Label(cell, textvariable=var, bg=COLORS['card2'], fg=COLORS['muted'],
                     font=('Sans', 8)).pack(side='left', padx=(3, 0))

        self.canvas = tk.Canvas(self.inner, height=118, bg=COLORS['card2'], highlightthickness=0, bd=0)
        self.canvas.pack(fill='x', expand=True, padx=7, pady=(0, 7))
        self.canvas.bind('<Configure>', lambda _e: self._draw())
        app.register_callback(self.refresh_language)
        self.refresh_language()

    def refresh_language(self):
        self.title_var.set(self.app.tr(self.title_key))
        for key, label_key, _ in self.series:
            self._update_legend(key, label_key)
        self._draw()

    def _format_value(self, value):
        if value is None:
            return '—'
        if self.percent:
            return f'{value:.2f}%'
        return f'{value:.4f}'

    def _update_legend(self, key, label_key=None):
        var, stored_label = self.legend_vars[key]
        label_key = label_key or stored_label
        var.set(f'{self.app.tr(label_key)}  {self._format_value(self.current[key])}')

    def add_point(self, key, epoch, value):
        if key not in self.history:
            return
        value = float(value)
        epoch = int(epoch)
        hist = self.history[key]
        if hist and hist[-1][0] == epoch:
            hist[-1] = (epoch, value)
        else:
            hist.append((epoch, value))
        self.current[key] = value
        self._update_legend(key)
        self._draw()

    def reset(self):
        for key in self.history:
            self.history[key] = []
            self.current[key] = None
            self._update_legend(key)
        self._draw()

    def _draw(self):
        c = self.canvas
        c.delete('all')
        w = max(260, c.winfo_width())
        h = max(90, c.winfo_height())
        left, right, top, bottom = 38, w - 8, 8, h - 23
        if right <= left or bottom <= top:
            return

        all_pts = [p for hist in self.history.values() for p in hist]
        if self.percent:
            ymin, ymax = 0.0, 100.0
        elif all_pts:
            vals = [v for _, v in all_pts]
            ymin = 0.0
            ymax = max(vals) * 1.12
            if ymax <= 0:
                ymax = 1.0
        else:
            ymin, ymax = 0.0, 1.0

        # subtle grid and y labels
        fs = max(7, int(round(7 * self.app.font_scale)))
        for i in range(4):
            frac = i / 3
            y = bottom - frac * (bottom - top)
            val = ymin + frac * (ymax - ymin)
            c.create_line(left, y, right, y, fill='#242429', width=1)
            label = f'{val:.0f}%' if self.percent else f'{val:.2f}'
            c.create_text(left - 5, y, text=label, fill=COLORS['muted'], anchor='e', font=('Sans', fs))

        if not all_pts:
            c.create_text((left + right) / 2, (top + bottom) / 2,
                          text='—', fill=COLORS['disabled'], font=('Sans', max(11, fs + 4), 'bold'))
            return

        epochs = [e for e, _ in all_pts]
        emin, emax = min(epochs), max(epochs)
        if emax == emin:
            emax = emin + 1
        span_y = max(1e-12, ymax - ymin)

        for key, _label_key, color in self.series:
            hist = self.history[key]
            if not hist:
                continue
            coords = []
            for epoch, value in hist:
                x = left + (right - left) * (epoch - emin) / (emax - emin)
                y = bottom - (bottom - top) * (value - ymin) / span_y
                y = max(top, min(bottom, y))
                coords.extend((x, y))
            if len(coords) >= 4:
                c.create_line(*coords, fill=color, width=2.2, smooth=True, splinesteps=12)
            x, y = coords[-2], coords[-1]
            c.create_oval(x-3, y-3, x+3, y+3, fill=color, outline=color)

        c.create_text(left, bottom + 12, text=f'{self.app.tr("epoch_short")} {emin}',
                      fill=COLORS['muted'], anchor='w', font=('Sans', fs))
        c.create_text(right, bottom + 12, text=f'{self.app.tr("epoch_short")} {max(epochs)}',
                      fill=COLORS['muted'], anchor='e', font=('Sans', fs))


class GradientBar(tk.Canvas):
    """Thin horizontal accent strip with a linear colour gradient.

    Drawn with a bounded number of vertical bands (not one line per pixel) and
    redrawn only when the widget is resized, so it costs nothing at runtime.
    """
    def __init__(self, parent, color_a, color_b, height=3, bands=64, **kwargs):
        try:
            bg = parent.cget('bg')
        except Exception:
            bg = COLORS['bg']
        super().__init__(parent, height=height, bg=bg, highlightthickness=0, bd=0, **kwargs)
        self._a = self._rgb(color_a)
        self._b = self._rgb(color_b)
        self._bands = max(2, int(bands))
        self._last_width = None
        self.bind('<Configure>', self._redraw)

    def _rgb(self, color):
        c = str(color).lstrip('#')
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))

    def _redraw(self, _event=None):
        w = max(1, self.winfo_width())
        if w == self._last_width:
            return
        self._last_width = w
        h = max(1, self.winfo_height())
        self.delete('all')
        n = self._bands
        for i in range(n):
            t = i / (n - 1)
            rgb = tuple(int(round(a + (b - a) * t)) for a, b in zip(self._a, self._b))
            x0 = w * i / n
            x1 = w * (i + 1) / n + 1
            self.create_rectangle(x0, 0, x1, h, fill='#%02x%02x%02x' % rgb, width=0)
