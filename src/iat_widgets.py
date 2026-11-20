#!/usr/bin/env python3
"""Reusable Tk widgets used by the integrated IAT workspace."""

import numpy as np
import tkinter as tk
from tkinter import ttk

from ui_resources import IAT_COLORS as C


class ScrollFrame(tk.Frame):
    """Vertical scroll container with portable mouse-wheel support.

    The left-side IAT panels can become taller than the available window on
    Windows (DPI/font metrics differ from Linux), so every child receives the
    wheel binding. Listboxes/Text widgets keep their native scrolling.
    """
    def __init__(self,parent,bg=C['panel'],gallery_mode=False):
        super().__init__(parent,bg=bg)
        # Optional fast path only for the 500-image Gallery. Other tabs keep
        # their original wheel-binding behavior.
        self._gallery_mode=gallery_mode
        self._gallery_layout_after=None
        self.canvas=tk.Canvas(self,bg=bg,highlightthickness=0)
        self.sb=ttk.Scrollbar(self,orient='vertical',command=self.canvas.yview)
        self.inner=tk.Frame(self.canvas,bg=bg)
        self.win=self.canvas.create_window((0,0),window=self.inner,anchor='nw')
        self.canvas.configure(yscrollcommand=self.sb.set)
        self.canvas.pack(side='left',fill='both',expand=True)
        self.sb.pack(side='right',fill='y')
        self.inner.bind('<Configure>',self._inner_changed)
        self.canvas.bind('<Configure>',lambda e:self.canvas.itemconfigure(self.win,width=e.width))
        self._bind_wheel(self.canvas)
        self._bind_wheel(self.inner)

    def _inner_changed(self,event=None):
        if self._gallery_mode:
            # Coalesce hundreds of inner-frame <Configure> events; the Gallery
            # binds each card's wheel handlers as it creates that card.
            if self._gallery_layout_after is None:
                try:self._gallery_layout_after=self.after(25,self._gallery_update_scrollregion)
                except Exception:pass
            return
        try:self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        except Exception:return
        self._bind_descendants(self.inner)

    def _gallery_update_scrollregion(self):
        self._gallery_layout_after=None
        try:self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        except Exception:pass

    def _bind_descendants(self,widget):
        try:children=widget.winfo_children()
        except Exception:return
        for child in children:
            # Preserve native wheel behavior for widgets whose primary purpose
            # is scrolling their own content.
            if not isinstance(child,(tk.Listbox,tk.Text,ttk.Combobox,tk.Spinbox,tk.Scale,ttk.Scale)):
                self._bind_wheel(child)
            self._bind_descendants(child)

    def _bind_wheel(self,widget):
        if getattr(widget,'_iat_wheel_bound',False):
            return
        try:
            widget.bind('<MouseWheel>',self._on_wheel,add='+')
            widget.bind('<Button-4>',self._on_wheel,add='+')
            widget.bind('<Button-5>',self._on_wheel,add='+')
            widget._iat_wheel_bound=True
        except Exception:
            pass

    def _on_wheel(self,event):
        try:
            # Windows reports multiples of 120. macOS may report small deltas.
            if getattr(event,'num',None)==4:
                step=-1
            elif getattr(event,'num',None)==5:
                step=1
            else:
                delta=float(getattr(event,'delta',0) or 0)
                if delta==0:return
                step=-max(1,int(abs(delta)/120)) if delta>0 else max(1,int(abs(delta)/120))
            first,last=self.canvas.yview()
            if (step<0 and first<=0.0) or (step>0 and last>=1.0):
                return
            self.canvas.yview_scroll(step,'units')
            return 'break'
        except Exception:
            return


class RangeSlider(tk.Canvas):
    """Two-handle range slider with an optional distribution histogram.

    The histogram is intentionally drawn inside the same Canvas as the handles,
    so changing a filter never rebuilds the surrounding Gallery controls.
    """
    def __init__(self, parent, low_var, high_var, minimum=0.0, maximum=1.0, step=0.01,
                 command=None, values=None, bins=28, **kw):
        super().__init__(parent, height=52, bg=C['card'], highlightthickness=0, **kw)
        self.low_var=low_var; self.high_var=high_var; self.minimum=float(minimum); self.maximum=float(maximum); self.step=float(step); self.command=command
        self.active=None
        self.bins=max(8,int(bins))
        self._hist_values=np.asarray([],dtype=float)
        self.set_values(values,redraw=False)
        self.bind('<Configure>',lambda e:self._draw())
        self.bind('<Button-1>',self._press); self.bind('<B1-Motion>',self._drag); self.bind('<ButtonRelease-1>',lambda e:self._release())
        self._draw()

    def set_values(self,values,redraw=True):
        try:
            a=np.asarray([] if values is None else values,dtype=float).reshape(-1)
            a=a[np.isfinite(a)]
            self._hist_values=a
        except Exception:
            self._hist_values=np.asarray([],dtype=float)
        # The histogram depends only on the values, never on the handles, so
        # it is computed once here instead of on every mouse-drag redraw.
        self._hist_cache=None
        vals=self._hist_values
        if vals.size:
            try:
                vals=vals[(vals>=self.minimum)&(vals<=self.maximum)]
                if vals.size:
                    hist,edges=np.histogram(vals,bins=self.bins,range=(self.minimum,self.maximum))
                    centers=((edges[:-1]+edges[1:])*0.5).tolist()
                    self._hist_cache=(hist.tolist(),centers,max(int(hist.max()),1))
            except Exception:
                self._hist_cache=None
        if redraw:
            self._draw()

    def _vals(self):
        try: lo=float(self.low_var.get())
        except Exception: lo=self.minimum
        try: hi=float(self.high_var.get())
        except Exception: hi=self.maximum
        lo=max(self.minimum,min(self.maximum,lo)); hi=max(self.minimum,min(self.maximum,hi))
        if lo>hi: lo,hi=hi,lo
        return lo,hi

    def _x(self,v):
        w=max(self.winfo_width(),160); pad=14
        return pad+(v-self.minimum)/max(self.maximum-self.minimum,1e-12)*(w-2*pad)

    def _value(self,x):
        w=max(self.winfo_width(),160); pad=14
        f=(x-pad)/max(w-2*pad,1); f=max(0,min(1,f)); v=self.minimum+f*(self.maximum-self.minimum)
        return round(v/self.step)*self.step

    def _draw(self):
        self.delete('all')
        w=max(self.winfo_width(),160); pad=14
        lo,hi=self._vals(); x1=self._x(lo); x2=self._x(hi)
        hist_top=3; hist_bottom=31

        cache=getattr(self,'_hist_cache',None)
        if cache:
            try:
                hist,centers,peak=cache
                usable=max(w-2*pad,1)
                for i,count in enumerate(hist):
                    bx1=pad+(i/self.bins)*usable
                    bx2=pad+((i+1)/self.bins)*usable-1
                    bh=(float(count)/peak)*(hist_bottom-hist_top)
                    center=centers[i]
                    fill=C['blue2'] if lo<=center<=hi else C['border']
                    self.create_rectangle(bx1,hist_bottom-bh,bx2,hist_bottom,fill=fill,outline='')
            except Exception:
                pass

        y=32
        self.create_line(pad,y,w-pad,y,fill=C['border'],width=4,capstyle='round')
        self.create_line(x1,y,x2,y,fill=C['blue'],width=4,capstyle='round')
        for x in (x1,x2):
            self.create_oval(x-7,y-7,x+7,y+7,fill='#ffffff',outline=C['blue'],width=2)
        self.create_text(pad,47,text=f'{lo:.2f}',fill=C['muted'],anchor='w',font=('Sans',7))
        self.create_text(w-pad,47,text=f'{hi:.2f}',fill=C['muted'],anchor='e',font=('Sans',7))

    def _press(self,e):
        lo,hi=self._vals(); self.active='lo' if abs(e.x-self._x(lo))<=abs(e.x-self._x(hi)) else 'hi'; self._set(e.x)

    def _drag(self,e): self._set(e.x)

    def _release(self):
        if self.command: self.command()
        self.active=None

    def _set(self,x):
        v=self._value(x); lo,hi=self._vals()
        if self.active=='lo': lo=min(v,hi); self.low_var.set(f'{lo:.2f}')
        else: hi=max(v,lo); self.high_var.set(f'{hi:.2f}')
        self._draw()
