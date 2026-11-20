#!/usr/bin/env python3
"""Tk-free stand-ins that let the existing dashboard mixins run unchanged.

The training/classification orchestration lives in ``ILTProcessMixin``,
``TrainingDatasetUIMixin`` and a few ``ILTApp`` methods.  They talk to Tk
through a small surface (``StringVar``/``BooleanVar``, ``after``, message
boxes, file dialogs and a handful of widgets).  This module provides
equivalents for that surface so the web interface can reuse the very same
methods instead of re-implementing them.

Only the web process installs these shims; the classic Tk interface is not
affected.
"""
from __future__ import annotations

import heapq
from contextlib import contextmanager
import itertools
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace


class HeadlessVar:
    """Minimal ``tk.StringVar``/``tk.BooleanVar`` replacement (get/set/trace)."""

    def __init__(self, master=None, value='', name=None):
        self._value = value
        self._traces = []

    def get(self):
        return self._value

    def set(self, value):
        self._value = value
        for callback in list(self._traces):
            callback('', '', 'write')

    def trace_add(self, mode, callback):
        self._traces.append(callback)
        return str(len(self._traces))


class HeadlessBoolVar(HeadlessVar):
    def __init__(self, master=None, value=False, name=None):
        super().__init__(master, bool(value), name)

    def get(self):
        return bool(self._value)

    def set(self, value):
        super().set(bool(value))


class HeadlessDoubleVar(HeadlessVar):
    def __init__(self, master=None, value=0.0, name=None):
        super().__init__(master, float(value), name)

    def get(self):
        try:
            return float(self._value)
        except (TypeError, ValueError):
            return 0.0


class HeadlessIntVar(HeadlessVar):
    def __init__(self, master=None, value=0, name=None):
        super().__init__(master, int(value), name)

    def get(self):
        try:
            return int(float(self._value))
        except (TypeError, ValueError):
            return 0


TK_SHIM = SimpleNamespace(StringVar=HeadlessVar, BooleanVar=HeadlessBoolVar,
                          DoubleVar=HeadlessDoubleVar, IntVar=HeadlessIntVar)


class StubWidget:
    """Records ``configure``/``[]`` calls so the web layer can read them."""

    def __init__(self, **state):
        self.state = dict(state)

    def configure(self, **kwargs):
        self.state.update(kwargs)

    config = configure

    def __getitem__(self, key):
        return self.state.get(key, 0)

    def __setitem__(self, key, value):
        self.state[key] = value

    def winfo_exists(self):
        return True

    def pack(self, *a, **k):
        return None

    grid = place = pack_forget = grid_forget = grid_remove = pack

    def __getattr__(self, name):
        # Any other widget method used by reused code becomes a no-op.
        if name.startswith('__'):
            raise AttributeError(name)
        return lambda *a, **k: None


class StubTree(StubWidget):
    """Replacement for the ttk.Treeview used by the model-class preview."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def get_children(self):
        return list(range(len(self.rows)))

    def delete(self, *_items):
        self.rows = []

    def insert(self, _parent, _index, values=()):
        self.rows.append(tuple(values))


class SeriesStore:
    """Replacement for ``TrainingPlotCard``: stores (epoch, value) series."""

    def __init__(self, keys):
        self.keys = tuple(keys)
        self.version = 0
        self.reset()

    def reset(self):
        self.series = {key: {} for key in self.keys}
        self.version += 1

    def add_point(self, key, epoch, value):
        self.series.setdefault(key, {})[int(epoch)] = float(value)
        self.version += 1

    def as_lists(self):
        out = {}
        for key, points in self.series.items():
            xs = sorted(points)
            out[key] = (xs, [points[x] for x in xs])
        return out


class Scheduler:
    """Single-threaded ``after``/``after_cancel`` loop (Tk mainloop stand-in).

    Every job runs on the scheduler thread while holding ``lock``.  Web
    callbacks take the same lock, so the reused mixin code always sees the
    same serialized execution model it had under Tk.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self._jobs = []
        self._cancelled = set()
        self._ids = itertools.count(1)
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, name='ilt-web-scheduler', daemon=True)
        self._thread.start()

    def after(self, ms, func=None, *args):
        if func is None:
            return None
        job = f'after#{next(self._ids)}'
        with self.lock:
            heapq.heappush(self._jobs, (time.monotonic() + max(0, ms) / 1000.0, job, func, args))
        self._wake.set()
        return job

    def after_cancel(self, job):
        if job:
            with self.lock:
                self._cancelled.add(job)

    def _run(self):
        while True:
            with self.lock:
                now = time.monotonic()
                due = []
                while self._jobs and self._jobs[0][0] <= now:
                    due.append(heapq.heappop(self._jobs))
                delay = (self._jobs[0][0] - now) if self._jobs else 0.5
            for _t, job, func, args in due:
                with self.lock:
                    if job in self._cancelled:
                        self._cancelled.discard(job)
                        continue
                    try:
                        func(*args)
                    except Exception as exc:  # keep the loop alive like Tk does
                        print(f'[ilt-web] scheduled job failed: {exc!r}', file=sys.stderr)
            self._wake.wait(timeout=max(0.01, min(delay, 0.5)))
            self._wake.clear()


class Notifications:
    """Collects messages that the Tk code would have shown in message boxes."""

    def __init__(self):
        self._items = deque(maxlen=50)
        self._seq = itertools.count(1)
        self._lock = threading.Lock()

    def push(self, kind, title, message):
        with self._lock:
            self._items.append((next(self._seq), kind, str(title or ''), str(message or '')))

    def since(self, seq):
        with self._lock:
            return [item for item in self._items if item[0] > seq]


NOTIFY = Notifications()


class MessageboxShim:
    """``tkinter.messagebox`` replacement that forwards to :data:`NOTIFY`.

    ``askyesno`` answers *yes*: every destructive action asks for
    confirmation in the web interface before reaching the reused code.
    """

    @staticmethod
    def showerror(title=None, message=None, **_kw):
        NOTIFY.push('error', title, message)

    @staticmethod
    def showwarning(title=None, message=None, **_kw):
        NOTIFY.push('warning', title, message)

    @staticmethod
    def showinfo(title=None, message=None, **_kw):
        NOTIFY.push('info', title, message)

    @staticmethod
    def askyesno(*_a, **_kw):
        return True

    askokcancel = askyesno


def _zenity(args):
    exe = shutil.which('zenity')
    if not exe:
        return None
    try:
        res = subprocess.run([exe, *args], capture_output=True, text=True, timeout=3600)
    except Exception:
        return None
    if res.returncode != 0:
        return ''
    return res.stdout.strip()


_TK_DIALOG = r'''
import sys, json, tkinter as tk
from tkinter import filedialog
kind, opts = sys.argv[1], json.loads(sys.argv[2])
root = tk.Tk(); root.withdraw()
try:
    root.attributes('-topmost', True)
except Exception:
    pass
fn = {'dir': filedialog.askdirectory, 'open': filedialog.askopenfilename,
      'save': filedialog.asksaveasfilename}[kind]
res = fn(parent=root, **opts)
print(res or '')
'''


def _tk_dialog(kind, opts):
    import json
    try:
        res = subprocess.run([sys.executable, '-c', _TK_DIALOG, kind, json.dumps(opts)],
                             capture_output=True, text=True, timeout=3600)
        return res.stdout.strip()
    except Exception:
        return ''


DIALOG_LOCK = None  # set by the controller: released while a dialog is open


@contextmanager
def _released():
    """Release the controller lock during a blocking dialog.

    Tk keeps processing events while a modal dialog is open (nested event
    loop); releasing the lock gives the scheduler the same behaviour, so the
    live log and progress keep updating while the user picks a folder.
    """
    lock = DIALOG_LOCK
    owned = lock is not None and lock._is_owned()
    state = lock._release_save() if owned else None
    try:
        yield
    finally:
        if owned:
            lock._acquire_restore(state)


class FiledialogShim:
    """Native file dialogs for the local web interface.

    The web server runs on the user's own machine, so a desktop dialog (zenity,
    or Tk in a short-lived helper process) returns real local paths.
    """

    @staticmethod
    def askdirectory(initialdir=None, title=None, **_kw):
        start = str(Path(initialdir).expanduser()) if initialdir else str(Path.home())
        with _released():
            res = _zenity(['--file-selection', '--directory', f'--filename={start.rstrip("/")}/',
                           f'--title={title or "Selecionar pasta"}'])
            if res is None:
                res = _tk_dialog('dir', {'initialdir': start, 'title': title or ''})
        return res or ''

    @staticmethod
    def askopenfilename(initialdir=None, title=None, filetypes=None, **_kw):
        start = str(Path(initialdir).expanduser()) if initialdir else str(Path.home())
        args = ['--file-selection', f'--filename={start.rstrip("/")}/', f'--title={title or "Abrir arquivo"}']
        for label, pattern in (filetypes or []):
            args.append(f'--file-filter={label} | {pattern}')
        with _released():
            res = _zenity(args)
            if res is None:
                res = _tk_dialog('open', {'initialdir': start, 'title': title or ''})
        return res or ''

    @staticmethod
    def asksaveasfilename(initialdir=None, initialfile='', title=None, defaultextension='', **_kw):
        start = Path(initialdir).expanduser() if initialdir else Path.home()
        with _released():
            res = _zenity(['--file-selection', '--save', '--confirm-overwrite',
                           f'--filename={start / (initialfile or "")}', f'--title={title or "Salvar como"}'])
            if res is None:
                res = _tk_dialog('save', {'initialdir': str(start), 'initialfile': initialfile or '',
                                          'defaultextension': defaultextension or ''})
        if res and defaultextension and not Path(res).suffix:
            res += defaultextension
        return res or ''


def install_module_shims(*modules):
    """Point ``tk``/``messagebox``/``filedialog`` of *modules* at the shims."""
    for module in modules:
        if hasattr(module, 'tk'):
            module.tk = TK_SHIM
        if hasattr(module, 'messagebox'):
            module.messagebox = MessageboxShim
        if hasattr(module, 'filedialog'):
            module.filedialog = FiledialogShim
