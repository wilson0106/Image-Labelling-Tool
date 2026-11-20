#!/usr/bin/env python3
"""CPU resource helpers shared by GUI and workers.

This module must stay free of heavy imports (no torch/numpy) so it can be used
from the Tk process without slowing down startup.  GPU-specific tuning lives in
``torch_runtime.py``.

Environment overrides
---------------------
``ILT_TSNE_JOBS``       CPU threads used by openTSNE / scikit-learn t-SNE / kNN.
``ILT_INFER_WORKERS``   DataLoader workers for feature extraction (see torch_runtime).
"""
from __future__ import annotations

import os


def available_cpus() -> int:
    """CPUs this process may actually use (respects taskset/cgroups on Linux)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def auto_num_workers(requested=-1, cap: int = 8) -> int:
    """Resolve a DataLoader worker count.

    ``requested >= 0`` is honoured exactly (0 keeps loading in the main
    process).  A negative value means *automatic*: leave two CPUs for the GUI,
    the main loop and the OS, and never exceed ``cap`` workers.
    """
    try:
        value = int(requested)
    except (TypeError, ValueError):
        value = -1
    if value >= 0:
        return value
    if os.name == 'nt':
        # Windows spawns (not forks) workers; keep the historic safe default.
        return 0
    return max(0, min(int(cap), available_cpus() - 2))


def env_num_workers(var_name: str, cap: int = 8) -> int:
    """Worker count from an environment variable, ``auto`` when unset/invalid."""
    raw = str(os.environ.get(var_name, '')).strip().lower()
    if raw in ('', 'auto'):
        return auto_num_workers(-1, cap=cap)
    try:
        return auto_num_workers(int(raw), cap=cap)
    except ValueError:
        return auto_num_workers(-1, cap=cap)


def cpu_jobs(env_var: str = 'ILT_TSNE_JOBS') -> int:
    """Thread count for CPU-bound scientific libraries (t-SNE, kNN)."""
    raw = str(os.environ.get(env_var, '')).strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return max(1, available_cpus() - 1)


def memory_budget_bytes(fraction: float = 0.30, hard_cap_gib: float = 12.0) -> int:
    """A conservative RAM budget for optional in-process caches."""
    cap = int(hard_cap_gib * 1024 ** 3)
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
    except Exception:
        return 512 * 1024 ** 2
    return max(256 * 1024 ** 2, min(cap, int(available * float(fraction))))


def raise_open_file_limit(target: int = 65536):
    """Raise this process's soft limit of open files towards *target*.

    Ubuntu sessions start with a soft limit of 1024 descriptors while the hard
    limit is much higher; any process may raise its own soft limit up to the
    hard limit without privileges.  This is only a safety margin for long
    runs (classification of hundreds of batches, the IAT integrated cycle):
    the descriptor use itself stays constant.  Children inherit the new limit.
    Returns ``(old, new)`` or ``None`` when unsupported (Windows) or refused.
    """
    try:
        import resource
    except ImportError:
        return None
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = int(target) if hard == resource.RLIM_INFINITY else min(int(target), int(hard))
        if soft == resource.RLIM_INFINITY or soft >= want:
            return soft, soft
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
        return soft, want
    except (ValueError, OSError):
        return None
