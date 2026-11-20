#!/usr/bin/env python3
"""Shared hardware/runtime tuning for the PyTorch workers (train.py, main.py).

Everything here is conservative by design:

* numerical results are never changed for deterministic runs (seed != -1);
* ``channels_last`` is only enabled on GPUs with Tensor Cores (compute
  capability >= 7.0, e.g. RTX 20xx/30xx/40xx).  Pascal cards such as the
  Quadro P4000 keep the classic NCHW layout, which is faster there;
* every automatic choice can be overridden by an environment variable.

Environment overrides
---------------------
``ILT_CHANNELS_LAST``   ``auto`` (default), ``1`` or ``0``.
``ILT_INFER_WORKERS``   DataLoader workers for classification feature extraction.
``ILT_INFER_BATCH_SIZE`` inference batch (``auto`` = from free VRAM, halved on OOM).
``ILT_AMP_DTYPE``       ``auto`` (bf16 on CC >= 8.0, else fp16), ``bf16`` or ``fp16``.
``ILT_TORCH_COMPILE``   ``1`` enables ``torch.compile`` for training/inference (opt-in).
"""
from __future__ import annotations

import os

import torch

# CPU helpers live in a torch-free module so the GUI can use them cheaply;
# they are re-exported here for the worker scripts.
from hw_resources import (  # noqa: F401
    auto_num_workers, available_cpus, cpu_jobs, env_num_workers,
)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def configure_backends(device: torch.device, deterministic: bool) -> None:
    """Enable cuDNN autotuning only when bitwise reproducibility is not requested."""
    if device.type != 'cuda':
        return
    if not deterministic:
        torch.backends.cudnn.benchmark = True


def amp_dtype(device: torch.device) -> torch.dtype:
    """Autocast dtype: bfloat16 on Ampere and newer (CC >= 8.0), float16 before.

    bfloat16 keeps the float32 exponent range, so no loss scaling and no
    overflow handling are needed (Kalamkar et al. 2019, *A Study of BFLOAT16
    for Deep Learning Training*, arXiv:1905.12322); float16 + GradScaler is the
    classic mixed-precision recipe (Micikevicius et al. 2018, ICLR) and is kept
    for Pascal/Volta/Turing, where bfloat16 is only emulated.
    ``ILT_AMP_DTYPE=bf16|fp16`` overrides the choice.
    """
    if device.type != 'cuda':
        return torch.float16
    mode = str(os.environ.get('ILT_AMP_DTYPE', 'auto')).strip().lower()
    if mode in ('bf16', 'bfloat16'):
        return torch.bfloat16
    if mode in ('fp16', 'float16'):
        return torch.float16
    try:
        major, _minor = torch.cuda.get_device_capability(device)
    except Exception:
        return torch.float16
    return torch.bfloat16 if major >= 8 else torch.float16


def cuda_free_gib(device: torch.device) -> float:
    if device.type != 'cuda':
        return 0.0
    try:
        free, _total = torch.cuda.mem_get_info(device)
        return free / 1024 ** 3
    except Exception:
        return 0.0


def _is_auto(value) -> bool:
    return str(value).strip().lower() in ('', 'auto', '-1', '0')


def resolve_batch_size(requested, device: torch.device, *, training: bool) -> int:
    """Integer batch size; ``'auto'`` picks a value from the free VRAM.

    The thresholds are conservative for ConvNeXt-Tiny at 224×224 with mixed
    precision; an OOM at runtime still halves the batch (callers retry).
    """
    if not _is_auto(requested):
        return max(1, int(float(str(requested).strip())))
    if device.type != 'cuda':
        return 16 if training else 64
    free = cuda_free_gib(device)
    if training:
        return 128 if free >= 20 else 64 if free >= 10 else 32 if free >= 6 else 16
    return 512 if free >= 20 else 256 if free >= 10 else 128


def maybe_compile(model):
    """``torch.compile`` only when ILT_TORCH_COMPILE=1 (opt-in: long first
    epoch, Triton/Linux only); returns the model unchanged otherwise."""
    if str(os.environ.get('ILT_TORCH_COMPILE', '')).strip() != '1':
        return model
    try:
        compiled = torch.compile(model)
        print('torch.compile: enabled (ILT_TORCH_COMPILE=1)')
        return compiled
    except Exception as exc:
        print('torch.compile unavailable, running eagerly:', exc)
        return model


def use_channels_last(device: torch.device) -> bool:
    """True when NHWC memory layout is expected to speed up ConvNeXt."""
    if device.type != 'cuda':
        return False
    mode = str(os.environ.get('ILT_CHANNELS_LAST', 'auto')).strip().lower()
    if mode in ('0', 'false', 'no', 'off'):
        return False
    if mode in ('1', 'true', 'yes', 'on'):
        return True
    try:
        major, _minor = torch.cuda.get_device_capability(device)
    except Exception:
        return False
    return major >= 7


def input_memory_format(channels_last: bool):
    return torch.channels_last if channels_last else torch.preserve_format


def loader_kwargs(batch_size: int, num_workers: int, pin_memory: bool, persistent: bool = True) -> dict:
    """Common DataLoader options.

    ``persistent`` keeps the worker processes between epochs and only helps
    when the same DataLoader is iterated again (training).  A DataLoader that
    is created for a single pass must use ``persistent=False``: with
    persistent workers and ``pin_memory`` PyTorch registers an ``atexit``
    hook per worker that keeps every finished worker (and two pipe
    descriptors each) alive until the program ends, which exhausted the
    1024-descriptor limit after ~60 classification batches.
    """
    workers = max(0, int(num_workers))
    # prefetch_factor stays at PyTorch's default (2): with large inference
    # batches a higher value only grows /dev/shm usage without extra speed.
    return dict(
        batch_size=max(1, int(batch_size)),
        num_workers=workers,
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent) and workers > 0,
    )


def describe_runtime(device: torch.device, num_workers: int, channels_last: bool) -> str:
    if device.type == 'cuda':
        try:
            name = torch.cuda.get_device_name(device)
        except Exception:
            name = 'CUDA'
        return f'GPU: {name} | DataLoader workers: {num_workers} | channels_last: {channels_last}'
    return f'CPU only | DataLoader workers: {num_workers} | torch threads: {torch.get_num_threads()}'
