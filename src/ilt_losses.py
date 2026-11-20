#!/usr/bin/env python3
"""Loss functions of the ILT trainer (PyTorch)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss computed from logits.

    ``FL(p_t) = -α_t · (1 − p_t)^γ · log p_t`` (Lin, Goyal, Girshick, He &
    Dollár 2017, *Focal Loss for Dense Object Detection*, ICCV), with
    ``p_t = softmax(logits)[target]``.

    The term ``log p_t`` comes from ``log_softmax`` (numerically stable, no
    ``eps`` needed) and the computation stays on the device: the previously
    used ``focal-loss-torch`` package took probabilities, asserted their range
    with ``torch.all`` (a host synchronisation on every batch) and used
    ``eps = 1e-16``.  With ``alpha=None`` and ``reduction='mean'`` the value is
    identical to that package up to the ``eps`` term.

    *alpha* is an optional per-class weight tensor of shape ``(C,)``.
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f'Unsupported reduction: {reduction!r}')
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
        self.register_buffer('alpha', alpha, persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=1)
        logpt = logp.gather(1, target.view(-1, 1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1.0 - pt) ** self.gamma) * logpt
        if self.alpha is not None:
            loss = self.alpha.to(loss.device)[target] * loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
