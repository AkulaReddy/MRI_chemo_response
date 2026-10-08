"""Slice -> patient aggregation (plan Section 4 M2): masked mean, gated attention-MIL."""
from __future__ import annotations

import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """x (B, K, ...) with valid (B, K) bool -> (B, ...)."""
    w = valid.to(x.dtype)
    w = w / w.sum(1, keepdim=True).clamp(min=1)
    return torch.einsum("bk,bk...->b...", w, x)


class GatedAttentionMIL(nn.Module):
    """Ilse et al. 2018 gated attention pooling. Returns (pooled (B, D), weights (B, K))."""

    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.u = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid())
        self.w = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor):
        a = self.w(self.drop(self.v(x) * self.u(x))).squeeze(-1)          # (B, K)
        a = a.float().masked_fill(~valid, float("-inf")).softmax(1).to(x.dtype)
        return torch.einsum("bk,bkd->bd", a, x), a
