"""Augmentation and normalisation of slice bags, on the GPU (D-009, plan Section 4 M2).

Input: uint8 (B, K, 3, 240, 240). Every slice of one patient gets the same geometric transform,
and the three channels (phases) always move together. Output: float (B, K, 3, 224, 224), normalised.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .rgb_slices import INPUT_SIZE

AUG_DEFAULTS = dict(flip=True, rotate_deg=15.0, scale=(0.9, 1.1), intensity=0.10, crop_jitter=8)


def _normalise(x: torch.Tensor, mean, std):
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
    return (x - m) / s


def center_crop(x: torch.Tensor, size: int = INPUT_SIZE):
    h, w = x.shape[-2:]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return x[..., y0:y0 + size, x0:x0 + size]


def eval_transform(x_u8: torch.Tensor, mean, std):
    x = center_crop(x_u8).float() / 255.0
    return _normalise(x, mean, std)


def train_transform(x_u8: torch.Tensor, mean, std, aug=None, generator=None):
    aug = {**AUG_DEFAULTS, **(aug or {})}
    b, k, c, h, w = x_u8.shape
    dev = x_u8.device

    def u(lo, hi, n=b):
        return torch.rand(n, device="cpu", generator=generator).to(dev) * (hi - lo) + lo

    x = x_u8.float().div_(255.0).reshape(b, k * c, h, w)
    # one affine per patient: rotation, isotropic scale, flips, translation (= crop jitter)
    ang = u(-aug["rotate_deg"], aug["rotate_deg"]) * math.pi / 180
    sc = u(*aug["scale"])
    fx = torch.where(u(0, 1) < 0.5, -1.0, 1.0) if aug["flip"] else torch.ones(b, device=dev)
    fy = torch.where(u(0, 1) < 0.5, -1.0, 1.0) if aug["flip"] else torch.ones(b, device=dev)
    j = aug["crop_jitter"] * 2.0 / w
    tx, ty = u(-j, j), u(-j, j)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * fx, -sin, tx], 1),
                         torch.stack([sin, cos * fy, ty], 1)], 1)            # (B, 2, 3)
    grid = F.affine_grid(theta, (b, k * c, h, w), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    x = center_crop(x.reshape(b, k, c, h, w))
    # per-channel intensity/contrast jitter, same for all slices of a patient
    if aug["intensity"]:
        i = aug["intensity"]
        gain = u(1 - i, 1 + i, b * c).view(b, 1, c, 1, 1)
        contrast = u(1 - i, 1 + i, b * c).view(b, 1, c, 1, 1)
        mu = x.mean(dim=(1, 3, 4), keepdim=True)
        x = ((x - mu) * contrast + mu) * gain
        x = x.clamp_(0, 1)
    return _normalise(x, mean, std)
