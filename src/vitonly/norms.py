"""Raw DCE phases -> 3-channel ViT images: intensity normalisations, crops and slice choice.

Every normalisation maps raw (k, H, W, 3) = (pre, early, late) to float [0, 1] (k, H, W, 3):

    joint_pct    v1 baseline: all phases clipped jointly at the volume's 0.5/99.5 pct (R/G/B = pre/early/late)
    joint_wide   same, upper limit at the 99.9th pct (v2 only; on v1 it equals joint_pct) -> less saturation
    per_phase    each phase clipped at its own 0.5/99.5 pct: destroys relative enhancement (ablation)
    joint_tumor  joint window from the tumour box (1/99.5 pct of all phases): radiologist-style lesion window
    pre_ref      every phase as log(1 + x / r) / log(13): removes scanner gain, keeps enhancement, no clipping
    zscore       (x - mean_pre_fg) / std_pre_fg, mapped (z + 2) / 12 (nnU-Net / MAMA-MIA style)
    sub          R = early (log-relative), G = wash-in (early - pre) / 6r, B = 0.5 + (late - early) / 2r
                 (B < 0.5 washout, > 0.5 persistent enhancement)
    kinetic      R = pre (log-relative), G = percent enhancement PE = (early - pre) / pre (/3),
                 B = signal enhancement ratio SER = (early - pre) / (late - pre) (/2, only where it enhances)
    early3       early phase only, copied to R, G, B (does the multi-phase RGB help at all?)

r = mean pre-contrast foreground intensity of the patient's volume. Measured on the demo archive, enhancing
I-SPY tumour reaches 4-8 r in the early phase, so linear scales such as x / 3r would saturate the tumour
exactly like v1 did; log(1 + x / r) maps r -> 0.27, 4r -> 0.63, 8r -> 0.86, 12r -> 1.
Crops (in-plane, around the tumour-box centre, output `out` x `out`, centre 224 taken later):
    fixed   a 240 window, no resizing (what the v1 pipeline did)
    tumor   side = zoom x the longer box side (>= 64 px), resized: the tumour fills ~1/zoom of the image
    mm      side = fov_mm / pixel spacing, resized: same physical field of view for every scanner
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .slabs import raw

NORMS = ("joint_pct", "joint_wide", "per_phase", "joint_tumor", "pre_ref", "zscore", "sub", "kinetic",
         "early3")
CROPS = ("fixed", "tumor", "mm")


def _clip01(a):
    return np.clip(a, 0.0, 1.0, out=a)


def _logrel(a, r):
    return np.log1p(np.maximum(a, 0) / r) / np.log1p(12.0)


def normalise(x: np.ndarray, stats: dict, method: str) -> np.ndarray:
    """x raw (k, H, W, 3) float32 -> (k, H, W, 3) float32 in [0, 1]."""
    return _normalise(x, stats, method).astype(np.float32, copy=False)


def _normalise(x, stats, method):
    pre, early, late = x[..., 0], x[..., 1], x[..., 2]
    r = max(float(stats["fg"][0, 0]), 1e-6)                 # mean pre-contrast foreground
    if method in ("joint_pct", "joint_wide", "joint_tumor", "early3"):
        lo, hi = {"joint_pct": ("joint_lo", "joint_hi"), "early3": ("joint_lo", "joint_hi"),
                  "joint_wide": ("joint_lo", "joint_hi999"),
                  "joint_tumor": ("tumor_lo", "tumor_hi")}[method]
        lo, hi = stats[lo], stats[hi]
        out = (x - lo) / max(hi - lo, 1e-6)
        if method == "early3":
            out = np.repeat(out[..., 1:2], 3, axis=-1)
        return _clip01(out)
    if method == "per_phase":
        lo, hi = np.asarray(stats["phase_lo"]), np.asarray(stats["phase_hi"])
        return _clip01((x - lo) / np.maximum(hi - lo, 1e-6))
    if method == "pre_ref":
        return _clip01(_logrel(x, r))
    if method == "zscore":
        mu, sd = float(stats["fg"][0, 0]), max(float(stats["fg"][0, 1]), 1e-6)
        return _clip01(((x - mu) / sd + 2) / 12)
    if method == "sub":
        return _clip01(np.stack([_logrel(early, r), (early - pre) / (6 * r), 0.5 + (late - early) / (2 * r)], -1))
    if method == "kinetic":
        from scipy.ndimage import uniform_filter               # 3x3 in-plane mean: ratios amplify noise
        pre, early, late = (uniform_filter(a, size=(1, 3, 3), mode="nearest") for a in (pre, early, late))
        wash_in = early - pre
        signal = np.clip(early / (0.5 * r), 0, 1)                                 # 0 in air: no ratio noise
        pe = wash_in / np.maximum(pre, 0.2 * r)                                    # percent enhancement
        ser = wash_in / np.maximum(late - pre, 0.05 * r)                           # signal enhancement ratio
        gate = np.clip(wash_in / (0.5 * r), 0, 1)                                  # SER only where it enhances
        return _clip01(np.stack([_logrel(pre, r), np.clip(pe / 3, 0, 1) * signal,
                                 np.clip(ser / 2, 0, 1) * gate], -1))
    raise ValueError(f"unknown normalisation {method!r}; choose from {NORMS}")


def crop_side(method: str, bbox, spacing_xy: float, extent: int, zoom: float = 1.6, fov_mm: float = 100.0,
              min_side: int = 64) -> int:
    if method == "fixed":
        return min(240, extent)
    if method == "tumor":
        y0, y1, x0, x1 = bbox
        side = zoom * max(y1 - y0, x1 - x0)
    elif method == "mm":
        side = fov_mm / max(spacing_xy, 1e-3)
    else:
        raise ValueError(f"unknown crop {method!r}; choose from {CROPS}")
    return int(np.clip(round(side), min_side, extent))


def crop_resize(img: np.ndarray, bbox, side: int, out: int = 240) -> torch.Tensor:
    """(k, H, W, 3) float -> (k, 3, out, out) float tensor: a `side` square around the box centre (shifted
    inward at the edges), resized with antialiasing."""
    k, h, w, _ = img.shape
    y0, y1, x0, x1 = bbox
    side = min(side, h, w)
    oy = min(max(int(round((y0 + y1) / 2 - side / 2)), 0), h - side)
    ox = min(max(int(round((x0 + x1) / 2 - side / 2)), 0), w - side)
    t = torch.from_numpy(np.ascontiguousarray(img[:, oy:oy + side, ox:ox + side])).permute(0, 3, 1, 2)
    if side != out:
        t = F.interpolate(t, size=(out, out), mode="bilinear", align_corners=False, antialias=side > out)
    return t


def select_slices(slab, k: int, mode: str = "mid", train: bool = False, rng=None, mid_frac: float = 0.6):
    """Slice indices into the slab. even: whole tumour range; mid: the central `mid_frac` of it (edge slices
    hold only a sliver of tumour); area: the k slices with the largest mask area (mid if no mask).
    Training samples k at random from the range; evaluation takes k evenly spaced."""
    t0, t1 = slab["t0"], max(slab["t1"], slab["t0"] + 1)
    if mode == "area" and slab["mask"] is not None:
        area = slab["mask"][t0:t1].reshape(t1 - t0, -1).sum(1)
        cand = t0 + np.argsort(-area, kind="stable")[:max(k, int(np.ceil((t1 - t0) * mid_frac)))]
        cand = np.sort(cand)
    else:
        if mode in ("mid", "area"):
            n = t1 - t0
            cut = int(np.floor(n * (1 - mid_frac) / 2))
            t0, t1 = t0 + cut, max(t1 - cut, t0 + cut + 1)
        elif mode != "even":
            raise ValueError(f"unknown slice mode {mode!r}")
        cand = np.arange(t0, t1)
    if train:
        rng = rng or np.random.default_rng()
        return np.sort(rng.choice(cand, k, replace=len(cand) < k))
    if len(cand) <= k:
        return cand
    return cand[np.unique(np.linspace(0, len(cand) - 1, k).round().astype(int))]


def make_views(slab, idx, norm: str, crop: str, spacing_xy: float, out: int = 240, zoom: float = 1.6,
               fov_mm: float = 100.0, x_raw=None) -> torch.Tensor:
    """(k, 3, out, out) uint8 views of the chosen slices. Pass `x_raw` (= raw(slab, idx)) to reuse it
    across several normalisations; normalise() never modifies its input."""
    img = normalise(raw(slab, idx) if x_raw is None else x_raw, slab["stats"], norm)
    side = crop_side(crop, slab["bbox"], spacing_xy, min(img.shape[1:3]), zoom, fov_mm)
    t = crop_resize(img, slab["bbox"], side, out)
    return (t.clamp_(0, 1) * 255).round_().to(torch.uint8)
