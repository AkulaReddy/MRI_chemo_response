"""Three DCE phases -> RGB uint8 slices (D-009).

This is the step the BreastDCEDL code gets wrong: it hands float [0, 1] arrays to
`Image.fromarray(mode="RGB")`, which reinterprets the float bytes as pixels (noise).
Here the conversion to uint8 is explicit and tested (tests/test_rgb_uint8.py).
"""
from __future__ import annotations

import numpy as np

CACHE_SIZE = 240   # cached crop; training takes a random 224 crop from it (±8 px jitter)
INPUT_SIZE = 224


def to_rgb_uint8(pre: np.ndarray, early: np.ndarray, late: np.ndarray,
                 lo_pct: float = 0.5, hi_pct: float = 99.5, stride: int = 4, clip=None):
    """Stack phases as R, G, B; clip jointly at volume percentiles; scale to 0..255 uint8.

    Returns (rgb (Z, H, W, 3) uint8, (lo, hi) clip values). Percentiles use every `stride`-th
    voxel of all three phases together, so the relative enhancement between phases is kept.
    `clip=(lo, hi)` uses those values instead (builder v2 computes them with `tumour_clip`).
    """
    if not (pre.shape == early.shape == late.shape):
        raise ValueError(f"phase shapes differ: {pre.shape} {early.shape} {late.shape}")
    stack = np.stack([pre, early, late], axis=-1).astype(np.float32, copy=False)
    if clip is None:
        lo, hi = np.percentile(stack.reshape(-1)[::stride], [lo_pct, hi_pct])
    else:
        lo, hi = clip
    scale = 255.0 / max(float(hi - lo), 1e-6)
    rgb = np.clip((stack - lo) * scale, 0, 255)
    return np.rint(rgb).astype(np.uint8), (float(lo), float(hi))


def tumour_clip(vols, box, lo_pct: float = 0.5, hi_pct: float = 99.5, tumour_pct: float = 99.9,
                stride: int = 4):
    """Joint (lo, hi) for builder v2: hi is raised to the `tumour_pct` percentile of the tumour box
    (all phases pooled), so the brightest phase no longer saturates inside the tumour (in v1 the
    late phase hit 255 there, hiding washout). `box` = (z0, z1, y0, y1, x0, x1), source voxels."""
    z0, z1, y0, y1, x0, x1 = box
    lo, hi = np.percentile(np.concatenate([v.reshape(-1)[::stride] for v in vols]), [lo_pct, hi_pct])
    tumour = np.concatenate([v[z0:z1, y0:y1, x0:x1].reshape(-1) for v in vols])
    return float(lo), float(max(hi, np.percentile(tumour, tumour_pct)))


def crop_origin(center: float, size: int, extent: int) -> int:
    """Start index of a `size` window centred on `center`, shifted inward to stay inside [0, extent)."""
    if size >= extent:
        return 0
    start = int(round(center - size / 2))
    return min(max(start, 0), extent - size)


def crop_slices(rgb: np.ndarray, cy: float, cx: float, size: int = CACHE_SIZE):
    """Crop (Z, H, W, 3) to (Z, size, size, 3) around (cy, cx); pads with 0 if the image is smaller."""
    z, h, w, c = rgb.shape
    if h < size or w < size:
        pad = np.zeros((z, max(h, size), max(w, size), c), rgb.dtype)
        pad[:, :h, :w] = rgb
        rgb, h, w = pad, pad.shape[1], pad.shape[2]
    y0, x0 = crop_origin(cy, size, h), crop_origin(cx, size, w)
    return rgb[:, y0:y0 + size, x0:x0 + size], (y0, x0)
