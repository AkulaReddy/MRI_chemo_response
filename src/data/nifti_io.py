"""NIfTI reading for BreastDCEDL volumes.

Checked on the Zenodo demo archive (2026-10-08): every volume is stored as (Z, 256, 256) float64,
axis 0 = slice, header spacing 1 mm (not the real spacing; use the CSV `xy_spacing`/`slice_thick`).
We do not reorient: the CSV coordinates (`sraw/eraw` = axis 1, `scol/ecol` = axis 2,
`mask_start/mask_end` = axis 0) refer to the array as stored.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import nibabel as nib
import numpy as np


def _from_bytes(raw: bytes):
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return nib.Nifti1Image.from_bytes(raw)


def load_volume(src: str | Path | bytes, dtype=np.float32) -> np.ndarray:
    """A phase volume (or mask) as a (Z, H, W) array. `src` is a path or the file's raw bytes."""
    img = _from_bytes(src) if isinstance(src, (bytes, bytearray)) else nib.load(str(src))
    a = np.asarray(img.dataobj, dtype=dtype)
    if a.ndim == 4 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {a.shape}")
    return a


def mask_bbox(mask: np.ndarray):
    """[start, end) per axis of a binary mask, or None if empty: ((z0, z1), (y0, y1), (x0, x1))."""
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return None
    lo, hi = idx.min(0), idx.max(0) + 1
    return tuple((int(a), int(b)) for a, b in zip(lo, hi))
