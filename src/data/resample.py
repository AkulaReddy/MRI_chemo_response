"""Resampling to a fixed physical grid around the tumour (builder v2, see challenges.md / EDA C8).

Every patient is resampled to the same voxel size (default 2.0 mm between slices, 0.7 mm in-plane),
on a window centred on the tumour box. The ViT then sees tumours at the same physical scale in every
cohort: a 224-px input covers 224 x 0.7 = 157 mm for everyone (in v1 it was 70-315 mm).

Coordinates: a voxel index i covers [i, i + 1) in "edge" coordinates, so its centre is i + 0.5. Box
coordinates (`[start, end)`) are edge coordinates. Out-of-volume voxels are 0 (zero padding).
Downsampling is anti-aliased with a Gaussian (sigma = (factor - 1) / 2 voxels, as in skimage).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def axis_grid(center_edge: float, src_mm: float, dst_mm: float, n: int):
    """(scale, offset) such that output index o samples source index o * scale + offset, for n
    output voxels of `dst_mm` centred on `center_edge` (source edge coordinates)."""
    scale = dst_mm / src_mm
    return scale, center_edge - 0.5 - (n - 1) / 2 * scale


def tumour_z_grid(z0: int, z1: int, depth: int, src_mm: float, dst_mm: float, margin: int):
    """Output slices of `dst_mm` over the tumour's physical z-extent [z0, z1) plus `margin` slices on
    each side, dropping margin slices whose centre falls outside the volume.

    Returns (scale, offset, n_slices, t0, t1): tumour slices are output[t0:t1]."""
    n_t = max(1, int(round((z1 - z0) * src_mm / dst_mm)))
    scale, offset = axis_grid((z0 + z1) / 2, src_mm, dst_mm, n_t)
    lo, hi = -margin, n_t + margin                                  # candidate output indices
    centres = np.arange(lo, hi) * scale + offset                    # source centre coordinates
    keep = np.flatnonzero((centres >= -0.5) & (centres <= depth - 0.5))
    first, last = lo + keep[0], lo + keep[-1] + 1
    return scale, float(offset + first * scale), int(last - first), int(-first), int(n_t - first)


def resample(vol: np.ndarray, scales, offsets, shape, order: int = 1, antialias: bool = True):
    """Sample `vol` (Z, H, W) on the axis-aligned grid out[o] = vol[o * scale + offset] (per axis).

    Only the source region the grid touches is filtered and read, so large volumes stay cheap."""
    scales, offsets = np.asarray(scales, float), np.asarray(offsets, float)
    sigma = np.where(antialias & (scales > 1), (scales - 1) / 2, 0.0)
    pad = np.ceil(3 * sigma).astype(int) + 2
    lo = np.floor(offsets).astype(int) - pad
    hi = np.ceil(offsets + (np.asarray(shape) - 1) * scales).astype(int) + pad + 1
    src_lo, src_hi = np.maximum(lo, 0), np.minimum(hi, vol.shape)
    if np.any(src_hi <= src_lo):                                     # grid entirely outside
        return np.zeros(shape, np.float32)
    sub = np.zeros(tuple(hi - lo), np.float32)                       # zero-padded source window
    sub[tuple(slice(a - l, b - l) for a, b, l in zip(src_lo, src_hi, lo))] = \
        vol[tuple(slice(a, b) for a, b in zip(src_lo, src_hi))]
    if sigma.any():
        sub = ndimage.gaussian_filter(sub, sigma, mode="constant", cval=0.0)
    return ndimage.affine_transform(sub, scales, offset=offsets - lo, output_shape=tuple(shape),
                                    order=order, mode="constant", cval=0.0, prefilter=False)
