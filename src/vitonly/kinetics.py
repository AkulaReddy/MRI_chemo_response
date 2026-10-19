"""Hand-crafted DCE kinetic features of the tumour: an image-only reference that needs no deep learning.

If these reach val AUC ~0.6 while a ViT stays at 0.5, the images carry signal and the ViT is failing to use
it. The tumour ROI is defined the same way for every cohort: voxels inside the tumour box and slice range
with percent enhancement PE = (early - pre) / pre >= 0.7 (the I-SPY functional-tumour-volume threshold).
The I-SPY masks are not used, because Duke has boxes only and a cohort-specific ROI would leak the cohort.
On v1 caches, enhancement above the 99.5th percentile is clipped, so PE and SER are underestimated.
"""
from __future__ import annotations

import numpy as np

from .slabs import raw

FEATURES = ["log_ftv_mm3", "log_box_mm3", "ftv_frac", "pe_mean", "pe_p50", "pe_p90", "pe_std",
            "ser_mean", "ser_p90", "frac_washout", "frac_plateau", "frac_persistent",
            "pre_rel", "early_rel", "late_rel", "early_rel_p90"]


def kinetic_features(slab, spacing_xy: float, slice_thick: float, pe_thr: float = 0.7) -> np.ndarray:
    t0, t1 = slab["t0"], max(slab["t1"], slab["t0"] + 1)
    y0, y1, x0, x1 = slab["bbox"]
    x = raw(slab, slice(t0, t1))[:, y0:y1, x0:x1].reshape(-1, 3)
    pre, early, late = x[:, 0], x[:, 1], x[:, 2]
    r = max(float(slab["stats"]["fg"][0, 0]), 1e-6)
    pe = (early - pre) / np.maximum(pre, 0.2 * r)
    roi = pe >= pe_thr
    if roi.sum() < 10:                                   # weakly enhancing tumour: top 10% PE of the box
        roi = pe >= np.quantile(pe, 0.9)
    ser = (early - pre)[roi] / np.maximum((late - pre)[roi], 0.05 * r)
    spacing_xy, slice_thick = slab.get("spacing_xy") or spacing_xy, slab.get("spacing_z") or slice_thick
    vox = spacing_xy ** 2 * slice_thick                     # cached voxel size (resampled caches differ from CSV)
    f = [np.log(roi.sum() * vox + 1), np.log(len(pe) * vox + 1), roi.mean(),
         pe[roi].mean(), np.median(pe[roi]), np.quantile(pe[roi], 0.9), pe[roi].std(),
         np.clip(ser, 0, 5).mean(), np.quantile(np.clip(ser, 0, 5), 0.9),
         (ser > 1.1).mean(), ((ser >= 0.9) & (ser <= 1.1)).mean(), (ser < 0.9).mean(),
         pre[roi].mean() / r, early[roi].mean() / r, late[roi].mean() / r, np.quantile(early[roi], 0.9) / r]
    return np.asarray(f, np.float32)
