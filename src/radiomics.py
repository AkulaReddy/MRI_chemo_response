"""Hand-made tumour features from one cache file (radiomics baseline, docs/ideas.md items 12 and 17).

Input: a builder v2/v3 cache `.npz` (uint8 pre/early/late on a fixed 0.7 mm grid, plus the clip range).
Intensities are converted back to raw-proportional units (cache value + lo / (hi - lo), see
`datasets.pre_offset`), so the kinetic ratios below do not depend on the scanner's gain. Texture is computed
on the relative-enhancement map with fixed bins, for the same reason.

Tumour region:
- I-SPY: the provided mask.
- Duke (boxes only): voxels inside the tumour box whose early relative enhancement is >= 70%, the
  threshold of the I-SPY functional tumour volume. If none pass, the whole box. Recorded in `mask_source`.

All features are per patient; NaN where undefined (e.g. no peritumoural voxels inside the scan).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

PE_THRESHOLD = 0.7     # early relative enhancement defining "enhancing tumour" (I-SPY FTV rule)
FLOOR = 0.05           # pre-contrast denominator floor, in units of the clip range (as in the model)
RING_MM = 5.0          # peritumoural ring width
WASHOUT = 0.1          # late vs early change (fraction of early) separating washout / plateau / persistent
RE_BINS = (0.0, 3.0, 16)   # fixed bins of the relative-enhancement map for texture and entropy


def _phases(d):
    lo, hi = (float(v) for v in d["clip"])
    off = max(lo, 0.0) / max(hi - lo, 1e-6)
    x = d["rgb"].astype(np.float32) / 255.0
    inside = d["rgb"].any(-1)                        # zero padding = outside the scan
    return x[..., 0] + off, x[..., 1] + off, x[..., 2] + off, inside


def tumour_region(d, re_e):
    """(mask (S, H, W) bool, source)."""
    if "mask" in d.files and d["mask"].any():
        return d["mask"] > 0, "mask"
    t0, t1 = int(d["t0"]), int(d["t1"])
    y0, y1, x0, x1 = (int(v) for v in d["bbox"])
    box = np.zeros(re_e.shape, bool)
    box[t0:t1, y0:y1, x0:x1] = True
    m = box & (re_e >= PE_THRESHOLD)
    return (m, "box_pe70") if m.sum() >= 10 else (box, "box")


def _glcm_features(q, mask, levels):
    """GLCM over the mask on each slice (distance 1, 4 angles, symmetric), pooled over slices.
    `q` holds levels 1..levels inside the mask; 0 outside (dropped from the matrix)."""
    from skimage.feature import graycomatrix, graycoprops
    total = np.zeros((levels + 1, levels + 1, 1, 4))
    for z in np.flatnonzero(mask.any((1, 2))):
        ys, xs = np.nonzero(mask[z])
        sl = np.where(mask[z], q[z], 0)[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
        total += graycomatrix(sl, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=levels + 1,
                              symmetric=True)
    g = total[1:, 1:]
    if g.sum() == 0:
        return {f"glcm_{k}": np.nan for k in ("contrast", "homogeneity", "energy", "correlation")}
    out = {}
    for k in ("contrast", "homogeneity", "energy", "correlation"):
        v = graycoprops(g, k)[0]
        out[f"glcm_{k}"] = float(np.nanmean(v))
    return out


def _shape_features(mask, spacing):
    from skimage.measure import marching_cubes, mesh_surface_area
    vox = float(np.prod(spacing))
    n = int(mask.sum())
    vol = n * vox
    out = {"log_volume_ml": np.log(vol / 1000.0)}
    try:
        verts, faces, _, _ = marching_cubes(np.pad(mask, 1).astype(np.float32), 0.5, spacing=tuple(spacing))
        area = mesh_surface_area(verts, faces)
        out["sphericity"] = float(np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / area)
        out["log_surface_to_volume"] = float(np.log(area / vol))
    except (ValueError, RuntimeError):            # too small for a surface
        out["sphericity"] = out["log_surface_to_volume"] = np.nan
    pts = np.argwhere(mask) * np.asarray(spacing)
    if len(pts) >= 4:
        ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1].clip(min=1e-9)
        out["major_axis_mm"] = float(4 * np.sqrt(ev[0]))
        out["elongation"] = float(np.sqrt(ev[1] / ev[0]))
        out["flatness"] = float(np.sqrt(ev[2] / ev[0]))
    else:
        out["major_axis_mm"] = out["elongation"] = out["flatness"] = np.nan
    lab, k = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())[1:]
    out["n_components"] = int((sizes * vox >= 10.0).sum())     # parts of at least 10 mm^3
    out["largest_component_frac"] = float(sizes.max() / n) if n else np.nan
    return out


def patient_features(d) -> dict:
    """Feature dict for one opened cache file (`np.load(...)`)."""
    pre, early, late, inside = _phases(d)
    den = np.maximum(pre, FLOOR)
    re_e = (early - pre) / den                           # early relative enhancement
    re_l = (late - pre) / den
    wo = (late - early) / np.maximum(early, FLOOR)       # late change relative to early
    mask, source = tumour_region(d, re_e)
    spacing = np.asarray(d["spacing"], float)
    f = {"mask_source": source, "n_voxels": int(mask.sum())}
    if mask.sum() == 0:
        return f
    f.update(_shape_features(mask, spacing))

    e, l, w = re_e[mask], re_l[mask], wo[mask]
    p10, p50, p90 = np.percentile(e, [10, 50, 90])
    f.update(re_early_mean=float(e.mean()), re_early_median=float(p50), re_early_p10=float(p10),
             re_early_p90=float(p90), re_early_std=float(e.std()), re_late_mean=float(l.mean()),
             re_late_p90=float(np.percentile(l, 90)), washout_mean=float(w.mean()),
             frac_washout=float((w < -WASHOUT).mean()), frac_plateau=float((np.abs(w) <= WASHOUT).mean()),
             frac_persistent=float((w > WASHOUT).mean()), frac_strong=float((e >= 1.0).mean()))

    lo, hi, n = RE_BINS
    hist = np.histogram(e.clip(lo, hi), bins=n, range=(lo, hi))[0] / len(e)
    nz = hist[hist > 0]
    f["re_early_entropy"] = float(-(nz * np.log2(nz)).sum())
    q = (np.clip(np.floor((re_e - lo) / (hi - lo) * n), 0, n - 1) + 1).astype(np.uint8)
    f.update(_glcm_features(q, mask, n))

    dist = ndimage.distance_transform_edt(~mask, sampling=spacing)
    ring = (dist > 0) & (dist <= RING_MM) & inside
    if ring.sum() >= 10:
        r = re_e[ring]
        f.update(ring_re_early_mean=float(r.mean()), ring_re_early_p90=float(np.percentile(r, 90)),
                 ring_frac_enhancing=float((r >= 0.5).mean()),
                 tumour_to_ring_re=float(e.mean() / max(r.mean(), 1e-3)))
    else:
        f.update(ring_re_early_mean=np.nan, ring_re_early_p90=np.nan, ring_frac_enhancing=np.nan,
                 tumour_to_ring_re=np.nan)
    return f


FEATURES = ["log_volume_ml", "sphericity", "log_surface_to_volume", "major_axis_mm", "elongation", "flatness",
            "n_components", "largest_component_frac",
            "re_early_mean", "re_early_median", "re_early_p10", "re_early_p90", "re_early_std", "re_late_mean",
            "re_late_p90", "washout_mean", "frac_washout", "frac_plateau", "frac_persistent", "frac_strong",
            "re_early_entropy", "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
            "ring_re_early_mean", "ring_re_early_p90", "ring_frac_enhancing", "tumour_to_ring_re"]
