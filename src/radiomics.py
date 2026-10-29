"""Hand-made tumour features from one cache file (radiomics baseline, docs/ideas.md items 12 and 17).

Every feature is computed in 3D over the whole tumour (all its slices), including the texture
(13-direction 3D GLCM; up to d1.10 the GLCM was 2D per slice, features `glcm_*`).

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


# the 13 unique neighbour directions of a voxel in 3D (the other 13 are their opposites)
DIRECTIONS_3D = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                 if (dz, dy, dx) > (0, 0, 0)]


def glcm_3d(q, mask, levels):
    """Grey-level co-occurrence over the tumour in 3D: for each of the 13 directions, every pair of
    neighbouring tumour voxels (both inside `mask`) counts once in each order (symmetric). `q` holds levels
    0..levels-1. Returns (13, levels, levels) probabilities (directions without pairs are all zero)."""
    out = np.zeros((len(DIRECTIONS_3D), levels, levels))
    for n, off in enumerate(DIRECTIONS_3D):
        a = tuple(slice(max(0, -o), q.shape[i] - max(0, o)) for i, o in enumerate(off))
        b = tuple(slice(max(0, o), q.shape[i] - max(0, -o)) for i, o in enumerate(off))
        both = mask[a] & mask[b]
        if not both.any():
            continue
        m = np.bincount(q[a][both] * levels + q[b][both], minlength=levels * levels).reshape(levels, levels)
        m = m + m.T
        out[n] = m / m.sum()
    return out


def _glcm_features(q, mask, levels):
    """Texture of the enhancement map from the 3D GLCM, averaged over the 13 directions (IBSI-style
    contrast, homogeneity, energy = sqrt(angular second moment), correlation). The voxels are 0.7 x 0.7 x
    2.0 mm, so the in-plane and through-plane neighbours are at different distances."""
    P = glcm_3d(q, mask, levels)
    P = P[P.sum((1, 2)) > 0]
    if not len(P):
        return {f"glcm3d_{k}": np.nan for k in ("contrast", "homogeneity", "energy", "correlation")}
    i, j = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    con = (P * (i - j) ** 2).sum((1, 2))
    hom = (P / (1 + (i - j) ** 2)).sum((1, 2))
    ene = np.sqrt((P ** 2).sum((1, 2)))
    mi, mj = (P * i).sum((1, 2)), (P * j).sum((1, 2))
    si = np.sqrt((P * (i - mi[:, None, None]) ** 2).sum((1, 2)))
    sj = np.sqrt((P * (j - mj[:, None, None]) ** 2).sum((1, 2)))
    cov = (P * (i - mi[:, None, None]) * (j - mj[:, None, None])).sum((1, 2))
    cor = np.where(si * sj > 0, cov / np.maximum(si * sj, 1e-12), 1.0)   # a flat region is fully correlated
    return {"glcm3d_contrast": float(con.mean()), "glcm3d_homogeneity": float(hom.mean()),
            "glcm3d_energy": float(ene.mean()), "glcm3d_correlation": float(cor.mean())}


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
    q = np.clip(np.floor((re_e - lo) / (hi - lo) * n), 0, n - 1).astype(np.int64)
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
            "re_early_entropy", "glcm3d_contrast", "glcm3d_homogeneity", "glcm3d_energy", "glcm3d_correlation",
            "ring_re_early_mean", "ring_re_early_p90", "ring_frac_enhancing", "tumour_to_ring_re"]
