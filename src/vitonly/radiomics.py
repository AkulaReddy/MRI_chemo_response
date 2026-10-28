"""Hand-crafted tumour features in 2D, 3D and per slice over time (professor's suggestions, 2026-10-28).

Three families, all from the same tumour region of interest (ROI), defined the same way for every cohort as in
kinetics.py: voxels inside the tumour box and slice range whose percent enhancement PE = (early - pre) / pre
is >= 0.7 (top 10% of the box if fewer than 10 voxels pass). The I-SPY masks are not used (Duke has boxes only).

    2d        the slice with the largest enhancing area: first-order statistics of 7 maps, 2D shape, 2D GLCM texture
    3d        the whole tumour volume: first-order statistics of the same 7 maps, 3D shape, 3D GLCM texture
    temporal  "temporal mapping of slices": every tumour slice's pre -> early -> late curve becomes per-slice
              kinetic numbers (enhancing area, PE, SER, wash-out fraction, ...), then aggregated across the slices
              (mean, std, max, mean of the 3 slices with the largest enhancing area, slice-to-slice change)

The 7 maps: pre, early, late (intensity, divided by a reference r), PE, SER (signal enhancement ratio
(early - pre) / (late - pre)), wash-in (early - pre) / r and wash-out (late - early) / r.
`ref` chooses r, the only thing the intensity normalisation changes:
    volume    r = mean pre-contrast foreground of the whole volume (one number per patient)
    slice     r = mean pre-contrast foreground of each slice on its own (the professor's per-slice normalisation)
PE and SER are ratios, so they do not depend on r.

The definitions follow pyradiomics / IBSI (first order, GLCM with 32 fixed bins over the ROI's 1-99th
percentile range, symmetric, averaged over directions: 4 in 2D, 13 in 3D). They are written in numpy because
pyradiomics needs a compiled install that often fails on Kaggle's Python. Uint8 caches (mnj-cache-*) clip the
brightest enhancement, so intensity and PE features are underestimated there; raw caches (mnj-cache2-*) are exact.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .norms import slice_fg_stats
from .slabs import raw

MAPS = ("pre", "early", "late", "pe", "ser", "washin", "washout")
FO = ("mean", "std", "p10", "p50", "p90", "skew", "kurt", "entropy")
GLCM = ("contrast", "dissimilarity", "homogeneity", "energy", "entropy", "correlation")
GLCM_MAPS = ("early", "pe")
SHAPE3D = ("log_vol_mm3", "log_box_mm3", "roi_frac", "surf_vol_ratio", "sphericity", "elongation", "flatness",
           "major_axis_mm", "n_components", "largest_comp_frac", "n_slices")
SHAPE2D = ("log_area_mm2", "perimeter_mm", "circularity", "elongation", "major_axis_mm", "n_components",
           "roi_frac", "slice_pos")
PER_SLICE = ("log_area_mm2", "pe_mean", "pe_p90", "ser_mean", "frac_washout", "frac_persistent", "early_rel")
AGG = ("mean", "std", "max", "top3")
DIRS_2D = ((0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, -1))
DIRS_3D = tuple(d for d in ((a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1))
                if d > (0, 0, 0))                                             # the 13 unique 3D directions
PE_THR = 0.7
FAMILIES = ("2d", "3d", "temporal")


def feature_names(family: str) -> list[str]:
    if family == "3d":
        return ([f"3d_fo_{m}_{s}" for m in MAPS for s in FO] + [f"3d_shape_{s}" for s in SHAPE3D]
                + [f"3d_glcm_{m}_{s}" for m in GLCM_MAPS for s in GLCM])
    if family == "2d":
        return ([f"2d_fo_{m}_{s}" for m in MAPS for s in FO] + [f"2d_shape_{s}" for s in SHAPE2D]
                + [f"2d_glcm_{m}_{s}" for m in GLCM_MAPS for s in GLCM])
    if family == "temporal":
        return [f"tmp_{f}_{a}" for f in PER_SLICE for a in AGG] + [f"tmp_{f}_slice_change" for f in
                                                                  ("pe_mean", "ser_mean", "frac_washout")]
    raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")


# ------------------------------------------------------------------------------------------- building blocks
def first_order(v: np.ndarray) -> list[float]:
    if v.size < 5:
        return [np.nan] * len(FO)
    m, sd = float(v.mean()), float(v.std())
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    z = (v - m) / sd if sd > 0 else np.zeros_like(v)
    h, _ = np.histogram(v, bins=32)
    h = h[h > 0] / v.size
    return [m, sd, p10, p50, p90, float((z ** 3).mean()), float((z ** 4).mean() - 3), float(-(h * np.log2(h)).sum())]


def quantise(img: np.ndarray, roi: np.ndarray, levels: int = 32) -> np.ndarray:
    """ROI values -> 0..levels-1 over the ROI's 1-99th percentile range; -1 outside the ROI."""
    q = np.full(img.shape, -1, np.int64)
    v = img[roi]
    if v.size == 0:
        return q
    lo, hi = np.percentile(v, [1, 99])
    q[roi] = np.clip(((v - lo) / max(hi - lo, 1e-9) * levels).astype(np.int64), 0, levels - 1)
    return q


def glcm_features(q: np.ndarray, dirs, levels: int = 32) -> list[float]:
    """Symmetric normalised grey-level co-occurrence matrix, one per direction, features averaged."""
    feats = []
    i, j = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    for d in dirs:
        sl_a, sl_b = [], []
        for off, n in zip(d, q.shape):
            if off >= 0:
                sl_a.append(slice(0, n - off)), sl_b.append(slice(off, n))
            else:
                sl_a.append(slice(-off, n)), sl_b.append(slice(0, n + off))
        a, b = q[tuple(sl_a)].ravel(), q[tuple(sl_b)].ravel()
        ok = (a >= 0) & (b >= 0)
        if ok.sum() < 10:
            continue
        P = np.bincount(a[ok] * levels + b[ok], minlength=levels * levels).reshape(levels, levels).astype(float)
        P = P + P.T
        P /= P.sum()
        mu_i, mu_j = (i * P).sum(), (j * P).sum()
        sd_i, sd_j = np.sqrt(((i - mu_i) ** 2 * P).sum()), np.sqrt(((j - mu_j) ** 2 * P).sum())
        nz = P[P > 0]
        feats.append([((i - j) ** 2 * P).sum(), (np.abs(i - j) * P).sum(), (P / (1 + (i - j) ** 2)).sum(),
                      (P ** 2).sum(), -(nz * np.log2(nz)).sum(),
                      ((i - mu_i) * (j - mu_j) * P).sum() / (sd_i * sd_j) if sd_i * sd_j > 0 else 1.0])
    return list(np.mean(feats, 0)) if feats else [np.nan] * len(GLCM)


def _pca_axes(coords: np.ndarray) -> np.ndarray:
    """Eigenvalues (descending) of the ROI coordinate covariance (in mm)."""
    if len(coords) < 4:
        return np.array([np.nan] * coords.shape[1])
    return np.sort(np.linalg.eigvalsh(np.cov(coords.T)))[::-1].clip(0)


def _faces(roi: np.ndarray, areas) -> float:
    """Boundary area: ROI voxel faces that touch a non-ROI voxel (or the edge), times the face area per axis."""
    total = 0.0
    p = np.pad(roi, 1)
    for ax, a in enumerate(areas):
        total += a * np.count_nonzero(np.diff(p.astype(np.int8), axis=ax))
    return total


# ------------------------------------------------------------------------------------------------- extraction
def tumour_maps(slab, ref: str = "volume", pe_thr: float = PE_THR):
    """The 7 maps over the tumour box and slices (dict of (S, h, w) arrays) and the ROI (S, h, w) bool."""
    t0, t1 = slab["t0"], max(slab["t1"], slab["t0"] + 1)
    y0, y1, x0, x1 = slab["bbox"]
    full = raw(slab, slice(t0, t1))                                      # (S, H, W, 3) whole planes
    r_vol = max(float(slab["stats"]["fg"][0, 0]), 1e-6)
    if ref == "slice":
        r = slice_fg_stats(full[..., 0], r_vol)[0][:, None, None]      # each slice's own reference
    elif ref == "volume":
        r = np.full((full.shape[0], 1, 1), r_vol, np.float32)
    else:
        raise ValueError(f"unknown ref {ref!r}: volume | slice")
    x = full[:, y0:y1, x0:x1]
    pre, early, late = x[..., 0], x[..., 1], x[..., 2]
    pe = (early - pre) / np.maximum(pre, 0.2 * r)                         # as kinetics.py
    roi = pe >= pe_thr
    if roi.sum() < 10:
        roi = pe >= np.quantile(pe, 0.9)
    ser = np.clip((early - pre) / np.maximum(late - pre, 0.05 * r), 0, 5)
    maps = {"pre": pre / r, "early": early / r, "late": late / r, "pe": pe, "ser": ser,
            "washin": (early - pre) / r, "washout": (late - early) / r}
    return maps, roi


def features_3d(maps, roi, sxy: float, sz: float) -> list[float]:
    fo = [f for m in MAPS for f in first_order(maps[m][roi])]
    vox = sxy * sxy * sz
    n = int(roi.sum())
    vol = n * vox
    area = _faces(roi, (sxy * sxy, sxy * sz, sxy * sz))
    ev = _pca_axes(np.argwhere(roi) * np.array([sz, sxy, sxy]))
    lab, ncomp = ndimage.label(roi)
    largest = np.bincount(lab.ravel())[1:].max() / max(n, 1) if ncomp else np.nan
    shape = [np.log(vol + 1), np.log(roi.size * vox + 1), n / roi.size, area / max(vol, 1e-9),
             np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / max(area, 1e-9),
             np.sqrt(ev[1] / ev[0]) if ev[0] > 0 else np.nan, np.sqrt(ev[2] / ev[0]) if ev[0] > 0 else np.nan,
             4 * np.sqrt(ev[0]), ncomp, largest, roi.shape[0]]
    tex = [f for m in GLCM_MAPS for f in glcm_features(quantise(maps[m], roi), DIRS_3D)]
    return fo + shape + tex


def features_2d(maps, roi, sxy: float) -> list[float]:
    areas = roi.reshape(roi.shape[0], -1).sum(1)
    z = int(np.argmax(areas))
    r2 = roi[z]
    fo = [f for m in MAPS for f in first_order(maps[m][z][r2])]
    a = r2.sum() * sxy * sxy
    per = _faces(r2, (sxy, sxy))
    ev = _pca_axes(np.argwhere(r2) * sxy)
    _, ncomp = ndimage.label(r2)
    shape = [np.log(a + 1), per, 4 * np.pi * a / max(per ** 2, 1e-9),
             np.sqrt(ev[1] / ev[0]) if ev[0] > 0 else np.nan, 4 * np.sqrt(ev[0]), ncomp, r2.mean(),
             (z + 0.5) / roi.shape[0]]
    tex = [f for m in GLCM_MAPS for f in glcm_features(quantise(maps[m][z:z + 1], roi[z:z + 1]), DIRS_2D)]
    return fo + shape + tex


def features_temporal(maps, roi, sxy: float) -> list[float]:
    """Per-slice kinetic curve summaries (each slice's pre -> early -> late), aggregated over the slices."""
    rows = []
    for z in range(roi.shape[0]):
        m = roi[z]
        if m.sum() < 3:
            continue
        pe, ser = maps["pe"][z][m], maps["ser"][z][m]
        rows.append([np.log(m.sum() * sxy * sxy + 1), pe.mean(), np.percentile(pe, 90), ser.mean(),
                     (ser > 1.1).mean(), (ser < 0.9).mean(), maps["early"][z][m].mean()])
    n_out = len(PER_SLICE) * len(AGG) + 3
    if not rows:
        return [np.nan] * n_out
    s = np.asarray(rows, float)                                            # (slices, PER_SLICE)
    top = s[np.argsort(-s[:, 0])[:3]].mean(0)                              # 3 slices with the largest area
    agg = np.stack([s.mean(0), s.std(0), s.max(0), top], 1).ravel()       # PER_SLICE x AGG, row-major
    change = [np.abs(np.diff(s[:, k])).mean() if len(s) > 1 else 0.0 for k in (1, 3, 4)]
    return list(agg) + change


def extract(slab, spacing_xy: float, slice_thick: float, ref: str = "volume") -> dict:
    """{feature name: value} for the 2d, 3d and temporal families of one patient."""
    sxy = slab.get("spacing_xy") or spacing_xy                             # resampled caches store their own
    sz = slab.get("spacing_z") or slice_thick
    maps, roi = tumour_maps(slab, ref)
    out = {}
    for fam, vals in (("3d", features_3d(maps, roi, sxy, sz)), ("2d", features_2d(maps, roi, sxy)),
                      ("temporal", features_temporal(maps, roi, sxy))):
        names = feature_names(fam)
        assert len(names) == len(vals), (fam, len(names), len(vals))
        out.update(zip(names, (float(v) for v in vals)))
    return out
