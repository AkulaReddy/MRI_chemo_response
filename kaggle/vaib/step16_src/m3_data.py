"""Step 16 data: 2D RGB-fused tumour slices, the BreastDCEDL paper's input (pre -> R, early -> G, late -> B).

Two versions per patient, both (n, 224, 224, 3) uint8:
  paper  as in the paper's code: slices range(mid - 2, mid + 2) around the tumour's middle slice, a 224-pixel crop
         at the native pixel size centred on the tumour box, min-max scaled per image over its three phases
  fixed  every tumour slice (up to `fix_max`, evenly spaced), the same physical field of view for every patient
         (`fix_crop_mm`, resized to 224), one intensity window per patient shared by all its slices and phases
         (keeps the enhancement and the slice-to-slice differences); each slice keeps its tumour area
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

import mnj_common as mc


def _crop(a, cy, cx, side):
    """(S, H, W) -> (S, side, side) centred on (cy, cx), zero padded outside the image."""
    S, H, W = a.shape
    y0, x0 = int(round(cy - side / 2)), int(round(cx - side / 2))
    o = np.zeros((S, side, side), np.float32)
    sy0, sy1, sx0, sx1 = max(0, y0), min(H, y0 + side), max(0, x0), min(W, x0 + side)
    if sy1 > sy0 and sx1 > sx0:
        o[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[:, sy0:sy1, sx0:sx1]
    return o


def _resize2d(a, size):
    """(S, h, w) -> (S, size, size), linear (light smoothing first when shrinking, against aliasing)."""
    S, h, w = a.shape
    if h == size and w == size:
        return a.astype(np.float32)
    f = h / size
    if f > 1.5:
        a = ndi.gaussian_filter(a, (0, 0.4 * f, 0.4 * f))
    yy = (np.arange(size) + 0.5) * (h / size) - 0.5
    xx = (np.arange(size) + 0.5) * (w / size) - 0.5
    gy, gx = np.meshgrid(yy, xx, indexing="ij")
    out = np.empty((S, size, size), np.float32)
    for s in range(S):
        out[s] = ndi.map_coordinates(a[s], [gy, gx], order=1, mode="nearest")
    return out


def _to_u8(x, lo, hi):
    return np.clip((x - lo) / np.maximum(hi - lo, 1e-6) * 255.0, 0, 255).round().astype(np.uint8)


def make_slices(pid, cohort, path, meta_row, clip=None, size=224, paper_n=4, fix_max=16, fix_crop_mm=100.0,
                pe_thr=0.7):
    out = {"pid": pid, "cohort": cohort}
    try:
        c = mc.load_case(path, meta_row, clip)
        pre, early, late = c["pre"], c["early"], c["late"]
        t0, t1 = c["t0"], c["t1"]
        y0, y1, x0, x1 = c["bbox"]
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2

        mid = (t0 + t1 - 1) // 2
        zp = list(range(max(mid - paper_n // 2, t0), min(mid + paper_n // 2, t1))) or [mid]
        P = np.stack([_crop(v[zp], cy, cx, size) for v in (pre, early, late)], -1)
        paper = _to_u8(P, P.min((1, 2, 3), keepdims=True), P.max((1, 2, 3), keepdims=True))

        zf = np.unique(np.round(np.linspace(t0, t1 - 1, min(t1 - t0, fix_max))).astype(int))
        side = max(16, int(round(fix_crop_mm / c["dxy"])))
        Fx = np.stack([_resize2d(_crop(v[zf], cy, cx, side), size) for v in (pre, early, late)], -1)
        box = np.stack([v[t0:t1, y0:y1, x0:x1] for v in (pre, early, late)])
        lo = float(np.percentile(Fx, 0.5))
        hi = max(float(np.percentile(box, 99.5)), lo + 1e-3)
        fixed = _to_u8(Fx, lo, hi)
        roi, method = mc.rule_roi(pre, early, t0, t1, c["bbox"], pe_thr)
        area = (roi[zf].sum((1, 2)) * c["dxy"] ** 2).astype(np.float32)

        out.update(paper=paper, paper_z=np.asarray(zp), fixed=fixed, fixed_z=zf, fixed_area=area, status="ok",
                   roi_method=method, n_tumour_slices=int(t1 - t0), n_paper=len(zp), n_fixed=len(zf),
                   paper_fov_mm=size * c["dxy"], fixed_fov_mm=side * c["dxy"], clip_known=c["clip_known"],
                   tumour_saturated=float(c["sat"][t0:t1, y0:y1, x0:x1].mean()) if c["sat"] is not None else np.nan)
    except Exception as e:  # noqa: BLE001
        out.update(status="error", note=f"{type(e).__name__}: {e}")
    return out
