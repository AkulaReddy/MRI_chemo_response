"""Input checks used by kaggle/vaib/01_input_diagnostics: saturation, tumour size in the crop, galleries."""
from __future__ import annotations

import numpy as np

from .norms import CROPS, NORMS, crop_side, make_views, normalise, select_slices
from .slabs import raw


def tumor_voxels(slab):
    """(n, 3) raw values of the tumour: the mask where there is one, else the box (Duke)."""
    t0, t1 = slab["t0"], max(slab["t1"], slab["t0"] + 1)
    x = raw(slab, slice(t0, t1))
    if slab["mask"] is not None and slab["mask"][t0:t1].any():
        return x[slab["mask"][t0:t1] > 0], "mask"
    y0, y1, x0, x1 = slab["bbox"]
    return x[:, y0:y1, x0:x1].reshape(-1, 3), "box"


def saturation(slab, norms=NORMS, k=8):
    """{norm: (3,) fraction of tumour pixels at the top of the range (>= 254/255) per channel}, over up to
    k tumour slices. Tumour = mask, or the box for Duke. On v1 caches this misses what v1 already clipped:
    compare against `v1_clipped`."""
    idx = select_slices(slab, k, "even")
    x = raw(slab, idx)
    if slab["mask"] is not None and slab["mask"][idx].any():
        sel = slab["mask"][idx] > 0
    else:
        y0, y1, x0, x1 = slab["bbox"]
        sel = np.zeros(x.shape[:3], bool)
        sel[:, y0:y1, x0:x1] = True
    return {n: (normalise(x, slab["stats"], n)[sel] >= 254 / 255).mean(0) for n in norms}


def v1_clipped(slab):
    """(3,) fraction of tumour voxels above the v1 ceiling (joint 99.5th pct). Exact on v2 caches; on v1
    caches it counts the voxels stored at 255."""
    tum, _ = tumor_voxels(slab)
    return (tum >= slab["stats"]["joint_hi"] - 1e-3 * abs(slab["stats"]["joint_hi"])).mean(0)


def tumor_fraction(slab, spacing_xy, crops=CROPS, zoom=1.6, fov_mm=100.0):
    """{crop: fraction of the 224 network input taken by the tumour box's longer side}."""
    y0, y1, x0, x1 = slab["bbox"]
    side_t = max(y1 - y0, x1 - x0)
    spacing_xy = slab.get("spacing_xy") or spacing_xy
    out = {}
    for c in crops:
        side = crop_side(c, slab["bbox"], spacing_xy, min(slab["x"].shape[1:3]), zoom, fov_mm)
        out[c] = min(1.0, side_t / side * 240 / 224)       # the eval centre crop keeps 224 of 240
    return out


def gallery(ax_row, slab, spacing_xy, norms, crop="tumor", contour=True):
    """One middle-tumour slice of a patient under every normalisation, mask contour in white."""
    import torch.nn.functional as F
    idx = select_slices(slab, 1, "mid")
    for ax, n in zip(ax_row, norms):
        v = make_views(slab, idx, n, crop, spacing_xy)[0].permute(1, 2, 0).numpy()
        ax.imshow(v)
        ax.axis("off")
        if contour and slab["mask"] is not None:
            from .norms import crop_resize
            side = crop_side(crop, slab["bbox"], slab.get("spacing_xy") or spacing_xy, min(slab["x"].shape[1:3]))
            m = crop_resize(slab["mask"][idx][..., None].astype(np.float32).repeat(3, -1), slab["bbox"], side)
            if m.max() > 0:
                ax.contour(m[0, 0].numpy(), levels=[0.5], colors="white", linewidths=0.5)
    return idx
