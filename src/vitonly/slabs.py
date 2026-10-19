"""Per-patient tumour slabs in raw scanner units, read from either cache format.

v1 = any uint8 RGB cache from scripts/build_cache.py, both of its builder versions:
    builder 1 (`mnj-cache-*`): native pixels, phases clipped jointly at the volume's 0.5/99.5 percentiles;
    builder 2 (main branch, `mnj-cache-v2`): resampled to a fixed 0.7 mm x 2 mm grid, ceiling raised to the
    tumour's 99.9th percentile (less saturation). Stores `spacing` (z, y, x) mm.
    We map back to raw units with lo + u8 * (hi - lo) / 255 (`clip`); anything above hi is lost.
v2 (scripts/build_cache_v2.py, `mnj-cache2-*`): float16 raw / scale, full 256 plane, plus whole-volume
    statistics. Nothing is clipped, so every normalisation in norms.py can be computed exactly.

A slab is a dict: x (S, H, W, 3) stored array (decoded lazily by `raw()`), t0/t1 (tumour slices are
x[t0:t1]), bbox (y0, y1, x0, x1) in slab pixels, mask (S, H, W) or None, stats (raw-unit references).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PCTS = (0.5, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9)   # v2 `vol_pct` columns


def raw(slab, idx=None) -> np.ndarray:
    """Selected slices in raw scanner units, float32 (k, H, W, 3)."""
    x = slab["x"] if idx is None else slab["x"][idx]
    if slab["format"] == "v2":
        return x.astype(np.float32) * slab["scale"]
    lo, hi = slab["stats"]["joint_lo"], slab["stats"]["joint_hi"]
    return lo + x.astype(np.float32) * ((hi - lo) / 255.0)


def _fg_stats(pre, early, late, stride=7):
    """Mean / std / median of each phase over the pre-contrast foreground (pre > 10% of its 99.5th pct)."""
    p, e, l = (a.reshape(-1)[::stride] for a in (pre, early, late))
    fg = p > 0.1 * np.percentile(p, 99.5)
    if fg.sum() < 50:
        fg = np.ones_like(p, bool)
    return np.array([[a[fg].mean(), a[fg].std(), np.median(a[fg])] for a in (p, e, l)], np.float32)


def load_slab(path: str | Path) -> dict:
    d = np.load(path)
    t0, t1 = int(d["t0"]), int(d["t1"])
    bbox = tuple(int(v) for v in d["bbox"])
    mask = d["mask"] if "mask" in d.files else None
    if "scale" in d.files:                                   # v2 (v1 also has `phases`: the phase indices)
        x, scale = d["phases"], float(d["scale"])
        vp = d["vol_pct"]                                    # (3, len(PCTS)) raw units
        fg = d["fg_stats"]                                   # (3, 3) raw units
        lo, hi = (float(v) for v in d["clip_v1"])
        stats = {"joint_lo": lo, "joint_hi": hi, "joint_hi999": float(d["joint_p999"]),
                 "phase_lo": vp[:, PCTS.index(0.5)], "phase_hi": vp[:, PCTS.index(99.5)],
                 "fg": fg, "saturated": False}
        fmt = "v2"
    else:                                                    # v1 uint8
        x, scale = d["rgb"], 1.0
        lo, hi = (float(v) for v in d["clip"]) if "clip" in d.files else (0.0, 255.0)
        stats = {"joint_lo": lo, "joint_hi": hi, "joint_hi999": hi, "saturated": True}
        fmt = "v1"
    sp = d["spacing"] if "spacing" in d.files else None       # v2 raw: (xy, z); uint8 builder 2: (z, y, x)
    spacing_xy = None if sp is None else float(sp[0] if fmt == "v2" else sp[-1])
    spacing_z = None if sp is None else float(sp[1] if fmt == "v2" else sp[0])
    slab = {"format": fmt, "x": x, "scale": scale, "t0": t0, "t1": t1, "bbox": bbox, "mask": mask,
            "stats": stats, "spacing_xy": spacing_xy, "spacing_z": spacing_z}
    if fmt == "v1":   # references from the slab itself (the whole volume is not in a v1 cache)
        r = raw(slab, slice(None, None, max(1, x.shape[0] // 8)))
        stats["phase_lo"] = np.percentile(r.reshape(-1, 3)[::5], 0.5, axis=0).astype(np.float32)
        stats["phase_hi"] = np.percentile(r.reshape(-1, 3)[::5], 99.5, axis=0).astype(np.float32)
        stats["fg"] = _fg_stats(r[..., 0], r[..., 1], r[..., 2], stride=5)
    y0, y1, x0, x1 = bbox   # tumour-window references, over the tumour slices and box, all phases
    tum = raw(slab, slice(t0, t1))[:, y0:y1, x0:x1].reshape(-1, 3)
    stats["tumor_lo"], stats["tumor_hi"] = (float(v) for v in np.percentile(tum, [1, 99.5]))
    return slab


def cache_format(cache_dir: str | Path) -> str:
    meta = json.loads((Path(cache_dir) / "cache_meta.json").read_text())
    return meta.get("format", "v1")


def cache_label(cache_dir: str | Path) -> str:
    """'v2 raw' (this branch's raw cache) or 'v1 uint8 builder 1/2' (scripts/build_cache.py)."""
    meta = json.loads((Path(cache_dir) / "cache_meta.json").read_text())
    if meta.get("format") == "v2":
        return "v2 raw"
    return f"v1 uint8 builder {meta.get('builder_version', 1)}"


def find_caches(roots, prefer="v2"):
    """{cohort: dir}. Any folder (<= 5 levels deep) with cache_meta.json. With several caches for a cohort:
    the `prefer` format ('v2' raw or 'v1' uint8) first, and within uint8 the higher builder version
    (builder 2 = main's resampled, less-clipped cache)."""
    import glob
    import os
    prefer = {"raw": "v2", "uint8": "v1"}.get(prefer, prefer)   # plain names: raw = mnj-cache2, uint8 = mnj-cache(-v2)
    found = {}
    for r in roots:
        for depth in range(6):
            for f in glob.glob(os.path.join(r, *["*"] * depth, "cache_meta.json")):
                meta = json.load(open(f))
                rank = (meta.get("format", "v1") == prefer, meta.get("builder_version", 1))
                c = meta["cohort"]
                if c not in found or rank > found[c][0]:
                    found[c] = (rank, os.path.dirname(f))
    return {c: d for c, (_, d) in found.items()}
