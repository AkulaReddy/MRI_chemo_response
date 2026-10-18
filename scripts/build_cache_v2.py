"""Build the raw-intensity slab cache (format v2) from raw BreastDCEDL archives.

    python scripts/build_cache_v2.py --cohort ispy1 --raw /kaggle/input/mnj-raw-ispy1 \
        --out /kaggle/working/cache2_ispy1 [--limit 10] [--workers 4] [--all-patients]

Why a second cache: v1 (`build_cache.py`) stores uint8 RGB clipped jointly at the volume's 99.5th percentile,
so enhancing tumour can saturate and only one normalisation is possible. v2 keeps the raw values (float16,
divided by the volume's joint 99.9th percentile) and the whole-volume statistics, so every normalisation in
src/vitonly/norms.py can be computed at training time without a rebuild. By default only pCR-labeled
patients are written (the image-only models use nothing else), which keeps I-SPY2 under ~10 GB.

Per patient `<pid>.npz`:
    phases     (S, 256, 256, 3) float16  raw / scale; R/G/B = pre/early/late; tumour slices +-2 (MARGIN)
    scale      joint 99.9th percentile of the whole volume (raw units; = joint_p999)
    clip_v1    (2,) the joint 0.5/99.5 percentiles v1 used, so norm `joint_pct` reproduces the v1 images
    vol_pct    (3, 10) per-phase whole-volume percentiles at slabs.PCTS (raw units)
    fg_stats   (3, 3) per-phase mean, std, median over the pre-contrast foreground (raw units)
    t0, t1, z_offset, bbox (y0, y1, x0, x1), crop_origin, phases (used), spacing (xy, z), mask (I-SPY)
plus `cache_index.csv` (one QC row per patient, same columns as v1 where they apply) and `cache_meta.json`
with `"format": "v2"`. Tumour location follows build_cache.py exactly (see challenges.md, item 3).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import build_cache as bc  # noqa: E402  (Source, resolve_phases, MARGIN: shared with v1)
from src.data.metadata import COHORTS, load_metadata  # noqa: E402
from src.data.nifti_io import load_volume, mask_bbox  # noqa: E402
from src.data.rgb_slices import crop_slices  # noqa: E402
from src.vitonly.slabs import PCTS, _fg_stats  # noqa: E402

BUILDER_VERSION = 2
PLANE = 256


def locate(row, shape, mask, qc):
    """Tumour z-range and in-plane box in volume voxels; the same rules as build_cache.process()."""
    depth, h, w = shape
    mb = mask_bbox(mask) if mask is not None else None
    y0, y1, x0, x1 = (int(row[c]) for c in ("sraw", "eraw", "scol", "ecol"))
    n_xy, qc["xy_source"] = int(row["n_xy"]), "csv"
    if (h, w) != (n_xy, n_xy):
        qc["note"] += f"in-plane {h}x{w} != n_xy {n_xy}; "
        if n_xy > h and n_xy > w:    # original cropped to a window centred on the box
            oy, ox = (int(min(max((a + b) // 2 - 127, 0), n_xy - size))
                      for a, b, size in ((y0, y1, h), (x0, x1, w)))
            y0, y1, x0, x1 = max(0, y0 - oy), min(h, y1 - oy), max(0, x0 - ox), min(w, x1 - ox)
            qc["xy_source"] = "csv_shifted"
        else:
            sy, sx = h / n_xy, w / n_xy
            y0, y1 = int(np.floor(y0 * sy)), min(h, int(np.ceil(y1 * sy)))
            x0, x1 = int(np.floor(x0 * sx)), min(w, int(np.ceil(x1 * sx)))
            qc["xy_source"] = "csv_scaled"
    if depth == int(row["n_z"]):
        z0, z1, qc["z_source"] = int(row["mask_start"]), int(row["mask_end"]), "csv"
    elif mb is not None:
        (z0, z1), qc["z_source"] = mb[0], "mask"
    else:
        raise ValueError(f"depth {depth} != n_z {int(row['n_z'])} and no mask to locate the tumor")
    if mb is not None and qc["xy_source"] != "csv":
        (_, _), (y0, y1), (x0, x1) = mb
        qc["xy_source"] = "mask"
    if z1 <= z0 and mb is not None:   # empty CSV z-range: fall back to the mask
        (z0, z1), qc["z_source"] = mb[0], "mask"
    if not (0 <= z0 < z1 <= depth and 0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
        raise ValueError(f"bbox z{z0}:{z1} y{y0}:{y1} x{x0}:{x1} outside volume {shape}")
    return z0, z1, y0, y1, x0, x1


def process(pid, row, phase_src, mask_src, out_dir):
    """QC dict; writes <pid>.npz on success. Runs in a worker process."""
    qc = {"pid": pid, "status": "ok", "note": ""}
    try:
        vols = [load_volume(phase_src[k]) for k in row["phases_used"]]
        depth, h, w = vols[0].shape
        qc.update(depth=depth, n_z_csv=int(row["n_z"]), h=h, w=w)
        mask = load_volume(mask_src, np.uint8) if mask_src is not None else None
        if mask is not None and mask.shape != vols[0].shape:
            qc["note"] += f"mask shape {mask.shape} != volume; mask ignored; "
            mask = None
        z0, z1, y0, y1, x0, x1 = locate(row, vols[0].shape, mask, qc)
        stack = np.stack(vols, axis=-1)                                     # (Z, H, W, 3) float32
        flat = stack.reshape(-1)[::4]                                       # same sample as v1
        lo, hi, p999 = (float(v) for v in np.percentile(flat, [0.5, 99.5, 99.9]))
        vol_pct = np.stack([np.percentile(stack[..., c].reshape(-1)[::4], PCTS) for c in range(3)])
        fg = _fg_stats(*vols)
        scale = max(p999, 1e-6)
        s0, s1 = max(0, z0 - bc.MARGIN), min(depth, z1 + bc.MARGIN)
        slab, (cy0, cx0) = crop_slices(stack[s0:s1], (y0 + y1) / 2, (x0 + x1) / 2, PLANE)
        arrays = dict(phases=(slab / scale).astype(np.float16), scale=np.float32(scale),
                      joint_p999=np.float32(p999), clip_v1=np.array([lo, hi], np.float32),
                      vol_pct=vol_pct.astype(np.float32), fg_stats=fg,
                      t0=z0 - s0, t1=z1 - s0, z_offset=s0,
                      bbox=np.array([y0 - cy0, y1 - cy0, x0 - cx0, x1 - cx0]),
                      crop_origin=np.array([cy0, cx0]), phases_used=np.array(row["phases_used"]),
                      spacing=np.array([row["xy_spacing"], row["slice_thick"]], np.float32))
        if mask is not None:
            m, _ = crop_slices(mask[s0:s1, :, :, None], (y0 + y1) / 2, (x0 + x1) / 2, PLANE)
            arrays["mask"] = (m[..., 0] > 0).astype(np.uint8)
        np.savez_compressed(Path(out_dir) / f"{pid}.npz", **arrays)
        tum = stack[z0:z1, y0:y1, x0:x1]
        qc.update(n_slices=int(slab.shape[0]), t0=z0 - s0, t1=z1 - s0, z0=z0, z1=z1,
                  clip_lo=round(lo, 2), clip_hi=round(hi, 2), p999=round(p999, 2),
                  # how much of the tumour box v1 lost to clipping at the 99.5th percentile, per phase
                  sat_pre=round(float((tum[..., 0] > hi).mean()), 4),
                  sat_early=round(float((tum[..., 1] > hi).mean()), 4),
                  sat_late=round(float((tum[..., 2] > hi).mean()), 4),
                  fg_pre_mean=round(float(fg[0, 0]), 2), has_mask=mask is not None)
    except Exception as e:   # recorded, not fatal
        qc.update(status="error", note=qc["note"] + f"{type(e).__name__}: {e}")
    return qc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True, choices=list(COHORTS))
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--limit", type=int, default=0, help="first N patients only (debug)")
    ap.add_argument("--all-patients", action="store_true", help="also unlabeled patients (default: labeled)")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--max-buffer-gb", type=float, default=4.0)
    a = ap.parse_args()

    t_start = time.time()
    meta = load_metadata(a.metadata) if a.metadata else load_metadata()
    meta = meta[meta.dataset == COHORTS[a.cohort]]
    if not a.all_patients:
        meta = meta[meta.labeled]
    a.out.mkdir(parents=True, exist_ok=True)
    src = bc.Source(a.raw, COHORTS[a.cohort])
    print(f"[{a.cohort}] indexing ...", flush=True)
    phases, masks = src.index()
    pids = sorted(set(phases) & set(meta.index))
    qc_rows = [{"pid": p, "status": "missing_in_archive"} for p in sorted(set(meta.index) - set(phases))]
    if a.limit:
        pids = pids[:a.limit]
    print(f"[{a.cohort}] {len(phases)} patients in archive, {len(meta)} wanted, building {len(pids)}", flush=True)

    rows, needed = {}, {}
    for pid in pids:
        r = meta.loc[pid]
        used, note = bc.resolve_phases(r, phases[pid])
        if used is None:
            qc_rows.append({"pid": pid, "status": "error", "note": note})
            continue
        rows[pid] = {**{c: r[c] for c in ("n_z", "n_xy", "sraw", "eraw", "scol", "ecol", "mask_start",
                                          "mask_end", "xy_spacing", "slice_thick")},
                     "phases_used": used, "phase_note": note}
        needed[pid] = set(used)

    buffers, buf_bytes, spill = defaultdict(dict), 0, Path(tempfile.mkdtemp(prefix="mnj_spill_"))
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {}

        def drain(block):
            nonlocal done
            if not futs:
                return
            finished = wait(list(futs), return_when=FIRST_COMPLETED)[0] if block else [f for f in futs if f.done()]
            for f in finished:
                pid = futs.pop(f)
                qc = f.result()
                qc.update(phases_used=",".join(map(str, rows[pid]["phases_used"])),
                          phase_fallback=rows[pid]["phase_note"])
                qc_rows.append(qc)
                done += 1
                if done % 50 == 0 or done == len(rows):
                    print(f"[{a.cohort}] {done}/{len(rows)} done, {time.time() - t_start:.0f}s", flush=True)

        for pid, k, data in src.iter_phases(needed):
            if isinstance(data, bytes):
                if buf_bytes + len(data) > a.max_buffer_gb * 1e9:
                    p = spill / f"{pid}_{k}.nii.gz"
                    p.write_bytes(data)
                    data = p
                else:
                    buf_bytes += len(data)
            buffers[pid][k] = data
            if set(buffers[pid]) >= needed[pid]:
                got = buffers.pop(pid)
                buf_bytes -= sum(len(v) for v in got.values() if isinstance(v, bytes))
                while len(futs) >= 2 * a.workers:
                    drain(block=True)
                futs[ex.submit(process, pid, rows[pid], got, masks.get(pid), a.out)] = pid
            drain(block=False)
        while futs:
            drain(block=True)
    for pid in buffers:
        qc_rows.append({"pid": pid, "status": "error", "note": f"incomplete phases {sorted(buffers[pid])}"})

    index = pd.DataFrame(qc_rows)
    index.insert(1, "cohort", a.cohort)
    index.to_csv(a.out / "cache_index.csv", index=False)
    ok = index[index.status == "ok"]
    info = {"cohort": a.cohort, "format": "v2", "builder_version": BUILDER_VERSION, "git": bc.git_commit(),
            "raw": str(a.raw), "plane": PLANE, "margin": bc.MARGIN, "labeled_only": not a.all_patients,
            "status_counts": index.status.value_counts().to_dict(),
            "z_source": ok.get("z_source", pd.Series(dtype=str)).value_counts().to_dict(),
            "xy_source": ok.get("xy_source", pd.Series(dtype=str)).value_counts().to_dict(),
            "v1_saturation_tumor_box_mean": {c: round(float(ok[c].mean()), 4) for c in
                                             ("sat_pre", "sat_early", "sat_late") if c in ok},
            "size_gb": round(sum(f.stat().st_size for f in a.out.glob("*.npz")) / 1e9, 2),
            "seconds": round(time.time() - t_start)}
    (a.out / "cache_meta.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    errs = index[index.status == "error"]
    if len(errs):
        print(f"\n{len(errs)} ERRORS (first 10):\n" + errs[["pid", "note"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
