"""Build the per-patient uint8 RGB slice cache from raw BreastDCEDL archives (D-009, D-022).

    python scripts/build_cache.py --cohort ispy1 --raw /kaggle/input/mnj-raw-ispy1 \
        --out /kaggle/working/cache_ispy1 [--limit 10] [--workers 4]

`--raw` may hold the original `.tar.gz` (streamed, never extracted to disk) or already-extracted
`.nii.gz` files (any depth). Output, per patient `<pid>.npz`:
    rgb    (S, 240, 240, 3) uint8   tumor slices +-2 margin, centred on the tumour box
    mask   (S, 240, 240) uint8      I-SPY only (Duke has bounding boxes only)
    area   (S,) int32               tumour-mask pixels per slice (I-SPY only)
    t0, t1                          tumor slices are rgb[t0:t1]
    bbox                            (y0, y1, x0, x1) of the tumor inside the crop
    spacing                         (z, y, x) mm of the cached voxels
plus `cache_index.csv` (one QC row per patient), `cache_meta.json` and `summary_grid.png`.

Builder v2 (default): every patient is resampled to `--z-mm` (2.0) x `--xy-mm` (0.7) mm voxels on a
240 x 240 window centred on the tumour box (zero padding outside the scan), so a 224 input covers
157 mm in every cohort. The clip ceiling is raised to the tumour's 99.9th percentile so the late
phase no longer saturates inside the tumour. `--xy-mm 0` reproduces v1 (native pixels, no resampling).
Builder v3 (`--z-mm 0`): as v2 in-plane (0.7 mm, tumour-centred, same clip), but the native slices are
kept (no resampling between slices), so the number of tumour slices differs between patients.

Coordinates: the CSV in-plane box (`sraw/eraw/scol/ecol`, [start, end)) matches the masks. The CSV
box is in the original `n_xy` image; larger originals (e.g. 512) were cropped to a 256 window centred
on the box, so the box is shifted by that window (see challenges.md) and, where a mask exists,
replaced by the mask's box. The CSV
z-range is used only if the volume depth equals `n_z`; otherwise (I-SPY volumes cropped in z, seen in
the demo archive) the mask's z-range is used. Missing CSV phase files fall back to the nearest
available phase. Every fallback is recorded in the QC table: check it before training.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.metadata import COHORTS, load_metadata, parse_name  # noqa: E402
from src.data.nifti_io import load_volume, mask_bbox  # noqa: E402
from src.data.resample import axis_grid, resample, tumour_z_grid  # noqa: E402
from src.data.rgb_slices import CACHE_SIZE, crop_slices, to_rgb_uint8, tumour_clip  # noqa: E402

MARGIN = 2
GRID = {"z_mm": 2.0, "xy_mm": 0.7}   # set from --z-mm / --xy-mm in main(); xy_mm 0 = v1, z_mm 0 = v3


def cache_version(grid) -> int:
    """1: native pixels and slices; 2: resampled in-plane and between slices; 3: in-plane only."""
    return 1 if not grid["xy_mm"] else (2 if grid["z_mm"] else 3)


# ----------------------------------------------------------------------------- sources
class Source:
    """Lists the phase/mask files of one cohort and yields their bytes, from tars or a folder."""

    def __init__(self, raw: Path, ds: str):
        self.ds = ds
        self.tars = sorted(raw.rglob("*.tar.gz")) + sorted(raw.rglob("*.tgz"))
        self.files = {} if self.tars else {
            p.name: p for p in raw.rglob("*.nii*") if (parse_name(p.name) or (0, 0, ""))[2] == ds}
        if not self.tars and not self.files:
            raise SystemExit(f"no .tar.gz or {ds} .nii.gz files under {raw}")

    def _members(self):
        for t in self.tars:
            with tarfile.open(t, "r|gz") as tf:
                for m in tf:
                    if m.isfile():
                        p = parse_name(m.name)
                        if p and p[2] == self.ds:
                            yield m, tf, p

    def index(self, want_masks=True):
        """(phases {pid: set(k)}, masks {pid: bytes or path})."""
        phases, masks = defaultdict(set), {}
        if self.tars:
            for m, tf, (kind, pid, _, k) in self._members():
                if kind == "phase":
                    phases[pid].add(k)
                elif want_masks:
                    masks[pid] = tf.extractfile(m).read()
        else:
            for name, path in self.files.items():
                kind, pid, _, k = parse_name(name)
                if kind == "phase":
                    phases[pid].add(k)
                else:
                    masks[pid] = path
        return phases, masks

    def iter_phases(self, needed):
        """Yields (pid, k, bytes_or_path) for the (pid, k) pairs in `needed`."""
        if self.tars:
            for m, tf, (kind, pid, _, k) in self._members():
                if kind == "phase" and k in needed.get(pid, ()):
                    yield pid, k, tf.extractfile(m).read()
        else:
            for name, path in self.files.items():
                kind, pid, _, k = parse_name(name)
                if kind == "phase" and k in needed.get(pid, ()):
                    yield pid, k, path


def resolve_phases(row, available):
    """CSV (pre, post_early, post_late) -> phase indices that exist, plus a fallback note."""
    want = [int(row.pre), int(row.post_early), int(row.post_late)]
    av = sorted(available)
    if all(k in av for k in want):
        return want, ""
    if len(av) < 3:
        return None, f"only phases {av}"
    pre = want[0] if want[0] in av else av[0]
    late = want[2] if want[2] in av else av[-1]
    mids = [k for k in av if pre < k < late]
    early = want[1] if want[1] in mids else (min(mids, key=lambda k: abs(k - want[1])) if mids else None)
    if early is None:
        return None, f"no early phase in {av}"
    return [pre, early, late], f"csv {want} -> used {[pre, early, late]} (available {av})"


# ----------------------------------------------------------------------------- per patient
def process(pid, row, phase_src, mask_src, out_dir, grid=None):
    """Returns a QC dict; writes <pid>.npz on success. Runs in a worker process.
    `grid` overrides GRID ({"z_mm", "xy_mm"}); passed explicitly so spawned workers see it."""
    qc = {"pid": pid, "status": "ok", "note": ""}
    try:
        vols = [load_volume(phase_src[k]) for k in row["phases_used"]]
        depth, h, w = vols[0].shape
        qc.update(depth=depth, n_z_csv=int(row["n_z"]), h=h, w=w)
        if (h, w) != (int(row["n_xy"]), int(row["n_xy"])):
            qc["note"] += f"in-plane {h}x{w} != n_xy {int(row['n_xy'])}; "
        mask = load_volume(mask_src, np.uint8) if mask_src is not None else None
        mb = mask_bbox(mask) if mask is not None else None
        if mask is not None and mask.shape != vols[0].shape:
            qc["note"] += f"mask shape {mask.shape} != volume; mask ignored; "
            mask, mb = None, None
        y0, y1, x0, x1 = (int(row[c]) for c in ("sraw", "eraw", "scol", "ecol"))
        n_xy, qc["xy_source"] = int(row["n_xy"]), "csv"
        if (h, w) != (n_xy, n_xy):   # CSV box is in the original n_xy image; the volume is a window of it
            if n_xy > h and n_xy > w:    # window centred on the box, clamped (fits 29/30 I-SPY1 offsets)
                oy, ox = (int(min(max((a + b) // 2 - 127, 0), n_xy - size))
                          for a, b, size in ((y0, y1, h), (x0, x1, w)))
                y0, y1, x0, x1 = max(0, y0 - oy), min(h, y1 - oy), max(0, x0 - ox), min(w, x1 - ox)
                qc["xy_source"] = "csv_shifted"
            else:                        # smaller original (I-SPY2 n_xy 128): rule unknown, rescale
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
        if mb is not None:   # agreement of the (scaled) CSV box with the mask, for QC
            (mz0, mz1), (my0, my1), (mx0, mx1) = mb
            inter = max(0, min(y1, my1) - max(y0, my0)) * max(0, min(x1, mx1) - max(x0, mx0))
            union = (y1 - y0) * (x1 - x0) + (my1 - my0) * (mx1 - mx0) - inter
            qc.update(mask_csv_iou_yx=round(inter / union, 3), mask_z=f"{mz0}:{mz1}",
                      mask_vox=int((mask > 0).sum()))
            if qc["xy_source"] != "csv":   # the mask is exact; the shifted/rescaled CSV box may not be
                y0, y1, x0, x1, qc["xy_source"] = my0, my1, mx0, mx1, "mask"
        if not (0 <= z0 < z1 <= depth and 0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
            raise ValueError(f"bbox z{z0}:{z1} y{y0}:{y1} x{x0}:{x1} outside volume {vols[0].shape}")
        grid = {**GRID, **(grid or {})}
        dz, dxy = float(row["slice_thick"]), float(row["xy_spacing"])
        if n_xy < h:   # smaller original stored at 256 (rescale rule above): pixels are smaller
            dxy *= n_xy / h
            qc["note"] += f"pixel {row['xy_spacing']:.3f} mm assumed upsampled to {dxy:.3f} mm; "
        qc.update(src_dz=dz, src_dxy=round(dxy, 4))
        if grid["xy_mm"]:   # v2/v3: fixed in-plane grid centred on the tumour
            lo, hi = tumour_clip(vols, (z0, z1, y0, y1, x0, x1))
            z_mm = grid["z_mm"] or dz   # v3: the source spacing, so z scale 1 = the native slices, unblurred
            zs, zo, n, t0, t1 = tumour_z_grid(z0, z1, depth, dz, z_mm, MARGIN)
            (ys, yo), (xs, xo) = (axis_grid((a + b) / 2, dxy, grid["xy_mm"], CACHE_SIZE)
                                  for a, b in ((y0, y1), (x0, x1)))
            shape = (n, CACHE_SIZE, CACHE_SIZE)
            res = [resample(v, (zs, ys, xs), (zo, yo, xo), shape) for v in vols]
            crop, _ = to_rgb_uint8(*res, clip=(lo, hi))

            def out_edge(e, s, o):   # source edge coordinate -> output edge coordinate
                return (e - 0.5 - o) / s + 0.5
            by0, by1, bx0, bx1 = (int(np.clip(f(out_edge(e, s, o)), 0, CACHE_SIZE)) for e, s, o, f in
                                  ((y0, ys, yo, np.floor), (y1, ys, yo, np.ceil),
                                   (x0, xs, xo, np.floor), (x1, xs, xo, np.ceil)))
            arrays = dict(rgb=crop, t0=t0, t1=t1, z_offset=zo, bbox=np.array([by0, by1, bx0, bx1]),
                          crop_origin=np.array([yo, xo]), clip=np.array([lo, hi], np.float32),
                          phases=np.array(row["phases_used"]),
                          spacing=np.array([z_mm, grid["xy_mm"], grid["xy_mm"]], np.float32))
            if mask is not None:
                m = resample(mask.astype(np.float32), (zs, ys, xs), (zo, yo, xo), shape, antialias=False)
                arrays["mask"] = (m >= 0.5).astype(np.uint8)
                arrays["area"] = arrays["mask"].sum((1, 2)).astype(np.int32)
            box = crop[t0:t1, by0:by1, bx0:bx1]
            qc.update(sat_frac=round(float((box == 255).any(-1).mean()), 4) if box.size else None,
                      fov224_mm=round(224 * grid["xy_mm"], 1))
            cy0, cx0 = round(yo, 2), round(xo, 2)
        else:               # v1: native pixels, window shifted inward, joint volume percentiles
            rgb, (lo, hi) = to_rgb_uint8(*vols)
            s0, s1 = max(0, z0 - MARGIN), min(depth, z1 + MARGIN)
            crop, (cy0, cx0) = crop_slices(rgb[s0:s1], (y0 + y1) / 2, (x0 + x1) / 2, CACHE_SIZE)
            t0, t1 = z0 - s0, z1 - s0
            arrays = dict(rgb=crop, t0=t0, t1=t1, z_offset=s0,
                          bbox=np.array([y0 - cy0, y1 - cy0, x0 - cx0, x1 - cx0]),
                          crop_origin=np.array([cy0, cx0]), clip=np.array([lo, hi], np.float32),
                          phases=np.array(row["phases_used"]),
                          spacing=np.array([dz, dxy, dxy], np.float32))
            if mask is not None:
                m, _ = crop_slices(mask[s0:s1, :, :, None], (y0 + y1) / 2, (x0 + x1) / 2, CACHE_SIZE)
                arrays["mask"] = (m[..., 0] > 0).astype(np.uint8)
        np.savez_compressed(Path(out_dir) / f"{pid}.npz", **arrays)
        qc.update(n_slices=int(crop.shape[0]), t0=t0, t1=t1, z0=z0, z1=z1,
                  clip_lo=round(lo, 2), clip_hi=round(hi, 2), crop_y0=cy0, crop_x0=cx0,
                  has_mask=mask is not None, tumor_mean=float(crop[t0:t1].mean()))
    except Exception as e:   # recorded, not fatal: one bad patient must not stop the build
        qc.update(status="error", note=qc["note"] + f"{type(e).__name__}: {e}")
    return qc


# ----------------------------------------------------------------------------- driver
def summary_grid(out_dir: Path, index: pd.DataFrame, n=20, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    ok = index[index.status == "ok"]
    pick = ok.sample(min(n, len(ok)), random_state=seed)
    cols = 5
    rows = max(1, int(np.ceil(len(pick) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, pid in zip(np.atleast_1d(axes).ravel(), pick.pid):
        d = np.load(out_dir / f"{pid}.npz")
        k = (int(d["t0"]) + int(d["t1"])) // 2
        ax.imshow(d["rgb"][k])
        y0, y1, x0, x1 = d["bbox"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=1))
        if "mask" in d and d["mask"][k].any():
            ax.contour(d["mask"][k], levels=[0.5], colors="cyan", linewidths=0.6)
        ax.set_title(f"{pid}\nslice {k}/{d['rgb'].shape[0]}", fontsize=7)
    fig.suptitle("RGB = pre / early / late. Yellow: CSV box. Cyan: mask. Tumor should be bright-ish and inside.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_grid.png", dpi=90)
    plt.close(fig)


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True, choices=list(COHORTS))
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--limit", type=int, default=0, help="first N patients only (debug)")
    ap.add_argument("--pids", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--z-mm", type=float, default=GRID["z_mm"],
                    help="slice spacing of the cache (mm); 0 = native slices (builder v3)")
    ap.add_argument("--xy-mm", type=float, default=GRID["xy_mm"],
                    help="in-plane pixel size of the cache (mm); 0 = native pixels (builder v1)")
    ap.add_argument("--max-buffer-gb", type=float, default=4.0,
                    help="phase bytes held in RAM while waiting for a patient's other phases")
    a = ap.parse_args()

    t_start = time.time()
    grid = {"z_mm": a.z_mm, "xy_mm": a.xy_mm}
    ds = COHORTS[a.cohort]
    meta = load_metadata(a.metadata) if a.metadata else load_metadata()
    meta = meta[meta.dataset == ds]
    a.out.mkdir(parents=True, exist_ok=True)

    src = Source(a.raw, ds)
    print(f"[{a.cohort}] indexing {'tar ' + ', '.join(t.name for t in src.tars) if src.tars else 'folder'} ...",
          flush=True)
    phases, masks = src.index()
    pids = sorted(set(phases) & set(meta.index))
    qc_rows = [{"pid": p, "status": "missing_in_archive"} for p in sorted(set(meta.index) - set(phases))]
    qc_rows += [{"pid": p, "status": "not_in_metadata"} for p in sorted(set(phases) - set(meta.index))]
    if a.pids:
        pids = [p for p in pids if p in set(a.pids)]
    if a.limit:
        pids = pids[:a.limit]
    print(f"[{a.cohort}] {len(phases)} patients in archive, {len(masks)} masks, "
          f"{len(meta)} in metadata, building {len(pids)}", flush=True)

    rows, needed = {}, {}
    for pid in pids:
        r = meta.loc[pid]
        used, note = resolve_phases(r, phases[pid])
        if used is None:
            qc_rows.append({"pid": pid, "status": "error", "note": note})
            continue
        rows[pid] = {**{c: r[c] for c in ("n_z", "n_xy", "sraw", "eraw", "scol", "ecol",
                                          "mask_start", "mask_end", "pre", "post_early", "post_late",
                                          "slice_thick", "xy_spacing")},
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
            finished, _ = wait(list(futs), return_when=FIRST_COMPLETED) if block else (
                [f for f in futs if f.done()], None)
            for f in finished:
                pid = futs.pop(f)
                qc = f.result()
                qc.update(phases_csv=f"{int(rows[pid]['pre'])},{int(rows[pid]['post_early'])},"
                                     f"{int(rows[pid]['post_late'])}",
                          phases_used=",".join(map(str, rows[pid]["phases_used"])),
                          phase_fallback=rows[pid]["phase_note"])
                qc_rows.append(qc)
                done += 1
                if done % 50 == 0 or done == len(rows):
                    print(f"[{a.cohort}] {done}/{len(rows)} done, {time.time() - t_start:.0f}s", flush=True)

        for pid, k, data in src.iter_phases(needed):
            if isinstance(data, bytes):
                if buf_bytes + len(data) > a.max_buffer_gb * 1e9:     # spill rather than run out of RAM
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
                futs[ex.submit(process, pid, rows[pid], got, masks.get(pid), a.out, grid)] = pid
            drain(block=False)
        while futs:
            drain(block=True)
    for pid in buffers:   # phases listed in the index but never streamed (should not happen)
        qc_rows.append({"pid": pid, "status": "error", "note": f"incomplete phases {sorted(buffers[pid])}"})

    index = pd.DataFrame(qc_rows)
    index.insert(1, "cohort", a.cohort)
    index.to_csv(a.out / "cache_index.csv", index=False)
    counts = index.status.value_counts().to_dict()
    info = {"cohort": a.cohort, "builder_version": cache_version(grid), "git": git_commit(),
            "grid_mm": {"z_mm": grid["z_mm"] or None, "xy_mm": grid["xy_mm"]} if a.xy_mm else None,
            "raw": str(a.raw), "cache_size": CACHE_SIZE, "margin": MARGIN, "status_counts": counts,
            "z_source": index.get("z_source", pd.Series(dtype=str)).value_counts().to_dict(),
            "xy_source": index.get("xy_source", pd.Series(dtype=str)).value_counts().to_dict(),
            "phase_fallbacks": int((index.get("phase_fallback", pd.Series(dtype=str)).fillna("") != "").sum()),
            "seconds": round(time.time() - t_start)}
    (a.out / "cache_meta.json").write_text(json.dumps(info, indent=2))
    if counts.get("ok"):
        summary_grid(a.out, index)
    print(json.dumps(info, indent=2))
    errs = index[index.status == "error"]
    if len(errs):
        print(f"\n{len(errs)} ERRORS (first 10):\n" + errs[["pid", "note"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
