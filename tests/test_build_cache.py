"""Builds a cache from a synthetic tar.gz laid out like the Zenodo demo archive."""
import gzip
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from src.data.metadata import load_metadata

ROOT = Path(__file__).resolve().parents[1]


def _add(tf, name, a):
    raw = gzip.compress(nib.Nifti1Image(a.astype(np.float64), np.eye(4)).to_bytes())
    info = tarfile.TarInfo(name)
    info.size = len(raw)
    tf.addfile(info, io.BytesIO(raw))


def test_build_from_tar(tmp_path):
    m = load_metadata()
    spy1 = m[m.cohort == "ispy1"]
    spy1 = pd.concat([spy1[spy1.n_xy == 256].head(2), spy1[spy1.n_xy == 512].head(1)])
    r_duke = next(m[m.cohort == "duke"].head(1).itertuples())
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    with tarfile.open(raw / "BreastDCEDL_test_min_crop.tar.gz", "w:gz") as tf:
        # I-SPY1 patient 1: depth == n_z (CSV z-range); patient 2: z-cropped (mask z-range);
        # patient 3: n_xy 512 resampled to 256 (CSV box in 512-pixel units, rescaled)
        for i, r in enumerate(spy1.itertuples()):
            depth = int(r.n_z) if i == 0 else 20
            z0, z1 = (int(r.mask_start), int(r.mask_end)) if i == 0 else (2, 12)
            f = 256 / r.n_xy
            mask = np.zeros((depth, 256, 256))
            mask[z0:z1, int(np.floor(r.sraw * f)):int(np.ceil(r.eraw * f)),
                 int(np.floor(r.scol * f)):int(np.ceil(r.ecol * f))] = 1
            for k in (int(r.pre), int(r.post_early), int(r.post_late)):
                _add(tf, f"x/dce/{r.pid}_spy1_vis1_dce_aqc_{k}.nii.gz",
                     rng.normal(100 + 20 * k, 10, (depth, 256, 256)) + 300 * mask)
            _add(tf, f"x/mask/{r.pid}_spy1_vis1_mask.nii.gz", mask)
        # Duke: the CSV late phase is missing -> falls back to the last available phase
        for k in (0, 1, int(r_duke.post_late) - 1):
            _add(tf, f"x/dce/{r_duke.pid}_duke_dce_aqc_{k}.nii.gz",
                 rng.normal(100, 10, (int(r_duke.n_z), 256, 256)))
        _add(tf, "x/mask/desktop.ini.nii.gz", np.zeros((2, 2, 2)))   # unrelated file: ignored
    for cohort, n in (("ispy1", 3), ("duke", 1)):
        out = tmp_path / f"cache_{cohort}"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_cache.py"), "--cohort", cohort,
                        "--raw", str(raw), "--out", str(out), "--workers", "1"], check=True, cwd=ROOT)
        idx = pd.read_csv(out / "cache_index.csv")
        ok = idx[idx.status == "ok"]
        assert len(ok) == n, idx[idx.status != "missing_in_archive"].to_string()
        assert json.loads((out / "cache_meta.json").read_text())["status_counts"]["ok"] == n
        assert (out / "summary_grid.png").exists()
        for pid in ok.pid:
            d = np.load(out / f"{pid}.npz")
            assert d["rgb"].dtype == np.uint8 and d["rgb"].shape[1:] == (240, 240, 3)
            assert 0 <= d["t0"] < d["t1"] <= d["rgb"].shape[0]
        if cohort == "ispy1":
            assert sorted(ok.z_source) == ["csv", "mask", "mask"]
            assert sorted(ok.xy_source) == ["csv", "csv", "mask"]
            assert (ok.mask_csv_iou_yx == 1.0).all()
            for pid in ok.pid:   # the tumor (bright inside the mask) sits inside the cached box
                d = np.load(out / f"{pid}.npz")
                y0, y1, x0, x1 = d["bbox"]
                t = d["rgb"][d["t0"]:d["t1"]].astype(float)
                assert t[:, y0:y1, x0:x1].mean() > t.mean() + 20
                assert d["mask"][d["t0"]:d["t1"], y0:y1, x0:x1].all()
        else:
            assert ok.phase_fallback.notna().all()
