"""BreastDCEDL metadata: loading, md5 check, split codes, file naming.

The canonical metadata is the Zenodo `BreastDCEDL_metadata_min_crop.csv` (D-003).
Its `test` column encodes the split: 0 = train, 2 = val, 1 = test.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_CSV = REPO_ROOT / "data" / "metadata" / "BreastDCEDL_metadata_min_crop.csv"
METADATA_MD5 = "d58b84d0f343b4b8338cbb222e6dda08"
SPLIT_CSV = REPO_ROOT / "data" / "splits" / "split_v1.csv"

SPLIT_CODES = {0: "train", 2: "val", 1: "test"}
COHORTS = {"ispy1": "spy1", "ispy2": "spy2", "duke": "duke"}   # our name -> CSV `dataset`
COHORT_OF_DATASET = {v: k for k, v in COHORTS.items()}

# File names seen in the Zenodo demo archive (2026-10-08) and in the repo's data_utils.py:
#   ISPY1_1001_spy1_vis1_dce_aqc_0.nii.gz   ACRIN-6698-102212_spy2_vis1_dce_aqc_2.nii.gz
#   Breast_MRI_001_duke_dce_aqc_3.nii.gz     ISPY1_1001_spy1_vis1_mask.nii.gz
# and older spellings like {pid}_spy1_vis1_acq{k}.nii.gz. One regex accepts all of them.
PHASE_RE = re.compile(
    r"^(?P<pid>.+?)_(?P<ds>spy1|spy2|duke)(?:_vis\d)?(?:_dce)?_a(?:qc|cq)_?(?P<k>\d+)\.nii(?:\.gz)?$")
MASK_RE = re.compile(r"^(?P<pid>.+?)_(?P<ds>spy1|spy2|duke)(?:_vis\d)?_mask\.nii(?:\.gz)?$")


def md5_file(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_metadata(path: str | Path = METADATA_CSV, check_md5: bool = True) -> pd.DataFrame:
    """Read the metadata CSV, verify its md5, and add `cohort`, `split`, `labeled`, bbox columns."""
    path = Path(path)
    if check_md5:
        got = md5_file(path)
        if got != METADATA_MD5:
            raise ValueError(f"{path}: md5 {got} != canonical {METADATA_MD5} (wrong CSV? see D-003)")
    df = pd.read_csv(path)
    df["cohort"] = df["dataset"].map(COHORT_OF_DATASET)
    df["split"] = df["test"].astype(int).map(SPLIT_CODES)
    df["labeled"] = df["pCR"].notna()
    # Coordinates are [start, end) in voxels (checked against the I-SPY masks: eraw/ecol/mask_end
    # are exclusive). Spacing: xy_spacing in-plane, slice_thick between slices (mm).
    dz = (df["mask_end"] - df["mask_start"]).clip(lower=1)
    dy = (df["eraw"] - df["sraw"]).clip(lower=1)
    dx = (df["ecol"] - df["scol"]).clip(lower=1)
    df["bbox_vol_mm3"] = dz * df["slice_thick"] * dy * dx * df["xy_spacing"] ** 2
    df["log_bbox_vol"] = np.log(df["bbox_vol_mm3"])
    return df.set_index("pid", drop=False).rename_axis(None)


def load_split(path: str | Path = SPLIT_CSV) -> pd.DataFrame:
    """The frozen split (D-003). Every training/eval script reads the split from here only."""
    return pd.read_csv(path, dtype={"pid": str}).set_index("pid", drop=False).rename_axis(None)


def parse_name(fname: str):
    """('phase', pid, ds, k) or ('mask', pid, ds, None) or None for unrelated files."""
    base = Path(fname).name
    m = PHASE_RE.match(base)
    if m:
        return "phase", m["pid"], m["ds"], int(m["k"])
    m = MASK_RE.match(base)
    if m:
        return "mask", m["pid"], m["ds"], None
    return None
