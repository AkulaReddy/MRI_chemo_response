"""Training on builder-v2 caches: weak-slice filter, required cache version, cache info in metrics."""
import json

import numpy as np
import pytest

from src.data.clinical import ClinicalEncoder
from src.data.datasets import PatientBags
from src.data.metadata import patient_table
from src.train import main
from tests.test_train_smoke import COMMON


def _bags(tmp_path, area, mask=False):
    t = patient_table("train", ["ispy1"]).head(1)
    pid = t.pid.iloc[0]
    s = len(area) + 2
    rgb = np.zeros((s, 240, 240, 3), np.uint8)
    rgb[:, 0, 0, 0] = np.arange(s)                       # slice id in pixel (0, 0)
    arrays = dict(rgb=rgb, t0=1, t1=s - 1, bbox=np.array([100, 140, 100, 140]))
    if mask:
        m = np.zeros((s, 240, 240), np.uint8)
        for i, a in enumerate(area):
            m[1 + i].reshape(-1)[:a] = 1
        arrays["mask"] = m
    else:
        arrays["area"] = np.array([0, *area, 0], np.int32)
    np.savez(tmp_path / f"{pid}.npz", **arrays)
    return t, {"ispy1": tmp_path}


@pytest.mark.parametrize("mask", [False, True])
def test_min_area_frac_drops_weak_slices(tmp_path, mask):
    area = [0, 5, 100, 400, 1000, 300, 20, 0]               # tumour slices 1..8 of the cache
    t, dirs = _bags(tmp_path, area, mask)
    enc = ClinicalEncoder([]).fit(t)
    kept = PatientBags(t, dirs, enc, k=4, train=False, min_area_frac=0.1)._load(0)[:, 0, 0, 0]
    assert kept.tolist() == [3, 4, 5, 6]                    # area >= 100 (10% of 1000)
    off = PatientBags(t, dirs, enc, k=4, train=False)._load(0)[:, 0, 0, 0]
    assert off.tolist() == list(range(1, 9))                # default: every tumour slice


def test_min_area_frac_keeps_duke_and_never_empties(tmp_path):
    t, dirs = _bags(tmp_path, [0, 0, 0])
    enc = ClinicalEncoder([]).fit(t)
    assert len(PatientBags(t, dirs, enc, k=4, train=False, min_area_frac=0.5)._load(0)) == 1
    np.savez(next(tmp_path.glob("*.npz")), rgb=np.zeros((5, 240, 240, 3), np.uint8), t0=1, t1=4,
             bbox=np.array([0, 1, 0, 1]))                    # no area, no mask (Duke)
    assert len(PatientBags(t, dirs, enc, k=4, train=False, min_area_frac=0.5)._load(0)) == 3


def test_cache_version_is_enforced_and_recorded(synthetic_cache, tmp_path):
    caches = [f"{c}={d}" for c, d in synthetic_cache.items()]
    args = ["--config", "b3_vit", "--cache", *caches, "--cache-root", str(tmp_path / "none"),
            "--out", str(tmp_path / "runs"), *COMMON]
    with pytest.raises(SystemExit, match="cache-version 2 required"):
        main(args[:2] + ["--cache-version", "2"] + args[2:])   # synthetic caches are v1
    run = main(args[:2] + ["--cache-version", "1"] + args[2:])
    m = json.loads((run / "metrics.json").read_text())
    assert set(m["caches"]) == {"ispy1", "duke"} and m["caches"]["duke"]["builder_version"] == 1
