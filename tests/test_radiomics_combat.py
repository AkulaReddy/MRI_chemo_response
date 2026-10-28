"""Radiomics features, ComBat harmonisation and the radiomics baselines, on synthetic data."""
import numpy as np
import pandas as pd

from src.combat import ComBat, protocol_groups
from src.radiomics import FEATURES, patient_features


def _cache_file(with_mask=True, r_mm=8.0, pre=0.2, early=0.5, late=0.4):
    """A v2-like cache dict: a ball tumour (pre/early/late signal levels in clip-range units) in tissue."""
    s, n = 20, 240
    z, y, x = np.meshgrid(np.arange(s) * 2.0, np.arange(n) * 0.7, np.arange(n) * 0.7, indexing="ij")
    ball = (z - 20) ** 2 + (y - 84) ** 2 + (x - 84) ** 2 <= r_mm ** 2
    vals = []
    for v in (pre, early, late):
        img = np.full((s, n, n), 0.2)                # background tissue, not enhancing
        img[ball] = v
        vals.append(img)
    rgb = np.rint(np.stack(vals, -1) * 255).astype(np.uint8)
    d = {"rgb": rgb, "clip": np.array([0.0, 1000.0], np.float32), "t0": 5, "t1": 15,
         "bbox": np.array([108, 132, 108, 132]), "spacing": np.array([2.0, 0.7, 0.7], np.float32)}
    if with_mask:
        d["mask"] = ball.astype(np.uint8)

    class F(dict):
        files = property(lambda self: list(self))
    return F(d), ball


def test_patient_features_on_a_ball():
    d, ball = _cache_file()
    f = patient_features(d)
    assert f["mask_source"] == "mask" and f["n_voxels"] == ball.sum()
    vol = 4 / 3 * np.pi * 8.0 ** 3 / 1000
    assert abs(np.exp(f["log_volume_ml"]) - vol) / vol < 0.1
    assert f["sphericity"] > 0.8 and f["n_components"] == 1
    assert abs(f["re_early_mean"] - 1.5) < 0.05             # (0.5 - 0.2) / 0.2
    assert abs(f["re_late_mean"] - 1.0) < 0.05              # (0.4 - 0.2) / 0.2
    assert f["frac_washout"] > 0.99                         # late 20% below early
    assert abs(f["ring_re_early_mean"]) < 0.05              # surrounding tissue does not enhance
    assert set(FEATURES) <= set(f)


def test_box_tumour_region_without_mask():
    d, ball = _cache_file(with_mask=False)
    f = patient_features(d)
    assert f["mask_source"] == "box_pe70"
    assert f["n_voxels"] <= ball.sum() and abs(f["re_early_mean"] - 1.5) < 0.05   # only enhancing box voxels


def test_combat_removes_batch_effects_and_keeps_covariates():
    rng = np.random.default_rng(0)
    n = 600
    batch = rng.integers(0, 3, n)
    cov = rng.integers(0, 2, n).astype(float)[:, None]
    shift, scale = np.array([0.0, 3.0, -2.0]), np.array([1.0, 2.5, 0.5])
    # ComBat's model: batch shifts the mean and scales the noise; the covariate effect (2.0) is shared
    Y = rng.normal(0, 1, (n, 4)) * scale[batch][:, None] + shift[batch][:, None] + 2.0 * cov
    tr = np.arange(n) < 400
    cb = ComBat().fit(Y[tr], batch[tr], cov[tr])
    H = cb.transform(Y, batch, cov)
    slope = np.polyfit(cov[:, 0], H[:, 0], 1)[0]
    assert slope > 1.0                                            # covariate effect kept (pooled scale)
    R = H - slope * cov                                           # remove it, then compare the batches
    means = np.stack([R[batch == k].mean(0) for k in range(3)])
    sds = np.stack([R[batch == k].std(0) for k in range(3)])
    assert np.ptp(means, 0).max() < 0.3 * sds.mean()              # before: means differed by 5 (in SD ~1-2.5)
    assert (sds.max(0) / sds.min(0)).max() < 1.3                  # before: SD ratio 5
    unseen = cb.transform(Y[:5], np.full(5, 9), cov[:5])
    assert np.array_equal(unseen, Y[:5])                          # unseen batch left unadjusted


def test_protocol_groups_merge_small_groups():
    meta = pd.DataFrame({"cohort": ["a"] * 30 + ["b"] * 5, "n_xy": [256] * 25 + [512] * 5 + [256] * 5,
                         "slice_thick": [2.0] * 35, "xy_spacing": [0.7] * 35}, index=[f"p{i}" for i in range(35)])
    g = protocol_groups(meta, min_size=20)
    assert g.iloc[:30].nunique() == 1 and g.iloc[0] == "a|256|2mm|coarse"     # the 512 group was too small
    assert (g.iloc[30:] == "b|all").all()


def test_radiomics_baselines_end_to_end(tmp_path):
    from src.baselines.radiomics import main
    from src.data.metadata import load_split
    sp = load_split()
    pick = pd.concat([sp[(sp.split == s) & (sp.cohort == c)].head(n)
                      for s, n in (("train", 120), ("val", 40)) for c in ("ispy1", "ispy2", "duke")])
    rng = np.random.default_rng(0)
    f = pd.DataFrame(rng.normal(0, 1, (len(pick), len(FEATURES))), columns=FEATURES)
    y = pick.pCR.fillna(0).to_numpy()
    f["re_early_mean"] += 1.5 * y                                 # one informative feature
    f.insert(0, "pid", pick.pid.to_numpy())
    f.insert(1, "cohort", pick.cohort.to_numpy())
    f["mask_source"] = "mask"
    f.to_csv(tmp_path / "features.csv", index=False)
    res = main(["--features", str(tmp_path / "features.csv"), "--out", str(tmp_path / "out")])
    by = {r["config"]: r for r in res}
    assert by["radiomics_rad"]["val"]["auc"] > 0.7
    assert "batch_acc_after" in by["radiomics_rad"] and "batch_acc_after" not in by["radiomics_rad_nocombat"]
    assert len(list((tmp_path / "out").glob("r_*/val_preds.csv"))) == 5
