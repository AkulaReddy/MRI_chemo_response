"""Step 9/10 additions: per-slice normalisations, ComBat, 2D/3D/temporal features, the feature-track CLI."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.metadata import load_split
from src.vitonly import radiomics
from src.vitonly.combat import ComBat
from src.vitonly.norms import SLICE_NORMS, normalise
from src.vitonly.slabs import load_slab, raw
from test_vitonly import _fake_v2


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    """Fake v2 caches with enough patients for 5-fold CV inside each cohort; enhancement depends on pCR."""
    split = load_split()
    root = tmp_path_factory.mktemp("cache_feat")
    rng = np.random.default_rng(1)
    dirs = {}
    for cohort in ("ispy1", "duke"):
        d = root / f"cache2_{cohort}"
        d.mkdir()
        rows = []
        for sp, n in (("train", 30), ("val", 10)):
            for pid in split[(split.cohort == cohort) & (split.split == sp) & split.labeled].pid.head(n):
                _fake_v2(d / f"{pid}.npz", rng, 250 if split.loc[pid, "pCR"] == 1 else 80, s=6, with_mask=False)
                rows.append(f"{pid},{cohort},ok")
        (d / "cache_index.csv").write_text("pid,cohort,status\n" + "\n".join(rows) + "\n")
        (d / "cache_meta.json").write_text(json.dumps({"cohort": cohort, "format": "v2"}))
        dirs[cohort] = d
    return dirs


def _slab(cache):
    return load_slab(next(Path(cache["ispy1"]).glob("*.npz")))


@pytest.mark.parametrize("norm", ["slice_joint", "slice_pre", "slice_zscore"])
def test_slice_norms_keep_the_enhancement(cache, norm):
    s = _slab(cache)
    out = normalise(raw(s, np.arange(1, 5)), s["stats"], norm)
    tum, bg = out[:, 100:140, 110:150].mean((0, 1, 2)), out[:, 10:50, 10:50].mean((0, 1, 2))
    assert out.min() >= 0 and out.max() <= 1
    assert tum[1] - tum[0] > 0.05 and abs(bg[1] - bg[0]) < tum[1] - tum[0]   # tumour brightens, background less


def test_slice_norms_use_only_their_own_slice(cache):
    s = _slab(cache)
    x = raw(s, np.arange(1, 4))
    for norm in SLICE_NORMS:
        a = normalise(x, s["stats"], norm)
        b = normalise(np.concatenate([x[:1], 5 * x[1:]]), s["stats"], norm)   # rescale slices 2-3 only
        assert np.allclose(a[0], b[0], atol=1e-5), norm                       # slice 1 is unaffected
        if norm != "slice_zscore":                                            # scale-free per slice
            assert np.allclose(a[1:], b[1:], atol=2e-2), norm


def test_combat_removes_batch_effects_and_keeps_covariates():
    rng = np.random.default_rng(0)
    n, p = 600, 20
    batch = rng.choice(["a", "b", "c"], n)
    cov = rng.integers(0, 2, (n, 1)).astype(float)
    shift = {"a": 0.0, "b": 3.0, "c": -2.0}
    scale = {"a": 1.0, "b": 2.0, "c": 0.5}
    X = np.array([x * scale[b] + shift[b] for x, b in zip(rng.normal(0, 1, (n, p)), batch)]) + 1.5 * cov
    tr = np.arange(n) < 400
    cb = ComBat().fit(X[tr], batch[tr], cov[tr])
    H = cb.transform(X[~tr], batch[~tr], cov[~tr])
    means = [H[batch[~tr] == b].mean() for b in "abc"]
    sds = [H[(batch[~tr] == b) & (cov[~tr, 0] == 0)].std() for b in "abc"]
    assert np.ptp(means) < 0.4 and np.ptp(sds) < 0.3                         # raw: 5.0 and 1.5
    eff = H[cov[~tr, 0] == 1].mean() - H[cov[~tr, 0] == 0].mean()
    assert 0.8 < eff < 2.5                                                    # the covariate effect survives
    with pytest.raises(ValueError):
        cb.transform(X[:3], np.array(["zz"] * 3), cov[:3])


def test_radiomics_features(cache):
    s = _slab(cache)
    f_vol = radiomics.extract(s, 0.7, 2.0, "volume")
    f_sl = radiomics.extract(s, 0.7, 2.0, "slice")
    names = [n for fam in radiomics.FAMILIES for n in radiomics.feature_names(fam)]
    assert set(names) == set(f_vol) == set(f_sl)
    assert np.mean([np.isfinite(v) for v in f_vol.values()]) > 0.95
    assert f_vol["3d_fo_pe_mean"] > 0.7 and f_vol["2d_fo_pe_mean"] > 0.7      # the ROI enhances
    for k in names:                                                           # ratios and shape: ref-independent
        if any(t in k for t in ("_pe_", "_ser_", "shape_", "frac_", "area")):
            assert np.isclose(f_vol[k], f_sl[k], equal_nan=True, rtol=1e-4), k


def test_feature_track_end_to_end(cache, tmp_path):
    from src.vitonly import features
    out = tmp_path / "feat"
    res = features.main(["--cache-root", str(Path(cache["ispy1"]).parent), "--cohorts", "ispy1", "duke",
                         "--out", str(out), "--jobs", "1", "--sets", "kinetic16", "3d", "temporal"])
    assert {"results.csv", "val_preds.csv", "RESULTS.md", "features_volume.csv", "features_slice.csv",
            "harmonisation_check.csv", "extraction_errors.csv"} <= {f.name for f in out.iterdir()}
    assert len(res[res.set == "3d"]) == 2 * 2 * 2                            # refs x ComBat x subtype
    assert res.val_auc.notna().all()
    assert "Does ComBat help" in (out / "RESULTS.md").read_text()
    # embeddings only (step 10): a fake probe output for the same patients, reusing the extracted features
    f = pd.read_csv(out / "features_volume.csv", dtype={"pid": str})
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "features_tiny.npz", pid=f.pid.values, combos=np.array(["slice_pre|tumor"]),
             **{"0_gap": rng.normal(size=(len(f), 16)).astype(np.float16)})
    res2 = features.main(["--cache-root", str(Path(cache["ispy1"]).parent), "--cohorts", "ispy1", "duke",
                          "--out", str(tmp_path / "deep"), "--jobs", "1", "--only-embeddings",
                          "--embeddings", str(tmp_path / "features_tiny.npz")])
    assert (res2.set == "deep:tiny|slice_pre|tumor|gap").sum() == 4
    # reuse the step-9 features (protocol batch / step 10): same columns, no duplicated metadata
    res3 = features.main(["--cache-root", str(Path(cache["ispy1"]).parent), "--cohorts", "ispy1", "duke",
                          "--out", str(tmp_path / "reuse"), "--jobs", "1", "--features-from", str(out),
                          "--refs", "volume", "--sets", "3d+temporal", "--batch", "protocol",
                          "--embeddings", str(tmp_path / "features_tiny.npz")])
    assert "deep:tiny|slice_pre|tumor|gap + 3d+temporal" in set(res3.set)
    again = pd.read_csv(tmp_path / "reuse" / "features_volume.csv")
    assert list(again.columns) == list(f.columns)
