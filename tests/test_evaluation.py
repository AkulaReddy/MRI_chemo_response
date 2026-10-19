"""src/evaluation.py (DeLong, operating points, calibration, subgroups) and src/predict_run.py."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import evaluation as ev


def _scores(seed=0, n=300, sep=1.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    return y, y * sep + rng.normal(0, 1, n), rng


def test_delong_auc_matches_sklearn_with_ties():
    y, p, _ = _scores()
    p = np.round(p, 1)                                     # ties
    assert ev.delong_ci(y, p)["auc"] == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_delong_paired_test():
    y, p, rng = _scores()
    same = ev.delong(y, p, p + 1e-9 * rng.normal(size=len(p)))
    assert abs(same["diff"]) < 1e-6 and same["p"] > 0.5
    worse = ev.delong(y, rng.normal(size=len(y)), p)       # b (real) vs a (noise)
    assert worse["diff"] > 0.2 and worse["p"] < 1e-4 and worse["ci"][0] > 0
    assert worse["auc_b"] == pytest.approx(roc_auc_score(y, p))
    lo, hi = ev.delong_ci(y, p)["ci"]
    assert lo < roc_auc_score(y, p) < hi


def test_operating_point_reaches_specificity():
    y, p, _ = _scores()
    t = ev.threshold_at_spec(y, p, 0.90)
    op = ev.at_threshold(y, p, t)
    assert op["spec"] >= 0.90 and 0 < op["sens"] < 1 and 0 < op["ppv"] <= 1
    assert op["n_called_pcr"] == int((p > t).sum())


def test_reliability_and_summary():
    y, p, _ = _scores()
    prob = 1 / (1 + np.exp(-p))
    r = ev.reliability(y, prob)
    assert r.n.sum() == len(y) and r.observed.between(0, 1).all()
    s = ev.summary_row(y, prob)
    assert s["ci_lo"] < s["auc"] < s["ci_hi"] and s["spec@90spec"] >= 0.9


def test_subtypes_and_ensemble():
    from src.data.metadata import load_split
    pids = load_split().query("labeled and split == 'val'").pid.head(60).tolist()
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"pid": pids, "cohort": "x", "label": rng.integers(0, 2, 60), "prob": rng.random(60)})
    sub = ev.with_subtype(df)
    assert set(sub.subtype) <= {"HR+/HER2-", "HER2+", "TN", "unknown"} and len(set(sub.subtype)) >= 2
    tab = ev.subgroup_table(df)
    assert tab.n.sum() == 60
    e = ev.ensemble([df, df.assign(prob=1 - df.prob)])
    assert np.allclose(e.prob, 0.5)


def test_clinical_baseline_scores_any_patients():
    from src.data.metadata import patient_table
    tr = patient_table("train", ["ispy1", "ispy2"])
    sc = patient_table("val", ["duke"])
    out = ev.clinical_preds(tr, sc, "b1_lr")
    assert len(out) == len(sc) and out.prob.between(0, 1).all()


def test_predict_run_reproduces_val_and_guards_test(synthetic_cache, tmp_path):
    from src.predict_run import score_run
    from src.train import main
    caches = [f"{c}={d}" for c, d in synthetic_cache.items()]
    run = main(["--config", "b3_vit", "--cache", *caches, "--cache-root", str(tmp_path / "none"),
                "--out", str(tmp_path / "runs"), "--cpu", "--set", "model.backbone=tiny-random",
                "model.grad_checkpointing=false", "optim.max_epochs=1", "data.num_workers=0",
                "optim.batch_patients=4", "data.k_train=3", "data.k_eval=4", "export.after_train=false"])
    root = str(next(iter(synthetic_cache.values())).parent)
    df = score_run(run, ["ispy1", "duke"], ["val"], cache_roots=[root], dev=__import__("torch").device("cpu"))
    ref = pd.read_csv(run / "val_preds.csv").set_index("pid").prob
    assert len(df) == 16 and np.allclose(df.set_index("pid").prob.loc[ref.index], ref, atol=1e-5)
    with pytest.raises(PermissionError):
        score_run(run, ["ispy1"], ["test"], cache_roots=[root])
