import json

import numpy as np
import pandas as pd

from src.baselines.tabular import main
from src.data.metadata import load_split


def test_baselines_val_only(tmp_path):
    res = main(["--models", "b0_lr", "b1_rf", "--seeds", "0", "--out", str(tmp_path)])
    assert [m["config"] for m in res] == ["b0_lr", "b1_rf"]
    split = load_split()
    test_pids = set(split.pid[split.split == "test"])
    for m in res:
        assert m["n_train"] == 1099 and m["n_val"] == 177
        d = tmp_path / m["run_id"]
        preds = pd.read_csv(d / "val_preds.csv", dtype={"pid": str})
        assert len(preds) == 177 and preds.prob.between(0, 1).all()
        assert not test_pids & set(preds.pid)
        assert set(json.loads((d / "model.json").read_text())["weights"])
    assert ("log_bbox_vol" in res[1]["features"]) and ("log_bbox_vol" not in res[0]["features"])


def test_paired_bootstrap_auc_diff():
    from src import stats
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good = y + rng.normal(0, 0.8, 300)
    same = good + rng.normal(0, 1e-6, 300)
    r = stats.paired_bootstrap_auc_diff(y, rng.normal(size=300), good, n=500)
    assert r["diff"] > 0.2 and r["ci"][0] > 0 and r["p"] < 0.01      # clearly better
    r = stats.paired_bootstrap_auc_diff(y, good, same, n=500)
    assert abs(r["diff"]) < 0.01 and r["ci"][0] <= 0 <= r["ci"][1]      # no difference
