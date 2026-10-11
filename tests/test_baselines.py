import json

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
