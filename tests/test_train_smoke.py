"""End-to-end: synthetic cache -> train 1 epoch -> metrics/preds -> export. CPU, tiny random ViT."""
import json

import pandas as pd
import pytest

from src.train import main

COMMON = ["--cpu", "--set", "model.backbone=tiny-random", "model.grad_checkpointing=false",
          "optim.max_epochs=1", "data.num_workers=0", "optim.batch_patients=4",
          "data.k_train=3", "data.k_eval=4", "export.k=4"]


@pytest.mark.parametrize("cfg", ["b3_vit", "f2_attn"])
def test_train_and_export(cfg, synthetic_cache, tmp_path):
    caches = [f"{c}={d}" for c, d in synthetic_cache.items()]
    no_root = ["--cache-root", str(tmp_path / "none")]
    run = main(["--config", cfg, "--cache", *caches, *no_root, "--out", str(tmp_path / "runs"),
                "--results-dir", str(tmp_path / "results"), *COMMON])
    m = json.loads((run / "metrics.json").read_text())
    assert m["n_train"] == 24 and m["n_val"] == 16
    preds = pd.read_csv(run / "val_preds.csv")
    assert len(preds) == 16 and preds.prob.between(0, 1).all()
    assert (tmp_path / "results" / run.name / "metrics.json").exists()
    if cfg == "b3_vit":                                       # B3 exports right after training
        assert len(pd.read_csv(run / "export_preds.csv")) == 40
        assert (run / "export_slices.csv").exists()
    main(["--config", cfg, "--mode", "export", "--checkpoint", str(run / "best.pt"),
          "--cache", *caches, *no_root, *COMMON])
    assert (run / "export_emb.npz").exists()
