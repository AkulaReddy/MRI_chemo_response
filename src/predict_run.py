"""Score a trained run (its best.pt) on chosen cohorts and splits: leave-one-cohort-out and the final test.

    python -m src.predict_run --run-dir runs/<run_id> --cohorts duke --splits train val --out loco_duke.csv
    python -m src.predict_run --run-dir runs/<run_id> --splits test --final --out test_preds.csv

Works for src/train.py runs (B3, F1, F2: config_resolved.yaml + encoder.json) and src/vitonly runs (vo_*).
The caches must be the same builder version the run was trained on (read from its metrics.json).
The test split is refused unless --final (D-006); every --final call is appended to final_eval_log.csv
in --log-dir, with time, run, git commit and patient count, so the one-time test use stays visible.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

import src

src.hide_broken_torchaudio()
from src.data.metadata import patient_table  # noqa: E402
from src.train import find_caches, git_sha, make_loader, predict, sigmoid  # noqa: E402


def load_run(run_dir: Path, dev):
    cfg = yaml.safe_load((run_dir / "config_resolved.yaml").read_text())
    vitonly = "pool" in cfg["model"] or cfg.get("name", "").startswith("vo_")
    if vitonly:
        from src.vitonly.model import VitOnly
        model = VitOnly(cfg["model"])
        enc = None
    else:
        from src.data.clinical import ClinicalEncoder
        from src.models.fusion import build_model
        enc = ClinicalEncoder.from_dict(json.loads((run_dir / "encoder.json").read_text()))
        model = build_model(cfg, enc)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)["model"])
    return cfg, enc, model.to(dev).eval(), vitonly


def dataset(cfg, enc, vitonly, table, caches):
    d = cfg["data"]
    if vitonly:
        from src.vitonly.data import SlabBags
        return SlabBags(table, caches, norm=d["norm"], crop=d["crop"], k=d.get("k_eval", 12), train=False,
                        slice_mode=d.get("slice_mode", "mid"), mid_frac=d.get("mid_frac", 0.6),
                        zoom=d.get("zoom", 1.6), fov_mm=d.get("fov_mm", 100.0))
    from src.data.datasets import PatientBags
    return PatientBags(table, caches, enc, k=d.get("k_eval", 16), train=False, margin=d.get("margin", 0),
                       min_area_frac=float(d.get("min_area_frac", 0.0)))


def score_run(run_dir, cohorts=("ispy1", "ispy2", "duke"), splits=("val",), final=False, cache_roots=None,
              dev=None):
    """DataFrame pid, cohort, split, label, prob for the run's checkpoint on the requested patients."""
    run_dir = Path(run_dir)
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "test" in splits and not final:
        raise PermissionError("the test split needs --final (D-006: the test set is used once)")
    cfg, enc, model, vitonly = load_run(run_dir, dev)
    roots = cache_roots or (["/kaggle/input", "/kaggle/working"] if Path("/kaggle/input").is_dir() else ["data/cache"])
    if vitonly:
        from src.vitonly.slabs import find_caches as vo_find
        caches = vo_find(roots, cfg["data"].get("prefer_cache", "v2"))
    else:
        caches = find_caches(roots)
    trained_on = json.loads((run_dir / "metrics.json").read_text()).get("caches", {}) \
        if (run_dir / "metrics.json").exists() else {}
    for c in cohorts:   # same cache builder as in training, else the inputs differ from what the model learned
        want = trained_on.get(c, next(iter(trained_on.values()), {})).get("builder_version") if trained_on else None
        got = json.loads((Path(caches[c]) / "cache_meta.json").read_text()).get("builder_version", 1) \
            if c in caches else None
        if c not in caches:
            raise SystemExit(f"no cache for {c} under {roots}")
        if want is not None and got != want:
            raise SystemExit(f"{c}: run trained on cache builder v{want}, attached cache is v{got}")
    frames = []
    for sp in splits:
        t = patient_table([sp], list(cohorts), cache_dirs=caches, allow_test=final)
        ds = dataset(cfg, enc, vitonly, t, caches)
        loader = make_loader(ds, {**cfg, "data": {**cfg["data"], "sampler": "random"},
                                  "optim": {**cfg["optim"], "batch_patients": 4}}, False)
        amp = dev.type == "cuda" and cfg["optim"].get("amp", True)
        pr = predict(model, loader, dev, amp)
        tt = t.reset_index(drop=True).iloc[pr["index"]]
        frames.append(pd.DataFrame({"pid": tt.pid.values, "cohort": tt.cohort.values, "split": sp,
                                    "label": tt.pCR.values, "prob": sigmoid(pr["logit"])}))
    return pd.concat(frames, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cohorts", nargs="+", default=["ispy1", "ispy2", "duke"])
    ap.add_argument("--splits", nargs="+", default=["val"], choices=["train", "val", "test"])
    ap.add_argument("--final", action="store_true", help="allow the test split (logged)")
    ap.add_argument("--cache-root", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-dir", default=".")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    df = score_run(a.run_dir, a.cohorts, a.splits, a.final, a.cache_root, dev)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    if a.final:
        log = Path(a.log_dir) / "final_eval_log.csv"
        row = pd.DataFrame([{"time": time.strftime("%Y-%m-%d %H:%M:%S"), "run": Path(a.run_dir).name,
                             "git": git_sha(), "splits": "+".join(a.splits), "cohorts": "+".join(a.cohorts),
                             "n": len(df), "out": a.out}])
        row.to_csv(log, mode="a", header=not log.exists(), index=False)
        print(f"FINAL evaluation logged in {log}")
    print(f"scored {len(df)} patients -> {a.out}")
    return df


if __name__ == "__main__":
    main()
