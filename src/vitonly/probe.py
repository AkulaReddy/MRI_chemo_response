"""Frozen-backbone feature probe: which input (normalisation x crop) and which backbone carry pCR signal?

    python -m src.vitonly.probe --backbones mae dinov2_b --norms joint_pct kinetic sub \
        --crops fixed tumor --cache-root /kaggle/input --out /kaggle/working/probe

Each patient's K evaluation slices go through a frozen backbone; slice features ([CLS] and patch-token mean)
are averaged into one patient vector and scored with an L2 logistic regression:
  cv_auc          5-fold stratified CV on the train split (~1,100 patients): low variance, use it to RANK
  val_auc         fit on all of train, scored on val (177 patients, 95% CI about +-0.08): a confirmation
  val_auc_within  mean of the per-cohort val AUCs. Cohorts differ in pCR rate (Duke 21%, I-SPY2 32%), so a
                  feature that only recognises the scanner already scores above 0.5 when pooled.
Reference rows: `cohort_only` (the cohort one-hot alone: the free AUC) and `kinetic_handcrafted` (tumour
enhancement statistics, no deep learning). Test patients are never loaded (patient_table refuses them).
Outputs: probe_results.csv (one row per backbone/norm/crop/pool), probe_val_preds.csv, features_<bb>.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import src

src.hide_broken_torchaudio()
from src import stats  # noqa: E402
from src.data.metadata import patient_table  # noqa: E402
from src.data.transforms import eval_transform  # noqa: E402

from .backbones import Backbone  # noqa: E402
from .data import MultiViews  # noqa: E402
from .kinetics import FEATURES, kinetic_features  # noqa: E402
from .norms import CROPS, NORMS  # noqa: E402
from .slabs import find_caches, load_slab  # noqa: E402

CS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)


def score(X, y, cohort, is_train, seed=0):
    """CV AUC on train (C chosen on the same folds), then refit on all train -> val predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr = X[is_train], y[is_train]
    strata = np.char.add(cohort[is_train].astype(str), ytr.astype(int).astype(str))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, strata))
    cv = {}
    for C in CS:
        aucs = []
        for tr, te in folds:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Xtr[tr], ytr[tr])
            aucs.append(stats.auc(ytr[te], m.decision_function(Xtr[te])))
        cv[C] = (float(np.nanmean(aucs)), float(np.nanstd(aucs)))    # nan: a fold with one class (tiny data)
    best = max(CS, key=lambda c: -np.inf if np.isnan(cv[c][0]) else cv[c][0])
    m = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=3000)).fit(Xtr, ytr)
    p = m.predict_proba(X[~is_train])[:, 1]
    yv, cv_ = y[~is_train], cohort[~is_train]
    by = {c: stats.auc(yv[cv_ == c], p[cv_ == c]) for c in sorted(set(cv_))}
    lo, hi = stats.bootstrap_auc_ci(yv, p, n=1000)
    row = {"cv_auc": cv[best][0], "cv_auc_sd": cv[best][1], "C": best, "val_auc": stats.auc(yv, p),
           "val_ci_lo": lo, "val_ci_hi": hi, **{f"val_auc_{c}": v for c, v in by.items()},
           "val_auc_within": float(np.nanmean(list(by.values())))}
    return row, p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["mae", "dinov2_b"])
    ap.add_argument("--norms", nargs="+", default=list(NORMS))
    ap.add_argument("--crops", nargs="+", default=["fixed", "tumor"], choices=list(CROPS))
    ap.add_argument("--k", type=int, default=8, help="evaluation slices per patient")
    ap.add_argument("--slice-mode", default="mid", choices=["even", "mid", "area"])
    ap.add_argument("--zoom", type=float, default=1.6)
    ap.add_argument("--fov-mm", type=float, default=100.0)
    ap.add_argument("--cohorts", nargs="*", default=["ispy1", "ispy2", "duke"])
    ap.add_argument("--cache-root", nargs="*", default=None)
    ap.add_argument("--prefer", default="v2", choices=["v1", "v2"], help="cache format when both are attached")
    ap.add_argument("--out", default="runs/probe")
    ap.add_argument("--batch", type=int, default=192, help="images per forward pass")
    ap.add_argument("--workers", type=int, default=4, help="data-loading processes")
    ap.add_argument("--jobs", type=int, default=4, help="parallel logistic-regression processes")
    ap.add_argument("--limit", type=int, default=0, help="patients per split (debug)")
    ap.add_argument("--no-kinetic", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)

    t_start = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    roots = a.cache_root or (["/kaggle/input", "/kaggle/working"] if os.path.isdir("/kaggle/input")
                             else ["data/cache"])
    caches = find_caches(roots, a.prefer)
    cohorts = [c for c in a.cohorts if c in caches]
    if not cohorts:
        raise SystemExit(f"no caches for {a.cohorts} under {roots}")
    fmts = {c: json.load(open(Path(caches[c]) / "cache_meta.json")).get("format", "v1") for c in cohorts}
    print("caches:", {c: (caches[c], fmts[c]) for c in cohorts}, flush=True)
    tr = patient_table(["train"], cohorts, cache_dirs=caches)
    va = patient_table(["val"], cohorts, cache_dirs=caches)
    if a.limit:
        tr, va = tr.sample(min(a.limit, len(tr)), random_state=0), va.sample(min(a.limit, len(va)), random_state=0)
    table = pd.concat([tr, va]).reset_index(drop=True)
    is_train = np.r_[np.ones(len(tr), bool), np.zeros(len(va), bool)]
    y, cohort = table.pCR.to_numpy(float), table.cohort.to_numpy(str)
    print(f"train {len(tr)} ({int(tr.pCR.sum())} pCR), val {len(va)} ({int(va.pCR.sum())} pCR)", flush=True)

    combos = [(n, c) for n in a.norms for c in a.crops]
    if any(f == "v1" for f in fmts.values()) and "joint_wide" in a.norms:
        print("note: on v1 caches joint_wide == joint_pct (the values above the 99.5th pct are gone)")
    n_gpu = torch.cuda.device_count() if not a.cpu else 0
    devs = [torch.device(f"cuda:{i}") for i in range(n_gpu)] or [torch.device("cpu")]
    models = {}
    for i, name in enumerate(a.backbones):   # spread backbones over the GPUs (Kaggle T4 x2)
        m = Backbone(name, train_blocks=0, grad_checkpointing=False).eval().to(devs[i % len(devs)])
        models[name] = m.half() if devs[0].type == "cuda" else m
    feats = {(b, j, p): np.zeros((len(table), m.dim), np.float32)
             for b, m in models.items() for j in range(len(combos)) for p in ("cls", "gap")}
    kin = np.zeros((len(table), len(FEATURES)), np.float32)

    ds = MultiViews(table, caches, combos, a.k, a.slice_mode, zoom=a.zoom, fov_mm=a.fov_mm)
    loader = DataLoader(ds, batch_size=None, num_workers=a.workers, prefetch_factor=4 if a.workers else None)
    with torch.no_grad():
        for it, item in enumerate(loader):
            i, n = int(item["index"]), int(item["n"])
            imgs = item["images"][:, :n]                                    # (V, n, 3, 240, 240) uint8
            v = imgs.shape[0]
            for b, m in models.items():
                dev = next(m.parameters()).device
                x = eval_transform(imgs.to(dev, non_blocking=True).flatten(0, 1).unsqueeze(0), m.mean, m.std)[0]
                x = x.half() if dev.type == "cuda" else x
                outs = [m(x[s:s + a.batch]) for s in range(0, len(x), a.batch)]
                for p in ("cls", "gap"):
                    f = torch.cat([o[p] for o in outs]).float().view(v, n, -1).mean(1).cpu().numpy()
                    for j in range(v):
                        feats[(b, j, p)][i] = f[j]
            if (it + 1) % 100 == 0 or it + 1 == len(ds):
                el = time.time() - t_start
                print(f"{it + 1}/{len(ds)} patients, {el / 60:.1f} min, eta {el / (it + 1) * (len(ds) - it - 1) / 60:.1f} min",
                      flush=True)
    if not a.no_kinetic:
        for i, r in enumerate(table.itertuples()):
            kin[i] = kinetic_features(load_slab(Path(caches[r.cohort]) / f"{r.pid}.npz"), r.xy_spacing, r.slice_thick)
    for b in models:
        np.savez_compressed(out / f"features_{b}.npz", pid=table.pid.values, is_train=is_train, y=y, cohort=cohort,
                            combos=np.array([f"{n}|{c}" for n, c in combos]),
                            **{f"{j}_{p}": feats[(b, j, p)].astype(np.float16)
                               for j in range(len(combos)) for p in ("cls", "gap")})

    from joblib import Parallel, delayed
    jobs = [({"backbone": b, "norm": combos[j][0], "crop": combos[j][1], "pool": p}, feats[(b, j, p)])
            for b in models for j in range(len(combos)) for p in ("cls", "gap")]
    jobs.append(({"backbone": "-", "norm": "-", "crop": "-", "pool": "cohort_only"},
                 pd.get_dummies(pd.Series(cohort)).to_numpy(float)))
    if not a.no_kinetic:
        jobs.append(({"backbone": "-", "norm": "raw", "crop": "tumor box", "pool": "kinetic_handcrafted"},
                     np.nan_to_num(kin)))
    print(f"scoring {len(jobs)} feature sets ...", flush=True)
    res = Parallel(n_jobs=a.jobs)(delayed(score)(X, y, cohort, is_train) for _, X in jobs)
    rows = [{**key, **r} for (key, _), (r, _) in zip(jobs, res)]
    df = pd.DataFrame(rows).sort_values("cv_auc", ascending=False)
    df.insert(4, "cache", "/".join(sorted(set(fmts.values()))))
    df.to_csv(out / "probe_results.csv", index=False)
    preds = pd.DataFrame({"pid": table.pid[~is_train].values, "cohort": cohort[~is_train], "label": y[~is_train]})
    for (key, _), (_, p) in zip(jobs, res):
        preds["|".join(key.values())] = p
    preds.to_csv(out / "probe_val_preds.csv", index=False)
    cols = ["backbone", "norm", "crop", "pool", "cv_auc", "cv_auc_sd", "val_auc", "val_ci_lo", "val_ci_hi",
            "val_auc_within"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df[cols].round(3).to_string(index=False))
    print(f"done in {(time.time() - t_start) / 60:.1f} min -> {out}")
    return df


if __name__ == "__main__":
    main()
