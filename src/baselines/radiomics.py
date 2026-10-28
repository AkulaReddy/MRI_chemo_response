"""Radiomics baselines (docs/ideas.md items 3, 12, 17): subtype lookup, radiomics, radiomics + subtype,
with and without ComBat harmonisation. Fitted on train, scored on val. CPU, under a minute.

    python -m src.baselines.radiomics --features results/A/radiomics_<id>/features.csv [--out results/A]

Models (one run folder each, same format as src.baselines.tabular / src.train):
- `subtype`: pCR rate of the patient's HR/HER2 subtype in train (a 3-row lookup table).
- `rad`: logistic regression on the radiomics features (src.radiomics.FEATURES).
- `rad_sub`: `rad` + HR and HER2 status.
- `rad_nocombat`, `rad_sub_nocombat`: the same without ComBat, to measure what harmonisation adds.

Pipeline: missing values -> train median; ComBat by protocol group (src.combat.protocol_groups), fitted on
all train-split patients with features (labeled or not: ComBat does not use the outcome), with HR/HER2 as
covariates so the subtype differences between groups are kept; standardise; LogisticRegressionCV (C by
5-fold CV AUC on train). metrics.json also reports how well the batch can be predicted from the features
before and after ComBat (balanced accuracy, 5-fold CV on train; chance = 1 / number of batches).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from src import stats
from src.baselines.tabular import git_sha
from src.combat import ComBat, protocol_groups
from src.data.metadata import load_metadata, load_split
from src.radiomics import FEATURES

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["subtype", "rad", "rad_sub", "rad_nocombat", "rad_sub_nocombat"]


def covariates(t: pd.DataFrame) -> np.ndarray:
    """HR+ and HER2+ indicators plus missing flags (negative status is the reference)."""
    return np.stack([t.HR.eq(1), t.HER2.eq(1), t.HR.isna(), t.HER2.isna()], 1).astype(float)


def load_table(paths):
    feats = pd.concat([pd.read_csv(p) for p in paths]).drop_duplicates("pid", keep="last").set_index("pid")
    meta = load_metadata().drop(columns=["split"]).join(load_split()[["split"]])
    t = meta.join(feats[[c for c in feats.columns if c not in meta.columns]], how="inner")
    if "error" in t:
        t = t[t["error"].isna()]
    return t[t.split.isin(["train", "val"])]       # test rows are never read here (D-006)


def batch_predictability(X, batch, seed=0):
    y = pd.factorize(batch)[0]
    k = min(5, np.bincount(y).min())
    if len(set(y)) < 2 or k < 2:
        return None
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    acc = cross_val_score(clf, StandardScaler().fit_transform(X), y, cv=StratifiedKFold(k, shuffle=True,
                          random_state=seed), scoring="balanced_accuracy")
    return round(float(acc.mean()), 3)


def run(name, t, out: Path, seed=0, min_batch=20):
    train_all = t[t.split == "train"]
    train, val = train_all[train_all.labeled], t[(t.split == "val") & t.labeled]
    info = {}
    if name == "subtype":
        rate = train.groupby("HR_HER2_STATUS").pCR.mean()
        prob = val.HR_HER2_STATUS.map(rate).fillna(train.pCR.mean()).to_numpy()
        model = {"kind": "lookup", "pCR_rate_by_subtype": rate.round(4).to_dict()}
    else:
        use_combat, with_sub = not name.endswith("_nocombat"), name.startswith("rad_sub")
        med = train_all[FEATURES].median()
        Xall = train_all[FEATURES].fillna(med).to_numpy(float)
        Xtr, Xva = train[FEATURES].fillna(med).to_numpy(float), val[FEATURES].fillna(med).to_numpy(float)
        groups = protocol_groups(t, min_size=min_batch, fit_index=train_all.index)
        if use_combat:
            cb = ComBat().fit(Xall, groups[train_all.index], covariates(train_all))
            info["batch_acc_before"] = batch_predictability(Xall, groups[train_all.index], seed)
            info["batch_acc_after"] = batch_predictability(
                cb.transform(Xall, groups[train_all.index], covariates(train_all)), groups[train_all.index], seed)
            Xtr = cb.transform(Xtr, groups[train.index], covariates(train))
            Xva = cb.transform(Xva, groups[val.index], covariates(val))
            info["batches"] = groups[train_all.index].value_counts().to_dict()
            info["batch_chance"] = round(1 / len(info["batches"]), 3)
        names = list(FEATURES)
        if with_sub:
            Xtr, Xva = np.hstack([Xtr, covariates(train)]), np.hstack([Xva, covariates(val)])
            names += ["HR+", "HER2+", "HR_missing", "HER2_missing"]
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                  scoring="roc_auc", max_iter=5000).fit(sc.transform(Xtr), train.pCR.astype(int))
        prob = lr.predict_proba(sc.transform(Xva))[:, 1]
        model = {"kind": "lr", "C": float(lr.C_[0]), "intercept": float(lr.intercept_[0]),
                 "weights_standardised": dict(zip(names, map(float, lr.coef_[0]))), "combat": use_combat}
    preds = pd.DataFrame({"pid": val.pid.to_numpy(), "cohort": val.cohort.to_numpy(),
                          "label": val.pCR.to_numpy(), "prob": prob})
    m = stats.summary(preds.label, preds.prob)
    m["by_cohort"] = {c: stats.summary(g.label, g.prob, ci=False) for c, g in preds.groupby("cohort")}
    run_id = f"r_{name}_s{seed}_{time.strftime('%m%d-%H%M')}_{git_sha()}"
    d = out / run_id
    d.mkdir(parents=True, exist_ok=True)
    preds.to_csv(d / "val_preds.csv", index=False)
    (d / "model.json").write_text(json.dumps(model, indent=2, default=float))
    metrics = {"run_id": run_id, "config": f"radiomics_{name}", "seed": seed, "git": git_sha(),
               "cohorts": sorted(t.cohort.unique()), "n_train": len(train), "n_train_combat": len(train_all),
               "n_val": len(val), "features": FEATURES, **info, "val": m}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return metrics, preds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", nargs="+", required=True, help="CSV(s) from scripts/extract_radiomics.py")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--cohorts", nargs="*", default=None)
    ap.add_argument("--min-batch", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "results" / "A"))
    a = ap.parse_args(argv)
    t = load_table(a.features)
    if a.cohorts:
        t = t[t.cohort.isin(a.cohorts)]
    print(f"train {int(((t.split == 'train') & t.labeled).sum())} labeled "
          f"({int((t.split == 'train').sum())} for ComBat) | val {int(((t.split == 'val') & t.labeled).sum())}")
    out, res, preds = Path(a.out), [], {}
    for name in a.models:
        m, preds[name] = run(name, t, out, min_batch=a.min_batch)
        v = m["val"]
        extra = (f" | batch acc {m['batch_acc_before']} -> {m['batch_acc_after']} (chance {m['batch_chance']})"
                 if "batch_acc_after" in m else "")
        print(f"{m['run_id']}: val AUC {v['auc']:.3f} (CI {v['auc_ci'][0]:.3f}-{v['auc_ci'][1]:.3f}) "
              + " ".join(f"{c} {x['auc']:.3f}" for c, x in v["by_cohort"].items()) + extra)
        res.append(m)
    if "subtype" in preds:   # paired comparison of every model with the subtype lookup
        b = preds["subtype"]
        for name, p in preds.items():
            if name != "subtype":
                d = stats.paired_bootstrap_auc_diff(b.label, b.prob, p.prob)
                print(f"  {name} - subtype: {d['diff']:+.3f} (CI {d['ci'][0]:+.3f} to {d['ci'][1]:+.3f}, p {d['p']:.3f})")
    return res


if __name__ == "__main__":
    main()
