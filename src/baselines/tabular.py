"""Clinical-only baselines B0 / B1 (LR + RF), fitted on train, scored on val. CPU, a few seconds.

    python -m src.baselines.tabular                     # all four, results/A/<run_id>/
    python -m src.baselines.tabular --models b1_lr --seeds 0

B0: HR, HER2, race, menopause, age.  B1: B0 + log bbox volume (D-008).
Hyperparameters (LR C, RF min_samples_leaf) are chosen by 5-fold CV AUC on train only.
Outputs per run: metrics.json and val_preds.csv in the same format as src.train, encoder.json,
and model.json (LR coefficients / RF feature importances).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src import stats
from src.data.clinical import CAT_LEVELS, ClinicalEncoder
from src.data.metadata import patient_table

ROOT = Path(__file__).resolve().parents[2]
COHORTS = ["ispy1", "ispy2", "duke"]
FEATURES = {"b0": ["HR", "HER2", "race", "menopause", "age"],
            "b1": ["HR", "HER2", "race", "menopause", "age", "log_bbox_vol"]}
MODELS = ["b0_lr", "b0_rf", "b1_lr", "b1_rf"]


def feature_names(enc: ClinicalEncoder):
    names = [f"{f}={v}" for f in enc.cat for v in [*CAT_LEVELS[f], "missing"]]
    return names + enc.cont + [f"{f}_missing" for f in enc.cont]


def fit(kind: str, X, y, seed: int):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    if kind == "lr":
        m = LogisticRegressionCV(Cs=np.logspace(-3, 2, 11), cv=cv, scoring="roc_auc", max_iter=5000)
        m.fit(X, y)
        return m, {"C": float(m.C_[0])}
    rf = RandomForestClassifier(n_estimators=500, max_features="sqrt", random_state=seed, n_jobs=-1)
    g = GridSearchCV(rf, {"min_samples_leaf": [1, 5, 10, 20, 40]}, cv=cv, scoring="roc_auc").fit(X, y)
    return g.best_estimator_, {"min_samples_leaf": g.best_params_["min_samples_leaf"],
                               "cv_auc": float(g.best_score_)}


def git_sha():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "nogit"
    except OSError:
        return "nogit"


def run(name: str, seed: int, out: Path, train: pd.DataFrame, val: pd.DataFrame):
    feats, kind = FEATURES[name.split("_")[0]], name.split("_")[1]
    enc = ClinicalEncoder(feats).fit(train)
    Xtr, Xva = enc.onehot(train), enc.onehot(val)
    ytr = train.pCR.astype(int).to_numpy()
    model, hp = fit(kind, Xtr, ytr, seed)
    prob = model.predict_proba(Xva)[:, 1]
    preds = pd.DataFrame({"pid": val.pid.to_numpy(), "cohort": val.cohort.to_numpy(),
                          "label": val.pCR.to_numpy(), "prob": prob})
    m = stats.summary(preds.label, preds.prob)
    m["by_cohort"] = {c: stats.summary(g.label, g.prob, ci=False) for c, g in preds.groupby("cohort")}
    run_id = f"{name}_s{seed}_{time.strftime('%m%d-%H%M')}_{git_sha()}"
    d = out / run_id
    d.mkdir(parents=True, exist_ok=True)
    names = feature_names(enc)
    weights = model.coef_[0] if kind == "lr" else model.feature_importances_
    (d / "model.json").write_text(json.dumps(
        {"kind": kind, "hyperparams": hp, "intercept": float(model.intercept_[0]) if kind == "lr" else None,
         "weights": dict(zip(names, map(float, weights)))}, indent=2))
    (d / "encoder.json").write_text(json.dumps(enc.to_dict(), indent=2))
    preds.to_csv(d / "val_preds.csv", index=False)
    metrics = {"run_id": run_id, "config": name, "seed": seed, "git": git_sha(), "cohorts": COHORTS,
               "features": feats, "hyperparams": hp, "n_train": len(train), "n_val": len(val), "val": m}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="RF seeds; LR is deterministic and always runs once (seed 0)")
    ap.add_argument("--out", default=str(ROOT / "results" / "A"))
    a = ap.parse_args(argv)
    train, val = patient_table("train", COHORTS), patient_table("val", COHORTS)
    print(f"train {len(train)} ({int(train.pCR.sum())} pCR) | val {len(val)} ({int(val.pCR.sum())} pCR)")
    results = []
    for name in a.models:
        for seed in (a.seeds if name.endswith("_rf") else [0]):
            m = run(name, seed, Path(a.out), train, val)
            v = m["val"]
            print(f"{m['run_id']}: val AUC {v['auc']:.3f} (CI {v['auc_ci'][0]:.3f}-{v['auc_ci'][1]:.3f}) "
                  f"AUPRC {v['auprc']:.3f} ECE {v['ece']:.3f} | "
                  + " ".join(f"{c} {x['auc']:.3f}" for c, x in v["by_cohort"].items()) + f" | {m['hyperparams']}")
            results.append(m)
    return results


if __name__ == "__main__":
    main()
