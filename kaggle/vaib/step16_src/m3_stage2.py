"""Step 16 stage 2 (the paper's second stage): patient-level features -> Random Forest or logistic regression,
repeated CV on the training patients (out-of-fold predictions) and one refit scored on val; tables in the
layout of the paper's Table 2."""
from __future__ import annotations

import numpy as np
import pandas as pd

import mnj_common as mc


def make(kind, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    if kind == "rf":
        return Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                         ("rf", RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                                                       n_jobs=1, random_state=seed))])
    return mc.make_model(kind, seed)


def _job(X, y, tr, te, kind, seed):
    import warnings
    warnings.filterwarnings("ignore")
    return make(kind, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]


def cv_refit(X, y, folds, Xval, kind="lr", seed=0, jobs=4):
    """{"oof": mean over repeats of the out-of-fold predictions, "fold_auc": [...], "val": refit-on-all prediction}."""
    from joblib import Parallel, delayed
    X, y, Xval = np.asarray(X, float), np.asarray(y, int), np.asarray(Xval, float)
    res = Parallel(n_jobs=jobs)(delayed(_job)(X, y, tr, te, kind, seed + 7 * r + f) for r, f, tr, te in folds)
    reps = max(r for r, *_ in folds) + 1
    oof, fa = np.full((reps, len(y)), np.nan), []
    for (r, f, tr, te), p in zip(folds, res):
        oof[r, te] = p
        fa.append(mc.auc(y[te], p))
    val = _job(np.r_[X, Xval], np.r_[y, np.zeros(len(Xval), int)], np.arange(len(y)),
               np.arange(len(y), len(y) + len(Xval)), kind, seed)
    return {"oof": np.nanmean(oof, 0), "fold_auc": fa, "val": val}


def size_frame(meta):
    """Tumour size, the paper's top-ranked features: the dataset's tumour volume and the tumour-box volume."""
    d = pd.DataFrame(index=meta.index)
    d["log_tum_vol"] = np.log1p(pd.to_numeric(meta["tum_vol"], errors="coerce"))
    d["log_bbox_vol"] = pd.to_numeric(meta["log_bbox_vol"], errors="coerce")
    return d


def paper_table(frame, prob, thr=0.5, n_boot=1000):
    """The paper's Table 2 layout (overall, per subtype, per cohort): N, pCR rate, accuracy, AUC with a 95%
    bootstrap CI, sensitivity, specificity, precision and NPV at threshold `thr`, and the accuracy of always
    predicting 'no pCR', which every accuracy has to be read against."""
    y, p = frame.label.to_numpy(int), np.asarray(prob, float)
    groups = [("overall", np.ones(len(y), bool))]
    groups += [(f"subtype {s}", (frame.subtype == s).to_numpy()) for s in ("TN", "HER2+", "HR+/HER2-")]
    groups += [(f"cohort {c}", (frame.cohort == c).to_numpy()) for c in mc.COH]
    div = lambda a, b: a / b if b else np.nan  # noqa: E731
    rows = []
    for name, s in groups:
        yy, pp = y[s], p[s]
        if not len(yy):
            continue
        pred = (pp >= thr).astype(int)
        tp, tn = int(((pred == 1) & (yy == 1)).sum()), int(((pred == 0) & (yy == 0)).sum())
        fp, fn = int(((pred == 1) & (yy == 0)).sum()), int(((pred == 0) & (yy == 1)).sum())
        lo, hi = mc.boot_ci(yy, pp, n=n_boot) if 0 < yy.sum() < len(yy) else (np.nan, np.nan)
        rows.append({"group": name, "n": len(yy), "n_pcr": int(yy.sum()), "pcr_rate": yy.mean(),
                     "acc": (tp + tn) / len(yy), "acc_if_all_no_pcr": 1 - yy.mean(), "auc": mc.auc(yy, pp),
                     "auc_lo": lo, "auc_hi": hi, "sens": div(tp, tp + fn), "spec": div(tn, tn + fp),
                     "prec": div(tp, tp + fp), "npv": div(tn, tn + fn)})
    return pd.DataFrame(rows)
