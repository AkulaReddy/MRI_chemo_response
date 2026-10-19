"""Report statistics (plan Section 6, D-016): DeLong, operating points, calibration, subgroup AUCs.

    delong(y, p_a, p_b)            paired DeLong test of two AUCs on the same patients (b - a)
    delong_ci(y, p)                AUC with DeLong 95% CI
    threshold_at_spec(y, p, 0.90)  decision threshold reaching >= 90% specificity on (y, p)
    at_threshold(y, p, t)          sensitivity / specificity / PPV / NPV at a fixed threshold
    reliability(y, p)              calibration curve (bins); plot_reliability draws it
    with_subtype(df)               adds `subtype` (HR+/HER2-, HER2+, TN) from the metadata
    subgroup_table(df, by)         n, pCR+, AUC (+ DeLong CI) per cohort / subtype
    load_preds / ensemble          read val_preds.csv / test_preds.csv; average seeds
    clinical_preds(...)            the B0/B1 clinical baselines fitted on any patients, scored on any others

Thresholds for test-set operating points must come from validation (threshold_at_spec on val, then
at_threshold on test); computed on the same set they are optimistic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import stats
from .data.metadata import load_metadata

SUBTYPES = {"HRposHER2neg": "HR+/HER2-", "HER2pos": "HER2+", "TripleNeg": "TN"}


# ----------------------------------------------------------------------------- DeLong
def _midrank(x):
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def _delong(y, preds):
    """AUCs (k,) and their covariance (k, k) for k score vectors on the same patients (Sun & Xu 2014)."""
    y = np.asarray(y).astype(int)
    preds = np.atleast_2d(np.asarray(preds, float))
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.concatenate([pos, neg], 1)])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1) / 2 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_ci(y, p, alpha=0.05):
    aucs, cov = _delong(y, p)
    se = float(np.sqrt(cov[0, 0]))
    z = sps.norm.ppf(1 - alpha / 2)
    return {"auc": float(aucs[0]), "ci": (float(max(0, aucs[0] - z * se)), float(min(1, aucs[0] + z * se))),
            "se": se}


def delong(y, p_a, p_b, alpha=0.05):
    """Paired DeLong test, AUC(b) - AUC(a). Two-sided p-value and 95% CI of the difference."""
    aucs, cov = _delong(y, np.vstack([p_a, p_b]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[1] - aucs[0])
    se = float(np.sqrt(max(var, 1e-12)))
    z = sps.norm.ppf(1 - alpha / 2)
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "ci": (diff - z * se, diff + z * se),
            "z": diff / se, "p": float(2 * sps.norm.sf(abs(diff / se)))}


# ----------------------------------------------------------------------------- operating points
def threshold_at_spec(y, p, spec=0.90):
    """Smallest threshold t such that specificity (p <= t counted negative) is >= spec on these patients."""
    neg = np.sort(np.asarray(p, float)[np.asarray(y) == 0])
    return float(neg[min(len(neg) - 1, int(np.ceil(spec * len(neg))) - 1)])


def at_threshold(y, p, t):
    y, pred = np.asarray(y).astype(int), (np.asarray(p, float) > t).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    tn, fn = int(((pred == 0) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    return {"threshold": float(t), "sens": div(tp, tp + fn), "spec": div(tn, tn + fp), "ppv": div(tp, tp + fp),
            "npv": div(tn, tn + fn), "n_called_pcr": tp + fp}


# ----------------------------------------------------------------------------- calibration
def reliability(y, p, bins=10):
    """Equal-width bins: mean predicted, observed pCR rate and count per non-empty bin."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    rows = [(p[idx == b].mean(), y[idx == b].mean(), int((idx == b).sum())) for b in range(bins) if (idx == b).any()]
    return pd.DataFrame(rows, columns=["mean_pred", "observed", "n"])


def plot_reliability(ax, curves: dict, title="Calibration (val)"):
    """curves: {name: (y, p)}. Diagonal = perfect calibration; marker size ~ patients in the bin."""
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    for name, (y, p) in curves.items():
        r = reliability(y, p)
        ax.plot(r.mean_pred, r.observed, "-o", ms=3, label=f"{name} (ECE {stats.ece(y, p):.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="predicted pCR probability", ylabel="observed pCR rate", title=title)
    ax.legend(fontsize=7)


# ----------------------------------------------------------------------------- subgroups
def with_subtype(df):
    meta = load_metadata()
    out = df.copy()
    out["subtype"] = out.pid.map(meta.HR_HER2_STATUS.map(SUBTYPES)).fillna("unknown")
    if "cohort" not in out:
        out["cohort"] = out.pid.map(meta.cohort)
    return out


def subgroup_table(df, by="subtype"):
    """n, pCR+, AUC with DeLong CI per group (groups with one class get AUC NaN)."""
    rows = []
    for g, d in with_subtype(df).groupby(by):
        ok = d.label.nunique() == 2
        r = delong_ci(d.label, d.prob) if ok else {"auc": np.nan, "ci": (np.nan, np.nan)}
        rows.append({by: g, "n": len(d), "pcr": int(d.label.sum()), "auc": r["auc"],
                     "ci_lo": r["ci"][0], "ci_hi": r["ci"][1]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- predictions
def load_preds(path):
    df = pd.read_csv(path)
    if "pid" not in df and "Unnamed: 0" in df:
        df = df.rename(columns={"Unnamed: 0": "pid"})
    return df[["pid", "cohort", "label", "prob"]].drop_duplicates("pid").reset_index(drop=True)


def ensemble(frames):
    """Mean probability of several runs (seeds) over the patients they share."""
    m = frames[0][["pid", "cohort", "label"]].copy()
    for i, f in enumerate(frames):
        m = m.merge(f[["pid", "prob"]].rename(columns={"prob": f"p{i}"}), on="pid")
    m["prob"] = m[[f"p{i}" for i in range(len(frames))]].mean(1)
    return m[["pid", "cohort", "label", "prob"]]


def align(preds: dict):
    """Restrict every model to the patients all of them scored (needed for paired tests)."""
    common = set.intersection(*(set(d.pid) for d in preds.values()))
    return {k: d[d.pid.isin(common)].sort_values("pid").reset_index(drop=True) for k, d in preds.items()}


def summary_row(y, p, t=None):
    """AUC (DeLong CI), AUPRC, Brier, ECE and the operating point at threshold t (default: 90% spec here)."""
    d = delong_ci(y, p)
    t = threshold_at_spec(y, p) if t is None else t
    op = at_threshold(y, p, t)
    return {"n": len(y), "pcr": int(np.sum(y)), "auc": d["auc"], "ci_lo": d["ci"][0], "ci_hi": d["ci"][1],
            "auprc": stats.auprc(y, p), "brier": stats.brier(y, p), "ece": stats.ece(y, p),
            **{f"{k}@90spec": v for k, v in op.items()}}


def clinical_preds(train: pd.DataFrame, score: pd.DataFrame, name="b1_lr", seed=0):
    """Fit a clinical baseline (src/baselines/tabular.py, same features and tuning) on `train`, score `score`."""
    from .baselines.tabular import FEATURES, fit
    from .data.clinical import ClinicalEncoder
    enc = ClinicalEncoder(FEATURES[name.split("_")[0]]).fit(train)
    model, _ = fit(name.split("_")[1], enc.onehot(train), train.pCR.astype(int).to_numpy(), seed)
    return pd.DataFrame({"pid": score.pid.to_numpy(), "cohort": score.cohort.to_numpy(),
                         "label": score.pCR.to_numpy(), "prob": model.predict_proba(enc.onehot(score))[:, 1]})


def find_run_dirs(roots, prefix=""):
    """Run folders (with best.pt and config_resolved.yaml) under `roots` whose name starts with `prefix`."""
    out = []
    for r in roots:
        for f in Path(r).rglob("best.pt"):
            d = f.parent
            if (d / "config_resolved.yaml").exists() and d.name.startswith(prefix):
                out.append(d)
    return sorted(set(out))
