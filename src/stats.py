"""Patient-level metrics (D-016). Day 0: the basics used during training.
Day 2 adds paired DeLong, sensitivity/PPV at 90% specificity and reliability plots."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _ok(y):
    y = np.asarray(y)
    return len(np.unique(y[~np.isnan(y)])) == 2


def auc(y, p):
    return float(roc_auc_score(y, p)) if _ok(y) else float("nan")


def auprc(y, p):
    return float(average_precision_score(y, p)) if _ok(y) else float("nan")


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def bootstrap_auc_ci(y, p, n=2000, seed=0, alpha=0.05):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if _ok(y[i]):
            vals.append(roc_auc_score(y[i], p[i]))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def summary(y, p, ci=True, n_boot=2000):
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "auc": auc(y, p), "auprc": auprc(y, p),
           "brier": brier(y, p), "ece": ece(y, p)}
    if ci:
        out["auc_ci"] = bootstrap_auc_ci(y, p, n_boot)
    return out
