"""ComBat harmonisation of features across batches (scanner / protocol groups), fitted on train only.

Parametric empirical-Bayes ComBat (Johnson, Li & Rabinovic 2007, Biostatistics 8:118), as used for
radiomics harmonisation (Fortin et al. 2017; Orlhac et al. 2018). For every feature it removes each batch's
additive and multiplicative effect, while keeping the effect of the covariates (e.g. subtype).

Rules: fit on train only and apply the stored parameters to val/test; never use the outcome (pCR) as a
covariate (that leaks the label). Batches unseen in training are left unadjusted.
"""
from __future__ import annotations

import numpy as np


class ComBat:
    def __init__(self, max_iter: int = 100, tol: float = 1e-4):
        self.max_iter, self.tol = max_iter, tol

    def fit(self, Y: np.ndarray, batch, covariates: np.ndarray | None = None):
        Y = np.asarray(Y, float)
        batch = np.asarray(batch)
        n, p = Y.shape
        X = np.zeros((n, 0)) if covariates is None else np.asarray(covariates, float).reshape(n, -1)
        self.batches = sorted(set(batch.tolist()))
        B = np.stack([batch == b for b in self.batches], 1).astype(float)          # (n, K)
        D = np.concatenate([B, X], 1)
        coef = np.linalg.lstsq(D, Y, rcond=None)[0]                                 # (K + q, p)
        n_k = B.sum(0)
        self.alpha = (n_k / n) @ coef[:len(self.batches)]                           # grand mean (p,)
        self.beta = coef[len(self.batches):]                                        # covariate effects
        resid = Y - D @ coef
        self.sigma = np.sqrt((resid ** 2).mean(0)).clip(min=1e-8)
        Z = (Y - self.alpha - X @ self.beta) / self.sigma

        self.gamma, self.delta = {}, {}
        g_hat = np.stack([Z[batch == b].mean(0) for b in self.batches])            # (K, p)
        d_hat = np.stack([Z[batch == b].var(0, ddof=1) if (batch == b).sum() > 1 else np.ones(p)
                          for b in self.batches]).clip(min=1e-8)
        g_bar, t2 = g_hat.mean(1), g_hat.var(1, ddof=1).clip(min=1e-8)              # prior on gamma (per batch)
        m, s2 = d_hat.mean(1), d_hat.var(1, ddof=1).clip(min=1e-8)                  # inverse-gamma prior on delta
        a, bb = (2 * s2 + m ** 2) / s2, (m ** 3 + m * s2) / s2
        for k, b in enumerate(self.batches):
            z = Z[batch == b]
            nk = len(z)
            g, d = g_hat[k], d_hat[k]
            for _ in range(self.max_iter):
                g_new = (nk * t2[k] * g_hat[k] + d * g_bar[k]) / (nk * t2[k] + d)
                d_new = (bb[k] + 0.5 * ((z - g_new) ** 2).sum(0)) / (nk / 2 + a[k] - 1)
                done = np.max(np.abs(g_new - g) / (np.abs(g) + 1e-8)) < self.tol and \
                    np.max(np.abs(d_new - d) / d) < self.tol
                g, d = g_new, d_new
                if done:
                    break
            self.gamma[b], self.delta[b] = g, d
        self.n_cov = X.shape[1]
        return self

    def transform(self, Y: np.ndarray, batch, covariates: np.ndarray | None = None) -> np.ndarray:
        Y = np.asarray(Y, float)
        batch = np.asarray(batch)
        n = len(Y)
        X = np.zeros((n, 0)) if covariates is None else np.asarray(covariates, float).reshape(n, -1)
        if X.shape[1] != self.n_cov:
            raise ValueError(f"{X.shape[1]} covariates, fitted with {self.n_cov}")
        fixed = self.alpha + X @ self.beta
        Z = (Y - fixed) / self.sigma
        out = Y.copy()
        for b in set(batch.tolist()):
            if b not in self.gamma:
                continue                                      # unseen batch: left as it is
            i = batch == b
            out[i] = (Z[i] - self.gamma[b]) / np.sqrt(self.delta[b]) * self.sigma + fixed[i]
        return out

    def fit_transform(self, Y, batch, covariates=None):
        return self.fit(Y, batch, covariates).transform(Y, batch, covariates)


def protocol_groups(meta, min_size: int = 20, fit_index=None):
    """Batch label per patient from the acquisition protocol: cohort x matrix (256 / 512 / other) x slice
    thickness (<= 1.5 / 1.6-2.4 / >= 2.5 mm) x pixel size (< 0.65 / >= 0.65 mm).

    Groups with fewer than `min_size` patients among `fit_index` (default: all rows) are merged into the
    largest group of the same cohort, so every batch has enough patients to estimate its effect."""
    import pandas as pd
    mat = np.where(meta.n_xy == 256, "256", np.where(meta.n_xy == 512, "512", "other"))
    dz = np.select([meta.slice_thick <= 1.5, meta.slice_thick < 2.5], ["thin", "2mm"], "thick")
    xy = np.where(meta.xy_spacing < 0.65, "fine", "coarse")
    g = pd.Series([f"{c}|{m}|{z}|{x}" for c, m, z, x in zip(meta.cohort, mat, dz, xy)], index=meta.index)
    counts = (g if fit_index is None else g.loc[g.index.intersection(fit_index)]).value_counts()
    out = g.copy()
    for cohort in meta.cohort.unique():
        rows = meta.cohort == cohort
        keep = [k for k, n in counts.items() if k.startswith(cohort + "|") and n >= min_size]
        # too-small groups (and groups seen only outside fit_index) join the cohort's largest group;
        # a cohort without any large-enough group becomes one batch
        out[rows & ~g.isin(keep)] = keep[0] if keep else f"{cohort}|all"
    return out
