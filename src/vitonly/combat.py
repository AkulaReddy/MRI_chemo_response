"""ComBat harmonisation (Johnson et al. 2007; as in neuroCombat / neuroHarmonize), fitted on train only.

Each feature is modelled as  x = alpha + covariates @ beta + gamma_batch + delta_batch * eps.  ComBat estimates the
batch shift gamma and scale delta (shrunk towards the mean over features by empirical Bayes, so small batches
are not over-corrected) and removes them, while the covariate effects beta are kept. Here:
    batch       the scanner / site source: cohort (Duke, I-SPY1, I-SPY2) or a finer acquisition protocol
    covariates  HR and HER2 (subtype): cohorts differ in subtype mix, and without them ComBat would remove real
                biology along with the scanner effect. NEVER pCR: it is the label, unknown for a new patient.

    cb = ComBat().fit(X_train, batch_train, cov_train)
    X_train_h, X_val_h = cb.transform(X_train, batch_train, cov_train), cb.transform(X_val, batch_val, cov_val)

`fit` uses only the patients it is given; `transform` applies the stored estimates to new patients from the
same batches (neuroCombat re-estimates on all data at once, which would leak validation patients into the fit).
"""
from __future__ import annotations

import numpy as np


class ComBat:
    def __init__(self, eb: bool = True, max_iter: int = 1000, tol: float = 1e-4):
        self.eb, self.max_iter, self.tol = eb, max_iter, tol

    @staticmethod
    def _cov(cov, n):
        return np.zeros((n, 0)) if cov is None else np.asarray(cov, float).reshape(n, -1)

    def fit(self, X, batch, cov=None):
        X = np.asarray(X, float)
        n, p = X.shape
        batch = np.asarray(batch).astype(str)
        self.batches = sorted(set(batch))
        B = np.stack([batch == b for b in self.batches], 1).astype(float)     # (n, n_batches)
        C = self._cov(cov, n)
        self.cov_mean = C.mean(0) if C.shape[1] else np.zeros(0)
        C = C - self.cov_mean                                                  # centred: grand mean stays put
        D = np.hstack([B, C])
        beta = np.linalg.lstsq(D, X, rcond=None)[0]                            # (n_batches + q, p)
        nb = B.sum(0)
        self.grand_mean = (nb / n) @ beta[:len(self.batches)]
        self.beta_cov = beta[len(self.batches):]
        self.var_pooled = np.maximum(((X - D @ beta) ** 2).mean(0), 1e-12)
        s = (X - self.grand_mean - C @ self.beta_cov) / np.sqrt(self.var_pooled)
        self.gamma, self.delta = {}, {}
        for k, b in enumerate(self.batches):
            sb = s[batch == b]
            g_hat = sb.mean(0)
            d_hat = sb.var(0, ddof=1) if len(sb) > 1 else np.ones(p)
            d_hat = np.maximum(d_hat, 1e-6)
            if self.eb and p > 1 and len(sb) > 1:
                g_hat, d_hat = self._eb(sb, g_hat, d_hat)
            self.gamma[b], self.delta[b] = g_hat, d_hat
        self.n_per_batch = dict(zip(self.batches, nb.astype(int)))
        return self

    def _eb(self, sb, g_hat, d_hat):
        """Parametric empirical-Bayes shrinkage of one batch's (gamma, delta) across features."""
        n = len(sb)
        g_bar, t2 = g_hat.mean(), max(g_hat.var(ddof=1), 1e-12)
        m, v = d_hat.mean(), max(d_hat.var(ddof=1), 1e-12)
        a, b = (2 * v + m ** 2) / v, (m * v + m ** 3) / v                      # inverse-gamma prior on delta
        g_old, d_old = g_hat.copy(), d_hat.copy()
        for _ in range(self.max_iter):
            g_new = (n * t2 * g_hat + d_old * g_bar) / (n * t2 + d_old)
            d_new = (0.5 * ((sb - g_new) ** 2).sum(0) + b) / (n / 2 + a - 1)
            change = max(np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-12)),
                         np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-12)))
            g_old, d_old = g_new, d_new
            if change < self.tol:
                break
        return g_old, np.maximum(d_old, 1e-6)

    def transform(self, X, batch, cov=None):
        X = np.asarray(X, float)
        n = len(X)
        batch = np.asarray(batch).astype(str)
        unknown = sorted(set(batch) - set(self.batches))
        if unknown:
            raise ValueError(f"ComBat was not fitted on batch(es) {unknown}; fitted on {self.batches}")
        C = self._cov(cov, n)
        C = C - self.cov_mean if C.shape[1] else C
        mean = self.grand_mean + C @ self.beta_cov
        s = (X - mean) / np.sqrt(self.var_pooled)
        out = np.empty_like(s)
        for b in set(batch):
            i = batch == b
            out[i] = (s[i] - self.gamma[b]) / np.sqrt(self.delta[b])
        return out * np.sqrt(self.var_pooled) + mean

    def fit_transform(self, X, batch, cov=None):
        return self.fit(X, batch, cov).transform(X, batch, cov)
