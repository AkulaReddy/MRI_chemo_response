"""Clinical feature encoding (D-008), fitted on train only.

Categorical (with an explicit "missing" level): HR, HER2, race, menopause, [cohort: ablation only].
Continuous (standardised with train mean/SD, missing -> train median + flag): age, log bbox volume.
Treatment arm and MammaPrint are deliberately absent (context baseline B2 only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAT_LEVELS = {            # value -> index; anything else (NaN) -> last index ("missing")
    "HR": [0, 1],
    "HER2": [0, 1],
    "race": ["white", "black", "other"],
    "menopause": [0, 1],
    "cohort": ["ispy1", "ispy2", "duke"],
}
CONT = ["age", "log_bbox_vol"]
DEFAULT_FEATURES = ["HR", "HER2", "race", "menopause", "age", "log_bbox_vol"]


def race_column(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=object)
    known = df["race_white"].notna() & df["race_black"].notna()
    out[known] = "other"
    out[known & (df["race_black"] == 1)] = "black"
    out[known & (df["race_white"] == 1)] = "white"
    return out


class ClinicalEncoder:
    def __init__(self, features=DEFAULT_FEATURES):
        unknown = set(features) - set(CAT_LEVELS) - set(CONT)
        if unknown:
            raise ValueError(f"unknown clinical features {unknown}")
        self.features = list(features)
        self.cat = [f for f in self.features if f in CAT_LEVELS]
        self.cont = [f for f in self.features if f in CONT]
        self.stats = {}

    @staticmethod
    def _column(df, f):
        return race_column(df) if f == "race" else df[f]

    def fit(self, train: pd.DataFrame):
        for f in self.cont:
            v = train[f].astype(float)
            med = float(v.median())
            filled = v.fillna(med)
            self.stats[f] = {"median": med, "mean": float(filled.mean()), "std": float(filled.std() or 1.0)}
        return self

    @property
    def cardinalities(self):
        return [len(CAT_LEVELS[f]) + 1 for f in self.cat]

    def transform(self, df: pd.DataFrame):
        """dict: cat (N, n_cat) int64, cont (N, n_cont) float32, cont_missing (N, n_cont) float32."""
        if self.cont and not self.stats:
            raise RuntimeError("fit() the encoder on train first")
        cat = np.zeros((len(df), len(self.cat)), np.int64)
        for j, f in enumerate(self.cat):
            levels = CAT_LEVELS[f]
            col = self._column(df, f)
            cat[:, j] = [levels.index(v) if v in levels else len(levels) for v in col.tolist()]
        cont = np.zeros((len(df), len(self.cont)), np.float32)
        miss = np.zeros_like(cont)
        for j, f in enumerate(self.cont):
            s = self.stats[f]
            v = df[f].astype(float)
            miss[:, j] = v.isna().to_numpy()
            cont[:, j] = ((v.fillna(s["median"]) - s["mean"]) / s["std"]).to_numpy()
        return {"cat": cat, "cont": cont, "cont_missing": miss}

    def onehot(self, df: pd.DataFrame) -> np.ndarray:
        """Flat design matrix (one-hot categoricals incl. missing, standardised continuous + flags)."""
        t = self.transform(df)
        blocks = [np.eye(k, dtype=np.float32)[t["cat"][:, j]] for j, k in enumerate(self.cardinalities)]
        blocks += [t["cont"], t["cont_missing"]]
        return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(df), 0), np.float32)

    def to_dict(self):
        return {"features": self.features, "stats": self.stats}

    @classmethod
    def from_dict(cls, d):
        enc = cls(d["features"])
        enc.stats = d["stats"]
        return enc
