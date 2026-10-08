"""Patient-level slice bags from the uint8 cache (D-009, D-016).

Every model sees a patient as a bag of K slices (training: K random tumor slices; evaluation:
up to K_eval slices evenly spaced over the tumor, padded with a mask). Labels are patient-level.
The test split is refused unless `allow_test=True`, which only scripts/final_eval.py passes (D-006).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .clinical import ClinicalEncoder
from .metadata import load_metadata, load_split


def patient_table(splits, cohorts=("ispy1", "ispy2", "duke"), labeled_only=True, allow_test=False,
                  cache_dirs=None):
    """Metadata rows for the requested splits/cohorts, joined with the frozen split.

    If `cache_dirs` ({cohort: dir}) is given, patients without a successfully cached volume are
    dropped and reported.
    """
    splits = [splits] if isinstance(splits, str) else list(splits)
    if "test" in splits and not allow_test:
        raise PermissionError("test split requested outside the final evaluation (D-006)")
    split = load_split()
    meta = load_metadata()
    meta = meta.drop(columns=["split"]).join(split[["split"]])
    t = meta[meta.split.isin(splits) & meta.cohort.isin(cohorts)]
    if labeled_only:
        t = t[t.labeled]
    if cache_dirs is not None:
        ok = set()
        for c in cohorts:
            if c in cache_dirs:
                idx = pd.read_csv(Path(cache_dirs[c]) / "cache_index.csv")
                ok |= set(idx.loc[idx.status == "ok", "pid"])
        missing = sorted(set(t.pid) - ok)
        if missing:
            print(f"[data] {len(missing)} of {len(t)} patients have no cache entry, dropped "
                  f"(first: {missing[:5]})")
        t = t[t.pid.isin(ok)]
    return t.copy()


class PatientBags(Dataset):
    def __init__(self, table: pd.DataFrame, cache_dirs: dict, encoder: ClinicalEncoder,
                 k: int = 8, train: bool = True, margin: int = 0, preload: bool = False, seed: int = 0):
        self.t = table.reset_index(drop=True)
        self.paths = [Path(cache_dirs[c]) / f"{p}.npz" for p, c in zip(self.t.pid, self.t.cohort)]
        self.clin = encoder.transform(self.t)
        self.k, self.train, self.margin = k, train, margin
        self.rng = np.random.default_rng(seed)
        self.labels = self.t.pCR.fillna(-1).to_numpy(np.float32)
        self.hr = self.t.HR.fillna(-1).to_numpy(np.float32)
        self.her2 = self.t.HER2.fillna(-1).to_numpy(np.float32)
        self._cache = [self._load(i) for i in range(len(self))] if preload else None

    def __len__(self):
        return len(self.t)

    def _load(self, i):
        d = np.load(self.paths[i])
        t0, t1 = int(d["t0"]), int(d["t1"])
        n = d["rgb"].shape[0]
        lo, hi = max(0, t0 - self.margin), min(n, t1 + self.margin)
        return d["rgb"][lo:hi]                                   # (S, 240, 240, 3) uint8

    def _pick(self, s):
        if self.train:
            return np.sort(self.rng.choice(s, self.k, replace=s < self.k))
        if s <= self.k:
            return np.arange(s)
        return np.unique(np.linspace(0, s - 1, self.k).round().astype(int))

    def __getitem__(self, i):
        rgb = self._cache[i] if self._cache is not None else self._load(i)
        idx = self._pick(rgb.shape[0])
        x = np.zeros((self.k,) + rgb.shape[1:], np.uint8)
        x[:len(idx)] = rgb[idx]
        valid = np.zeros(self.k, bool)
        valid[:len(idx)] = True
        return {
            "images": torch.from_numpy(x).permute(0, 3, 1, 2),     # (K, 3, 240, 240) uint8
            "valid": torch.from_numpy(valid),
            "slice_pos": torch.from_numpy(np.pad(idx / max(rgb.shape[0] - 1, 1),
                                                 (0, self.k - len(idx))).astype(np.float32)),
            "cat": torch.from_numpy(self.clin["cat"][i]),
            "cont": torch.from_numpy(self.clin["cont"][i]),
            "cont_missing": torch.from_numpy(self.clin["cont_missing"][i]),
            "label": torch.tensor(self.labels[i]),
            "hr": torch.tensor(self.hr[i]),
            "her2": torch.tensor(self.her2[i]),
            "index": torch.tensor(i),
        }

    def set_epoch(self, epoch: int, seed: int = 0):
        self.rng = np.random.default_rng(seed * 1000 + epoch)


def worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    ds.rng = np.random.default_rng([int(ds.rng.integers(1 << 31)), worker_id])
