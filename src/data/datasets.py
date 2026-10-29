"""Patient-level slice bags from the uint8 cache (D-009, D-016).

Every model sees a patient as a bag of K slices (training: K random tumor slices; evaluation:
up to K_eval slices evenly spaced over the tumor, padded with a mask). Labels are patient-level.
Each item also carries `pre_offset` = lo / (hi - lo) of the patient's clip range, so the model can
recover intensities proportional to the raw signal (`model.input_phases: enhancement`).
The test split is refused unless `allow_test=True`, which only scripts/final_eval.py passes (D-006).

Two options change what a bag holds (set per run by `models.fusion.build_model` from the model config,
through `configure_bags`, or passed to PatientBags directly):
- `slices`: "phases" (default) = each slice's pre/early/late as its 3 channels; "early_25d" = the early
  phase of the slice and of its two neighbours (z-1, z, z+1), so every 2D slice carries some depth (2.5D).
- `sampling`: "spread" (default) = K random (train) / evenly spaced (eval) tumour slices; "block" = K
  *contiguous* slices (a 3D block for 3D encoders): a random block position in training, the centred
  block in evaluation, and short tumours centred in the block with empty slices on both sides.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .clinical import ClinicalEncoder
from .metadata import load_metadata, load_split, patient_table  # noqa: F401  (re-exported)

SLICES, SAMPLINGS = ("phases", "early_25d"), ("spread", "block")
BAG_OPTIONS = {"slices": "phases", "sampling": "spread"}   # defaults for PatientBags; see configure_bags


def configure_bags(slices: str = "phases", sampling: str = "spread"):
    """Default bag options for datasets created after this call (build_model calls it, so the training
    script needs no model-specific arguments)."""
    assert slices in SLICES and sampling in SAMPLINGS, (slices, sampling)
    BAG_OPTIONS.update(slices=slices, sampling=sampling)


class PatientBags(Dataset):
    def __init__(self, table: pd.DataFrame, cache_dirs: dict, encoder: ClinicalEncoder,
                 k: int = 8, train: bool = True, margin: int = 0, preload: bool = False, seed: int = 0,
                 min_area_frac: float = 0.0, slices: str | None = None, sampling: str | None = None):
        self.t = table.reset_index(drop=True)
        self.min_area_frac = min_area_frac
        self.slices = slices or BAG_OPTIONS["slices"]
        self.sampling = sampling or BAG_OPTIONS["sampling"]
        assert self.slices in SLICES and self.sampling in SAMPLINGS, (self.slices, self.sampling)
        if self.sampling == "block" and min_area_frac > 0:
            raise ValueError("min_area_frac removes slices, so a block would not be contiguous")
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
        """(tumour slices (S, 240, 240, 3) uint8, pre_offset)."""
        d = np.load(self.paths[i])
        off = pre_offset(d["clip"]) if "clip" in d.files else 0.0
        return self._slices(d), off

    def _slices(self, d):
        t0, t1 = int(d["t0"]), int(d["t1"])
        rgb = d["rgb"]
        n = rgb.shape[0]
        if self.slices == "early_25d":   # channels = early phase of slices z-1, z, z+1 (edges repeated)
            e = rgb[..., 1]
            z = np.arange(n)
            rgb = np.stack([e[np.maximum(z - 1, 0)], e, e[np.minimum(z + 1, n - 1)]], -1)
        if self.min_area_frac > 0 and ("area" in d.files or "mask" in d.files):
            # drop tumour slices whose mask area is under min_area_frac of the largest one (I-SPY only:
            # Duke has no mask, so all its tumour slices are kept). Margin slices are not added here.
            a = d["area"][t0:t1] if "area" in d.files else d["mask"][t0:t1].sum((1, 2))
            keep = np.flatnonzero(a >= self.min_area_frac * max(int(a.max()), 1))
            return rgb[t0:t1][keep if len(keep) else [int(np.argmax(a))]]
        lo, hi = max(0, t0 - self.margin), min(n, t1 + self.margin)
        return rgb[lo:hi]                                        # (S, 240, 240, 3) uint8

    def _pick(self, s):
        """(slice indices, position of the first one in the bag)."""
        if self.sampling == "block":
            if s <= self.k:
                return np.arange(s), (self.k - s) // 2
            start = int(self.rng.integers(0, s - self.k + 1)) if self.train else (s - self.k) // 2
            return np.arange(start, start + self.k), 0
        return self._pick_spread(s), 0

    def _pick_spread(self, s):
        if self.train:
            return np.sort(self.rng.choice(s, self.k, replace=s < self.k))
        if s <= self.k:
            return np.arange(s)
        return np.unique(np.linspace(0, s - 1, self.k).round().astype(int))

    def __getitem__(self, i):
        rgb, off = self._cache[i] if self._cache is not None else self._load(i)
        idx, at = self._pick(rgb.shape[0])
        x = np.zeros((self.k,) + rgb.shape[1:], np.uint8)
        x[at:at + len(idx)] = rgb[idx]
        valid = np.zeros(self.k, bool)
        valid[at:at + len(idx)] = True
        pos = np.zeros(self.k, np.float32)
        pos[at:at + len(idx)] = idx / max(rgb.shape[0] - 1, 1)
        return {
            "images": torch.from_numpy(x).permute(0, 3, 1, 2),     # (K, 3, 240, 240) uint8
            "valid": torch.from_numpy(valid),
            "slice_pos": torch.from_numpy(pos),
            "cat": torch.from_numpy(self.clin["cat"][i]),
            "cont": torch.from_numpy(self.clin["cont"][i]),
            "cont_missing": torch.from_numpy(self.clin["cont_missing"][i]),
            "pre_offset": torch.tensor(off, dtype=torch.float32),
            "label": torch.tensor(self.labels[i]),
            "hr": torch.tensor(self.hr[i]),
            "her2": torch.tensor(self.her2[i]),
            "index": torch.tensor(i),
        }

    def set_epoch(self, epoch: int, seed: int = 0):
        self.rng = np.random.default_rng(seed * 1000 + epoch)


def pre_offset(clip) -> float:
    """Cache value v (0..1) maps to raw = lo + v * (hi - lo), so raw / (hi - lo) = v + lo / (hi - lo)."""
    lo, hi = float(clip[0]), float(clip[1])
    return max(lo, 0.0) / max(hi - lo, 1e-6)


def worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    ds.rng = np.random.default_rng([int(ds.rng.integers(1 << 31)), worker_id])
