"""Patient slice bags with a chosen normalisation / crop, from v1 or v2 caches.

Same batch keys and attributes as src/data/datasets.PatientBags (images uint8 (K, 3, 240, 240), valid,
slice_pos, label, hr, her2, index; .labels, .rng, set_epoch), so src/train.py's loaders, augmentation
(train_transform: random 224 crop + affine + intensity jitter) and predict/evaluate work unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .norms import make_views, select_slices
from .slabs import load_slab, raw


class SlabBags(Dataset):
    def __init__(self, table: pd.DataFrame, cache_dirs: dict, norm="joint_pct", crop="tumor", k=8,
                 train=True, slice_mode="mid", mid_frac=0.6, zoom=1.6, fov_mm=100.0, seed=0, encoder=None):
        self.t = table.reset_index(drop=True)
        # clinical inputs for the fusion model (VitFusion); empty for the image-only model
        self.clin = encoder.transform(self.t) if encoder is not None else {
            "cat": np.zeros((len(self.t), 0), np.int64), "cont": np.zeros((len(self.t), 0), np.float32),
            "cont_missing": np.zeros((len(self.t), 0), np.float32)}
        self.paths = [Path(cache_dirs[c]) / f"{p}.npz" for p, c in zip(self.t.pid, self.t.cohort)]
        self.norm, self.crop, self.k, self.train = norm, crop, k, train
        self.slice_mode, self.mid_frac, self.zoom, self.fov_mm = slice_mode, mid_frac, zoom, fov_mm
        self.spacing = self.t.xy_spacing.to_numpy(float)
        self.rng = np.random.default_rng(seed)
        self.labels = self.t.pCR.fillna(-1).to_numpy(np.float32)
        self.hr = self.t.HR.fillna(-1).to_numpy(np.float32)
        self.her2 = self.t.HER2.fillna(-1).to_numpy(np.float32)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        slab = load_slab(self.paths[i])
        idx = select_slices(slab, self.k, self.slice_mode, self.train, self.rng, self.mid_frac)
        v = make_views(slab, idx, self.norm, self.crop, self.spacing[i], zoom=self.zoom, fov_mm=self.fov_mm)
        x = torch.zeros((self.k,) + tuple(v.shape[1:]), dtype=torch.uint8)
        x[:len(idx)] = v
        valid = torch.zeros(self.k, dtype=torch.bool)
        valid[:len(idx)] = True
        n = max(slab["x"].shape[0] - 1, 1)
        return {"images": x, "valid": valid,
                "slice_pos": torch.from_numpy(np.pad(idx / n, (0, self.k - len(idx))).astype(np.float32)),
                "cat": torch.from_numpy(self.clin["cat"][i]), "cont": torch.from_numpy(self.clin["cont"][i]),
                "cont_missing": torch.from_numpy(self.clin["cont_missing"][i]),
                "label": torch.tensor(self.labels[i]), "hr": torch.tensor(self.hr[i]),
                "her2": torch.tensor(self.her2[i]), "index": torch.tensor(i)}

    def set_epoch(self, epoch: int, seed: int = 0):
        self.rng = np.random.default_rng(seed * 1000 + epoch)


class MultiViews(Dataset):
    """For the frozen-feature probe: every (normalisation, crop) view of a patient's evaluation slices at
    once, so each slab is read and decoded a single time. images: (V, K, 3, 240, 240) uint8."""

    def __init__(self, table, cache_dirs, combos, k=8, slice_mode="mid", mid_frac=0.6, zoom=1.6, fov_mm=100.0):
        self.t = table.reset_index(drop=True)
        self.paths = [Path(cache_dirs[c]) / f"{p}.npz" for p, c in zip(self.t.pid, self.t.cohort)]
        self.combos, self.k = list(combos), k
        self.slice_mode, self.mid_frac, self.zoom, self.fov_mm = slice_mode, mid_frac, zoom, fov_mm
        self.spacing = self.t.xy_spacing.to_numpy(float)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        slab = load_slab(self.paths[i])
        idx = select_slices(slab, self.k, self.slice_mode, False, None, self.mid_frac)
        xr = raw(slab, idx)
        views = torch.zeros((len(self.combos), self.k, 3, 240, 240), dtype=torch.uint8)
        for j, (norm, crop) in enumerate(self.combos):
            views[j, :len(idx)] = make_views(slab, idx, norm, crop, self.spacing[i], zoom=self.zoom,
                                             fov_mm=self.fov_mm, x_raw=xr)
        return {"images": views, "n": torch.tensor(len(idx)), "index": torch.tensor(i)}
