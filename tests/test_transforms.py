"""Training augmentation must be rigid (rotation, flips, isotropic scale): a disc stays a disc."""
import numpy as np
import torch

from src.data.transforms import train_transform

MEAN, STD = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)


def _aspect(img):
    ys, xs = np.nonzero(img > 0.5)
    ev = np.linalg.eigvalsh(np.cov(np.stack([ys, xs])))
    return float(np.sqrt(ev[-1] / ev[0]))


def test_augmentation_keeps_shapes():
    yy, xx = np.mgrid[0:240, 0:240]
    disc = ((yy - 120) ** 2 + (xx - 120) ** 2 <= 50 ** 2).astype(np.uint8) * 255
    x = torch.from_numpy(np.repeat(disc[None, None, None], 3, axis=2))          # (1, 1, 3, 240, 240)
    aug = dict(rotate_deg=15.0, scale=(1.0, 1.0), intensity=0.0, crop_jitter=0, flip=True)
    ratios = []
    for seed in range(40):                                                       # covers all flip combinations
        out = train_transform(x, MEAN, STD, aug=aug, generator=torch.Generator().manual_seed(seed))
        ratios.append(_aspect(out[0, 0, 0].numpy()))
    assert max(ratios) < 1.03, f"augmentation distorts shapes: max aspect ratio {max(ratios):.2f}"
