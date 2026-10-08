"""Guards against the BreastDCEDL conversion bug (float [0, 1] passed to PIL as RGB = noise)."""
import numpy as np
from PIL import Image

from src.data.rgb_slices import crop_origin, crop_slices, to_rgb_uint8


def _phases(seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[0:6, 0:64, 0:64]
    base = 200 + 50 * np.sin(x / 7.0) + 30 * np.cos(y / 5.0) + rng.normal(0, 5, (6, 64, 64))
    return base, base * 1.6 + 20, base * 1.4 + 10


def test_dtype_range_and_correlation():
    pre, early, late = _phases()
    rgb, (lo, hi) = to_rgb_uint8(pre, early, late)
    assert rgb.dtype == np.uint8 and rgb.shape == (6, 64, 64, 3)
    assert rgb.min() == 0 and rgb.max() == 255 and lo < hi
    for c, src in enumerate((pre, early, late)):
        r = np.corrcoef(rgb[..., c].ravel().astype(float), src.ravel())[0, 1]
        assert r > 0.99, f"channel {c} correlation {r:.3f}"


def test_pil_roundtrip_is_lossless():
    rgb, _ = to_rgb_uint8(*_phases())
    back = np.asarray(Image.fromarray(rgb[3]))
    assert back.dtype == np.uint8 and np.array_equal(back, rgb[3])


def test_joint_clipping_keeps_enhancement_order():
    rgb, _ = to_rgb_uint8(*_phases())
    assert rgb[..., 1].mean() > rgb[..., 0].mean()   # post-contrast brighter than pre


def test_crop_shifts_inward():
    assert crop_origin(10, 224, 256) == 0
    assert crop_origin(250, 224, 256) == 32
    assert crop_origin(128, 224, 256) == 16
    out, (y0, x0) = crop_slices(np.zeros((2, 256, 256, 3), np.uint8), 255, 0, 240)
    assert out.shape == (2, 240, 240, 3) and (y0, x0) == (16, 0)
