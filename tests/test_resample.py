"""Builder v2 resampling: physical sizes are preserved, the grid is centred, the mask stays aligned."""
import numpy as np

from src.data.resample import axis_grid, resample, tumour_z_grid
from src.data.rgb_slices import to_rgb_uint8, tumour_clip


def _cube(shape, z, y, x, value=1.0):
    v = np.zeros(shape, np.float32)
    v[z[0]:z[1], y[0]:y[1], x[0]:x[1]] = value
    return v


def test_identity_grid_is_exact():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(6, 40, 40)).astype(np.float32)
    s, o = axis_grid(20, 1.0, 1.0, 40)
    out = resample(v, (1, s, s), (0, o, o), (6, 40, 40))
    assert np.allclose(out, v, atol=1e-5)


def test_same_physical_size_after_resampling():
    """A 50 mm box is 50 / 0.7 = 71 px wide whatever the scan's pixel size."""
    for dxy in (0.5, 0.78, 1.0):
        n = int(round(50 / dxy))
        c = 100
        v = _cube((3, 200, 200), (0, 3), (c - n // 2, c - n // 2 + n), (c - n // 2, c - n // 2 + n))
        s, o = axis_grid(c - n // 2 + n / 2, dxy, 0.7, 240)
        out = resample(v, (1, s, s), (0, o, o), (3, 240, 240), antialias=False)
        width = (out[1, 120] >= 0.5).sum()
        assert abs(width - 50 / 0.7) <= 2, (dxy, width)
        cols = np.flatnonzero(out[1, 120] >= 0.5)
        assert abs((cols[0] + cols[-1] + 1) / 2 - 120) <= 1, "tumour not centred"


def test_tumour_z_grid_counts_and_margins():
    # 10 slices of 1 mm -> 5 slices of 2 mm, +2 margin each side
    scale, offset, n, t0, t1 = tumour_z_grid(10, 20, 40, 1.0, 2.0, 2)
    assert (n, t0, t1) == (9, 2, 7) and scale == 2.0
    # tumour at the top of the volume: margin slices outside the volume are dropped
    scale, offset, n, t0, t1 = tumour_z_grid(0, 6, 6, 2.0, 2.0, 2)
    assert (n, t0, t1) == (6, 0, 6)
    assert offset == 0.0


def test_mask_stays_on_the_bright_tumour():
    shape, z, y, x = (30, 128, 128), (8, 20), (40, 70), (50, 90)
    img = 100 + _cube(shape, z, y, x, 300)
    mask = _cube(shape, z, y, x)
    zs, zo, n, t0, t1 = tumour_z_grid(*z, shape[0], 1.2, 2.0, 2)
    (ys, yo), (xs, xo) = axis_grid(sum(y) / 2, 0.6, 0.7, 240), axis_grid(sum(x) / 2, 0.6, 0.7, 240)
    im = resample(img, (zs, ys, xs), (zo, yo, xo), (n, 240, 240))
    m = resample(mask, (zs, ys, xs), (zo, yo, xo), (n, 240, 240), antialias=False) >= 0.5
    assert m[t0:t1].any(-1).any(-1).all(), "every tumour slice has mask"
    assert im[m].mean() > 350 and im[t0:t1][~m[t0:t1] & (im[t0:t1] > 0)].mean() < 200


def test_tumour_clip_avoids_late_saturation():
    rng = np.random.default_rng(0)
    shape = (10, 128, 128)                                   # tumour = 1% of the volume
    box = (3, 7, 50, 70, 50, 70)
    pre = rng.normal(100, 5, shape)
    tum = _cube(shape, box[:2], box[2:4], box[4:])
    early, late = pre + 400 * tum, pre + 600 * tum          # tumour is a tiny, very bright region
    rgb_v1, _ = to_rgb_uint8(pre, early, late)
    rgb_v2, _ = to_rgb_uint8(pre, early, late, clip=tumour_clip([pre, early, late], box))
    t = tum.astype(bool)
    assert (rgb_v1[..., 2][t] == 255).mean() > 0.9          # v1: late phase clipped in the tumour
    assert (rgb_v2[..., 2][t] == 255).mean() < 0.01
    assert rgb_v2[..., 2][t].mean() > rgb_v2[..., 1][t].mean()   # late > early still visible
