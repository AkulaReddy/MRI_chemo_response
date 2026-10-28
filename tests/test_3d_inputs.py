"""3D information: slice attention (A), 2.5D input (B), 3D CNN on slice blocks (C), 3D GLCM (D)."""
import json

import numpy as np
import pytest
import torch

from src.data import datasets
from src.data.clinical import ClinicalEncoder
from src.data.datasets import PatientBags, configure_bags, patient_table
from src.models.backbones import R3DEncoder
from src.radiomics import DIRECTIONS_3D, glcm_3d
from src.train import main
from tests.test_train_smoke import COMMON


@pytest.fixture(autouse=True)
def default_bags():
    configure_bags()          # every test starts (and ends) with the default bag options
    yield
    configure_bags()


def _bags(synthetic_cache, **kw):
    t = patient_table(["train"], ["ispy1"], cache_dirs=synthetic_cache)
    enc = ClinicalEncoder().fit(t)
    return PatientBags(t, synthetic_cache, enc, **kw)


def test_early_25d_channels_are_neighbouring_slices(synthetic_cache):
    ds = _bags(synthetic_cache, k=4, train=False, margin=1, slices="early_25d")
    d = np.load(ds.paths[0])
    rgb, _ = ds._load(0)
    e, t0 = d["rgb"][..., 1], int(d["t0"])
    lo = max(0, t0 - 1)
    for s in range(1, len(rgb) - 1):                       # interior slices: z-1, z, z+1 of the early phase
        z = lo + s
        assert np.array_equal(rgb[s, ..., 0], e[z - 1]) and np.array_equal(rgb[s, ..., 1], e[z])
        assert np.array_equal(rgb[s, ..., 2], e[z + 1])


def test_block_sampling_is_contiguous_and_centred(synthetic_cache):
    ds = _bags(synthetic_cache, k=12, train=False, sampling="block")
    item = ds[0]
    s = ds._load(0)[0].shape[0]
    v = item["valid"].numpy()
    first = (12 - s) // 2                                  # a short tumour sits in the middle of the block
    assert v.sum() == s and v[first:first + s].all()
    ds = _bags(synthetic_cache, k=2, train=True, sampling="block", seed=1)
    idx, at = ds._pick(7)
    assert at == 0 and np.array_equal(idx, np.arange(idx[0], idx[0] + 2))
    with pytest.raises(ValueError, match="contiguous"):
        _bags(synthetic_cache, k=4, sampling="block", min_area_frac=0.1)


def test_r3d_encoder_shapes_and_freezing():
    enc = R3DEncoder(pretrained=False, train_blocks=2, crop=64, size=32)
    trained = [any(p.requires_grad for p in st.parameters()) for st in enc.stages]
    assert trained == [False, False, False, True, True]
    out = enc(torch.randn(2, 3, 8, 224, 224))
    assert out["cls"].shape == (2, 512)


@pytest.mark.parametrize("extra", [
    ["model.aggregation=slice_attn"],                                                     # A
    ["model.input_phases=early_25d", "data.margin=1"],                                     # B
    ["model.backbone=r3d_18", "model.pretrained=false", "model.aggregation=mean", "model.vol_crop=64",
     "model.vol_size=32", "data.k_train=6", "data.k_eval=6", "export.k=6", "data.margin=1"],   # C
    ["model.backbone=r3d_18", "model.pretrained=false", "model.aggregation=mean", "model.vol_crop=64",
     "model.vol_size=32", "data.k_train=6", "data.k_eval=6", "model.input_phases=enhancement"],
])
def test_train_with_3d_options(extra, synthetic_cache, tmp_path):
    caches = [f"{c}={d}" for c, d in synthetic_cache.items()]
    common = [a for a in COMMON if not a.startswith(("data.k_", "export.k")) or "r3d_18" not in " ".join(extra)]
    run = main(["--config", "b3_vit", "--cache", *caches, "--cache-root", str(tmp_path / "none"),
                "--out", str(tmp_path / "runs"), *common, *extra])
    m = json.loads((run / "metrics.json").read_text())
    assert m["n_val"] == 16 and 0 <= m["val"]["auc"] <= 1
    if "model.backbone=r3d_18" in extra:
        assert datasets.BAG_OPTIONS["sampling"] == "block"


def test_glcm_3d_matches_2d_glcm_in_plane_and_sees_depth():
    from skimage.feature import graycomatrix
    rng = np.random.default_rng(0)
    q = rng.integers(0, 4, (5, 9, 9))
    mask = np.ones(q.shape, bool)
    P = glcm_3d(q, mask, 4)
    k = DIRECTIONS_3D.index((0, 0, 1))                      # along columns = skimage angle 0, per slice
    ref = sum(graycomatrix(q[z].astype(np.uint8), [1], [0], levels=4, symmetric=True)[:, :, 0, 0]
              for z in range(5)).astype(float)
    assert np.allclose(P[k], ref / ref.sum())
    layers = np.repeat(np.arange(5)[:, None, None] % 2, 9, 1).repeat(9, 2)   # alternating in z only
    P = glcm_3d(layers, mask, 2)
    assert P[DIRECTIONS_3D.index((1, 0, 0))][0, 1] > 0.4    # through-plane neighbours differ
    assert P[DIRECTIONS_3D.index((0, 0, 1))][0, 1] == 0     # in-plane neighbours never differ
