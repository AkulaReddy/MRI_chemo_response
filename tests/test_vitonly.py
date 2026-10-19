"""src/vitonly: v2 cache builder, normalisations, crops, slice choice, probe and fine-tuning (CPU, tiny ViTs)."""
import gzip
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from src.data.metadata import load_metadata, load_split
from src.vitonly.norms import CROPS, NORMS, crop_resize, crop_side, make_views, normalise, select_slices
from src.vitonly.slabs import PCTS, load_slab, raw

ROOT = Path(__file__).resolve().parents[1]


def _fake_v2(path, rng, tumor_gain, s=8, with_mask=True):
    """A v2 slab: background ~100, a 40x40 tumour that enhances by `tumor_gain` in early/late."""
    x = rng.normal(100, 10, (s, 256, 256, 3)).clip(0)
    x[..., 1] *= 1.2
    x[1:s - 1, 100:140, 110:150, 1:] += tumor_gain
    scale = np.percentile(x, 99.9)
    flat = x.reshape(-1)[::4]
    mask = np.zeros((s, 256, 256), np.uint8)
    mask[1:s - 1, 100:140, 110:150] = 1
    arrays = dict(phases=(x / scale).astype(np.float16), scale=np.float32(scale),
                  joint_p999=np.float32(scale), clip_v1=np.percentile(flat, [0.5, 99.5]).astype(np.float32),
                  vol_pct=np.stack([np.percentile(x[..., c], PCTS) for c in range(3)]).astype(np.float32),
                  fg_stats=np.array([[x[..., c].mean(), x[..., c].std(), np.median(x[..., c])] for c in range(3)],
                                    np.float32),
                  t0=1, t1=s - 1, z_offset=0, bbox=np.array([100, 140, 110, 150]), crop_origin=np.array([0, 0]),
                  phases_used=np.array([0, 2, 5]), spacing=np.array([0.7, 2.0], np.float32))
    if with_mask:
        arrays["mask"] = mask
    np.savez_compressed(path, **arrays)


@pytest.fixture(scope="module")
def v2_cache(tmp_path_factory):
    """Fake v2 caches for real I-SPY1 / Duke train+val patient IDs; tumour enhancement depends on pCR."""
    split = load_split()
    root = tmp_path_factory.mktemp("cache2")
    rng = np.random.default_rng(0)
    dirs = {}
    for cohort in ("ispy1", "duke"):
        d = root / f"cache2_{cohort}"
        d.mkdir()
        rows = []
        for sp, n in (("train", 12), ("val", 8)):
            for pid in split[(split.cohort == cohort) & (split.split == sp) & split.labeled].pid.head(n):
                _fake_v2(d / f"{pid}.npz", rng, 250 if split.loc[pid, "pCR"] == 1 else 80, with_mask=cohort == "ispy1")
                rows.append(f"{pid},{cohort},ok")
        (d / "cache_index.csv").write_text("pid,cohort,status\n" + "\n".join(rows) + "\n")
        (d / "cache_meta.json").write_text(json.dumps({"cohort": cohort, "format": "v2"}))
        dirs[cohort] = d
    return dirs


def _one_slab(v2_cache):
    return load_slab(next(Path(v2_cache["ispy1"]).glob("*.npz")))


def test_every_norm_in_unit_range(v2_cache):
    s = _one_slab(v2_cache)
    x = raw(s, np.arange(2, 5))
    for n in NORMS:
        out = normalise(x, s["stats"], n)
        assert out.shape == x.shape and out.dtype == np.float32, n
        assert out.min() >= 0 and out.max() <= 1, n
    assert np.allclose(x, raw(s, np.arange(2, 5)))           # normalise() must not modify its input
    with pytest.raises(ValueError):
        normalise(x, s["stats"], "nope")


def test_enhancement_ordering_is_kept(v2_cache):
    """Joint normalisations keep early > pre in the enhancing tumour; per_phase does not have to."""
    s = _one_slab(v2_cache)
    x = raw(s, np.arange(1, 7))
    for n in ("joint_pct", "joint_wide", "joint_tumor", "pre_ref", "zscore"):
        out = normalise(x, s["stats"], n)
        tum = out[:, 105:135, 115:145]
        assert tum[..., 1].mean() > tum[..., 0].mean() + 0.02, n
    k = normalise(x, s["stats"], "kinetic")                  # wash-in channel bright in the tumour only
    assert k[:, 105:135, 115:145, 1].mean() > k[:, :60, :60, 1].mean() + 0.1


def test_crops_contain_the_tumour(v2_cache):
    s = _one_slab(v2_cache)
    for c in CROPS:
        side = crop_side(c, s["bbox"], 0.7, 256)
        assert 64 <= side <= 256
        v = make_views(s, np.array([3]), "joint_pct", c, 0.7)
        assert v.shape == (1, 3, 240, 240) and str(v.dtype) == "torch.uint8"
    assert crop_side("tumor", s["bbox"], 0.7, 256, zoom=1.6) == 64        # 40 px box * 1.6 -> 64
    t = crop_resize(np.zeros((1, 256, 256, 3), np.float32), (0, 10, 0, 10), 64)   # box at the corner
    assert t.shape == (1, 3, 240, 240)


def test_slice_selection(v2_cache):
    s = _one_slab(v2_cache)                       # tumour slices 1..6
    assert list(select_slices(s, 8, "even")) == [1, 2, 3, 4, 5, 6]
    mid = select_slices(s, 8, "mid")
    assert mid.min() >= 1 and mid.max() <= 6 and len(mid) < 6
    assert len(select_slices(s, 4, "area")) == 4
    tr = select_slices(s, 8, "mid", train=True, rng=np.random.default_rng(0))
    assert len(tr) == 8 and set(tr) <= set(range(1, 7))


def _add(tf, name, a):
    raw_ = gzip.compress(nib.Nifti1Image(a.astype(np.float64), np.eye(4)).to_bytes())
    info = tarfile.TarInfo(name)
    info.size = len(raw_)
    tf.addfile(info, io.BytesIO(raw_))


def test_v2_builder_reproduces_v1(tmp_path):
    """Both builders on the same synthetic archive: v2 + joint_pct gives the v1 uint8 images (+-1)."""
    m = load_metadata()
    rows = m[(m.cohort == "ispy1") & (m.n_xy == 256) & m.labeled].head(2)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rng = np.random.default_rng(0)
    with tarfile.open(raw_dir / "BreastDCEDL_test_min_crop.tar.gz", "w:gz") as tf:
        for r in rows.itertuples():
            mask = np.zeros((20, 256, 256))
            mask[2:12, int(r.sraw):int(r.eraw), int(r.scol):int(r.ecol)] = 1
            for i, k in enumerate((int(r.pre), int(r.post_early), int(r.post_late))):
                _add(tf, f"x/dce/{r.pid}_spy1_vis1_dce_aqc_{k}.nii.gz",
                     rng.normal(100 + 20 * i, 10, (20, 256, 256)) + 300 * mask * (i > 0))
            _add(tf, f"x/mask/{r.pid}_spy1_vis1_mask.nii.gz", mask)
    for script, out, extra in (("build_cache.py", "v1", ["--xy-mm", "0"]), ("build_cache_v2.py", "v2", []),
                               ("build_cache.py", "b2", [])):          # b2: main's resampled builder 2
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--cohort", "ispy1", "--raw", str(raw_dir),
                        "--out", str(tmp_path / out), "--workers", "1", *extra], check=True, cwd=ROOT)
    meta2 = json.loads((tmp_path / "v2" / "cache_meta.json").read_text())
    assert meta2["format"] == "v2" and meta2["status_counts"]["ok"] == 2
    idx = pd.read_csv(tmp_path / "v2" / "cache_index.csv")
    ok = idx[idx.status == "ok"]
    assert (ok[["sat_early", "sat_late"]].max(axis=1) > 0.1).all()   # the bright tumour is partly clipped in v1
    for r in rows.itertuples():
        v1 = np.load(tmp_path / "v1" / f"{r.pid}.npz")
        s2 = load_slab(tmp_path / "v2" / f"{r.pid}.npz")
        assert s2["format"] == "v2" and load_slab(tmp_path / "v1" / f"{r.pid}.npz")["format"] == "v1"
        o2 = np.load(tmp_path / "v2" / f"{r.pid}.npz")["crop_origin"]
        dy, dx = (int(a) - int(b) for a, b in zip(v1["crop_origin"], o2))
        img = normalise(raw(s2), s2["stats"], "joint_pct")[:, dy:dy + 240, dx:dx + 240]
        assert np.abs(np.rint(img * 255) - v1["rgb"]).max() <= 1
        assert s2["t0"] == int(v1["t0"]) and s2["t1"] == int(v1["t1"])
        b2 = load_slab(tmp_path / "b2" / f"{r.pid}.npz")          # builder 2: read as uint8, own spacing
        assert b2["format"] == "v1" and b2["spacing_xy"] == pytest.approx(0.7) and b2["spacing_z"] == 2.0
        assert make_views(b2, np.array([b2["t0"]]), "kinetic", "mm", 99.0).shape == (1, 3, 240, 240)
        from src.vitonly.kinetics import kinetic_features
        assert np.isfinite(kinetic_features(b2, 99.0, 99.0)).all()      # CSV spacing ignored for b2
    from src.vitonly.slabs import cache_label, find_caches
    for c in ("v1", "v2", "b2"):
        json.loads((tmp_path / c / "cache_meta.json").read_text())
    assert cache_label(tmp_path / "b2") == "v1 uint8 builder 2" and cache_label(tmp_path / "v2") == "v2 raw"
    assert find_caches([str(tmp_path)], "v2")["ispy1"] == str(tmp_path / "v2")
    assert find_caches([str(tmp_path)], "v1")["ispy1"] == str(tmp_path / "b2")   # builder 2 beats builder 1


def test_probe_end_to_end(v2_cache, tmp_path):
    from src.vitonly.probe import main
    df = main(["--backbones", "tiny-random", "tiny-dinov2", "--norms", "joint_pct", "kinetic",
               "--crops", "fixed", "tumor", "--k", "3", "--cohorts", "ispy1", "duke", "--cache-root",
               str(Path(v2_cache["ispy1"]).parent), "--out", str(tmp_path / "probe"), "--workers", "0", "--jobs", "1", "--cpu"])
    assert len(df) == 2 * 2 * 2 * 2 + 2                     # backbones x norms x crops x pools + 2 references
    assert {"cohort_only", "kinetic_handcrafted"} <= set(df["pool"])
    hand = df[df["pool"] == "kinetic_handcrafted"].iloc[0]
    assert hand.cv_auc > 0.8                                  # synthetic tumours enhance more when pCR = 1
    assert (tmp_path / "probe" / "features_tiny-dinov2.npz").exists()
    assert len(pd.read_csv(tmp_path / "probe" / "probe_val_preds.csv")) == 16


@pytest.mark.parametrize("backbone,pool,agg", [("tiny-random", "gap", "mean_logit"),
                                               ("tiny-dinov2", "both", "mil")])
def test_finetune_end_to_end(v2_cache, tmp_path, backbone, pool, agg):
    from src.vitonly.finetune import main
    run = main(["--config", "vo_base", "--cpu", "--cache-root", str(Path(v2_cache["ispy1"]).parent),
                "--out", str(tmp_path / "runs"), "--results-dir", str(tmp_path / "results"), "--set",
                f"model.backbone={backbone}", f"model.pool={pool}", f"model.aggregation={agg}",
                "model.grad_checkpointing=false", "optim.max_epochs=2", "optim.freeze_epochs=1",
                "data.num_workers=0", "optim.batch_patients=4", "data.k_train=3", "data.k_eval=4"])
    m = json.loads((run / "metrics.json").read_text())
    assert m["n_train"] == 24 and m["n_val"] == 16 and m["epochs_run"] == 2
    log = pd.read_csv(run / "log.csv")
    assert list(log.frozen) == [True, False] and log.lr_enc_top.iloc[0] == 0.0
    preds = pd.read_csv(run / "val_preds.csv")
    assert len(preds) == 16 and preds.prob.between(0, 1).all()
    assert (tmp_path / "results" / run.name / "metrics.json").exists()


@pytest.mark.parametrize("backbone", ["tiny-random", "tiny-dinov2"])
def test_frozen_backbone_gets_no_gradient(backbone):
    """LP warm-up: with freeze_backbone only the head receives gradients; afterwards the top blocks do too."""
    import torch
    from src.vitonly.model import VitOnly
    m = VitOnly({"backbone": backbone, "train_blocks": 2, "grad_checkpointing": False})
    batch = {"images": torch.randn(2, 3, 3, 224, 224), "valid": torch.tensor([[1, 1, 0], [1, 0, 0]]).bool()}
    for frozen in (True, False):
        m.zero_grad(set_to_none=True)
        m.freeze_backbone = frozen
        m(batch)["logit"].sum().backward()
        enc = [p.grad for p in m.encoder.parameters() if p.requires_grad]
        assert m.head.weight.grad is not None
        assert all(g is None for g in enc) if frozen else any(g is not None for g in enc)
    n_blocks = m.encoder.n_layers
    trained = [i for i, b in enumerate(m.encoder.blocks) if any(p.requires_grad for p in b.parameters())]
    assert trained == list(range(n_blocks - 2, n_blocks))


@pytest.mark.parametrize("config", ["vo_fusion", "vo_base"])
def test_fusion_and_image_only_runs_rescore_identically(v2_cache, tmp_path, config):
    """Our fusion model (VitFusion + clinical tokens) trains, reports train AUC, and src.predict_run reloads
    the checkpoint (with its clinical encoder) to the same val predictions."""
    import torch
    from src.predict_run import score_run
    from src.vitonly.finetune import main
    root = str(Path(v2_cache["ispy1"]).parent)
    run = main(["--config", config, "--cpu", "--cache-root", root, "--out", str(tmp_path / "runs"), "--set",
                "model.backbone=tiny-random", "model.grad_checkpointing=false", "optim.max_epochs=1",
                "optim.freeze_epochs=0", "data.num_workers=0", "optim.batch_patients=4", "data.k_train=3",
                "data.k_eval=4"])
    m = json.loads((run / "metrics.json").read_text())
    assert 0 <= m["val"]["train_auc"] <= 1 and (run / "train_preds.csv").exists()
    assert (run / "encoder.json").exists() == (config == "vo_fusion")
    df = score_run(run, ["ispy1", "duke"], ["val"], cache_roots=[root], dev=torch.device("cpu"))
    ref = pd.read_csv(run / "val_preds.csv").set_index("pid").prob
    assert len(df) == 16 and np.allclose(df.set_index("pid").prob.loc[ref.index], ref, atol=1e-5)


def test_runs_in_the_same_minute_get_their_own_folders(v2_cache, tmp_path):
    from src.vitonly.finetune import main
    args = ["--config", "vo_base", "--cpu", "--cache-root", str(Path(v2_cache["ispy1"]).parent), "--out",
            str(tmp_path / "runs"), "--set", "model.backbone=tiny-random", "model.grad_checkpointing=false",
            "optim.max_epochs=1", "optim.freeze_epochs=0", "data.num_workers=0", "optim.batch_patients=4",
            "data.k_train=2", "data.k_eval=2", "eval_train=false"]
    a, b = main(args), main(args)
    assert a != b and (a / "best.pt").exists() and (b / "best.pt").exists()
