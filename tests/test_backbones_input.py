"""New slice encoders (ResNet-18, DINOv2) and the early-phase-only input; no downloads (random weights)."""
import json

import pytest
import torch

from src.models.backbones import ResNetEncoder, build_encoder
from src.models.fusion import PCRModel
from src.train import main
from tests.test_train_smoke import COMMON


def test_resnet_encoder_freezing_and_shapes():
    enc = ResNetEncoder("resnet18", pretrained=False, train_blocks=2)        # layer3 + layer4 trained
    trained = [any(p.requires_grad for p in st.parameters()) for st in enc.stages]
    assert trained == [False, False, False, True, True]
    enc.train()
    assert not enc.stages[1].training and enc.stages[4].training           # frozen BN stays in eval
    out = enc(torch.randn(3, 3, 224, 224))
    assert out["cls"].shape == (3, 512)
    lrs = sorted({g["lr"] for g in enc.param_groups(1e-4, 0.5, 0.05)})
    assert lrs == [5e-5, 1e-4]                                              # layer decay over stages
    assert all(p.requires_grad for p in ResNetEncoder("resnet18", False, "all").parameters())


def test_dinov2_encoder_through_factory():
    enc = build_encoder({"backbone": "tiny-dinov2", "pretrained": False, "train_blocks": 2,
                         "grad_checkpointing": False})
    out = enc(torch.randn(2, 3, 224, 224))
    assert out["cls"].shape == (2, enc.dim) and out["tokens"].shape[1] == 1 + 16 * 16   # 224 / 14 = 16
    assert [any(p.requires_grad for p in b.parameters()) for b in enc.blocks] == [False, False, True, True]


def test_early_input_feeds_the_early_phase_to_all_channels():
    m = {"kind": "image_only", "aggregation": "mean_logit", "backbone": "tiny-random", "train_blocks": 1,
         "grad_checkpointing": False, "input_phases": "early"}
    model = PCRModel(m, [], 0)
    seen = {}
    model.encoder.register_forward_pre_hook(lambda mod, args: seen.update(x=args[0]))   # returns None
    x = torch.randn(2, 3, 3, 224, 224)
    model({"images": x, "valid": torch.ones(2, 3, dtype=torch.bool)})
    got = seen["x"]
    assert torch.equal(got[:, 0], got[:, 1]) and torch.equal(got[:, 2], got[:, 1])
    assert torch.equal(got[:, 1], x.flatten(0, 1)[:, 1])                   # channel 1 = early (G)


@pytest.mark.parametrize("extra", [["model.backbone=resnet18", "model.pretrained=false", "model.train_blocks=all",
                                    "model.input_phases=early"],
                                   ["model.backbone=tiny-dinov2", "model.train_blocks=2", "model.input_phases=early"]])
def test_train_and_export_with_new_backbones(extra, synthetic_cache, tmp_path):
    caches = [f"{c}={d}" for c, d in synthetic_cache.items()]
    no_root = ["--cache-root", str(tmp_path / "none")]
    common = [a for a in COMMON if a != "model.backbone=tiny-random"]
    run = main(["--config", "b3_vit", "--cache", *caches, *no_root, "--out", str(tmp_path / "runs"),
                *common, *extra])
    m = json.loads((run / "metrics.json").read_text())
    assert m["n_val"] == 16 and 0 <= m["val"]["auc"] <= 1
    assert (run / "export_preds.csv").exists()


def test_drop_path_only_in_trained_dinov2_blocks():
    from src.models.vit_encoder import DropPath
    enc = build_encoder({"backbone": "tiny-dinov2", "pretrained": False, "train_blocks": 2,
                         "grad_checkpointing": False, "drop_path": 0.3})
    kinds = [type(b.drop_path).__name__ for b in enc.blocks]
    assert kinds == ["Identity", "Identity", "DropPath", "DropPath"]
    x = torch.randn(4, 3, 224, 224)
    enc.eval()
    with torch.no_grad():
        assert torch.equal(enc(x)["cls"], enc(x)["cls"])                       # deterministic at eval
    enc.train()
    torch.manual_seed(0)
    with torch.no_grad():
        assert not torch.equal(enc(x)["cls"], enc(x)["cls"])                   # stochastic in training
    assert DropPath(0.5).eval()(x) is x
    with pytest.raises(ValueError, match="DINOv2"):
        build_encoder({"backbone": "tiny-random", "pretrained": False, "train_blocks": 2,
                       "grad_checkpointing": False, "drop_path": 0.1})


def test_train_on_ispy_only_with_drop_path(synthetic_cache, tmp_path):
    caches = [f"{c}={d}" for c, d in synthetic_cache.items()]
    common = [a for a in COMMON if a != "model.backbone=tiny-random"]
    run = main(["--config", "b3_vit", "--cache", *caches, "--cache-root", str(tmp_path / "none"),
                "--out", str(tmp_path / "runs"), *common, "model.backbone=tiny-dinov2", "model.train_blocks=2",
                "model.input_phases=early", "model.drop_path=0.2", "model.dropout=0.5", "data.cohorts=[ispy1]"])
    m = json.loads((run / "metrics.json").read_text())
    assert m["cohorts"] == ["ispy1"] and m["n_val"] == 8 and set(m["caches"]) == {"ispy1"}
