import pytest
import torch

from src.config import load_config
from src.data.clinical import ClinicalEncoder
from src.data.datasets import patient_table
from src.losses import total_loss
from src.models.fusion import build_model
from src.models.vit_encoder import TINY, mae_to_vit

CONFIGS = ["b3_vit", "f1_concat", "f2_attn", "f2_meanpool", "f2_mil", "f2_no_bbox", "f2_no_aux"]


def _model(name, **over):
    cfg = load_config(name)
    cfg["model"].update(backbone="tiny-random", train_blocks=2, grad_checkpointing=False, **over)
    enc = ClinicalEncoder(cfg["clinical"]["features"]).fit(patient_table("train"))
    return cfg, enc, build_model(cfg, enc)


def _batch(enc, b=3, k=4):
    c = enc.transform(patient_table("train").head(b))
    valid = torch.ones(b, k, dtype=torch.bool)
    valid[0, 2:] = False                                      # patient 0 has 2 padded slices
    return {"images": torch.randn(b, k, 3, 224, 224), "valid": valid, "slice_pos": torch.rand(b, k),
            "cat": torch.from_numpy(c["cat"]), "cont": torch.from_numpy(c["cont"]),
            "cont_missing": torch.from_numpy(c["cont_missing"]),
            "label": torch.tensor([0.0, 1.0, 0.0]), "hr": torch.tensor([1.0, -1.0, 0.0]),
            "her2": torch.tensor([0.0, 1.0, -1.0])}


@pytest.mark.parametrize("name", CONFIGS)
def test_forward_backward(name):
    cfg, enc, model = _model(name)
    batch = _batch(enc)
    out = model(batch)
    assert out["logit"].shape == (3,) and torch.isfinite(out["logit"]).all()
    assert out["img_emb"].shape == (3, 32)
    if cfg["model"]["aux_heads"]:
        assert out["aux_hr"].shape == (3,)
    if cfg["model"]["kind"] == "attn":
        n_img = 4 if cfg["model"]["aggregation"] == "tokens" else 1
        t = 1 + n_img + len(cfg["clinical"]["features"])
        assert out["attn"].shape == (3, t, t)
        if n_img == 4:                                        # padded slice tokens get no attention
            assert out["attn"][0, 0, 3:5].abs().max() < 1e-6
    loss, _ = total_loss(out, batch, cfg["loss"])
    loss.backward()
    assert all(p.grad is None for p in model.encoder.blocks[0].parameters())   # frozen
    assert all(p.grad is not None for p in model.encoder.blocks[-1].parameters())


@pytest.mark.parametrize("name", ["f2_attn", "b3_vit", "f2_mil"])
def test_padding_does_not_change_prediction(name):
    _, enc, model = _model(name, dropout=0.0)
    model.eval()
    b = _batch(enc)
    with torch.no_grad():
        p1 = model(b)["logit"][0]
        b["images"][0, 2:] = torch.randn(2, 3, 224, 224) * 10     # change padded slices only
        p2 = model(b)["logit"][0]
    assert torch.allclose(p1, p2, atol=1e-5)


def test_mae_weights_copy_exactly():
    from transformers import ViTMAEConfig, ViTMAEModel
    torch.manual_seed(0)
    mae = ViTMAEModel(ViTMAEConfig(**TINY, image_size=224, patch_size=16, mask_ratio=0.0)).eval()
    vit = mae_to_vit(mae).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        a = mae(pixel_values=x, noise=torch.arange(196).float().expand(2, 196)).last_hidden_state
        b = vit(pixel_values=x).last_hidden_state
    assert torch.allclose(a, b, atol=1e-6)
