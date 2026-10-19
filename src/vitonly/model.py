"""Image-only pCR model on slice bags (the improved B3).

Same interface as src/models/fusion.PCRModel (encoder.mean/std, param_groups, forward(batch) -> logit,
slice_logits, img_emb), so src/train.py's predict/evaluate/export work on it. Differences from B3:
pooling of patch tokens (`gap`) instead of MAE's untrained [CLS], other backbones, optional linear-probe
warm-up (`freeze_backbone`: the encoder runs without gradients, only the head learns).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.pooling import GatedAttentionMIL, masked_mean

from .backbones import Backbone


class VitOnly(nn.Module):
    POOLS = ("cls", "gap", "both")
    AGGS = ("mean_logit", "mean", "mil")

    def __init__(self, m: dict):
        super().__init__()
        self.pool, self.agg = m.get("pool", "gap"), m.get("aggregation", "mean_logit")
        assert self.pool in self.POOLS and self.agg in self.AGGS, (self.pool, self.agg)
        self.encoder = Backbone(m.get("backbone", "mae"), m.get("train_blocks", 4), m.get("grad_checkpointing", True))
        d = self.encoder.dim * (2 if self.pool == "both" else 1)
        self.drop = nn.Dropout(m.get("dropout", 0.2))
        self.head = nn.Linear(d, 1)
        if self.agg == "mil":
            self.mil = GatedAttentionMIL(d, dropout=m.get("dropout", 0.2))
        self.freeze_backbone = False

    def param_groups(self, lr_head, lr_backbone, layer_decay, weight_decay):
        groups = self.encoder.param_groups(lr_backbone, layer_decay, weight_decay)
        enc = {id(p) for p in self.encoder.parameters()}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in enc]
        groups.append({"params": [p for p in rest if p.ndim >= 2], "lr": lr_head, "weight_decay": weight_decay})
        groups.append({"params": [p for p in rest if p.ndim < 2], "lr": lr_head, "weight_decay": 0.0})
        return [g for g in groups if g["params"]]

    def features(self, images, valid):
        b, k = images.shape[:2]
        sel = valid.flatten()
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_backbone):
            out = self.encoder(images.flatten(0, 1)[sel])
        f = {"cls": out["cls"], "gap": out["gap"]}.get(self.pool)
        if f is None:
            f = torch.cat([out["cls"], out["gap"]], -1)
        h = f.new_zeros((b * k, f.shape[-1]))
        h[sel] = f
        return h.view(b, k, -1)

    def forward(self, batch):
        valid = batch["valid"]
        h = self.features(batch["images"], valid)                        # (B, K, D)
        res = {"img_emb": masked_mean(h, valid)}
        if self.agg == "mean_logit":
            sl = self.head(self.drop(h)).squeeze(-1)
            res["slice_logits"] = sl
            res["logit"] = masked_mean(sl.unsqueeze(-1), valid).squeeze(-1)
        elif self.agg == "mil":
            pooled, res["mil_weights"] = self.mil(h, valid)
            res["logit"] = self.head(self.drop(pooled)).squeeze(-1)
        else:
            res["logit"] = self.head(self.drop(res["img_emb"])).squeeze(-1)
        return res


class VitFusion(VitOnly):
    """Our ViT input pipeline (normalisation, tumour crop, backbone) + the F2 attention fusion with clinical data.

    Tokens, as in src/models/fusion.PCRModel(kind="attn", aggregation="tokens"): a learned [CLS], one token per
    slice (backbone feature -> d_model, + slice position), one token per clinical feature (HR, HER2, race,
    menopause, age, log tumour-box volume), then n_layers pre-norm transformer layers; [CLS] -> logit.
    `attn` (B, T, T) of the last layer is returned for interpretation (row 0 = what [CLS] attended to).
    """

    def __init__(self, m: dict, cardinalities, n_cont):
        super().__init__(m)
        from src.models.fusion import ClinicalTokens, FusionBlock
        D, d, p = self.head.in_features, m.get("d_model", 256), m.get("dropout", 0.2)
        self.img_proj = nn.Sequential(nn.Linear(D, d), nn.LayerNorm(d))
        self.slice_pos = nn.Linear(1, d)
        self.img_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.clin = ClinicalTokens(cardinalities, n_cont, d)
        self.blocks = nn.ModuleList(FusionBlock(d, m.get("n_heads", 4), m.get("ffn", 512), p)
                                    for _ in range(m.get("n_layers", 2)))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        if self.agg == "mil":
            del self.mil

    def forward(self, batch):
        valid = batch["valid"]
        h = self.features(batch["images"], valid)                        # (B, K, D)
        res = {"img_emb": masked_mean(h, valid)}
        img = self.img_proj(h) + self.slice_pos(batch["slice_pos"].unsqueeze(-1).to(h.dtype)) + self.img_type
        clin = self.clin(batch["cat"], batch["cont"], batch["cont_missing"]).to(img.dtype)
        b = img.shape[0]
        x = torch.cat([self.cls.expand(b, -1, -1).to(img.dtype), img, clin], 1)
        keep = torch.cat([valid.new_ones((b, 1)), valid, valid.new_ones((b, clin.shape[1]))], 1)
        for blk in self.blocks:
            x, w = blk(x, ~keep)
        res["logit"] = self.head(self.drop(self.norm(x[:, 0]))).squeeze(-1)
        res["attn"] = w
        return res


def build(m: dict, encoder=None):
    """VitFusion when model.kind is "attn" (needs the fitted ClinicalEncoder), else image-only VitOnly."""
    if m.get("kind", "image_only") == "attn":
        return VitFusion(m, encoder.cardinalities, len(encoder.cont))
    return VitOnly(m)
