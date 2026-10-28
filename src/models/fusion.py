"""pCR models on slice bags: B3 (image only), F1 (concat + MLP), F2 (attention fusion, M3).

Image encoders: 2D (each slice encoded alone, then aggregated over the bag) or 3D (`r3d_18`: the bag is a
block of contiguous slices encoded as one volume; the aggregation then sees a single "slice").
Aggregations: mean_logit / mean / mil / tokens (F2), and slice_attn (image only): a small transformer over
the slice features with their position in the tumour, so the model can use how slices relate in depth.

All take a batch dict from `PatientBags` with images already transformed to (B, K, 3, 224, 224)
and return a dict with `logit` (B,), optional `aux_hr`/`aux_her2` (B,), `img_emb` (B, D) and,
depending on the model, `slice_logits` (B, K), `mil_weights` (B, K) or `attn` (B, T, T).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.datasets import configure_bags
from .pooling import GatedAttentionMIL, masked_mean
from .backbones import build_encoder

RE_MAX = 3.0      # relative enhancement 0..300% is mapped to 0..1 (larger values are clipped)
RE_FLOOR = 0.05   # floor of the pre-contrast denominator, in units of the patient's clip range


def enhancement_maps(x01: torch.Tensor, pre_offset: torch.Tensor) -> torch.Tensor:
    """(N, 3, H, W) cache values in 0..1 (R/G/B = pre/early/late) -> (N, 3, H, W) in 0..1:
    early (anatomy, per-patient scale), (early - pre) / pre and (late - pre) / pre.

    The ratios use raw-proportional intensities (cache value + `pre_offset`, see datasets.pre_offset),
    so they cancel the scanner's overall gain: the same tissue gives the same value on any scanner.
    Dark pre-contrast voxels (air, suppressed fat) are floored at RE_FLOOR to keep the ratio finite."""
    pre = x01[:, 0:1]
    den = (pre + pre_offset.view(-1, 1, 1, 1)).clamp_min(RE_FLOOR)
    rel = ((x01[:, 1:3] - pre) / den).clamp(0, RE_MAX) / RE_MAX
    return torch.cat([x01[:, 1:2], rel], 1)


class FusionBlock(nn.Module):
    """Pre-norm transformer layer that also returns its (head-averaged) attention weights."""

    def __init__(self, d, heads, ffn, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        h = self.n1(x)
        a, w = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=True, average_attn_weights=True)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n2(x)))
        return x, w


class ClinicalTokens(nn.Module):
    """One d-dim token per clinical feature: embedding table (categorical) or linear (continuous),
    plus a learned missing embedding (continuous) and a learned type embedding per feature."""

    def __init__(self, cardinalities, n_cont, d):
        super().__init__()
        self.cat = nn.ModuleList(nn.Embedding(k, d) for k in cardinalities)
        self.cont = nn.ModuleList(nn.Linear(1, d) for _ in range(n_cont))
        self.cont_missing = nn.Parameter(torch.zeros(max(n_cont, 1), d))
        self.type_emb = nn.Parameter(torch.randn(len(cardinalities) + n_cont, d) * 0.02)

    @property
    def n_tokens(self):
        return len(self.cat) + len(self.cont)

    def forward(self, cat, cont, cont_missing):
        toks = [emb(cat[:, j]) for j, emb in enumerate(self.cat)]
        toks += [lin(cont[:, j:j + 1]) + cont_missing[:, j:j + 1] * self.cont_missing[j]
                 for j, lin in enumerate(self.cont)]
        if not toks:
            return cat.new_zeros((cat.shape[0], 0, self.type_emb.shape[1]), dtype=torch.float)
        return torch.stack(toks, 1) + self.type_emb


class PCRModel(nn.Module):
    KINDS = ("image_only", "concat", "attn")
    AGGS = ("mean_logit", "mean", "mil", "tokens", "slice_attn")
    PHASES = ("rgb", "early", "enhancement", "early_25d")

    def __init__(self, m: dict, cardinalities, n_cont):
        super().__init__()
        self.kind, self.agg = m["kind"], m["aggregation"]
        assert self.kind in self.KINDS and self.agg in self.AGGS, (self.kind, self.agg)
        assert not (self.agg == "tokens" and self.kind != "attn"), "tokens aggregation needs kind=attn"
        assert not (self.agg == "mean_logit" and self.kind != "image_only"), "mean_logit is for image_only"
        assert not (self.agg == "slice_attn" and self.kind != "image_only"), "slice_attn is for image_only"
        self.encoder = build_encoder(m)
        self.is_3d = getattr(self.encoder, "is_3d", False)
        # which DCE phases the encoder sees: "rgb" = pre/early/late as R/G/B (default); "early" = the early
        # post-contrast phase copied into all 3 channels (= a 1-channel input with the RGB filters summed);
        # "enhancement" = early, (early - pre)/pre, (late - pre)/pre (relative-enhancement maps);
        # "early_25d" = the early phase of slices z-1, z, z+1 (built by the dataset; passed through here)
        self.input_phases = m.get("input_phases", "rgb")
        assert self.input_phases in self.PHASES, self.input_phases
        assert not (self.is_3d and self.input_phases == "early_25d"), "a 3D encoder already sees the neighbours"
        D, d, p = self.encoder.dim, m.get("d_model", 256), m.get("dropout", 0.2)
        self.drop = nn.Dropout(p)
        if self.agg == "mil":
            self.mil = GatedAttentionMIL(D, dropout=p)

        if self.kind == "image_only" and self.agg == "slice_attn":
            self.sa_proj = nn.Sequential(nn.Linear(D, d), nn.LayerNorm(d))
            self.sa_pos = nn.Linear(1, d)
            self.sa_cls = nn.Parameter(torch.zeros(1, 1, d))
            self.sa_blocks = nn.ModuleList(FusionBlock(d, m.get("n_heads", 4), m.get("ffn", 512), p)
                                           for _ in range(m.get("n_layers", 2)))
            self.sa_norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)
            aux_dim = d
        elif self.kind == "image_only":
            self.head = nn.Linear(D, 1)
            aux_dim = D
        else:
            self.img_proj = nn.Sequential(nn.Linear(D, d), nn.LayerNorm(d))
            aux_dim = d
        if self.kind == "concat":
            n_clin = sum(min(8, k) for k in cardinalities) + 2 * n_cont
            self.clin_emb = nn.ModuleList(nn.Embedding(k, min(8, k)) for k in cardinalities)
            self.clin_mlp = nn.Sequential(nn.Linear(n_clin, d), nn.GELU(), nn.Dropout(p))
            self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, 1))
        if self.kind == "attn":
            self.clin = ClinicalTokens(cardinalities, n_cont, d)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.img_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.slice_pos = nn.Linear(1, d)
            self.blocks = nn.ModuleList(FusionBlock(d, m.get("n_heads", 4), m.get("ffn", 512), p)
                                        for _ in range(m.get("n_layers", 2)))
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)
        self.aux = m.get("aux_heads", False)
        if self.aux:
            self.aux_hr, self.aux_her2 = nn.Linear(aux_dim, 1), nn.Linear(aux_dim, 1)

    # ------------------------------------------------------------------
    def param_groups(self, lr_head, lr_backbone, layer_decay, weight_decay):
        groups = self.encoder.param_groups(lr_backbone, layer_decay, weight_decay)
        enc = {id(p) for p in self.encoder.parameters()}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in enc]
        groups.append({"params": [p for p in rest if p.ndim >= 2], "lr": lr_head, "weight_decay": weight_decay})
        groups.append({"params": [p for p in rest if p.ndim < 2], "lr": lr_head, "weight_decay": 0.0})
        return [g for g in groups if g["params"]]

    def encode(self, images, valid, output_attentions=False, pre_offset=None):
        b, k = images.shape[:2]
        flat = images.flatten(0, 1)
        if self.input_phases == "early":
            flat = flat[:, 1:2].expand(-1, 3, -1, -1)
        elif self.input_phases == "enhancement":   # undo the normalisation, compute the maps, redo it
            mean, std = (flat.new_tensor(v).view(1, 3, 1, 1) for v in (self.encoder.mean, self.encoder.std))
            off = (pre_offset if pre_offset is not None else flat.new_zeros(b)).to(flat.dtype)
            flat = (enhancement_maps(flat * std + mean, off.repeat_interleave(k)) - mean) / std
        if self.is_3d:   # the whole block at once: (B, 3, K, H, W) -> one feature vector per patient
            vol = flat.view(b, k, *flat.shape[1:]).transpose(1, 2)
            out = self.encoder(vol)
            return out["cls"][:, None], out, valid.new_ones((b, 1))
        sel = valid.flatten()
        out = self.encoder(flat[sel], output_attentions=output_attentions)   # only real slices
        cls = out["cls"].new_zeros((b * k, out["cls"].shape[-1]))
        cls[sel] = out["cls"]
        return cls.view(b, k, -1), out, valid

    def forward(self, batch, output_attentions=False):
        h, enc_out, valid = self.encode(batch["images"], batch["valid"], output_attentions,
                                        batch.get("pre_offset"))                   # (B, K, D); K = 1 for 3D
        res = {"img_emb": masked_mean(h, valid)}
        if output_attentions:
            res["vit_attentions"] = enc_out["attentions"]
        if self.agg == "mil":
            pooled, res["mil_weights"] = self.mil(h, valid)
        else:
            pooled = res["img_emb"]

        if self.kind == "image_only":
            if self.agg == "slice_attn":
                pos = batch["slice_pos"][:, :h.shape[1]] if not self.is_3d else h.new_zeros(h.shape[:2])
                x = self.sa_proj(h) + self.sa_pos(pos.unsqueeze(-1).to(h.dtype))
                b = x.shape[0]
                x = torch.cat([self.sa_cls.expand(b, -1, -1).to(x.dtype), x], 1)
                keep = torch.cat([valid.new_ones((b, 1)), valid], 1)
                for blk in self.sa_blocks:
                    x, w = blk(x, ~keep)
                z = self.sa_norm(x[:, 0])
                res["logit"] = self.head(self.drop(z)).squeeze(-1)
                res["slice_attn"] = w[:, 0, 1:]                 # what [CLS] attended to, per slice
                pooled = z
            elif self.agg == "mean_logit":
                sl = self.head(self.drop(h)).squeeze(-1)
                res["slice_logits"] = sl
                res["logit"] = masked_mean(sl.unsqueeze(-1), valid).squeeze(-1)
            else:
                res["logit"] = self.head(self.drop(pooled)).squeeze(-1)
            aux_in = pooled
        elif self.kind == "concat":
            img = self.img_proj(pooled)
            parts = [emb(batch["cat"][:, j]) for j, emb in enumerate(self.clin_emb)]
            parts += [batch["cont"], batch["cont_missing"]]
            clin = self.clin_mlp(torch.cat(parts, 1))
            res["logit"] = self.head(torch.cat([self.drop(img), clin], 1)).squeeze(-1)
            aux_in = img
        else:
            if self.agg == "tokens":
                pos = batch["slice_pos"] if not self.is_3d else h.new_zeros(h.shape[:2])
                img = self.img_proj(h) + self.slice_pos(pos.unsqueeze(-1).to(h.dtype))
                img_valid = valid
                aux_in = masked_mean(img, valid)
            else:
                img = self.img_proj(pooled).unsqueeze(1)
                img_valid = valid.new_ones((valid.shape[0], 1))
                aux_in = img[:, 0]
            img = img + self.img_type
            clin = self.clin(batch["cat"], batch["cont"], batch["cont_missing"]).to(img.dtype)
            b = img.shape[0]
            x = torch.cat([self.cls.expand(b, -1, -1).to(img.dtype), img, clin], 1)
            keep = torch.cat([valid.new_ones((b, 1)), img_valid, valid.new_ones((b, clin.shape[1]))], 1)
            attn = []
            for blk in self.blocks:
                x, w = blk(x, ~keep)
                attn.append(w)
            res["logit"] = self.head(self.drop(self.norm(x[:, 0]))).squeeze(-1)
            res["attn"] = attn[-1]                      # (B, T, T): row 0 = what [CLS] attended to
            res["token_layout"] = {"n_img": img.shape[1], "n_clin": clin.shape[1]}
        if self.aux:
            res["aux_hr"] = self.aux_hr(aux_in).squeeze(-1)
            res["aux_her2"] = self.aux_her2(aux_in).squeeze(-1)
        return res


def build_model(cfg: dict, encoder) -> PCRModel:
    model = PCRModel(cfg["model"], encoder.cardinalities, len(encoder.cont))
    # the datasets created next (by the training script) build the bags this model needs
    configure_bags(slices="early_25d" if model.input_phases == "early_25d" else "phases",
                   sampling="block" if model.is_3d else "spread")
    return model
