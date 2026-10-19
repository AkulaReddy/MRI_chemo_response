"""ViT-B/16 slice encoder (M2, D-010).

Default: MAE self-supervised weights `facebook/vit-mae-base` (the BreastDCEDL initialisation).
They are copied into a plain HF `ViTModel`, because `ViTMAEModel` shuffles and masks patches in
its forward pass. Ablations: supervised `google/vit-base-patch16-224` (ImageNet-21k + 1k), and
`facebook/dinov2-base` (DINOv2 ViT-B/14, self-supervised on LVD-142M; strong frozen features). DINOv2
has the same module layout (embeddings, encoder.layer, layernorm); at 224 px it gives 16 x 16 patches
(its position embeddings are interpolated). A CNN alternative lives in `backbones.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src import hide_broken_torchaudio

hide_broken_torchaudio()   # before transformers is imported below

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
BACKBONES = {
    "facebook/vit-mae-base": dict(kind="mae", mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "google/vit-base-patch16-224": dict(kind="vit", mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    "facebook/dinov2-base": dict(kind="dinov2", mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "tiny-random": dict(kind="mae", mean=IMAGENET_MEAN, std=IMAGENET_STD),   # tests only
    "tiny-dinov2": dict(kind="dinov2", mean=IMAGENET_MEAN, std=IMAGENET_STD),   # tests only
}
TINY = dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=64)


def mae_to_vit(mae) -> "ViTModel":
    """A ViTModel with the encoder weights of a ViTMAEModel (identical layer names)."""
    from transformers import ViTConfig, ViTModel
    c = mae.config
    cfg = ViTConfig(hidden_size=c.hidden_size, num_hidden_layers=c.num_hidden_layers,
                    num_attention_heads=c.num_attention_heads, intermediate_size=c.intermediate_size,
                    hidden_act=c.hidden_act, layer_norm_eps=c.layer_norm_eps, qkv_bias=c.qkv_bias,
                    image_size=c.image_size, patch_size=c.patch_size, num_channels=c.num_channels,
                    hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    vit = ViTModel(cfg, add_pooling_layer=False)
    missing, unexpected = vit.load_state_dict(mae.state_dict(), strict=False)
    missing = [k for k in missing if "mask_token" not in k]
    if missing or unexpected:
        raise RuntimeError(f"MAE -> ViT weight mapping failed: missing {missing}, unexpected {unexpected}")
    return vit


def load_backbone(name: str, pretrained: bool = True):
    from transformers import ViTMAEConfig, ViTMAEModel, ViTModel
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name}; choose from {list(BACKBONES)}")
    if name == "tiny-random":
        return mae_to_vit(ViTMAEModel(ViTMAEConfig(**TINY, image_size=224, patch_size=16)))
    if BACKBONES[name]["kind"] == "dinov2":
        from transformers import Dinov2Config, Dinov2Model
        if name == "tiny-dinov2":
            return Dinov2Model(Dinov2Config(**TINY, patch_size=14, image_size=518))
        return Dinov2Model.from_pretrained(name) if pretrained else Dinov2Model(Dinov2Config())
    if BACKBONES[name]["kind"] == "mae":
        mae = ViTMAEModel.from_pretrained(name) if pretrained else ViTMAEModel(ViTMAEConfig())
        return mae_to_vit(mae)
    return ViTModel.from_pretrained(name, add_pooling_layer=False)


class ViTEncoder(nn.Module):
    def __init__(self, name="facebook/vit-mae-base", pretrained=True, train_blocks=4, grad_checkpointing=True):
        super().__init__()
        self.name = name
        self.vit = load_backbone(name, pretrained)
        self.dim = self.vit.config.hidden_size
        self.n_layers = self.vit.config.num_hidden_layers
        self.mean, self.std = BACKBONES[name]["mean"], BACKBONES[name]["std"]
        # transformers 4.x: vit.encoder.layer; 5.x: vit.layers
        self.blocks = self.vit.layers if hasattr(self.vit, "layers") else self.vit.encoder.layer
        n_train = self.n_layers if train_blocks in ("all", -1) else int(train_blocks)
        self.first_trained = self.n_layers - n_train
        for p in self.vit.embeddings.parameters():
            p.requires_grad = n_train == self.n_layers
        for i, layer in enumerate(self.blocks):
            for p in layer.parameters():
                p.requires_grad = i >= self.first_trained
        if grad_checkpointing and n_train > 0:
            self.vit.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def param_groups(self, lr: float, layer_decay: float, weight_decay: float):
        """Layer-wise LR decay over the trained blocks; the top block and final norm get `lr`."""
        groups = []

        def add(params, scale):
            params = [p for p in params if p.requires_grad]
            decay = [p for p in params if p.ndim >= 2]
            no_decay = [p for p in params if p.ndim < 2]
            if decay:
                groups.append({"params": decay, "lr": lr * scale, "weight_decay": weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr * scale, "weight_decay": 0.0})

        add(self.vit.embeddings.parameters(), layer_decay ** self.n_layers)
        for i, layer in enumerate(self.blocks):
            add(layer.parameters(), layer_decay ** (self.n_layers - 1 - i))
        add(self.vit.layernorm.parameters(), 1.0)
        return groups

    def forward(self, x: torch.Tensor, output_attentions: bool = False):
        """x: (N, 3, 224, 224) normalised -> {'cls': (N, D), 'tokens': (N, 1+P, D)[, 'attentions']}."""
        out = self.vit(pixel_values=x, output_attentions=output_attentions)
        h = out.last_hidden_state
        res = {"cls": h[:, 0], "tokens": h}
        if output_attentions:
            res["attentions"] = out.attentions
        return res
