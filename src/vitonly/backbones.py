"""ViT backbones behind one interface: features `cls` and `gap` (mean of patch tokens), partial freezing,
layer-wise LR decay. MAE and supervised ViT reuse src/models/vit_encoder.py; DINO / DINOv2 are added here.

Why `gap`: MAE pre-training never trains the [CLS] token (its loss is pixel reconstruction of patches), and
MAE's own fine-tuning recipe pools the patch tokens instead. B3 used [CLS] from MAE.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src import hide_broken_torchaudio

hide_broken_torchaudio()
from src.models.vit_encoder import IMAGENET_MEAN, IMAGENET_STD, TINY, load_backbone  # noqa: E402

SPECS = {
    "mae":        dict(hf="facebook/vit-mae-base", kind="mae", mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "vit_sup":    dict(hf="google/vit-base-patch16-224", kind="vit", mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    "dino_b16":   dict(hf="facebook/dino-vitb16", kind="vit", mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "dinov2_s":   dict(hf="facebook/dinov2-small", kind="dinov2", mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "dinov2_b":   dict(hf="facebook/dinov2-base", kind="dinov2", mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "tiny-random": dict(hf="tiny-random", kind="mae", mean=IMAGENET_MEAN, std=IMAGENET_STD),     # tests
    "tiny-dinov2": dict(hf="tiny-dinov2", kind="dinov2", mean=IMAGENET_MEAN, std=IMAGENET_STD),  # tests
}


def _load(name: str):
    from transformers import Dinov2Config, Dinov2Model, ViTModel
    s = SPECS[name]
    if name == "tiny-dinov2":
        return Dinov2Model(Dinov2Config(**TINY, image_size=224, patch_size=14))
    if s["kind"] == "mae":
        return load_backbone(s["hf"])               # MAE weights copied into a plain ViTModel
    if s["kind"] == "dinov2":
        return Dinov2Model.from_pretrained(s["hf"])
    return ViTModel.from_pretrained(s["hf"], add_pooling_layer=False)


def _blocks(m):
    for path in ("encoder.layer", "layers", "encoder.layers"):   # transformers 4.x / 5.x layouts
        obj = m
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise AttributeError(f"no transformer blocks found in {type(m).__name__}")


class Backbone(nn.Module):
    def __init__(self, name="mae", train_blocks=4, grad_checkpointing=True):
        super().__init__()
        if name not in SPECS:
            raise ValueError(f"unknown backbone {name}; choose from {list(SPECS)}")
        self.name = name
        self.model = _load(name)
        self.mean, self.std = SPECS[name]["mean"], SPECS[name]["std"]
        self.dim = self.model.config.hidden_size
        self.blocks = _blocks(self.model)
        self.n_layers = len(self.blocks)
        n_train = self.n_layers if train_blocks in ("all", -1) else int(train_blocks)
        self.first_trained = self.n_layers - n_train
        for p in self.model.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.blocks):
            for p in layer.parameters():
                p.requires_grad = i >= self.first_trained
        for p in self.model.layernorm.parameters():
            p.requires_grad = n_train > 0
        if n_train == self.n_layers:
            for p in self.model.embeddings.parameters():
                p.requires_grad = True
        if grad_checkpointing and n_train > 0:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def param_groups(self, lr: float, layer_decay: float, weight_decay: float):
        groups = []

        def add(params, scale):
            params = [p for p in params if p.requires_grad]
            for ps, wd in (([p for p in params if p.ndim >= 2], weight_decay), ([p for p in params if p.ndim < 2], 0.0)):
                if ps:
                    groups.append({"params": ps, "lr": lr * scale, "weight_decay": wd})

        add(self.model.embeddings.parameters(), layer_decay ** self.n_layers)
        for i, layer in enumerate(self.blocks):
            add(layer.parameters(), layer_decay ** (self.n_layers - 1 - i))
        add(self.model.layernorm.parameters(), 1.0)
        return groups

    def forward(self, x: torch.Tensor):
        """x (N, 3, 224, 224) normalised -> {'cls': (N, D), 'gap': (N, D)}."""
        h = self.model(pixel_values=x).last_hidden_state          # layer-normed by both model classes
        return {"cls": h[:, 0], "gap": h[:, 1:].mean(1)}
