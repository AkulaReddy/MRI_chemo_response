"""Slice encoders other than the ViTs in `vit_encoder.py`, and the factory that picks one.

`model.backbone` in the config selects the encoder:
- a ViT name from `vit_encoder.BACKBONES` (MAE, supervised ViT, DINOv2) -> `ViTEncoder`;
- `resnet18` -> `ResNetEncoder` (torchvision, ImageNet-1k weights). CNNs have locality built in and often
  do better than ViTs on small medical datasets.

Every encoder has the same interface: `dim`, `mean`, `std`, `param_groups(lr, layer_decay, weight_decay)`
and `forward(x) -> {"cls": (N, dim), "tokens": ...}` on (N, 3, 224, 224) normalised images.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .vit_encoder import IMAGENET_MEAN, IMAGENET_STD, ViTEncoder

RESNETS = {"resnet18": 512}


class ResNetEncoder(nn.Module):
    """torchvision ResNet without its classifier; the global-average-pooled feature is the slice vector.

    `train_blocks` counts stages from the top: the stages are [stem, layer1, layer2, layer3, layer4], so
    4 trains layer1-4 (stem frozen), "all" trains everything, 0 is a linear probe. BatchNorm layers of
    frozen stages stay in eval mode, so their running statistics are not changed either."""

    def __init__(self, name="resnet18", pretrained=True, train_blocks="all"):
        super().__init__()
        import torchvision
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = getattr(torchvision.models, name)(weights=weights)
        self.name, self.dim, self.mean, self.std = name, RESNETS[name], IMAGENET_MEAN, IMAGENET_STD
        self.stages = nn.ModuleList([nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool),
                                     net.layer1, net.layer2, net.layer3, net.layer4])
        self.n_layers = len(self.stages)
        n_train = self.n_layers if train_blocks in ("all", -1) else int(train_blocks)
        self.first_trained = self.n_layers - n_train
        for i, st in enumerate(self.stages):
            for p in st.parameters():
                p.requires_grad = i >= self.first_trained

    def train(self, mode: bool = True):
        super().train(mode)
        for i, st in enumerate(self.stages):
            if i < self.first_trained:
                st.eval()
        return self

    def param_groups(self, lr: float, layer_decay: float, weight_decay: float):
        """Layer-wise LR decay over the stages: layer4 gets `lr`, each stage below x layer_decay."""
        groups = []
        for i, st in enumerate(self.stages):
            params = [p for p in st.parameters() if p.requires_grad]
            scale = layer_decay ** (self.n_layers - 1 - i)
            decay = [p for p in params if p.ndim >= 2]
            no_decay = [p for p in params if p.ndim < 2]
            if decay:
                groups.append({"params": decay, "lr": lr * scale, "weight_decay": weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr * scale, "weight_decay": 0.0})
        return groups

    def forward(self, x: torch.Tensor, output_attentions: bool = False):
        for st in self.stages:
            x = st(x)
        cls = x.mean(dim=(2, 3))                                 # global average pool -> (N, 512)
        return {"cls": cls, "tokens": cls[:, None]}


def build_encoder(m: dict) -> nn.Module:
    """The slice encoder named by `model.backbone`."""
    name = m["backbone"]
    if name in RESNETS:
        if m.get("drop_path"):
            raise ValueError("drop_path is not supported for ResNet backbones")
        return ResNetEncoder(name, m.get("pretrained", True), m.get("train_blocks", "all"))
    return ViTEncoder(name, m.get("pretrained", True), m.get("train_blocks", 4), m.get("grad_checkpointing", True),
                      drop_path=float(m.get("drop_path", 0.0)))
