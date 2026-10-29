"""Slice encoders other than the ViTs in `vit_encoder.py`, and the factory that picks one.

`model.backbone` in the config selects the encoder:
- a ViT name from `vit_encoder.BACKBONES` (MAE, supervised ViT, DINOv2) -> `ViTEncoder`;
- `resnet18` -> `ResNetEncoder` (torchvision, ImageNet-1k weights). CNNs have locality built in and often
  do better than ViTs on small medical datasets.
- `r3d_18` -> `R3DEncoder`: a **3D** ResNet-18 (torchvision, pretrained on Kinetics-400 video). It sees a
  block of contiguous slices as one volume, so it learns 3D shape and how the tumour changes across slices.

Every encoder has the same interface: `dim`, `mean`, `std`, `param_groups(lr, layer_decay, weight_decay)`
and `forward(x) -> {"cls": (N, dim), "tokens": ...}`. 2D encoders take (N, 3, 224, 224) normalised slices;
3D encoders (`is_3d = True`) take (B, 3, K, 224, 224) normalised blocks of K slices.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_encoder import IMAGENET_MEAN, IMAGENET_STD, ViTEncoder

RESNETS = {"resnet18": 512}
RESNETS_3D = {"r3d_18": 512}
KINETICS_MEAN, KINETICS_STD = (0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)


class _StagedEncoder(nn.Module):
    """Shared by the CNN encoders: `self.stages` from input to output; the top `train_blocks` stages are
    trained, frozen stages keep BatchNorm in eval mode, layer-wise LR decay over the stages."""
    is_3d = False

    def _freeze(self, train_blocks):
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
        """Layer-wise LR decay over the stages: the last stage gets `lr`, each stage below x layer_decay."""
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


class ResNetEncoder(_StagedEncoder):
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
        self._freeze(train_blocks)

    def forward(self, x: torch.Tensor, output_attentions: bool = False):
        for st in self.stages:
            x = st(x)
        cls = x.mean(dim=(2, 3))                                 # global average pool -> (N, 512)
        return {"cls": cls, "tokens": cls[:, None]}


class R3DEncoder(_StagedEncoder):
    """torchvision 3D ResNet-18 (Kinetics-400 video weights) without its classifier.

    Input (B, 3, K, 224, 224): channels = DCE phases (or enhancement maps), K contiguous slices. The centre
    `crop` x `crop` pixels are kept and resized to `size` x `size` (default 160 px = 112 mm at 0.7 mm,
    resized to 112 px = 1.0 mm per pixel, the input size the network was trained at). Stages: [stem,
    layer1..4]; output = global average over (slices, rows, columns) -> (B, 512)."""
    is_3d = True

    def __init__(self, name="r3d_18", pretrained=True, train_blocks="all", crop=160, size=112):
        super().__init__()
        from torchvision.models.video import R3D_18_Weights, r3d_18
        net = r3d_18(weights=R3D_18_Weights.KINETICS400_V1 if pretrained else None)
        self.name, self.dim, self.mean, self.std = name, RESNETS_3D[name], KINETICS_MEAN, KINETICS_STD
        self.crop, self.size = int(crop), int(size)
        self.stages = nn.ModuleList([net.stem, net.layer1, net.layer2, net.layer3, net.layer4])
        self._freeze(train_blocks)

    def forward(self, x: torch.Tensor, output_attentions: bool = False):
        h, w = x.shape[-2:]
        y0, x0 = (h - self.crop) // 2, (w - self.crop) // 2
        x = x[..., y0:y0 + self.crop, x0:x0 + self.crop]
        if self.size != self.crop:
            x = F.interpolate(x, size=(x.shape[2], self.size, self.size), mode="trilinear", align_corners=False)
        for st in self.stages:
            x = st(x)
        cls = x.mean(dim=(2, 3, 4))
        return {"cls": cls, "tokens": cls[:, None]}


def build_encoder(m: dict) -> nn.Module:
    """The slice (2D) or block (3D) encoder named by `model.backbone`."""
    name = m["backbone"]
    if name in RESNETS_3D:
        if m.get("drop_path"):
            raise ValueError("drop_path is not supported for ResNet backbones")
        return R3DEncoder(name, m.get("pretrained", True), m.get("train_blocks", "all"),
                          m.get("vol_crop", 160), m.get("vol_size", 112))
    if name in RESNETS:
        if m.get("drop_path"):
            raise ValueError("drop_path is not supported for ResNet backbones")
        return ResNetEncoder(name, m.get("pretrained", True), m.get("train_blocks", "all"))
    return ViTEncoder(name, m.get("pretrained", True), m.get("train_blocks", 4), m.get("grad_checkpointing", True),
                      drop_path=float(m.get("drop_path", 0.0)))
