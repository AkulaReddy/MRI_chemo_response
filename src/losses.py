"""pCR loss: 0.5 BCE + 0.5 focal (gamma 2), following Nishizawa; aux HR/HER2 BCE (D-013)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pcr_loss(logit, y, bce_weight=0.5, focal_weight=0.5, focal_gamma=2.0, pos_weight=None):
    logit, y = logit.float(), y.float()
    pw = torch.tensor(pos_weight, device=logit.device) if pos_weight else None
    bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw, reduction="none")
    loss = bce_weight * bce.mean()
    if focal_weight:
        p = torch.sigmoid(logit)
        pt = torch.where(y > 0.5, p, 1 - p)
        plain = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        loss = loss + focal_weight * ((1 - pt) ** focal_gamma * plain).mean()
    return loss


def masked_bce(logit, target):
    """BCE over entries with a known target (target < 0 = missing)."""
    keep = target >= 0
    if not keep.any():
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logit[keep].float(), target[keep].float())


def total_loss(out, batch, cfg_loss):
    loss = pcr_loss(out["logit"], batch["label"], cfg_loss.get("bce_weight", 0.5),
                    cfg_loss.get("focal_weight", 0.5), cfg_loss.get("focal_gamma", 2.0),
                    cfg_loss.get("pos_weight"))
    parts = {"pcr": loss.item()}
    w = cfg_loss.get("aux_weight", 0.0)
    if w and "aux_hr" in out:
        aux = masked_bce(out["aux_hr"], batch["hr"]) + masked_bce(out["aux_her2"], batch["her2"])
        loss = loss + w * aux
        parts["aux"] = aux.item()
    return loss, parts
