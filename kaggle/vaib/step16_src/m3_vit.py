"""Step 16 model: a pretrained ViT fine-tuned on 2D RGB-fused tumour slices, as in the BreastDCEDL paper
(facebook/vit-mae-base, the [CLS] token -> a pCR head, plus HR and HER2 heads as in its Figure 2). Every slice
carries its patient's label; a patient is scored from its slices (mean, min, max, tumour-area-weighted mean)."""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mnj_common as mc

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SCORES = ("mean", "min", "max", "wmean")


def load_backbone(name):
    from transformers import ViTConfig, ViTModel
    if name == "tiny-random":                      # offline smoke tests only
        cfg = ViTConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128,
                        image_size=224, patch_size=16)
        return ViTModel(cfg, add_pooling_layer=False)
    return ViTModel.from_pretrained(name, add_pooling_layer=False)


class SliceNet(nn.Module):
    def __init__(self, backbone, n_out=1, drop=0.1):
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(backbone.config.hidden_size, n_out)

    def forward(self, x):
        return self.head(self.drop(self.backbone(pixel_values=x).last_hidden_state[:, 0]))


def _amp(dev):
    return torch.autocast("cuda", dtype=torch.float16) if dev.type == "cuda" else torch.autocast(
        "cpu", dtype=torch.bfloat16, enabled=False)


def to_input(u8, dev):
    """uint8 (B, H, W, 3) numpy -> ImageNet-normalised float (B, 3, H, W) on `dev`."""
    x = torch.from_numpy(np.ascontiguousarray(u8)).to(dev).permute(0, 3, 1, 2).float().div_(255)
    return (x - MEAN.to(dev)) / STD.to(dev)


def augment2d(x, strength=1.0):
    """Random flips, rotation (+-15 deg), zoom (+-15%), shift, brightness / contrast jitter; on the GPU."""
    B, dev = x.shape[0], x.device
    x = x.clone()
    for d in (2, 3):
        m = torch.rand(B, device=dev) < 0.5
        if m.any():
            x[m] = x[m].flip(d)
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(15 * strength)
    sc = 1 + (torch.rand(B, device=dev) * 2 - 1) * 0.15 * strength
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1] = torch.cos(ang) / sc, -torch.sin(ang) / sc
    theta[:, 1, 0], theta[:, 1, 1] = torch.sin(ang) / sc, torch.cos(ang) / sc
    theta[:, :, 2] = (torch.rand(B, 2, device=dev) * 2 - 1) * 0.08 * strength
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    gain = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.1 * strength
    shift = (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.1 * strength
    return x * gain + shift


@torch.no_grad()
def slice_probs(net, IMG, idx, dev, batch=64, tta=True):
    net.eval()
    out = []
    for i in range(0, len(idx), batch):
        x = to_input(IMG[idx[i:i + batch]], dev)
        with _amp(dev):
            lg = net(x)[:, 0].float()
            if tta:
                lg = (lg + net(x.flip(3))[:, 0].float()) / 2
        out.append(torch.sigmoid(lg).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


def patient_scores(net, IMG, SL, AREA, pats, dev, tta=True):
    """(len(pats), 4): mean, min, max and tumour-area-weighted mean of each patient's slice probabilities."""
    pats = np.asarray(pats)
    idx = np.concatenate([SL[p] for p in pats])
    pr = slice_probs(net, IMG, idx, dev, tta=tta)
    out, k = np.zeros((len(pats), 4), np.float32), 0
    for i, p in enumerate(pats):
        n = len(SL[p])
        v, w = pr[k:k + n], AREA[SL[p]] + 1e-3
        k += n
        out[i] = [v.mean(), v.min(), v.max(), float((v * w).sum() / w.sum())]
    return out


def train_fold(backbone, IMG, SL, AREA, pats, y, hr, her2, evals, multitask=True, train_slices=None, epochs=12,
               patience=3, batch=32, lr_backbone=3e-5, lr_head=1e-3, wd=0.05, seed=0, dev=None, say=print):
    """Fine-tune on patients `pats` (rows of SL) with labels y (and hr / her2 for the extra heads); an inner,
    stratified 15% of them is held out for early stopping on the patient-level AUC (mean score).
    Returns (patient_scores for every array of patients in `evals`, best epoch, best inner AUC, history)."""
    from sklearn.model_selection import StratifiedShuffleSplit
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pats, y = np.asarray(pats), np.asarray(y, int)
    lab = np.c_[y, np.asarray(hr, float), np.asarray(her2, float)].astype(np.float32)
    a, b = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=seed).split(pats, y))
    fit_p, fit_lab, in_p, in_y = pats[a], lab[a], pats[b], y[b]

    net = SliceNet(load_backbone(backbone), 3 if multitask else 1).to(dev)
    head = [p for n, p in net.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": net.backbone.parameters(), "lr": lr_backbone},
                             {"params": head, "lr": lr_head}], weight_decay=wd)
    per_ep = sum(min(len(SL[p]), train_slices or 10 ** 9) for p in fit_p)
    steps = max(1, math.ceil(per_ep / batch))
    total, warm = epochs * steps, steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    pos_w = torch.tensor(float((fit_lab[:, 0] == 0).sum() / max(1, (fit_lab[:, 0] == 1).sum())), device=dev)

    best, best_state, best_ep, bad, hist = -1.0, None, 0, 0, []
    for ep in range(epochs):
        t = time.time()
        imgs, labs = [], []
        for p, l in zip(fit_p, fit_lab):
            s = SL[p]
            if train_slices and len(s) > train_slices:
                w = AREA[s] + 1e-3
                s = rng.choice(s, train_slices, replace=False, p=w / w.sum())
            imgs.append(s)
            labs.append(np.repeat(l[None], len(s), 0))
        imgs, labs = np.concatenate(imgs), np.concatenate(labs)
        perm = rng.permutation(len(imgs))
        net.train()
        tot, nb = 0.0, 0
        for i in range(0, len(perm), batch):
            sel = perm[i:i + batch]
            if len(sel) < 2:
                continue
            with torch.no_grad():
                x = augment2d(to_input(IMG[imgs[sel]], dev))
            tl = torch.as_tensor(labs[sel], device=dev)
            with _amp(dev):
                lg = net(x)
            lg = lg.float()
            loss = F.binary_cross_entropy_with_logits(lg[:, 0], tl[:, 0], pos_weight=pos_w)
            if multitask:
                for j in (1, 2):
                    m = ~torch.isnan(tl[:, j])
                    if m.any():
                        loss = loss + 0.5 * F.binary_cross_entropy_with_logits(lg[m, j], tl[m, j])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
        a_in = mc.auc(in_y, patient_scores(net, IMG, SL, AREA, in_p, dev, tta=False)[:, 0])
        a_in = 0.5 if a_in != a_in else a_in
        hist.append({"epoch": ep, "loss": tot / max(nb, 1), "inner_auc": a_in, "minutes": (time.time() - t) / 60})
        say(f"     epoch {ep + 1}/{epochs}: loss {tot / max(nb, 1):.4f}, inner AUC {a_in:.3f} "
            f"({len(imgs)} slices, {(time.time() - t) / 60:.1f} min)")
        if a_in > best:
            best, best_ep, bad = a_in, ep, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    preds = [patient_scores(net, IMG, SL, AREA, e, dev) for e in evals]
    del net, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return preds, best_ep, best, hist
