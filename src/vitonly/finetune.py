"""Fine-tune the improved ViT image-only model (configs/vo_*.yaml).

    python -m src.vitonly.finetune --config vo_base --seed 0 --set data.norm=kinetic model.backbone=dinov2_b
    python -m src.vitonly.finetune --config vo_b3_repro --debug      # the old B3 recipe through this code

What differs from B3 (src/train.py with b3_vit.yaml), each switchable in the config:
  data.norm / data.crop   normalisation and tumour-centred zoom (src/vitonly/norms.py); B3 = joint_pct / fixed
  data.slice_mode         central 60% of the tumour slices (`mid`); B3 used the whole range incl. edge slivers
  model.pool              patch-token mean (`gap`); B3 used MAE's [CLS], which MAE pre-training never trains
  model.backbone          mae | vit_sup | dino_b16 | dinov2_s | dinov2_b
  optim.freeze_epochs     linear-probe warm-up (LP-FT): the head trains on frozen features first, then the
                          top blocks unfreeze with their own warm-up, so a random head can't wreck them
  loss                    plain BCE by default (B3: 0.5 BCE + 0.5 focal)
Selection uses val patient-level AUC (the test split is never loaded). Outputs in <out>/<run_id>/ as in
src/train.py: config_resolved.yaml, log.csv, metrics.json, val_preds.csv, train.log, best.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import src

src.hide_broken_torchaudio()
from src import stats  # noqa: E402
from src.config import config_hash, load_config, set_dotted  # noqa: E402
from src.data.metadata import patient_table  # noqa: E402
from src.data.transforms import train_transform  # noqa: E402
from src.losses import pcr_loss  # noqa: E402
from src.train import evaluate, git_sha, lr_lambda, make_loader, seed_all, to_device  # noqa: E402

from .data import SlabBags  # noqa: E402
from .model import VitOnly  # noqa: E402
from .slabs import find_caches  # noqa: E402


def datasets(cfg, tables, caches, seed):
    d = cfg["data"]
    kw = dict(norm=d["norm"], crop=d["crop"], slice_mode=d.get("slice_mode", "mid"),
              mid_frac=d.get("mid_frac", 0.6), zoom=d.get("zoom", 1.6), fov_mm=d.get("fov_mm", 100.0))
    tr = SlabBags(tables["train"], caches, k=d.get("k_train", 8), train=True, seed=seed, **kw)
    va = SlabBags(tables["val"], caches, k=d.get("k_eval", 12), train=False, **kw)
    return tr, va


def fit(model, cfg, train_ds, val_ds, val_t, dev, run_dir, log):
    o, amp = cfg["optim"], cfg["optim"].get("amp", True) and dev.type == "cuda"
    tr_loader, va_loader = make_loader(train_ds, cfg, True), make_loader(val_ds, cfg, False)
    groups = model.param_groups(o["lr_head"], o["lr_backbone"], o["layer_decay"], o["weight_decay"])
    enc = {id(p) for p in model.encoder.parameters()}
    opt = torch.optim.AdamW(groups)
    steps, freeze = len(tr_loader), int(o.get("freeze_epochs", 0))
    warm, total = int(o.get("warmup_epochs", 1) * steps), o["max_epochs"] * steps

    def head_lr(s):
        return lr_lambda(s, warm, total)

    def enc_lr(s):   # zero while frozen, then its own warm-up and the remaining cosine
        return 0.0 if s < freeze * steps else lr_lambda(s - freeze * steps, warm, total - freeze * steps)

    enc_groups = [i for i, g in enumerate(groups) if any(id(p) in enc for p in g["params"])]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, [enc_lr if i in enc_groups else head_lr for i in range(len(groups))])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    mean, std = model.encoder.mean, model.encoder.std
    gen = torch.Generator().manual_seed(cfg["seed"])
    best, bad, rows = -1.0, 0, []
    for epoch in range(o["max_epochs"]):
        model.freeze_backbone = epoch < freeze
        train_ds.set_epoch(epoch, cfg["seed"])
        model.train()
        lr_enc = max((opt.param_groups[i]["lr"] for i in enc_groups), default=0.0)   # used this epoch
        t0, n_slices, losses = time.time(), 0, []
        for batch in tr_loader:
            batch = to_device(batch, dev)
            batch["images"] = train_transform(batch["images"], mean, std, cfg.get("aug"), gen)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                out = model(batch)
            L = cfg["loss"]
            loss = pcr_loss(out["logit"], batch["label"], L.get("bce_weight", 1.0), L.get("focal_weight", 0.0),
                            L.get("focal_gamma", 2.0), L.get("pos_weight"))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], o.get("grad_clip", 1.0))
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            n_slices += int(batch["valid"].sum())
        t_train = time.time() - t0
        _, m, _ = evaluate(model, va_loader, val_t, dev, amp)
        row = {"epoch": epoch, "frozen": epoch < freeze, "train_loss": float(np.mean(losses)), "val_auc": m["auc"],
               "val_auprc": m["auprc"], "val_brier": m["brier"],
               **{f"val_auc_{c}": v["auc"] for c, v in m["by_cohort"].items()},
               "lr_head": opt.param_groups[-1]["lr"],
               "lr_enc_top": lr_enc,
               "slices_per_s": n_slices / max(t_train, 1e-6), "epoch_min": (time.time() - t0) / 60}
        rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "log.csv", index=False)
        score = m["auc"] if not np.isnan(m["auc"]) else -row["train_loss"]
        improved = score > best
        if improved:
            best, bad = score, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_auc": m["auc"]}, run_dir / "best.pt")
        else:
            bad += 1
        log(f"epoch {epoch:2d}{' (frozen)' if epoch < freeze else ''} loss {row['train_loss']:.4f} "
            f"val AUC {m['auc']:.3f} AUPRC {m['auprc']:.3f} | "
            + " ".join(f"{c} {v['auc']:.2f}" for c, v in m["by_cohort"].items())
            + f" | {row['slices_per_s']:.0f} slices/s, {row['epoch_min']:.1f} min{'  *best' if improved else ''}")
        if bad >= o.get("patience", 6) and epoch >= freeze:
            log(f"early stop: no val AUC gain for {bad} epochs")
            break
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="vo_base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-root", nargs="*", default=None)
    ap.add_argument("--cohorts", nargs="*", default=None)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. data.norm=kinetic model.backbone=dinov2_b")
    ap.add_argument("--debug", action="store_true", help="2 epochs on a few patients")
    ap.add_argument("--results-dir", default=None, help="copy the small outputs here (results/<person>)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    cfg["seed"] = a.seed
    for s in a.set:
        set_dotted(cfg, *s.split("=", 1))
    for sec in ("optim", "loss"):   # YAML 1.1 reads `1e-5` (no decimal point) as a string
        for k, v in cfg[sec].items():
            if isinstance(v, str):
                try:
                    cfg[sec][k] = float(v)
                except ValueError:
                    pass
    if a.debug:
        cfg["optim"]["max_epochs"] = min(cfg["optim"]["max_epochs"], 2)
        cfg["optim"]["freeze_epochs"] = min(cfg["optim"].get("freeze_epochs", 0), 1)
        cfg["data"]["limit"] = cfg["data"].get("limit") or 48
    seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    amp = cfg["optim"].get("amp", True) and dev.type == "cuda"

    roots = a.cache_root or (["/kaggle/input", "/kaggle/working"] if os.path.isdir("/kaggle/input") else ["data/cache"])
    d = cfg["data"]
    caches = find_caches(roots, d.get("prefer_cache", "v2"))
    cohorts = [c for c in (a.cohorts or d["cohorts"]) if c in caches]
    if not cohorts:
        raise SystemExit(f"no caches found for {a.cohorts or d['cohorts']} under {roots}")
    fmts = {c: json.loads((Path(caches[c]) / "cache_meta.json").read_text()).get("format", "v1") for c in cohorts}
    tables = {"train": patient_table(["train"], cohorts, cache_dirs=caches),
              "val": patient_table(["val"], cohorts, cache_dirs=caches)}
    if d.get("limit"):
        tables = {k: t.sample(min(d["limit"], len(t)), random_state=a.seed) for k, t in tables.items()}

    tag = f"{cfg['name']}_{cfg['model']['backbone']}_{d['norm']}_{d['crop']}"
    run_id = f"{tag}_s{a.seed}_{time.strftime('%m%d-%H%M')}_{git_sha()}" + ("_debug" if a.debug else "")
    run_dir = Path(a.out) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / "train.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    model = VitOnly(cfg["model"]).to(dev)
    tr, va = tables["train"], tables["val"]
    log(f"run {run_id} | {dev} | caches {fmts} | norm {d['norm']} crop {d['crop']} slices {d.get('slice_mode')} | "
        f"train {len(tr)} ({int(tr.pCR.sum())} pCR) | val {len(va)} ({int(va.pCR.sum())} pCR) | "
        f"trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    train_ds, val_ds = datasets(cfg, tables, caches, a.seed)
    t0 = time.time()
    rows = fit(model, cfg, train_ds, val_ds, va, dev, run_dir, log)
    hours = (time.time() - t0) / 3600
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    val_df, m, _ = evaluate(model, make_loader(val_ds, cfg, False), va, dev, amp)
    val_df.to_csv(run_dir / "val_preds.csv", index=False)
    m["auc_ci"] = stats.bootstrap_auc_ci(val_df.label, val_df.prob)
    m["auc_within"] = float(np.nanmean([v["auc"] for v in m["by_cohort"].values()]))
    metrics = {"run_id": run_id, "config": cfg["name"], "config_hash": config_hash(cfg), "seed": a.seed,
               "git": git_sha(), "cohorts": cohorts, "cache_formats": fmts, "norm": d["norm"], "crop": d["crop"],
               "backbone": cfg["model"]["backbone"], "best_epoch": ck["epoch"], "epochs_run": len(rows),
               "gpu_hours": round(hours, 3), "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
               "n_train": len(tr), "n_val": len(va), "val": m}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    log(f"best epoch {ck['epoch']}: val AUC {m['auc']:.3f} (95% CI {m['auc_ci'][0]:.3f}-{m['auc_ci'][1]:.3f}), "
        f"within-cohort {m['auc_within']:.3f}, AUPRC {m['auprc']:.3f}, {hours:.2f} GPU-h")
    if a.results_dir:
        dst = Path(a.results_dir) / run_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("config_resolved.yaml", "log.csv", "metrics.json", "val_preds.csv", "train.log"):
            if (run_dir / f).exists():
                shutil.copy(run_dir / f, dst / f)
        log(f"copied summary files to {dst}")
    return run_dir


if __name__ == "__main__":
    main()
