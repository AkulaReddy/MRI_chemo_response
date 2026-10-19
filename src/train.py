"""Config-driven training / export for B3, F1, F2 and their ablations.

    python -m src.train --config f2_attn --seed 0 --cache-root /kaggle/input --out /kaggle/working/runs
    python -m src.train --config b3_vit --debug                         # 2 epochs, few patients
    python -m src.train --config b3_vit --mode export --checkpoint runs/<run_id>/best.pt

Selection uses val patient-level AUC only; the test split is never loaded here (D-006).
Outputs in <out>/<run_id>/: config_resolved.yaml, encoder.json, log.csv, metrics.json,
val_preds.csv, best.pt and, after export, export_preds.csv / export_slices.csv / export_emb.npz.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

import src

src.hide_broken_torchaudio()   # before transformers is imported
from src import stats
from src.config import config_hash, load_config, set_dotted
from src.data.clinical import ClinicalEncoder
from src.data.datasets import PatientBags, patient_table, worker_init
from src.data.transforms import eval_transform, train_transform
from src.losses import total_loss
from src.models.fusion import build_model

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- setup
def find_caches(roots, explicit=()):
    """{cohort: dir} from `cohort=dir` pairs, else any folder (<= 4 levels deep) with cache_meta.json.
    If one cohort has several caches (v1 and v2 both attached), the highest builder_version wins."""
    found, version = {}, {}
    for r in roots:
        for depth in range(5):
            for f in glob.glob(os.path.join(r, *["*"] * depth, "cache_meta.json")):
                meta = json.load(open(f))
                c, v = meta["cohort"], meta.get("builder_version", 1)
                if c not in found or v > version[c]:
                    found[c], version[c] = os.path.dirname(f), v
    for pair in explicit:
        c, d = pair.split("=", 1)
        found[c] = d
    return found


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_sha():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "nogit"
    except OSError:
        return "nogit"


def to_device(batch, dev):
    return {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def make_loader(ds, cfg, train):
    d = cfg["data"]
    sampler = None
    if train and d.get("sampler") == "balanced":
        y = ds.labels
        w = np.where(y == 1, 0.5 / max((y == 1).sum(), 1), 0.5 / max((y == 0).sum(), 1))
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(ds), replacement=True)
    nw = d.get("num_workers", 4)
    return DataLoader(ds, batch_size=cfg["optim"]["batch_patients"], shuffle=train and sampler is None,
                      sampler=sampler, num_workers=nw, pin_memory=torch.cuda.is_available(),
                      drop_last=train and len(ds) > cfg["optim"]["batch_patients"],
                      worker_init_fn=worker_init if nw else None)


# ----------------------------------------------------------------------------- loops
@torch.no_grad()
def predict(model, loader, dev, amp, want_attn=False):
    model.eval()
    mean, std = model.encoder.mean, model.encoder.std
    keep = {"logit": [], "img_emb": [], "aux_hr": [], "aux_her2": [], "index": [], "slice_logits": [],
            "valid": [], "cls_attn": [], "mil_weights": []}
    for batch in loader:
        batch = to_device(batch, dev)
        batch["images"] = eval_transform(batch["images"], mean, std)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            out = model(batch)
        keep["index"].append(batch["index"].cpu())
        keep["valid"].append(batch["valid"].cpu())
        for k in ("logit", "img_emb", "aux_hr", "aux_her2", "slice_logits", "mil_weights"):
            if k in out:
                keep[k].append(out[k].float().cpu())
        if want_attn and "attn" in out:
            keep["cls_attn"].append(out["attn"][:, 0].float().cpu())
    return {k: torch.cat(v).numpy() for k, v in keep.items() if v}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def evaluate(model, loader, table, dev, amp):
    pr = predict(model, loader, dev, amp)
    t = table.iloc[pr["index"]]
    p = sigmoid(pr["logit"])
    df = pd.DataFrame({"pid": t.pid.values, "cohort": t.cohort.values, "label": t.pCR.values, "prob": p})
    m = stats.summary(df.label, df.prob, ci=False)
    m["by_cohort"] = {c: stats.summary(g.label, g.prob, ci=False) for c, g in df.groupby("cohort")}
    if "aux_hr" in pr:
        for k, col in (("aux_hr", "HR"), ("aux_her2", "HER2")):
            y = t[col].values
            ok = ~np.isnan(y)
            m[f"{k}_auc"] = stats.auc(y[ok], pr[k][ok])
    return df, m, pr


def lr_lambda(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(total - warmup, 1))))


def fit(model, cfg, train_ds, val_ds, val_t, dev, run_dir, log):
    o, amp = cfg["optim"], cfg["optim"].get("amp", True) and dev.type == "cuda"
    tr_loader, va_loader = make_loader(train_ds, cfg, True), make_loader(val_ds, cfg, False)
    groups = model.param_groups(o["lr_head"], o["lr_backbone"], o["layer_decay"], o["weight_decay"])
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups)
    steps = len(tr_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, int(o.get("warmup_epochs", 1) * steps), o["max_epochs"] * steps))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    mean, std = model.encoder.mean, model.encoder.std
    gen = torch.Generator().manual_seed(cfg["seed"])
    best, bad, rows = -1.0, 0, []
    for epoch in range(o["max_epochs"]):
        train_ds.set_epoch(epoch, cfg["seed"])
        model.train()
        t0, n_slices, losses = time.time(), 0, []
        for batch in tr_loader:
            batch = to_device(batch, dev)
            batch["images"] = train_transform(batch["images"], mean, std, cfg.get("aug"), gen)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                out = model(batch)
            loss, parts = total_loss(out, batch, cfg["loss"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           o.get("grad_clip", 1.0))
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            n_slices += int(batch["valid"].sum())
        t_train = time.time() - t0
        _, m, _ = evaluate(model, va_loader, val_t, dev, amp)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auc": m["auc"],
               "val_auprc": m["auprc"], "val_brier": m["brier"], "val_ece": m["ece"],
               **{f"val_auc_{c}": v["auc"] for c, v in m["by_cohort"].items()},
               "lr_head": opt.param_groups[-1]["lr"], "train_min": t_train / 60,
               "slices_per_s": n_slices / max(t_train, 1e-6), "epoch_min": (time.time() - t0) / 60}
        for k in ("aux_hr_auc", "aux_her2_auc"):
            if k in m:
                row[f"val_{k}"] = m[k]
        rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "log.csv", index=False)
        score = m["auc"] if not np.isnan(m["auc"]) else -row["train_loss"]
        improved = score > best
        if improved:
            best, bad = score, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_auc": m["auc"]}, run_dir / "best.pt")
        else:
            bad += 1
        log(f"epoch {epoch:2d} loss {row['train_loss']:.4f} val AUC {m['auc']:.3f} AUPRC {m['auprc']:.3f} "
            f"| {row['slices_per_s']:.0f} slices/s, {row['epoch_min']:.1f} min{'  *best' if improved else ''}")
        if bad >= o.get("patience", 8):
            log(f"early stop: no val AUC gain for {bad} epochs")
            break
    return rows


def export(model, cfg, tables, cache_dirs, enc, dev, amp, run_dir, log):
    """Train/val predictions, slice scores and patient embeddings (for B4, F2-from-B3, explain)."""
    k = cfg.get("export", {}).get("k", 32)
    preds, slices, embs = [], [], {}
    for name, t in tables.items():
        ds = PatientBags(t, cache_dirs, enc, k=k, train=False, margin=cfg["data"].get("margin", 0),
                         min_area_frac=float(cfg["data"].get("min_area_frac", 0.0)))
        loader = make_loader(ds, {**cfg, "optim": {**cfg["optim"], "batch_patients": 4}}, False)
        pr = predict(model, loader, dev, amp, want_attn=True)
        tt = t.reset_index(drop=True).iloc[pr["index"]]
        preds.append(pd.DataFrame({"pid": tt.pid.values, "split": name, "cohort": tt.cohort.values,
                                   "label": tt.pCR.values, "prob": sigmoid(pr["logit"])}))
        embs[name] = (tt.pid.values, pr["img_emb"])
        if "slice_logits" in pr:
            for i, pid in enumerate(tt.pid.values):
                v = pr["valid"][i]
                for j in np.flatnonzero(v):
                    slices.append((pid, name, j, float(sigmoid(pr["slice_logits"][i, j]))))
        if "cls_attn" in pr:
            np.save(run_dir / f"export_cls_attn_{name}.npy", pr["cls_attn"])
    pd.concat(preds).to_csv(run_dir / "export_preds.csv", index=False)
    if slices:
        pd.DataFrame(slices, columns=["pid", "split", "slice", "prob"]).to_csv(run_dir / "export_slices.csv",
                                                                              index=False)
    np.savez(run_dir / "export_emb.npz", **{f"{n}_pid": v[0] for n, v in embs.items()},
             **{f"{n}_emb": v[1] for n, v in embs.items()})
    log(f"exported {sum(len(p) for p in preds)} patients to {run_dir}")


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-root", nargs="*", default=None,
                    help="folders searched for caches (default: /kaggle/input on Kaggle, else data/cache)")
    ap.add_argument("--cache", nargs="*", default=[], help="explicit cohort=dir pairs")
    ap.add_argument("--cache-version", type=int, default=None,
                    help="require this cache builder version for every cohort (e.g. 2); stops otherwise")
    ap.add_argument("--cohorts", nargs="*", default=None, help="restrict to these cohorts (e.g. ispy1)")
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--mode", choices=["train", "export"], default="train")
    ap.add_argument("--checkpoint", default=None, help="for --mode export")
    ap.add_argument("--init-from", default=None, help="checkpoint to initialise from (e.g. f2_from_b3)")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. optim.max_epochs=2")
    ap.add_argument("--debug", action="store_true", help="2 epochs on a few patients")
    ap.add_argument("--results-dir", default=None, help="copy the small outputs here (results/<person>)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    cfg["seed"] = a.seed
    for s in a.set:
        set_dotted(cfg, *s.split("=", 1))
    if a.debug:
        cfg["optim"]["max_epochs"] = min(cfg["optim"]["max_epochs"], 2)
        cfg["data"]["limit"] = cfg["data"].get("limit") or 48
    seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    amp = cfg["optim"].get("amp", True) and dev.type == "cuda"

    roots = a.cache_root or (["/kaggle/input", "/kaggle/working"] if os.path.isdir("/kaggle/input")
                             else [str(ROOT / "data" / "cache")])
    caches = find_caches(roots, a.cache)
    d = cfg["data"]
    cohorts = [c for c in (a.cohorts or d["cohorts"]) if c in caches]
    if not cohorts:
        raise SystemExit(f"no caches found for {a.cohorts or d['cohorts']} under {roots} (found {caches})")
    missing = set(a.cohorts or d["cohorts"]) - set(cohorts)
    if missing:
        print(f"WARNING: no cache for {sorted(missing)}; training on {cohorts} only")
    cache_info = {}
    for c in cohorts:
        m = json.load(open(os.path.join(caches[c], "cache_meta.json")))
        cache_info[c] = {"dir": str(caches[c]), "builder_version": m.get("builder_version", 1),
                         "grid_mm": m.get("grid_mm"), "git": m.get("git")}
    print("caches:", {c: f"v{i['builder_version']} {i['dir']}" for c, i in cache_info.items()})
    if a.cache_version is not None:
        wrong = {c: i["builder_version"] for c, i in cache_info.items() if i["builder_version"] != a.cache_version}
        if wrong:
            raise SystemExit(f"--cache-version {a.cache_version} required, but found {wrong}: attach mnj-cache-v2")
    train_t = patient_table(d.get("train_splits", ["train"]), cohorts, cache_dirs=caches)
    val_t = patient_table(["val"], [c for c in d.get("val_cohorts", cohorts) if c in caches], cache_dirs=caches)
    if d.get("limit"):
        train_t = train_t.sample(min(d["limit"], len(train_t)), random_state=a.seed)
        val_t = val_t.sample(min(d["limit"], len(val_t)), random_state=a.seed)

    run_id = f"{cfg['name']}_s{a.seed}_{time.strftime('%m%d-%H%M')}_{git_sha()}" + ("_debug" if a.debug else "")
    if a.mode == "export":
        run_dir = Path(a.checkpoint).resolve().parent
    else:
        run_dir = Path(a.out) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / ("train.log" if a.mode == "train" else "export.log"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    enc = ClinicalEncoder(cfg["clinical"]["features"]).fit(train_t)
    model = build_model(cfg, enc).to(dev)
    if a.init_from or cfg.get("init_from"):
        sd = torch.load(a.init_from or cfg["init_from"], map_location="cpu", weights_only=True)["model"]
        if cfg.get("init_encoder_only", True):
            sd = {k: v for k, v in sd.items() if k.startswith("encoder.")}
        missing_k, unexpected = model.load_state_dict(sd, strict=False)
        log(f"init from {a.init_from or cfg['init_from']}: {len(sd)} tensors, "
            f"{len(missing_k)} missing, {len(unexpected)} unexpected")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"run {run_id} | {dev} | cohorts {cohorts} | train {len(train_t)} ({int(train_t.pCR.sum())} pCR) "
        f"| val {len(val_t)} ({int(val_t.pCR.sum())} pCR) | trainable params {n_train / 1e6:.1f}M")

    if a.mode == "train":
        (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (run_dir / "encoder.json").write_text(json.dumps(enc.to_dict(), indent=2))
        k_tr, k_ev, margin = d.get("k_train", 8), d.get("k_eval", 16), d.get("margin", 0)
        maf = float(d.get("min_area_frac", 0.0))
        train_ds = PatientBags(train_t, caches, enc, k=k_tr, train=True, margin=margin,
                               preload=d.get("preload", False), seed=a.seed, min_area_frac=maf)
        val_ds = PatientBags(val_t, caches, enc, k=k_ev, train=False, margin=margin, preload=d.get("preload", False),
                             min_area_frac=maf)
        t0 = time.time()
        rows = fit(model, cfg, train_ds, val_ds, val_t, dev, run_dir, log)
        hours = (time.time() - t0) / 3600
        ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        val_df, m, _ = evaluate(model, make_loader(val_ds, cfg, False), val_t, dev, amp)
        val_df.to_csv(run_dir / "val_preds.csv", index=False)
        m["auc_ci"] = stats.bootstrap_auc_ci(val_df.label, val_df.prob)
        metrics = {"run_id": run_id, "config": cfg["name"], "config_hash": config_hash(cfg), "seed": a.seed,
                   "git": git_sha(), "cohorts": cohorts, "best_epoch": ck["epoch"], "epochs_run": len(rows),
                   "gpu_hours": round(hours, 3), "device": torch.cuda.get_device_name(0) if dev.type == "cuda"
                   else "cpu", "n_train": len(train_t), "n_val": len(val_t), "caches": cache_info, "val": m}
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
        log(f"best epoch {ck['epoch']}: val AUC {m['auc']:.3f} (95% CI {m['auc_ci'][0]:.3f}-{m['auc_ci'][1]:.3f}),"
            f" AUPRC {m['auprc']:.3f}, ECE {m['ece']:.3f}, {hours:.2f} GPU-h")
        if cfg.get("export", {}).get("after_train"):
            export(model, cfg, {"train": train_t, "val": val_t}, caches, enc, dev, amp, run_dir, log)
    else:
        if not a.checkpoint:
            raise SystemExit("--mode export needs --checkpoint runs/<run_id>/best.pt")
        saved = json.loads((run_dir / "encoder.json").read_text())
        enc = ClinicalEncoder.from_dict(saved)
        model.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=True)["model"])
        export(model, cfg, {"train": train_t, "val": val_t}, caches, enc, dev, amp, run_dir, log)

    if a.results_dir:
        dst = Path(a.results_dir) / run_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("config_resolved.yaml", "encoder.json", "log.csv", "metrics.json", "val_preds.csv",
                  "train.log", "export.log", "export_preds.csv"):
            if (run_dir / f).exists():
                shutil.copy(run_dir / f, dst / f)
        log(f"copied summary files to {dst}")
    return run_dir


if __name__ == "__main__":
    main()
