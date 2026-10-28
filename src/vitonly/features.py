"""Step 9 feature track: 2D / 3D / temporal tumour features and frozen ViT embeddings, each with and without
ComBat and with and without subtype, scored with an L2 logistic regression (professor's suggestions, 2026-10-28).

    python -m src.vitonly.features --out runs/features                         # hand-crafted, both refs
    python -m src.vitonly.features --out runs/feat_deep --only-embeddings \\
        --embeddings runs/probe/features_dinov2_b.npz                           # step 10: ComBat on ViT features

For every configuration (ref x feature set x ComBat x subtype):
  cv_auc     5-fold stratified CV on the train split (C and the ComBat fit are inside each fold): use it to RANK
  val_auc    refit on all train, scored on val (177 patients, 95% CI about +-0.08): a confirmation
  vs_lookup  paired bootstrap of val AUC minus the subtype lookup table (each subtype -> its train pCR rate, 0.662)
Feature sets: kinetic16 (the old 16 kinetic numbers), 2d, 3d, temporal and their combinations (radiomics.py).
ref: volume or slice intensity reference (the slice one is the professor's per-slice normalisation).
ComBat (combat.py): batch = cohort (or protocol), covariates HR + HER2, fitted on the training patients only.
Outputs in --out: features_<ref>.csv, extraction_errors.csv, results.csv, val_preds.csv,
harmonisation_check.csv, RESULTS.md (the tables to paste into the results document).
The test split is never loaded (patient_table refuses it).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import src

src.hide_broken_torchaudio()
from src import stats  # noqa: E402
from src.data.metadata import patient_table  # noqa: E402

from . import radiomics  # noqa: E402
from .combat import ComBat  # noqa: E402
from .kinetics import FEATURES as KIN_FEATURES, kinetic_features  # noqa: E402
from .slabs import cache_label, find_caches, load_slab  # noqa: E402

CS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
SETS = {"kinetic16": ("kin_",), "2d": ("2d_",), "3d": ("3d_",), "temporal": ("tmp_",),
        "2d+temporal": ("2d_", "tmp_"), "3d+temporal": ("3d_", "tmp_"), "2d+3d+temporal": ("2d_", "3d_", "tmp_"),
        "all": ("kin_", "2d_", "3d_", "tmp_")}
SUBTYPE = {"HRposHER2neg": "HR+/HER2-", "HER2pos": "HER2+", "TripleNeg": "TN"}
T0 = time.time()


def log(msg):
    print(f"[features {time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:.1f} min] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ extraction
def _one(path, sxy, sz, ref):
    try:
        slab = load_slab(path)
        f = radiomics.extract(slab, sxy, sz, ref)
        f.update({f"kin_{k}": float(v) for k, v in zip(KIN_FEATURES, kinetic_features(slab, sxy, sz))})
        return f, None
    except Exception as e:                                    # one bad slab must not stop the whole run
        return None, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=2).splitlines()[-1]}"


def extract_table(table, caches, ref, jobs=4, chunk=100):
    """One row per patient with every hand-crafted feature; errors are printed and returned, never hidden."""
    from joblib import Parallel, delayed
    rows, errors = [], []
    t = time.time()
    for s in range(0, len(table), chunk):
        part = table.iloc[s:s + chunk]
        res = Parallel(n_jobs=jobs)(delayed(_one)(Path(caches[r.cohort]) / f"{r.pid}.npz", r.xy_spacing,
                                                  r.slice_thick, ref) for r in part.itertuples())
        for r, (f, err) in zip(part.itertuples(), res):
            if err:
                errors.append({"pid": r.pid, "cohort": r.cohort, "ref": ref, "error": err})
                log(f"  ERROR {r.pid} ({r.cohort}): {err}")
                f = {}
            rows.append({"pid": r.pid, **f})
        done = min(s + chunk, len(table))
        el = time.time() - t
        log(f"  ref={ref}: {done}/{len(table)} patients, {el / 60:.1f} min, eta {el / done * (len(table) - done) / 60:.1f} min, "
            f"{len(errors)} errors so far")
    return pd.DataFrame(rows), errors


# --------------------------------------------------------------------------------------------------- scoring
def subtype_of(table):
    return table.HR_HER2_STATUS.map(SUBTYPE).fillna("unknown").to_numpy(str)


def batches_of(table, how="cohort", is_train=None, min_n=20):
    """cohort, or protocol = cohort x pixel-size bin x slice-thickness bin (groups with < min_n training
    patients fall back to their cohort, so ComBat never estimates a batch from a handful of patients)."""
    coh = table.cohort.to_numpy(str)
    if how == "cohort":
        return coh
    if how != "protocol":
        raise ValueError(f"unknown batch {how!r}: cohort | protocol")
    xy = np.digitize(table.xy_spacing.to_numpy(float), [0.6, 0.8])
    z = np.digitize(table.slice_thick.to_numpy(float), [1.6, 2.4])
    b = np.array([f"{c}|xy{a}|z{d}" for c, a, d in zip(coh, xy, z)])
    tr = b if is_train is None else b[is_train]
    small = {g for g, n in zip(*np.unique(tr, return_counts=True)) if n < min_n}
    return np.array([c if g in small or g not in set(tr) else g for g, c in zip(b, coh)])


def covariates_of(table):
    """HR, HER2 (0/1; a missing value -> 0.5): kept by ComBat so the subtype biology is not removed."""
    return table[["HR", "HER2"]].astype(float).fillna(0.5).to_numpy()


def _prep(Xtr, Xte, btr, bte, ctr, cte, combat):
    med = np.nanmedian(Xtr, 0)
    med = np.where(np.isnan(med), 0.0, med)
    Xtr, Xte = np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)
    keep = Xtr.std(0) > 1e-9
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    if combat:
        cb = ComBat().fit(Xtr, btr, ctr)
        Xtr, Xte = cb.transform(Xtr, btr, ctr), cb.transform(Xte, bte, cte)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _with_subtype(X, sub):
    return np.hstack([X, np.stack([sub == s for s in ("HR+/HER2-", "HER2+", "TN")], 1).astype(float)])


def fit_predict(Xtr, ytr, Xte, btr, bte, ctr, cte, str_, ste, C, combat, subtype):
    from sklearn.linear_model import LogisticRegression
    if Xtr.shape[1]:
        Xtr, Xte = _prep(Xtr, Xte, btr, bte, ctr, cte, combat)
    if subtype:
        Xtr, Xte = _with_subtype(Xtr, str_), _with_subtype(Xte, ste)
    m = LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def score(X, y, cohort, batch, cov, sub, is_train, combat, subtype, seed=0):
    """CV AUC on train (C and ComBat chosen / fitted inside the folds), then refit on train -> val predictions."""
    from sklearn.model_selection import StratifiedKFold
    tr = is_train
    strata = np.char.add(cohort[tr], y[tr].astype(int).astype(str))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X[tr], strata))
    Xt, yt, bt, ct, st = X[tr], y[tr], batch[tr], cov[tr], sub[tr]
    cv = {}
    for C in CS:
        aucs = [stats.auc(yt[b], fit_predict(Xt[a], yt[a], Xt[b], bt[a], bt[b], ct[a], ct[b], st[a], st[b], C,
                                             combat, subtype)) for a, b in folds]
        cv[C] = (float(np.nanmean(aucs)), float(np.nanstd(aucs)))
    best = max(CS, key=lambda c: -np.inf if np.isnan(cv[c][0]) else cv[c][0])
    p = fit_predict(Xt, yt, X[~tr], bt, batch[~tr], ct, cov[~tr], st, sub[~tr], best, combat, subtype)
    yv, cv_ = y[~tr], cohort[~tr]
    by = {c: stats.auc(yv[cv_ == c], p[cv_ == c]) for c in sorted(set(cv_))}
    lo, hi = stats.bootstrap_auc_ci(yv, p, n=1000)
    return {"cv_auc": cv[best][0], "cv_auc_sd": cv[best][1], "C": best, "val_auc": stats.auc(yv, p),
            "val_ci_lo": lo, "val_ci_hi": hi, **{f"val_auc_{c}": v for c, v in by.items()},
            "val_auc_within": float(np.nanmean(list(by.values())))}, p


def subtype_lookup(y, sub, is_train):
    """Each patient -> the training pCR rate of their subtype (the 3-row table that every model must beat)."""
    rate = pd.Series(y[is_train]).groupby(sub[is_train]).mean()
    return pd.Series(sub).map(rate).fillna(y[is_train].mean()).to_numpy(float)


def harmonisation_check(X, cohort, batch, cov, is_train, seed=0):
    """How well do the features recognise the cohort (macro one-vs-rest AUC, 5-fold CV on train) before and
    after ComBat? 0.5 = the cohort can no longer be recognised; ComBat only aligns each feature's mean and
    variance, so a multivariate classifier usually still finds some cohort signal."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    Xt, ct, bt, cv_ = X[is_train], cohort[is_train], batch[is_train], cov[is_train]
    out = {}
    for combat in (False, True):
        prob = np.zeros((len(Xt), len(set(ct))))
        labels = sorted(set(ct))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xt, ct):
            A, Bm = _prep(Xt[a], Xt[b], bt[a], bt[b], cv_[a], cv_[b], combat)
            m = LogisticRegression(C=0.1, max_iter=5000).fit(A, ct[a])
            prob[b] = m.predict_proba(Bm)[:, [list(m.classes_).index(c) for c in labels]]
        auc = (roc_auc_score(ct == labels[1], prob[:, 1]) if len(labels) == 2      # e.g. I-SPY only
               else roc_auc_score(ct, prob, multi_class="ovr", labels=labels))
        out["after_combat" if combat else "before_combat"] = float(auc)
    return out


def md_table(df, floatfmt=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if v != v else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


# ------------------------------------------------------------------------------------------------------ main
def load_embeddings(paths, table, pool="gap"):
    """{name: (n, d) array aligned to `table` (NaN rows for patients the probe did not see)} from probe npz."""
    out = {}
    for path in paths:
        d = np.load(path, allow_pickle=True)
        bb = Path(path).stem.replace("features_", "")
        pos = {p: i for i, p in enumerate(d["pid"].astype(str))}
        idx = np.array([pos.get(p, -1) for p in table.pid.astype(str)])
        log(f"embeddings {path}: backbone {bb}, {len(d['combos'])} views, {int((idx >= 0).sum())}/{len(table)} "
            f"patients matched")
        for j, combo in enumerate(d["combos"].astype(str)):
            key = f"{j}_{pool}"
            if key not in d.files:
                continue
            E = d[key].astype(np.float32)
            X = np.full((len(table), E.shape[1]), np.nan, np.float32)
            X[idx >= 0] = E[idx[idx >= 0]]
            out[f"deep:{bb}|{combo}|{pool}"] = X
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", nargs="+", default=["volume", "slice"], choices=["volume", "slice"])
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--batch", default="cohort", choices=["cohort", "protocol"], help="ComBat batch variable")
    ap.add_argument("--embeddings", nargs="*", default=[], help="features_<backbone>.npz files from the probe")
    ap.add_argument("--emb-pool", default="gap", choices=["gap", "cls"])
    ap.add_argument("--only-embeddings", action="store_true", help="skip the hand-crafted features")
    ap.add_argument("--features-from", default=None, help="a folder with features_<ref>.csv to reuse")
    ap.add_argument("--cohorts", nargs="*", default=["ispy1", "ispy2", "duke"])
    ap.add_argument("--cache-root", nargs="*", default=None)
    ap.add_argument("--prefer", default="v2", choices=["v1", "v2", "raw", "uint8"])
    ap.add_argument("--out", default="runs/features")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="patients per split (debug)")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log(f"step 1/6: find the caches and the train / val patients | args {vars(a)}")
    roots = a.cache_root or (["/kaggle/input", "/kaggle/working"] if os.path.isdir("/kaggle/input") else ["data/cache"])
    caches = find_caches(roots, a.prefer)
    cohorts = [c for c in a.cohorts if c in caches]
    if not cohorts:
        raise SystemExit(f"no caches for {a.cohorts} under {roots}: attach mnj-cache2-* / mnj-cache-v2 / mnj-cache-*")
    labels = {c: cache_label(caches[c]) for c in cohorts}
    for c in cohorts:
        log(f"  cache {c}: {labels[c]} at {caches[c]}")
    if any(v.startswith("v1") for v in labels.values()):
        log("  NOTE: uint8 cache(s): the brightest enhancement is clipped, so intensity / PE features are "
            "underestimated (raw mnj-cache2-* caches are exact)")
    missing = sorted(set(a.cohorts) - set(cohorts))
    if missing:
        log(f"  WARNING: no cache for {missing}: those cohorts are left out")
    tr = patient_table(["train"], cohorts, cache_dirs=caches)
    va = patient_table(["val"], cohorts, cache_dirs=caches)
    if a.limit:
        tr, va = tr.sample(min(a.limit, len(tr)), random_state=0), va.sample(min(a.limit, len(va)), random_state=0)
    table = pd.concat([tr, va]).reset_index(drop=True)
    is_train = np.r_[np.ones(len(tr), bool), np.zeros(len(va), bool)]
    y, cohort, sub = table.pCR.to_numpy(float), table.cohort.to_numpy(str), subtype_of(table)
    batch, cov = batches_of(table, a.batch, is_train), covariates_of(table)
    log(f"  train {len(tr)} ({int(tr.pCR.sum())} pCR) | val {len(va)} ({int(va.pCR.sum())} pCR) | "
        f"per cohort (train/val): " + ", ".join(f"{c} {int((cohort[is_train] == c).sum())}/{int((cohort[~is_train] == c).sum())}"
                                                 for c in cohorts))
    log(f"  ComBat batches ({a.batch}): " + ", ".join(f"{b} {n}" for b, n in zip(*np.unique(batch[is_train], return_counts=True))))

    log("step 2/6: hand-crafted features (2d, 3d, temporal, kinetic16)" + (" -- skipped" if a.only_embeddings else ""))
    feats, errors = {}, []
    for ref in ([] if a.only_embeddings else a.refs):
        f_csv = Path(a.features_from) / f"features_{ref}.csv" if a.features_from else None
        if f_csv is not None and f_csv.exists():
            df = pd.read_csv(f_csv, dtype={"pid": str}).drop(columns=["cohort", "split", "label", "subtype"],
                                                             errors="ignore")
            log(f"  ref={ref}: reusing {f_csv} ({len(df)} rows, {df.shape[1] - 1} features)")
        else:
            log(f"  ref={ref}: extracting {len(table)} patients with {a.jobs} processes ...")
            df, err = extract_table(table, caches, ref, a.jobs)
            errors += err
        df = table[["pid"]].merge(df, on="pid", how="left")
        meta = pd.DataFrame({"cohort": cohort, "split": np.where(is_train, "train", "val"), "label": y, "subtype": sub})
        pd.concat([df[["pid"]], meta, df.drop(columns="pid")], axis=1).to_csv(out / f"features_{ref}.csv", index=False)
        feats[ref] = df
        nan = df.drop(columns="pid").isna().mean()
        log(f"  ref={ref}: {df.shape[1] - 1} features; patients with no features {int(df.drop(columns='pid').isna().all(1).sum())}; "
            f"features with > 20% missing: {list(nan[nan > 0.2].index[:10])}")
    pd.DataFrame(errors, columns=["pid", "cohort", "ref", "error"]).to_csv(out / "extraction_errors.csv", index=False)
    if errors:
        log(f"  {len(errors)} extraction errors -> extraction_errors.csv (those patients get the train median)")

    log("step 3/6: list the configurations")
    jobs = []
    for ref, df in feats.items():
        for name in a.sets:
            cols = [c for c in df.columns if c.startswith(SETS[name])]
            if cols:
                jobs.append(({"ref": ref, "set": name}, df[cols].to_numpy(float)))
    emb = load_embeddings(a.embeddings, table, a.emb_pool) if a.embeddings else {}
    for name, X in emb.items():
        jobs.append(({"ref": "-", "set": name}, X.astype(float)))
        if "volume" in feats:                                  # deep + the best hand-crafted family
            hc = feats["volume"][[c for c in feats["volume"].columns if c.startswith(SETS["3d+temporal"])]]
            jobs.append(({"ref": "volume", "set": name + " + 3d+temporal"}, np.hstack([X, hc.to_numpy(float)])))
    if not jobs:
        raise SystemExit("nothing to score: no hand-crafted features and no --embeddings")
    grid = [({**k, "combat": cb, "subtype": st}, X) for k, X in jobs for cb in (False, True) for st in (False, True)]
    log(f"  {len(jobs)} feature sets x ComBat (no/yes) x subtype (no/yes) = {len(grid)} configurations, "
        f"plus references (subtype lookup, subtype LR, cohort LR)")

    log("step 4/6: references")
    rows, preds = [], pd.DataFrame({"pid": table.pid[~is_train].values, "cohort": cohort[~is_train],
                                    "label": y[~is_train], "subtype": sub[~is_train]})
    look = subtype_lookup(y, sub, is_train)[~is_train]
    lk = {"val_auc": stats.auc(y[~is_train], look),
          **{f"val_auc_{c}": stats.auc(y[~is_train][cohort[~is_train] == c], look[cohort[~is_train] == c]) for c in cohorts}}
    lk["val_auc_within"] = float(np.nanmean([lk[f"val_auc_{c}"] for c in cohorts]))
    lk["val_ci_lo"], lk["val_ci_hi"] = stats.bootstrap_auc_ci(y[~is_train], look, n=1000)
    rows.append({"ref": "-", "set": "subtype_lookup", "combat": False, "subtype": True, "n_features": 0, **lk})
    preds["subtype_lookup"] = look
    log(f"  subtype lookup: val AUC {lk['val_auc']:.3f} (" + ", ".join(f"{c} {lk[f'val_auc_{c}']:.3f}" for c in cohorts) + ")")
    empty = np.zeros((len(table), 0))
    onehot = pd.get_dummies(pd.Series(cohort)).to_numpy(float)
    for name, X, st in (("subtype_LR", empty, True), ("cohort_LR", onehot, False), ("cohort+subtype_LR", onehot, True)):
        r, p = score(X, y, cohort, batch, cov, sub, is_train, False, st)
        rows.append({"ref": "-", "set": name, "combat": False, "subtype": st, "n_features": X.shape[1], **r})
        preds[name] = p
        log(f"  {name}: cv AUC {r['cv_auc']:.3f}, val AUC {r['val_auc']:.3f}")

    log(f"step 5/6: score {len(grid)} configurations (each: 5-fold CV x {len(CS)} C values, then refit)")
    from joblib import Parallel, delayed
    t = time.time()
    for s in range(0, len(grid), a.jobs * 2):
        part = grid[s:s + a.jobs * 2]
        res = Parallel(n_jobs=a.jobs)(delayed(score)(X, y, cohort, batch, cov, sub, is_train, k["combat"], k["subtype"])
                                      for k, X in part)
        for i, ((k, X), (r, p)) in enumerate(zip(part, res)):
            col = f"{k['ref']}|{k['set']}|{'combat' if k['combat'] else 'raw'}|{'+subtype' if k['subtype'] else 'image'}"
            preds[col] = p
            d = stats.paired_bootstrap_auc_diff(y[~is_train], look, p, n=1000)
            rows.append({**k, "n_features": X.shape[1], **r, "vs_lookup": d["diff"], "vs_lookup_lo": d["ci"][0],
                         "vs_lookup_hi": d["ci"][1], "vs_lookup_p": d["p"]})
            log(f"  [{s + i + 1}/{len(grid)}] {col}: cv {r['cv_auc']:.3f} | "
                f"val {r['val_auc']:.3f} ({r['val_ci_lo']:.2f}-{r['val_ci_hi']:.2f}) | within-cohort "
                f"{r['val_auc_within']:.3f} | vs lookup {d['diff']:+.3f} (p {d['p']:.2f})")
        el = time.time() - t
        done = min(s + len(part), len(grid))
        log(f"  {done}/{len(grid)} done, {el / 60:.1f} min, eta {el / done * (len(grid) - done) / 60:.1f} min")
    res = pd.DataFrame(rows).sort_values("cv_auc", ascending=False, na_position="last")
    res.to_csv(out / "results.csv", index=False)
    preds.to_csv(out / "val_preds.csv", index=False)

    log("step 6/6: harmonisation check and summary tables")
    checks = []
    for k, X in jobs:
        if k["set"] in ("all", "3d+temporal") or k["set"].startswith("deep:"):
            h = harmonisation_check(X, cohort, batch, cov, is_train)
            checks.append({**k, **h})
            log(f"  cohort recognisable from {k['ref']}|{k['set']}: macro AUC {h['before_combat']:.3f} before ComBat, "
                f"{h['after_combat']:.3f} after (0.5 = not recognisable)")
    pd.DataFrame(checks).to_csv(out / "harmonisation_check.csv", index=False)
    write_report(res, preds, pd.DataFrame(checks), out, labels, len(tr), len(va), a)
    cols = ["ref", "set", "combat", "subtype", "n_features", "cv_auc", "val_auc", "val_ci_lo", "val_ci_hi",
            "val_auc_within"] + [f"val_auc_{c}" for c in cohorts] + ["vs_lookup", "vs_lookup_p"]
    with pd.option_context("display.width", 250, "display.max_rows", 300, "display.max_columns", 30):
        print(res[[c for c in cols if c in res]].round(3).to_string(index=False), flush=True)
    log(f"done -> {out} (results.csv, val_preds.csv, RESULTS.md)")
    return res


def write_report(res, preds, checks, out, labels, n_tr, n_va, a):
    """RESULTS.md: the top configurations and the three questions (ComBat? per-slice? 2D vs 3D vs temporal?)."""
    y = preds.label.to_numpy()
    cohorts = [c for c in ("duke", "ispy1", "ispy2") if f"val_auc_{c}" in res]
    show = ["ref", "set", "combat", "subtype", "n_features", "cv_auc", "val_auc", "val_ci_lo", "val_ci_hi",
            "val_auc_within"] + [f"val_auc_{c}" for c in cohorts] + ["vs_lookup", "vs_lookup_p"]
    lines = [f"# Step 9 results: features, ComBat, per-slice normalisation ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"Train {n_tr} / val {n_va} patients. Caches: " + ", ".join(f"{c} = {v}" for c, v in labels.items())
             + f". ComBat batch = {a.batch}, covariates HR + HER2, fitted on train only.", "",
             f"Rank by `cv_auc` (5-fold CV on the {n_tr} train patients); `val_auc` has a 95% CI of about +-0.08. "
             "`vs_lookup` = paired val AUC difference to the subtype lookup table (counts only if p < 0.05).", "",
             "## Top 15 by CV AUC", "", md_table(res[[c for c in show if c in res]].head(15)), ""]

    def paired(key_a, key_b):
        if key_a not in preds or key_b not in preds:
            return None
        d = stats.paired_bootstrap_auc_diff(y, preds[key_a], preds[key_b], n=1000)
        return d["diff"], d["ci"][0], d["ci"][1], d["p"]

    def effect(title, pairs, note):
        rows = []
        for label, ka, kb in pairs:
            r = paired(ka, kb)
            if r:
                rows.append({"comparison": label, "val AUC a": stats.auc(y, preds[ka]), "val AUC b": stats.auc(y, preds[kb]),
                             "b - a": r[0], "ci_lo": r[1], "ci_hi": r[2], "p": r[3]})
        if rows:
            lines.extend([f"## {title}", "", note, "", md_table(pd.DataFrame(rows)), ""])

    hc = [s for s in SETS if any(f"|{s}|" in c for c in preds)]
    refs = [r for r in ("volume", "slice") if any(c.startswith(r + "|") for c in preds)]
    effect("Does ComBat help? (a = no ComBat, b = ComBat)",
           [(f"{r}|{s}|{m}", f"{r}|{s}|raw|{m}", f"{r}|{s}|combat|{m}") for r in refs for s in hc
            for m in ("image", "+subtype")],
           "Same features, same patients; only the harmonisation differs.")
    effect("Does per-slice normalisation help? (a = volume reference, b = per-slice reference)",
           [(f"{s}|{c}|{m}", f"volume|{s}|{c}|{m}", f"slice|{s}|{c}|{m}") for s in hc if s != "kinetic16"
            for c in ("raw", "combat") for m in ("image", "+subtype")] if len(refs) == 2 else [],
           "Only intensity features change (PE, SER and shape do not depend on the reference).")
    base = "volume" if "volume" in refs else (refs[0] if refs else None)
    if base:
        effect("2D vs 3D vs temporal (a = 2d, b = the other family; ComBat, image only)",
               [(f"2d -> {s}", f"{base}|2d|combat|image", f"{base}|{s}|combat|image")
                for s in ("3d", "temporal", "3d+temporal", "2d+3d+temporal", "kinetic16")],
               f"ref = {base}. Positive = the other family beats the single largest slice.")
        effect("Do the image features add to subtype? (a = subtype lookup, b = features + subtype, ComBat)",
               [(s, "subtype_lookup", f"{base}|{s}|combat|+subtype") for s in hc],
               "The question of results section 6, now with trained models on all cohorts.")
    deep = sorted({c.split("|combat|")[0] for c in preds if c.startswith("-|deep:") and "|combat|" in c})
    if deep:
        effect("ViT embeddings: ComBat and subtype (a = raw image-only, b = ComBat + subtype)",
               [(d, f"{d}|raw|image", f"{d}|combat|+subtype") for d in deep], "Frozen backbone features from the probe.")
    if len(checks):
        lines.extend(["## Harmonisation check", "", "Macro one-vs-rest AUC of recognising the cohort from the features "
                      "(5-fold CV on train). Lower after ComBat = the scanner / site signal was reduced.", "",
                      md_table(checks), ""])
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
