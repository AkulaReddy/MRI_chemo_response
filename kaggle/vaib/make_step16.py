"""Build step16.ipynb: setup / helpers / shared code reused from step15.ipynb, the step-16 modules from step16_src/."""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
S15 = json.load(open(f"{HERE}/step15.ipynb"))
SRC = lambda c: "".join(S15["cells"][c]["source"])  # noqa: E731


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


def module(name):
    return code(f"%%writefile /tmp/mnjlib/{name}.py\n" + open(f"{HERE}/step16_src/{name}.py").read())


INTRO = r"""
# Step 16: the BreastDCEDL paper's model, reproduced as published and with our fixes (GPU T4, ~2-3 h)
The BreastDCEDL paper (Fridman et al., arXiv 2506.12190) reports **AUC 0.94 / accuracy 0.93**. That number is one
subgroup: I-SPY2 HR+/HER2- test patients with clinical features, **N = 40 with 5 pCR** (their Figure 4C: 35 true
negatives, 2 true positives, 3 missed). Always predicting "no pCR" already scores 87.5% accuracy there, and an AUC from
5 positives has a 95% CI roughly 0.8-1.0. Their whole test set (177 patients, all cohorts) scored **AUC 0.72**,
Duke 0.54. Their I-SPY2 model also used the MammaPrint score and the I-SPY2 drug arms, which are **not in the
released metadata** (checked: the public CSV has the same 29 columns as ours).

This notebook does what they did, on our data, under our evaluation rules, so the numbers can be compared fairly:

| | paper (as published) | fixed (this project's lessons) |
|---|---|---|
| input | RGB fusion: pre -> R, early -> G, late -> B | same |
| slices | 4 around the tumour's middle slice | every tumour slice (up to 16); 4 random, area-weighted, per epoch |
| crop | 224 pixels at the native pixel size (70-315 mm) | the same 100 mm for every patient, resized to 224 |
| intensity | min-max per image | one window per patient (keeps enhancement + slice differences) |
| network | `facebook/vit-mae-base` fine-tuned, [CLS] -> pCR + HR + HER2 heads | same backbone, pCR head only (step 12: extra heads hurt) |
| patient score | mean / min of slice probabilities | + max, tumour-area-weighted mean |
| stage 2 | Random Forest: DL scores + tumour volume + box volume + age, HR, HER2, race | RF **and** logistic regression; same features |

Evaluation (unchanged project rules): the ViT is trained in **5-fold CV on the 1,098 training patients**, so every
training patient gets a score from a model that never saw it (no leakage into stage 2); val (177) = mean of the 5
fold models. Stage 2: 5 x 5-fold CV. **The test split is never loaded.** Results are also printed in the
**paper's Table 2 layout** (accuracy, AUC with 95% CI, sensitivity, specificity, precision, NPV per subtype and
cohort) next to the accuracy of always predicting "no pCR", plus an **I-SPY2-only** model like theirs.

Kaggle setup: Accelerator **GPU T4 x2** (one GPU is used), Internet **ON** (downloads `facebook/vit-mae-base`).
**+ Add Input**:
- the caches, best first: **`mnj-cache2-ispy1/-ispy2/-duke`** (raw), else **`mnj-cache-v2`**, else the
  `mnj-cache-ispy1/-ispy2/-duke` v1 caches (works; the brightest enhancement is clipped);
- the code: the **`mnj-code-vaib`** Dataset (the same version as for step 15);
- optional: **your step-15 notebook output** (its `stack_inputs.csv` adds the self-supervised 3D score to stage 2).

Run with `DEBUG = True` first (25 patients per split and cohort, 2 epochs: ~10 min), then `DEBUG = False` with
**Save Version -> Save & Run All (Commit)**. Every step prints a timestamped `[...]` line.
Results: Output tab, `results/<PERSON>/step16/RESULTS_step16.md`.
"""

PARAMS = r"""
# ==== PARAMETERS (edit these) ====
REPO = "AkulaReddy/MRI_chemo_response"
GIT_TAG = "vaib"           # branch (or tag) to clone
PERSON = ""               # optional: "A", "B" or "C" -> results/<PERSON>/ (empty -> results/vaib/)
STEP = "step16"
CACHE_ORDER = ["raw", "uint8_v2", "uint8_v1"]   # per cohort, the first kind attached that loads
VARIANTS = ["paper", "fixed"]                  # "paper" = as published, "fixed" = with our fixes
BACKBONE = "facebook/vit-mae-base"             # the paper's checkpoint
IMG_SIZE = 224
PAPER_SLICES = 4             # the paper's code: range(mid - 2, mid + 2)
FIX_MAX_SLICES = 16          # fixed: every tumour slice, at most this many (evenly spaced)
FIX_CROP_MM = 100.0          # fixed: field of view of every crop
TRAIN_SLICES = 4             # fixed: tumour slices drawn per patient per epoch (area-weighted)
MULTITASK = {"paper": True, "fixed": False}   # HR + HER2 heads next to pCR (paper, Figure 2)
EPOCHS = 12
PATIENCE = 3                 # early stopping on an inner 15% of each training fold (patient-level AUC)
BATCH = 32
LR_BACKBONE, LR_HEAD, WD = 3e-5, 1e-3, 0.05
VIT_REPEATS = 1              # outer CV repeats for the ViT (x 5 folds)
STAGE2_REPEATS = 5           # repeated 5-fold CV for stage 2
STAGE2_KINDS = ["rf", "lr"]  # rf = the paper's Random Forest, lr = logistic regression
THRESHOLD = 0.5              # for accuracy / sensitivity / specificity in the paper-style tables
USE_STEP15 = True            # add the step-15 self-supervised 3D score when its output is attached
JOBS = 4
SEED = 0
EXTRA_PIP = []
TIME_BUDGET_H = 11.0
DEBUG = False                # True: 25 patients per split and cohort, 2 epochs, 2 CV repeats
PUSH_RESULTS = False
NOTEBOOK_NAME = "step16"
"""

CACHES = r"""
# ==== which cache is used for each cohort: the best attached kind that loads (raw > uint8 v2 > uint8 v1) ====
import importlib
MODS = [importlib.reload(importlib.import_module(m)) for m in ["mnj_common", "m2_eval", "m3_data", "m3_vit", "m3_stage2"]]
mc, m2e, m3d, m3v, m3s = MODS
META = mc.load_meta()
CACHES = mc.find_caches(order=tuple(CACHE_ORDER), meta=META)
say(f"caches chosen: {len(CACHES)} cohorts")
for c in COH:
    print(f"  {c:6s}", f"{CACHES[c]['kind']:9s} {CACHES[c]['dir']}" if c in CACHES else "MISSING -> left out", flush=True)
if not CACHES:
    raise SystemExit("no cache attached: + Add Input -> mnj-cache2-* (raw), mnj-cache-v2, or mnj-cache-ispy1/-ispy2/-duke")
if any(v["kind"] == "uint8_v1" for v in CACHES.values()):
    print("NOTE: a v1 uint8 cache is used (99.5% clip: the brightest enhancement is cut). The paper also used 8-bit "
          "images, so the 'paper' variant is faithful; the 'fixed' variant would gain from mnj-cache-v2 / mnj-cache2-*.",
          flush=True)
"""

SLICES = r"""
import torch
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAB = META[META.labeled & META.split.isin(["train", "val"])].copy()          # never the test split (D-012)
if DEBUG:
    LAB = pd.concat([g.sample(min(len(g), 25), random_state=0) for _, g in LAB.groupby(["split", "cohort"])])
FILES = mc.cache_files(CACHES, list(LAB.pid))
say(f"1. 2D RGB slices for {len(FILES)} labelled train/val patients: paper ({PAPER_SLICES} middle slices, "
    f"{IMG_SIZE} native pixels) and fixed (up to {FIX_MAX_SLICES} tumour slices, {FIX_CROP_MM:.0f} mm field of view)")
from concurrent.futures import ProcessPoolExecutor, as_completed
rows, t = [], time.time()
with ProcessPoolExecutor(JOBS) as ex:
    futs = [ex.submit(m3d.make_slices, p, c, f, META.loc[p].to_dict(), mc.clip_of(CACHES[c]["index"], p), IMG_SIZE,
                      PAPER_SLICES, FIX_MAX_SLICES, FIX_CROP_MM) for p, (c, f) in FILES.items()]
    for i, fu in enumerate(as_completed(futs), 1):
        rows.append(fu.result())
        if i % 200 == 0 or i == len(futs):
            say(f"1. slices {i}/{len(futs)} ({(time.time() - t) / 60:.1f} min)")
OKR = sorted([r for r in rows if r["status"] == "ok"], key=lambda r: r["pid"])
PIDS = [r["pid"] for r in OKR]
ROW = {p: i for i, p in enumerate(PIDS)}

def stack(kind):
    n = [len(r[kind]) for r in OKR]
    off = np.r_[0, np.cumsum(n)]
    sl = [np.arange(off[i], off[i + 1]) for i in range(len(OKR))]
    area = np.concatenate([r["fixed_area"] if kind == "fixed" else np.ones(len(r[kind]), np.float32) for r in OKR])
    return np.concatenate([r[kind] for r in OKR]), sl, area.astype(np.float32)

DATA = {v: stack(v) for v in VARIANTS}
for r in rows:                                   # the stacked arrays are the only copy kept
    r.pop("paper", None), r.pop("fixed", None)
SQC = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in rows])
SQC.to_csv(f"{OUT}/slices_qc.csv", index=False)
print("status:", SQC.status.value_counts().to_dict())
if (SQC.status != "ok").any():
    print(SQC.loc[SQC.status != "ok", ["pid", "cohort", "note"]].head(10).to_string(index=False))
ok = SQC[SQC.status == "ok"]
print(ok.groupby("cohort").agg(patients=("pid", "size"), tumour_slices=("n_tumour_slices", "median"),
                               paper_slices=("n_paper", "mean"), fixed_slices=("n_fixed", "mean"),
                               paper_fov_mm=("paper_fov_mm", "median"), fixed_fov_mm=("fixed_fov_mm", "median"),
                               tumour_saturated=("tumour_saturated", "mean")).round(2).to_string())
for v, (imgs, sl, _) in DATA.items():
    print(f"{v}: {len(imgs)} slices, {imgs.nbytes / 1e9:.2f} GB")
show = [p for c in COH for p in ok[ok.cohort == c].pid.head(2)]
fig, axes = plt.subplots(2, len(show), figsize=(2.3 * len(show), 4.8), squeeze=False)
for j, p in enumerate(show):
    for i, v in enumerate(["paper", "fixed"]):
        if v in DATA:
            imgs, sl, _ = DATA[v]
            s = sl[ROW[p]]
            axes[i][j].imshow(imgs[s[len(s) // 2]])
        axes[i][j].axis("off")
        axes[i][j].set_title(f"{v}: {p}"[:24], fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/slice_examples.png", dpi=110); plt.show()

LAB = LAB[LAB.pid.isin(PIDS)].reset_index(drop=True)
LAB["label"] = LAB.pCR.astype(int)
TR, VA = LAB[LAB.split == "train"].reset_index(drop=True), LAB[LAB.split == "val"].reset_index(drop=True)
Y = TR.label.to_numpy(int)
R_TR, R_VA = np.array([ROW[p] for p in TR.pid]), np.array([ROW[p] for p in VA.pid])
HR_TR, HER2_TR = pd.to_numeric(TR.HR, errors="coerce").to_numpy(float), pd.to_numeric(TR.HER2, errors="coerce").to_numpy(float)
FOLDS = mc.make_folds(TR.cohort + "_" + TR.label.astype(str), 2 if DEBUG else STAGE2_REPEATS, seed=SEED)
VIT_FOLDS = [f for f in FOLDS if f[0] < (1 if DEBUG else VIT_REPEATS)]      # stage 2's first repeat = the ViT's folds
MODELS = {}
lk = m2e.lookup_preds(TR, VA, FOLDS)
MODELS["subtype lookup"] = {"oof": lk[0], "fold_auc": lk[1], "val": lk[2]}
say(f"train {len(TR)} ({Y.sum()} pCR), val {len(VA)} ({VA.label.sum()} pCR); ViT folds {len(VIT_FOLDS)}, stage-2 folds {len(FOLDS)}")
"""

VIT = r"""
SCORE_COL = {"paper": 0, "fixed": 3}          # paper: mean of the slice probabilities; fixed: area-weighted mean
DL, VLOG, HIST = {}, [], []
for var in VARIANTS:
    IMGS, SL, AREA = DATA[var]
    reps = max(r for r, *_ in VIT_FOLDS) + 1
    oof, vals, fa = np.full((reps, len(TR), 4), np.nan), [], []
    t = time.time()
    say(f"2. ViT '{var}': {BACKBONE}, {'pCR + HR + HER2 heads' if MULTITASK[var] else 'pCR head'}, {len(VIT_FOLDS)} folds")
    for r, f, a, b in VIT_FOLDS:
        if budget_left() < 1.0:
            say(f"2. time budget nearly used: '{var}' stops after {len(fa)} folds"); break
        (pb, pv), ep, ain, hist = m3v.train_fold(
            BACKBONE, IMGS, SL, AREA, R_TR[a], Y[a], HR_TR[a], HER2_TR[a], [R_TR[b], R_VA], multitask=MULTITASK[var],
            train_slices=None if var == "paper" else TRAIN_SLICES, epochs=2 if DEBUG else EPOCHS, patience=PATIENCE,
            batch=BATCH, lr_backbone=LR_BACKBONE, lr_head=LR_HEAD, wd=WD, seed=SEED + 10 * r + f, dev=DEV, say=say)
        oof[r, b], vals = pb, vals + [pv]
        fa.append(mc.auc(Y[b], pb[:, SCORE_COL[var]]))
        VLOG.append({"variant": var, "repeat": r, "fold": f, "best_epoch": ep, "inner_auc": ain, "fold_auc": fa[-1],
                     **{f"fold_auc_{k}": mc.auc(Y[b], pb[:, j]) for j, k in enumerate(m3v.SCORES)}})
        HIST += [{"variant": var, "repeat": r, "fold": f, **h} for h in hist]
        say(f"2. {var} fold {f}: fold AUC {fa[-1]:.3f} (best epoch {ep + 1}, inner {ain:.3f}) | {(time.time() - t) / 60:.1f} min")
    if fa:
        DL[var] = {"oof": np.nanmean(oof, 0), "val": np.mean(vals, 0), "fold_auc": fa}
        o, v = DL[var]["oof"], DL[var]["val"]
        MODELS[f"{var} ViT: image only"] = {"oof": o[:, SCORE_COL[var]], "fold_auc": fa, "val": v[:, SCORE_COL[var]]}
        say(f"2. {var} ViT image only: cv {np.mean(fa):.3f} | oof " +
            ", ".join(f"{k} {mc.auc(Y, o[:, j]):.3f}" for j, k in enumerate(m3v.SCORES)) +
            " | val " + ", ".join(f"{k} {mc.auc(VA.label, v[:, j]):.3f}" for j, k in enumerate(m3v.SCORES)))
pd.DataFrame(VLOG).to_csv(f"{OUT}/vit_folds.csv", index=False)
H = pd.DataFrame(HIST)
H.to_csv(f"{OUT}/vit_epochs.csv", index=False)
if len(H):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for (var, f), g in H.groupby(["variant", "fold"]):
        axes[0].plot(g.epoch + 1, g.loss, "-" if var == "paper" else "--", lw=1)
        axes[1].plot(g.epoch + 1, g.inner_auc, "-" if var == "paper" else "--", lw=1, label=f"{var} f{f}")
    axes[0].set_title("training loss"); axes[1].set_title("inner (early-stopping) AUC"); axes[1].axhline(0.5, color="gray", ls=":")
    axes[1].legend(fontsize=6, ncol=2); plt.tight_layout(); plt.savefig(f"{OUT}/vit_curves.png", dpi=110); plt.show()
"""

STAGE2 = r"""
BASE_TR = pd.concat([mc.clinical_frame(TR), m3s.size_frame(TR)], axis=1).reset_index(drop=True)
BASE_VA = pd.concat([mc.clinical_frame(VA), m3s.size_frame(VA)], axis=1).reset_index(drop=True)
CLIN = [c for c in BASE_TR.columns if c.startswith("clin_")]

def dl_frames(var):
    cols = [f"{var}_vit_{k}" for k in m3v.SCORES]
    return pd.DataFrame(DL[var]["oof"], columns=cols), pd.DataFrame(DL[var]["val"], columns=cols)

SSL = None
if USE_STEP15:
    for f in sorted(glob.glob("/kaggle/input/**/stack_inputs.csv", recursive=True)):
        s = pd.read_csv(f, dtype={"pid": str})
        if "m2_probe_img" in s.columns:
            SSL = s.set_index("pid")["m2_probe_img"]
            say(f"3. step-15 self-supervised 3D score found: {f}")
            break
    if SSL is None:
        say("3. no step-15 output attached: the stage-2 sets without the self-supervised 3D score only")

SETS = {"clinical": (BASE_TR[CLIN], BASE_VA[CLIN]), "clinical + size": (BASE_TR, BASE_VA)}
for var in DL:
    a, b = dl_frames(var)
    SETS[f"clinical + size + {var} ViT"] = (pd.concat([BASE_TR, a], axis=1), pd.concat([BASE_VA, b], axis=1))
if SSL is not None:
    s_tr, s_va = TR.pid.map(SSL).to_numpy(float), VA.pid.map(SSL).to_numpy(float)
    SETS["clinical + size + SSL 3D (step 15)"] = (BASE_TR.assign(ssl3d=s_tr), BASE_VA.assign(ssl3d=s_va))
    if "fixed" in DL:
        a, b = dl_frames("fixed")
        SETS["clinical + size + fixed ViT + SSL 3D"] = (pd.concat([BASE_TR.assign(ssl3d=s_tr), a], axis=1),
                                                        pd.concat([BASE_VA.assign(ssl3d=s_va), b], axis=1))
for name, (A, B) in SETS.items():
    for kind in STAGE2_KINDS:
        r = m3s.cv_refit(A.to_numpy(float), Y, FOLDS, B.to_numpy(float), kind, SEED, JOBS)
        MODELS[f"{name} ({kind.upper()})"] = r
        say(f"3. {name} ({kind.upper()}), {A.shape[1]} features: cv {np.nanmean(r['fold_auc']):.3f} | oof "
            f"{mc.auc(Y, r['oof']):.3f} (within subtype {mc.within_auc(Y, r['oof'], TR.subtype):.3f}) | val "
            f"{mc.auc(VA.label, r['val']):.3f}")
if "clinical + size (RF)" in MODELS:          # what the Random Forest relies on (the paper's Figure 4)
    A, _ = SETS[max((k for k in SETS if "ViT" in k), default="clinical + size", key=len)]
    rf = m3s.make("rf", SEED).fit(A.to_numpy(float), Y)
    imp = pd.Series(rf.named_steps["rf"].feature_importances_[:A.shape[1]], index=A.columns).sort_values(ascending=False)
    print("\nRandom Forest feature importance (largest set, refit on all training patients):")
    print(imp.round(3).to_string())
    imp.to_csv(f"{OUT}/rf_feature_importance.csv")
"""

TABLES = r"""
def pick(names):
    return [n for n in names if n in MODELS]

SHOW = pick(["subtype lookup", "clinical (LR)", "clinical + size (RF)", "paper ViT: image only", "fixed ViT: image only",
             "clinical + size + paper ViT (RF)", "clinical + size + paper ViT (LR)", "clinical + size + fixed ViT (RF)",
             "clinical + size + fixed ViT (LR)", "clinical + size + fixed ViT + SSL 3D (LR)",
             "clinical + size + fixed ViT + SSL 3D (RF)"])
PT_VAL = pd.concat([m3s.paper_table(VA, MODELS[m]["val"], THRESHOLD).assign(model=m) for m in SHOW])
PT_OOF = pd.concat([m3s.paper_table(TR, MODELS[m]["oof"], THRESHOLD).assign(model=m) for m in SHOW])
cols = ["model", "group", "n", "n_pcr", "pcr_rate", "acc", "acc_if_all_no_pcr", "auc", "auc_lo", "auc_hi", "sens", "spec",
        "prec", "npv"]
PT_VAL, PT_OOF = PT_VAL[cols], PT_OOF[cols]
PT_VAL.to_csv(f"{OUT}/paper_style_val.csv", index=False)
PT_OOF.to_csv(f"{OUT}/paper_style_train_oof.csv", index=False)
print(f"Paper-style table, VAL (177), threshold {THRESHOLD} (compare acc with acc_if_all_no_pcr):")
for m in SHOW:
    print(f"\n== {m}")
    print(PT_VAL[PT_VAL.model == m].drop(columns="model").round(3).to_string(index=False))

# I-SPY2 only, like the paper's "I-SPY2 with clinical features" (their extra MammaPrint / drug-arm columns are not public)
ITR, IVA = (TR.cohort == "ispy2").to_numpy(), (VA.cohort == "ispy2").to_numpy()
FOLDS_I = mc.make_folds(TR[ITR].label.to_numpy(), 2 if DEBUG else STAGE2_REPEATS, seed=SEED + 1)
ISPY, rows_i = {}, []
for name in [n for n in ("clinical + size", "clinical + size + paper ViT", "clinical + size + fixed ViT") if n in SETS]:
    A, B = SETS[name]
    for kind in STAGE2_KINDS:
        r = m3s.cv_refit(A.to_numpy(float)[ITR], Y[ITR], FOLDS_I, B.to_numpy(float)[IVA], kind, SEED, JOBS)
        ISPY[f"I-SPY2 only: {name} ({kind.upper()})"] = r
        t_ = m3s.paper_table(VA[IVA].reset_index(drop=True), r["val"], THRESHOLD)
        rows_i.append(t_[t_.group.isin(["overall", "subtype TN", "subtype HER2+", "subtype HR+/HER2-"])]
                      .assign(model=f"{name} ({kind.upper()})", train_cv=np.nanmean(r["fold_auc"]),
                              train_oof=mc.auc(Y[ITR], r["oof"])))
PT_ISPY = pd.concat(rows_i)[["model", "group", "train_cv", "train_oof"] + cols[2:]] if rows_i else pd.DataFrame()
PT_ISPY.to_csv(f"{OUT}/ispy2_only_val.csv", index=False)
print(f"\nI-SPY2-only models (train on {ITR.sum()} I-SPY2 patients, val = {IVA.sum()} I-SPY2 val patients):")
print(PT_ISPY.round(3).to_string(index=False))
"""

SUMMARY = r"""
for k_, m in MODELS.items():   # a ViT stopped by the time budget leaves NaN OOF for some patients: fill with the median
    if np.isnan(m["oof"]).any():
        m["oof"] = np.where(np.isnan(m["oof"]), np.nanmedian(m["oof"]), m["oof"])
PAIRS = [("subtype lookup -> clinical (LR)", "subtype lookup", "clinical (LR)"),
         ("clinical (LR) -> clinical + size (LR)", "clinical (LR)", "clinical + size (LR)"),
         ("clinical + size: LR -> RF", "clinical + size (LR)", "clinical + size (RF)"),
         ("paper ViT -> fixed ViT (image only)", "paper ViT: image only", "fixed ViT: image only"),
         ("clinical + size -> + paper ViT (RF, the paper's pipeline)", "clinical + size (RF)", "clinical + size + paper ViT (RF)"),
         ("clinical + size -> + paper ViT (LR)", "clinical + size (LR)", "clinical + size + paper ViT (LR)"),
         ("clinical + size -> + fixed ViT (LR)", "clinical + size (LR)", "clinical + size + fixed ViT (LR)"),
         ("+ paper ViT -> + fixed ViT (LR)", "clinical + size + paper ViT (LR)", "clinical + size + fixed ViT (LR)"),
         ("clinical + size -> + fixed ViT + SSL 3D (LR)", "clinical + size (LR)", "clinical + size + fixed ViT + SSL 3D (LR)"),
         ("clinical (LR) -> best image + clinical (LR)", "clinical (LR)", "clinical + size + fixed ViT + SSL 3D (LR)")]
STACK = {"s16_paper_vit": "paper ViT: image only", "s16_fixed_vit": "fixed ViT: image only"}
note = (f"Train {len(TR)} / val {len(VA)}. ViT: {len(VIT_FOLDS)}-fold CV on the training patients (OOF), val = mean of "
        f"the fold models; stage 2: {2 if DEBUG else STAGE2_REPEATS} x 5-fold CV + refit for val. The test split is never "
        f"loaded. Caches: " + ", ".join(f"{c} {v['kind']}" for c, v in CACHES.items()))
RES, CMPS = m2e.report(TR, VA, MODELS, PAIRS, OUT, "Step 16 results: the BreastDCEDL paper's ViT pipeline, as published and fixed",
                       note, STACK)
print(RES.round(3).to_string(index=False))
print("\nPaired comparisons (b - a, same patients):")
print(CMPS.round(3).to_string(index=False))
r = RES.set_index("model")[["oof_auc", "val_auc"]]
ax = r.plot.barh(figsize=(8, 0.42 * len(r) + 1.5), width=0.8)
ax.axvline(0.662, ls="--", lw=1, color="k"); ax.axvline(0.72, ls=":", lw=1, color="#b3261e")
ax.text(0.722, -0.6, "paper, all test patients: 0.72", color="#b3261e", fontsize=7)
ax.set_xlim(0.45, 0.85); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/auc_by_model.png", dpi=120); plt.show()

CMP = None
try:
    def as_preds(col):
        V = pd.read_csv(f"{OUT}/val_preds.csv", dtype={"pid": str})
        return V[["pid", "cohort", "label"]].assign(prob=V[col].values)
    best = RES[RES.model != "subtype lookup"].iloc[0].model
    preds = {"B1 LR (clinical + size)": B1_VAL, "subtype lookup": as_preds("subtype lookup"), "F2 fusion v1 (old cache)": F2_V1}
    preds.update({f"{STEP}: {k_}": as_preds(k_) for k_ in dict.fromkeys(
        ["clinical (LR)", "clinical + size + paper ViT (RF)", best]) if k_ in MODELS})
    CMP = compare(preds, "subtype lookup")
    print(CMP.round(3).to_string(index=False))
    CMP.to_csv(f"{OUT}/comparison_vs_team.csv", index=False)
except Exception as e:
    print("comparison with the team's models skipped:", type(e).__name__, e)
"""

RESULTS = r"""
txt = open(f"{OUT}/RESULTS.md").read()
txt += ("\n## Paper-style table (val, threshold %.2f): compare `acc` with `acc_if_all_no_pcr`\n\n" % THRESHOLD
        + mc.md_table(PT_VAL) + "\n")
if len(PT_ISPY):
    txt += "\n## I-SPY2-only models (val = 99 I-SPY2 patients)\n\n" + mc.md_table(PT_ISPY) + "\n"
if CMP is not None:
    txt += "\n## Best models vs the team's models (val, paired DeLong vs the subtype lookup)\n\n" + mc.md_table(CMP) + "\n"
open(f"{OUT}/RESULTS_step16.md", "w").write(txt)
say(f"wrote {OUT}/RESULTS_step16.md (+ paper_style_val.csv, ispy2_only_val.csv, vit_folds.csv, stack_inputs.csv)")
"""

READING = r"""
## How to read the results against the paper
- **Same metric, same kind of set:** the paper's all-cohort test AUC is **0.72** (177 patients). Compare it with our
  `val_auc` (177 different patients, same size) and, better, with `oof_auc` (all 1,098 training patients, much less
  noise). Their test set's cohort counts (I-SPY1 35, I-SPY2 99, Duke 43) match our held-out test split, which this
  notebook never touches.
- **Accuracy:** read `acc` against `acc_if_all_no_pcr`. In a subgroup with 12.5% pCR, 87.5% accuracy needs no model.
- **Subgroup AUCs:** look at `n_pcr` and the CI (`auc_lo`-`auc_hi`). With 5 pCR cases even an AUC of 0.94 has a
  95% CI of roughly 0.8-1.0.
- **What could still explain a gap:** their I-SPY2 model used MammaPrint and the drug arm (not in the public data),
  and the subgroup / model they report was the best of many on one split.
"""

cells = [md(INTRO), code(PARAMS), code(SRC(2)), code(SRC(3)), code(SRC(4)),
         code(SRC(5).replace("Shared code for steps 14 and 15", "Shared code for steps 14-16")), code(SRC(8)),
         module("m3_data"), module("m3_vit"), module("m3_stage2"), code(CACHES),
         md("## 1. 2D RGB-fused tumour slices, paper and fixed versions (CPU workers, ~5-10 min)"), code(SLICES),
         md("## 2. ViT fine-tuning, 5-fold CV on the training patients (GPU, ~6-10 min per fold)\n"
            "Every slice carries its patient's label; each fold is early-stopped on an inner 15% of its training patients."),
         code(VIT),
         md("## 3. Stage 2: patient-level scores + clinical data + tumour size -> Random Forest / logistic regression"),
         code(STAGE2),
         md("## 4. The paper's Table 2 layout, and I-SPY2-only models"), code(TABLES),
         md("## 5. Summary and paired comparisons"), code(SUMMARY), md(READING),
         md("## 6. Results file"), code(RESULTS), code(SRC(22))]
nb = copy.deepcopy(S15)
nb["cells"] = cells
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open(f"{HERE}/step16.ipynb", "w"), indent=1)
print("wrote", f"{HERE}/step16.ipynb", "with", len(cells), "cells")
