"""Generates the Kaggle notebooks in this folder (A runs this after changing a notebook's cells).

    python kaggle/make_notebooks.py

Every notebook starts with the same two cells: PARAMETERS, then SETUP (clone the git tag into
/tmp/repo using the GITHUB_TOKEN Kaggle secret, install requirements-kaggle.txt). The token is
passed as an HTTP header for each git call and is never written to disk or to the notebook output.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_SLUG = "AkulaReddy/MRI_chemo_response"
CURRENT_TAG = "d1.10"   # bump this whenever a new code tag is made, then regenerate

SETUP = r'''# ==== SETUP (do not edit) ====
import base64, json, os, shutil, subprocess, sys
assert PERSON in ("A", "B", "C"), 'set PERSON = "A", "B" or "C" in the parameters cell'
from kaggle_secrets import UserSecretsClient
_tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
_hdr = "Authorization: Basic " + base64.b64encode(f"x-access-token:{_tok}".encode()).decode()

def git(*args, cwd="/tmp/repo"):
    """git with the token as a one-off header (never stored in .git/config or the output)."""
    r = subprocess.run(["git", "-c", f"http.extraHeader={_hdr}", *args], cwd=cwd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace(_tok, "***")
    if r.returncode:
        raise RuntimeError(f"git {args[0]} failed:\n{out}")
    return out

shutil.rmtree("/tmp/repo", ignore_errors=True)
git("clone", "--quiet", "--depth", "1", "--branch", GIT_TAG, f"https://github.com/{REPO}.git", "/tmp/repo", cwd="/tmp")
os.chdir("/tmp/repo")
print("code at", GIT_TAG, git("log", "-1", "--format=%h %s"))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"], check=True)
import torch; print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
'''

PUSH = r'''# ==== PUSH the small result files to GitHub (results/<PERSON>/<run_id>/) ====
if PUSH_RESULTS:
    git("config", "user.name", f"Person {PERSON} (Kaggle)")
    git("config", "user.email", "kaggle@users.noreply.github.com")
    # the clone holds only the tag, so fetch main into origin/main explicitly; on a rejected push
    # (someone else pushed first) rebuild the commit on the new main instead of rebasing a shallow clone
    for attempt in range(3):
        git("fetch", "--quiet", "--depth", "1", "origin", "+refs/heads/main:refs/remotes/origin/main")
        git("checkout", "--quiet", "-f", "-B", "main", "origin/main")
        shutil.copytree(f"/kaggle/working/results/{PERSON}", f"results/{PERSON}", dirs_exist_ok=True)
        git("add", f"results/{PERSON}")
        print(git("commit", "-m", f"results: {PERSON} {CONFIG} seed {SEED} ({GIT_TAG})"))
        try:
            git("push", "--quiet", "origin", "main"); print("pushed"); break
        except RuntimeError as e:
            print(f"push attempt {attempt + 1} failed:", e)
    else:
        print("NOT pushed: download results/ from the Output and send it to A")
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def notebook(cells):
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}}}


def params(extra):
    return code(f'''# ==== PARAMETERS (edit these) ====
REPO = "{REPO_SLUG}"
GIT_TAG = "{CURRENT_TAG}"        # code version; A posts the tag to use
PERSON = ""             # set to "A", "B" or "C"
''' + extra)


NOTEBOOKS = {
    "_template.ipynb": [
        md("""# Template
Copy this for new notebooks. Session options: GPU (T4 or P100) only if training; **Internet ON**;
secret `GITHUB_TOKEN` attached (Add-ons -> Secrets)."""),
        params(""), code(SETUP),
    ],
    "01_build_cache.ipynb": [
        md("""# 01 Build the slice caches, all cohorts (CPU, no GPU needed; A runs this)
Builder v2: every patient resampled to 2.0 mm slices x 0.7 mm pixels, centred on the tumour, clip
ceiling raised so the late phase doesn't saturate in the tumour (see `scripts/build_cache.py`).
1. Session options: Accelerator **None**, Internet **ON**, secret `GITHUB_TOKEN` attached.
2. **+ Add Input** -> all three raw Datasets (`mnj-raw-ispy1`, `mnj-raw-ispy2`, `mnj-raw-duke`;
   B and C must have shared theirs with you).
3. Quick check first: `LIMIT = 10`, run interactively, look at the grids. Then `LIMIT = 0` and
   **Save Version -> Save & Run All (Commit)** (several hours; keeps running with the browser closed).
4. When it finishes: check the QC tables and grids, then **New Dataset** from the output, named
   `mnj-cache-v2` (Private, shared with the team). It holds one folder per cohort, `cache_v2_<cohort>`.
   Training picks the highest builder version automatically if old caches are attached too.

**Builder v3** (`Z_MM = 0`): same as v2 in-plane (0.7 mm, tumour-centred), but the **native slices are
kept** (no resampling between slices). Output folders `cache_v3_<cohort>`; name the Dataset `mnj-cache-v3`.
Notebooks that compare v2 and v3 (`14_v3_enhancement`) choose the cache per run."""),
        params('''COHORTS = ["ispy1", "ispy2", "duke"]
LIMIT = 0               # 0 = all patients; 10 for a quick check
WORKERS = 4
RAW = {}                # {cohort: path} to override the automatic search for mnj-raw-<cohort>
Z_MM = 2.0              # 2.0 = builder v2 (2 mm slices); 0 = builder v3 (native slices, 0.7 mm in-plane)
'''),
        code(SETUP),
        code(r'''import time

def find_raw(cohort):
    """Kaggle may unpack the .tar.gz into a folder; either form works."""
    want = f"BreastDCEDL_{cohort.upper()}_min_crop"
    hits = []
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if root.count("/") > 7:
            dirs[:] = []
            continue
        hits += [os.path.join(root, n) for n in dirs + files if n.startswith(want)]
        dirs[:] = [d for d in dirs if not d.startswith("BreastDCEDL_")]
    if not hits:
        return None
    return hits[0] if os.path.isdir(hits[0]) else os.path.dirname(hits[0])

raws = {c: RAW.get(c) or find_raw(c) for c in COHORTS}
print("raw inputs:", raws)
if not all(raws.values()):
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        print(root, dirs[:5], files[:5])
        if root.count("/") > 5:
            dirs[:] = []
    raise SystemExit(f"missing raw input for {[c for c, r in raws.items() if not r]}: attach mnj-raw-<cohort>")

OUTS, status = {}, {}
for c in COHORTS:   # one cohort at a time; a failed cohort is reported, the others still build
    OUTS[c] = f"/kaggle/working/cache_v{2 if Z_MM else 3}_{c}"
    cmd = [sys.executable, "scripts/build_cache.py", "--cohort", c, "--raw", raws[c], "--out", OUTS[c],
           "--workers", str(WORKERS), "--z-mm", str(Z_MM)] + (["--limit", str(LIMIT)] if LIMIT else [])
    t = time.time()
    print(f"\n===== {c}: {' '.join(cmd)}", flush=True)
    status[c] = subprocess.run(cmd).returncode
    print(f"===== {c}: exit {status[c]}, {(time.time() - t) / 60:.1f} min", flush=True)
print("\nexit codes:", status)
'''),
        code(r'''import pandas as pd
from IPython.display import Image, display

for c, out in OUTS.items():
    print(f"\n==================== {c}")
    if not os.path.exists(f"{out}/cache_index.csv"):
        print("no cache_index.csv: the build failed, see the log above")
        continue
    idx = pd.read_csv(f"{out}/cache_index.csv")
    meta = json.load(open(f"{out}/cache_meta.json"))
    print({k: meta.get(k) for k in ("builder_version", "grid_mm", "git", "status_counts", "seconds")})
    for col in ("z_source", "xy_source"):
        if col in idx:
            print(col, idx[col].value_counts(dropna=False).to_dict())
    if "mask_csv_iou_yx" in idx and idx.mask_csv_iou_yx.notna().any():
        print("mask vs CSV box IoU:", idx.mask_csv_iou_yx.describe().round(3).to_dict())
    ok = idx[idx.status == "ok"]
    show = [k for k in ("src_dxy", "src_dz", "n_slices", "sat_frac") if k in ok]
    print("per patient (ok rows):")
    display(ok[show].describe(percentiles=[0.5]).round(3))
    if "sat_frac" in ok:
        print(f"tumour-box pixels with a saturated channel: median {ok.sat_frac.median():.4f}, "
              f"patients over 1%: {(ok.sat_frac > 0.01).sum()} of {len(ok)}")
    mb = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e6
    print(f"size on disk: {mb:.0f} MB")
    err = idx[idx.status == "error"]
    if len(err):
        display(err[["pid", "note"]].head(20))
    if os.path.exists(f"{out}/summary_grid.png"):
        display(Image(f"{out}/summary_grid.png"))
'''),
    ],
    "10_train.ipynb": [
        md("""# 10 Train / export (GPU)
1. Session options: Accelerator **GPU T4 x2 or P100** (one GPU is used), Internet **ON**, secret `GITHUB_TOKEN`.
2. **+ Add Input**: the cache Dataset **`mnj-cache-v2`** (builder v2, all three cohorts; from `d1.5`)
   (and, for `f2_from_b3` or export, the weights Dataset `mnj-w-<run_id>`). With `CACHE_VERSION = 2`
   the run stops if any cohort's cache is not v2; the caches used are printed and saved in metrics.json.
3. Set the parameters. First run of the day: `DEBUG = True` (2 epochs, 48 patients, a few minutes).
4. Real runs: **Save Version -> Save & Run All (Commit)** so it keeps running with the browser closed.
5. Afterwards: **New Dataset** from the output -> name it `mnj-w-<run_id>` (the run_id is printed at the end).
   Post the val AUC (with CI), AUPRC, ECE and GPU-h in the chat."""),
        params('''CONFIG = "b3_vit"       # b3_vit, b3_vit_supervised, f1_concat, f2_attn, f2_meanpool, f2_mil, ...
SEED = 0
MODE = "train"          # "train" or "export"
DEBUG = False
DATASETS = []           # [] = all cohorts in the config; ["ispy1"] if only that cache exists yet
CHECKPOINT = ""         # MODE="export": path to best.pt under /kaggle/input/mnj-w-<run_id>/...
INIT_FROM = ""          # f2_from_b3: path to the B3 best.pt
CACHE_VERSION = 2       # required cache builder version (2 = mnj-cache-v2); None = take whatever is attached
EXTRA = []              # optional overrides, e.g. ["optim.patience=15"] or ["data.min_area_frac=0.1"]
PUSH_RESULTS = True     # commit results/<PERSON>/<run_id>/ (small files only) to GitHub
'''),
        code(SETUP),
        code(r'''cmd = [sys.executable, "-m", "src.train", "--config", CONFIG, "--seed", str(SEED), "--mode", MODE,
       "--cache-root", "/kaggle/input", "--out", "/kaggle/working/runs",
       "--results-dir", f"/kaggle/working/results/{PERSON}"]
if DEBUG: cmd.append("--debug")
if CACHE_VERSION is not None: cmd += ["--cache-version", str(CACHE_VERSION)]
if DATASETS: cmd += ["--cohorts", *DATASETS]
if CHECKPOINT:
    # export writes next to the checkpoint; /kaggle/input is read-only, so copy the run folder first
    src_dir = os.path.dirname(CHECKPOINT)
    dst = f"/kaggle/working/runs/{os.path.basename(src_dir)}"
    shutil.copytree(src_dir, dst, dirs_exist_ok=True)
    cmd += ["--checkpoint", f"{dst}/best.pt"]
if INIT_FROM: cmd += ["--init-from", INIT_FROM]
if EXTRA: cmd += ["--set", *EXTRA]
print(" ".join(cmd))
subprocess.run(cmd, check=True)
import glob
RUN_DIR = sorted(glob.glob("/kaggle/working/runs/*/"), key=os.path.getmtime)[-1]
print("\nrun_id:", os.path.basename(RUN_DIR.rstrip("/")))
if os.path.exists(f"{RUN_DIR}/metrics.json"):
    m = json.load(open(f"{RUN_DIR}/metrics.json"))
    v = m["val"]
    print(f"val AUC {v['auc']:.3f} (CI {v['auc_ci'][0]:.3f}-{v['auc_ci'][1]:.3f}) AUPRC {v['auprc']:.3f} "
          f"ECE {v['ece']:.3f} | best epoch {m['best_epoch']} | {m['gpu_hours']} GPU-h")
    print("caches:", {c: f"v{i['builder_version']}" for c, i in m.get("caches", {}).items()})
'''),
        code(PUSH.replace("if PUSH_RESULTS:", "if PUSH_RESULTS and not DEBUG:")),
    ],
    "11_sweep.ipynb": [
        md("""# 11 Sweep: several training settings in one session (GPU)
Runs every entry of `RUNS` (x every seed) one after the other, then compares them on the validation set.
1. Session options: Accelerator **GPU T4 x2 or P100**, Internet **ON**, secret `GITHUB_TOKEN`.
2. **+ Add Input**: `mnj-cache-v2`.
3. Edit `RUNS`. The **first entry is the baseline**; every other run is compared with it. Change one
   thing per entry, so each difference has one cause.
4. `DEBUG = True` first (2 epochs, 48 patients per run, minutes), then `DEBUG = False` and
   **Save Version -> Save & Run All (Commit)**. Runs that won't fit in `TIME_BUDGET_H` are skipped
   (Kaggle stops a session after 12 h); `summary.csv` is rewritten after every run, so a cut session
   still keeps what finished.

**Reading the table (val = 177 patients, 53 pCR+):** one AUC has a 95% CI of about +-0.08, so most
small differences are noise. Trust a setting only if its **paired** difference vs the baseline has a
CI above 0, and confirm it with more seeds before adopting it. Never tune on the test set."""),
        params('''CONFIG = "b3_vit"
SEEDS = [0]                       # add seeds (e.g. [0, 1, 2]) to confirm a winner
RUNS = {                          # name: overrides; the FIRST entry is the baseline
    # B3 on v2 learns but overfits (train AUC 0.87, val 0.56): every entry tests one way to reduce that
    "base":         ["optim.patience=15"],
    "frozen":       ["optim.patience=15", "model.train_blocks=0"],             # linear probe on frozen MAE features
    "blocks_2":     ["optim.patience=15", "model.train_blocks=2"],             # fewer trainable blocks (default 4)
    "min_area_0.1": ["optim.patience=15", "data.min_area_frac=0.1"],           # skip near-empty tumour slices
    "strong_aug":   ["optim.patience=15", "aug.rotate_deg=30", "aug.intensity=0.2", "aug.scale=[0.8, 1.2]"],
    "dropout_0.5":  ["optim.patience=15", "model.dropout=0.5", "optim.weight_decay=0.1"],
    "supervised":   ["optim.patience=15", "model.backbone=google/vit-base-patch16-224"],  # ImageNet-supervised ViT
}
BASELINE = None                   # or {"name": ..., "val_preds": path to an earlier run's val_preds.csv}
CACHE_VERSION = 2
DEBUG = False
TIME_BUDGET_H = 11.0              # stop starting new runs after this (Kaggle limit: 12 h)
KEEP_CHECKPOINTS = True           # best.pt is ~350 MB per run (7 runs ~2.5 GB of the 20 GB output)
PUSH_RESULTS = True               # commit results/<PERSON>/<sweep id>/ (small files only)
'''),
        code(SETUP),
        code(r'''import glob, time
import pandas as pd

SEED = "sweep"   # used in the push commit message
SWEEP_ID = f"sweep_{CONFIG}_{time.strftime('%m%d-%H%M')}" + ("_debug" if DEBUG else "")
RES = f"/kaggle/working/results/{PERSON}/{SWEEP_ID}"
os.makedirs(RES, exist_ok=True)
json.dump({"config": CONFIG, "seeds": SEEDS, "runs": RUNS, "git_tag": GIT_TAG, "debug": DEBUG},
          open(f"{RES}/sweep.json", "w"), indent=2)
rows, t_start, durations = [], time.time(), []

def run_spec(spec):
    """A RUNS entry: a list of overrides, or {"set": [overrides], "cache": builder version}."""
    return (spec, CACHE_VERSION) if isinstance(spec, list) else (spec.get("set", []), spec.get("cache", CACHE_VERSION))

CACHES = {}   # {builder version: {cohort: dir}} of the attached caches; each run is pointed at one version
for depth in range(6):
    for f in glob.glob(os.path.join("/kaggle/input", *["*"] * depth, "cache_meta.json")):
        m = json.load(open(f))
        CACHES.setdefault(m.get("builder_version", 1), {})[m["cohort"]] = os.path.dirname(f)
print("attached caches:", {f"v{v}": sorted(d) for v, d in sorted(CACHES.items())})

def finished_run_dir(before):
    new = sorted(set(glob.glob("/kaggle/working/runs/*/")) - before, key=os.path.getmtime)
    return os.path.normpath(new[-1]) if new else None

for name, spec in RUNS.items():
    extra, version = run_spec(spec)
    for seed in SEEDS:
        used_h = (time.time() - t_start) / 3600
        est_h = (sum(durations) / len(durations) / 3600) if durations else 0
        if used_h + est_h > TIME_BUDGET_H:
            print(f"skip {name} seed {seed}: {used_h:.1f} h used, next run needs ~{est_h:.1f} h")
            rows.append({"name": name, "seed": seed, "status": "skipped (time)"})
            continue
        cmd = [sys.executable, "-m", "src.train", "--config", CONFIG, "--seed", str(seed), "--mode", "train",
               "--cache-root", "/kaggle/input", "--out", "/kaggle/working/runs", "--results-dir", RES]
        if version is not None:
            cmd += ["--cache-version", str(version)]
            if len(CACHES) > 1 and version in CACHES:   # several versions attached: name this one's folders
                cmd += ["--cache", *[f"{c}={d}" for c, d in CACHES[version].items()]]
        if DEBUG:
            cmd.append("--debug")
        if extra:
            cmd += ["--set", *extra]
        print(f"\n===== {name} seed {seed} (cache v{version}): {' '.join(extra) or '(defaults)'}", flush=True)
        before, t = set(glob.glob("/kaggle/working/runs/*/")), time.time()
        rc = subprocess.run(cmd).returncode
        durations.append(time.time() - t)
        run_dir = finished_run_dir(before)
        row = {"name": name, "seed": seed, "cache": version, "overrides": " ".join(extra),
               "status": "ok" if rc == 0 else f"exit {rc}",
               "run_id": os.path.basename(run_dir) if run_dir else None, "minutes": round(durations[-1] / 60, 1)}
        if rc == 0 and run_dir and os.path.exists(f"{run_dir}/metrics.json"):
            m = json.load(open(f"{run_dir}/metrics.json"))
            v = m["val"]
            log = pd.read_csv(f"{run_dir}/log.csv")
            row.update(val_auc=round(v["auc"], 4), ci_lo=round(v["auc_ci"][0], 3), ci_hi=round(v["auc_ci"][1], 3),
                       auprc=round(v["auprc"], 3), ece=round(v["ece"], 3), best_epoch=m["best_epoch"],
                       epochs_run=m["epochs_run"], max_val_auc_any_epoch=round(log.val_auc.max(), 4),
                       **{f"auc_{c}": round(x["auc"], 3) for c, x in v.get("by_cohort", {}).items()})
            print(f"===== {name} seed {seed}: val AUC {v['auc']:.3f} (CI {v['auc_ci'][0]:.3f}-{v['auc_ci'][1]:.3f}), "
                  f"best epoch {m['best_epoch']} of {m['epochs_run']}, {row['minutes']} min", flush=True)
        if run_dir and not KEEP_CHECKPOINTS:
            for f in glob.glob(f"{run_dir}/*.pt"):
                os.remove(f)
        rows.append(row)
        pd.DataFrame(rows).to_csv(f"{RES}/summary.csv", index=False)   # saved after every run
print(f"\nsweep done in {(time.time() - t_start) / 3600:.2f} h -> {RES}/summary.csv")
'''),
        code(r'''from src import stats
from IPython.display import display
import matplotlib.pyplot as plt

S = pd.DataFrame(rows)
ok = S[S.status == "ok"].copy()
preds = {(r.name, r.seed): pd.read_csv(f"{RES}/{r.run_id}/val_preds.csv").set_index("pid") for r in ok.itertuples()}
if BASELINE:   # an earlier run (in the repo clone) is the baseline; every run here is compared with it
    base = BASELINE["name"]
    ext = pd.read_csv(BASELINE["val_preds"]).set_index("pid")
    print(f"baseline '{base}' from {BASELINE['val_preds']}: val AUC {stats.auc(ext.label, ext.prob):.4f}")
    get_base = lambda seed: ext
else:          # the first entry of RUNS, same seed
    base = next(iter(RUNS))
    get_base = lambda seed: preds.get((base, seed))
for col in ("vs_base", "vs_base_ci", "vs_base_p"):
    ok[col] = None
for i, r in ok.iterrows():   # paired bootstrap against the baseline
    b = get_base(r.seed)
    if b is None or r["name"] == base:
        continue
    p = preds[(r["name"], r.seed)]
    common = b.index.intersection(p.index)          # same val patients in both (all 177 normally)
    if len(common) < len(b):
        print(f"{r['name']}: compared on {len(common)} of {len(b)} baseline patients "
              f"(baseline AUC on these: {stats.auc(b.loc[common, 'label'], b.loc[common, 'prob']):.4f})")
    d = stats.paired_bootstrap_auc_diff(b.loc[common, "label"], b.loc[common, "prob"], p.loc[common, "prob"])
    ok.at[i, "vs_base"], ok.at[i, "vs_base_ci"], ok.at[i, "vs_base_p"] = (
        round(d["diff"], 3), f"{d['ci'][0]:+.3f} to {d['ci'][1]:+.3f}", round(d["p"], 3))
cols = ["name", "seed", "val_auc", "ci_lo", "ci_hi", "vs_base", "vs_base_ci", "vs_base_p", "auprc", "ece",
        "best_epoch", "epochs_run", "max_val_auc_any_epoch", "auc_duke", "auc_ispy1", "auc_ispy2", "minutes"]
print(f"baseline = '{base}'. vs_base = AUC(run) - AUC(baseline), paired bootstrap on the same val patients.")
display(ok[[c for c in cols if c in ok]].sort_values("val_auc", ascending=False))
if len(SEEDS) > 1:
    print("mean over seeds:")
    display(ok.groupby("name")[["val_auc", "auprc", "ece", "best_epoch"]].agg(["mean", "std"]).round(3))
    # each setting's seeds averaged into one prediction per patient (less seed noise), then compared
    # with the baseline's seed average on the same patients
    avg = {}
    for n in ok["name"].unique():
        ps = [preds[(n, s)] for s in ok[ok["name"] == n].seed]
        common = ps[0].index
        for q in ps[1:]:
            common = common.intersection(q.index)
        avg[n] = pd.DataFrame({"label": ps[0].loc[common, "label"],
                               "prob": sum(q.loc[common, "prob"] for q in ps) / len(ps)})
    b = ext if BASELINE else avg.get(base)
    if b is not None:
        rows_avg = []
        for n, p in avg.items():
            common = b.index.intersection(p.index)
            d = stats.paired_bootstrap_auc_diff(b.loc[common, "label"], b.loc[common, "prob"], p.loc[common, "prob"])
            rows_avg.append({"name": n, "seeds": int((ok["name"] == n).sum()),
                             "auc_seed_avg": round(stats.auc(p.label, p.prob), 4), "vs_base": round(d["diff"], 3),
                             "vs_base_ci": f"{d['ci'][0]:+.3f} to {d['ci'][1]:+.3f}", "vs_base_p": round(d["p"], 3)})
        print(f"seed-averaged predictions, paired vs '{base}':")
        A = pd.DataFrame(rows_avg)
        display(A)
        A.to_csv(f"{RES}/summary_seed_avg.csv", index=False)
if (S.status != "ok").any():
    display(S[S.status != "ok"][["name", "seed", "status"]])
ok.to_csv(f"{RES}/summary.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for r in ok.itertuples():
    log = pd.read_csv(f"{RES}/{r.run_id}/log.csv")
    axes[0].plot(log.epoch, log.val_auc, label=f"{r.name} s{r.seed}")
    axes[1].plot(log.epoch, log.train_loss, label=f"{r.name} s{r.seed}")
axes[0].set(title="val AUC per epoch", xlabel="epoch"); axes[1].set(title="train loss per epoch", xlabel="epoch")
axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{RES}/curves.png", dpi=90); plt.show()
'''),
        code(PUSH.replace("if PUSH_RESULTS:", "if PUSH_RESULTS and not DEBUG:")),
    ],
}


# 12_backbones: the sweep notebook with other defaults (DINOv2 and ResNet-18 on the early phase only)
NOTEBOOKS["12_backbones.ipynb"] = [
    md("""# 12 Backbones: DINOv2 ViT-B and ResNet-18 on the early post-contrast phase (GPU)
Same machinery as `11_sweep` (runs one after the other, paired comparison with the first entry, curves,
push), with these defaults:
- **baseline** (not retrained): sweep 1's `blocks_2` run (MAE ViT, 2 trainable blocks, pre/early/late as RGB,
  val AUC 0.579), read from its `val_preds.csv` in the repo;
- `mae_early`: the same model on the **early phase only** (copied into the 3 channels = a 1-channel input), to
  separate the effect of the input from the effect of the backbone;
- `dinov2_early`: **DINOv2 ViT-B/14** (`facebook/dinov2-base`), top 2 blocks fine-tuned;
- `resnet18_early`: **ImageNet ResNet-18**, all stages fine-tuned, backbone LR 1e-4.

1. Session options: **GPU**, Internet **ON** (downloads DINOv2 and ResNet weights), secret `GITHUB_TOKEN`.
2. **+ Add Input**: `mnj-cache-v2`.
3. `DEBUG = True` first (minutes), then `DEBUG = False` and **Save Version -> Save & Run All (Commit)**
   (about 1.5-2 h).

Read the table as in `11_sweep`: a setting is better only if its paired CI vs the baseline is above 0."""),
    params('''CONFIG = "b3_vit"
SEEDS = [0]
P = ["optim.patience=15"]
RUNS = {                          # name: overrides; all are compared with BASELINE below
    "mae_early":      P + ["model.train_blocks=2", "model.input_phases=early"],
    "dinov2_early":   P + ["model.backbone=facebook/dinov2-base", "model.train_blocks=2", "model.input_phases=early"],
    "resnet18_early": P + ["model.backbone=resnet18", "model.train_blocks=all", "model.input_phases=early",
                           "optim.lr_backbone=1.0e-4"],
}
# baseline = sweep 1's blocks_2 run (MAE ViT, RGB, 2 blocks, seed 0, val AUC 0.579), not retrained here
BASELINE = {"name": "mae_rgb (sweep 1 blocks_2)",
            "val_preds": "results/A/sweep_b3_vit_1019-0406/b3_vit_s0_1019-0500_4662d29/val_preds.csv"}
CACHE_VERSION = 2
DEBUG = False
TIME_BUDGET_H = 11.0              # stop starting new runs after this (Kaggle limit: 12 h)
KEEP_CHECKPOINTS = True           # best.pt: ~350 MB (ViT) / ~45 MB (ResNet-18) per run
PUSH_RESULTS = True               # commit results/<PERSON>/<sweep id>/ (small files only)
'''),
    *NOTEBOOKS["11_sweep.ipynb"][2:],
]


# 13_dinov2_ispy: DINOv2 (early phase) trained and validated on I-SPY1 + I-SPY2 only, with/without dropout
NOTEBOOKS["13_dinov2_ispy.ipynb"] = [
    md("""# 13 DINOv2 on I-SPY only, with regularisation (GPU)
Same machinery as `11_sweep`. Every run: DINOv2 ViT-B/14, early phase only, top 2 blocks trained, and
**I-SPY1 + I-SPY2 only** for training *and* validation (`data.cohorts=[ispy1, ispy2]`: 888 train, 132 val
patients). Duke was at chance (0.44) for DINOv2 in `12_backbones`.
- **baseline** (not retrained): `dinov2_early` from `12_backbones` (all cohorts), scored on the **same 132
  I-SPY val patients**;
- `ispy`: the same model trained on I-SPY only (effect of dropping Duke);
- `ispy_dropout_0.5`: + dropout 0.5 before the head;
- `ispy_droppath_0.2`: + stochastic depth 0.2 in the 2 trained DINOv2 blocks (`model.drop_path`);
- `ispy_both`: + both.

1. Session options: **GPU**, Internet **ON**, secret `GITHUB_TOKEN`; **+ Add Input** `mnj-cache-v2`.
2. `DEBUG = True` first, then `DEBUG = False` and **Save Version -> Save & Run All (Commit)** (about 2 h).

Note: AUCs here are on 132 patients (43 pCR+), so the CIs are wider than on the full 177."""),
    params('''CONFIG = "b3_vit"
SEEDS = [0]
P = ["optim.patience=15", "model.backbone=facebook/dinov2-base", "model.train_blocks=2", "model.input_phases=early",
     "data.cohorts=[ispy1, ispy2]"]
RUNS = {                          # name: overrides; all are compared with BASELINE below
    "ispy":              P,
    "ispy_dropout_0.5":  P + ["model.dropout=0.5"],
    "ispy_droppath_0.2": P + ["model.drop_path=0.2"],
    "ispy_both":         P + ["model.dropout=0.5", "model.drop_path=0.2"],
}
# baseline = dinov2_early from 12_backbones (all cohorts, seed 0), compared on the I-SPY val patients
BASELINE = {"name": "dinov2_early (all cohorts)",
            "val_preds": "results/A/sweep_b3_vit_1019-1102/b3_vit_s0_1019-1129_a7e574e/val_preds.csv"}
CACHE_VERSION = 2
DEBUG = False
TIME_BUDGET_H = 11.0              # stop starting new runs after this (Kaggle limit: 12 h)
KEEP_CHECKPOINTS = True           # best.pt ~350 MB per run
PUSH_RESULTS = True               # commit results/<PERSON>/<sweep id>/ (small files only)
'''),
    *NOTEBOOKS["11_sweep.ipynb"][2:],
]


# 02_radiomics: hand-made tumour features from cache v2 + ComBat + subtype baselines (CPU)
NOTEBOOKS["02_radiomics.ipynb"] = [
    md("""# 02 Radiomics + ComBat + subtype baselines (CPU, no GPU needed)
1. **Features** (`scripts/extract_radiomics.py`, `src/radiomics.py`) for every cached patient. They're
   computed on cache v2 (same 0.7 x 0.7 x 2 mm voxels for everyone), with intensities converted back to
   raw-proportional units:
   - **shape:** volume, sphericity, axes, number of parts;
   - **kinetics:** relative enhancement (early - pre)/pre and (late - pre)/pre; washout / plateau / persistent
     fractions;
   - **heterogeneity:** entropy and GLCM texture of the enhancement map;
   - **peritumoural 5 mm ring:** enhancement just outside the tumour.

   Tumour = the I-SPY mask; for Duke (boxes only), the box voxels enhancing >= 70% (the I-SPY rule).
2. **Baselines** (`src/baselines/radiomics.py`), fitted on train and scored on val:
   - `subtype`: HR/HER2 lookup;
   - `rad`: radiomics, ComBat-harmonised by protocol group (matrix x slice thickness x pixel size, per
     cohort);
   - `rad_sub`: `rad` + HR/HER2;
   - `rad_nocombat`, `rad_sub_nocombat`: the same without ComBat.

   Each model is compared with `subtype` by paired bootstrap.

Session: Accelerator **None**, Internet **ON**, secret `GITHUB_TOKEN`, **+ Add Input** `mnj-cache-v2`.
`LIMIT = 30` first (a minute), then `LIMIT = 0` and **Save & Run All (Commit)** (about 30-60 min).
The features CSV is pushed with the results, so A can rerun the baselines locally."""),
    params('''LIMIT = 0               # patients per cohort; 30 for a quick check
WORKERS = 4
CACHE_VERSION = 2
PUSH_RESULTS = True
'''),
    code(SETUP),
    code(r'''import time
CONFIG, SEED = "radiomics", "-"     # used in the push commit message
RES = f"/kaggle/working/results/{PERSON}/radiomics_{time.strftime('%m%d-%H%M')}" + ("_debug" if LIMIT else "")
os.makedirs(RES, exist_ok=True)
cmd = [sys.executable, "scripts/extract_radiomics.py", "--cache-root", "/kaggle/input", "--cache-version",
       str(CACHE_VERSION), "--out", f"{RES}/features.csv", "--workers", str(WORKERS)] + (["--limit", str(LIMIT)] if LIMIT else [])
print(" ".join(cmd)); subprocess.run(cmd, check=True)
cmd = [sys.executable, "-m", "src.baselines.radiomics", "--features", f"{RES}/features.csv", "--out", RES]
print(" ".join(cmd)); subprocess.run(cmd, check=True)
'''),
    code(r'''import glob
import pandas as pd
from IPython.display import display
f = pd.read_csv(f"{RES}/features.csv")
print("mask source per cohort:"); display(f.groupby("cohort").mask_source.value_counts().unstack(fill_value=0))
from src.radiomics import FEATURES
print("feature medians per cohort (raw, before ComBat):"); display(f.groupby("cohort")[FEATURES].median().T.round(3))
rows = []
for m in sorted(glob.glob(f"{RES}/r_*/metrics.json")):
    j = json.load(open(m)); v = j["val"]
    rows.append({"model": j["config"], "val_auc": round(v["auc"], 3), "ci": f"{v['auc_ci'][0]:.3f}-{v['auc_ci'][1]:.3f}",
                 **{f"auc_{c}": round(x["auc"], 3) for c, x in v["by_cohort"].items()},
                 "batch_acc_before": j.get("batch_acc_before"), "batch_acc_after": j.get("batch_acc_after"),
                 "chance": j.get("batch_chance")})
display(pd.DataFrame(rows))
'''),
    code(PUSH.replace("if PUSH_RESULTS:", "if PUSH_RESULTS and not LIMIT:")),
]


# 14_v3_enhancement: cache v2 vs v3 (native slices) and relative-enhancement input, 2 seeds each
NOTEBOOKS["14_v3_enhancement.ipynb"] = [
    md("""# 14 Native slices (cache v3) and relative-enhancement input (GPU)
Same machinery as `11_sweep`. Every run: DINOv2 ViT-B/14, top 2 blocks trained, dropout 0.5, all cohorts.
Two questions (see `docs/ideas.md`, items 6 and 13):
1. **Does resampling between slices matter?** v2 = 2 mm slices, v3 = native slices (both 0.7 mm in-plane).
2. **Does a scanner-independent input help?** `enhancement` = early, (early - pre)/pre, (late - pre)/pre.

Runs (each with seeds 0 and 1); the **first entry is the baseline**:
- `v2_early`: cache v2, early phase only (the current best setting);
- `v3_early`: cache v3, early phase only -> effect of native slices;
- `v2_rgb`: cache v2, raw pre/early/late -> effect of using all three phases;
- `v2_enh`: cache v2, relative-enhancement maps -> effect of the normalisation, compared with `v2_rgb`;
- `v3_enh`: cache v3, relative-enhancement maps.

1. Session options: **GPU**, Internet **ON**, secret `GITHUB_TOKEN`.
2. **+ Add Input**: **both** `mnj-cache-v2` and `mnj-cache-v3` (built with `01_build_cache`, `Z_MM = 0`).
   Each run is pointed at its cache version; the attached versions are printed first.
3. `DEBUG = True` first, then `DEBUG = False` and **Save Version -> Save & Run All (Commit)** (about 5.5 h).

Read the **seed-averaged** table: it averages each setting's two seeds before the paired comparison, so it
is less noisy than the single-seed rows."""),
    params('''CONFIG = "b3_vit"
SEEDS = [0, 1]
P = ["optim.patience=15", "model.backbone=facebook/dinov2-base", "model.train_blocks=2", "model.dropout=0.5"]
RUNS = {                          # name: {"cache": builder version, "set": overrides}; FIRST = baseline
    "v2_early": {"cache": 2, "set": P + ["model.input_phases=early"]},
    "v3_early": {"cache": 3, "set": P + ["model.input_phases=early"]},
    "v2_rgb":   {"cache": 2, "set": P + ["model.input_phases=rgb"]},
    "v2_enh":   {"cache": 2, "set": P + ["model.input_phases=enhancement"]},
    "v3_enh":   {"cache": 3, "set": P + ["model.input_phases=enhancement"]},
}
BASELINE = None                   # None = the first entry of RUNS, same seed
CACHE_VERSION = 2                 # used for entries without a "cache" key
DEBUG = False
TIME_BUDGET_H = 11.0              # stop starting new runs after this (Kaggle limit: 12 h)
KEEP_CHECKPOINTS = False          # 10 runs x ~350 MB; set True only if you will reuse the weights
PUSH_RESULTS = True               # commit results/<PERSON>/<sweep id>/ (small files only)
'''),
    *NOTEBOOKS["11_sweep.ipynb"][2:],
]


def main():
    for name, cells in NOTEBOOKS.items():
        (HERE / name).write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
