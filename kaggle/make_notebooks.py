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
CURRENT_TAG = "d1.5"   # bump this whenever a new code tag is made, then regenerate

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
   Training picks the highest builder version automatically if old caches are attached too."""),
        params('''COHORTS = ["ispy1", "ispy2", "duke"]
LIMIT = 0               # 0 = all patients; 10 for a quick check
WORKERS = 4
RAW = {}                # {cohort: path} to override the automatic search for mnj-raw-<cohort>
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
    OUTS[c] = f"/kaggle/working/cache_v2_{c}"
    cmd = [sys.executable, "scripts/build_cache.py", "--cohort", c, "--raw", raws[c], "--out", OUTS[c],
           "--workers", str(WORKERS)] + (["--limit", str(LIMIT)] if LIMIT else [])
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
2. **+ Add Input**: the cache Datasets `mnj-cache-ispy1`, `mnj-cache-ispy2`, `mnj-cache-duke`
   (and, for `f2_from_b3` or export, the weights Dataset `mnj-w-<run_id>`).
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
EXTRA = []              # optional overrides, e.g. ["optim.max_epochs=20"]
PUSH_RESULTS = True     # commit results/<PERSON>/<run_id>/ (small files only) to GitHub
'''),
        code(SETUP),
        code(r'''cmd = [sys.executable, "-m", "src.train", "--config", CONFIG, "--seed", str(SEED), "--mode", MODE,
       "--cache-root", "/kaggle/input", "--out", "/kaggle/working/runs",
       "--results-dir", f"/kaggle/working/results/{PERSON}"]
if DEBUG: cmd.append("--debug")
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
'''),
        code(PUSH.replace("if PUSH_RESULTS:", "if PUSH_RESULTS and not DEBUG:")),
    ],
}


def main():
    for name, cells in NOTEBOOKS.items():
        (HERE / name).write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
