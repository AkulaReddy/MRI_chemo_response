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
CURRENT_TAG = "d1.3"   # bump this whenever a new code tag is made, then regenerate

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
        md("""# 01 Build the slice cache (CPU, no GPU needed)
1. Session options: Accelerator **None**, Internet **ON**, secret `GITHUB_TOKEN` attached.
2. **+ Add Input** -> your raw Dataset (`mnj-raw-ispy1`, `mnj-raw-ispy2` or `mnj-raw-duke`).
3. Set the parameters, then **Save Version -> Save & Run All (Commit)**.
4. When it finishes: check the QC numbers and the grid below, then **New Dataset** from the output,
   named `mnj-cache-<cohort>` (Private, shared with the team). Post the status counts in the chat."""),
        params('''COHORT = "ispy1"       # "ispy1", "ispy2" or "duke"
LIMIT = 0               # 0 = all patients; 10 for a quick check
RAW = ""                # leave "" to find the mnj-raw-<cohort> input automatically
'''),
        code(SETUP),
        code(r'''if not RAW:   # Kaggle may unpack the .tar.gz into a folder; either form works
    want = f"BreastDCEDL_{COHORT.upper()}_min_crop"
    hits = []
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if root.count("/") > 7:
            dirs[:] = []
            continue
        hits += [os.path.join(root, n) for n in dirs + files if n.startswith(want)]
        dirs[:] = [d for d in dirs if not d.startswith("BreastDCEDL_")]
    if not hits:
        for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
            print(root, dirs[:5], files[:5])
            if root.count("/") > 5:
                dirs[:] = []
    assert hits, f"no {want} folder or .tar.gz under /kaggle/input: attach mnj-raw-{COHORT} or set RAW"
    RAW = hits[0] if os.path.isdir(hits[0]) else os.path.dirname(hits[0])
OUT = f"/kaggle/working/cache_{COHORT}"
print("raw:", RAW, os.listdir(RAW))
cmd = [sys.executable, "scripts/build_cache.py", "--cohort", COHORT, "--raw", RAW, "--out", OUT, "--workers", "4"]
if LIMIT:
    cmd += ["--limit", str(LIMIT)]
subprocess.run(cmd, check=True)
'''),
        code(r'''import pandas as pd
from IPython.display import Image, display
idx = pd.read_csv(f"{OUT}/cache_index.csv")
print(json.load(open(f"{OUT}/cache_meta.json")))
print(idx.status.value_counts())
if "z_source" in idx:
    print(idx.z_source.value_counts(dropna=False))
if "xy_source" in idx:
    print(idx.xy_source.value_counts(dropna=False))
    sc = idx[idx.xy_source != "csv"]
    if "mask_csv_iou_yx" in sc and sc.mask_csv_iou_yx.notna().any():
        print("shifted CSV box vs mask IoU:", sc.mask_csv_iou_yx.describe().round(3).to_dict())
if "mask_csv_iou_yx" in idx:
    print("mask vs CSV box IoU:", idx.mask_csv_iou_yx.describe().round(3).to_dict())
display(idx[idx.status == "error"].head(20))
display(Image(f"{OUT}/summary_grid.png"))
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
