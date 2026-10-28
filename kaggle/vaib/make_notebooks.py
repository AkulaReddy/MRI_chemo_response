"""Generates the branch-`vaib` Kaggle notebooks in this folder (the team's kaggle/*.ipynb are untouched).

    python kaggle/vaib/make_notebooks.py

Run order: step1 (x3 cohorts) -> step2 (x3) -> step3 -> step4 -> step5; step0 is an optional quick check.
step0 and step1 are self-contained (no repo, no secret). step2-5 need the repo code, either cloned from GitHub at
branch `vaib` with the GITHUB_TOKEN Kaggle secret, or from an attached code Dataset (zip of the repo; see
kaggle/vaib/README.md), because the repository is private.
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("team_nb", HERE.parent / "make_notebooks.py")
team = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(team)
md, code, notebook = team.md, team.code, team.notebook
BRANCH = "vaib"
ROOT = HERE.parents[1]

SETUP = r'''# ==== SETUP (do not edit): get the repo code, install requirements ====
import base64, json, os, shutil, subprocess, sys
if PERSON not in ("A", "B", "C"):           # only names the results folder
    PERSON = "vaib"
    print('PERSON not set: results go to results/vaib/ (set PERSON = "A", "B" or "C" to change)')
_tok = None
try:
    from kaggle_secrets import UserSecretsClient
    _tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception as e:
    print("no GITHUB_TOKEN secret attached to this notebook (fine if the mnj-code-vaib Dataset is attached):",
          type(e).__name__)

def git(*args, cwd="/tmp/repo"):
    """git with the token as a one-off header (never stored in .git/config or the output)."""
    if not _tok:
        raise RuntimeError("no GITHUB_TOKEN secret: git push/pull is not available")
    hdr = "Authorization: Basic " + base64.b64encode(f"x-access-token:{_tok}".encode()).decode()
    r = subprocess.run(["git", "-c", f"http.extraHeader={hdr}", *args], cwd=cwd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace(_tok, "***")
    if r.returncode:
        raise RuntimeError(f"git {args[0]} failed:\n{out}")
    return out

shutil.rmtree("/tmp/repo", ignore_errors=True)
if _tok:                                    # 1) clone the branch from GitHub
    git("clone", "--quiet", "--depth", "1", "--branch", GIT_TAG, f"https://github.com/{REPO}.git", "/tmp/repo", cwd="/tmp")
    print("code: GitHub", REPO, GIT_TAG, git("log", "-1", "--format=%h %s"))
else:                                       # 2) or a code Dataset (zip of the repo) attached as input
    import glob, zipfile
    hits = [h for d in range(1, 8) for h in glob.glob("/kaggle/input/" + "*/" * d + "src/vitonly/__init__.py")]
    zips = [z for d in range(1, 6) for z in glob.glob("/kaggle/input/" + "*/" * d + "mnj-code-vaib*.zip")]
    if not hits and zips:                   # Kaggle kept the uploaded zip as a file: unpack it
        zipfile.ZipFile(zips[0]).extractall("/tmp/code_zip")
        hits = glob.glob("/tmp/code_zip/**/src/vitonly/__init__.py", recursive=True)
    if not hits:
        print("attached inputs:")
        for root, dirs, files in os.walk("/kaggle/input"):
            if root.count("/") <= 5:
                print("  ", root, dirs[:6], files[:4])
            else:
                dirs[:] = []
        raise SystemExit(
            "\nThe notebook can't find the project code. Do ONE of these, then Run All again:\n"
            "  (a) Add-ons -> Secrets -> add GITHUB_TOKEN (a GitHub token with access to the repo) and switch it ON\n"
            "      for this notebook, or\n"
            "  (b) + Add Input -> Datasets -> Your Datasets -> mnj-code-vaib (made by uploading mnj-code-vaib.zip).")
    code_root = os.path.dirname(os.path.dirname(os.path.dirname(hits[0])))
    shutil.copytree(code_root, "/tmp/repo")
    print("code: attached Dataset", code_root)
os.chdir("/tmp/repo")
sys.path.insert(0, "/tmp/repo")         # so `from src...` works in this kernel
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"], check=True)
import torch; print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
'''


def params(extra):
    return code(f'''# ==== PARAMETERS (edit these) ====
REPO = "{team.REPO_SLUG}"
GIT_TAG = "{BRANCH}"           # branch (or tag) to clone
PERSON = ""               # optional: "A", "B" or "C" -> results/<PERSON>/ (empty -> results/vaib/)
''' + extra)


PUSH = r'''# ==== PUSH the small result files to the vaib branch (results/<PERSON>/...) ====
if PUSH_RESULTS and not _tok:
    print("PUSH_RESULTS needs the GITHUB_TOKEN secret; download results/ from the Output tab instead")
elif PUSH_RESULTS:
    git("config", "user.name", f"Person {PERSON} (Kaggle)")
    git("config", "user.email", "kaggle@users.noreply.github.com")
    for attempt in range(3):
        git("fetch", "--quiet", "--depth", "1", "origin", f"+refs/heads/{GIT_TAG}:refs/remotes/origin/{GIT_TAG}")
        git("checkout", "--quiet", "-f", "-B", GIT_TAG, f"origin/{GIT_TAG}")
        shutil.copytree(f"/kaggle/working/results/{PERSON}", f"results/{PERSON}", dirs_exist_ok=True)
        git("add", f"results/{PERSON}")
        print(git("commit", "-m", f"results: {PERSON} {NOTEBOOK_NAME} ({GIT_TAG})"))
        try:
            git("push", "--quiet", "origin", GIT_TAG); print("pushed to", GIT_TAG); break
        except RuntimeError as e:
            print(f"push attempt {attempt + 1} failed:", e)
    else:
        print("NOT pushed: download results/ from the Output tab")
'''

# --------------------------------------------------------------------------------------------- 00
NB00 = [
    md("""# Step 0 (optional): is BreastDCEDL downloadable? Download, verify, look inside (CPU, ~5 min)
**Self-contained:** no repo and no GitHub secret. Session options: Accelerator **None**, Internet **ON**.

What it shows:
1. The data is open on Zenodo (record 18114231, CC BY-NC 4.0, no login). It downloads with parallel byte ranges,
   and each file is checked against its published md5. Zenodo sometimes returns short or `504` responses, so
   parts are size-checked and resumed.
2. What one patient looks like: one 3D NIfTI per DCE phase, raw float64 scanner values, and the tumour mask.
3. How a 3D scan becomes the 2D RGB slices the ViT sees, and why the original joint 99.5th-percentile clipping
   saturates the tumour.

`WHAT = "demo"` (55 MB, a 7-patient sample) is enough to answer the question. **This is not the training data:**
the full dataset is downloaded by `step1a/1b/1c_download_*.ipynb`."""),
    code('''WHAT = "demo"      # "demo" (55 MB) | "ispy1" (1.25 GB) | "ispy2" (13.6 GB) | "duke" (8.6 GB)
PARTS = 8          # parallel connections
PID = None         # a patient to inspect, or None = first I-SPY patient found
'''),
    code(r'''import hashlib, os, subprocess, sys, tarfile, time
from concurrent.futures import ThreadPoolExecutor
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nibabel"], check=True)
import numpy as np, pandas as pd, nibabel as nib, matplotlib.pyplot as plt

BASE = "https://zenodo.org/records/18114231/files"
FILES = {   # name, exact size (bytes), md5 (from https://zenodo.org/api/records/18114231)
    "meta":  ("BreastDCEDL_metadata_min_crop.csv", 339901, "d58b84d0f343b4b8338cbb222e6dda08"),
    "demo":  ("BreastDCEDL_demo_data.tar.gz", 54838099, "5b7d8701309c2c3807f8d126f7ddcede"),
    "ispy1": ("BreastDCEDL_ISPY1_min_crop.tar.gz", 1252209311, "d2d2630940a861819e6a385877f45ba5"),
    "ispy2": ("BreastDCEDL_ISPY2_min_crop.tar.gz", 13634158199, "2f1a00310bd00d344664319fb7bd679c"),
    "duke":  ("BreastDCEDL_DUKE_min_crop.tar.gz", 8643461864, "16e6c752b9679138e88c188d7a01895f"),
}
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()

def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

def download(key):
    name, size, md5 = FILES[key]
    out = os.path.join(OUT, name)
    if os.path.exists(out) and os.path.getsize(out) == size and md5sum(out) == md5:
        print(f"{name}: already here, md5 OK"); return out
    url, step, t = f"{BASE}/{name}?download=1", -(-size // PARTS), time.time()
    def part(i):
        a, b = i * step, min(size, (i + 1) * step) - 1
        p = f"{out}.part{i}"
        for attempt in range(60):          # resume until the part has exactly the right size
            have = os.path.getsize(p) if os.path.exists(p) else 0
            if have >= b - a + 1:
                return
            subprocess.run(["curl", "-sL", "--fail", "-m", "600", "-r", f"{a + have}-{b}", url], stdout=open(p, "ab"))
            time.sleep(min(30, 2 * attempt))
        raise RuntimeError(f"part {i} did not complete")
    with ThreadPoolExecutor(PARTS) as ex:
        list(ex.map(part, range(PARTS)))
    with open(out, "wb") as f:
        for i in range(PARTS):
            with open(f"{out}.part{i}", "rb") as p:
                while (b := p.read(1 << 24)):
                    f.write(b)
            os.remove(f"{out}.part{i}")
    dt = time.time() - t
    ok = md5sum(out) == md5
    print(f"{name}: {size / 1e6:.1f} MB in {dt:.0f} s = {size / 1e6 / dt:.2f} MB/s, md5 {'OK' if ok else 'MISMATCH'}")
    assert ok, "md5 mismatch: delete the file and rerun"
    return out

META = pd.read_csv(download("meta")).set_index("pid", drop=False)
ARCHIVE = download(WHAT)
'''),
    md("""## The archive: one 3D NIfTI file per DCE phase, plus a tumour mask (I-SPY) per patient
Only the three phases used for the model are shipped: pre-contrast, early and late post-contrast. Their indices
are the CSV columns `pre`, `post_early` and `post_late`."""),
    code(r'''import re
DATA = os.path.join(OUT, "extracted"); os.makedirs(DATA, exist_ok=True)
with tarfile.open(ARCHIVE) as tf:
    members = [m for m in tf.getmembers() if m.name.endswith(".nii.gz")]
    if WHAT == "demo" or not os.listdir(DATA):
        tf.extractall(DATA, members=members if WHAT == "demo" else members[:60])   # full cohorts: a few patients
rx = re.compile(r"^(?P<pid>.+?)_(?P<ds>spy1|spy2|duke)(?:_vis\d)?(?:_dce)?_a(?:qc|cq)_?(?P<k>\d+)\.nii\.gz$")
rows = []
for root, _, files in os.walk(DATA):
    for f in files:
        m = rx.match(f)
        if m: rows.append({"pid": m["pid"], "cohort": m["ds"], "phase": int(m["k"]), "file": os.path.join(root, f)})
        elif f.endswith("_mask.nii.gz"): rows.append({"pid": f.split("_spy")[0], "cohort": "mask", "phase": -1, "file": os.path.join(root, f)})
files = pd.DataFrame(rows)
print(f"{len(members)} NIfTI files in the archive; extracted {len(files)}")
display(files[files.phase >= 0].groupby(["cohort", "pid"]).phase.apply(sorted).to_frame("phases shipped"))
cols = ["dataset", "pCR", "HR", "HER2", "age", "n_xy", "n_z", "xy_spacing", "slice_thick", "pre", "post_early", "post_late",
        "mask_start", "mask_end", "sraw", "eraw", "scol", "ecol", "test"]
display(META.loc[[p for p in files.pid.unique() if p in META.index], cols])
'''),
    md("""## One patient: shapes, raw values, real voxel size
- Arrays are `(slices, 256, 256)` float64 with raw scanner values. Nothing is normalised.
- The NIfTI header spacing says 1 mm, which is wrong. The real in-plane pixel size (`xy_spacing`) and slice
  thickness (`slice_thick`) come from the CSV.
- The `test` column is the split: 0 train, 2 validation, 1 test."""),
    code(r'''pids = [p for p in files[files.cohort.isin(["spy1", "spy2"])].pid.unique()] or list(files.pid.unique())
PID = PID or pids[0]
r = META.loc[PID]
ph = files[(files.pid == PID) & (files.phase >= 0)].set_index("phase").file
use = [int(r.pre), int(r.post_early), int(r.post_late)]
use = [k if k in ph.index else sorted(ph.index)[min(i, len(ph) - 1)] for i, k in enumerate(use)]
vols = [nib.load(ph[k]).get_fdata(dtype=np.float32) for k in use]
mf = files[(files.pid == PID) & (files.phase == -1)].file
mask = nib.load(mf.iloc[0]).get_fdata().astype(np.uint8) if len(mf) else None
img = nib.load(ph[use[0]])
print(f"{PID}: pCR = {r.pCR}, split = {({0: 'train', 1: 'test', 2: 'val'})[int(r.test)]}")
print("shape (slices, rows, cols):", vols[0].shape, "| dtype on disk:", img.header.get_data_dtype(),
      "| header spacing:", img.header.get_zooms(), "<- wrong; CSV:", r.xy_spacing, "mm in-plane,", r.slice_thick, "mm slices")
for name, v in zip(("pre", "early", "late"), vols):
    print(f"  {name:5s} min {v.min():8.1f}  median {np.median(v):8.1f}  99.5th pct {np.percentile(v, 99.5):8.1f}  max {v.max():8.1f}")
if mask is not None:
    z = np.flatnonzero(mask.any((1, 2)))
    z0, z1 = z.min(), z.max() + 1
    yy, xx = np.nonzero(mask.any(0))
    box = (yy.min(), yy.max() + 1, xx.min(), xx.max() + 1)
else:   # Duke: boxes only. The CSV box is in the original n_xy image (see challenges.md); fine when n_xy == 256
    z0, z1, box = int(r.mask_start), int(r.mask_end), (int(r.sraw), int(r.eraw), int(r.scol), int(r.ecol))
print(f"tumour: slices {z0}-{z1} of {vols[0].shape[0]}, box rows {box[0]}-{box[1]} cols {box[2]}-{box[3]}")
'''),
    md("""## The 3D volume is a stack of 2D slices. The tumour spans a range of them (red titles)."""),
    code(r'''zs = np.unique(np.linspace(0, vols[1].shape[0] - 1, 8).round().astype(int))
fig, ax = plt.subplots(1, len(zs), figsize=(2.2 * len(zs), 2.6))
for a, z in zip(ax, zs):
    a.imshow(vols[1][z], cmap="gray", vmax=np.percentile(vols[1], 99.5)); a.axis("off")
    a.set_title(f"slice {z}", color="red" if z0 <= z < z1 else "black", fontsize=8)
plt.suptitle("early post-contrast volume"); plt.tight_layout(); plt.show()

zm = (z0 + z1) // 2
fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
for a, v, n in zip(ax, vols, ("pre-contrast", "early post", "late post")):
    a.imshow(v[zm], cmap="gray", vmax=np.percentile(vols[1], 99.5)); a.set_title(n); a.axis("off")
    if mask is not None: a.contour(mask[zm], levels=[0.5], colors="r", linewidths=0.6)
sel = mask[z0:z1] > 0 if mask is not None else np.s_[:, box[0]:box[1], box[2]:box[3]]
curve = [v[z0:z1][sel].mean() for v in vols]
ax[3].plot(["pre", "early", "late"], curve, "o-"); ax[3].set_title("mean tumour intensity (DCE curve)")
plt.tight_layout(); plt.show()
print("enhancement: early/pre = %.2f, late/early = %.2f (<1 washout, ~1 plateau, >1 persistent)" % (curve[1] / curve[0], curve[2] / curve[1]))
'''),
    md("""## 3D -> 2D RGB, as the ViT pipeline does it, and the saturation problem
The original (v1) pipeline puts R = pre, G = early and B = late. It clips all three phases **jointly** at the
volume's 0.5 / 99.5 percentiles and stores the result as uint8. Each tumour slice becomes one 2D RGB image, which
is cropped to 240 px around the tumour box (224 px at the network input).

Enhancing tumour is among the brightest tissue in a tumour-centred crop, so a large share of it sits **above**
the 99.5th percentile and is stored as 255. Washout vs persistent enhancement (G vs B) and internal texture are
then lost. On the demo archive, 48-96% of the I-SPY1 tumour-mask voxels were clipped this way."""),
    code(r'''stack = np.stack(vols, -1)
lo, hi = np.percentile(stack.reshape(-1)[::4], [0.5, 99.5])
rgb = np.clip((stack - lo) / (hi - lo), 0, 1)
tum = stack[z0:z1][sel] if mask is not None else stack[z0:z1, box[0]:box[1], box[2]:box[3]].reshape(-1, 3)
print("tumour voxels clipped at the v1 ceiling: pre %.0f%%, early %.0f%%, late %.0f%%" % tuple(100 * (tum > hi).mean(0)))
cy, cx = (box[0] + box[1]) // 2, (box[2] + box[3]) // 2
def win(a, side):
    y0 = int(np.clip(cy - side // 2, 0, a.shape[0] - side)); x0 = int(np.clip(cx - side // 2, 0, a.shape[1] - side))
    return a[y0:y0 + side, x0:x0 + side]
side_t = int(np.clip(1.6 * max(box[1] - box[0], box[3] - box[2]), 64, 256))
p999 = np.percentile(stack.reshape(-1)[::4], 99.9)
fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
ax[0].imshow(rgb[zm]); ax[0].set_title("RGB = pre/early/late, v1 clip")
ax[1].imshow(win(rgb[zm], 224)); ax[1].set_title("224 px around the tumour (v1 input)")
ax[2].imshow(win(rgb[zm], side_t)); ax[2].set_title(f"tumour zoom ({side_t} px -> 224)")
ax[3].imshow(win(np.clip((stack[zm] - lo) / (p999 - lo), 0, 1), side_t)); ax[3].set_title("zoom, 99.9th pct ceiling")
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()
print(f"{z1 - z0} tumour slices -> {z1 - z0} 2D RGB images for this patient; the patient's label (pCR) is shared by all of them")
'''),
]

# --------------------------------------------------------------------------------------------- step 1
_DL = (ROOT / "scripts" / "download_breastdcedl.py").read_text().split('\nif __name__ == "__main__":')[0]
NB_STEP1 = [
    md("""# Step 1: download the FULL BreastDCEDL dataset into Kaggle (CPU, no GPU, no secret)
All 2,070 patients from Zenodo record 18114231 (CC BY-NC 4.0, no login): I-SPY1 172, I-SPY2 982, Duke 916.
A Kaggle notebook can save at most 20 GB of output, and the full set is 23.85 GB. So run this notebook
**three times**, once per cohort; the three Datasets attached together are the full dataset.

| run | `COHORT` | files | size | time (~1-3 MB/s) | save the output as Dataset |
|---|---|---|---|---|---|
| 1 | `"ispy1"` | I-SPY1 archive + metadata CSV + authors' ViT weights | 1.6 GB | ~15-30 min | **`mnj-raw-ispy1`** |
| 2 | `"ispy2"` | I-SPY2 archive (982 patients) | 13.6 GB | ~2-4 h | **`mnj-raw-ispy2`** |
| 3 | `"duke"` | Duke archive (916 patients) | 8.6 GB | ~1.5-3 h | **`mnj-raw-duke`** |

How to run:
1. Kaggle -> **Create -> New Notebook -> File -> Import Notebook** -> upload this `.ipynb`.
2. Session options (right panel): **Accelerator None**, **Internet ON** (needs a phone-verified account).
3. Set `COHORT` below, then **Save Version -> Save & Run All (Commit)**. It keeps running with the browser closed.
4. When the version has finished: open it -> **Output** tab -> check the log ends with `ALL MD5 OK` ->
   **New Dataset** -> name from the table -> Private. You can do the three runs in parallel.
5. If a run stops at Kaggle's 12 h limit, run it again. Finished files are skipped, but parts are not kept
   between sessions, so a very slow I-SPY2 run starts over. That is rare at Kaggle speeds.

If your teammates already made `mnj-raw-ispy1/-ispy2/-duke`, ask them to share those instead (same files)."""),
    code('''COHORT = "ispy1"     # "ispy1", then "ispy2", then "duke" (one per run)
PARTS = 8            # parallel connections to Zenodo
'''),
    code(_DL),
    code('''files = {"ispy1": ["ispy1", "meta", "models"], "ispy2": ["ispy2"], "duke": ["duke"]}[COHORT]
ok = main(["--out", "/kaggle/working" if os.path.isdir("/kaggle/working") else "BreastDCEDL", "--files", *files,
           "--parts", str(PARTS)])
assert ok, "some files are incomplete: run the notebook again"
print(f"next: save this output as the Dataset mnj-raw-{COHORT}")
'''),
]

# --------------------------------------------------------------------------------------------- 01
NB01 = [
    md("""# Step 3: input diagnostics on the full dataset: what does the ViT actually see? (CPU, ~15 min)
1. Session options: Accelerator **None**, Internet **ON**, secret `GITHUB_TOKEN` (or attach the code Dataset).
2. **+ Add Input**: `mnj-cache2-ispy1`, `mnj-cache2-ispy2`, `mnj-cache2-duke` (from step 2). The team's v1 caches
   `mnj-cache-*` also work, and attaching both lets you compare.

Measures, on **training patients only**:
- **Saturation:** how much of the tumour is clipped to white by the v1 joint 99.5th-percentile scaling, and by
  each alternative normalisation (v2 caches are exact; on v1 the clipped part is already gone).
- **Tumour size in the input:** the share of the 224 px input width that the tumour box takes, per crop.
- **Gallery:** the same slices under every normalisation, pCR vs non-pCR.
- **Is there signal at all?** Hand-crafted tumour enhancement features (no deep learning) -> logistic regression:
  5-fold CV AUC on train and AUC on val. The `cohort_only` row is the AUC you get for free from cohort pCR rates."""),
    params('''N_PER_COHORT = 80        # training patients per cohort for the saturation / size statistics
PREFER = "v2"            # cache format to use when both are attached
'''),
    code(SETUP),
    code(r'''import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from src.data.metadata import patient_table
from src.vitonly.slabs import find_caches, load_slab
from src.vitonly.norms import NORMS, CROPS
from src.vitonly.diagnostics import saturation, v1_clipped, tumor_fraction, gallery
from src.vitonly.kinetics import kinetic_features, FEATURES
from src.vitonly.probe import score
caches = find_caches(["/kaggle/input", "/kaggle/working"], PREFER)
fmts = {c: json.load(open(Path(d) / "cache_meta.json")).get("format", "v1") for c, d in caches.items()}
print({c: (d, fmts[c]) for c, d in caches.items()})
cohorts = [c for c in ("ispy1", "ispy2", "duke") if c in caches]
train = patient_table(["train"], cohorts, cache_dirs=caches)
val = patient_table(["val"], cohorts, cache_dirs=caches)
print(f"train {len(train)}, val {len(val)}")
'''),
    md("## 1. Saturation inside the tumour and tumour size in the input"),
    code(r'''rows = []
for c in cohorts:
    for r in train[train.cohort == c].sample(min(N_PER_COHORT, (train.cohort == c).sum()), random_state=0).itertuples():
        s = load_slab(Path(caches[c]) / f"{r.pid}.npz")
        sat = saturation(s)
        rows.append({"cohort": c, "pid": r.pid, "pCR": r.pCR, "v1_clipped": float(v1_clipped(s)[1:].max()),
                     **{n: float(sat[n][1:].max()) for n in NORMS},
                     **{f"size_{k}": v for k, v in tumor_fraction(s, r.xy_spacing).items()}})
diag = pd.DataFrame(rows)
print("fraction of tumour pixels saturated (worst of early/late channel), mean per cohort; cache:", fmts)
display(diag.groupby("cohort")[["v1_clipped", *NORMS]].mean().round(3))
print("share of the 224 px input width taken by the tumour box, mean per cohort")
display(diag.groupby("cohort")[[f"size_{k}" for k in CROPS]].mean().round(2))
fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
diag.groupby("cohort")[["v1_clipped", *NORMS]].mean().T.plot.bar(ax=ax[0]); ax[0].set_title("tumour pixels saturated")
diag.groupby("cohort")[[f"size_{k}" for k in CROPS]].mean().T.plot.bar(ax=ax[1]); ax[1].set_title("tumour share of input width")
plt.tight_layout(); plt.show()
if "v1" in fmts.values():
    print("NOTE: on v1 caches only `v1_clipped` measures the loss; other columns can't see what v1 already clipped.")
'''),
    md("## 2. Gallery: the same middle-tumour slice under every normalisation (tumour crop, mask in white)"),
    code(r'''for c in cohorts:
    t = train[train.cohort == c]
    pick = pd.concat([g.sample(min(2, len(g)), random_state=1) for _, g in t.groupby("pCR")])
    fig, axes = plt.subplots(len(pick), len(NORMS), figsize=(1.9 * len(NORMS), 2.0 * len(pick)), squeeze=False)
    for row, r in zip(axes, pick.itertuples()):
        gallery(row, load_slab(Path(caches[c]) / f"{r.pid}.npz"), r.xy_spacing, NORMS, "tumor")
        row[0].set_title(f"{r.pid} pCR={int(r.pCR)}", fontsize=7, loc="left")
    for a, n in zip(axes[-1], NORMS): a.set_xlabel(n); a.axis("on"); a.set_xticks([]); a.set_yticks([])
    plt.suptitle(f"{c} ({fmts[c]} cache): " + " | ".join(NORMS), fontsize=8); plt.tight_layout(); plt.show()
'''),
    md("""## 3. Is there pCR signal in the images without deep learning?
Tumour ROI = voxels in the tumour box with percent enhancement >= 70% (the I-SPY functional tumour volume rule),
the same rule for every cohort. Features: functional tumour volume, PE and SER statistics, washout/plateau/
persistent fractions, relative intensities (16 in total)."""),
    code(r'''table = pd.concat([train, val]).reset_index(drop=True)
is_train = np.r_[np.ones(len(train), bool), np.zeros(len(val), bool)]
X = np.stack([kinetic_features(load_slab(Path(caches[r.cohort]) / f"{r.pid}.npz"), r.xy_spacing, r.slice_thick)
              for r in table.itertuples()])
y, coh = table.pCR.to_numpy(float), table.cohort.to_numpy(str)
res = {}
for name, F in (("kinetic_handcrafted", np.nan_to_num(X)), ("cohort_only", pd.get_dummies(pd.Series(coh)).to_numpy(float)),
                ("kinetic + cohort", np.c_[np.nan_to_num(X), pd.get_dummies(pd.Series(coh)).to_numpy(float)])):
    res[name] = score(F, y, coh, is_train)[0]
display(pd.DataFrame(res).T[["cv_auc", "cv_auc_sd", "val_auc", "val_ci_lo", "val_ci_hi", "val_auc_within",
                             *[f"val_auc_{c}" for c in cohorts]]].round(3))
from sklearn.metrics import roc_auc_score
tr = is_train
print("single-feature AUC on train (>0.5: higher in pCR):")
display(pd.Series({f: roc_auc_score(y[tr], np.nan_to_num(X[tr, j])) for j, f in enumerate(FEATURES)}).round(3).sort_values())
print("Reference: the old B3 ViT reached val AUC 0.530; clinical-only LR (B0) 0.639.")
'''),
]

# --------------------------------------------------------------------------------------------- 02
NB02 = [
    md("""# Step 2: build the training cache from the full raw dataset (CPU, ~20-60 min per cohort, run x3)
Why: the v1 cache stored uint8 clipped at the volume's 99.5th percentile, which saturates much of the tumour,
and it fixes one normalisation. v2 stores the raw values (float16) of the tumour slices, the full 256 plane and
whole-volume statistics. Every normalisation can then be chosen at training time without a rebuild. Only
pCR-labeled patients (1,452 of 2,070) are written by default. The other 618, mostly Duke, have no pCR outcome,
so they can be neither trained on nor scored; `ALL_PATIENTS = True` includes them anyway.

1. Session options: Accelerator **None**, Internet **ON**, secret `GITHUB_TOKEN` (or attach the code Dataset).
2. **+ Add Input** -> the raw Dataset of the cohort from step 1 (`mnj-raw-ispy1`, `mnj-raw-ispy2` or `mnj-raw-duke`;
   `breastdcedl-full` if you uploaded everything as one Dataset).
3. Set `COHORT`, then **Save Version -> Save & Run All (Commit)**. Run it three times (ispy1, ispy2, duke).
4. When it finishes: **New Dataset** from the output, named **`mnj-cache2-<cohort>`** (Private).
   Measured size per cohort (labeled / all): I-SPY1 1.1 / 1.1 GB, I-SPY2 8.2 / 8.2 GB, Duke 2.1 / 5.2 GB."""),
    params('''COHORT = "ispy1"        # "ispy1", "ispy2" or "duke"
LIMIT = 0                # 0 = every patient (full cohort); 10 only for a quick check
ALL_PATIENTS = False     # True: also the patients without a pCR label (Duke 916 instead of 298)
RAW = ""                 # leave "" to find the BreastDCEDL_<COHORT>_min_crop input automatically
'''),
    code(SETUP),
    code(r'''if not RAW:
    want = f"BreastDCEDL_{COHORT.upper()}_min_crop"
    hits = []
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if root.count("/") > 7:
            dirs[:] = []; continue
        hits += [os.path.join(root, n) for n in dirs + files if n.startswith(want)]
        dirs[:] = [d for d in dirs if not d.startswith("BreastDCEDL_")]
    assert hits, f"no {want} under /kaggle/input: attach mnj-raw-{COHORT} or set RAW"
    RAW = hits[0] if os.path.isdir(hits[0]) else os.path.dirname(hits[0])
OUT = f"/kaggle/working/cache2_{COHORT}"
cmd = [sys.executable, "scripts/build_cache_v2.py", "--cohort", COHORT, "--raw", RAW, "--out", OUT, "--workers", "4"]
if LIMIT: cmd += ["--limit", str(LIMIT)]
if ALL_PATIENTS: cmd.append("--all-patients")
print(" ".join(cmd)); subprocess.run(cmd, check=True)
'''),
    code(r'''import pandas as pd
idx = pd.read_csv(f"{OUT}/cache_index.csv")
print(json.dumps(json.load(open(f"{OUT}/cache_meta.json")), indent=1))
ok = idx[idx.status == "ok"]
print("tumour-box share clipped by the v1 scaling (per phase):")
display(ok[["sat_pre", "sat_early", "sat_late"]].describe().round(3))
display(idx[idx.status == "error"].head(20))
'''),
]

# --------------------------------------------------------------------------------------------- 03
NB03 = [
    md("""# Step 4: frozen-feature probe: which normalisation x crop x backbone carries pCR signal? (GPU, ~30 min)
Fine-tuning one ViT costs ~30 min and its val AUC is noisy (177 patients, 95% CI about +-0.08). So first,
every input variant goes through **frozen** backbones. Each patient's slice features are averaged and an L2
logistic regression is fitted. Variants are ranked by **5-fold CV AUC on the ~1,100 training patients** (low
variance), and val AUC is shown as a confirmation. The best rows tell step 5 what to fine-tune.

1. Session options: **GPU T4 x2** (backbones are spread over both GPUs), Internet **ON**, secret `GITHUB_TOKEN`
   (or attach the code Dataset).
2. **+ Add Input**: `mnj-cache2-ispy1`, `mnj-cache2-ispy2`, `mnj-cache2-duke` (all three = all labeled patients).
3. First run: `LIMIT = 60` (a few minutes) to check everything works, then `LIMIT = 0`, Save & Run All."""),
    params('''BACKBONES = ["mae", "dinov2_b"]   # also: "vit_sup", "dino_b16", "dinov2_s"
NORMS = ["joint_pct", "joint_wide", "per_phase", "joint_tumor", "pre_ref", "zscore", "sub", "kinetic", "early3"]
CROPS = ["fixed", "tumor"]          # also: "mm"
K = 8                               # slices per patient (central 60% of the tumour)
PREFER = "v2"
LIMIT = 0                           # patients per split; 60 for a quick check
PUSH_RESULTS = False
NOTEBOOK_NAME = "probe"
'''),
    code(SETUP),
    code(r'''import time
OUTD = f"/kaggle/working/probe_{time.strftime('%m%d-%H%M')}"
cmd = [sys.executable, "-m", "src.vitonly.probe", "--backbones", *BACKBONES, "--norms", *NORMS, "--crops", *CROPS,
       "--k", str(K), "--prefer", PREFER, "--out", OUTD, "--cache-root", "/kaggle/input", "/kaggle/working"]
if LIMIT: cmd += ["--limit", str(LIMIT)]
print(" ".join(cmd)); subprocess.run(cmd, check=True)
'''),
    code(r'''import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv(f"{OUTD}/probe_results.csv")
cols = ["backbone", "norm", "crop", "pool", "cv_auc", "cv_auc_sd", "val_auc", "val_ci_lo", "val_ci_hi", "val_auc_within"]
cols += [c for c in df.columns if c.startswith("val_auc_") and c != "val_auc_within"]
pd.set_option("display.max_rows", 300, "display.width", 220)
display(df[cols].round(3).head(30))
ref = df[df["pool"].isin(["cohort_only", "kinetic_handcrafted"])]
display(ref[cols].round(3))
best = df[df.backbone != "-"].groupby(["norm", "crop"]).cv_auc.max().unstack()
best.plot.barh(figsize=(8, 5), xlim=(0.45, max(0.75, best.max().max() + 0.02)))
plt.axvline(float(ref[ref["pool"] == "cohort_only"].cv_auc.iloc[0]), color="grey", ls="--", label="cohort only")
plt.xlabel("best 5-fold CV AUC on train"); plt.legend(); plt.tight_layout(); plt.show()
top = df[df.backbone != "-"].iloc[0]
print(f"TOP: backbone={top.backbone} norm={top.norm} crop={top.crop} pool={top['pool']} "
      f"cv {top.cv_auc:.3f}, val {top.val_auc:.3f} -> use in step 5")
os.makedirs(f"/kaggle/working/results/{PERSON}/{os.path.basename(OUTD)}", exist_ok=True)
for f in ("probe_results.csv", "probe_val_preds.csv"):
    shutil.copy(f"{OUTD}/{f}", f"/kaggle/working/results/{PERSON}/{os.path.basename(OUTD)}/{f}")
'''),
    code(PUSH),
]

# --------------------------------------------------------------------------------------------- 04
NB04 = [
    md("""# Step 5: fine-tune the improved ViT image-only model (GPU, ~30 min per seed)
Uses the best input from step 4. Changes from B3 (val AUC 0.530), all in `configs/vo_base.yaml`:
- **v2 cache:** the tumour is not clipped (in v1, roughly half of the I-SPY tumour was pure white).
- **Normalisation** (`NORM`) and **tumour-zoom crop**: the tumour fills ~60% of the input instead of ~20-40%.
- **Central tumour slices** instead of every slice down to edge slivers.
- **Patch-token mean pooling** instead of MAE's untrained [CLS] token.
- **LP-FT:** 2 epochs head-only, then the top 4 blocks unfreeze with their own warm-up.
- **Plain BCE** loss, and several seeds averaged (an ensemble), because one val AUC on 177 patients is noisy.

`RUN_CONTROL = True` also runs `vo_b3_repro`, the old B3 recipe through this code. It should land near 0.53,
which shows that any gain comes from the changes rather than the new code.

1. Session options: **GPU T4 x2 or P100**, Internet **ON**, secret `GITHUB_TOKEN` (or the code Dataset);
   attach `mnj-cache2-ispy1`, `mnj-cache2-ispy2`, `mnj-cache2-duke`.
2. First `DEBUG = True` (2 epochs, 48 patients), then the real run: **Save & Run All (Commit)**."""),
    params('''CONFIG = "vo_base"          # vo_base | vo_dinov2 | vo_b3_repro
NORM = "kinetic"            # from step 4's top rows
CROP = "tumor"
BACKBONE = "mae"            # mae | dinov2_b | vit_sup | dino_b16 | dinov2_s
SEEDS = [0, 1, 2]
EXTRA = []                  # more overrides, e.g. ["model.train_blocks=6", "optim.lr_backbone=2.0e-5"]
RUN_CONTROL = False         # also run vo_b3_repro (seed 0)
DEBUG = False
PUSH_RESULTS = False
NOTEBOOK_NAME = "finetune"
'''),
    code(SETUP),
    code(r'''import glob, pandas as pd
def run(config, seed, sets):
    cmd = [sys.executable, "-m", "src.vitonly.finetune", "--config", config, "--seed", str(seed),
           "--cache-root", "/kaggle/input", "/kaggle/working", "--out", "/kaggle/working/runs",
           "--results-dir", f"/kaggle/working/results/{PERSON}", "--set", *sets]
    if DEBUG: cmd.append("--debug")
    print(" ".join(cmd)); subprocess.run(cmd, check=True)
    d = sorted(glob.glob("/kaggle/working/runs/*/"), key=os.path.getmtime)[-1]
    return d, json.load(open(f"{d}/metrics.json"))

sets = [f"data.norm={NORM}", f"data.crop={CROP}", f"model.backbone={BACKBONE}", *EXTRA]
runs = [run(CONFIG, s, sets) for s in SEEDS]
if RUN_CONTROL:
    runs.append(run("vo_b3_repro", 0, ["model.backbone=mae"]))
'''),
    code(r'''from src import stats
rows = [{"run": os.path.basename(d.rstrip("/")), "val_auc": m["val"]["auc"], "ci": m["val"]["auc_ci"],
         "within_cohort": m["val"]["auc_within"], **{f"{c}": v["auc"] for c, v in m["val"]["by_cohort"].items()},
         "best_epoch": m["best_epoch"], "gpu_h": m["gpu_hours"]} for d, m in runs]
display(pd.DataFrame(rows).round(3))
main_runs = [d for d, m in runs[:len(SEEDS)]]
p = pd.concat([pd.read_csv(f"{d}/val_preds.csv").set_index("pid") for d in main_runs], axis=1, keys=range(len(main_runs)))
ens = pd.DataFrame({"label": p[0].label, "cohort": p[0].cohort, "prob": p.xs("prob", axis=1, level=1).mean(1)})
s = stats.summary(ens.label, ens.prob)
by = {c: stats.auc(g.label, g.prob) for c, g in ens.groupby("cohort")}
print(f"ENSEMBLE of {len(main_runs)} seeds: val AUC {s['auc']:.3f} (95% CI {s['auc_ci'][0]:.3f}-{s['auc_ci'][1]:.3f}), "
      f"AUPRC {s['auprc']:.3f} | " + " ".join(f"{c} {v:.3f}" for c, v in by.items()))
print("references (val): old B3 ViT 0.530 | clinical LR B0 0.639 | clinical + size B1 0.653-0.660 | F2 fusion 0.690")
tag = f"vo_{BACKBONE}_{NORM}_{CROP}_ensemble"
os.makedirs(f"/kaggle/working/results/{PERSON}/{tag}", exist_ok=True)
ens.to_csv(f"/kaggle/working/results/{PERSON}/{tag}/val_preds.csv")
json.dump({"runs": [os.path.basename(d.rstrip("/")) for d in main_runs], "val": s, "by_cohort": by},
          open(f"/kaggle/working/results/{PERSON}/{tag}/metrics.json", "w"), indent=2, default=float)
'''),
    code(PUSH.replace("if PUSH_RESULTS:", "if PUSH_RESULTS and not DEBUG:")),
]

# ---------------------------------------------------------------------------------- steps 6-8 (update 2)
HELPERS = r'''# ==== HELPERS (do not edit): training runs, prediction files, comparisons ====
import glob, time
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src import evaluation as ev, stats
from src.train import find_caches
from src.data.metadata import patient_table
COH = ["ispy1", "ispy2", "duke"]
OUT = f"/kaggle/working/results/{PERSON}/{STEP}"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
RUNS_JSON = {}
pd.set_option("display.width", 250, "display.max_columns", 40)

def budget_left():
    return TIME_BUDGET_H - (time.time() - T0) / 3600

def say(msg):
    """Timestamped progress line (elapsed and remaining budget), flushed so it shows in the Kaggle log at once."""
    el = (time.time() - T0) / 3600
    print(f"[{time.strftime('%H:%M:%S')} | {el:.2f} h elapsed, {budget_left():.2f} h left] {msg}", flush=True)

def run_cmd(cmd):
    """Run a training subprocess and stream its output line by line into this notebook (subprocess.run alone
    shows nothing until it ends, and the child's stdout is block-buffered without PYTHONUNBUFFERED)."""
    print("\n" + " ".join(cmd), flush=True)
    t = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise subprocess.CalledProcessError(p.returncode, cmd)
    say(f"finished in {(time.time() - t) / 60:.1f} min")

def check_caches(version):
    caches = find_caches(["/kaggle/input", "/kaggle/working"])
    info = {c: json.load(open(f"{d}/cache_meta.json")).get("builder_version", 1) for c, d in caches.items()}
    print("caches:", {c: f"builder v{v}: {caches[c]}" for c, v in info.items()})
    if version is not None:
        bad = {c: v for c, v in info.items() if v != version}
        assert not bad and len(info) == 3, (f"need cache builder v{version} for all 3 cohorts (attach mnj-cache-v2); "
                                            f"found {info}")
    return caches

def _copy(d, label):
    os.makedirs(f"{OUT}/{label}", exist_ok=True)
    for f in ("val_preds.csv", "metrics.json", "log.csv", "config_resolved.yaml", "export_preds.csv",
              "train_preds.csv", "encoder.json"):
        if os.path.exists(f"{d}/{f}"):
            shutil.copy(f"{d}/{f}", f"{OUT}/{label}/{f}")

def train(config, seed, sets=(), label=None, cohorts=None):
    """One src.train run (B3 / F2 ...) -> (run_dir, metrics). Its small files are copied to OUT/<label>/."""
    cmd = [sys.executable, "-m", "src.train", "--config", config, "--seed", str(seed), "--out", "/kaggle/working/runs"]
    if CACHE_VERSION is not None: cmd += ["--cache-version", str(CACHE_VERSION)]
    if cohorts: cmd += ["--cohorts", *cohorts]
    if DEBUG: cmd.append("--debug")
    if sets: cmd += ["--set", *sets]
    run_cmd(cmd)
    d = max(glob.glob("/kaggle/working/runs/*/"), key=os.path.getmtime).rstrip("/")
    if label: _copy(d, label)
    return d, json.load(open(f"{d}/metrics.json"))

def finetune(config, seed, sets=(), label=None, cohorts=None):
    """One src.vitonly.finetune run (our pipeline: ViT-only vo_base / fusion vo_fusion) -> run_dir."""
    cmd = [sys.executable, "-m", "src.vitonly.finetune", "--config", config, "--seed", str(seed),
           "--cache-root", "/kaggle/input", "/kaggle/working", "--out", "/kaggle/working/runs"]
    if cohorts: cmd += ["--cohorts", *cohorts]
    if DEBUG: cmd.append("--debug")
    if sets: cmd += ["--set", *sets]
    run_cmd(cmd)
    d = max(glob.glob("/kaggle/working/runs/*/"), key=os.path.getmtime).rstrip("/")
    if label: _copy(d, label)
    return d

def note_runs(key, dirs):
    """label -> run folder names, in OUT/runs.json: step 8 uses it to find the checkpoints."""
    RUNS_JSON[key] = [os.path.basename(d) for d in dirs]
    json.dump(RUNS_JSON, open(f"{OUT}/runs.json", "w"), indent=1)

def preds_of(dirs):
    fr = [ev.load_preds(f"{d}/val_preds.csv") for d in dirs]
    return fr[0] if len(fr) == 1 else ev.ensemble(fr)

def save_preds(label, df):
    os.makedirs(f"{OUT}/{label}", exist_ok=True)
    df.to_csv(f"{OUT}/{label}/val_preds.csv", index=False)
    return df

def repo_preds(pattern):
    hits = sorted(glob.glob(f"results/*/{pattern}/val_preds.csv"))
    return preds_of([os.path.dirname(h) for h in hits]) if hits else None

def train_auc(d):
    """AUC on the training patients from export_preds.csv (B3 exports after training): the overfitting gap."""
    f = f"{d}/export_preds.csv"
    if not os.path.exists(f): return np.nan
    e = pd.read_csv(f); e = e[e.split == "train"]
    return stats.auc(e.label, e.prob)

def vo_train_auc(d):
    """Training-set AUC of a src.vitonly run (metrics.json): the overfitting gap."""
    return json.load(open(f"{d}/metrics.json"))["val"].get("train_auc", np.nan)

def compare(preds, ref):
    """One row per model on the patients all models share: AUC (DeLong CI), AUPRC, Brier, ECE, the 90%-spec
    operating point (threshold chosen on these same val patients: optimistic), and the paired DeLong test
    and paired bootstrap of the AUC difference vs `ref`."""
    preds = ev.align({k: v for k, v in preds.items() if v is not None})
    rows = []
    for k, d in preds.items():
        r = {"model": k, **ev.summary_row(d.label, d.prob)}
        if k != ref and ref in preds:
            y = preds[ref].label
            dl = ev.delong(y, preds[ref].prob, d.prob)
            pb = stats.paired_bootstrap_auc_diff(y, preds[ref].prob, d.prob)
            r.update(diff_vs_ref=dl["diff"], delong_ci_lo=dl["ci"][0], delong_ci_hi=dl["ci"][1], delong_p=dl["p"],
                     boot_p=pb["p"])
        rows.append(r)
    cols = ["model", "n", "pcr", "auc", "ci_lo", "ci_hi", "auprc", "brier", "ece", "sens@90spec", "ppv@90spec",
            "npv@90spec", "diff_vs_ref", "delong_ci_lo", "delong_ci_hi", "delong_p", "boot_p"]
    t = pd.DataFrame(rows)
    return t[[c for c in cols if c in t]]

def subtype_wide(preds, by="subtype"):
    """AUC (95% CI) per subtype (or cohort) for every model, all on the same patients; n / pCR+ per group."""
    preds = ev.align({k: v for k, v in preds.items() if v is not None})
    tabs = {k: ev.subgroup_table(d, by).set_index(by) for k, d in preds.items()}
    first = next(iter(tabs.values()))
    out = pd.DataFrame({"n": first.n.astype(int), "pCR+": first.pcr.astype(int)})
    for k, t in tabs.items():
        out[k] = [f"{a:.3f} ({lo:.2f}-{hi:.2f})" if a == a else "n/a (one class)"
                  for a, lo, hi in zip(t.auc, t.ci_lo, t.ci_hi)]
    return out

B1_VAL = repo_preds("b1_lr_s0_*")                       # clinical + tumour size, logistic regression (val 0.653)
B0_VAL = repo_preds("b0_lr_s0_*")                       # clinical only (val 0.639)
AKULA_B3_V2 = repo_preds("b3_vit_s0_1019-0302_*")       # Akula's B3 on cache v2 (val 0.559)
B3_V1 = repo_preds("b3_vit_s0_1009-1616_*")             # B3 on the old cache (val 0.530)
F2_V1 = repo_preds("f2_attn_s0_1009-1605_*")            # fusion on the old cache (val 0.690)
print("reference predictions from the repo:", {k: v is not None for k, v in
      dict(B1=B1_VAL, B0=B0_VAL, B3_v2_Akula=AKULA_B3_V2, B3_v1=B3_V1, F2_v1=F2_V1).items()})
'''

ATTACH = """Kaggle setup: Accelerator **GPU T4 x2** (one GPU is used), Internet **ON**. **+ Add Input**:
- **`mnj-cache-v2`** (Akula's builder-v2 cache, all 3 cohorts). With `CACHE_VERSION = 2` the run stops if
  another cache is picked.
- the code: the `mnj-code-vaib` Dataset (upload the new zip as a **new version** first) or the `GITHUB_TOKEN` secret.

Use **Save Version -> Save & Run All (Commit)**: these runs take hours and continue with the browser closed.
Results go to the Output tab under `results/<PERSON>/<step>/`. Keep this notebook's output: step 8 reads it."""

NB06 = [
    md("""# Step 6: reproduce Akula's cache-v2 result, then our ViT-only model on cache v2 (GPU, ~3 h)
Akula's update: B3 (ViT image-only, `b3_vit`, seed 0) on cache v2 gave **val AUC 0.559** (old cache: 0.530).
1. **B3 reproduction:** the same procedure (`b3_vit`, seed 0, cache v2, no overrides) and a paired
   comparison with his committed predictions. GPU training is not bit-exact; an AUC within ~0.03 and highly
   correlated predictions count as a match.
2. **Probe on cache v2:** our step-4 sweep again. The first probe used the old clipped cache; v2 has a
   tumour-based clip ceiling and a fixed mm scale.
3. **Our ViT-only model on cache v2:** the probe's top setting, 3 seeds, ensemble, then paired DeLong and
   bootstrap tests against B3.

""" + ATTACH),
    params('''STEP = "step6"
CACHE_VERSION = 2          # Akula's mnj-cache-v2
RUN_B3 = True              # 1. reproduce Akula's B3 (~0.5 h)
RUN_PROBE = True           # 2. probe on cache v2 (~0.5 h)
PROBE_BACKBONES = ["mae", "dinov2_b"]
PROBE_NORMS = ["joint_pct", "per_phase", "joint_tumor", "pre_ref", "zscore", "sub", "kinetic", "early3"]
PROBE_CROPS = ["fixed", "tumor"]
RUN_VITONLY = True         # 3. fine-tune the probe's top setting (~0.5 h per seed)
VITONLY_SEEDS = [0, 1, 2]
VITONLY_EXTRA = []         # extra overrides, e.g. ["model.train_blocks=2"]
TIME_BUDGET_H = 11.0
DEBUG = False              # True: 2 epochs on 48 patients per run (a few minutes) to check the notebook
PUSH_RESULTS = False
NOTEBOOK_NAME = "step6"
'''),
    code(SETUP),
    code(HELPERS),
    code('''caches = check_caches(CACHE_VERSION)'''),
    md("## 1. B3 reproduction (Akula's procedure)"),
    code(r'''preds = {"B3, old cache v1 (repo)": B3_V1, "B3, cache v2 (Akula, repo)": AKULA_B3_V2}
if RUN_B3:
    d, m = train("b3_vit", 0, [], label="step6_b3_repro")
    note_runs("b3_repro", [d])
    preds["B3, cache v2 (reproduced)"] = ev.load_preds(f"{d}/val_preds.csv")
    a, b = ev.align({"a": AKULA_B3_V2, "b": preds["B3, cache v2 (reproduced)"]}).values()
    print(f"\nAkula val AUC {stats.auc(a.label, a.prob):.3f} | reproduced {stats.auc(b.label, b.prob):.3f} | "
          f"correlation of the two runs' predictions r = {np.corrcoef(a.prob, b.prob)[0, 1]:.2f} | "
          f"train AUC (overfitting) {train_auc(d):.3f} (Akula: 0.873)")
display(compare(preds, "B3, cache v2 (Akula, repo)").round(3))
'''),
    md("## 2. Probe on cache v2 (frozen backbones, CV AUC on train)"),
    code(r'''if RUN_PROBE:
    pdir = "/kaggle/working/probe_v2"
    cmd = [sys.executable, "-m", "src.vitonly.probe", "--backbones", *PROBE_BACKBONES, "--norms", *PROBE_NORMS,
           "--crops", *PROBE_CROPS, "--prefer", "uint8", "--out", pdir, "--cache-root", "/kaggle/input", "/kaggle/working"]
    if DEBUG: cmd += ["--limit", "40"]
    print(" ".join(cmd)); subprocess.run(cmd, check=True)
    probe = pd.read_csv(f"{pdir}/probe_results.csv")
    os.makedirs(f"{OUT}/step6_probe", exist_ok=True); probe.to_csv(f"{OUT}/step6_probe/probe_results.csv", index=False)
    display(probe.head(15).round(3))
    top = probe[probe.backbone != "-"].iloc[0]
    TOP = dict(backbone=top.backbone, norm=top.norm, crop=top.crop, pool=top["pool"])
else:
    TOP = dict(backbone="dinov2_b", norm="sub", crop="tumor", pool="gap")     # step-4 winner on the old cache
print("TOP setting for fine-tuning:", TOP)
'''),
    md("## 3. Our ViT-only model on cache v2 (3 seeds + ensemble) vs B3"),
    code(r'''if RUN_VITONLY:
    cfg = "vo_dinov2" if TOP["backbone"].startswith("dinov2") else "vo_base"
    sets = [f"data.norm={TOP['norm']}", f"data.crop={TOP['crop']}", f"model.backbone={TOP['backbone']}",
            f"model.pool={TOP['pool']}", "data.prefer_cache=uint8", *VITONLY_EXTRA]
    dirs = []
    for s in VITONLY_SEEDS:
        if budget_left() < 0.7:
            print(f"seed {s} skipped: time budget"); continue
        dirs.append(finetune(cfg, s, sets, f"step6_vitonly_s{s}"))
    note_runs("vitonly", dirs)
    preds["ViT-only vaib, cache v2 (ensemble)"] = save_preds("step6_vitonly_ensemble", preds_of(dirs))
    for d in dirs:
        print(os.path.basename(d), "val AUC", round(json.load(open(f"{d}/metrics.json"))["val"]["auc"], 3))
tab = compare(preds, "B3, cache v2 (Akula, repo)")
tab.to_csv(f"{OUT}/step6_summary.csv", index=False)
display(tab.round(3))
print("diff_vs_ref > 0 with delong_p < 0.05 = significantly better than Akula's B3 on the same 177 patients")
'''),
    code(PUSH),
]

NB07 = [
    md("""# Step 7: regularisation sweep for the ViT (point 1) + ViT-clinical fusion on cache v2 (point 2) (GPU, ~5-6 h)
**Point 1, regularisation.** B3 on cache v2 overfits (train AUC 0.87 vs val 0.56). Each run below changes
**one** thing against `base`, which is Akula's sweep list from `kaggle/11_sweep.ipynb`: frozen ViT (linear
probe), 2 trainable blocks, skipping near-empty slices, stronger augmentation, dropout 0.5 + weight decay, and
the supervised ImageNet backbone. The table shows val AUC, **train AUC (the overfitting gap)**, and the paired
DeLong / bootstrap difference vs `base`.

**Point 2, fusion.** F2 (`f2_attn`: ViT slice tokens + clinical tokens through attention) on cache v2, 3 seeds
+ ensemble. AUC overall, per cohort and **within subtypes** (HR+/HER2-, HER2+, triple-negative), compared
with B1 (clinical + tumour size) and B3 by paired DeLong. `FUSION_USE_BEST_SWEEP = True` also applies the
best sweep setting to F2. That choice is made on val, so mark it as such in the report.

""" + ATTACH),
    params('''STEP = "step7"
CACHE_VERSION = 2
RUN_SWEEP = False                 # Akula already ran this sweep (results/A/sweep_b3_vit_1019-0406: best blocks_2
                                  # 0.579 vs base 0.559, nothing significant). True = rerun it here (~3.5 h)
SWEEP_SEEDS = [0]                 # [0, 1, 2] to confirm a winner (3x the time)
RUNS = {                          # name: overrides of b3_vit; the FIRST entry is the baseline (Akula's list)
    "base":         ["optim.patience=15"],
    "frozen":       ["optim.patience=15", "model.train_blocks=0"],
    "blocks_2":     ["optim.patience=15", "model.train_blocks=2"],
    "min_area_0.1": ["optim.patience=15", "data.min_area_frac=0.1"],
    "strong_aug":   ["optim.patience=15", "aug.rotate_deg=30", "aug.intensity=0.2", "aug.scale=[0.8, 1.2]"],
    "dropout_0.5":  ["optim.patience=15", "model.dropout=0.5", "optim.weight_decay=0.1"],
    "supervised":   ["optim.patience=15", "model.backbone=google/vit-base-patch16-224"],
}
RUN_FUSION = True
FUSION_CONFIG = "f2_attn"         # F2 (main model); "f1_concat" = concatenation baseline
FUSION_SEEDS = [0, 1, 2]
FUSION_EXTRA = []                 # overrides for F2, e.g. ["model.train_blocks=2"] (Akula's best sweep setting)
FUSION_USE_BEST_SWEEP = False     # True: add the best sweep entry's overrides to F2 (chosen on val)
TIME_BUDGET_H = 11.0
DEBUG = False
PUSH_RESULTS = False
NOTEBOOK_NAME = "step7"
'''),
    code(SETUP),
    code(HELPERS),
    code('''caches = check_caches(CACHE_VERSION)'''),
    md("## Point 1: regularisation sweep (B3 on cache v2)"),
    code(r'''sweep = {}
if RUN_SWEEP:
    for name, sets in RUNS.items():
        for s in SWEEP_SEEDS:
            if budget_left() < 0.8:
                print(f"{name} seed {s} skipped: time budget"); continue
            d, m = train("b3_vit", s, sets, label=f"step7_sweep_{name}_s{s}")
            sweep.setdefault(name, []).append(d)
            print(f"{name} s{s}: val AUC {m['val']['auc']:.3f}, train AUC {train_auc(d):.3f}, best epoch {m['best_epoch']}")
    note_runs("sweep", [d for ds in sweep.values() for d in ds])
    sp = {n: preds_of(ds) for n, ds in sweep.items()}
    tab = compare(sp, "base")
    tab.insert(1, "overrides", [" ".join(RUNS[n]) for n in tab.model])
    tab.insert(2, "train_auc", [np.nanmean([train_auc(d) for d in sweep[n]]) for n in tab.model])
    tab.insert(3, "gap", tab.train_auc - tab.auc)
    tab.to_csv(f"{OUT}/step7_sweep_summary.csv", index=False)
    display(tab.round(3))
    best = tab.sort_values("auc", ascending=False).model.iloc[0]
    note_runs("sweep_best", sweep[best])
    save_preds("step7_sweep_best", sp[best])
    print(f"best on val: {best} (overrides {RUNS[best]}). Trust it only if delong_p < 0.05 vs base; "
          "confirm with SWEEP_SEEDS = [0, 1, 2].")
'''),
    md("## Point 2: ViT + clinical fusion (F2) on cache v2, overall / per cohort / within subtypes"),
    code(r'''if RUN_FUSION:
    extra = list(FUSION_EXTRA)
    if FUSION_USE_BEST_SWEEP and sweep:
        extra += [s for s in RUNS[best] if s not in extra]
    dirs = []
    for s in FUSION_SEEDS:
        if budget_left() < 0.8:
            print(f"fusion seed {s} skipped: time budget"); continue
        d, m = train(FUSION_CONFIG, s, extra, label=f"step7_fusion_s{s}")
        dirs.append(d)
        print(f"{FUSION_CONFIG} s{s}: val AUC {m['val']['auc']:.3f}")
    note_runs("fusion", dirs)
    f2 = save_preds("step7_fusion_ensemble", preds_of(dirs))
    b3 = preds_of(sweep["base"]) if sweep.get("base") else AKULA_B3_V2
    models = {"B1 clinical + size (LR)": B1_VAL, "B0 clinical (LR)": B0_VAL, "B3 ViT image-only, v2": b3,
              "F2 fusion, old cache v1": F2_V1, f"F2 fusion, cache v2 ({len(dirs)} seeds)": f2}
    tab = compare(models, "B1 clinical + size (LR)")
    tab.to_csv(f"{OUT}/step7_fusion_summary.csv", index=False)
    display(tab.round(3))
    print("\nAUC within subtypes [95% CI] n=patients/pCR+:")
    st = subtype_wide(models); st.to_csv(f"{OUT}/step7_fusion_by_subtype.csv"); display(st)
    print("\nAUC per cohort:")
    sc = subtype_wide(models, "cohort"); sc.to_csv(f"{OUT}/step7_fusion_by_cohort.csv"); display(sc)
'''),
    code(PUSH),
]

NB08 = [
    md("""# Step 8: statistics (point 3), leave-one-cohort-out (point 4), final test (point 4, OFF by default)
**Point 3, statistics** on the validation predictions of every model: AUC with DeLong 95% CI, AUPRC, Brier,
ECE, sensitivity / PPV / NPV at 90% specificity, the paired DeLong test vs the clinical baseline B1,
calibration (reliability) plots, and AUC within subtypes and cohorts. It reads the repo's committed results plus
any attached step-6 / step-7 notebook outputs (CPU part, minutes).

**Point 4a, leave-one-cohort-out (LOCO):** for each cohort, train on the other two (early stopping on their
val), then score the held-out cohort's train + val patients, which the model never saw. The clinical B1 is run
the same way for comparison (GPU, ~0.5 h per model per cohort).

**Point 4b, the final test evaluation:** **once**, on the frozen final models, with thresholds taken from val.
`FINAL_TEST = False` by default. Agree with the team before running it (plan D-006: the test set is used once;
every use is logged in `final_eval_log.csv`). It needs the step-6/7 outputs attached (their checkpoints).

Kaggle setup: GPU T4 x2 (for LOCO), Internet ON. **+ Add Input**: `mnj-cache-v2`, the code (`mnj-code-vaib`
new version or the `GITHUB_TOKEN` secret), and the **outputs of your step-6 and step-7 notebooks**
(+ Add Input -> Your Work -> Notebooks)."""),
    params('''STEP = "step8"
CACHE_VERSION = 2
RUN_STATS = True
REFERENCE = "B1 clinical + size (LR)"
MODELS = {   # name: run-folder name pattern (glob) under results/ of the repo or the attached outputs; seeds averaged
    "B1 clinical + size (LR)":            "b1_lr_s0_*",
    "B0 clinical (LR)":                   "b0_lr_s0_*",
    "B3 ViT, old cache v1":               "b3_vit_s0_1009-1616_*",
    "B3 ViT, cache v2 (Akula)":           "b3_vit_s0_1019-0302_*",
    "B3 ViT, cache v2, 2 blocks (Akula sweep)": "b3_vit_s0_1019-0500_*",
    "B3 ViT, cache v2, best regularised (step 7)": "step7_sweep_best",
    "ViT-only vaib, old cache (ens.)":    "vo_mae_kinetic_tumor_ensemble",
    "ViT-only vaib, cache v2 (ens.)":     "step6_vitonly_ensemble",
    "F2 fusion, old cache v1":            "f2_attn_s0_1009-1605_*",
    "F2 fusion, cache v2 (ens.)":         "step7_fusion_ensemble",
}
RUN_LOCO = True
LOCO_CONFIGS = ["f2_attn", "b3_vit"]     # models retrained per held-out cohort
LOCO_SEED = 0
LOCO_EXTRA = []                          # e.g. the best sweep overrides from step 7
FINAL_TEST = False                       # True ONCE, with the team's agreement
FINAL_CONFIRM = ""                       # must be "I use the test set once" when FINAL_TEST = True
FINAL_MODELS = {   # name: "step:key" of runs.json written by step 6 / 7 (checkpoints from the attached outputs)
    "F2 fusion, cache v2": "step7:fusion",
    "B3 ViT, cache v2, best regularised": "step7:sweep_best",
    "ViT-only vaib, cache v2": "step6:vitonly",
}
TIME_BUDGET_H = 11.0
DEBUG = False
PUSH_RESULTS = False
NOTEBOOK_NAME = "step8"
'''),
    code(SETUP),
    code(HELPERS),
    md("## Point 3: statistics on validation"),
    code(r'''ROOTS = ["results", "/kaggle/input"]
def collect(pattern):
    seen, hits = set(), []
    for r in ROOTS:
        for h in sorted(glob.glob(f"{r}/**/{pattern}/val_preds.csv", recursive=True)):
            name = os.path.basename(os.path.dirname(h))
            if name not in seen:            # the same run can sit in the repo and in an attached output
                seen.add(name); hits.append(h)
    return hits

if RUN_STATS:
    say(f"POINT 3: collecting val predictions of {len(MODELS)} models from {ROOTS}")
    preds = {}
    for i, (name, pat) in enumerate(MODELS.items(), 1):
        hits = collect(pat)
        if not hits:
            print(f"  [{i}/{len(MODELS)}] missing (not run / not attached): {name} [{pat}]", flush=True); continue
        preds[name] = preds_of([os.path.dirname(h) for h in hits])
        print(f"  [{i}/{len(MODELS)}] {name}: {len(hits)} run(s), {len(preds[name])} val patients", flush=True)
    say(f"computing AUC / DeLong / bootstrap / 90%-spec operating points for {len(preds)} models ...")
    tab = compare(preds, REFERENCE)
    say("statistics done")
    tab.to_csv(f"{OUT}/step8_val_statistics.csv", index=False)
    display(tab.round(3))
    print("sens/ppv/npv@90spec: threshold chosen on these val patients (optimistic); the final test uses val thresholds.")
    say("calibration plots ...")
    al = ev.align(preds)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ev.plot_reliability(ax[0], {k: (v.label, v.prob) for k, v in al.items() if "B1" in k or "F2" in k or "B0" in k})
    ev.plot_reliability(ax[1], {k: (v.label, v.prob) for k, v in al.items() if "ViT" in k or "B3" in k},
                        "Calibration, image-only models (val)")
    plt.tight_layout(); plt.savefig(f"{OUT}/step8_calibration.png", dpi=110); plt.show()
    say("AUC within subtypes and cohorts ...")
    st = subtype_wide(preds); st.to_csv(f"{OUT}/step8_val_by_subtype.csv"); print("AUC within subtypes:"); display(st)
    sc = subtype_wide(preds, "cohort"); sc.to_csv(f"{OUT}/step8_val_by_cohort.csv"); print("AUC per cohort:"); display(sc)
    say(f"POINT 3 done -> {OUT}")
'''),
    md("## Point 4a: leave-one-cohort-out"),
    code(r'''if RUN_LOCO:
    caches = check_caches(CACHE_VERSION)
    from src.predict_run import score_run
    rows = []
    for held in COH:
        others = [c for c in COH if c != held]
        tr = patient_table("train", others, cache_dirs=caches)                 # same patients as the image models
        sc_t = patient_table(["train", "val"], [held], cache_dirs=caches)
        b1 = ev.clinical_preds(tr, sc_t, "b1_lr")
        r = ev.delong_ci(b1.label, b1.prob)
        rows.append({"model": "B1 clinical + size (LR)", "held_out": held, "n": len(b1), "pcr": int(b1.label.sum()),
                     "auc": r["auc"], "ci_lo": r["ci"][0], "ci_hi": r["ci"][1]})
        for cfg in LOCO_CONFIGS:
            if budget_left() < 0.7:
                print(f"{cfg} / {held} skipped: time budget"); continue
            d, m = train(cfg, LOCO_SEED, ["export.after_train=false", *LOCO_EXTRA],
                         label=f"step8_loco_{cfg}_{held}", cohorts=others)
            p = score_run(d, [held], ["train", "val"])
            p.to_csv(f"{OUT}/step8_loco_{cfg}_{held}/heldout_preds.csv", index=False)
            r = ev.delong_ci(p.label, p.prob)
            rows.append({"model": cfg, "held_out": held, "n": len(p), "pcr": int(p.label.sum()), "auc": r["auc"],
                         "ci_lo": r["ci"][0], "ci_hi": r["ci"][1], "val_auc_training_cohorts": m["val"]["auc"]})
            print(f"{cfg} trained on {others} -> {held}: AUC {r['auc']:.3f} ({r['ci'][0]:.2f}-{r['ci'][1]:.2f})")
    loco = pd.DataFrame(rows)
    loco.to_csv(f"{OUT}/step8_loco.csv", index=False)
    display(loco.round(3))
    display(loco.pivot(index="model", columns="held_out", values="auc").round(3))
'''),
    md("## Point 4b: final test evaluation (once)"),
    code(r'''if FINAL_TEST:
    assert FINAL_CONFIRM == "I use the test set once", 'set FINAL_CONFIRM = "I use the test set once"'
    from src.predict_run import score_run
    say("POINT 4b: FINAL TEST, looking for runs.json and checkpoints ...")
    rj = {}
    for f in glob.glob("/kaggle/input/**/runs.json", recursive=True) + glob.glob("/kaggle/working/results/**/runs.json", recursive=True):
        step = os.path.basename(os.path.dirname(f))
        rj.setdefault(step, {}).update(json.load(open(f)))
    all_runs = {d.name: d for d in ev.find_run_dirs(["/kaggle/input", "/kaggle/working"])}
    train_t, val_t = patient_table("train", COH), patient_table("val", COH)
    test_t = patient_table("test", COH, allow_test=True)
    final, log = {}, []
    b1_val = ev.clinical_preds(train_t, val_t, "b1_lr")
    final["B1 clinical + size (LR)"] = (ev.clinical_preds(train_t, test_t, "b1_lr"),
                                        ev.threshold_at_spec(b1_val.label, b1_val.prob))
    say(f"B1 done; test set {len(test_t)} patients. Scoring {len(FINAL_MODELS)} models ...")
    for name, key in FINAL_MODELS.items():
        step, k = key.split(":")
        ids = rj.get(step, {}).get(k, [])
        dirs = [all_runs[i] for i in ids if i in all_runs]
        say(f"{name}: {len(dirs)} checkpoint(s) found for {key}")
        if not dirs:
            print(f"SKIPPED {name}: no checkpoints for {key} (attach the {step} notebook output; "
                  "a model reused from the repo has predictions but no checkpoint)"); continue
        tests = [score_run(d, COH, ["test"], final=True) for d in dirs]
        log += [{"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": name, "run": d.name, "n": len(t)} for d, t in zip(dirs, tests)]
        vals = preds_of(dirs)
        final[name] = (ev.ensemble(tests) if len(tests) > 1 else tests[0], ev.threshold_at_spec(vals.label, vals.prob))
    fdir = f"/kaggle/working/results/{PERSON}/final_test"; os.makedirs(fdir, exist_ok=True)
    pd.DataFrame(log).to_csv(f"{fdir}/final_eval_log.csv", mode="a", index=False,
                             header=not os.path.exists(f"{fdir}/final_eval_log.csv"))
    aligned = ev.align({k: v[0] for k, v in final.items()})       # the test patients every model scored
    ref = aligned["B1 clinical + size (LR)"]
    rows = []
    for name, (_, t) in final.items():
        p = aligned[name]
        r = {"model": name, **ev.summary_row(p.label, p.prob, t=t)}
        if name != "B1 clinical + size (LR)":
            dl = ev.delong(ref.label, ref.prob, p.prob)
            r.update(diff_vs_B1=dl["diff"], delong_ci_lo=dl["ci"][0], delong_ci_hi=dl["ci"][1], delong_p=dl["p"])
        rows.append(r)
        p.to_csv(f"{fdir}/{name.replace(' ', '_').replace(',', '')}_test_preds.csv", index=False)
    ft = pd.DataFrame(rows); ft.to_csv(f"{fdir}/final_test_statistics.csv", index=False)
    say(f"POINT 4b done -> {fdir}")
    print(f"TEST SET: {len(ref)} patients, {int(ref.label.sum())} pCR. Thresholds from val (90% spec)."); display(ft.round(3))
    fp = aligned
    st = subtype_wide(fp); st.to_csv(f"{fdir}/final_test_by_subtype.csv"); display(st)
    sc = subtype_wide(fp, "cohort"); sc.to_csv(f"{fdir}/final_test_by_cohort.csv"); display(sc)
    fig, ax = plt.subplots(figsize=(6.5, 5)); ev.plot_reliability(ax, {k: (v.label, v.prob) for k, v in fp.items()}, "Calibration (test)")
    plt.tight_layout(); plt.savefig(f"{fdir}/final_test_calibration.png", dpi=110); plt.show()
else:
    print("final test not run (FINAL_TEST = False)")
'''),
    code(PUSH),
]


LOCO_OURS = r'''if RUN_LOCO:
    import re
    caches = check_caches(CACHE_VERSION)
    from src.predict_run import score_run
    bs = {}
    if LOCO_SETS is None:
        found = sorted(glob.glob("/kaggle/input/**/best_setting.json", recursive=True))
        bs = json.load(open(found[0])) if found else {}
        sets = bs.get("overrides", ["model.backbone=dinov2_b", "optim.lr_backbone=1.0e-5", "data.norm=sub",
                                    "data.crop=tumor", "model.pool=gap"])
        print("LOCO setting:", bs.get("name", "default: dinov2_b + sub + tumour crop (attach the step-7 output)"), sets)
    else:
        sets = list(LOCO_SETS)
    fsets = bs.get("fusion_overrides", sets)
    rows = []
    n_jobs, job = len(COH) * len(LOCO_MODELS), 0
    say(f"POINT 4a: LOCO, {n_jobs} trainings ({', '.join(LOCO_MODELS)}) x held-out {COH}")
    for held in COH:
        others = [c for c in COH if c != held]
        say(f"=== held-out cohort {held.upper()} (train on {others}) ===")
        tr = patient_table("train", others, cache_dirs=caches)               # same patients as the image models
        sc_t = patient_table(["train", "val"], [held], cache_dirs=caches)
        b1 = ev.clinical_preds(tr, sc_t, "b1_lr")
        r = ev.delong_ci(b1.label, b1.prob)
        rows.append({"model": "B1 clinical + size (LR)", "held_out": held, "n": len(b1), "pcr": int(b1.label.sum()),
                     "auc": r["auc"], "ci_lo": r["ci"][0], "ci_hi": r["ci"][1]})
        say(f"B1 clinical + size -> {held}: AUC {r['auc']:.3f} ({len(b1)} patients, {int(b1.label.sum())} pCR)")
        for name, cfg in LOCO_MODELS.items():
            job += 1
            if budget_left() < 0.9:
                say(f"[{job}/{n_jobs}] {name} / {held} skipped: time budget"); continue
            label = "step8_loco_" + re.sub(r"[^a-z0-9]+", "_", name.lower()) + f"_{held}"
            say(f"[{job}/{n_jobs}] training {name} ({cfg}) on {others} ({len(tr)} train patients) ...")
            d = finetune(cfg, LOCO_SEED, fsets if cfg == "vo_fusion" else sets, label, cohorts=others)
            m = json.load(open(f"{d}/metrics.json"))
            say(f"[{job}/{n_jobs}] {name}: val AUC on {others} {m['val']['auc']:.3f}; scoring held-out {held} ...")
            p = score_run(d, [held], ["train", "val"])
            p.to_csv(f"{OUT}/{label}/heldout_preds.csv", index=False)
            r = ev.delong_ci(p.label, p.prob)
            rows.append({"model": name, "held_out": held, "n": len(p), "pcr": int(p.label.sum()), "auc": r["auc"],
                         "ci_lo": r["ci"][0], "ci_hi": r["ci"][1], "val_auc_training_cohorts": m["val"]["auc"]})
            print(f"{name} trained on {others} -> {held}: AUC {r['auc']:.3f} ({r['ci'][0]:.2f}-{r['ci'][1]:.2f})")
        pd.DataFrame(rows).to_csv(f"{OUT}/step8_loco.csv", index=False)     # partial table after each cohort
    loco = pd.DataFrame(rows)
    loco.to_csv(f"{OUT}/step8_loco.csv", index=False)
    say(f"POINT 4a done -> {OUT}/step8_loco.csv")
    display(loco.round(3))
    display(loco.pivot(index="model", columns="held_out", values="auc").round(3))
'''

ATTACH_OURS = """Kaggle setup: Accelerator **GPU T4 x2** (one GPU is used), Internet **ON**. **+ Add Input**:
- the caches: `mnj-cache-ispy1`, `mnj-cache-ispy2`, `mnj-cache-duke`. Our pipeline normalises the images
  itself (`data.norm`), so any attached cache works; `mnj-cache2-*` (raw) or `mnj-cache-v2` are used
  automatically if attached;
- the code: the `mnj-code-vaib` Dataset (upload the new zip as a **new version** first) or the `GITHUB_TOKEN` secret.

Use **Save Version -> Save & Run All (Commit)**: these runs take hours and continue with the browser closed.
Results go to the Output tab under `results/<PERSON>/<step>/`. Keep this notebook's output: step 8 reads it."""

NB07 = [
    md("""# Step 7: best normalisation, regularisation (point 1), fusion with clinical data (point 2) (GPU, ~6-8 h)
Everything uses **our pipeline** (`src/vitonly`): our normalisation, tumour crop and backbone. Akula's cache v2
uses one fixed normalisation; we compare against it later.
- **A. Best normalisation:** the two strongest ViT-only settings so far, 3 seeds each, ensemble, paired DeLong.
  The "(step 5)" one is reused from the repo (val 0.574) with `REUSE_STEP5`. The best on val is carried
  forward (`best_setting.json`); that choice is made on val, so say so in the report.
- **B. Regularisation (point 1)** of the best setting: frozen ViT (linear probe), 2 trainable blocks,
  largest-tumour-area slices, stronger augmentation, dropout 0.5 + weight decay. Each run changes one thing
  vs `base`, and train AUC is reported to show the overfitting gap.
- **C. Fusion (point 2):** the best setting + clinical tokens (HR, HER2, race, menopause, age, tumour-box
  volume) through attention (the F2 design, `configs/vo_fusion.yaml`), 3 seeds. AUC overall, per cohort and
  **within subtypes**, paired DeLong vs B1 (clinical + size).

""" + ATTACH_OURS),
    params('''STEP = "step7"
CACHE_VERSION = None              # our pipeline: no cache-version requirement
CANDIDATES = {                    # ViT-only settings (overrides of configs/vo_base.yaml)
    "dinov2_b + sub + tumour crop (probe #1)": ["model.backbone=dinov2_b", "optim.lr_backbone=1.0e-5",
                                               "data.norm=sub", "data.crop=tumor", "model.pool=gap"],
    "mae + kinetic + tumour crop (step 5)":    ["model.backbone=mae", "data.norm=kinetic", "data.crop=tumor",
                                               "model.pool=gap"],
}
CAND_SEEDS = [0, 1, 2]
REUSE_STEP5 = True                # the "(step 5)" candidate: use its 3 seeds from results/B instead of retraining
RUN_SWEEP = True
SWEEP = {                         # name: overrides added to the best candidate; "base" = the best candidate itself
    "base":        [],
    "frozen":      ["model.train_blocks=0", "optim.freeze_epochs=0"],
    "blocks_2":    ["model.train_blocks=2"],
    "area_slices": ["data.slice_mode=area"],
    "strong_aug":  ["aug.rotate_deg=30", "aug.intensity=0.2", "aug.scale=[0.8, 1.2]"],
    "dropout_0.5": ["model.dropout=0.5", "optim.weight_decay=0.1"],
}
SWEEP_SEEDS = [0]
RUN_FUSION = True
FUSION_SEEDS = [0, 1, 2]
FUSION_EXTRA = []                 # extra overrides for the fusion model
FUSION_USE_BEST_SWEEP = False     # True: also add the best sweep entry's overrides (chosen on val)
TIME_BUDGET_H = 11.0
DEBUG = False
PUSH_RESULTS = False
NOTEBOOK_NAME = "step7"
'''),
    code(SETUP),
    code(HELPERS),
    code('''caches = check_caches(CACHE_VERSION)'''),
    md("## A. Which normalisation / backbone is best? (ViT-only, 3 seeds each)"),
    code(r'''import re
slug = lambda s: re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
cand_runs, cand_preds = {}, {}
for name, sets in CANDIDATES.items():
    if REUSE_STEP5 and "(step 5)" in name and repo_preds("vo_base_mae_kinetic_tumor_s*") is not None:
        cand_preds[name] = repo_preds("vo_base_mae_kinetic_tumor_s*")
        print(f"{name}: reused the 3 step-5 seeds from results/B"); continue
    dirs = []
    for s in CAND_SEEDS:
        if budget_left() < 1.0:
            print(f"{name} seed {s} skipped: time budget"); continue
        dirs.append(finetune("vo_base", s, sets, f"step7_cand_{slug(name)}_s{s}"))
    cand_runs[name], cand_preds[name] = dirs, preds_of(dirs)
first = next(iter(cand_preds))
tab = compare(cand_preds, first)
tab.insert(1, "train_auc", [np.nanmean([vo_train_auc(d) for d in cand_runs.get(k, [])]) if cand_runs.get(k) else np.nan
                            for k in tab.model])
tab.to_csv(f"{OUT}/step7_candidates.csv", index=False); display(tab.round(3))
BEST = tab.sort_values("auc", ascending=False).model.iloc[0]
BEST_SETS = CANDIDATES[BEST]
save_preds("step7_vitonly_best_ensemble", cand_preds[BEST])
note_runs("vitonly_best", cand_runs.get(BEST, []))
json.dump({"name": BEST, "overrides": BEST_SETS}, open(f"{OUT}/best_setting.json", "w"), indent=1)
print(f"BEST (on val): {BEST} -> {BEST_SETS}")
'''),
    md("## B. Point 1: regularisation of the best setting"),
    code(r'''sweep = {}
if RUN_SWEEP:
    for name, extra in SWEEP.items():
        for s in SWEEP_SEEDS:
            reuse = cand_runs.get(BEST, [])
            if name == "base" and s in CAND_SEEDS and CAND_SEEDS.index(s) < len(reuse):
                sweep.setdefault(name, []).append(reuse[CAND_SEEDS.index(s)]); continue   # same run as candidate A
            if budget_left() < 0.9:
                print(f"{name} seed {s} skipped: time budget"); continue
            sweep.setdefault(name, []).append(finetune("vo_base", s, BEST_SETS + extra, f"step7_sweep_{name}_s{s}"))
    note_runs("sweep", [d for ds in sweep.values() for d in ds])
    sp = {n: preds_of(ds) for n, ds in sweep.items()}
    tab = compare(sp, "base")
    tab.insert(1, "overrides", [" ".join(SWEEP[n]) or "(best candidate)" for n in tab.model])
    tab.insert(2, "train_auc", [np.nanmean([vo_train_auc(d) for d in sweep[n]]) for n in tab.model])
    tab.insert(3, "gap", tab.train_auc - tab.auc)
    tab.to_csv(f"{OUT}/step7_sweep_summary.csv", index=False); display(tab.round(3))
    best_sweep = tab.sort_values("auc", ascending=False).model.iloc[0]
    note_runs("sweep_best", sweep[best_sweep]); save_preds("step7_sweep_best", sp[best_sweep])
    if not RUNS_JSON.get("vitonly_best") and sweep.get("base"):
        note_runs("vitonly_best", sweep["base"])   # best setting was reused from step 5 (no checkpoint in git):
                                                   # the sweep's base run is that same setting, trained here
    print(f"best on val: {best_sweep} {SWEEP[best_sweep]}. Trust it only if delong_p < 0.05 vs base; "
          "confirm with SWEEP_SEEDS = [0, 1, 2]. gap = train AUC - val AUC (overfitting).")
'''),
    md("## C. Point 2: fusion with clinical data (our pipeline), overall / per cohort / within subtypes"),
    code(r'''if RUN_FUSION:
    fsets = BEST_SETS + list(FUSION_EXTRA)
    if FUSION_USE_BEST_SWEEP and sweep:
        fsets += [x for x in SWEEP[best_sweep] if x not in fsets]
    dirs = []
    for s in FUSION_SEEDS:
        if budget_left() < 0.9:
            print(f"fusion seed {s} skipped: time budget"); continue
        dirs.append(finetune("vo_fusion", s, fsets, f"step7_fusion_s{s}"))
    note_runs("fusion", dirs)
    fus = save_preds("step7_fusion_ensemble", preds_of(dirs))
    json.dump({"name": BEST, "overrides": BEST_SETS, "fusion_overrides": fsets}, open(f"{OUT}/best_setting.json", "w"), indent=1)
    models = {"B1 clinical + size (LR)": B1_VAL, "B0 clinical (LR)": B0_VAL, "F2 fusion, old pipeline (repo)": F2_V1,
              "B3 ViT, cache v2 (Akula)": AKULA_B3_V2, f"ViT-only ours: {BEST}": cand_preds[BEST],
              f"Fusion ours ({len(dirs)} seeds)": fus}
    tab = compare(models, "B1 clinical + size (LR)")
    tab.to_csv(f"{OUT}/step7_fusion_summary.csv", index=False); display(tab.round(3))
    print("\nAUC within subtypes (95% CI), all models on the same patients:")
    st = subtype_wide(models); st.to_csv(f"{OUT}/step7_fusion_by_subtype.csv"); display(st)
    print("\nAUC per cohort:")
    sc = subtype_wide(models, "cohort"); sc.to_csv(f"{OUT}/step7_fusion_by_cohort.csv"); display(sc)
'''),
    code(PUSH),
]

NB08 = [
    md("""# Step 8: statistics (point 3), leave-one-cohort-out (point 4), final test (point 4, OFF by default)
**Point 3, statistics** on the validation predictions of every model: AUC with DeLong 95% CI, AUPRC, Brier,
ECE, sensitivity / PPV / NPV at 90% specificity, the paired DeLong test vs the clinical baseline B1,
calibration (reliability) plots, and AUC within subtypes and cohorts. It reads the repo's committed results plus
the attached step-7 (and step-6) outputs.

**Point 4a, leave-one-cohort-out (LOCO):** for each cohort, our ViT-only and fusion models (the best setting
from step 7, read from its `best_setting.json`) are trained on the other two cohorts (early stopping on their
val), then score the held-out cohort's train + val patients, which they never saw. B1 is done the same way.

**Point 4b, the final test evaluation:** **once**, on the frozen final models, with thresholds from val.
`FINAL_TEST = False` by default. Agree with the team first (plan D-006: the test set is used once; every use
is logged). It needs the step-7 output attached (its checkpoints).

Kaggle setup: GPU T4 x2 (for LOCO), Internet ON. **+ Add Input**: the caches (`mnj-cache-ispy1/-ispy2/-duke`),
the code (`mnj-code-vaib` new version, or the `GITHUB_TOKEN` secret), and **your step-7 notebook's output**
(+ Add Input -> Your Work -> Notebooks)."""),
    params('''STEP = "step8"
CACHE_VERSION = None
RUN_STATS = True
REFERENCE = "B1 clinical + size (LR)"
MODELS = {   # name: run-folder name pattern (glob) under results/ of the repo or the attached outputs; seeds averaged
    "B1 clinical + size (LR)":              "b1_lr_s0_*",
    "B0 clinical (LR)":                     "b0_lr_s0_*",
    "B3 ViT, old pipeline":                 "b3_vit_s0_1009-1616_*",
    "B3 ViT, cache v2 (Akula)":             "b3_vit_s0_1019-0302_*",
    "B3 ViT, cache v2, 2 blocks (Akula)":   "b3_vit_s0_1019-0500_*",
    "F2 fusion, old pipeline":              "f2_attn_s0_1009-1605_*",
    "ViT-only ours, step 5 (mae+kinetic)":  "vo_base_mae_kinetic_tumor_s*",
    "ViT-only ours, best (step 7)":         "step7_vitonly_best_ensemble",
    "ViT-only ours, best regularised":      "step7_sweep_best",
    "Fusion ours (step 7)":                 "step7_fusion_ensemble",
}
RUN_LOCO = True
LOCO_MODELS = {"ViT-only ours": "vo_base", "Fusion ours": "vo_fusion"}
LOCO_SEED = 0
LOCO_SETS = None             # None = the best setting from the attached step-7 output; or a list of overrides
FINAL_TEST = False           # True ONCE, with the team's agreement
FINAL_CONFIRM = ""           # must be "I use the test set once" when FINAL_TEST = True
FINAL_MODELS = {             # name: "step:key" of runs.json written by step 7 (checkpoints from the attached output)
    "Fusion ours": "step7:fusion",
    "ViT-only ours, best": "step7:vitonly_best",
}
TIME_BUDGET_H = 11.0
DEBUG = False
PUSH_RESULTS = False
NOTEBOOK_NAME = "step8"
'''),
    code(SETUP),
    code(HELPERS),
] + NB08[4:6] + [md("## Point 4a: leave-one-cohort-out (our ViT-only and fusion models, B1)"), code(LOCO_OURS)] + NB08[8:]


# ------------------------------------------------------------------- steps 9-10 (professor, 2026-10-28)
ATTACH_FEAT = """**+ Add Input** (any cache works; the best one attached per cohort is used and printed):
- the caches, best first: **`mnj-cache2-ispy1/-ispy2/-duke`** (raw, step 2: nothing clipped, exact features), or
  Akula's **`mnj-cache-v2`** (uint8, 0.7 mm, less clipping), or `mnj-cache-ispy1/-ispy2/-duke` (uint8, clipped);
- the code: the `mnj-code-vaib` Dataset (upload the new zip as a **new version** first) or the `GITHUB_TOKEN` secret.

Run with `DEBUG = True` first (a few minutes, 40 patients per split) to check the notebook, then `DEBUG = False`
with **Save Version -> Save & Run All (Commit)**. Every step prints a timestamped `[...]` line, so the log shows
where it is and which patient or configuration failed. Results: Output tab, `results/<PERSON>/<step>/`."""

CACHES_CELL = r'''# ==== which cache is used for each cohort ====
from src.vitonly.slabs import find_caches as vo_find_caches, cache_label
VO_CACHES = vo_find_caches(["/kaggle/input", "/kaggle/working"], PREFER)
say(f"caches found: {len(VO_CACHES)} cohorts")
for c in COH:
    print(f"  {c:6s}", f"{cache_label(VO_CACHES[c])}  {VO_CACHES[c]}" if c in VO_CACHES else "MISSING -> left out", flush=True)
if not VO_CACHES:
    raise SystemExit("no cache attached: + Add Input -> mnj-cache2-* / mnj-cache-v2 / mnj-cache-*")
if any(cache_label(d).startswith("v1") for d in VO_CACHES.values()):
    print("NOTE: uint8 cache(s): the brightest enhancement is clipped, so intensity / PE features are underestimated. "
          "Raw mnj-cache2-* caches (step 2) are exact.", flush=True)
'''

NB09 = [
    md("""# Step 9: 2D / 3D / temporal tumour features, ComBat harmonisation, per-slice normalisation (CPU, ~30-60 min)
The professor's suggestions (2026-10-28), all in one table, no GPU needed:
1. **ComBat harmonisation** (`src/vitonly/combat.py`): removes the scanner / site shift of every feature.
   Batch = cohort (`BATCH`), covariates = HR + HER2 (so subtype biology is kept), **fitted on the training
   patients only** (inside every CV fold too). pCR is never a covariate: it is the label.
2. **2D features**: the slice with the largest enhancing tumour area: first-order statistics of 7 maps
   (pre, early, late, PE, SER, wash-in, wash-out), 2D shape, 2D GLCM texture.
3. **3D features**: the same over the whole tumour volume, with 3D shape and 3D GLCM texture.
4. **Temporal mapping of slices**: every tumour slice's pre -> early -> late curve gives per-slice kinetic
   numbers (enhancing area, PE, SER, wash-out fraction ...), aggregated over the slices (mean, std, max, top-3,
   slice-to-slice change).
5. **Normalise each slice separately**: `REFS = ["volume", "slice"]` extracts everything twice, with the
   intensity reference from the whole volume or from each slice on its own. PE, SER and shape are ratios and
   do not change.

Every set is scored with and without ComBat and with and without subtype, by 5-fold CV on train (rank by this)
and on val (confirmation), with the paired difference to the **subtype lookup (0.662)**, the number to beat.
`RESULTS.md` in the output has the tables ready to paste: does ComBat help, does per-slice help,
2D vs 3D vs temporal, does the image add to subtype, and the harmonisation check.

Kaggle setup: Accelerator **None** (CPU), Internet **ON**.
""" + ATTACH_FEAT),
    params('''STEP = "step9"
PREFER = "v2"                # cache format when several are attached: v2 = raw mnj-cache2-*, v1 = uint8
REFS = ["volume", "slice"]   # intensity reference: whole volume vs each slice on its own (professor)
SETS = ["kinetic16", "2d", "3d", "temporal", "2d+temporal", "3d+temporal", "2d+3d+temporal", "all"]
BATCH = "cohort"             # ComBat batch: cohort | protocol (cohort x pixel-size x slice-thickness groups)
ALSO_PROTOCOL = True         # also run ComBat with BATCH = "protocol" (reuses the extracted features, ~10 min)
JOBS = 4                     # CPU processes (Kaggle CPU sessions have 4 cores)
TIME_BUDGET_H = 11.0
DEBUG = False                # True: 40 patients per split, a few minutes, to check the notebook
PUSH_RESULTS = False
NOTEBOOK_NAME = "step9"
'''),
    code(SETUP),
    code(HELPERS),
    code(CACHES_CELL),
    md("## 1. Extract the features and score every configuration (batch = cohort)"),
    code(r'''FEAT = f"{OUT}/features_{BATCH}"
cmd = [sys.executable, "-m", "src.vitonly.features", "--out", FEAT, "--refs", *REFS, "--sets", *SETS,
       "--batch", BATCH, "--jobs", str(JOBS), "--prefer", PREFER, "--cache-root", "/kaggle/input", "/kaggle/working"]
if DEBUG: cmd += ["--limit", "40"]
say("1. feature extraction + scoring starts (progress every 100 patients, then every configuration)")
run_cmd(cmd)
RES = pd.read_csv(f"{FEAT}/results.csv")
PREDS = pd.read_csv(f"{FEAT}/val_preds.csv", dtype={"pid": str})
say(f"1. done: {len(RES)} rows in {FEAT}/results.csv")
'''),
    md("## 2. ComBat with a finer batch: acquisition protocol (pixel size x slice thickness within each cohort)"),
    code(r'''if ALSO_PROTOCOL and BATCH != "protocol":
    FEAT_P = f"{OUT}/features_protocol"
    cmd = [sys.executable, "-m", "src.vitonly.features", "--out", FEAT_P, "--refs", *REFS, "--sets", *SETS,
           "--batch", "protocol", "--features-from", FEAT, "--jobs", str(JOBS), "--prefer", PREFER,
           "--cache-root", "/kaggle/input", "/kaggle/working"]
    if DEBUG: cmd += ["--limit", "40"]
    say("2. protocol-batch ComBat starts (features reused from step 1)")
    run_cmd(cmd)
    RES_P = pd.read_csv(f"{FEAT_P}/results.csv")
    m = RES.merge(RES_P, on=["ref", "set", "combat", "subtype"], suffixes=("_cohort", "_protocol"))
    m = m[m.combat]
    print(m[["ref", "set", "subtype", "cv_auc_cohort", "cv_auc_protocol", "val_auc_cohort", "val_auc_protocol"]]
          .round(3).to_string(index=False))
else:
    say("2. skipped (ALSO_PROTOCOL = False)")
'''),
    md("## 3. Summary: the best configurations, plots, and the comparison with the team's models"),
    code(r'''cols = ["ref", "set", "combat", "subtype", "n_features", "cv_auc", "val_auc", "val_ci_lo", "val_ci_hi",
        "val_auc_within", "val_auc_duke", "val_auc_ispy1", "val_auc_ispy2", "vs_lookup", "vs_lookup_p"]
print("TOP 20 by CV AUC (rank by CV, val is the confirmation):")
print(RES[[c for c in cols if c in RES]].head(20).round(3).to_string(index=False))

hc = RES[RES.ref.isin(REFS) & ~RES.set.str.startswith("deep")]
fig, axes = plt.subplots(1, len(REFS), figsize=(7 * len(REFS), 4.5), squeeze=False)
for ax, ref in zip(axes[0], REFS):
    d = hc[hc.ref == ref].pivot_table(index="set", columns=["combat", "subtype"], values="cv_auc").reindex(SETS)
    d.columns = [f"{'ComBat' if cb else 'raw'}{' + subtype' if st else ''}" for cb, st in d.columns]
    d.plot.barh(ax=ax, width=0.8)
    lk = RES.loc[RES.set == "subtype_lookup", "val_auc"].iloc[0]
    ax.axvline(lk, color="k", ls="--", lw=1, label=f"subtype lookup (val) {lk:.3f}")
    ax.set_xlim(0.45, max(0.75, d.max().max() + 0.02)); ax.set_title(f"CV AUC, ref = {ref}"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/cv_auc_by_set.png", dpi=120); plt.show()

# best image-only and best image + subtype, against the team's models (paired DeLong on the shared val patients)
def as_preds(col):
    return PREDS[["pid", "cohort", "label"]].assign(prob=PREDS[col].values)
ok = RES[RES.ref.isin(REFS)]
best_img = ok[~ok.subtype].iloc[0]; best_sub = ok[ok.subtype].iloc[0]
key = lambda r: f"{r.ref}|{r.set}|{'combat' if r.combat else 'raw'}|{'+subtype' if r.subtype else 'image'}"
say(f"best image-only (by CV): {key(best_img)} | best image + subtype: {key(best_sub)}")
preds = {"B1 LR (clinical + size)": B1_VAL, "subtype lookup": as_preds("subtype_lookup"),
         "F2 fusion v1 (old cache)": F2_V1,
         f"features, image only: {key(best_img)}": as_preds(key(best_img)),
         f"features + subtype: {key(best_sub)}": as_preds(key(best_sub))}
CMP = compare(preds, "subtype lookup")
print(CMP.round(3).to_string(index=False))
CMP.to_csv(f"{OUT}/comparison_vs_team.csv", index=False)
print(subtype_wide({k: v for k, v in preds.items() if k.startswith("features")}).to_string())
'''),
    md("## 4. Results file\n`RESULTS_step9.md` = the tables of `features_cohort/RESULTS.md` + the comparison with the team's models."),
    code(r'''txt = open(f"{FEAT}/RESULTS.md").read()
from src.vitonly.features import md_table
txt += "\n## Best configurations vs the team's models (paired DeLong vs the subtype lookup)\n\n" + md_table(CMP) + "\n"
if ALSO_PROTOCOL and BATCH != "protocol":
    txt += ("\n## ComBat batch: cohort vs protocol (ComBat rows)\n\n"
            + md_table(m[["ref", "set", "subtype", "cv_auc_cohort", "cv_auc_protocol", "val_auc_cohort", "val_auc_protocol"]]) + "\n")
open(f"{OUT}/RESULTS_step9.md", "w").write(txt)
say(f"wrote {OUT}/RESULTS_step9.md -- download it from the Output tab (or PUSH_RESULTS)")
'''),
    code(PUSH.replace("if PUSH_RESULTS:", "if PUSH_RESULTS and not DEBUG:")),
]

NB10 = [
    md("""# Step 10: per-slice normalisation for the ViT + ComBat on ViT features (GPU, ~4-5 h)
The professor's "normalise each slice separately", tested on the deep models. Each per-slice method has a
**volume twin**, so each pair differs only in where the statistics come from:

| per slice (new) | volume twin | what it does |
|---|---|---|
| `slice_joint` | `joint_wide` | one 0.5/99.9 pct window for the 3 phases |
| `slice_pre` | `pre_ref` | log(1 + x / r), r = mean pre-contrast foreground |
| `slice_zscore` | `zscore` | z-score by the pre-contrast foreground |
| `slice_phase` | `per_phase` | each phase its own window: destroys the enhancement, ablation only |

1. **Probe** (frozen DINOv2 + MAE, ~30 min): every norm above, CV AUC on train + val AUC; per-slice vs twin.
2. **ComBat on the ViT features** (~10 min): the probe's patient embeddings, with and without ComBat and with
   and without subtype (same scoring as step 9). Attach the step-9 output to also get "ViT + 3d+temporal".
3. **Fine-tuning** (~0.4 h per run): DINOv2, 2 trainable blocks (Akula's best), each norm in `FT_NORMS` x
   `FT_SEEDS`. Reports the best-epoch val AUC **and the mean val AUC over epochs 5-15** (the honest number;
   the peak epoch is optimistic), per cohort (is Duke still at chance?) and paired tests per-slice vs twin.

Kaggle setup: Accelerator **GPU T4 x2**, Internet **ON**.
""" + ATTACH_FEAT + "\nOptional: **+ Add Input -> Your Work -> Notebooks -> your step-9 notebook** (for ViT + hand-crafted)."),
    params('''STEP = "step10"
PREFER = "v2"
RUN_PROBE = True
PROBE_BACKBONES = ["dinov2_b", "mae"]
PROBE_NORMS = ["joint_wide", "slice_joint", "pre_ref", "slice_pre", "zscore", "slice_zscore", "per_phase",
               "slice_phase", "early3", "kinetic"]
PROBE_CROPS = ["tumor"]
RUN_DEEP_COMBAT = True       # ComBat / subtype on the probe's embeddings
RUN_FT = True
FT_CONFIG = "vo_dinov2"      # DINOv2-B/14 (configs/vo_dinov2.yaml)
FT_NORMS = ["pre_ref", "slice_pre", "joint_wide", "slice_joint"]   # twins next to each other
FT_SEEDS = [0, 1]
FT_EXTRA = ["model.train_blocks=2", "data.crop=tumor"]           # Akula's best: 2 trainable blocks
TWINS = {"slice_joint": "joint_wide", "slice_pre": "pre_ref", "slice_zscore": "zscore", "slice_phase": "per_phase"}
TIME_BUDGET_H = 11.0
DEBUG = False                # True: probe on 40 patients, fine-tuning 2 epochs on 48 patients
PUSH_RESULTS = False
NOTEBOOK_NAME = "step10"
'''),
    code(SETUP),
    code(HELPERS),
    code(CACHES_CELL),
    md("## 1. Probe: frozen features for every normalisation, per-slice vs volume twin"),
    code(r'''PROBE_OUT = f"{OUT}/probe"
if RUN_PROBE:
    cmd = [sys.executable, "-m", "src.vitonly.probe", "--backbones", *PROBE_BACKBONES, "--norms", *PROBE_NORMS,
           "--crops", *PROBE_CROPS, "--prefer", PREFER, "--out", PROBE_OUT, "--cache-root", "/kaggle/input", "/kaggle/working"]
    if DEBUG: cmd += ["--limit", "40"]
    say("1. probe starts (progress every 100 patients)")
    run_cmd(cmd)
    PR = pd.read_csv(f"{PROBE_OUT}/probe_results.csv")
    PP = pd.read_csv(f"{PROBE_OUT}/probe_val_preds.csv", dtype={"pid": str})
    rows = []
    for (bb, crop, pool), g in PR[PR.backbone != "-"].groupby(["backbone", "crop", "pool"]):
        g = g.set_index("norm")
        for s, v in TWINS.items():
            if s in g.index and v in g.index:
                ks, kv = f"{bb}|{s}|{crop}|{pool}", f"{bb}|{v}|{crop}|{pool}"
                d = stats.paired_bootstrap_auc_diff(PP.label, PP[kv], PP[ks], n=1000)
                rows.append({"backbone": bb, "pool": pool, "crop": crop, "per_slice": s, "volume_twin": v,
                             "cv_twin": g.loc[v, "cv_auc"], "cv_slice": g.loc[s, "cv_auc"],
                             "cv_diff": g.loc[s, "cv_auc"] - g.loc[v, "cv_auc"],
                             "val_twin": g.loc[v, "val_auc"], "val_slice": g.loc[s, "val_auc"],
                             "val_diff": d["diff"], "val_diff_lo": d["ci"][0], "val_diff_hi": d["ci"][1], "p": d["p"]})
    TWIN_PROBE = pd.DataFrame(rows)
    TWIN_PROBE.to_csv(f"{OUT}/probe_per_slice_vs_volume.csv", index=False)
    print("per-slice minus volume twin (positive = per-slice better; CV on train is the low-noise number):")
    print(TWIN_PROBE.round(3).to_string(index=False))
    print("\ntop 10 probe rows by CV AUC:")
    print(PR.head(10)[["backbone", "norm", "crop", "pool", "cv_auc", "val_auc", "val_ci_lo", "val_ci_hi",
                       "val_auc_within"]].round(3).to_string(index=False))
else:
    say("1. probe skipped")
'''),
    md("## 2. ComBat (+ subtype) on the ViT features"),
    code(r'''DEEP_OUT = f"{OUT}/deep_combat"
if RUN_DEEP_COMBAT and os.path.isdir(PROBE_OUT):
    npz = sorted(glob.glob(f"{PROBE_OUT}/features_*.npz"))
    step9 = sorted(glob.glob("/kaggle/input/**/features_cohort/features_volume.csv", recursive=True))
    cmd = [sys.executable, "-m", "src.vitonly.features", "--out", DEEP_OUT, "--embeddings", *npz, "--prefer", PREFER,
           "--cache-root", "/kaggle/input", "/kaggle/working", "--jobs", "4"]
    if step9:
        say(f"2. step-9 features found ({step9[0]}): also scoring ViT + 3d+temporal")
        cmd += ["--features-from", os.path.dirname(step9[0]), "--refs", "volume", "--sets", "3d+temporal"]
    else:
        say("2. no step-9 output attached: ViT features only")
        cmd += ["--only-embeddings"]
    if DEBUG: cmd += ["--limit", "40"]
    run_cmd(cmd)
    DEEP = pd.read_csv(f"{DEEP_OUT}/results.csv")
    print(DEEP.head(20)[["ref", "set", "combat", "subtype", "cv_auc", "val_auc", "val_auc_within", "vs_lookup",
                         "vs_lookup_p"]].round(3).to_string(index=False))
else:
    say("2. skipped (RUN_DEEP_COMBAT = False or no probe output)")
'''),
    md("## 3. Fine-tuning DINOv2 with per-slice vs volume normalisation"),
    code(r'''def epoch_mean(d, lo=5, hi=15):
    """Mean val AUC over epochs lo..hi of a run (log.csv): less optimistic than the best epoch."""
    lg = pd.read_csv(f"{d}/log.csv")
    w = lg[(lg.epoch >= lo) & (lg.epoch <= hi)]
    return float(w.val_auc.mean()) if len(w) else float(lg.val_auc.mean())

FT_RUNS, FT_ROWS = {}, []
if RUN_FT:
    for norm in FT_NORMS:
        for seed in FT_SEEDS:
            if budget_left() < 0.6:
                say(f"3. time budget nearly used: skipping {norm} seed {seed}"); continue
            say(f"3. fine-tune {FT_CONFIG} norm={norm} seed={seed} ({len(FT_RUNS)} runs done)")
            try:
                d = finetune(FT_CONFIG, seed, [f"data.norm={norm}", *FT_EXTRA], label=f"ft_{norm}_s{seed}")
            except Exception as e:                       # one failed run must not lose the others
                say(f"3. FAILED {norm} seed {seed}: {type(e).__name__}: {e}"); continue
            m = json.load(open(f"{d}/metrics.json"))
            FT_RUNS.setdefault(norm, []).append(d)
            FT_ROWS.append({"norm": norm, "seed": seed, "best_epoch": m["best_epoch"], "val_auc_best": m["val"]["auc"],
                            "val_auc_ep5_15": epoch_mean(d), "train_auc": m["val"].get("train_auc", np.nan),
                            **{f"val_{c}": v["auc"] for c, v in m["val"]["by_cohort"].items()}})
            say(f"3. {norm} s{seed}: best-epoch val {FT_ROWS[-1]['val_auc_best']:.3f}, "
                f"epochs 5-15 mean {FT_ROWS[-1]['val_auc_ep5_15']:.3f}, train {FT_ROWS[-1]['train_auc']:.3f}")
            pd.DataFrame(FT_ROWS).to_csv(f"{OUT}/finetune_runs.csv", index=False)
            note_runs(f"ft_{norm}", FT_RUNS[norm])
    FT = pd.DataFrame(FT_ROWS)
    if len(FT):
        print(FT.round(3).to_string(index=False))
        print("\nmean over seeds:")
        print(FT.groupby("norm").mean(numeric_only=True).drop(columns="seed").round(3).to_string())
else:
    say("3. fine-tuning skipped")
'''),
    code(r'''FT_PREDS, FT_CMP = {}, None
for norm, dirs in FT_RUNS.items():
    FT_PREDS[f"DINOv2 {norm} ({len(dirs)} seeds)"] = save_preds(f"ft_{norm}_ensemble", preds_of(dirs))
if FT_PREDS:
    FT_CMP = compare({"B1 LR (clinical + size)": B1_VAL, **FT_PREDS}, "B1 LR (clinical + size)")
    print(FT_CMP.round(3).to_string(index=False))
    rows = []
    for s, v in TWINS.items():
        ks = [k for k in FT_PREDS if f" {s} (" in k]; kv = [k for k in FT_PREDS if f" {v} (" in k]
        if ks and kv:
            a, b = ev.align({"v": FT_PREDS[kv[0]], "s": FT_PREDS[ks[0]]}).values()
            dl = ev.delong(a.label, a.prob, b.prob)
            rows.append({"per_slice": s, "volume_twin": v, "auc_twin": stats.auc(a.label, a.prob),
                         "auc_slice": stats.auc(b.label, b.prob), "diff": dl["diff"], "ci_lo": dl["ci"][0],
                         "ci_hi": dl["ci"][1], "delong_p": dl["p"]})
    FT_TWINS = pd.DataFrame(rows)
    print("\nfine-tuned, seed ensembles: per-slice minus volume twin (paired DeLong)")
    print(FT_TWINS.round(3).to_string(index=False))
    FT_TWINS.to_csv(f"{OUT}/finetune_per_slice_vs_volume.csv", index=False)
    print(subtype_wide(FT_PREDS, by="cohort").to_string())
'''),
    md("## 4. Results file"),
    code(r'''from src.vitonly.features import md_table
parts = [f"# Step 10 results: per-slice normalisation for the ViT, ComBat on ViT features ({time.strftime('%Y-%m-%d %H:%M')})", ""]
if RUN_PROBE and os.path.isdir(PROBE_OUT):
    parts += ["## Probe (frozen backbones): per-slice minus volume twin", "", md_table(TWIN_PROBE), ""]
if os.path.exists(f"{DEEP_OUT}/RESULTS.md"):
    parts += [open(f"{DEEP_OUT}/RESULTS.md").read().replace("# Step 9 results", "## ComBat on ViT features (from step 9 code)"), ""]
if FT_ROWS:
    parts += ["## Fine-tuned DINOv2 (2 blocks): every run", "", md_table(pd.DataFrame(FT_ROWS)), ""]
if FT_CMP is not None:
    parts += ["## Fine-tuned seed ensembles vs B1 (paired DeLong)", "", md_table(FT_CMP), "",
              "## Fine-tuned: per-slice minus volume twin", "", md_table(FT_TWINS), ""]
open(f"{OUT}/RESULTS_step10.md", "w").write("\n".join(parts) + "\n")
say(f"wrote {OUT}/RESULTS_step10.md")
'''),
    code(PUSH.replace("if PUSH_RESULTS:", "if PUSH_RESULTS and not DEBUG:")),
]


def step1_for(cohort, letter, size):
    """step1 with COHORT fixed: one notebook per cohort, so each keeps its own output (Kaggle makes a Dataset
    from a notebook's latest version only; re-running one notebook with another COHORT would replace it)."""
    cells = [dict(c) for c in NB_STEP1]
    cells[0] = md(f"""# Step 1{letter}: download the FULL {cohort.upper()} cohort ({size}) -> Dataset `mnj-raw-{cohort}`
This is **1 of 3** download notebooks; together they are the full BreastDCEDL dataset (23.85 GB, 2,070 patients):
`step1a_download_ispy1` (1.6 GB) + `step1b_download_ispy2` (13.6 GB) + `step1c_download_duke` (8.6 GB).
A Kaggle notebook can save at most 20 GB, so the full set cannot come out of a single notebook.
Import all three as **separate notebooks** and run them (in parallel is fine). Nothing needs editing.

1. Kaggle -> **Create -> New Notebook -> File -> Import Notebook** -> this file.
2. Session options: **Accelerator None**, **Internet ON**.
3. **Save Version -> Save & Run All (Commit)**; you can close the browser.
4. When finished: open the version -> **Output** -> log ends with `ALL MD5 OK` -> **New Dataset** named
   **`mnj-raw-{cohort}`** (Private)."""
    )
    cells[1] = code(f'''COHORT = "{cohort}"     # fixed for this notebook; the other cohorts have their own step1 notebook
PARTS = 8''')
    return cells


NOTEBOOKS = {"step0_demo_download_check.ipynb": NB00,
             "step1a_download_ispy1.ipynb": step1_for("ispy1", "a", "1.6 GB incl. metadata + weights"),
             "step1b_download_ispy2.ipynb": step1_for("ispy2", "b", "13.6 GB"),
             "step1c_download_duke.ipynb": step1_for("duke", "c", "8.6 GB"),
             "step2_build_cache_v2.ipynb": NB02, "step3_input_diagnostics.ipynb": NB01,
             "step4_probe_sweep.ipynb": NB03, "step5_finetune_vit.ipynb": NB04,
             "step6_reproduce_akula_cache_v2.ipynb": NB06, "step7_regularisation_and_fusion.ipynb": NB07,
             "step8_statistics_loco_final_test.ipynb": NB08,
             "step9_features_combat_per_slice.ipynb": NB09, "step10_per_slice_vit_deep_combat.ipynb": NB10}


def main():
    for old in [*HERE.glob("0*_*.ipynb"), HERE / "step1_download_full_dataset.ipynb"]:   # earlier names
        old.unlink(missing_ok=True)
    for name, cells in NOTEBOOKS.items():
        (HERE / name).write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
