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
assert PERSON in ("A", "B", "C"), 'set PERSON = "A", "B" or "C" in the parameters cell'
_tok = None
try:
    from kaggle_secrets import UserSecretsClient
    _tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass

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
    import glob
    hits = [h for d in range(1, 6) for h in glob.glob("/kaggle/input/" + "*/" * d + "src/vitonly/__init__.py")]
    assert hits, ("no GITHUB_TOKEN secret and no code Dataset attached. Either add the secret (Add-ons -> Secrets) "
                  "or attach the Dataset made from mnj-code-vaib.zip (see kaggle/vaib/README.md)")
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
PERSON = ""               # "A", "B" or "C" (your role in the team): results go to results/<PERSON>/
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
the full dataset is downloaded by `step1_download_full_dataset.ipynb`."""),
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

NOTEBOOKS = {"step0_demo_download_check.ipynb": NB00, "step1_download_full_dataset.ipynb": NB_STEP1,
             "step2_build_cache_v2.ipynb": NB02, "step3_input_diagnostics.ipynb": NB01,
             "step4_probe_sweep.ipynb": NB03, "step5_finetune_vit.ipynb": NB04}


def main():
    for old in HERE.glob("0*_*.ipynb"):        # earlier names of these notebooks (00_ ... 04_)
        old.unlink()
    for name, cells in NOTEBOOKS.items():
        (HERE / name).write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
