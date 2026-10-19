# Branch `vaib`: run the improved ViT-only pipeline on Kaggle with the FULL real dataset

Everything here runs on the **real BreastDCEDL data**: all 2,070 patients, of which 1,452 have pCR labels.
Synthetic data is used only by the local unit tests (`tests/`), which need no download. Background and
findings: `docs/vaib/EXPLAINED.md`. The team's `kaggle/*.ipynb` are unchanged.

## What goes to Kaggle
**Notebooks.** Upload with Kaggle -> Create -> New Notebook -> File -> Import Notebook.

| step | notebook (this folder) | accelerator | attach as input | runs | save output as Dataset |
|---|---|---|---|---|---|
| 0 (optional) | `step0_demo_download_check.ipynb` | None | – | 1 | – (a 7-patient sample, just to see the data) |
| **1a** | `step1a_download_ispy1.ipynb` | None | – | 1 | `mnj-raw-ispy1` (1.6 GB) |
| **1b** | `step1b_download_ispy2.ipynb` | None | – | 1 | `mnj-raw-ispy2` (13.6 GB) |
| **1c** | `step1c_download_duke.ipynb` | None | – | 1 | `mnj-raw-duke` (8.6 GB) |
| **2** | `step2_build_cache_v2.ipynb` | None | the matching `mnj-raw-<cohort>` + code | **3** | `mnj-cache2-ispy1`, `mnj-cache2-ispy2`, `mnj-cache2-duke` |
| **3** | `step3_input_diagnostics.ipynb` | None | all 3 `mnj-cache2-*` + code | 1 | – |
| **4** | `step4_probe_sweep.ipynb` | GPU T4 x2 | all 3 `mnj-cache2-*` + code | 1 | – (read the TOP line) |
| **5** | `step5_finetune_vit.ipynb` | GPU T4 x2 / P100 | all 3 `mnj-cache2-*` + code | 1 | – (results in Output) |

Every notebook needs **Internet ON** (Session options; requires a phone-verified Kaggle account).

**Datasets.** You create them yourself from the notebook outputs (all Private):
- `mnj-raw-ispy1` (1.6 GB), `mnj-raw-ispy2` (13.6 GB), `mnj-raw-duke` (8.6 GB): together these are the full raw
  release, 23.85 GB, all 2,070 patients. Three Datasets are needed only because one Kaggle notebook can save at
  most 20 GB.
- `mnj-cache2-ispy1` (1.1 GB), `mnj-cache2-ispy2` (8.2 GB), `mnj-cache2-duke` (2.1 GB): every pCR-labeled
  patient, ready for training.
- If your teammates already have `mnj-raw-*`, they can share those instead, and you skip step 1.

**The repo code** (steps 2-5). The GitHub repo is private, so choose one:
- **(a) GitHub token:** Kaggle notebook -> Add-ons -> Secrets -> add `GITHUB_TOKEN` (a GitHub personal access
  token with read access to the repo). The notebook clones branch `vaib`.
- **(b) No token:** upload the code as a Dataset. On your laptop:
  `git -C MRI_chemo_response archive --format=zip --prefix=MRI_chemo_response/ -o mnj-code-vaib.zip vaib`
  (a ready copy is in `../kaggle_upload/`, next to the repo). Then Kaggle -> Datasets -> New Dataset -> upload
  the zip -> name `mnj-code-vaib` -> attach it to every step-2-5 notebook.
  Make a new zip after every `git pull` of `vaib`.

In every step-2-5 notebook set `PERSON` ("A", "B" or "C", your role in the team) in the first cell.

## Which cache to attach in steps 3-5 (after merging main, 2026-10-19)
Three kinds of cache now exist. The notebooks take the best one attached for each cohort:
1. **`mnj-cache2-<cohort>`** (this branch, step 2): raw float16, nothing clipped. Preferred.
2. **`mnj-cache-v2`** (main, `kaggle/01_build_cache.ipynb`): uint8, resampled to 0.7 mm x 2 mm, tumour clip
   ceiling raised. Much less saturation than builder 1. The code reads its stored 0.7 mm spacing.
3. **`mnj-cache-ispy1/-ispy2/-duke`** (builder 1): uint8, native pixels, tumour often clipped.

Attaching a better one alongside an older one is safe: the better one wins (printed as `v2 raw`,
`v1 uint8 builder 2` or `v1 uint8 builder 1`).

## Order and timing
1. **step1a, 1b, 1c** as three separate notebooks, in parallel; nothing to edit: ~30 min (I-SPY1), ~2-4 h (I-SPY2), ~1.5-3 h (Duke). Check that the log ends with
   `ALL MD5 OK`, then New Dataset from the Output.
2. **step2** x3, each with its `mnj-raw-*` attached: 20-60 min each. New Dataset `mnj-cache2-<cohort>`.
   Leave `LIMIT = 0` (= every patient). `ALL_PATIENTS = True` also stores the 618 patients without a pCR label;
   they are not used for training or scoring.
3. **step3** (optional but recommended for the report): saturation, tumour size and the hand-crafted signal check.
4. **step4:** first a quick run with `LIMIT = 60` (a few minutes), then `LIMIT = 0` (~30 min). The last line prints
   `TOP: backbone=... norm=... crop=...`.
5. **step5:** copy the TOP values into `BACKBONE`, `NORM` and `CROP`. First run with `DEBUG = True` (minutes),
   then `DEBUG = False` (3 seeds, ~1.5 h) with Save Version -> Save & Run All (Commit). `RUN_CONTROL = True`
   also runs the old B3 recipe as a check (~0.53).

`LIMIT` and `DEBUG` are the only settings that use part of the data, and only for quick checks. With
`LIMIT = 0` / `DEBUG = False` every patient in the train and val splits is used. The test split is never
used, by design.

## Steps 6-8: project update 2
| step | notebook | what it does | attach | GPU time |
|---|---|---|---|---|
| 6 | `step6_reproduce_akula_cache_v2.ipynb` | reproduces Akula's B3 on cache v2 (0.559) with his exact procedure; probe on cache v2; our ViT-only model with the probe's top setting (3 seeds) vs B3, with paired DeLong / bootstrap | `mnj-cache-v2` + code | ~3 h |
| 7 | `step7_regularisation_and_fusion.ipynb` | **our pipeline** (our normalisation, any cache): **A** best normalisation (dinov2 + sub vs mae + kinetic, 3 seeds); **B, point 1** regularisation of the best (frozen, 2 blocks, area slices, strong aug, dropout) with train AUC / gap; **C, point 2** fusion with clinical data (`vo_fusion`), AUC overall / per cohort / **within subtypes**, DeLong vs B1 | `mnj-cache-ispy1/-ispy2/-duke` + code | ~6-8 h |
| 8 | `step8_statistics_loco_final_test.ipynb` | **point 3:** DeLong CIs / tests vs B1, sens / PPV / NPV at 90% specificity, calibration, subtype / cohort AUCs for every model; **point 4:** leave-one-cohort-out of our ViT-only + fusion (best setting from step 7) and B1; the final test (`FINAL_TEST = False` until the team agrees) | caches + code + **your step-7 notebook output** | ~2-3 h |

Step 6 needs Akula's `mnj-cache-v2` (it reproduces his cache-v2 result); skip it if you only have `mnj-cache-*`.
Each writes `results/<PERSON>/step6|7|8/` (summary CSVs, prediction files, `runs.json`) to its Output. Step 8 finds
the step 6/7 predictions and checkpoints in the attached notebook outputs (+ Add Input -> Your Work -> Notebooks).
Run each with `DEBUG = True` first (minutes), then `DEBUG = False` + Save & Run All (Commit).

## Option: download on your laptop instead of step 1
`python scripts/download_breastdcedl.py` downloads the full release into `datasets/BreastDCEDL/`. It resumes
and checks md5, and that folder is ignored by git. Then upload it as one Dataset with the Kaggle CLI (see
`datasets/BreastDCEDL/README.md`) and attach that one Dataset in step 2. At the measured 0.23 MB/s this takes
more than a day, so step 1 on Kaggle is much faster.

## Checked locally (2026-10-18)
- step1: real Zenodo downloads (metadata CSV + demo archive), md5 OK.
- step0 and step2: run on real patients from the demo archive.
- step3-5: run end to end on synthetic caches with tiny models, through the real SETUP cell (code-Dataset route).
- `python -m pytest -q tests`: 42 passed.

Regenerate the notebooks after editing: `python kaggle/vaib/make_notebooks.py`.
