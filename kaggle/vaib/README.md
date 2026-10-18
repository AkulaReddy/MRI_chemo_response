# Branch `vaib`: run the improved ViT-only pipeline on Kaggle with the FULL real dataset

Everything here runs on the **real BreastDCEDL data**: all 2,070 patients, of which 1,452 have pCR labels.
Synthetic data is used only by the local unit tests (`tests/`), which need no download. Background and
findings: `docs/vaib/EXPLAINED.md`. The team's `kaggle/*.ipynb` are unchanged.

## What goes to Kaggle
**Notebooks.** Upload with Kaggle -> Create -> New Notebook -> File -> Import Notebook.

| step | notebook (this folder) | accelerator | attach as input | runs | save output as Dataset |
|---|---|---|---|---|---|
| 0 (optional) | `step0_demo_download_check.ipynb` | None | – | 1 | – (a 7-patient sample, just to see the data) |
| **1** | `step1_download_full_dataset.ipynb` | None | – | **3** (`COHORT` = ispy1, ispy2, duke) | `mnj-raw-ispy1`, `mnj-raw-ispy2`, `mnj-raw-duke` |
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

## Order and timing
1. **step1** x3, in parallel: ~30 min (I-SPY1), ~2-4 h (I-SPY2), ~1.5-3 h (Duke). Check that the log ends with
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
