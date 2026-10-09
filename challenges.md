# Challenges

Problems we hit while building the pipeline, what caused them and how we fixed them. Newest first.

## 1. Results push failed at the end of the first real training run (2026-10-09)

**Symptom.** A's first full `f2_attn` run trained fine (val AUC 0.690), but the last cell stopped with
`git checkout failed: fatal: 'origin/main' is not a commit and a branch 'main' cannot be created from it`.
Nothing was pushed to GitHub. The results were still in the notebook version's Output.

**Cause.** SETUP clones only the code tag (`git clone --depth 1 --branch <tag>`). A clone like that
fetches only the tag, so `git fetch origin main` updated `FETCH_HEAD` but never created `origin/main`.
Debug runs skip the push, so the debug runs couldn't catch this.

**Fix (tag `d1.3`, `kaggle/make_notebooks.py`).**
- The push cell fetches `+refs/heads/main:refs/remotes/origin/main` explicitly.
- If a push is rejected because someone else pushed first, the commit is rebuilt on the new `main`.
  It no longer uses `git pull --rebase`, which is unreliable in a shallow clone. Each person writes only
  to `results/<PERSON>/`, so rebuilding can't conflict.
- After 3 failed attempts it prints `NOT pushed` instead of failing silently.
- Tested locally with a tag-only shallow clone and a push in between from another clone.

**Lessons.** Any step that only runs in real runs (here, the push) needs a test of its own. A green
debug run doesn't cover it.

## 2. CSV tumour boxes in the original image's pixels (2026-10-09)

**Symptom.** The first full I-SPY1 cache build (`d1`) gave 158 `ok` and 14 `error`. All 14 errors said
`in-plane 256x256 != n_xy 512; ValueError: bbox ... outside volume`. One more patient was `ok`
but had a mask vs CSV box IoU of 0.0. Its crop was in the wrong place, and no error was raised.

**Cause.** The metadata box (`sraw/eraw/scol/ecol`) is in the pixels of the original image, whose
in-plane size is `n_xy`. Every archive volume is 256×256. The Zenodo demo archive happened to
contain only `n_xy` = 256 patients, so this never showed up there.

| cohort | `n_xy` other than 256 |
|---|---|
| I-SPY1 | 15 of 172 (512), 12 train / 3 test |
| I-SPY2 | 215 of 982 (384, 320, 312, 448, and 2 at 128), 172 train / 25 val / 18 test |
| Duke | 2 of 916 (448), both train, 1 labeled |

**First guess was wrong (`d1.1`).** We assumed the volumes had been *resampled* to 256 and scaled
the box by `256 / n_xy`. The I-SPY masks showed otherwise: the scaled box against the mask gave an
IoU of only 0.11–0.27. The I-SPY1 `d1.1` cache was still correct, because it used the mask box.

**What was really done (checked on the 15 I-SPY1 patients at 512).** The originals were *cropped*,
not resampled:
- The CSV box sizes equal the mask box sizes pixel for pixel.
- Each volume is a 256-pixel window at the original resolution, starting at
  `clamp(floor((start + end) / 2) - 127, 0, n_xy - 256)` on each axis. That is, the window is
  centred on the tumour box.
- That rule matches 29 of the 30 measured offsets exactly. The exception is ISPY1_1103, whose
  tumour is 272 px tall: taller than the window, so its mask box is cut off at 256.
- So in-plane pixel size really does vary between patients: 0.39 mm for the 512 I-SPY1 scans, 0.78 mm
  for the 256 ones. That is a source of scanner/domain variation, not a bug.
- `log_bbox_vol` (a clinical feature) needed no change: CSV box pixels × `xy_spacing` are the
  original pixels, so the volume in mm³ was already correct.

**Fix (`scripts/build_cache.py`, tag `d1.2`).**
- If `n_xy` is bigger than the volume, the CSV box is shifted by the centred-window offset above
  (`xy_source = csv_shifted`).
- Where a mask exists (every I-SPY patient), the mask's box is used instead (`xy_source = mask`), and
  the IoU of the shifted CSV box against the mask is printed as a check. It should be about 1.0.
- If `n_xy` is smaller than the volume (2 I-SPY2 patients at 128), the rule is unknown. The box is
  rescaled (`csv_scaled`), and the mask overrides it anyway.
- Only Duke's 2 patients at 448 depend on the shifted box alone.
- `tests/test_build_cache.py` includes a 512-pixel patient cropped this way.

**Lessons.**
- The demo archive is not representative. Check the value ranges of every metadata column
  (`n_xy` here) before assuming they match the data.
- A box that falls inside the image can still be wrong. Keep the mask vs CSV IoU in the QC and
  look at its minimum, not only the error count.
- Test a hypothesis against ground truth before building on it. The mask IoU check in `d1.1` is
  what showed the resampling guess was wrong.
- Caches built with `d1` must be rebuilt. The I-SPY1 cache from `d1.1` is fine (mask boxes),
  but I-SPY2 and Duke should be built with `d1.2`.

## 3. Raw Dataset not found on Kaggle (2026-10-09)

**Symptom.** `01_build_cache` stopped with `attach the mnj-raw-ispy1 Dataset`, even though the
Dataset was attached.

**Cause.** The notebook searched for a folder named `raw-<cohort>`. The Dataset slug was
`mnj-raw-ispy` (not `-ispy1`), and Kaggle mounts it under `/kaggle/input/datasets/<user>/<slug>/`.
Kaggle had also unpacked the `.tar.gz` into a folder.

**Fix (tag `d1`).** The notebook now searches `/kaggle/input` for a folder or file starting with
`BreastDCEDL_<COHORT>_min_crop`, so the Dataset name doesn't matter. If nothing is found, it prints
the input tree. `build_cache.py` already accepted either an extracted folder or a tarball.
