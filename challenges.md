# Challenges

Problems we hit while building the pipeline, what caused them and how we fixed them. Newest first.

## 1. CSV tumour boxes in original-resolution pixels (2026-10-09)

**Symptom.** The first full I-SPY1 cache build (`d1`) gave 158 `ok` and 14 `error`. All 14 errors said
`in-plane 256x256 != n_xy 512; ValueError: bbox ... outside volume`. One more patient was `ok`
but had a mask vs CSV box IoU of 0.0. Its crop was in the wrong place, and no error was raised.

**Cause.** The metadata box (`sraw/eraw/scol/ecol`) is in the pixels of the original scan, whose
in-plane size is `n_xy`. In the BreastDCEDL archive every volume is resampled in-plane to
256×256. The Zenodo demo archive happened to contain only `n_xy` = 256 patients, so the
mismatch never showed up there. `xy_spacing` is also in original pixels: the 512-pixel I-SPY1
scans have 0.39 mm spacing, against 0.78 mm for 256, with the same 200 mm field of view.

| cohort | `n_xy` other than 256 |
|---|---|
| I-SPY1 | 15 of 172 (512) |
| I-SPY2 | 215 of 982 (384, 320, 312, 128, 448) |
| Duke | 2 of 916 (448) |

The I-SPY2 numbers are why this mattered: built with `d1`, about 1 in 5 I-SPY2 patients would
have failed or been cropped around the wrong place.

**Fix (`scripts/build_cache.py`, tag `d1.1`).**
- If the volume's in-plane size differs from `n_xy`, the CSV box is scaled by `256 / n_xy`, rounding outwards.
- If the patient has a mask (every I-SPY patient), the mask's own box is used instead. The IoU of
  the scaled CSV box against the mask is still recorded, as a check on the scaling.
- Only the 2 Duke patients with `n_xy` 448 depend on the scaled box alone.
- A new QC column, `xy_source` (`csv`, `csv_scaled` or `mask`), records which box each patient got.
  It is also counted in `cache_meta.json` and printed by `01_build_cache`.
- `log_bbox_vol` (a clinical feature) needed no change: box pixels × `xy_spacing` are in the
  same units, so the volume in mm³ was already correct.
- `tests/test_build_cache.py` now includes a 512→256 patient.

**Lessons.**
- The demo archive is not representative. Check the value ranges of every metadata column
  (`n_xy` here) before assuming they match the data.
- A box that falls inside the image can still be wrong. Keep the mask vs CSV IoU in the QC and
  look at its minimum, not only the error count.
- Any cache built with `d1` must be rebuilt with `d1.1`.

## 2. Raw Dataset not found on Kaggle (2026-10-09)

**Symptom.** `01_build_cache` stopped with `attach the mnj-raw-ispy1 Dataset`, even though the
Dataset was attached.

**Cause.** The notebook searched for a folder named `raw-<cohort>`. The Dataset slug was
`mnj-raw-ispy` (not `-ispy1`), and Kaggle mounts it under `/kaggle/input/datasets/<user>/<slug>/`.
Kaggle had also unpacked the `.tar.gz` into a folder.

**Fix (tag `d1`).** The notebook now searches `/kaggle/input` for a folder or file starting with
`BreastDCEDL_<COHORT>_min_crop`, so the Dataset name doesn't matter. If nothing is found, it prints
the input tree. `build_cache.py` already accepted either an extracted folder or a tarball.
