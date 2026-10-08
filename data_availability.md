# Data Availability — Verified Findings

**Checked:** 2026-10-06, live queries from the student's Windows machine (not a cloud sandbox). All sources below were reachable from here.
**Plan:** `IMPLEMENTATION_PLAN_FINAL.md`; decisions in `docs/DECISIONS.md`.
**Updated:** 2026-10-08 with findings from the BreastDCEDL Scientific Data paper (2026), the BreastDCEDL arXiv paper (2506.12190v3), Nishizawa et al. (2025) and the Nishizawa GitHub code. See Section 6.
**Context:** MNJ project #11, pCR prediction from breast DCE-MRI. **Terminology:** "Option 1" below = Stage 1 (pre-treatment prediction, this project); "Option 2" = Stage 2 (paired pre + post-chemo, future work). The earlier draft plans this file corrects (Section 5) have been deleted; `IMPLEMENTATION_PLAN_FINAL.md` is the only plan.

This file overrides the plan wherever they disagree.

---

## 1. Summary

| Source | Reachable | Login needed | Needed for |
|---|---|---|---|
| Zenodo BreastDCEDL (latest, record 18114231) | Yes, resumable (HTTP 206 range requests work) | No | Option 1: everything (images, masks, labels, split). Option 2: pCR labels. |
| MAMA-MIA (Synapse syn60868042) | Yes | **Yes, free Synapse account** (anonymous users can view but not download; no extra access requirement or DUA) | Only the pretrained nnU-Net weights for Module 1 (tumor localization). Optional. |
| TCIA I-SPY1 / I-SPY2 / ACRIN-6698 | Yes, NBIA API works anonymously | No for metadata; downloads expected open (CC BY), not yet tested | **Option 2 only:** the only public source of post-chemo scans |
| GitHub repos (BreastDCEDL, MAMA-MIA, Nishizawa bc_pcr_prediction) | Yes | No | Code and reference preprocessing |

**Bottom line**
- **Option 1:** Zenodo alone is enough. MAMA-MIA is needed only for its pretrained segmentation weights. Instead, you can train your own segmenter on BreastDCEDL's I-SPY masks, which also avoids leakage (the MAMA-MIA nnU-Net was likely trained on the same patients). However, the Zenodo I-SPY2 and Duke volumes are tumor-centered crops, so evaluating localization on a full scan needs the TCIA full-res BreastDCEDL-ISPY2 or MAMA-MIA (Section 2).
- **Option 2:** TCIA is required. Zenodo, MAMA-MIA and Duke contain pre-treatment scans only. About **1,020–1,040 labeled paired patients** exist, far more than the plan's estimate of "about 160 to a few hundred".

---

## 2. Zenodo — BreastDCEDL

- Concept DOI: 10.5281/zenodo.15627232 (always resolves to the newest version)
- **Use record 18114231** (the newest, published 2025-09-24). `https://zenodo.org/records/15627233/latest` redirects there.
- Versions: 15627233 (v1.0, June 2025, **no I-SPY2**), 17274053, 17580237, 18114231.
- License: CC BY-NC 4.0 (non-commercial, because of the Duke data).
- Code: https://github.com/naomifridman/BreastDCEDL

### Files in record 18114231

| File | Size | md5 |
|---|---|---|
| `BreastDCEDL_ISPY2_min_crop.tar.gz` | 13.634 GB | 2f1a00310bd00d344664319fb7bd679c |
| `BreastDCEDL_DUKE_min_crop.tar.gz` | 8.643 GB | 16e6c752b9679138e88c188d7a01895f |
| `BreastDCEDL_ISPY1_min_crop.tar.gz` | 1.252 GB | d2d2630940a861819e6a385877f45ba5 |
| `BreastDCEDL_spy1.zip` (older I-SPY1 package from v1.0) | 2.998 GB | e6509c6efa5b0e3370528e8a12ed6b23 |
| `BreastDCEDL_models.tar.gz` (pretrained ViT weights) | 0.320 GB | 8096eea314a326abd284b7d95837c0d5 |
| `BreastDCEDL_demo_data.tar.gz` | 0.055 GB | 5b7d8701309c2c3807f8d126f7ddcede |
| `BreastDCEDL_metadata_min_crop.csv` | small (2,071 lines) | d58b84d0f343b4b8338cbb222e6dda08 |
| `BreastDCEDL_dataset.pdf`, `LICENSE`, `overview_breastdcedl.png` | small | |

- **The three `*_min_crop` cohort archives plus the models file total about 23.9 GB**, not the 12 GB the plans state.
- Download pattern: `https://zenodo.org/records/18114231/files/<FILE>?download=1`. Use `curl -L -C - -o <FILE> <url>` (resumable), then verify the md5.
- **Speed (measured 2026-10-08 from the project laptop):**
  - **Per-connection limit:** 1 connection = 13–65 KB/s; 4 parallel connections ≈ 80–90 KB/s each; 8 parallel connections ≈ 0.9–1.2 MB/s total. The laptop's line does about 3.9 MB/s elsewhere.
  - **From Kaggle (measured 2026-10-08, single connection):** I-SPY1 1.25 GB at 1.00 MB/s (21 min), models 0.32 GB at 0.78 MB/s, md5 OK for all three of A's files. One connection from Kaggle is about 15× faster than one from the laptop.
  - **Distance:** round trip to Zenodo (CERN, Geneva) is about 258 ms.
  - **Overload:** Zenodo intermittently answers `504 Gateway Time-out`, a 92-byte error body that naive scripts save as the file.
  - Response headers show `x-ratelimit-limit: 133` requests per minute (a request-count limit, not bandwidth).
  - **Use `kaggle/00_download_zenodo.py`:** 8 parallel byte ranges, `curl --fail`, exact sizes + md5, back-off on 504, and resume across sessions. Tested here: md5 OK on the demo archive, including a resume after an interruption.
- File list via API: `https://zenodo.org/api/records/18114231`
- **"min_crop" (mostly answered by the Scientific Data paper):** I-SPY2 is "cropped to 256 × 256 × Z containing tumor regions" and Duke is "cropped to 256 × 256 × Z tumor-centered volumes". I-SPY1 is described as the "complete dataset with all DCE-MRI acquisitions". So the I-SPY2 and Duke Zenodo volumes **cannot test full-field-of-view tumor localization (Module 1)**. For that, use the full-resolution I-SPY2 NIfTI on TCIA ("BreastDCEDL-ISPY2", DOI 10.7937/42WQ-TH78, about 38.2 GB of volumes plus 59.1 MB of masks), MAMA-MIA, or I-SPY1 if it turns out to be uncropped. Still confirm by inspecting one volume of each cohort.
- **Format:** 3D NIfTI per DCE phase (`<id>_<ds>_acq<N>.nii.gz`) plus `<id>_mask.nii.gz`, stored as float64 with original DICOM intensities. **No intensity normalization or resampling has been applied.** The RGB fusion and 8-bit conversion used for the ViT are done in the authors' code, not in the data. We must build the 2D RGB slices ourselves.
- **Checked on the Zenodo demo archive (`BreastDCEDL_demo_data.tar.gz`, 7 patients, 2026-10-08):**
  - File names are `{pid}_spy1_vis1_dce_aqc_{k}.nii.gz`, `{pid}_spy2_vis1_dce_aqc_{k}.nii.gz`, `{pid}_duke_dce_aqc_{k}.nii.gz` and `{pid}_spy{1,2}_vis1_mask.nii.gz`, in `dce/` and `mask/` folders. This differs from `utils/data_utils.py`. Only the three phases used are shipped.
  - Arrays are `(Z, 256, 256)` float64, **axis 0 = slice**, including I-SPY1. The header spacing is 1 mm (wrong), so use the CSV `xy_spacing`/`slice_thick`.
  - **In-plane CSV boxes (`sraw/eraw` = axis 1, `scol/ecol` = axis 2) match the masks exactly** (IoU 1.0). Ends are exclusive, and so is `mask_end`.
  - **The demo I-SPY volumes are also cropped in z** (e.g. ISPY1_1001: 30 slices, mask at z 2–27). But the **Zenodo CSV gives z for the uncropped depth** (`n_z` 60, mask 12–38); the GitHub min_crop CSV matches the demo instead. Duke depths match the Zenodo CSV.
  - The demo Duke_001 ships phase 3, but the Zenodo CSV says `post_late = 4`.
  - Both are handled in `scripts/build_cache.py`: the CSV z-range is used only when depth equals `n_z`, otherwise the mask's z-range is used; a missing phase falls back to the nearest available one. Every case is logged in `cache_index.csv`. **Check `z_source` and `phase_fallback` counts on the real archives (day 1).**
- **Annotations:** I-SPY1 and I-SPY2 have binary 3D masks (1,154). These were decoded from the trials' analysis-mask bit flags (a manual VOI plus an enhancement threshold), so they are **semi-automatic FTV-style masks, not hand-drawn outlines**. Duke has **bounding boxes only** (916), covering only the largest tumor. Only the primary tumor is annotated in any cohort, even when the disease is multifocal (up to 9 lesions per Duke patient).

### Metadata CSV (downloaded and verified)

Columns: `pid, pCR, n_xy, n_z, n_times, pre, post_early, post_late, slice_thick, xy_spacing, mask_start, mask_end, sraw, eraw, scol, ecol, tum_vol, age, menopause, race_white, race_black, HR, HER2, HR_HER2_STATUS, TripleNeg, HER2pos, HRposHER2neg, dataset, test`

- `dataset` values: `spy1`, `spy2`, `duke`
- `pre/post_early/post_late` give the DCE phase indices that map to R/G/B. All three are phases of **one pre-treatment scan**.

| Cohort | Patients | pCR-labeled | pCR rate |
|---|---|---|---|
| I-SPY1 (`spy1`) | 172 | 172 | 28.5% |
| I-SPY2 (`spy2`) | 982 (717 `ISPY2-*` + 265 `ACRIN-*`) | 982 | 32.2% |
| Duke (`duke`) | 916 | 298 | 21.1% |
| **Total** | **2,070** | **1,452** | **29.5%** |

**Split column `test`:** `0.0` = train (1,532 rows, including unlabeled patients), `1.0` = test (270 rows), `2.0` = second held-out set, probably validation (268 rows).
- **Labeled test patients: 176** (I-SPY1 35, I-SPY2 99, **Duke 42**). The plan's "Duke 43" was wrong, which explains its 177-vs-176 mismatch.
- Test rows per cohort, labeled or not: spy1 35, spy2 99, duke 136. Held-out set 2: spy1 33, spy2 99, duke 136.
- **Resolved:** The Scientific Data paper (Table 4) defines `test` as 0 = train, 1 = test, 2 = val. Labeled counts are train 1,099 (29.3% pCR), val 177 (29.9%) and test 176 (30.1%). Totals including unlabeled patients are 1,532 / 268 / 270.
- **Three different metadata CSVs exist** (details in Section 6.4). **Use the Zenodo `BreastDCEDL_metadata_min_crop.csv` (md5 d58b84d0…) as the canonical split.** In it, `test==1` has 176 labeled patients and `test==2` has 177.
- **The arXiv results were computed on none of the released splits.** The article's own prediction file has 175 patients (Duke 41). Five of them are not in Zenodo `test==1` (4 are in val, `ISPY1_1173` is in **train**), and 6 Zenodo test patients are missing from it. Our results on Zenodo `test==1` are therefore close to, but not exactly comparable with, the published numbers.

**Phase indices (`pre`, `post_early`, `post_late`):** for I-SPY the arXiv paper uses SER timepoints 0, 2 and min(last, 6). In the Zenodo CSV, I-SPY is mostly (0, 2, 5) or (0, 2, 6), with about 140 I-SPY1 patients at (0, 1, 2). **Duke is `post_early = 1` for all 916 patients**, with late = 3 or 4 (the last phase). This matches the arXiv paper ("0, 1, last"), not the Scientific Data paper's "index 2". Use the CSV columns as the source of truth.

**Cohort-specific biomarker definitions** (affects HR/HER2 inputs and any cross-cohort comparison):

| | I-SPY2 | I-SPY1 | Duke |
|---|---|---|---|
| HR+ threshold | ≥1% staining | ≥10% staining | Allred >3 |
| HER2+ threshold | FISH ≥2.0 | FISH ≥2.0 | FISH >2.2 |

Tumor volume (`tum_vol`) is computed from the masks for I-SPY but is a five-level size class for Duke, so it is not comparable across cohorts.

**Missing clinical values (of 2,070 patients):** menopause 210, HER2 5, age 3, race 3, HR 2.

**Patient ID formats:** `ISPY1_1001`; `ISPY2-756412` / `ACRIN-...` (the I-SPY2 cohort includes ACRIN-6698 patients); Duke IDs not yet inspected. To join with TCIA, match on the numeric part after the last `-`.

---

## 3. MAMA-MIA — Synapse

- Project: `syn60868042`. Restriction level OPEN with no unmet access requirement, but **downloading needs a logged-in, free Synapse account**. Use `pip install synapseclient`, then `syn.login(authToken=...)`.
- GitHub: https://github.com/LidiaGarrucho/MAMA-MIA
- Top-level contents:

| ID | Name | Type |
|---|---|---|
| syn61247992 | `nnUNet_pretrained_weights/` → **syn61906708 `full_image_dce_mri_tumor_segmentation.zip`** | folder → file |
| syn64871114 | `images/` (one folder per patient, e.g. `DUKE_001`) | folder |
| syn64871173 | `segmentations/` | folder |
| syn64889927 | `patient_info_files/` | folder |
| syn64854989 | `clinical_and_imaging_info.xlsx` | file |
| syn60880777 | `train_test_splits.csv` | file |
| syn61338741 | `preliminary_automatic_segmentations_quality_scores.csv` | file |

- For Module 1, download **only** syn61906708 (weights) and syn60880777 (split). Use the split to evaluate the nnU-Net only on cases it wasn't trained on.
- File sizes aren't visible anonymously; check them after logging in.
- Example: `synapseclient.Synapse().login(authToken=TOKEN); syn.get("syn61906708", downloadLocation="weights/")`

---

## 4. TCIA — needed only for Option 2 (post-chemo scans)

The NBIA API works anonymously: `https://services.cancerimagingarchive.net/nbia-api/services/v1/` with `getPatient`, `getPatientStudy`, `getSeries`, each called as `?Collection=<name>&format=json`.

| Collection | Patients | Studies | Timepoint labels |
|---|---|---|---|
| ISPY1 | 222 | 847 | **Not labeled.** Study descriptions are free text, so order each patient's studies by `StudyDate`. |
| ISPY2 | 719 | 2,688 | `ISPY2_MRI_T0` 717, `_T1` 690, `_T2` 647, `_T3` 634 |
| ACRIN-6698 | 385 | 1,123 | `ACRIN-6698_ISPY2_MRI_T0` 385, `_T1` 258, `_T2` 243, `_T3` 237 |

I-SPY2 timepoints: **T0 = baseline (pre-treatment), T3 = pre-surgery (post-chemo).** Use T0 + T3.

### Paired-patient yield (baseline + pre-surgery study both present)

| Cohort | Paired patients | With BreastDCEDL pCR label |
|---|---|---|
| ISPY2 (T0 + T3) | 633 (595 have all 4 timepoints) | |
| ACRIN-6698 (T0 + T3) | 237 (228 have all 4) | |
| **I-SPY2 combined** (no patient overlap between the two collections) | **870** | **869, 34.2% pCR** |
| ISPY1 (≥2 studies) | 216 (studies per patient: 1→1 patient, 2→5, 3→28, 4→188) | **171** (≥2 studies); **148** (all 4 visits) |
| **Total labeled pairs** | | **About 1,020–1,040** |

- pCR labels for Option 2 can come from the BreastDCEDL metadata CSV (join on the numeric patient ID), so the separate TCIA clinical spreadsheets are needed only for extra covariates such as treatment arm or RCB.
- **Leakage warning:** 91 of the 869 labeled I-SPY2 pairs are in BreastDCEDL's official test set (`test==1.0`). If Option 1 weights initialize Option 2, keep these patients (and any Option 2 test patients) out of Option 1 training, or transfer only self-supervised weights.
- I-SPY1 caveat: "first study = baseline, last study = pre-surgery" is an assumption based on dates. Nishizawa's code does not settle it either (the preprocessing that assigns I-SPY1 timepoints is not in the repo). Verify it against the I-SPY1 docs.
- **Timepoint mapping, confirmed in Nishizawa's code:** their data loader reads `image_t0_first` … `image_t3_first` and the model uses `t0` and `t3`. So the paper's "T1 (pre) / T4 (post)" is TCIA's I-SPY2 **T0 / T3**. The loader requires all four timepoints, which likely explains part of their 985 → 660 drop.
- **Series selection hint from their README:** they used the TCIA "unilateral original cropped DCE-MRI" series and used the sizes of the SER / PE2 / PE6 folders to locate the first post-contrast slices. This tells us which I-SPY series to filter for in the DCE-series check below.

### Not yet done (Option 2 go/no-go)
1. **Check that DCE series exist within each paired study.** Use `getSeries` per collection and classify by `SeriesDescription`. The full `getSeries` lists are large and take more than 5 minutes per collection, so use `curl --compressed` with a long timeout or no `-m`, or query per study (`getSeries?StudyInstanceUID=...`). A first attempt failed because a 300 s curl timeout truncated the JSON.
2. **Estimate the DCE-only download size.** Sum the `FileSize` field from `getSeries` (if present) for the selected series, or use the NBIA Search cart.
3. **Test an actual image download** (`getImage?SeriesInstanceUID=...`) to confirm no login is needed.
4. Confirm the pCR definition is the same across I-SPY1 and I-SPY2.

---

## 5. Corrections to the earlier draft plans (deleted; kept here as a record)

| Plan statement | Verified fact |
|---|---|
| "I-SPY2 does not appear in the Zenodo files" | It is in record 18114231 (`BreastDCEDL_ISPY2_min_crop.tar.gz`, 13.6 GB) |
| BreastDCEDL is "about 12 GB" | About 24 GB for the three `*_min_crop` cohorts plus models |
| Test split = I-SPY1 35 / I-SPY2 99 / Duke 43 | Duke **42**, total 176 |
| v1: Duke 916 counted among the 1,452 labeled | Duke has 298 labeled; 916 counts all Duke patients, labeled or not |
| Option 2: "roughly 160 to a few hundred" labeled pairs | About 869 (I-SPY2) + 148–171 (I-SPY1) ≈ 1,020–1,040 |
| Option 2: I-SPY2 "T0 (optional baseline)" | T0 is *the* baseline; T3 is pre-surgery |
| v1: TCIA needs account + DUA | The metadata API is anonymous; image download not yet tested |
| "Cloud sandbox cannot reach TCIA/Zenodo/Synapse/GitHub" | Doesn't apply on this local machine; all are reachable |
| MAMA-MIA "free account" | Confirmed: login needed to download; no access requirement or DUA |
| v1: BreastDCEDL ships "RGB-fused 2D images" | It ships raw-intensity float64 3D NIfTI per phase; RGB fusion is done in code |
| I-SPY masks are "expert-verified" (Dice target) | BreastDCEDL I-SPY masks are semi-automatic threshold (FTV-style) masks; Duke has boxes only |
| Option 1 plan: train our own segmenter on BreastDCEDL masks | Possible only for I-SPY (1,154 masks), and the Zenodo I-SPY2/Duke crops are tumor-centered, so full-FOV evaluation needs TCIA BreastDCEDL-ISPY2 or MAMA-MIA |
| The 0.85 result is "ViT features → Random Forest + clinical" | The RF also uses **tumor volume and bounding-box volume from the ground-truth masks**, which are its top features (arXiv Fig. 4). At deployment those must come from Module 1 |
| The 0.72 baseline is on the 176-patient test set | It is on a 175-patient set (Duke 41) that matches no released split; see Section 6.4 |
| The 0.85 is ViT features + "clinical" | The RF inputs are HR, HER2, **MammaPrint**, age, ethnicity, race, subtype flags, **13 I-SPY2 treatment-arm drug indicators**, **tumor volume and bounding-box volume** (ground-truth masks), plus **two scalar ViT scores** (mean/min slice pCR probability, not ViT feature embeddings). Most of the 0.85 is non-imaging; see Section 6.4 |
| The shipped ViT weights reproduce 0.72 | The repo's own all-patient ViT prediction file scores AUC about 0.55 on test (predictions range 0.47–0.58); see Section 6.4 |
| Duke early post-contrast = index 2 (Scientific Data) | The CSVs use index 1 for all Duke patients |
| BreastDCEDL ViT = "ImageNet ViT-B/16" | HuggingFace `ViTForImageClassification` initialised from **`facebook/vit-mae-base`** (MAE self-supervised ImageNet weights, not supervised ViT-B/16) |
| Nishizawa's "T1/T4" naming is unknown | It is TCIA I-SPY2 T0/T3 (confirmed in code) |
| Nishizawa: "batch 4, Adam, lr 1e-3, early stopping" | Code default is AdamW (lr 1e-3, wd 5e-6, 100 epochs) and has no early stopping; see Section 6.3 |

---

## 6. Findings from the papers and the Nishizawa code (2026-10-08)

### 6.1 BreastDCEDL arXiv paper (2506.12190v3), source of all the BreastDCEDL benchmark numbers

ViT input: 2D RGB slices from tumor-containing slices (R = pre, G = early post, B = late post), min-max normalized to 8-bit. The paper does **not** state the ViT variant, pretraining, slice selection, slice-to-patient aggregation, hyperparameters or decision threshold. Get these from the GitHub notebooks (`BreastDCEDL_vit_predict.ipynb`) and the shipped weights.

ViT imaging-only results (arXiv Table 2):

| Subset | N | pCR rate | AUC | Sens | Spec |
|---|---|---|---|---|---|
| Overall | 177 | 30.1% | **0.72** | 0.27 | 0.95 |
| I-SPY1 | 35 | 34.3% | 0.68 | 0.17 | 1.00 |
| I-SPY2 | 99 | 32.3% | **0.78** | 0.28 | 0.94 |
| Duke | 43 | 21.0% | 0.54 | 0.22 | 0.91 |
| HR−/HER2− | 63 | 36.5% | 0.63 | | |
| HER2+ | 43 | 42.0% | 0.68 | | |
| HR+/HER2− | 69 | 16.0% | 0.64 | 0.10 | 1.00 |

I-SPY2 "with clinical features" (ViT features + clinical + demographic + **tumor features**, Random Forest, two-stage):

| Subset | N | pCR rate | AUC |
|---|---|---|---|
| Overall | 99 | 32.3% | **0.85** |
| HR−/HER2− | 40 | 47.5% | 0.75 |
| HER2+ | 19 | 42.1% | 0.84 |
| HR+/HER2− | 40 | 12.5% (**5 pCR cases**) | 0.94 |

Implications:
- The 0.94 rests on **5 positive patients**, so its confidence interval is very wide. Never use it as a target.
- The 0.85 is not "image + clinical". The code (Section 6.4) shows it is clinical + **MammaPrint** + **I-SPY2 treatment arm** + **ground-truth tumor volume**, plus two weak scalar ViT scores. Tumor volume was the top-ranked feature. A fair comparison needs (a) a clinical + tumor-volume-only baseline, and (b) the same model with tumor volume taken from Module 1's predicted masks.
- Sensitivity at the default threshold is very low (0.27). Report threshold-free metrics plus sensitivity at fixed specificity.
- The authors list as limitations: 2D input, RGB fusion blurring phase information, 8-bit quantization, fixed ViT patches, and the two-stage pipeline. They name end-to-end transformer fusion with clinical data as future work, which supports our Module 3 novelty claim.

### 6.2 BreastDCEDL Scientific Data paper (2026)

This is a dataset descriptor with no model results. Its key facts (format, crops, masks, split definition, biomarker thresholds) are folded into Section 2. It also says conversion of the mid-treatment I-SPY timepoints is planned for a future release. Several DOIs in its reference list look wrong (for example, Dammu et al. is listed as IEEE TMI; it is PLoS ONE 2023, doi 10.1371/journal.pone.0280148). Cite primary sources directly.

### 6.3 Nishizawa et al. (2025): paper vs code (https://github.com/TaishiNishizawa/bc_pcr_prediction)

**The repo is incomplete and does not run as published.** It has 4 Python files and is missing `resnet.py`, `combined_loss.py`, all DICOM preprocessing and the data-list JSONs. File paths in `main.py` are empty strings, one import path is broken, and `forward()` calls an undefined `self.encoder1`. Use it as an architecture reference only; we must re-implement it.

What the code actually does:

| Aspect | Paper | Code |
|---|---|---|
| Image input | first post-contrast, T1 + T4 | `image_t0_first`, `image_t3_first` (TCIA T0/T3); loader also loads t1/t2 |
| Encoder | 3D ResNet-50 | 3D ResNet-50 (Hara et al.), **one encoder shared** by both timepoints; a checkpoint is loaded before training (source/pretraining not stated) |
| Image embedding | "matching dimension" | 2048 → 500 → 100 → 30 → **8** (LeakyReLU MLP) |
| Clinical branch | binary HR/HER2, age 0–1, race/menopause embeddings | [HR, HER2, age] + race embedding (6→3) + menopause embedding (3→2) → Linear(8→8) + dropout 0.1 + BatchNorm |
| Fusion | transformer block, multi-head self-attention | 3 tokens of dim 8, `nn.MultiheadAttention(8, 2 heads)` + residual/LayerNorm + FFN (8→24→8) + residual/LayerNorm |
| Head | FC + sigmoid | **flatten the 3 tokens (24-d) → Linear → 1 logit** (no pooling, no CLS token) |
| Volume | 80×256×256 | `CropOrPad(80,256,256)` + per-volume `RescaleIntensity(0,1)`; README says resampled to spacing [0.7, 0.7, 2.0] |
| Augmentation | flips (Methods); translation/rotation (Discussion) | one random flip along one axis, but the flip probability is drawn **once per loader** as 0 or 1, so a training run either always flips or never flips. Effectively no augmentation |
| Class imbalance | 0.5 BCE + 0.5 focal | combined loss (file missing) **plus oversampling positives** by ⌊neg/pos⌋, not mentioned in the paper |
| Optimizer | Adam, lr 1e-3, batch 4 | default AdamW, lr 1e-3, weight decay 5e-6, batch 4, 100 epochs |
| Early stopping / model selection | "early stopping" | none in code; `train()` receives the validation loaders but never uses them |
| Decision threshold | not stated | 0.4 |
| Internal evaluation | 5-fold CV, 528/132 | fold indices are empty strings in the repo; there is no separate I-SPY2 test set, so if early stopping used the validation fold, the internal 0.73 is optimistic |

Results that matter for us: on external I-SPY1, **clinical-only (0.71) equals the fused model (0.71)**. Imaging-only with pre + post scans reached only 0.68 internal / 0.70 external. Shuffling features showed HER2, then HR, drives most of the performance. Their limitations section suggests tumor segmentation to focus the model, and names ViTs as an alternative. Both point directly at our Module 1 + Module 2 design.

### 6.4 BreastDCEDL GitHub code (https://github.com/naomifridman/BreastDCEDL, checked 2026-10-08)

The repo is 1.27 GB, mostly 2,764 sample DICOMs plus sample NIfTIs. Don't clone it; fetch individual files from `raw.githubusercontent.com`. Read so far: `README.md`, `ISPY2/README.md`, `utils/data_utils.py`, `BrestDCEDL_vit_predict.ipynb`, `plot_pCR_article_results.ipynb`, `BrestDCEDL_demo.ipynb`, both metadata CSVs, `df_pcr_pred_test_article.csv` and `transformer_models/BreastDCEDL_vit_pcr_predictions.csv`. **Not read** (GitHub raw downloads were too slow): `DUKE/crop_spy2_spy1.ipynb` (how min_crop was made), `duke_crop.ipynb`, the two large demo notebooks, and the per-cohort metadata/conversion notebooks.

**No training code is published**, either for the ViT or for the per-slice dataset build. Only an inference notebook and a results/plotting notebook exist.

**ViT inference pipeline (`BrestDCEDL_vit_predict.ipynb`):**
- Model: HuggingFace `ViTForImageClassification.from_pretrained('facebook/vit-mae-base')` with a 2-class softmax head, then `breastdcedl_pcr_vit_model_weights.pth` from Zenodo. That is a ViT-B/16 initialised from **MAE self-supervised** weights.
- Slices: the **4 slices** `idx-2 … idx+1` around the middle of the tumor's z-range (`(mask_start + mask_end) // 2`).
- Crop: 224×224 around the bounding-box center (`sraw/eraw/scol/ecol`), shifted inward at image edges. Then `Resize(224)` and ImageNet mean/std normalization.
- Patient score: the per-slice pCR probabilities, summarised as mean and min (`DL_pred_pcr_mean`, `DL_pred_pcr_min`).
- **Bug:** `to_rgb()` returns float64 in [0, 1], which is passed to `Image.fromarray(im, mode="RGB")` with no ×255 / uint8 cast. Under Pillow ≤ 10 this reinterprets the raw float bytes as pixels. We tested Pillow 10.4: the output has correlation −0.01 with the true image, i.e. noise. Pillow 12 raises a TypeError instead. This also contradicts the paper's "converted to 8-bit".
- Consequence: the notebook's sample predictions are all 0.49–0.52. The shipped all-patient file `BreastDCEDL_vit_pcr_predictions.csv` (1,452 patients) has predictions between 0.474 and 0.575, with AUC 0.553 on train, **0.558 on test and 0.477 on val**. It is chance level, and it is the `DL_pred_pcr_*` input to the Random Forest.
- Whether the shipped weights are good when fed correct uint8 images is **unknown until we test them**.

**Article results (`df_pcr_pred_test_article.csv`, 175 patients):** the published numbers reproduce exactly from column `pred_pcr_vnew`. That column comes from a different ("vnew") model run whose code is not released. 95% CIs are ours (2,000-sample bootstrap):

| | N (pCR) | AUC | 95% CI |
|---|---|---|---|
| ViT overall | 175 (63) | 0.720 | 0.63–0.80 |
| ViT I-SPY2 | 99 (32) | 0.780 | |
| ViT I-SPY1 | 35 (12) | 0.679 | |
| ViT Duke | 41 (8) | 0.540 | |
| RF, I-SPY2 | 99 (32) | 0.853 | 0.77–0.92 |
| RF, I-SPY2 HR+/HER2− | 40 (5) | 0.937 | 0.84–1.00 |

**The 0.85 Random Forest (`plot_pCR_article_results.ipynb`):**
- Training data: I-SPY2 only, train + val (`test != 1`). Evaluated on I-SPY2 `test == 1` (99 patients). `StandardScaler` + `RandomForestClassifier(n_estimators=200, random_state=42)`, no tuning shown.
- Features (25): `HR, HER2, MP` (MammaPrint), `age, e_hispanic_latino, race_white`, subtype flags `HER2pos, HRposHER2neg, TripleNeg`, **13 treatment-arm drug indicators** (`ABT 888, AMG 386, Carboplatin, Ganetespib, Ganitumab, MK-2206, Neratinib, Paclitaxel, Pembrolizumab, Pertuzumab, T-DM1, Trastuzumab`), **`bbox_vol, tum_vol`** (from the ground-truth masks), and **`DL_pred_pcr_mean, DL_pred_pcr_min`**. These are the only imaging-model inputs: two scalar slice-probability summaries, **not ViT feature embeddings**. If they equal the shipped file's values (same column names, but the RF reads a private CSV so this is unproven), their AUC on these 99 test patients is 0.59 (mean) and 0.66 (min): weak, not chance. The RF output tracks the article's ViT score `pred_pcr_vnew` (Spearman 0.73) far more than these inputs (0.24 / 0.33), likely because both reflect tumor subtype.
- **Novelty gap (to our knowledge, for these two papers):** BreastDCEDL uses a ViT but fuses with clinical data in two stages through scalar scores. Nishizawa fuses end-to-end, but with a 3D CNN. **ViT image embeddings fused with clinical data end-to-end** has not been done in either.
- It reads a private file (`C:\Users\naomi\...\df_breastdcedl_pcr.csv`), so it cannot be rerun as published. MP and treatment arm are in the TCIA `ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx` (mirrored in the repo's `ISPY2/TCIA_metadata/`).
- Implication: **the 0.85 is mostly a clinical + treatment + tumor-volume model, not an imaging result.** Treatment arm strongly predicts pCR in I-SPY2: for example, T-DM1 + pertuzumab 59% and paclitaxel control 18% (Nishizawa Table 1). It is not available for MNJ patients in the same form. **Do not treat 0.85 as an image + clinical benchmark.** To compare fairly, rebuild the RF without imaging scores and with or without treatment arm, and report our model against those.

**The three metadata CSVs differ:**

| File | Patients | Duke labeled | Labeled train / test (1) / val (2) | Notes |
|---|---|---|---|---|
| Zenodo `BreastDCEDL_metadata_min_crop.csv` (md5 d58b84d0…) | 2,070 | 298 | 1,099 / 176 / 177 | **canonical**; pCR identical to GitHub full |
| GitHub `BreastDCEDL_metadata.csv` | 2,070 | 298 | 1,099 / 177 / 176 | the README's split table matches this one |
| GitHub `BreastDCEDL_metadata_min_crop.csv` | 2,072 (Duke 918) | 278 | 1,083 / 175 / 174 | 4 extra / 2 missing Duke patients, 19 Duke pCR labels differ, 12 split changes |

- The full and min_crop CSVs also differ (as expected) in `n_xy, n_z, sraw…, mask_start/end`, because the coordinates refer to the cropped vs full-FOV volume. **Use the coordinates from the CSV that matches the volumes being loaded.**
- All min_crop volumes, **including I-SPY1**, are 256×256 in-plane. A full-FOV I-SPY1 copy is presumably the older `BreastDCEDL_spy1.zip`; verify after download.
- Duke `tum_vol` differs between files (it is a size class for Duke).

**`utils/data_utils.py` file naming (needed for our loader):** NIfTI files are `{pid}_spy2_vis1_dce_aqc_{k}.nii.gz`, `{pid}_spy1_vis1_acq{k}.nii.gz` and `{pid}_duke_aqc_{k}.nii.gz`. Masks are `{pid}_spy2_vis1_mask`, `{pid}_spy1_vis1_mask` and `{pid}_duke_mask`. Note the inconsistent "aqc"/"acq" spellings. Loading uses `nibabel.get_fdata()` (float64).

---

## 7. Suggested next steps

1. Download `BreastDCEDL_ISPY1_min_crop.tar.gz` (1.25 GB, smallest) and `BreastDCEDL_models.tar.gz` first. Inspect the volume format, crop extent and mask format, and run the authors' demo notebook.
2. Load the shipped ViT weights (`BreastDCEDL_models.tar.gz`) and run them on a few Zenodo min_crop patients with a **correct** uint8 RGB conversion (Section 6.4). This tells us whether the weights are usable at all or whether the repo's chance-level predictions come from the inference bug. Either way, plan to **train our own ViT baseline**; the repo has no training code.
3. Download the I-SPY2 (13.6 GB) and Duke (8.6 GB) archives; verify md5s; mirror them to a private Kaggle Dataset. Check the Duke `pre/post_early/post_late` values against the two papers.
4. Build the baselines that the papers show are hard to beat: clinical-only (HR, HER2, age, race, menopause) and clinical + tumor volume. Report the image models against these with DeLong tests.
5. Decide the Module 1 evaluation source: TCIA BreastDCEDL-ISPY2 full-res (about 38 GB) or MAMA-MIA (Synapse account; pull only the nnU-Net weights and `train_test_splits.csv`).
6. For Option 2: finish the DCE-series check and size estimate (Section 4), filtering for the unilateral cropped DCE / SER / PE2 / PE6 series. Then download one paired patient end-to-end as a test. Re-implement Nishizawa's model from Section 6.3 as the Option 2 reference baseline.
