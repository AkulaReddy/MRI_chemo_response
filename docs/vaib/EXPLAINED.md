# BreastDCEDL from A to Z, and why the ViT-only model failed (branch `vaib`)

Written 2026-10-18. Numbers marked *(demo)* were measured on the 7-patient Zenodo demo archive, downloaded and
processed locally with this repo's code. Everything else comes from the papers, `data_availability.md`,
`challenges.md` and the team's logged runs in `results/`.

## 1. The question
Predict **pathologic complete response (pCR)**, meaning no invasive cancer left at surgery after neoadjuvant
chemotherapy, from the **pre-treatment** breast DCE-MRI. pCR = 1 is the good outcome. About 30% of patients
reach it.

## 2. Is the data downloadable? Yes
- **Source:** Zenodo record 18114231 (BreastDCEDL, Fridman et al., *Scientific Data* 2026). Licence CC BY-NC 4.0,
  **no login**. The cohort archives total ~24 GB: I-SPY1 1.25 GB, I-SPY2 13.6 GB, Duke 8.6 GB, plus a 55 MB
  demo archive and the metadata CSV.
- **Speed:** Zenodo throttles each connection.
  - This laptop: ~13–65 KB/s on one connection. **0.23 MB/s with 8 parallel byte ranges** (demo archive, 238 s,
    md5 OK, 2026-10-18).
  - Kaggle: ~1 MB/s on one connection (I-SPY1 in 21 min, 2026-10-08).
  - Zenodo sometimes returns short or `504` responses. **A first local attempt came back 531 KB short with a wrong
    md5**, so every download must check part sizes and the md5. `kaggle/00_download_zenodo.py` and
    `kaggle/vaib/00_download_check.ipynb` both do this.
- **Where it lives for training:** the team downloaded it inside Kaggle (Datasets `mnj-raw-ispy1/-ispy2/-duke`)
  and built slice caches from it (`mnj-cache-*`).
- **Post-chemo scans** (needed for the Nishizawa-style pre + post model): not in BreastDCEDL, which is
  pre-treatment only. They come from TCIA I-SPY2 / ACRIN-6698 (T0 baseline + T3 pre-surgery, ~870 labeled pairs)
  and I-SPY1. The NBIA API is anonymous; see `data_availability.md` §4. MAMA-MIA (Synapse, free login) has
  expert segmentations and nnU-Net weights.

## 3. What is in the data
| | I-SPY1 | I-SPY2 (incl. ACRIN-6698) | Duke |
|---|---|---|---|
| patients / pCR-labeled | 172 / 172 | 982 / 982 | 916 / 298 |
| pCR rate | 28.5% | 32.2% | 21.1% |
| tumour annotation | 3D mask (semi-automatic, enhancement threshold) | 3D mask (same) | bounding box only |
| sites / vendors | 10 sites | >22 sites | 1 site |

- **Files:** one NIfTI per **DCE phase** (`<pid>_spy1_vis1_dce_aqc_<k>.nii.gz`) plus `<pid>_..._mask.nii.gz`.
  Arrays are `(slices, 256, 256)` **float64 raw scanner values, never normalised**. The volumes are "min_crop":
  a 256×256 in-plane window around the tumour, at the original resolution.
- **The header spacing (1 mm) is wrong.** The real pixel size (`xy_spacing`, 0.31–1.41 mm) and slice thickness
  (`slice_thick`, 0.8–4 mm) are in the CSV.
- **Metadata CSV:** labels, HR/HER2, age, race, menopause, phase indices (`pre`, `post_early`, `post_late`),
  tumour box (`sraw/eraw/scol/ecol`, `mask_start/mask_end`) and the **split** (`test`: 0 train, 2 val, 1 test →
  1,099 / 177 / 176 labeled). The box is in the *original* `n_xy` image's pixels; `challenges.md` item 3
  explains the shift this needs.
- Intensities differ by orders of magnitude between scanners: the mean pre-contrast foreground ranged
  **17 to 543** across the 7 demo patients. *(demo)*

## 4. Why three phases (DCE physics in one paragraph)
A pre-contrast scan is taken, gadolinium is injected, and the scan is repeated every 1–2 min. Tumours have leaky,
dense vessels, so they **enhance early** (wash-in). Afterwards they either keep rising (*persistent*, more often
benign or slow), level off (*plateau*), or drop (*washout*, typical of aggressive cancer). Standard numbers:
- **PE** = (early − pre) / pre, the percent enhancement;
- **SER** = (early − pre) / (late − pre), the signal enhancement ratio (> 1.1 means washout).

These kinetics relate to tumour subtype and proliferation, which is why they carry some pCR signal. Example:
ISPY1_1001's tumour mean goes pre → early ×1.95, early → late ×1.06 (persistent). *(demo)*

## 5. 3D → 2D: what the pipeline does and why
**Why 2D:** pretrained ViTs (MAE, DINOv2, ImageNet) are 2D RGB models, and there are only ~1,100 labeled training
patients, far too few to train a 3D transformer from scratch. Each tumour slice becomes one 2D image, which also
multiplies the training examples (~30 slices per patient).

**How (repo v1, `scripts/build_cache.py` → `src/train.py`):**
1. Load the three phases given by the CSV (pre, early, late) and find the tumour slice range and box (mask for
   I-SPY, CSV box for Duke).
2. **Normalise:** clip all three phases *jointly* at the volume's 0.5 / 99.5 percentiles and scale to uint8.
   Clipping jointly keeps "early is brighter than pre".
3. **Fuse:** R = pre, G = early, B = late. One RGB image per slice. Bright cyan/white means strong enhancement;
   green > blue means washout.
4. **Crop** 240×240 around the tumour-box centre and keep the tumour slices ±2 → `<pid>.npz`.
5. **Train:** a patient is a *bag* of K slices (8 random for training; 16 evenly spaced for evaluation). Each slice
   gets a random 224 crop, rotation, flips and intensity jitter → ViT-B/16 (MAE weights, top 4 blocks trained) →
   one logit per slice → **mean over slices = patient logit**. The label (pCR) belongs to the patient.
   Metrics are patient-level AUC on val.
6. The test split is refused by `patient_table()` and touched once at the end (D-006).

## 6. Why B3 (ViT image-only) scored val AUC 0.530
Evidence from `results/B/b3_vit_s0_1009-1616_a10edc6`:
- **The train loss never moved: 0.393 → 0.370 over 10 epochs.** For this loss (0.5 BCE + 0.5 focal) a constant
  prediction of the 29% base rate costs ≈ 0.40. The model did not even fit the training set. This is
  underfitting caused by the inputs or the recipe, not overfitting. Val AUC: 0.53 overall, **0.36 on I-SPY1**,
  0.57 I-SPY2, 0.57 Duke.
- The F2 fusion model's auxiliary heads (HR / HER2 predicted *from the image*) reached val AUC only ~0.55 / ~0.6.
  The image features carried little information.

Causes found (all in the input, none in the training code):

| problem | measured | effect |
|---|---|---|
| **Tumour saturates in the RGB images.** The joint 99.5th-percentile ceiling sits *below* the enhancing tumour, which is among the brightest tissue in a tumour-centred crop | share of tumour-**mask** voxels clipped to 255 in early/late: I-SPY1 **48–96%**, I-SPY2 1–47%, Duke 0–2% *(demo)* | G = B = 255 across the tumour: washout vs persistent and the internal texture are gone. Worst for I-SPY1, which matches its 0.36 |
| **The tumour is small in the 224 input** | median box side 84 px (I-SPY), 53 px (Duke); 29% of Duke tumours < 40 px ≈ 2.5 ViT patches; tumour = 17–43% of the input width *(demo)* | most of each image is surrounding tissue |
| **MAE [CLS] token** | MAE pre-training never trains [CLS]; MAE's own fine-tuning pools patch tokens | weak features at the start of fine-tuning |
| **Edge slices** | all tumour slices were used, including the end slices that hold only a sliver of tumour | label noise at the slice level |

Also: the published 0.72 / 0.78 (arXiv Table 2) came from an unreleased model on a 175-patient set that matches
no released split, and the shipped weights have an RGB bug (`data_availability.md` §6.4). Realistic image-only
targets from the literature are ~0.65–0.72 (Nishizawa MRI-only: 0.68 internal).

## 7. What branch `vaib` adds (nothing existing is changed)
| change | where | why |
|---|---|---|
| **Cache v2:** raw float16 phases, full 256 plane, whole-volume statistics; labeled patients only | `scripts/build_cache_v2.py`, `kaggle/vaib/02` | no clipping; any normalisation at train time. `joint_pct` on v2 reproduces the v1 images to ±1 grey level (tested) |
| **9 normalisations** | `src/vitonly/norms.py` | `joint_pct` (v1), `joint_wide` (99.9th pct), `per_phase`, `joint_tumor` (lesion window), `pre_ref` (log of intensity relative to pre-contrast foreground), `zscore`, `sub` (subtraction channels), `kinetic` (pre / PE / SER maps), `early3` (single phase). Tumour saturation on v2: `joint_tumor`, `pre_ref`, `kinetic` ≤ 12% vs `joint_pct` up to 96% *(demo)* |
| **Tumour-zoom crop** | `norms.py` (`crop=tumor`, also `mm`) | the tumour fills ~65% of the input width instead of 17–43% |
| **Central slices** (`slice_mode=mid`) | `norms.py` | drop the edge slivers |
| **Backbones + pooling** | `src/vitonly/backbones.py`, `model.py` | MAE, supervised ViT, DINO, DINOv2-S/B; patch-token mean (`gap`) |
| **LP-FT** | `finetune.py` (`freeze_epochs`) | the head first learns on frozen features, then the top blocks unfreeze with their own warm-up |
| **Frozen-feature probe** | `src/vitonly/probe.py`, `kaggle/vaib/03` | ranks every norm × crop × backbone by 5-fold CV AUC on ~1,100 training patients. This is much less noisy than one fine-tune on 177 val patients (95% CI ±0.08) |
| **Hand-crafted kinetic baseline** | `src/vitonly/kinetics.py`, `kaggle/vaib/01` | FTV, PE, SER, washout fractions → LR. Shows whether the images carry pCR signal at all, without deep learning |
| **Cohort-only floor and within-cohort AUC** | probe and fine-tune outputs | pCR rates differ by cohort (21% vs 32%), so "recognising the scanner" alone scores > 0.5 pooled |
| **B3 control config** | `configs/vo_b3_repro.yaml` | the old recipe through the new code; should land near 0.53 |

## 8. How to run (Kaggle)
See `kaggle/vaib/README.md`. All steps use the full real dataset:
1. step1: download the full release into Kaggle (×3 cohorts → `mnj-raw-*`).
2. step2: build `mnj-cache2-*` (×3, CPU).
3. step3: diagnostics (CPU).
4. step4: probe (GPU, ~30 min).
5. step5: fine-tune the top probe setting, 3 seeds + ensemble (GPU, ~1.5 h).

step0 is an optional look at the 7-patient demo sample.

## 9. Honest expectations
- No new AUC is claimed here: real numbers need the Kaggle runs. The fixes target measured input defects, so the
  ViT should at least *fit* the training data now. The probe will show which input carries the most signal.
- Report val AUC with its CI, the within-cohort AUC and the cohort-only floor. Compare against B0 (0.639) / B1
  (~0.655), which are clinical, not imaging. Never select on test.

## 10. Nishizawa (2025) and the pre + post-chemo idea
Nishizawa et al. use the **first post-contrast** whole-breast 3D volume at **T0 (baseline) and T3 (after chemo)**
from TCIA I-SPY2, a 3D ResNet-50 per scan, and attention fusion with HR/HER2/age/race/menopause. Their results:
internal 0.73 (I-SPY2), external 0.71 (I-SPY1). Clinical-only reached 0.71 externally, so imaging added little
there. The post-chemo scan answers "is there residual tumour?", which is easier than predicting response before
treatment. It needs the TCIA download (DICOM, pre + post), not BreastDCEDL. Feasibility counts and the leakage
caveat (91 paired I-SPY2 patients are in BreastDCEDL's test split) are in `data_availability.md` §4 and plan §9.
