# Results so far

All results through 2026-10-28. Every number is on the **validation set**; the test set has not been
touched.
- **Validation set:** 177 patients (53 pCR+). The I-SPY-only runs use 132 of them (43 pCR+).
- **Uncertainty:** one AUC on 177 patients has a 95% CI of about ±0.08. Differences of a few points
  between single runs are usually noise.
- **Paired difference:** both models scored on the same patients, with a bootstrap CI. A difference
  counts only if its CI excludes 0.

Each run is stored in `results/<person>/<run folder>/`: `metrics.json`, `val_preds.csv` and `log.csv`.

---

## 1. Headline

| model | input | val AUC (95% CI) | vs subtype lookup (paired) |
|---|---|---|---|
| Subtype lookup (HR/HER2, 3 rows) | clinical | **0.662** | — |
| B1 logistic regression | clinical + tumour size | 0.653 (0.57–0.74) | −0.009 (−0.08 to +0.06) |
| Radiomics + subtype, logistic regression (section 8) | tumour features + HR/HER2 | **0.700** (0.62–0.78) | +0.038 (−0.03 to +0.11) |
| Radiomics only (section 8) | tumour features | 0.631 (0.54–0.72) | −0.030 (−0.15 to +0.09) |
| F2 attention fusion *(v1 cache, old code)* | image + clinical | **0.690** (0.60–0.77) | +0.028 (−0.00 to +0.06), p 0.06 |
| F1 concat fusion *(v1 cache, old code)* | image + clinical | 0.678 (0.59–0.76) | +0.016 (−0.06 to +0.09) |
| Best image-only: ResNet-18, early phase (v2 cache) | image | 0.626 (0.54–0.72) | −0.036 (−0.16 to +0.09) |
| Best image-only ViT: DINOv2, early phase (v2 cache) | image | 0.608 (0.52–0.70) | −0.054 (−0.18 to +0.07) |

- **No image-only model beats the 3-row subtype lookup yet.**
- **Fusion models are slightly above clinical-only, but not significantly.** F2 comes closest (p 0.06).
  Both fusion runs used the old v1 cache and the augmentation bug fixed in d1.4, so they need a rerun.
- **The images do carry information beyond subtype** (section 6). On I-SPY, a simple combination of image
  and subtype scores 0.70 vs 0.65 for subtype alone.

---

## 2. Clinical-only baselines (A, code ee4d7e9)

`src/baselines/tabular.py`. B0 = HR, HER2, race, menopause, age; B1 = B0 + log tumour-box volume.
Hyperparameters were chosen by 5-fold CV on train.

| model | seed | val AUC (95% CI) | AUPRC | ECE | Duke | I-SPY1 | I-SPY2 |
|---|---|---|---|---|---|---|---|
| B0 LR (C 0.01) | 0 | 0.639 (0.55–0.73) | 0.461 | 0.039 | 0.669 | 0.736 | 0.598 |
| B0 RF (leaf 40) | 0 | 0.645 (0.55–0.73) | 0.480 | 0.035 | 0.646 | 0.736 | 0.622 |
| B0 RF | 1 | 0.654 (0.56–0.74) | 0.493 | 0.049 | 0.666 | 0.736 | 0.628 |
| B0 RF | 2 | 0.649 (0.56–0.74) | 0.480 | 0.036 | 0.657 | 0.731 | 0.627 |
| B1 LR (C 0.03) | 0 | 0.653 (0.57–0.74) | 0.448 | 0.041 | 0.666 | 0.690 | 0.654 |
| B1 RF (leaf 40) | 0 | 0.659 (0.57–0.74) | 0.460 | 0.019 | 0.683 | 0.686 | 0.648 |
| B1 RF | 1 | 0.654 (0.57–0.74) | 0.456 | 0.027 | 0.683 | 0.678 | 0.646 |
| B1 RF | 2 | 0.660 (0.57–0.75) | 0.457 | 0.011 | 0.694 | 0.690 | 0.645 |

**Subtype lookup** (computed in analysis; it will be a saved run in `02_radiomics`):
- **Rule:** each val patient gets the train pCR rate of their subtype:
  - HR+/HER2−: 0.141
  - HER2+: 0.427
  - triple negative: 0.379
- **Val AUC:** 0.662 overall; Duke 0.717, I-SPY1 0.674, I-SPY2 0.638.
- **Takeaway:** subtype explains almost all of the clinical signal; B1 LR − subtype = −0.009. The
  stronger clinical predictors (grade, Ki-67, T/N stage, treatment arm) are not in the dataset's CSV.

---

## 3. First image and fusion runs (cache v1, code a10edc6, 2026-10-09)

These runs used cache v1 (native pixels, so 224 px covered 70–315 mm) and the augmentation that sheared
images (fixed in d1.4). Read them as preliminary.

| model | who | val AUC (95% CI) | AUPRC | ECE | best epoch / run | Duke | I-SPY1 | I-SPY2 |
|---|---|---|---|---|---|---|---|---|
| B3 ViT (MAE), image only | B | 0.530 (0.44–0.63) | 0.338 | 0.063 | 1 / 10 | 0.566 | 0.362 | 0.569 |
| F1 concat fusion | C | 0.678 (0.59–0.76) | 0.463 | 0.095 | 6 / 15 | 0.674 | 0.653 | 0.697 |
| F2 attention fusion | A | 0.690 (0.60–0.77) | 0.516 | 0.080 | 2 / 11 | 0.760 | 0.742 | 0.651 |

- **B3 on v1 did not learn:** its predictions were nearly constant (SD 0.03) and the best epoch was 1.
- **F2 already reached 0.658 at epoch 0**, so its score comes mostly from the clinical tokens.

---

## 4. Image-only B3 on cache v2 (A, 2026-10-19)

Cache v2 (d1.5): 0.7 mm pixels and 2 mm slices for every patient, tumour-centred, clip ceiling raised.

| run | val AUC (95% CI) | train AUC | best epoch | prediction SD |
|---|---|---|---|---|
| B3 on v1 (B, above) | 0.530 | 0.595 | 1 | 0.03 |
| B3 on v2 (`b3_vit_s0_1019-0302_2b57e15`) | 0.559 (0.47–0.65) | 0.873 | 18 | 0.21 |

- **The model now learns, but overfits:** train AUC 0.87 vs val 0.56.
- **Paired v2 − v1:** +0.029 (−0.08 to +0.14), not significant.

### 4a. Sweep 1: regularisation (`sweep_b3_vit_1019-0406`, MAE ViT, RGB input, seed 0)

Baseline = `base` (patience 15). Paired differences vs `base`.

| run | change | val AUC | vs base (95% CI) | p | best epoch | Duke | I-SPY1 | I-SPY2 |
|---|---|---|---|---|---|---|---|---|
| base | — | 0.559 | — | — | 18 | 0.511 | 0.583 | 0.568 |
| frozen | linear probe (0 blocks trained) | 0.515 | −0.044 (−0.13 to +0.04) | 0.30 | 17 | 0.501 | 0.512 | 0.514 |
| **blocks_2** | 2 blocks trained (default 4) | **0.579** | +0.020 (−0.02 to +0.06) | 0.31 | 26 | 0.546 | 0.599 | 0.584 |
| min_area_0.1 | skip near-empty slices | 0.568 | +0.009 (−0.02 to +0.04) | 0.56 | 19 | 0.491 | 0.585 | 0.589 |
| strong_aug | rotation 30°, intensity 0.2, scale 0.8–1.2 | 0.555 | −0.004 (−0.04 to +0.03) | 0.80 | 18 | 0.494 | 0.616 | 0.568 |
| dropout_0.5 | dropout 0.5 + weight decay 0.1 | 0.558 | −0.001 (−0.01 to +0.01) | 0.91 | 18 | 0.526 | 0.591 | 0.563 |
| supervised | ImageNet-supervised ViT-B/16 | 0.538 | −0.021 (−0.11 to +0.07) | 0.64 | 7 | 0.580 | 0.632 | 0.489 |

**Result:** nothing significant. `blocks_2` was the best and became the reference for sweep 2.

### 4b. Sweep 2: backbones on the early phase only (`sweep_b3_vit_1019-1102`, seed 0)

Baseline = sweep 1 `blocks_2` (MAE, RGB input, 0.579). Early phase only = the early image in all 3
channels.

| run | backbone | val AUC (95% CI) | vs blocks_2 (95% CI) | p | best epoch | Duke | I-SPY1 | I-SPY2 |
|---|---|---|---|---|---|---|---|---|
| mae_early | MAE ViT-B/16, 2 blocks | 0.564 (0.47–0.66) | −0.015 (−0.08 to +0.05) | 0.69 | 13 | 0.620 | 0.488 | 0.558 |
| dinov2_early | DINOv2 ViT-B/14, 2 blocks | 0.608 (0.52–0.70) | +0.029 (−0.05 to +0.11) | 0.47 | 10 | 0.443 | 0.669 | 0.629 |
| resnet18_early | ResNet-18, all stages, lr 1e-4 | **0.626** (0.54–0.72) | +0.046 (−0.03 to +0.13) | 0.25 | 7 | 0.477 | 0.686 | 0.684 |

- **Early-only input does not help MAE** (−0.015).
- **DINOv2 and ResNet-18 look better** but are not significant.
- **ResNet's 0.626 is a single-epoch spike**; its epoch-to-epoch level is about 0.57.
- **Both new backbones are at chance on Duke** (0.44–0.48) and 0.63–0.69 on I-SPY.

### 4c. Sweep 3: DINOv2 on I-SPY only, with regularisation (`sweep_b3_vit_1019-1820`, seed 0)

- **Setup:** DINOv2, early phase, 2 blocks, trained and validated on I-SPY1 + I-SPY2 only.
- **Validation:** 132 patients.
- **Baseline:** `dinov2_early` from sweep 2 (trained on all cohorts), scored on the same 132 patients:
  0.636.

| run | change | val AUC (95% CI) | vs baseline (95% CI) | p | mean AUC, epochs 5–15 | I-SPY1 | I-SPY2 |
|---|---|---|---|---|---|---|---|
| ispy | I-SPY-only training | 0.623 (0.53–0.72) | −0.013 (−0.07 to +0.04) | 0.64 | 0.593 | 0.636 | 0.619 |
| **ispy_dropout_0.5** | + dropout 0.5 | **0.648** (0.55–0.74) | +0.012 (−0.05 to +0.07) | 0.68 | **0.612** | 0.694 | 0.637 |
| ispy_droppath_0.2 | + stochastic depth 0.2 | 0.625 (0.53–0.72) | −0.011 (−0.07 to +0.05) | 0.77 | 0.590 | 0.632 | 0.618 |
| ispy_both | + both | 0.623 (0.53–0.72) | −0.013 (−0.07 to +0.04) | 0.70 | 0.592 | 0.603 | 0.627 |

- **Nothing significant.**
- **The four runs are nearly the same model:** their predictions correlate 0.87–0.98, and averaging all
  four gives 0.633.
- **The peak epoch is optimistic:** all runs peak at epoch 11, then decline as the training loss keeps
  falling (to about 0.1). The epoch-5–15 averages (0.59–0.61) are more honest.
- **Training without Duke does not help.**

---

## 5. Image-only vs clinical, paired on all 177 val patients

| image model | − subtype lookup (95% CI) | p | − B1 LR (95% CI) | p |
|---|---|---|---|---|
| B3 MAE, v1 | −0.132 (−0.27 to +0.01) | 0.07 | −0.122 (−0.25 to +0.01) | 0.07 |
| B3 MAE, v2 | −0.103 (−0.22 to +0.02) | 0.09 | −0.093 (−0.22 to +0.03) | 0.13 |
| DINOv2 early, v2 | −0.054 (−0.18 to +0.07) | 0.40 | −0.044 (−0.16 to +0.08) | 0.47 |
| ResNet-18 early, v2 | −0.036 (−0.16 to +0.09) | 0.58 | −0.027 (−0.15 to +0.09) | 0.67 |
| F1 concat (v1) | +0.016 (−0.06 to +0.09) | 0.65 | +0.026 (−0.01 to +0.07) | 0.20 |
| F2 attention (v1) | +0.028 (−0.00 to +0.06) | 0.06 | +0.038 (−0.02 to +0.09) | 0.21 |

The image-only models have been closing the gap: −0.13 → −0.10 → −0.05 → −0.04. None is significantly
different from clinical-only, in either direction.

---

## 6. Does the image add to subtype? (quick analysis, I-SPY val, 132 patients)

Computed in analysis; no model was trained for this.
- **Image + subtype** = average of the image model's ranking and the subtype lookup's ranking.
- **Lookup table:** fitted on train I-SPY patients.

| | val AUC |
|---|---|
| subtype lookup | 0.648 |
| DINOv2 early, dropout 0.5 (I-SPY trained) | 0.648 |
| image + subtype (rank average) | **0.697** |
| DINOv2 early, all cohorts + subtype | 0.695 |

AUC of the image model **within** each subtype (dropout 0.5 run):

| subtype | patients | pCR+ | AUC |
|---|---|---|---|
| HR+/HER2− | 48 | 9 | 0.73 |
| HER2+ | 37 | 18 | 0.62 |
| triple negative | 47 | 16 | 0.59 |

- The image ranks patients **within** subtypes, so it carries information that receptor status does not.
- **Caveats:** the subgroups are small and no significance test was run. A trained fusion model (F2 on
  cache v2) is needed to confirm this.

---

## 7. What the results say so far

1. **Clinical ceiling:** about 0.65–0.66, and almost all of it is HR/HER2 subtype.
2. **Cache v2 fixed the "not learning" problem**, but image-only models overfit and plateau at about
   0.60–0.65.
3. **Neither regularisation nor I-SPY-only training changed that significantly** (13 runs, 3 sweeps).
4. **Backbones:** DINOv2 and ResNet-18 are the best image encoders so far, but both are at chance on Duke.
5. **The image adds to subtype on I-SPY** (+0.05 in a quick check). The fusion models are the most
   promising route, but have not been rerun since v2 and the augmentation fix.

## 8. Radiomics + ComBat baselines (`results/A/radiomics_1028-1414`)

- **Features:** 29 per patient (shape, enhancement kinetics, enhancement texture, 5 mm peritumoural ring),
  from cache v2, for all 2,068 patients, with no errors.
- **Duke tumour region:** box voxels enhancing ≥ 70%, for 913 of 914 patients.
- **ComBat:** 14 protocol batches (cohort × matrix × slice thickness × pixel size), 20–343 training
  patients each.
- **Model:** logistic regression, C by 5-fold CV on train.

**Use the reruns `r_*_1028-2101/2102_2c4396f`.** The first runs (`r_*_1028-1422/1423_c8564d9`) had a
flaw, described below.

| model | val AUC (95% CI) | Duke | I-SPY1 | I-SPY2 | vs subtype (paired, 95% CI) | p |
|---|---|---|---|---|---|---|
| subtype lookup | 0.662 (0.58–0.74) | 0.717 | 0.674 | 0.638 | — | — |
| radiomics only, ComBat | 0.625 (0.53–0.72) | 0.617 | 0.574 | 0.645 | −0.037 (−0.15 to +0.08) | 0.57 |
| radiomics only, no ComBat | 0.631 (0.54–0.72) | 0.646 | 0.591 | 0.599 | −0.030 (−0.15 to +0.09) | 0.65 |
| **radiomics + subtype, ComBat** | **0.699** (0.61–0.78) | 0.657 | 0.715 | 0.704 | +0.037 (−0.03 to +0.10) | 0.22 |
| **radiomics + subtype, no ComBat** | **0.700** (0.62–0.78) | 0.680 | 0.744 | 0.685 | +0.038 (−0.03 to +0.11) | 0.27 |

- **Radiomics alone ≈ 0.63**, the same level as the best image-only network (ResNet-18 at 0.626). Within
  subtypes it scores 0.57–0.63.
- **Radiomics + subtype ≈ 0.70**, the best saved result so far. It is +0.04 over subtype, not significant.
- **ComBat removes the batch signal but does not change AUC.**
  - Batch predictability from the features drops from 0.379 to 0.111 balanced accuracy (chance 0.071),
    with ComBat fitted inside each CV fold.
  - The model with ComBat and the model without score the same AUC.
- **Largest weights (radiomics only):**
  - enhancement of the 5 mm ring around the tumour (`ring_re_early_mean`, `tumour_to_ring_re`);
  - enhancement heterogeneity (entropy, GLCM energy);
  - tumour volume.

  Peritumoural enhancement as a pCR predictor matches published work (Braman et al. 2017).

**Flaw in the first runs (fixed):**
- **The bug:** the "radiomics only" model ran ComBat with HR/HER2 as covariates. ComBat keeps the
  covariate part of each feature, so the harmonised features carried subtype information.
- **Evidence:** HR status became more predictable from the features (AUC 0.637 → 0.668), and "radiomics
  only" scored 0.709. That 0.709 is not a radiomics result.
- **The fix:** `rad` now uses ComBat without covariates; only `rad_sub` keeps them.
- **Second fix:** the batch check first fitted ComBat on all rows and then cross-validated, which pushed
  the score below chance (0.032). It now fits ComBat inside each fold.

## 9. Runs prepared, not yet run (code d1.10)

- `01_build_cache` with `Z_MM = 0`: cache v3 (native slices).
- `14_v3_enhancement`: v2 vs v3 and the relative-enhancement input, DINOv2, 2 seeds each.
- **Still to do:** F1/F2 fusion on cache v2.

See `docs/ideas.md` for the reasoning behind each experiment.
