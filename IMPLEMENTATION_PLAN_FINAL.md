# Implementation Plan (Final): pCR Prediction from Breast DCE-MRI

**Project:** individual capstone, MNJ Institute of Oncology project #11 ("MRI-Based Assessment of Chemo Response").
**Version:** 1.2, 2026-10-08. **5-week timeline** (D-019); training on Kaggle (D-020); **3-person split, parallel tracks from day 1** (D-021, D-022); see `docs/team/`. This is the only plan: it consolidates and replaces all earlier draft plans (D-001). In the decision log and data notes, "Option 1" means Stage 1 and "Option 2" means Stage 2.
**Read with:** `data_availability.md` (verified data facts; it wins any disagreement) and `docs/DECISIONS.md` (the reasoning behind each choice, cited here as D-xxx).

---

## 1. Goal and honest framing

**What we build:** a deployable model that takes a breast DCE-MRI plus routine clinical variables and outputs a calibrated probability of pathologic complete response (pCR) to neoadjuvant chemotherapy, with an explanation. It must work on a new scan that has no tumor annotation.

Scope for the 5 weeks (D-019):

| | Stage 1: pre-treatment (**this project**) | Stage 2: paired pre + post-chemo (**future work**) |
|---|---|---|
| Question | "Will this patient respond?", from the baseline scan | "Is there residual cancer?", MNJ's actual question |
| Data | BreastDCEDL (Zenodo), 1,452 labeled | TCIA I-SPY1/I-SPY2 raw DICOM, about 1,020 labeled pairs |
| In this project | built, evaluated, demoed | design note + verified feasibility counts only (Section 9) |

Every Stage 1 deliverable states plainly: *"This predicts response before treatment; it does not assess the post-chemo scan."*

**Our contribution**, to our knowledge and based on the two reference papers:
- BreastDCEDL used a ViT, but combined it with clinical data in two stages, through two scalar ViT scores fed to a Random Forest.
- Nishizawa fused image and clinical data end-to-end, but with a 3D CNN.
- **End-to-end fusion of ViT image embeddings with clinical features, plus a localization step that makes it deployable on unannotated scans,** has not been reported in either.

**Primary hypothesis (D-007):** the fusion model beats the **clinical + tumor-size baseline** on the held-out test set (paired DeLong). It is the hard bar: on external data Nishizawa's clinical-only model matched their fused model, and tumor volume was BreastDCEDL's top feature.

## 2. Facts that shape the design

Details and sources are in data_availability.md.

- BreastDCEDL ships **3D NIfTI per DCE phase**, raw intensity (float64), with no normalisation. We build the RGB slices ourselves.
- Zenodo min_crop volumes are **256×256 in-plane crops around the tumor for every cohort**. Full-field-of-view (FOV) images come from TCIA BreastDCEDL-ISPY2 full-res (about 38 GB), MAMA-MIA, or the older `BreastDCEDL_spy1.zip`.
- Masks: I-SPY has 1,154 **semi-automatic** threshold masks; Duke has **bounding boxes only**.
- **The BreastDCEDL repo has no training code.** Its inference code has an RGB conversion bug (images become noise), and its shipped predictions are chance level. The published 0.72 / 0.78 come from an unreleased model run, on a 175-patient set that matches no released split.
- BreastDCEDL's **0.85** Random Forest uses MammaPrint, 13 treatment-arm drug indicators and ground-truth tumor/bbox volume. It is mostly not an imaging result.
- The Nishizawa repo is incomplete and does not run. Their real model encodes each scan to **8 numbers**, then fuses 3 tokens with one attention block.
- Receptor definitions differ by cohort: HR ≥1% (I-SPY2) / ≥10% (I-SPY1) / Allred >3 (Duke); HER2 FISH ≥2.0 vs >2.2.
- Duke: 298 of 916 labeled, published ViT AUC 0.54, test set 42 patients (8 pCR).

## 3. Data and split

### 3.1 Sources and roles (D-005)

| Source | Use | Size |
|---|---|---|
| Zenodo 18114231: `BreastDCEDL_{ISPY1,ISPY2,DUKE}_min_crop.tar.gz` | classifier training and evaluation, all patients | 1.25 + 13.6 + 8.6 GB |
| Zenodo: `BreastDCEDL_metadata_min_crop.csv` | labels, clinical data, phase indices, coordinates, **the split** | small |
| Zenodo: `BreastDCEDL_models.tar.gz` | sanity check of the shipped ViT weights only (D-010) | 0.32 GB |
| TCIA BreastDCEDL-ISPY2 full-res | fallback full-FOV source for Module 1 if MAMA-MIA coverage is too small | about 38 GB (download test/val patients only) |
| MAMA-MIA (Synapse, free login) | pretrained nnU-Net weights and its train/test split; full-FOV images for the cohorts it covers | pull selectively |
| TCIA I-SPY1 / I-SPY2 / ACRIN-6698 | Stage 2 only (future work) | not downloaded in this project |

**Where data is downloaded (D-020, D-022):** Zenodo serves this machine at about 13 KB/s, so the archives are downloaded **inside Kaggle**. `kaggle/00_download_zenodo.py` runs overnight with no GPU, one cohort per person, and saves the md5-verified raw archives as Datasets `mnj-raw-ispy1/-ispy2/-duke`. The uint8 slice caches (`mnj-cache-*`) are then built from those Datasets, so a builder fix means a quick rebuild, not a new download. Only the small CSVs and caches come to the local machine. For Module 1, prefer MAMA-MIA images over TCIA BreastDCEDL-ISPY2 full-res: MAMA-MIA images come with expert-verified segmentations and its train/test split, all from one source.

### 3.2 Split (D-003, D-006)

| Split | `test` value | Labeled (pCR) | Use |
|---|---|---|---|
| train | 0 | 1,099 (322) | fitting |
| val | 2 | 177 (53) | model selection, early stopping, threshold choice |
| test | 1 | 176 (53): I-SPY1 35, I-SPY2 99, Duke 42 | **once per final model** |

- Freeze the split as `data/splits/split_v1.csv`, with the md5 of the source CSV, and read it only from there.
- Unlabeled patients (618, mostly Duke) count toward their split for SSL purposes. SSL never uses `test==1` images (D-012).
- Stage 2 keeps each patient's assignment (D-014).

### 3.3 Inputs

**Clinical (D-008):**

| Feature | Encoding | Missing |
|---|---|---|
| HR, HER2 | binary | category "missing" (2 / 5 patients) |
| age | standardised using train mean/SD | impute train median + missing flag (3) |
| race | white / black / other | missing category (3) |
| menopause | pre / post | missing category (210) |
| bbox tumor volume | log(mm³), from CSV coordinates × spacing; at deployment from Module 1 | none |

- Excluded from the main model: treatment arm and MammaPrint (used only in context baseline B2).
- Cohort indicator: ablation only.

**Image (D-009):**
1. Load the `pre`, `post_early` and `post_late` phase volumes using the CSV indices.
2. Clip the three phases jointly at the 0.5–99.5 percentiles of the volume, scale to [0, 1], then **explicitly** ×255 → uint8.
3. Stack them as R, G, B.
4. Slice along axis 0 (confirmed on the demo archive for all three cohorts).
5. Cache a 240×240 crop around the tumor bbox center, shifted inward at the edges. Training takes a random 224 crop (±8 px jitter); evaluation takes the center 224.
6. For training, use all tumor slices (`mask_start..mask_end`), with a margin of 2 slices as an ablation.
7. Cache per patient as `uint8` `.npz` (about 3–8 MB each).

## 4. Architecture

```
full-FOV DCE (deployment) ─► [M1 localization: nnU-Net] ─► bbox + volume ─┐
                                                                           ▼
3 phases ─► RGB uint8 slices, 224×224 around bbox ─► [M2 ViT-B/16] ─► slice embeddings ─► pooling ─► image token(s)
                                                                                               │
clinical (HR, HER2, age, race, menopause, log bbox volume) ─► per-feature token embeddings ────┤
                                                                                               ▼
                                                           [M3 fusion transformer, 1–2 layers] ─► [CLS] ─► pCR probability
                                                                   ▲ aux HR/HER2 heads on the image token (pre-fusion)
```

### M1: tumor localization (D-011)
- Primary: MAMA-MIA pretrained nnU-Net, `nnUNetv2_predict -d 101 -c 3d_fullres`. Its input needs 1 mm isotropic resampling and z-scoring over all phases; check the repo README for the `_0000…` naming.
- Output: largest connected component → bbox, centroid and volume. Map back to native spacing for cropping.
- **Leakage control:** evaluate only on patients in neither MAMA-MIA's train split nor ours. If fewer than about 50 qualify, report on what qualifies and label it as limited. Training our own nnU-Net on I-SPY train masks is a **stretch goal** (D-019).
- Metrics: Dice and bbox IoU (I-SPY), centroid distance (mm), bbox-volume agreement, and **downstream AUC with predicted vs ground-truth crops**. Note that the I-SPY masks are semi-automatic.

### M2: image encoder (D-010)
- ViT-B/16, 224 px, initialised from `facebook/vit-mae-base` (HuggingFace).
- Ablation: supervised ImageNet-21k + 1k ViT-B/16, `google/vit-base-patch16-224` (HuggingFace, same code path as MAE; replaces the timm checkpoint named earlier).
- Fine-tuning: freeze patch embedding + blocks 0–7, train blocks 8–11 (ablate 2 / 4 / all). Layer-wise LR decay of 0.75. Gradient checkpointing + AMP.
- Patient aggregation (ablate): mean of slice logits; attention-MIL pooling of slice CLS embeddings into one image token; or k central slice tokens passed to M3.
- Augmentation (the same transform for all three channels): horizontal/vertical flip, ±15° rotation, 0.9–1.1 scale, ±10% intensity/contrast jitter per channel, small crop jitter (±8 px). No elastic deformation.

### M3: fusion (main contribution)
- Tokens: a learnable [CLS] token, the image token(s) (768 → d = 256 via a linear layer + LayerNorm), and one token per clinical feature (an embedding table for categorical features, a linear layer for continuous ones), each with a learned type embedding.
- 1–2 pre-norm transformer layers: d = 256, 4 heads, FFN 512, dropout 0.1–0.3. Output: [CLS] → linear → logit.
- Auxiliary heads: HR and HER2 from the image token before fusion, with loss weight 0.1–0.3 (D-013).
- Loss: 0.5 BCE + 0.5 focal (γ = 2), following Nishizawa. Ablate against BCE with `pos_weight` and against balanced sampling.
- Optimiser: AdamW, weight decay 0.05. LR 1e-3 for M3/heads and 3e-5 for the ViT top blocks. Cosine schedule with 1 warm-up epoch. Up to 30 epochs, early stopping on **val patient-level AUC**, batch of 32 slices or 8 patients.
- Training: **F1/F2 train end-to-end from the `facebook/vit-mae-base` weights**, the same initialisation as B3, so the image-only and fusion tracks run in parallel (D-022). Initialising F2 from the trained B3 is an ablation (`f2_from_b3`).

Interpretability: attention-weight tables per patient (as in Nishizawa), aggregated by column; attention rollout or Grad-CAM on ViT slices; and a feature-shuffle check of clinical importance on val.

## 5. Experiments

All experiments use the D-003 split and patient-level metrics (D-016). Final models (B3, B5, F2) use 3 seeds; ablations use 1.

| ID | Model | Purpose |
|---|---|---|
| B0 | Logistic regression on clinical features (no tumor size) | non-imaging floor |
| B1 | LR and RF on clinical + log bbox volume | **main comparator** (H1) |
| B2 | RF re-creating BreastDCEDL's I-SPY2 setup: + MammaPrint + treatment arm, with/without tumor volume | context only, I-SPY2-only; quantifies how much of the 0.85 is non-imaging |
| B3 | ViT image-only (ours, built correctly) | reproduces the reported ≈0.72 overall / ≈0.78 I-SPY2 range |
| B4 | Two-stage: B3 slice-score summaries (mean/min) **and** frozen B3 embeddings (PCA 32) → LR/RF with B1 features | the fair version of BreastDCEDL's two-stage design (H2) |
| B5 | Nishizawa re-implementation, single timepoint: 3D ResNet-50, first post-contrast, 80×256×256, 3 → 2 tokens (D-015) | CNN fusion reference |
| F1 | End-to-end ViT + clinical, concatenation + MLP | is attention needed? |
| **F2** | **End-to-end ViT + clinical, attention fusion (Section 4)** | **main model** |

**Tiers (D-019, updated by D-021):**
- **Core:** B0, B1, B2, B3, B4, B5, F1, F2.
- **Planned robustness:** leave-one-cohort-out; supervised-ImageNet backbone (B3-supervised).
- **Stretch:** our own nnU-Net; SSL pretraining; cohort token; loss variants; number of fine-tuned blocks.

Core ablations, all on F2 and selected on val only (1 seed each):
- aggregation (mean vs attention-MIL);
- with/without bbox-volume token;
- with/without aux heads;
- F2 initialised from MAE vs from the trained B3;
- ground-truth vs M1-predicted crops (deployment test, on the M1 subset).

Robustness:
- Per-cohort results; per-subtype results (HR+/HER2−, HER2+, triple-negative); calibration per cohort.
- Leave-one-cohort-out (train on two cohorts; evaluate on the third cohort's non-test patients), reported as a secondary analysis.

Sanity check of the shipped BreastDCEDL weights: run them on our uint8 inputs for val. If val AUC ≥0.65, add a row "shipped BreastDCEDL ViT (our inputs)"; if not, record the result and drop it.

## 6. Evaluation protocol (D-006, D-016)

- Per model: AUC with 2,000-sample bootstrap 95% CI, AUPRC, sensitivity and PPV at 90% specificity, Brier score, ECE and a reliability plot. Thresholds are chosen on val.
- Comparisons: paired DeLong (and paired bootstrap) on the same test patients, for F2 vs B1 (H1) and F2 vs B4 (H2). Mention the multiple-comparison count when reporting.
- Every breakdown lists N and the number of pCR cases. Duke test has 8 pCR, so its CI will be wide; report it, don't hide it.
- Each test-set evaluation is logged in `docs/RESULTS.md` with date, commit and config hash.
- Clinical framing for MNJ: a false "pCR" call could lead to under-treatment. Report the operating point at high specificity and its PPV/NPV.

## 7. Repository layout

```
TIAIH_porj/
  IMPLEMENTATION_PLAN_FINAL.md   data_availability.md   README.md   requirements.txt   requirements-kaggle.txt
  docs/        DECISIONS.md  DATA_NOTES.md  RESULTS_val.md  RESULTS.md
               team/ TEAM_OVERVIEW.md  PLAN_PERSON_A.md  PLAN_PERSON_B.md  PLAN_PERSON_C.md
               report/ (report sections, one file per author/section)
  kaggle/      00_download_zenodo.py      # paste-into-one-cell script, no GPU
               _template.ipynb            # clones a git tag with GITHUB_TOKEN, installs deps, mounts datasets
               01_build_cache.ipynb       # COHORT=ispy1|ispy2|duke, MODE=slices|vol3d|from_m1
               10_train.ipynb             # CONFIG, SEED, MODE=train|export
               20_b5.ipynb  30_m1.ipynb  40_explain.ipynb  50_final_eval.ipynb
  configs/     base.yaml  b0_clinical.yaml  b1_clin_size.yaml  b2_ispy2_context.yaml  b3_vit.yaml  b3_vit_supervised.yaml
               b4_twostage.yaml  b5_nishizawa.yaml  f1_concat.yaml  f2_attn.yaml  f2_meanpool.yaml  f2_mil.yaml
               f2_no_bbox.yaml  f2_no_aux.yaml  f2_from_b3.yaml  f2_final.yaml  loco_*.yaml
  data/        splits/split_v1.csv        # committed; everything else under data/ is gitignored
  scripts/     make_split.py  build_cache.py  map_ids.py  download_mamamia.py  final_eval.py
  src/
    data/      metadata.py  nifti_io.py  rgb_slices.py  clinical.py  datasets.py  transforms.py
    models/    vit_encoder.py  pooling.py  fusion.py  nishizawa3d.py  localization.py
    baselines/ tabular.py                 # B0, B1, B2, B4 heads
    train.py  evaluate.py  stats.py  explain.py  infer.py
  notebooks/   01_inspect_volumes.ipynb  02_eda.ipynb  03_shipped_weights_check.ipynb
  tests/       test_split_frozen.py  test_no_patient_leakage.py  test_rgb_uint8.py  test_fusion_shapes.py  test_clinical_encoding.py
  results/     A/  B/  C/  final_manifest.csv  final_eval_log.csv
```

What stays out of git (`.gitignore`): all data, caches, archives and weights (they live in Kaggle Datasets), secrets, and the paper PDFs.

Key tests:
- `test_rgb_uint8.py` guards against the BreastDCEDL conversion bug: the output dtype is uint8 and correlates with the source image at r > 0.99.
- `test_no_patient_leakage.py` asserts there is no patient overlap between splits, or between SSL data and test.
- `test_split_frozen.py` checks the md5 of `split_v1.csv`.

Dependencies: torch, torchvision, transformers, timm, nibabel, SimpleITK, numpy, pandas, scikit-learn, scipy, matplotlib, pyyaml, tqdm, nnunetv2, synapseclient. Stage 2 (future) would add pydicom, dcm2niix, torchio and tcia_utils.

## 8. Schedule (5 weeks, 3 people)

Day 0 = Thu 2026-10-08. Per-person tasks, handoffs (H0–H12) and fallbacks are in `docs/team/`; this is the combined view (D-022).

| Days | A (code owner) | B (image-only track) | C (fusion track) |
|---|---|---|---|
| 0 | overnight download (I-SPY1 + models + CSV); write the core code | overnight download (I-SPY2) | overnight download (Duke) |
| 1 | inspect a real volume, finalise the builder, tag `d1`; I-SPY1 cache | I-SPY2 cache; B3 debug run | Duke cache; F2 debug run |
| 2–6 | `stats.py`; B0/B1/B2; code for B5, LOCO and M1 (tag `d6`) | B3 ×3 seeds, B3-supervised; exports (day 5) | F1; F2 ablations (aggregation, bbox token, aux heads) |
| 7–10 | B4; deployment-crop builder + `explain.py` (tag `d9`); `final_eval.py` | B5 ×3; M1 inference (Synapse); LOCO ×3 | **freeze F2 (day 8)**; F2 ×3 seeds; F2-from-B3 ablation |
| 11–15 | merge val results; freeze the code | deployment test on val | explainability; per-cohort/subtype analysis on val |
| **16** | **single final test evaluation** → `RESULTS.md` | | |
| 17–24 | `infer.py` (tag `d21`); evaluation write-up; Stage 2 note | report sections | second-machine test of `infer.py`; report sections |
| 25–34 | merge and edit the report; slides | slides; fixes | slides; fixes |

Rules:
- Real dependencies only: B4 needs B3 (day 5); the deployment test and explainability need the frozen F2 (day 10); the final evaluation needs every model (day 15).
- If something slips, cut stretch work first. Never cut the final evaluation, the stats or the write-up.
- Days 25–34 are buffer and writing time.

**Compute budget** (three Kaggle accounts, about 30 GPU-h/week each). These are estimates; re-measure on day 1–2.

| Job | Estimate |
|---|---|
| ViT-B/16, 224 px, AMP, top 4 blocks trained, about 20–25k tumor slices/epoch | about 2–4 min/epoch on T4 → about 1–2 h per run (≤30 epochs, early stopping) |
| B: B3 ×4 + exports, then B5 ×3, M1, LOCO ×3 | about 20 h (week 1) + 20–25 h (week 2) |
| C: F1/F2 debug + 6 ablations, then F2 ×3, F1 ×2, F2-from-B3 | about 20–25 h (week 1) + 15–20 h (week 2) |
| A: downloads and cache (CPU), final evaluation | about 3–5 h total |

## 9. Stage 2: paired pre + post-chemo (future work, design note only)

This is not implemented in the 5 weeks (D-019). The deliverable (Person A, days 22–24) is a 1–2 page design note containing:
- **Feasibility, already verified** (data_availability.md Section 4): about 870 labeled I-SPY2 pairs (T0 + T3) and 148–171 I-SPY1 pairs. Still to check: DCE-series presence, DCE-only download size, image download without login, and pCR-definition consistency.
- **Split:** inherit D-003 per patient (D-014). This lets Stage 1 weights initialise Stage 2 without leakage.
- **Pipeline:**
  - TCIA → NIfTI by process-and-discard.
  - Filter for the unilateral cropped DCE + SER/PE2/PE6 series (as in Nishizawa's README).
  - Run M1 on the baseline scan; transfer the tumor-bed ROI to the post-chemo scan. An empty post-chemo segmentation is a feature, not an error.
- **Model:** F2-paired with tokens [CLS, f_pre, f_post, f_post − f_pre, clinical], plus a shared ViT initialised from Stage 1. Baselines: B5 on T0 + T3, and ΔFTV + clinical logistic regression. External validation: I-SPY2 → I-SPY1.
- **Effort estimate:** about 4–6 additional weeks. This is the path for MNJ's data once IRB approval arrives.

## 10. MNJ handoff

- **Input contract:** a DICOM folder for the DCE series (pre-contrast plus at least 2 post-contrast phases), and a clinical JSON with HR, HER2, age, race and menopause. For Stage 2, add the post-chemo DCE folder.
- **Pipeline:** DICOM → NIfTI → M1 → crop → F2 → probability, heatmap and attention table. It runs locally, and no data leaves the hospital.
- **When MNJ's roughly 100 cases arrive:**
  1. External validation first, with no training on them.
  2. Then, optionally, fine-tuning with cross-validation.
- **Questions for MNJ:**
  - scanner/protocol and phase timing;
  - interval from MRI to surgery;
  - pCR definition (ypT0/is vs ypT0 ypN0);
  - receptor thresholds used;
  - de-identification and IRB constraints.

## 11. Risks

| Risk | Mitigation |
|---|---|
| B3 lands well below about 0.72 | debug in order: inputs, slice selection, normalisation, label join, augmentation; try the shipped weights; report honestly, since the published number came from an unreleased run on a different split |
| F2 does not beat B1 | a legitimate finding, consistent with Nishizawa's external result; report it with CIs, and analyse subgroups and calibration where the image may add value |
| Too few leakage-free patients for M1 evaluation | report on what qualifies, labelled as limited; TCIA full-res as a second source; own nnU-Net is stretch (D-011) |
| Duke drags pooled metrics | per-cohort reporting; cohort-token ablation; leave-one-cohort-out |
| Receptor definition differences across cohorts | per-cohort calibration; noted in the report |
| Kaggle time and disk limits | uint8 npz cache, checkpoints saved to a Kaggle Dataset, resumable training |
| Zenodo / GitHub / TCIA download slowness | resumable downloads; fetch single files; mirror once |
| 5-week deadline | parallel tracks; handoff table with fallbacks (`docs/team/`); cut stretch first; days 25–34 are buffer |
| Code owner (A) is a single point of failure | small tagged releases; B and C report errors with logs; core code is written on day 0 |
| Zenodo throttling (13 KB/s locally) | overnight Kaggle downloads, one cohort per person; `RESUME_FROM` if a run hits the 12 h limit; develop on I-SPY1 first |
| Misframing vs the MNJ brief | the framing sentence in Section 1 goes in every deliverable |

## 12. Open items

1. Day 1: did the overnight downloads finish, and at what speed (D-020)?
2. Day 1: slice axis and orientation per cohort; whether the CSV coordinates match the min_crop volumes; whether `BreastDCEDL_spy1.zip` is full FOV (only if needed for M1).
3. Day 7: the MAMA-MIA ID format and its overlap with our split (sets the M1 evaluation size).
4. Stage 2 feasibility items, for the design note only (data_availability.md Section 4).

## 13. Deliverables

- Frozen split and cache build scripts.
- Trained B0–B5, F1 and F2 with configs, plus the LOCO and ablation runs.
- `docs/RESULTS.md`: val ablations, the single test evaluation, CIs, DeLong, per-cohort/subtype results, calibration and the deployment test.
- Module 1 report.
- Explainability figures.
- `infer.py` demo.
- Report, slides and the MNJ handoff note.
- Stage 2 design note (future work).
