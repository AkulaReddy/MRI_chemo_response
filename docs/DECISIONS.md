# Decisions Log

Project: MNJ project #11, pCR prediction from breast DCE-MRI.
Companion files: `IMPLEMENTATION_PLAN_FINAL.md` (what we build), `data_availability.md` (verified data facts).

Terminology: "Option 1" = **Stage 1** (pre-treatment prediction, this project); "Option 2" = **Stage 2** (paired pre + post-chemo, future work). Earlier entries use the old names.

Each entry records what was decided, why, and what follows from it. Status is **Accepted** (agreed), **Proposed** (default chosen, awaiting confirmation) or **Superseded**. Add new entries at the bottom; never delete one, mark it superseded instead.

---

## D-001 Plan documents
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:** `IMPLEMENTATION_PLAN_FINAL.md` is the only plan. It replaces three earlier draft plans (a v1 plan, an "Option 1" pre-treatment plan and an "Option 2" paired plan), which have been deleted. `data_availability.md` is the source of truth for data facts and overrides the plan; its Section 5 records what the drafts got wrong.
- **Why:** The earlier plans had several wrong facts: the dataset size, "I-SPY2 missing", the split counts, "RGB-fused 2D images", the ViT backbone, and what the 0.85 means. They were corrected only in data_availability.md.

## D-002 Scope: Option 1 first, Option 2 as a gated extension
- **Date:** 2026-10-08 · **Status:** Superseded by D-019 (5-week timeline)
- **Decision:**
  - The primary deliverable is pre-treatment pCR prediction (Option 1) on BreastDCEDL.
  - Paired pre + post-chemo prediction (Option 2) is a second stage. It starts only after the Option 1 pipeline works and the Option 2 go/no-go check passes.
  - The cheap Option 2 feasibility check (DCE-series metadata, download size) runs in parallel during the first weeks.
- **Why:** Option 1 has preprocessed data, 1,452 labeled patients and published numbers to compare against. Its code (loader, ViT, fusion, evaluation, Module 1) carries over to Option 2. Option 2 matches MNJ's real question, but needs raw TCIA DICOM work.
- **Consequence:** Every Option 1 deliverable must state that it predicts response *before* treatment and does not answer MNJ's post-chemo question.

## D-003 Canonical split: Zenodo `BreastDCEDL_metadata_min_crop.csv`
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:**
  - Use the `test` column of the Zenodo CSV (record 18114231, md5 `d58b84d0f343b4b8338cbb222e6dda08`): 0 = train, 2 = val, 1 = test.
  - Labeled patients: 1,099 / 177 / 176. The test set is I-SPY1 35, I-SPY2 99 and Duke 42.
  - We do **not** try to reproduce either paper's exact split.
- **Why:**
  - Three metadata CSVs exist, and they disagree on Duke labels (19 patients), split membership and patient count.
  - The BreastDCEDL article's 175-patient test set matches none of them; one of its test patients is in Zenodo train.
  - Nishizawa's folds were never released.
  - The Zenodo file is the published, checksummed artifact.
- **Consequence:** Published numbers are context, not head-to-head comparisons (see D-004). Store a frozen copy as `data/splits/split_v1.csv` with its md5; all code reads only that file.

## D-004 Rerun every baseline on our split
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:** All comparisons in our results table come from models we train and evaluate on the D-003 split: clinical-only, clinical + tumor size, ViT image-only, two-stage ViT → classifier, a Nishizawa-style re-implementation and our fusion model. Published AUCs (0.72 / 0.78 / 0.85, Nishizawa 0.73 / 0.71) are quoted only as "reported", never as rows in our comparison.
- **Why:** The splits differ (D-003). BreastDCEDL's 0.85 uses features that are not imaging (D-007). Nishizawa's internal number has no held-out test set.

## D-005 One image source per pipeline stage
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:**

  | Stage | Image source | Patients |
  |---|---|---|
  | Classifier training and evaluation (Modules 2 + 3) | Zenodo min_crop volumes for every patient, one preprocessing pipeline | all 1,452 labeled (+ unlabeled for SSL, see D-012) |
  | Module 1 (localization) evaluation | full-field-of-view volumes: TCIA BreastDCEDL-ISPY2 full-res first, MAMA-MIA second, I-SPY1 full (`BreastDCEDL_spy1.zip`) if confirmed uncropped | test (and val) patients that have them |
  | End-to-end deployment test | full FOV → our Module 1 → crop → classifier | the same subset; report the AUC drop from ground-truth crops to predicted crops |

- **Why:** Mixing sources within the classifier would give patients systematically different inputs:
  - MAMA-MIA is resampled to 1 mm and z-scored, while Zenodo is raw intensity at native spacing;
  - full-FOV coverage differs by cohort (MAMA-MIA has 291 of 916 Duke).

  The source would correlate with cohort and pCR rate and act as a confounder.
- **Consequence:** Full-FOV images are used only for localization and the deployment test. Coordinates (`sraw…`, `mask_start/end`) must come from the CSV that matches the volumes being loaded.

## D-006 Test set is touched once
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:**
  - All model selection, hyperparameter search, early stopping and ablations use val (`test==2`) or stratified k-fold on train + val.
  - `test==1` is evaluated once per final model, after the configuration is frozen.
  - Each test evaluation is logged in `docs/RESULTS.md` with date and git commit.
- **Why:** The test set is small (176 patients, 53 pCR), so repeated peeking would quickly overfit to it.

## D-007 What counts as the benchmark
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:**
  - The primary hypothesis is that **end-to-end ViT + clinical fusion beats the clinical + tumor-size baseline** on `test==1` (paired DeLong test). The secondary hypothesis is that it beats the two-stage ViT → classifier.
  - BreastDCEDL's 0.85 is not a target. It is a Random Forest on I-SPY2 only, using MammaPrint, 13 treatment-arm drug indicators, ground-truth tumor and bbox volume, and two scalar ViT scores.
  - The 0.94 (HR+/HER2−) rests on 5 positive patients and is never cited as a target.
- **Why:** Nishizawa found clinical-only = fused (0.71) on external data. BreastDCEDL's RF ranks tumor volume first, and treatment arm strongly predicts pCR. Beating the strong non-imaging baseline is the claim that matters.

## D-008 Clinical inputs for the main model
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:**
  - Inputs: HR, HER2, age, race (white/black/other) and menopause, each with an explicit "missing" category. Menopause is missing for 210 patients.
  - Tumor size: **bounding-box volume in mm³**, from `sraw/eraw/scol/ecol/mask_start/mask_end` × voxel spacing. It is available for all cohorts and can be reproduced from Module 1 at deployment. Duke `tum_vol` is a size class and is not used as a continuous feature.
  - **Excluded from the main model:** treatment arm and MammaPrint (I-SPY2 only, and not available at MNJ in the same form). They are used only in a labelled I-SPY2 context baseline (B2).
  - A cohort indicator is tested as an ablation only.
- **Why:** The main model should use only inputs available for every cohort and at MNJ.
- **Known issue:** HR/HER2 thresholds differ by cohort (HR ≥1% / ≥10% / Allred >3; HER2 FISH ≥2.0 vs >2.2). Note this in the report and check per-cohort calibration.

## D-009 Image input construction (our own pipeline)
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:**
  - Build RGB slices ourselves from the CSV phase indices (`pre`, `post_early`, `post_late` → R, G, B).
  - Intensity: clip each patient's three phases **jointly** at the 0.5–99.5 percentiles of the volume, scale to [0, 1], then convert to uint8 explicitly (×255, round, cast). Scaling the three phases together keeps their relative enhancement.
  - Crop 224×224 around the tumor bbox center; at deployment the bbox comes from Module 1.
  - Train on all tumor slices; aggregate per patient by mean probability, ablating against central-k and attention pooling.
  - Do **not** reuse the BreastDCEDL inference code.
- **Why:** The repo's `to_rgb()` → `Image.fromarray(float64, mode="RGB")` turns images into byte noise under Pillow ≤10 (we measured correlation −0.01 with the true image) and errors under Pillow 12. Its shipped all-patient predictions score AUC ≈ 0.56 on test.
- **To verify in Phase 0:** that the slice axis is axis 0 (the repo indexes `acq[k]` and `mask_start/end` on axis 0), and the orientation for sagittal I-SPY1.

## D-010 ViT backbone
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:**
  - Main backbone: ViT-B/16 initialised from `facebook/vit-mae-base`, the same initialisation BreastDCEDL used.
  - Ablation: supervised ImageNet ViT-B/16 (timm `vit_base_patch16_224.augreg_in21k_ft_in1k`).
  - No SAM/MedSAM.
  - The shipped BreastDCEDL weights get a sanity check only: run on correctly built uint8 inputs. They are used as an initialisation or a row in the table only if they score well on val.
- **Why:** This matches the reference model. The earlier plans' "ImageNet supervised ViT-B/16" was not what BreastDCEDL used.

## D-011 Module 1 and nnU-Net leakage
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:**
  - First option: MAMA-MIA's pretrained nnU-Net, evaluated only on patients who are in neither MAMA-MIA's training split nor our train split. This requires mapping the IDs to MAMA-MIA's naming.
  - If too few patients qualify (target ≥50 labeled test/val patients with a full FOV), train our own nnU-Net on I-SPY masks from our train split.
  - Metrics: Dice, bbox IoU, centroid error, bbox-volume agreement and downstream AUC. Note that I-SPY masks are semi-automatic threshold masks, not expert outlines.
- **Why:** The MAMA-MIA model was very likely trained on many of our test patients.

## D-012 Self-supervised pretraining data
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:** Optional MAE-style continued pretraining uses only patients with `test != 1`: train + val, including the unlabeled Duke patients in those splits. Test-split images, labeled or not, are excluded.
- **Why:** Pretraining on test images (transductive learning) would blur the held-out claim.

## D-013 Auxiliary heads and leakage
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:** Auxiliary HR and HER2 heads attach to the image embedding **before** fusion. They are used as a training signal only and are not part of the pCR output path.
- **Why:** HR/HER2 are also clinical inputs to the fusion. Predicting them after fusion would be trivial and add no image signal.

## D-014 Stage 2 inherits the split
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:** In Stage 2, every I-SPY1/I-SPY2 patient keeps their D-003 assignment (train/val/test). All 869 labeled I-SPY2 pairs and the I-SPY1 pairs are in BreastDCEDL, so the assignment covers them all.
- **Why:** It removes the 91-patient leakage risk noted in data_availability.md Section 4, so Stage 1 weights can initialise Stage 2.

## D-015 Nishizawa re-implementation
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:**
  - Re-implement from the paper and code (the repo is incomplete and does not run).
  - Keep the 3-token, 8-dim fusion and the shared 3D ResNet-50 encoder, on first post-contrast 80×256×256 volumes.
  - Fix the flip augmentation, which is drawn once per loader in their code.
  - Use the D-003 split, not their CV.
  - Document every deviation.
  - In Stage 1 it runs on a single timepoint; in Stage 2 on T0 + T3.
- **Why:** It's the closest end-to-end fusion precedent and the natural reference baseline.

## D-016 Evaluation protocol
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:**
  - Patient-level metrics only.
  - Report AUC with 2,000-sample bootstrap 95% CI, AUPRC, sensitivity and PPV at 90% specificity, calibration (ECE and reliability plot), and Brier score.
  - Break results down per cohort (I-SPY1 / I-SPY2 / Duke) and per subtype, always with N and the number of pCR cases.
  - Use ≥3 seeds (mean ± SD) and paired DeLong tests between models.
  - Choose any threshold on val, never on test.
- **Why:** The test set is small, Duke has only 42 test patients (8 pCR), and published reports often omit CIs.

## D-017 Data download practice
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:**
  - Zenodo: resumable `curl -L -C -` plus md5 verification against data_availability.md.
  - GitHub: fetch individual files from `raw.githubusercontent.com`; never clone the BreastDCEDL repo (1.27 GB, mostly sample DICOM).
  - Extract archives into their own empty directories.
  - Scripts that read downloaded data run from outside those directories (`python -I` where possible).

## D-018 Compute
- **Date:** 2026-10-08 · **Status:** Superseded by D-020

## D-019 Five-week timeline: Stage 1 only, with trimmed scope
- **Date:** 2026-10-08 · **Status:** Accepted (the user confirmed 5 weeks for the whole project)
- **Decision:**
  - **Stage 1 (pre-treatment) is the whole project.**
  - **Stage 2 (paired pre + post-chemo) becomes documented future work.** Its deliverable is a design note plus the feasibility counts already verified in data_availability.md (about 870 labeled I-SPY2 pairs + 148–171 I-SPY1). No TCIA DICOM processing.
  - **Core (must ship):** B0, B1, B3, B4, F1, F2. Also Module 1 *inference* with the MAMA-MIA pretrained nnU-Net on a leakage-free subset, the single test evaluation with CIs and DeLong, per-cohort/subtype and calibration results, explainability figures, `infer.py`, and the report.
  - **Stretch (only if ahead of schedule):**
    - B2 (BreastDCEDL-style RF with treatment arm + MammaPrint);
    - B5 (Nishizawa 3D ResNet re-implementation);
    - training our own nnU-Net;
    - SSL continued pretraining;
    - leave-one-cohort-out;
    - supervised-ImageNet backbone ablation.
  - **Ablations kept (val only):** fusion type (F1 vs F2), bbox-volume token on/off, aux heads on/off, slice aggregation (mean vs attention-MIL).
  - **Seeds:** 3 for F2, B3 and B4; 1 for ablations.
- **Why:** Raw TCIA DICOM conversion, series selection and post-chemo localization alone would take more than 5 weeks. Stage 1 delivers the novel contribution and the deployable pipeline.
- **Consequence:**
  - Every deliverable states that the model predicts response before treatment.
  - If the MAMA-MIA leakage-free subset is too small (D-011), the localization evaluation is reported on whatever qualifies and labelled as limited. Training our own nnU-Net stays a stretch goal.

## D-020 Compute and data transfer
- **Date:** 2026-10-08 · **Status:** Accepted
- **Measured on the local machine:**
  - GPU: NVIDIA Quadro P520, 4 GB, compute capability 6.1, about 0.9 TFLOPS fp32; fp16 is no faster.
  - CPU: i7-10510U, 4 cores / 8 threads. RAM: 15.8 GB.
  - Free disk: C 85 GB, F 205 GB. PyTorch 2.6.0+cu118 with CUDA available.
  - General download speed: about 3.9 MB/s. **Zenodo download speed: about 13 KB/s** (throttled).
- **Decision:**
  - **All training runs on Kaggle GPU (T4 or P100, 16 GB).** The weekly free quota (~30 h) is enough for the core runs (see the plan's compute budget).
  - **Zenodo archives are downloaded inside a Kaggle notebook**, verified by md5, converted there into the uint8 slice cache, and saved as a private Kaggle Dataset. Test this on day 1; if Zenodo is slow from Kaggle too, start all three downloads in parallel immediately and work on I-SPY1 first.
  - The local machine is for code development, unit tests, small-batch debugging on CPU/GPU, tabular baselines (B0/B1/B4 heads), statistics and figures, and the `infer.py` demo (inference fits in 4 GB).
  - Code is device-agnostic and config-driven, so the same scripts run locally and on Kaggle.
- **Why:** Training ViT-B/16 on the P520 would be about 40–70× slower than on a T4, with 4 GB memory limiting batch size. At 13 KB/s, the 24 GB download would take weeks locally.

## D-021 Three-person split; one code owner
- **Date:** 2026-10-08 · **Status:** Accepted
- **Decision:**
  - **Person A** writes all code on the project laptop and pushes to one private GitHub repo. A also owns the data pipeline, the tabular baselines (B0/B1/B2/B4 heads), the single final test evaluation and `infer.py`.
  - **Person B** trains the image models: B3, B4 exports, F1, F2, ablations and seeds.
  - **Person C** owns Module 1 + the deployment test, B5, leave-one-cohort-out, explainability, and the second-machine test of `infer.py`.
  - B and C run code from tagged releases in Kaggle notebooks, and commit only to `results/` and `docs/report/`.
  - Week-1 Zenodo downloads are split by cohort: A takes I-SPY1 + models, B takes I-SPY2, C takes Duke.
  - Handoffs H1–H14 and their due dates are in `docs/team/TEAM_OVERVIEW.md`.
- **Why:**
  - Three Kaggle accounts give about 90 GPU-h/week, so GPU time stops being the bottleneck.
  - A single code owner keeps one consistent codebase, which matters because others only run it.
  - The split runs the three long, independent workstreams in parallel: image models, localization, and the CNN baseline/robustness work.
- **Consequence:**
  - These move from stretch to planned: B2, B5, leave-one-cohort-out and the supervised-ImageNet backbone ablation. This supersedes that part of D-019.
  - Our own nnU-Net, SSL pretraining and Stage 2 stay stretch / future work.
  - The schedule's critical path is A's code delivery. Every plan includes fallback tasks for when a handoff is late.

## D-022 Parallel tracks from day 1; overnight downloads
- **Date:** 2026-10-08 · **Status:** Accepted (supersedes the role assignment and schedule in D-021; the single-code-owner rule stands)
- **Decision:**
  - **Person B:** image-only track (B3, B4 exports, B5, leave-one-cohort-out, Module 1 inference, deployment test).
  - **Person C:** fusion track (F1, F2 and ablations, trained **from `facebook/vit-mae-base`** rather than from B3; F2-from-B3 becomes an ablation), plus explainability.
  - **Person A:** writes all code, runs the tabular baselines and the single final evaluation.
  - **Day 0 (2026-10-08):** each person starts a no-GPU Kaggle commit run of `kaggle/00_download_zenodo.py` overnight (A: I-SPY1 + models + CSV; B: I-SPY2; C: Duke). It saves md5-checked raw archives as `mnj-raw-*` Datasets.
  - **Day 1:** A inspects a real volume, finalises the cache builder (tag `d1`), each person builds their cohort cache from their raw Dataset, and B and C start training.
- **Why:**
  - F2 initialised from B3 created a needless serial chain. Training both from the same MAE weights lets the two tracks run at once.
  - Splitting download from cache build means a builder bug costs a quick rebuild, not another night of downloading. The builder can't be finalised before a real volume is inspected (axis order, orientation, coordinates).
- **Consequence:**
  - The schedule now has about a week of buffer (days 25–34).
  - The remaining real dependencies are B4 → B3 (day 5), the deployment test/explainability → frozen F2 (day 10), and the final evaluation → all models (days 15–16).
  - Module 1 moves from C to B, so B needs a Synapse account by day 5.
