# Ideas log

What we discussed, what we decided, and why. All AUCs are on the **validation set** (the test set is
untouched). Validation is small: 177 patients (53 pCR+), or 132 (43 pCR+) for I-SPY only. A 95% CI on
one AUC is about ±0.08, so differences of a few points between single runs are usually noise.

Status: **Done** (tried, result below) · **Accepted** (will do) · **Accepted, changed** (will do, in a
modified form) · **Proposed** (suggested, not decided yet) · **Deferred** (maybe later) · **Rejected**.

---

## 2026-10-26

### Where we start

- The best image-only model is DINOv2 ViT-B/14 on the early post-contrast phase, with 2 trainable blocks.
  It reaches **AUC 0.61–0.65** depending on the run.
- **Subtype lookup:** predict each val patient's pCR as the train pCR rate of their HR/HER2 subtype.
  This scores 0.648 on the 132 I-SPY val patients and 0.662 on all 177.
- The image model **overfits**: val AUC peaks around epoch 11 while training loss keeps falling.

### 1. Regularise DINOv2 (dropout, drop path) and train on I-SPY only: Done; no effect

Sweep `results/A/sweep_b3_vit_1019-1820` (tag d1.9, seed 0). Baseline: DINOv2 early-phase trained on
all cohorts, 0.636 on the same 132 I-SPY val patients.

| run | val AUC (best epoch) | vs baseline (paired bootstrap) | mean AUC, epochs 5–15 |
|---|---|---|---|
| I-SPY only | 0.623 | −0.013 (−0.07 to +0.04) | 0.593 |
| + dropout 0.5 | 0.648 | +0.012 (−0.05 to +0.07) | 0.612 |
| + drop path 0.2 | 0.625 | −0.011 | 0.590 |
| + both | 0.623 | −0.013 | 0.592 |

- **No difference is significant.** The four runs' predictions correlate 0.87–0.98: they are nearly
  the same model.
- **Overfitting is not fixed.** Every run still peaks at epoch 11 and then declines.
- **Training on I-SPY only does not help**, so Duke is not hurting and stays in training.
- **Decision:**
  - Keep `dropout 0.5`: it was ahead on every measure, but this is a weak preference.
  - Stop tuning regularisation on its own. Four sweeps put image-only AUC at about 0.60–0.65 whatever we
    changed.

### 2. Does the image add anything beyond subtype? Done (quick check); promising

A check computed outside the repo:
- the image model's AUC within each subtype;
- an average of the image ranking and the subtype ranking, with no training.

| (132 I-SPY val patients) | AUC |
|---|---|
| subtype lookup | 0.648 |
| image (dropout 0.5 run) | 0.648 |
| image + subtype (rank average) | **0.697** |
| baseline image + subtype | 0.695 |

AUC within each subtype (dropout run):

| subtype | patients | pCR+ | AUC |
|---|---|---|---|
| HR+/HER2− | 48 | 9 | 0.73 |
| HER2+ | 37 | 18 | 0.62 |
| triple negative | 47 | 16 | 0.59 |

- The image model ranks patients **within** subtypes, so it carries information that receptor status
  does not.
- **Caveats:**
  - The groups are small; HR+/HER2− has only 9 positives.
  - No significance test was run.
  - The lookup was never a saved run, which is why item 3 exists.

### 3. Save the subtype lookup as a proper baseline run: Proposed

Make the lookup a baseline config with saved `val_preds.csv`, so image + clinical models are compared
against it by paired bootstrap between saved runs instead of by hand.

### 4. Hyperparameter tuning with Optuna: Accepted, changed; after the EDA

Search over learning rates, dropout, weight decay, trainable layers, backbone, slices per patient and
input type.

- **Rejected form: tuning on the validation set.** After ~50 trials the best val AUC is about +0.05
  high by chance, and val stops being a fair yardstick.
- **Accepted form:**
  - **Tune on a held-out 20% of train** (a "tune" split, stratified by cohort and pCR).
  - **Score each trial by mean tune AUC over its last epochs**, not the single best epoch.
  - **Prune bad trials early** with Optuna, using per-epoch AUC read from `log.csv`. No changes to
    `src/train.py`.
  - **Save the study to sqlite** so it resumes across Kaggle sessions.
  - **Confirm only the top 2–3 settings on val**, with 3 seeds each.
- **Search space:**
  - backbone: dinov2 / resnet18 / mae
  - lr_backbone: 1e-6 – 1e-4
  - lr_head: 1e-4 – 3e-3
  - train_blocks: 0 / 1 / 2 / 4
  - dropout: 0 – 0.6
  - weight_decay: 0.01 – 0.3
  - k_train: 1 / 4 / 8 / 16
  - input: early / rgb, plus subtraction (item 6) once it exists
  - max_epochs: 30, with pruning
- **Budget:** about 30–40 trials per 12-hour session.
- **Expected gain:** realistically +0.02–0.04.

### 5. DINOv2 with 16-pixel patches (/16) instead of /14: Proposed, as DINOv3

- **DINOv2 /16 does not exist.** Meta released DINOv2 only with 14-pixel patches.
- **The /16 option is DINOv3 ViT-B/16** (Meta, 2025). It is on Hugging Face, but gated: accept the
  licence and add an HF token as a Kaggle secret.
- **Patch size matters little:** 256 tokens at 224 px (/14) vs 196 (/16).
- **Why consider it:** DINOv3 is a newer, stronger model.
- **Open decision:** the user sets up the HF token.

### 6. Subtraction / relative-enhancement input: Proposed

- **Input:** feed (early − pre)/pre and (late − pre)/pre maps instead of, or alongside, raw phases.
- **Why:**
  - These maps show contrast uptake directly.
  - They largely cancel scanner gain, so they are also the most scanner-independent intensity
    normalisation for DCE (see item 14).

### 7. Tighter tumour crop: Proposed

- **Problem:** the 224-px input covers 157 mm, so a 2 cm tumour is only about 30 px wide.
- **Proposal:** crop about 80 mm around the tumour and scale it to 224 px, giving roughly twice the
  detail on the tumour.
- **Trade-off:** less surrounding tissue.

### 8. Run the EDA notebook on cache v2 first: Accepted; next step

The EDA notebook already handles v2. Run it on Kaggle with:
- the cache v2 datasets and the raw I-SPY1 dataset attached;
- `GIT_TAG = "d1.9"`.

Review the output before choosing the next experiments. The aim is to catch pipeline errors like the
ones the v1 EDA found.

### 9. Why 8 slices per patient, and shouldn't the model see one slice at a time? Explained; k_train goes in the search

- **pCR is a patient label, not a slice label.** The model encodes 8 slices, scores each one, and
  averages the scores into one patient prediction. The loss is on that prediction.
- **Single-slice training gives every slice the patient's label**, including edge slices with almost no
  tumour. That makes the labels noisy.
- **8 is a GPU memory/speed compromise**, not a tuned value.
- **Decision:** add `k_train` (1 / 4 / 8 / 16) to the Optuna search. `k_train=1`, averaged over 16
  slices at evaluation, is the single-slice version.

### 10. Use the leftover tumour slices as extra data points: Rejected

- **They are already used.** Each epoch draws a new random 8, so over 30 epochs the model sees every
  tumour slice.
- **They are not independent samples.** Slices from one patient are nearly identical and share one
  label. About 1,000 patients × 15 slices is still about 1,000 independent examples, and the patient
  count is what limits us.

### 11. Use 3D information: Accepted; after the radiomics baseline

- **Today the model is "2.5D":** slices are encoded separately and averaged. It ignores slice order and
  how the tumour changes across slices.
- **11a. Slice attention:**
  - Replace the averaging with a small attention/transformer layer over slice features, with
    slice-position encoding.
  - It keeps the pretrained 2D encoders and is cheap. Do this first.
- **11b. 3D CNN:**
  - A 3D ResNet-18 on a tumour-centred crop (e.g. 96×96×32 voxels), with the 3 phases as channels.
  - Pro: it sees true 3D shape and how enhancement varies across the tumour.
  - Cons: 3D pretrained weights (MedicalNet, video models) are weaker than ImageNet/DINOv2, and ~1,000
    patients is little for a 3D network.
  - It needs a consistent voxel grid; v2 provides one (see item 13).
  - Published 3D CNNs on I-SPY2 report about 0.70–0.80, usually with more timepoints.

### 12. Feature engineering (radiomics) + subtype baseline: Accepted

Hand-made tumour features from the mask, fed to logistic regression or gradient boosting together with
subtype:
- **kinetic:** wash-in (early − pre)/pre, washout (late − early)/early, fraction of tumour that washes out;
- **shape:** volume, sphericity, surface/volume;
- **texture:** pyradiomics enhancement heterogeneity.

Why:
- It runs on CPU in minutes and is easy to explain.
- It shows whether the images hold signal the ViT is missing.

**Kinetic features must come from the raw arrays**, not the uint8 cache: the cache's clipping and
offset distort the intensity ratios.

### 13. Professor: "keep all native slices, don't resample to 2 mm": Accepted, changed; ablation

- **For the 2D bag models, he is right.**
  - They never use the distance between slices.
  - z-resampling creates blended in-between slices for thick-slice scans (e.g. 3.4 mm) and slightly
    blurs thin-slice scans.
  - We never tested whether it helps.
- **Why it was done:**
  - Each tumour gets equal sampling weight per mm. With native slices, a 1.2 mm scan has about 25 tumour
    slices and a 3.4 mm scan about 9, and we sample 8 from either. That effect is mild.
  - A 3D CNN (item 11b) needs a fixed voxel size.
- **Decision:**
  - Build a **v3 cache: 0.7 mm in-plane, native slices** (no z-resampling). This needs a small builder
    change, because `--z-mm 0` currently divides by zero.
  - Train the same DINOv2 setting on v2 and v3 and compare by paired bootstrap.
  - Keep v2 for the 3D CNN.

### 14. Professor: "different scanners are common; normalise intensity only, not resolution": Rejected for in-plane resolution (testable); agreed on intensity

- **In-plane resampling fixed a measured problem.** In v1, 224 px covered 70–315 mm, so the same 2 cm
  tumour was about 64 px wide on one scan and about 14 px on another. CNNs/ViTs are not scale-invariant,
  and ~1,000 patients is too few to learn it.
- **It is common practice:**
  - deep-learning MRI pipelines (e.g. nnU-Net) resample to a common spacing;
  - the radiomics standard (IBSI) recommends a common voxel size, because texture features change with
    voxel size.
- **Agreed:**
  - Scanner differences show up mostly as **intensity** differences, so intensity normalisation matters
    most. See item 6, relative-enhancement maps.
  - Upsampling adds no real detail.
- **Caveat:** our v1 → v2 gain (0.53 → 0.56) mixed several fixes and was not significant, so there is
  no clean evidence yet. An in-plane ablation (native pixels vs 0.7 mm, everything else equal) would
  settle it.

### 15. Professor: domain adaptation / generalisation: Proposed

- **The domains:** our cohorts and their scanners. A model can learn site shortcuts; this is likely one
  reason Duke comes out at chance.
- **Proposed method: domain-adversarial training (DANN).**
  - A small extra head predicts the cohort (or protocol group, item 17) from the encoder's features.
  - A gradient-reversal layer trains the encoder to make that impossible.
  - The features then keep pCR information and lose site information.
- **Cost:** one extra head and one loss term.

### 16. Professor: GANs: Deferred (low priority)

- **Possible uses:**
  - **harmonisation:** CycleGAN to translate one scanner's image "style" into another's;
  - **synthetic training images;**
  - **synthesising missing phases.**
- **Why deferred:**
  - With ~1,000 patients GANs are hard to train.
  - They can hallucinate or erase subtle tumour texture, which is exactly the signal we need.
  - High effort, high risk.

### 17. Professor: ComBat harmonisation: Accepted, with the radiomics baseline (item 12)

- **What it is:** batch-effect correction from genomics, used in radiomics on **features**, not images.
  For each feature it removes the shift and scale between batches (sites/scanners) with an
  empirical-Bayes adjustment, and keeps chosen covariates.
- **Apply to:** radiomics features, and possibly DINOv2 slice features.
- **Rules:**
  - Fit on train only.
  - Keep subtype as a covariate; cohorts differ in true pCR rates.
  - Never use pCR as a covariate (label leak).
- **Batch definition (revised in discussion):** first proposed as "batch = cohort", because there is no
  scanner column. The user pointed out that acquisition resolution identifies the scanner/protocol, so
  we use **protocol groups** instead of cohort.
- **Raw combos are too fragmented to use directly.** Each combo is (matrix, pixel size, slice thickness):

  | cohort | patients | distinct combos | combos with ≥ 20 patients |
  |---|---|---|---|
  | Duke | 916 | 118 | 10 (523 patients) |
  | I-SPY1 | 172 | 43 | 2 (63 patients) |
  | I-SPY2 | 982 | 199 | 10 (415 patients) |

  ComBat estimates a shift and scale per batch, so batches need roughly ≥ 20–30 patients. Most combos
  have 1–10, and Duke has only 298 labeled patients.
- **Resolution identifies the protocol, not the scanner exactly.** One scanner runs several protocols,
  and different vendors can share a protocol.
- **Plan: coarse protocol groups.**
  - **Matrix:** 256 / 512.
  - **Slice thickness:** ≤ 1.5 / 1.6–2.4 / ≥ 2.5 mm.
  - **Pixel size:** < 0.65 / ≥ 0.65 mm.
  - **Batch:** cohort × group, with small groups merged.
  - Expect about 6–12 batches.
  - **Check:** after ComBat, a classifier should not be able to predict the batch from the features.
- **Better, if available:**
  - Duke's TCIA release has a table of MRI technical details, possibly with manufacturer and field
    strength (to confirm).
  - I-SPY's original DICOM headers have these fields.
  - True scanner batches (vendor × field strength) would replace the protocol groups.

---

### Order of work agreed today

1. Run the EDA on cache v2 (item 8) and review it.
2. Build the v3 cache (in-plane only) and run the z-resampling ablation (item 13).
3. Radiomics + ComBat + subtype baseline (items 12, 17); save the subtype lookup as a baseline (item 3).
4. Slice attention (item 11a), subtraction input (item 6), tighter crop (item 7).
5. Optuna search on a tune split (item 4), then confirm the top settings with 3 seeds.
6. Later: DANN (item 15), 3D CNN (item 11b), DINOv3 (item 5). GANs last (item 16).

### Implemented (code tag d1.10), waiting for Kaggle runs

- **Item 13, cache v3:** `scripts/build_cache.py --z-mm 0` keeps the native slices; in-plane and clip are
  as in v2. Builder version 3, folders `cache_v3_<cohort>`. Build it with `01_build_cache`, `Z_MM = 0`.
- **Item 6, enhancement input:** `model.input_phases: enhancement`. The channels are:
  - early;
  - (early − pre)/pre, mapped from 0–300% to 0–1;
  - (late − pre)/pre, same mapping.

  The ratios use raw-proportional intensities, from the cache's stored clip range.
- **Items 6 + 13, experiment:** notebook `14_v3_enhancement`. DINOv2, 2 trained blocks, dropout 0.5,
  seeds 0 and 1. Runs: v2_early (baseline), v3_early, v2_rgb, v2_enh, v3_enh. Each run picks its own cache
  version.
- **Items 3, 12, 17, radiomics + ComBat:** notebook `02_radiomics` (CPU).
  - **Features:** `src/radiomics.py`, 29 of them (shape, kinetics, enhancement texture, 5 mm ring).
  - **ComBat:** `src/combat.py`, fitted on train, batches = protocol groups.
  - **Baselines:** `src/baselines/radiomics.py`: subtype lookup (now a saved run), rad, rad_sub, and both
    without ComBat.

### 2026-10-28: radiomics result and a ComBat pitfall

- **Results** (details in `docs/results.md`, section 8):
  - radiomics only: 0.63;
  - radiomics + subtype: 0.70, the best so far but not significant vs subtype (+0.04);
  - ComBat removes the batch signal but does not change AUC.
- **Pitfall found: Accepted fix.**
  - **Problem:** ComBat with HR/HER2 as covariates leaks subtype into "radiomics-only" features, which
    gave a false 0.709.
  - **Rule:** use ComBat covariates only in models that are allowed to see those covariates anyway.
  - **Applies to ComBat on deep features too.**

### 2026-10-28: 3D information (item 11): Implemented (d1.11), waiting for Kaggle runs

- **Where things stood:**
  - Radiomics was already mostly 3D (shape, kinetics, entropy and ring all use every tumour slice); only
    the GLCM texture was per slice.
  - Every deep model was 2D: each slice encoded alone, the scores averaged, slice order ignored.
- **Why the deep models were 2D:** pretrained weights are 2D, the data is small (~1,100 labeled training
  patients), and the paper's baseline was 2D. These made 3D harder, not pointless.
- **Implemented:**
  - **A. Slice attention** (`model.aggregation: slice_attn`): a 2-layer transformer over the slice
    features plus their position in the tumour, with a [CLS] token, instead of averaging.
  - **B. 2.5D input** (`model.input_phases: early_25d`): the 3 channels are the early phase of slices
    z−1, z, z+1.
  - **C. 3D CNN** (`model.backbone: r3d_18`): torchvision 3D ResNet-18 with Kinetics-400 video weights.
    - **Input:** a block of 32 contiguous slices (64 mm) × 112 × 112 mm around the tumour, resized to
      1 mm pixels.
    - **Channels:** pre/early/late, or the enhancement maps.
    - **Bag:** the dataset's `block` sampling provides the slices (random block position in training,
      centred in evaluation).
  - **D. 3D GLCM:** the radiomics texture now uses all 13 3D neighbour directions (`glcm3d_*`). The
    features must be re-extracted with `02_radiomics`.
- **Experiment:** notebook `15_3d`, 2 seeds per run.
  - **Baseline:** DINOv2 early with mean aggregation.
  - **Runs:** A, B, 3D CNN with RGB channels, 3D CNN with enhancement channels.
