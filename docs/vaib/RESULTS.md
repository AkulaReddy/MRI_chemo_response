# Results: branch `vaib` (image-only work + the professor's 2026-10-28 suggestions)

Status on 2026-10-28. Every number is on the **validation set** (177 patients, 53 pCR+); the test set has not
been touched. One val AUC has a 95% CI of about ±0.08, so a difference counts only if its **paired** CI excludes 0.
Rank settings by **CV AUC on train** (5-fold, ~1,100 patients, much less noise), and use val as the confirmation.

---

## 1. Where the team stands (summary of Akula's `results.md`, code up to d1.9)

| model | input | val AUC (95% CI) | note |
|---|---|---|---|
| Subtype lookup (HR/HER2, 3 rows) | clinical | **0.662** | the number to beat |
| B1 logistic regression | clinical + tumour size | 0.653 (0.57–0.74) | |
| F2 attention fusion (v1 cache) | image + clinical | **0.690** (0.60–0.77) | vs lookup +0.028, p 0.06; needs a rerun on v2 |
| F1 concat fusion (v1 cache) | image + clinical | 0.678 (0.59–0.76) | |
| ResNet-18, early phase (v2) | image | 0.626 (0.54–0.72) | single-epoch spike; level ~0.57 |
| DINOv2, early phase (v2) | image | 0.608 (0.52–0.70) | Duke at chance (0.44) |
| DINOv2, I-SPY only, dropout 0.5 | image | 0.648 (132 I-SPY patients) | epochs 5–15 mean 0.61 |
| image + subtype rank average | image + clinical | 0.697 (132 I-SPY patients) | the image adds to subtype |

13 regularisation / backbone runs in 3 sweeps changed nothing significantly. Every image model is at chance on Duke.

## 2. Our runs so far (`vaib`, all on the old uint8 v1 caches)

**Frozen-feature probe**: `results/vaib/probe_1019-0603/probe_results.csv`, 74 settings (MAE / DINOv2 ×
9 normalisations × 2 crops × 2 poolings).

| setting | CV AUC (train) | val AUC (95% CI) | within-cohort val |
|---|---|---|---|
| DINOv2 + `sub` + tumour crop + gap (best by CV) | **0.609** | 0.582 (0.49–0.67) | 0.553 |
| MAE + `sub` + tumour crop + cls | 0.601 | 0.542 | 0.515 |
| cohort one-hot only (the free AUC) | 0.552 | 0.544 | 0.500 |
| hand-crafted kinetic (16 features) | 0.531 | 0.615 (0.52–0.71) | 0.611 |

The kinetic features' 0.615 on val is **not** supported by CV (0.531, below the cohort-only 0.552). Read it
as validation-set luck. Raw caches remove the clipping and may change this; step 9 retests it.

**Fine-tuned ViT-only**: MAE + `kinetic` + tumour crop, 3 seeds (`results/B/vo_base_mae_kinetic_tumor_*`):
0.570, 0.573, 0.578; ensemble **0.574** (0.47–0.67); Duke 0.49, I-SPY1 0.50, I-SPY2 0.62.

## 3. The professor's suggestions (2026-10-28) and where each one is

| suggestion | what was built | code | notebook |
|---|---|---|---|
| ComBat harmonisation | ComBat with empirical Bayes, **fitted on train only** (inside every CV fold); batch = cohort or acquisition protocol; covariates HR + HER2 (never pCR) | `src/vitonly/combat.py` | step 9 (features), step 10 (ViT features) |
| 2D features | largest enhancing slice: first-order stats of 7 maps (pre, early, late, PE, SER, wash-in, wash-out) + 2D shape + 2D GLCM | `src/vitonly/radiomics.py` | step 9 |
| 3D features | whole tumour volume: the same 7 maps + 3D shape (volume, sphericity, elongation ...) + 3D GLCM (13 directions) | `src/vitonly/radiomics.py` | step 9 |
| temporal mapping of slices | each slice's pre → early → late curve → per-slice kinetics (area, PE, SER, wash-out fraction ...), aggregated across slices (mean, std, max, top-3, slice-to-slice change) | `src/vitonly/radiomics.py` | step 9 |
| normalise each slice separately | (a) features: intensity reference per slice vs per volume (`ref`); (b) ViT input: `slice_joint`, `slice_pre`, `slice_zscore`, `slice_phase`, each with a volume twin | `src/vitonly/norms.py`, `radiomics.py` | step 9 (a), step 10 (b) |

**Our interpretation of "normalise each slice separately":** one window per slice, **shared by its 3 phases**
(`slice_joint`, `slice_pre`, `slice_zscore`). Normalising every phase on its own erases the contrast
enhancement, which is the main signal (`per_phase`: probe val 0.53). That literal reading is kept as the
ablation `slice_phase`. Please confirm with the professor.

ROI for all hand-crafted features: tumour box × tumour slices, voxels with PE ≥ 0.7 (the I-SPY functional
tumour volume threshold). It is the same for every cohort, because Duke has only boxes and a cohort-specific
ROI would leak the cohort.

## 4. Step 9 results (CPU): fill in from `results/<PERSON>/step9/RESULTS_step9.md`

| feature set | ref | ComBat | + subtype | CV AUC | val AUC (95% CI) | Duke | I-SPY1 | I-SPY2 | vs lookup (p) |
|---|---|---|---|---|---|---|---|---|---|
| kinetic16 | volume | no / yes | no / yes | | | | | | |
| 2d | volume / slice | | | | | | | | |
| 3d | volume / slice | | | | | | | | |
| temporal | volume / slice | | | | | | | | |
| 3d+temporal | volume / slice | | | | | | | | |
| all | volume / slice | | | | | | | | |

The questions `RESULTS_step9.md` answers with paired tests:
1. Does ComBat help? (same features, with vs without)
2. Does the per-slice reference help? (volume vs slice)
3. 2D vs 3D vs temporal
4. Do the features add to subtype? (features + subtype vs the lookup)
5. Harmonisation check: can the cohort still be recognised from the features after ComBat?

## 5. Step 10 results (GPU): fill in from `results/<PERSON>/step10/RESULTS_step10.md`

| | volume twin | per slice | difference (paired) |
|---|---|---|---|
| probe DINOv2 CV AUC: `pre_ref` vs `slice_pre` | | | |
| probe DINOv2 CV AUC: `joint_wide` vs `slice_joint` | | | |
| probe DINOv2 CV AUC: `zscore` vs `slice_zscore` | | | |
| fine-tuned DINOv2 (2 blocks, 2 seeds): `pre_ref` vs `slice_pre`, val / epochs 5–15 mean | | | |
| fine-tuned DINOv2 (2 blocks, 2 seeds): `joint_wide` vs `slice_joint`, val / epochs 5–15 mean | | | |
| DINOv2 features + ComBat + subtype (best) | | | vs lookup: |

## 6. What would change the plan
- **Features + subtype significantly above the lookup** (vs-lookup CI above 0): the image adds information
  on all cohorts, not only I-SPY. That is the main result for the report, and a cheap, explainable model.
- **ComBat lifts Duke off chance** (Duke AUC or within-cohort AUC up): scanner effects were hiding the signal.
  Then apply ComBat to the fusion model's image embedding too.
- **Per-slice beats the volume twin in CV for both the probe and fine-tuning**: switch the default `data.norm`.
  If not, report it as tested, no gain.
- Still open for the team: F1/F2 fusion on cache v2.
