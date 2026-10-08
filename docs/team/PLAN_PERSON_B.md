# Person B: Image-only track, CNN baseline, localization

**Machine:** your Kaggle account (about 30 GPU-h/week); Synapse account by day 5.
**You own:** **B3** (ViT image-only), B4 exports, **B5** (Nishizawa 3D CNN), **leave-one-cohort-out**, **Module 1 inference** and the **deployment test**; the I-SPY2 data; report sections on the image-only models, the localization and robustness.
**You do not:** edit code (send errors to A); touch the test set.
**Read first:** `TEAM_OVERVIEW.md`; plan Sections 3–5 of `../../IMPLEMENTATION_PLAN_FINAL.md`.

**Running code:** use `kaggle/<nn>_*.ipynb` and set the parameter cell (`GIT_TAG`, `CONFIG`, `SEED`). For GPU work, choose T4×2 or P100 with Internet on, and attach the cache Datasets. After a run, save the weights as `mnj-w-<run_id>`, commit `results/B/<run_id>/`, and post the val AUC and GPU-h.

---

## Day 0 (tonight)
- Phone-verify Kaggle; accept the GitHub invite; add the `GITHUB_TOKEN` secret.
- Run `kaggle/00_download_zenodo.py` with `ME="B"` (I-SPY2, 13.6 GB): new notebook, Accelerator None, Internet on, **Save & Run All (Commit)**.

## Day 1
- Morning: check that the log says **MD5 OK** → create Dataset **`mnj-raw-ispy2`** from the output and share it (H1).
- After A tags `d1`: run `01_build_cache` with `COHORT=ispy2` (CPU) → **`mnj-cache-ispy2`** (H3). View 20 random patients in the summary grid.
- Afternoon: **B3 debug run** (`b3_vit.yaml`, 2 epochs). Post the slices/s and minutes/epoch.
- **Fallback if the download failed:** rerun it (it resumes), and meanwhile run the B3 debug on I-SPY1 only (`DATASETS=ispy1`).

## Days 2–6: B3
- B3 seed 0 (full run, early stopping). **Check:** val AUC between about 0.65 and 0.78. If it is below 0.6, stop and send A the log plus a few input grids.
- B3 seeds 1, 2; `b3_vit_supervised.yaml` (backbone ablation), seed 0.
- Day 5: `10_train` with `MODE=export` on the best B3: train/val slice predictions + patient embeddings. Save as `mnj-w-<run_id>` → **H5** (A for B4; C for F2-from-B3).
- Day 5: create the Synapse account; add the `SYNAPSE_TOKEN` secret.

## Days 7–10: B5, M1, LOCO (code from A's tag `d6`)
- Day 7: `01_build_cache` with `MODE=vol3d` for all three cohorts (CPU) → `mnj-vol3d-all`.
- Days 7–9: **B5** seeds 0, 1, 2 (`kaggle/20_b5.ipynb`, `b5_nishizawa.yaml`). If one run takes more than about 6 h, tell A to switch to ResNet-18.
- Day 7 in parallel: `kaggle/30_m1.ipynb` step 1: run `map_ids` → **M1 evaluation list** = MAMA-MIA *test* ∩ our *val ∪ test*. Post the counts per cohort to A. If fewer than about 50, it's reported as limited (D-011). Step 2: `download_mamamia` for the list only → `mnj-mamamia-m1`.
- Days 8–10: M1 inference + metrics (Dice, bbox IoU, centroid mm, volume ratio) → `mnj-m1-preds` + `results/B/m1/report.md` (H9, due day 11).
- Days 9–10: **LOCO** ×3 (B3 config), 1 seed each.

## Days 11–15
- After H7 (frozen F2, day 10) and H8 (tag `d9`): build the M1-crop cache for the M1-list patients. Run the **deployment dry run on val patients**: F2 on M1 crops vs ground-truth crops. Report the AUC difference and how many crops miss the tumor. The test version runs in A's final evaluation.
- Day 15: check your rows in `results/final_manifest.csv` (B3 ×3, B3-supervised, B5 ×3, LOCO, M1 preds) → H10.

## Days 18–34
- Report sections: **Image-only models (B3, B5)**, **Localization and deployment test**, **Robustness (LOCO, per cohort)**. Numbers only from `RESULTS.md` / `RESULTS_val.md`.
- Slides (3–4) and the dry run.

## GPU estimate
About 20 h in week 1 (B3 ×4 runs + exports), about 20–25 h in week 2 (B5 ×3, M1, LOCO ×3), a few hours in week 3.
