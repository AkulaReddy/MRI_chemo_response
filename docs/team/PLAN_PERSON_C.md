# Person C: Fusion track (the main model) and explainability

**Machine:** your Kaggle account (about 30 GPU-h/week).
**You own:** **F1** and **F2** (end-to-end ViT + clinical fusion, the project's main contribution), their ablations, the frozen F2 with 3 seeds, the F2-from-B3 ablation; **explainability**; the second-machine test of `infer.py`; the Duke data; report sections on fusion and explainability.
**You do not:** edit code (send errors to A); touch the test set.
**Read first:** `TEAM_OVERVIEW.md`; plan Section 4 (M3 fusion) and Section 5 of `../../IMPLEMENTATION_PLAN_FINAL.md`; `../DECISIONS.md` D-008, D-013.

**You do not wait for B.** F1 and F2 start from the same `facebook/vit-mae-base` weights as B3, so your track runs in parallel from day 1.

**Running code:** use `kaggle/10_train.ipynb` and set the parameter cell (`GIT_TAG`, `CONFIG`, `SEED`). Choose T4×2 or P100 with Internet on, and attach the three cache Datasets. After a run, save the weights as `mnj-w-<run_id>`, commit `results/C/<run_id>/`, and post the val AUC, AUPRC, ECE and GPU-h.

---

## Day 0 (tonight)
- Phone-verify Kaggle; accept the GitHub invite; add the `GITHUB_TOKEN` secret.
- Run `kaggle/00_download_zenodo.py` with `ME="C"` (Duke, 8.6 GB): new notebook, Accelerator None, Internet on, **Save & Run All (Commit)**.

## Day 1
- Morning: check that the log says **MD5 OK** → create Dataset **`mnj-raw-duke`** from the output and share it (H1).
- After A tags `d1`: run `01_build_cache` with `COHORT=duke` (CPU) → **`mnj-cache-duke`** (H3). View 20 random patients in the summary grid. Duke crops come from bounding boxes; check that the tumor is inside.
- Afternoon: **F2 debug run** (`f2_attn.yaml`, 2 epochs). Confirm that the loss falls, the val patient AUC prints and the attention weights are logged.
- **Fallback if the download failed:** rerun it (it resumes), and debug F2 on I-SPY1 only.

## Days 2–7: fusion ablations (1 seed each, val only)

| Run | Config | Question |
|---|---|---|
| F1 | `f1_concat.yaml` | does attention beat simple concatenation? |
| F2 | `f2_attn.yaml` | the default main model |
| F2-mean vs F2-MIL | `f2_meanpool.yaml`, `f2_mil.yaml` | slice aggregation |
| F2-no-bbox | `f2_no_bbox.yaml` | does the tumor-size token matter? |
| F2-no-aux | `f2_no_aux.yaml` | do the auxiliary HR/HER2 heads help? |

Post one table: val AUC with CI, AUPRC, ECE and GPU-h. If GPU time is short, drop `f2_no_aux` first.

## Day 8: freeze F2
Choose the final F2 configuration together with A: highest val AUC, with ties broken by calibration, then simplicity. A commits `configs/f2_final.yaml`.

## Days 8–10
- F2-final seeds 0, 1, 2; F1 seeds 1–2 if there's time. → **H7** (day 10): weights + config hash to A and B.
- After H5 (B3 exports, day 5): `f2_from_b3.yaml` (F2 initialised from the best B3), 1 seed, as an extra ablation.

## Days 11–15: explainability and analysis (after tag `d9`)
- `kaggle/40_explain.ipynb` (`explain.py`) on the frozen F2, **val patients**:
  - attention-weight tables aggregated over val (by column, as in Nishizawa), plus 4 example patients (pCR / non-pCR, correct / wrong);
  - attention-rollout heatmaps; count, over 30 random val patients, how many highlight the tumor rather than the crop border;
  - clinical feature-shuffle importance.
- Per-cohort and per-subtype val breakdown of F2 vs B3 vs B1 (using A's `stats.py`).
- Day 15: check your rows in `results/final_manifest.csv` → H10.

## Days 21–24: second-machine test
After A tags `d21`: on your own laptop (or a fresh Kaggle session), follow **only** `README.md` to install and run `infer.py` on 3 sample patients. Report every failed or unclear step to A.

## Days 18–34
- Report sections: **Fusion model and ablations**, **Explainability**. Numbers only from `RESULTS.md` / `RESULTS_val.md`. Draw the architecture figure.
- Slides (3–4) and the dry run.

## GPU estimate
About 20–25 h in week 1 (debug + 6 ablation runs), about 15–20 h in week 2 (F2 ×3, F1 ×2, F2-from-B3), a few hours in week 3 (explainability).
