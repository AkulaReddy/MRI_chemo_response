# Person A: Code owner, data and evaluation lead

**Machine:** the project laptop (Quadro P520 4 GB). Code, tests, tabular models, stats and the demo run here. Your Kaggle account handles the I-SPY1 download, the cache build and the final evaluation.
**You own:** all code; the split and the data pipeline; B0/B1/B2/B4-head; the **single final test evaluation**; `RESULTS.md`; `infer.py`; final report editing.
**Read first:** `TEAM_OVERVIEW.md` (handoffs H0–H12).

Your rule: never block B or C. A code handoff beats your own analysis.

---

## Day 0 (today)
1. **Now:** send `kaggle/00_download_zenodo.py` to B and C (H0). Each sets `ME` and commits the notebook tonight.
2. Run it yourself with `ME="A"` (I-SPY1 + models + CSV).
3. Write the core code with Claude: `git init` + private GitHub repo (add B and C); `make_split.py` + frozen `split_v1.csv`; `metadata.py`, `nifti_io.py`, `rgb_slices.py`, `clinical.py`, `datasets.py`, `transforms.py`; `build_cache.py` (reads from the `mnj-raw-*` dataset); `vit_encoder.py`, `pooling.py`, `fusion.py` (F1, F2, aux heads); `train.py` (config-driven, val patient AUC early stopping, run-ID outputs); configs `base`, `b3_vit`, `b3_vit_supervised`, `f1_concat`, `f2_attn`, `f2_meanpool`, `f2_mil`, `f2_no_bbox`, `f2_no_aux`, `f2_from_b3`; Kaggle notebooks `kaggle/01_build_cache.ipynb` and `kaggle/10_train.ipynb` from `kaggle/_template.ipynb` (clones a tag with `GITHUB_TOKEN`); tests (`test_rgb_uint8`, `test_split_frozen`, `test_no_patient_leakage`, `test_fusion_shapes`, `test_clinical_encoding`).
4. Smoke-test locally on synthetic volumes (no real data yet). Push, but don't tag yet.

## Day 1
- 08:00–10:00: run `01_build_cache` on `mnj-raw-ispy1` with `LIMIT=10`. The loader was already checked on the Zenodo demo archive (axis 0 = slice, CSV boxes = masks), so check what differs in the real archive: the `z_source` counts (csv vs mask), `phase_fallback` count, `mask_csv_iou_yx` (expect 1.0), errors, and the grid (tumor visible, inside the box, no noise). Fix anything new, then run all 172.
- 11:00: tag **`d1`** → H2. Post "build caches now" to B and C.
- Build `mnj-cache-ispy1`. Help B and C with their first runs (expect small fixes, tagged `d1.1`, …).

## Days 2–6
- Day 2: `stats.py` (bootstrap 2,000, paired DeLong, AUPRC, sensitivity/PPV at 90% specificity, Brier, ECE, reliability plot), `evaluate.py` (val; test requires `--final`). Tag `d2` → H4.
- Days 2–3: **B0, B1** (LR + RF; clinical ± log bbox volume) on train → val, locally. **B2** (I-SPY2 RF with MammaPrint + treatment arm, from TCIA `ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx`; with/without tumor volume). Commit to `results/A/`.
- Days 3–6: code for B's week 2. **B5** `nishizawa3d.py` + `build_cache --mode vol3d` + `b5_nishizawa.yaml`. **LOCO** configs (3), which train on two cohorts and evaluate on the third cohort's non-test patients. **M1**: `map_ids.py`, `download_mamamia.py` (weights `syn61906708`, split `syn60880777`, selected folders), `localization.py` (nnU-Net v2 wrapper: 1 mm resample, z-score, `_000X` naming, largest component → bbox/volume, back to native spacing), metrics. Notebooks `kaggle/20_b5.ipynb`, `kaggle/30_m1.ipynb`. Tag `d6` → H6.

## Days 7–10
- Day 7: **B4** on B's B3 exports (H5): slice-score summaries + embedding PCA → LR/RF with the B1 features; val results.
- Days 8–9: `build_cache --from-m1` (crops from M1 bboxes), `explain.py` (attention tables aggregated by column, attention rollout, clinical feature-shuffle) + `kaggle/40_explain.ipynb`. Tag `d9` → H8.
- Days 9–10: `scripts/final_eval.py`: reads `results/final_manifest.csv`, scores every model on test once, writes all tables (overall, per cohort, per subtype, calibration, high-specificity point with the val-chosen threshold) and DeLong H1 (F2 vs B1) and H2 (F2 vs B4).

## Days 11–17
- Days 11–14: merge val results into `docs/RESULTS_val.md`; check the manifest (H10 due day 15). Freeze the code: tag `final-eval`.
- **Day 16: run `final_eval.py` once** via `kaggle/50_final_eval.ipynb` (your Kaggle, GPU). Commit `docs/RESULTS.md` + figures (H11). Any rerun needs a logged bug reason.
- Day 17: evaluation write-up draft.

## Days 18–34
- Days 18–21: `infer.py` (NIfTI/DICOM folder + `clinical.json` → M1 → crop → F2 seed ensemble → probability, spread, heatmap, attention table; runs on the P520 or CPU) + README. Tag `d21` → H12.
- Days 22–24: fix whatever C reports; write the Stage 2 design note (plan Section 9).
- Days 25–34: report (intro, data, evaluation, discussion, MNJ handoff); merge and edit all sections; slides; buffer.

## GPU use
About 3–5 h total: the shipped-weights check (day 2, optional) and the final evaluation.
