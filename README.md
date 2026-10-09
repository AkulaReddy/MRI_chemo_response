# pCR prediction from breast DCE-MRI (MNJ project #11, Stage 1)

ViT-B/16 slice encoder + attention fusion with routine clinical variables, trained on BreastDCEDL
(I-SPY1, I-SPY2, Duke). Plan: `IMPLEMENTATION_PLAN_FINAL.md`. Data facts: `data_availability.md`.
Decisions: `docs/DECISIONS.md`. Team plans: `docs/team/`.

## Layout
```
configs/   YAML configs (inherit from base.yaml)          kaggle/   notebooks + overnight download script
data/      metadata CSV + frozen split (committed)        scripts/  make_split.py, build_cache.py
src/       data/ (loading, RGB, clinical, bags), models/ (ViT, pooling, fusion), train.py, stats.py
tests/     pytest suite (runs on CPU with a tiny random ViT, ~2 min)
```

## Local setup (project laptop)
```
pip install -r requirements.txt
python -m pytest -q tests
python scripts/make_split.py --check
```

Notes for the laptop:
- A `torchaudio` built for a different torch (here 2.11 vs torch 2.6) breaks `import transformers`.
  `src/__init__.py` hides it automatically; nothing in this project uses audio.
- Code works with transformers 4.x (`vit.encoder.layer`) and 5.x (`vit.layers`).

## Verified so far (2026-10-08, day 0)
- Split frozen: labeled train / val / test = 1099 / 177 / 176 (pCR 322 / 53 / 53), md5 `856c3f59...`.
- Cache builder run on the Zenodo demo archive (7 patients, all 3 cohorts): axis 0 = slice, CSV boxes
  match the I-SPY masks exactly (IoU 1.0), tumor visible inside the crop. Real-archive differences
  (z-cropped I-SPY volumes, a missing Duke phase) are handled and logged in `cache_index.csv`.
- `facebook/vit-mae-base` loads into a plain `ViTModel` with output identical to `ViTMAEModel`
  (max difference 0.0; the unused MAE decoder weights are dropped).
- 30 tests pass (`python -m pytest -q tests`), including train -> export end-to-end on a synthetic cache.

## Pipeline
1. `kaggle/00_download_zenodo.py` -> Kaggle Datasets `mnj-raw-<cohort>` (md5-verified archives).
2. `kaggle/01_build_cache.ipynb` -> `mnj-cache-<cohort>`: per patient `<pid>.npz` with uint8 RGB slices
   (pre / early / late phase), 240x240 around the tumor, plus `cache_index.csv` (QC) and `summary_grid.png`.
3. `kaggle/10_train.ipynb` -> `python -m src.train --config <name> --seed <n>`; outputs in `runs/<run_id>/`,
   small files copied to `results/<person>/<run_id>/`.

Locally, with caches under `data/cache/` (gitignored):
```
python scripts/build_cache.py --cohort ispy1 --raw <folder with the tar.gz> --out data/cache/cache_ispy1
python -m src.train --config f2_attn --debug
```

## Rules
- The split is frozen in `data/splits/split_v1.csv` (Zenodo `test` column: 0 train, 2 val, 1 test).
  `patient_table()` refuses the test split; only the final evaluation (day 16) reads it.
- No data, caches or weights in git (they live in Kaggle Datasets). Secrets go in Kaggle secrets only.

