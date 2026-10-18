# Branch `vaib` notebooks: improving the ViT image-only model

New notebooks only; the team's `kaggle/*.ipynb` are unchanged. Background and findings: `docs/vaib/EXPLAINED.md`.
Regenerate after editing: `python kaggle/vaib/make_notebooks.py`.

Notebooks 01–04 clone this repo at `GIT_TAG = "vaib"` with the `GITHUB_TOKEN` Kaggle secret (as the team's do).
Set `PERSON` ("A"/"B"/"C"); results are written to `results/<PERSON>/`, and pushed to the `vaib` branch only if
`PUSH_RESULTS = True`.

| # | notebook | accelerator | inputs to attach | time | output |
|---|---|---|---|---|---|
| 00 | `00_download_check` | None, Internet ON, **no secret** | – | ~5 min (demo) | answers "is it downloadable"; one patient from raw NIfTI to RGB slice |
| 01 | `01_input_diagnostics` | None | `mnj-cache-*` (v1) and/or `mnj-cache2-*` | ~15 min | saturation per normalisation, tumour size in the input, gallery, hand-crafted kinetic AUC |
| 02 | `02_build_cache_v2` (once per cohort) | None | `mnj-raw-<cohort>` | 20–60 min | save the output as Dataset **`mnj-cache2-<cohort>`** (~2 / 10 / 4 GB) |
| 03 | `03_probe_sweep` | GPU T4 ×2 | `mnj-cache2-*` (or v1) | ~30 min (`LIMIT=60` first) | `probe_results.csv`: norm × crop × backbone ranked by CV AUC |
| 04 | `04_finetune_vit` | GPU T4 / P100 | `mnj-cache2-*` (or v1) | ~30 min per seed (`DEBUG=True` first) | per-seed runs + 3-seed ensemble val AUC |

Order: 00 → 01 (works on the existing v1 caches today) → 02 ×3 → 03 → put the top row of 03 into 04
(`NORM`, `CROP`, `BACKBONE`) → 04. Optionally set `RUN_CONTROL = True` in 04: it runs the old B3 recipe through the
new code, which should give ~0.53.

Without v2 caches, 03 and 04 still run on the v1 caches. But v1 has already clipped the tumour, so `joint_wide`
equals `joint_pct`, and the other normalisations work from clipped values.

Overrides in `EXTRA` use the config keys in `configs/vo_base.yaml`, e.g. `"model.train_blocks=6"`. Write numbers
with a decimal point (`3.0e-5`); YAML reads `3e-5` as text, although finetune.py converts it.

Local checks: `python -m pytest -q tests/test_vitonly.py`. All five notebooks were executed end to end locally:
00 on the real demo archive; 01, 03 and 04 on synthetic v2 caches with tiny random ViTs.
