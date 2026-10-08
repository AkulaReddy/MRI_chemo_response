# Team Overview: who does what, and when work is handed over

**Version 2 (2026-10-08):** two parallel training tracks from day 1; downloads overnight on day 0 (D-022).
Read this first, then your own plan: `PLAN_PERSON_A.md`, `PLAN_PERSON_B.md` or `PLAN_PERSON_C.md`.
Background: `../../IMPLEMENTATION_PLAN_FINAL.md` (what we build) and `../DECISIONS.md` (why; cited as D-xxx).

**Days are counted from Day 0 = Thu 2026-10-08 (today).** Day 34 is the end of week 5. Replace "Person A/B/C" with names.

---

## 1. Roles

| Person | Track | Trains / runs | Also |
|---|---|---|---|
| **A** (project laptop) | **Code owner**, data, evaluation | tabular baselines B0, B1, B2, B4-head; **the single final test evaluation** | writes *all* code; `infer.py`; edits the report |
| **B** | **Image-only track** | B3 ViT (3 seeds + supervised-backbone ablation); B4 exports; B5 Nishizawa 3D; leave-one-cohort-out; **Module 1 inference** + deployment test | Synapse account |
| **C** | **Fusion track** | F1 and F2 **from MAE weights** (no wait for B3); F2 ablations; frozen F2 ×3 seeds; F2-initialised-from-B3 ablation | explainability; tests `infer.py` on a second machine |

**Why this is parallel:** F2 does not need B3. Both start from `facebook/vit-mae-base`, so B and C train at the same time from day 1. The only real dependencies are:
- B4 needs B3 (day 5);
- the deployment test and explainability need the frozen F2 (day 10);
- the final evaluation needs every model (day 15).

## 2. Ground rules

1. **All code is written by A,** pushed to one private GitHub repo and released as tags (`d1`, `d2`, …). B and C run tagged code in Kaggle notebooks and change only the parameter cell. Errors go to A with the log and the command.
2. B and C commit only to `results/<person>/` and `docs/report/`. Weights go in Kaggle Datasets (`mnj-w-<run_id>`), never in git.
3. **Only A evaluates on the test set (`test==1`), once, on day 16** (D-006). `evaluate.py` refuses test without `--final` and logs every final run. All selection uses val (`test==2`).
4. Everyone uses `data/splits/split_v1.csv` (D-003) and the shared caches.
5. A run ID is `<model>_<confighash>_s<seed>`. Each run writes `results/<person>/<run_id>/{metrics.json, log.txt, val_predictions.csv}`.

## 3. Setup each person needs (day 0)

| Item | A | B | C |
|---|---|---|---|
| Kaggle account, **phone-verified** (needed for internet + GPU) | ✔ | ✔ | ✔ |
| GitHub account; repo access; GitHub token saved as Kaggle Secret `GITHUB_TOKEN` | owner | ✔ | ✔ |
| Synapse account (MAMA-MIA) + Kaggle Secret `SYNAPSE_TOKEN` | – | ✔ (by day 5) | – |

Each Kaggle account gets about 30 GPU-h/week and 12-h sessions. Downloads and cache builds use **no GPU** (Accelerator: None).

## 4. Shared Kaggle Datasets (names fixed)

| Dataset | Content | Owner | Ready |
|---|---|---|---|
| `mnj-raw-ispy1` | I-SPY1 archive + models archive + metadata CSV (md5-checked) | A | Day 1 morning |
| `mnj-raw-ispy2` | I-SPY2 archive | B | Day 1 morning |
| `mnj-raw-duke` | Duke archive | C | Day 1 morning |
| `mnj-cache-ispy1/-ispy2/-duke` | uint8 RGB tumor-slice caches + index CSV | A / B / C | Day 1 afternoon |
| `mnj-vol3d-all` | 3D first-post-contrast volumes for B5 | B | Day 6 |
| `mnj-mamamia-m1`, `mnj-m1-preds` | nnU-Net weights + evaluation-subset images; M1 predictions | B | Day 8 / Day 11 |
| `mnj-w-<run_id>` | weights per run | the trainer | as produced |

All are private and shared with the other two.

## 5. Handoffs

| ID | What | From → To | Due | Unblocks |
|---|---|---|---|---|
| H0 | `kaggle/00_download_zenodo.py` | A → B, C | **Day 0, now** | overnight downloads |
| H1 | `mnj-raw-*` datasets | A, B, C → all | Day 1 09:00 | cache builds |
| H2 | Core code, tag `d1`: repo, split, cache builder, datasets, clinical encoding, ViT trainer, F1/F2 fusion, configs, Kaggle train notebook. Written day 0, **finalised day 1 morning after A inspects a real I-SPY1 volume** | A → B, C | Day 1 11:00 | cache builds + all training |
| H3 | `mnj-cache-*` | A, B, C → all | Day 1 15:00 | B3, F1/F2 |
| H4 | `stats.py`, `evaluate.py` (val), tag `d2` | A → B, C | Day 2 | consistent metrics |
| H5 | B3 best-seed weights + exports (slice predictions, embeddings) | B → A, C | **Day 5** | B4 (A); F2-from-B3 ablation (C) |
| H6 | B5 code + `vol3d` cache mode, LOCO configs, M1 code (`map_ids`, `download_mamamia`, nnU-Net wrapper, metrics), tag `d6` | A → B | **Day 6** | B's week-2 work |
| H7 | **Frozen F2 config + 3 seed weights**; F1 weights | C → A, B | **Day 10** | deployment test, explainability, final evaluation |
| H8 | Deployment crop builder + `explain.py`, tag `d9` | A → B, C | Day 9 | B's deployment test; C's explainability |
| H9 | M1 predictions + localization report | B → A | Day 11 | deployment rows in the final evaluation |
| H10 | `results/final_manifest.csv` complete (every val-selected checkpoint) | all → A | **Day 15** | final evaluation |
| H11 | Final test results `docs/RESULTS.md` | A → all | Day 16–17 | report |
| H12 | `infer.py` + README, tag `d21` | A → C | Day 21 | second-machine test |

**If a handoff is late, use the fallback task in your plan.** Don't wait idle.

## 6. Calendar at a glance

| Days | A | B (image-only) | C (fusion) |
|---|---|---|---|
| 0 | send H0; write the core code with Claude | start the I-SPY2 download tonight | start the Duke download tonight |
| 1 | inspect a volume, fix, tag `d1`; build the I-SPY1 cache | build the I-SPY2 cache; B3 debug run | build the Duke cache; F2 debug run |
| 2–6 | stats, B0/B1/B2; M1 + B5 + LOCO code | B3 ×3, supervised-backbone B3, exports | F1, F2 ablations (aggregation, bbox token, aux heads) |
| 7–10 | B4; deployment crops; `explain.py` | B5 ×3; M1 inference (Synapse); LOCO ×3 | freeze F2 (day 8), F2 ×3 seeds, F2-from-B3 ablation |
| 11–15 | `final_eval.py`; collect the manifest | deployment test on val; finish LOCO | explainability (val); per-cohort/subtype analysis on val |
| 16–17 | **final test evaluation (once)** → RESULTS.md | | |
| 18–24 | `infer.py`; evaluation write-up | report: localization, B3/B5, robustness | `infer.py` second-machine test; report: fusion, explainability |
| 25–34 | edit and merge the report; Stage 2 note; slides | slides; fixes | slides; fixes |

This leaves about a week of buffer (days 25–34) for reruns and writing.

## 7. Weekly rhythm
- **Daily, 10 minutes (async in chat):** runs started/finished, val AUC, GPU-h used, blockers.
- **Days 1, 7, 14, 21, 28:** a 30-minute sync to check the handoffs due that week; A tags a release.
