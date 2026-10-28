"""Radiomics features for every patient of one or more builder v2/v3 caches -> one CSV (CPU).

    python scripts/extract_radiomics.py --cache-root /kaggle/input --out features.csv [--workers 4] [--limit 20]

One row per cached patient (`status == ok` in cache_index.csv), all splits (train/val/test): the test rows
are only *computed* here; models never see them until the final evaluation (src.baselines.radiomics reads
train and val only). Columns: pid, cohort, cache_version, mask_source, n_voxels and src.radiomics.FEATURES.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.radiomics import patient_features  # noqa: E402


def find_caches(roots, version):
    found = {}
    for r in roots:
        for depth in range(6):
            for f in glob.glob(os.path.join(r, *["*"] * depth, "cache_meta.json")):
                m = json.load(open(f))
                if m.get("builder_version", 1) == version:
                    found[m["cohort"]] = os.path.dirname(f)
    return found


def one(args):
    pid, cohort, path = args
    row = {"pid": pid, "cohort": cohort}
    try:
        with np.load(path) as d:
            row.update(patient_features(d))
    except Exception as e:     # recorded, not fatal
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-root", nargs="+", default=[str(ROOT / "data" / "cache")])
    ap.add_argument("--cache-version", type=int, default=2)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--limit", type=int, default=0, help="first N patients per cohort (debug)")
    a = ap.parse_args(argv)

    caches = find_caches(a.cache_root, a.cache_version)
    if not caches:
        raise SystemExit(f"no v{a.cache_version} caches under {a.cache_root}")
    jobs = []
    for c, d in sorted(caches.items()):
        idx = pd.read_csv(Path(d) / "cache_index.csv")
        pids = idx.loc[idx.status == "ok", "pid"].tolist()
        pids = pids[:a.limit] if a.limit else pids
        jobs += [(p, c, Path(d) / f"{p}.npz") for p in pids]
        print(f"{c}: {len(pids)} patients from {d}", flush=True)
    t, rows = time.time(), []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=8), 1):
            rows.append(r)
            if i % 200 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} done, {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.insert(2, "cache_version", a.cache_version)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    err = df["error"].notna().sum() if "error" in df else 0
    print(f"wrote {a.out}: {len(df)} patients, {err} errors; mask sources "
          f"{df.groupby('cohort').mask_source.value_counts().to_dict() if 'mask_source' in df else {}}")


if __name__ == "__main__":
    main()
