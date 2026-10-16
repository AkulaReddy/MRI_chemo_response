import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src  # noqa: E402

src.hide_broken_torchaudio()   # test_fusion_shapes imports transformers directly


@pytest.fixture(scope="session")
def synthetic_cache(tmp_path_factory):
    """A fake I-SPY1 + Duke cache for real train/val patient IDs: random slices with a bright
    'tumor' whose intensity depends on pCR, so a model can learn something."""
    from src.data.metadata import load_split
    split = load_split()
    root = tmp_path_factory.mktemp("cache")
    dirs = {}
    rng = np.random.default_rng(0)
    for cohort in ("ispy1", "duke"):
        d = root / f"cache_{cohort}"
        d.mkdir()
        rows = []
        for sp, n in (("train", 12), ("val", 8)):
            pids = split[(split.cohort == cohort) & (split.split == sp) & split.labeled].pid.head(n)
            for pid in pids:
                s = int(rng.integers(4, 9))
                rgb = rng.integers(0, 60, (s, 240, 240, 3), dtype=np.uint8)
                rgb[:, 100:140, 100:140] += np.uint8(80 if split.loc[pid, "pCR"] == 1 else 30)
                np.savez_compressed(d / f"{pid}.npz", rgb=rgb, t0=1, t1=s - 1,
                                    bbox=np.array([100, 140, 100, 140]))
                rows.append(f"{pid},{cohort},ok")
        (d / "cache_index.csv").write_text("pid,cohort,status\n" + "\n".join(rows) + "\n")
        (d / "cache_meta.json").write_text(json.dumps({"cohort": cohort}))
        dirs[cohort] = d
    return dirs
