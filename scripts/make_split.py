"""Freeze the Zenodo split as data/splits/split_v1.csv (D-003). Run once; the output is committed.

    python scripts/make_split.py            # writes split_v1.csv + split_v1.meta.json
    python scripts/make_split.py --check    # verifies the committed files, writes nothing
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.metadata import METADATA_CSV, METADATA_MD5, SPLIT_CSV, load_metadata, md5_file  # noqa: E402

EXPECTED_LABELED = {"train": 1099, "val": 177, "test": 176}


def build():
    df = load_metadata(METADATA_CSV)
    out = df[["pid", "cohort", "split", "test", "labeled", "pCR"]].rename(columns={"test": "test_code"})
    out["test_code"] = out["test_code"].astype(int)
    out = out.sort_values("pid").reset_index(drop=True)
    counts = out[out.labeled].split.value_counts().to_dict()
    assert counts == EXPECTED_LABELED, f"labeled counts {counts} != {EXPECTED_LABELED}"
    assert out.pid.is_unique
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = build()
    meta_path = SPLIT_CSV.with_suffix(".meta.json")
    if a.check:
        meta = json.loads(meta_path.read_text())
        assert md5_file(SPLIT_CSV) == meta["split_md5"], "split_v1.csv changed since it was frozen"
        print("split_v1.csv OK", meta["labeled_counts"])
        return
    if SPLIT_CSV.exists():
        sys.exit(f"{SPLIT_CSV} exists; the split is frozen. Use --check.")
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SPLIT_CSV, index=False, lineterminator="\n")
    lab = out[out.labeled]
    meta = {
        "source": METADATA_CSV.name, "source_md5": METADATA_MD5,
        "split_md5": md5_file(SPLIT_CSV),
        "codes": {"0": "train", "2": "val", "1": "test"},
        "labeled_counts": lab.split.value_counts().to_dict(),
        "labeled_pcr": lab.groupby("split").pCR.sum().astype(int).to_dict(),
        "labeled_by_cohort": lab.groupby(["split", "cohort"]).size().unstack(fill_value=0).to_dict("index"),
        "all_counts": out.split.value_counts().to_dict(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
