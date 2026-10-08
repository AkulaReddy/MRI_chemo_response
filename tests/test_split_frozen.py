import json

from src.data.metadata import METADATA_CSV, METADATA_MD5, SPLIT_CSV, load_split, md5_file

FROZEN_SPLIT_MD5 = "856c3f59070e8a5cb5c9f31dad1ccc62"


def test_metadata_is_canonical():
    assert md5_file(METADATA_CSV) == METADATA_MD5


def test_split_file_unchanged():
    assert md5_file(SPLIT_CSV) == FROZEN_SPLIT_MD5
    meta = json.loads(SPLIT_CSV.with_suffix(".meta.json").read_text())
    assert meta["split_md5"] == FROZEN_SPLIT_MD5


def test_split_counts():
    lab = load_split()
    lab = lab[lab.labeled]
    assert lab.split.value_counts().to_dict() == {"train": 1099, "val": 177, "test": 176}
    assert lab.groupby("split").pCR.sum().astype(int).to_dict() == {"train": 322, "val": 53, "test": 53}
    assert lab[lab.split == "test"].cohort.value_counts().to_dict() == {"ispy2": 99, "duke": 42, "ispy1": 35}
