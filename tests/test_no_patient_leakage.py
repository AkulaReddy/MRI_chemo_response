import pytest

from src.data.datasets import patient_table
from src.data.metadata import load_metadata, load_split


def test_each_patient_in_one_split():
    s = load_split()
    assert s.pid.is_unique
    assert set(s.split) == {"train", "val", "test"}


def test_split_matches_metadata_codes():
    s, m = load_split(), load_metadata()
    codes = {0: "train", 2: "val", 1: "test"}
    assert (m.loc[s.pid, "test"].astype(int).map(codes).values == s.split.values).all()


def test_test_split_refused_by_default():
    with pytest.raises(PermissionError):
        patient_table(["train", "test"])
    assert len(patient_table(["test"], allow_test=True)) == 176


def test_train_and_val_tables_disjoint():
    tr, va = patient_table("train"), patient_table("val")
    assert not set(tr.pid) & set(va.pid)
    assert (len(tr), len(va)) == (1099, 177)
