import numpy as np

from src.data.clinical import ClinicalEncoder, race_column
from src.data.datasets import patient_table
from src.data.metadata import load_metadata


def test_fit_on_train_and_valid_codes():
    tr, va = patient_table("train"), patient_table("val")
    enc = ClinicalEncoder().fit(tr)
    for t in (tr, va):
        out = enc.transform(t)
        assert not np.isnan(out["cont"]).any()
        assert out["cat"].min() >= 0
        assert (out["cat"].max(0) < np.array(enc.cardinalities)).all()
    assert abs(enc.transform(tr)["cont"].mean(0)).max() < 1e-4      # standardised on train
    assert enc.onehot(va).shape[1] == sum(enc.cardinalities) + 2 * len(enc.cont)


def test_missing_values_get_missing_category_and_flag():
    m = load_metadata()
    enc = ClinicalEncoder().fit(m[m.split == "train"])
    out = enc.transform(m)
    j = enc.cat.index("menopause")
    assert (out["cat"][m.menopause.isna().values, j] == 2).all()
    a = enc.cont.index("age")
    assert m.age.isna().sum() == 3
    assert (out["cont_missing"][m.age.isna().values, a] == 1).all()


def test_race_mapping():
    r = race_column(load_metadata())
    assert r.isna().sum() == 3
    assert set(r.dropna()) == {"white", "black", "other"}


def test_roundtrip_and_no_features():
    tr = patient_table("train")
    enc = ClinicalEncoder().fit(tr)
    enc2 = ClinicalEncoder.from_dict(enc.to_dict())
    assert np.allclose(enc.transform(tr)["cont"], enc2.transform(tr)["cont"])
    empty = ClinicalEncoder([]).fit(tr).transform(tr)
    assert empty["cat"].shape == (len(tr), 0) and empty["cont"].shape == (len(tr), 0)


def test_bbox_volume_positive():
    m = load_metadata()
    assert (m.bbox_vol_mm3 > 0).all() and np.isfinite(m.log_bbox_vol).all()
