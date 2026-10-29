import numpy as np
import pandas as pd
import pytest

from churnguard.config import RAW_PATH
from churnguard.data.clean import clean
from churnguard.data.load import load_raw
from churnguard.data.split import load_split, save_splits, split_data
from churnguard.data.validate import (
    RAW_CATEGORIES,
    ValidationError,
    validate_clean,
    validate_raw,
)


def make_raw(n: int = 1000, churn_rate: float = 0.265, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame with the raw schema, incl. blank TotalCharges for tenure == 0 rows."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.choice(sorted(vals), n) for col, vals in RAW_CATEGORIES.items()})
    df["customerID"] = [f"C{i}" for i in range(n)]
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(1, 73, n)
    df["MonthlyCharges"] = rng.uniform(18, 119, n).round(2)
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2).astype(str)
    df["Churn"] = np.where(np.arange(n) < round(n * churn_rate), "Yes", "No")
    df.loc[:2, ["tenure", "TotalCharges"]] = [0, " "]  # three new customers
    return df


def test_blank_total_charges_becomes_zero() -> None:
    raw = make_raw()
    out = clean(raw)
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])
    assert (out.loc[:2, "TotalCharges"] == 0).all()
    assert raw.loc[0, "TotalCharges"] == " "  # input untouched


def test_no_nulls_and_valid_after_cleaning() -> None:
    out = validate_clean(clean(validate_raw(make_raw())))
    assert not out.isna().any().any()
    assert set(out["Churn"].unique()) <= {0, 1}


def test_redundant_categories_normalized_to_no() -> None:
    out = clean(make_raw())
    for col in ["MultipleLines", "OnlineSecurity", "StreamingTV"]:
        assert set(out[col].unique()) == {"Yes", "No"}


def test_validate_raw_rejects_bad_category() -> None:
    raw = make_raw()
    raw.loc[5, "Contract"] = "Weekly"
    with pytest.raises(ValidationError, match="Contract"):
        validate_raw(raw)


def test_validate_raw_rejects_blank_total_charges_with_tenure() -> None:
    raw = make_raw()
    raw.loc[10, "TotalCharges"] = " "
    with pytest.raises(ValidationError, match="blank TotalCharges"):
        validate_raw(raw)


def test_validate_raw_rejects_out_of_range_tenure() -> None:
    raw = make_raw()
    raw.loc[10, "tenure"] = -1
    with pytest.raises(ValidationError, match="tenure"):
        validate_raw(raw)


def test_split_sizes_and_stratification() -> None:
    df = clean(make_raw(n=2000))
    splits = split_data(df, seed=42)
    sizes = {k: len(v) for k, v in splits.items()}
    assert sum(sizes.values()) == len(df)
    assert sizes == {"train": 1400, "val": 300, "test": 300}
    overall = df["Churn"].mean()
    for part in splits.values():
        assert abs(part["Churn"].mean() - overall) < 0.01
    ids = [set(p["customerID"]) for p in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_split_is_deterministic_and_roundtrips(tmp_path) -> None:
    df = clean(make_raw())
    a, b = split_data(df, seed=42), split_data(df, seed=42)
    assert a["train"]["customerID"].tolist() == b["train"]["customerID"].tolist()
    save_splits(a, tmp_path)
    assert load_split("val", tmp_path).shape == a["val"].shape


def test_split_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError):
        split_data(clean(make_raw()), train=0.5, val=0.2, test=0.2)


@pytest.mark.skipif(not RAW_PATH.exists(), reason="run `make data` first")
def test_real_data_cleans_and_splits() -> None:
    df = validate_clean(clean(validate_raw(load_raw())))
    assert len(df) == 7043 and not df.isna().any().any()
    splits = split_data(df, seed=42)
    for part in splits.values():
        assert abs(part["Churn"].mean() - df["Churn"].mean()) < 0.01
