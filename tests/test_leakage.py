"""Fit on train only; nothing about the test rows may influence what a row transforms to."""

import pandas as pd
import pytest
from sklearn.base import clone

from churnguard.config import PROCESSED_DIR
from churnguard.data.split import load_split, split_xy
from churnguard.features.pipeline import NUMERIC, build_preprocessor


def _learned_state(pre) -> dict:
    cols = pre.named_steps["columns"]
    num = cols.named_transformers_["num"]
    return {
        "median": num["impute"].statistics_.copy(),
        "mean": num["scale"].mean_.copy(),
        "scale": num["scale"].scale_.copy(),
        "categories": [c.copy() for c in cols.named_transformers_["cat"].categories_],
    }


@pytest.mark.filterwarnings("ignore:Found unknown categories")
def test_row_transform_is_independent_of_test_content(random_customers) -> None:
    X_train, _ = split_xy(random_customers(n=400, seed=1))
    X_test_a, _ = split_xy(random_customers(n=200, seed=2))
    X_test_b = X_test_a.copy()  # same row 0, wildly different neighbours
    X_test_b.loc[X_test_b.index[1:], ["tenure", "MonthlyCharges", "TotalCharges"]] = 1e6
    X_test_b.loc[X_test_b.index[1:], "PaymentMethod"] = "Brand new method"

    pre = build_preprocessor().fit(X_train)
    out_a = pre.transform(X_test_a).iloc[[0]]
    out_b = pre.transform(X_test_b).iloc[[0]]
    alone = pre.transform(X_test_a.iloc[[0]])

    pd.testing.assert_frame_equal(out_a, out_b)
    pd.testing.assert_frame_equal(out_a, alone)


def test_transform_does_not_change_learned_state(random_customers) -> None:
    X_train, _ = split_xy(random_customers(n=400, seed=1))
    X_test, _ = split_xy(random_customers(n=200, seed=2))
    X_test["MonthlyCharges"] = 9999.0
    pre = build_preprocessor().fit(X_train)
    before = _learned_state(pre)
    pre.transform(X_test)
    after = _learned_state(pre)
    for key in ("median", "mean", "scale"):
        assert (before[key] == after[key]).all()
    assert all(
        (b == a).all() for b, a in zip(before["categories"], after["categories"], strict=True)
    )


def test_learned_statistics_come_from_train_only(random_customers) -> None:
    X_train, _ = split_xy(random_customers(n=400, seed=1))
    X_test, _ = split_xy(random_customers(n=200, seed=2))
    X_test["tenure"] = 500  # would shift the mean if it leaked

    pre = build_preprocessor().fit(X_train)
    engineered_train = pre.named_steps["engineer"].transform(X_train)
    assert pre.named_steps["columns"].named_transformers_["num"]["scale"].mean_ == pytest.approx(
        engineered_train[NUMERIC].mean().to_numpy()
    )

    # Guard against a vacuous test: fitting on train + test must give different output.
    leaky = clone(pre).fit(pd.concat([X_train, X_test]))
    assert not leaky.transform(X_train.iloc[[0]]).equals(pre.transform(X_train.iloc[[0]]))


@pytest.mark.skipif(not (PROCESSED_DIR / "train.parquet").exists(), reason="run `make data` first")
def test_real_splits_fit_on_train_only() -> None:
    X_train, _ = split_xy(load_split("train"))
    X_val, _ = split_xy(load_split("val"))
    pre = build_preprocessor().fit(X_train)
    state = _learned_state(pre)
    pre.transform(X_val)
    assert (state["mean"] == _learned_state(pre)["mean"]).all()
    # train rows are standardized exactly; validation rows only approximately
    assert abs(pre.transform(X_train)[NUMERIC].mean().abs().max()) < 1e-9
    assert pre.transform(X_val)[NUMERIC].mean().abs().max() > 1e-9
