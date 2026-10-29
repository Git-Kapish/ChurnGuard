import numpy as np
import pandas as pd
import pytest

from churnguard.data.split import split_xy
from churnguard.features.engineering import ENGINEERED, FeatureEngineer
from churnguard.features.pipeline import BINARY, CATEGORICAL, NUMERIC, build_preprocessor


def test_engineered_values_match_hand_computation(make_customers) -> None:
    df = make_customers(
        # A: new customer. avg = 0 / max(0, 1) = 0, delta = 50 - 0 = 50.
        {"tenure": 0, "MonthlyCharges": 50.0, "TotalCharges": 0.0, "OnlineSecurity": "Yes"},
        # B: avg = 1200 / 24 = 50, delta = 80 - 50 = 30. Add-ons: TechSupport, TV, Movies = 3.
        {
            "tenure": 24,
            "MonthlyCharges": 80.0,
            "TotalCharges": 1200.0,
            "Contract": "One year",
            "PaymentMethod": "Bank transfer (automatic)",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
        },
        # C: avg = 4900 / 49 = 100, delta = 0. All six add-ons = 6.
        {
            "tenure": 49,
            "MonthlyCharges": 100.0,
            "TotalCharges": 4900.0,
            "Contract": "Two year",
            "PaymentMethod": "Credit card (automatic)",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
        },
    )
    out = FeatureEngineer().transform(df)
    expected = pd.DataFrame(
        {
            "tenure_bucket": ["0-12", "13-24", "49+"],
            "n_services": [1, 3, 6],
            "avg_monthly_spend": [0.0, 50.0, 100.0],
            "charge_delta": [50.0, 30.0, 0.0],
            "is_month_to_month": [1, 0, 0],
            "has_support_bundle": [1, 1, 1],
            "auto_pay": [0, 1, 1],
        }
    )
    pd.testing.assert_frame_equal(out[ENGINEERED].reset_index(drop=True), expected)


@pytest.mark.parametrize(
    ("tenure", "bucket"),
    [
        (0, "0-12"),
        (12, "0-12"),
        (13, "13-24"),
        (24, "13-24"),
        (25, "25-48"),
        (48, "25-48"),
        (49, "49+"),
        (72, "49+"),
    ],
)
def test_tenure_bucket_boundaries(make_customers, tenure: int, bucket: str) -> None:
    out = FeatureEngineer().transform(make_customers({"tenure": tenure}))
    assert out.loc[0, "tenure_bucket"] == bucket


def test_support_bundle_is_security_or_techsupport(make_customers) -> None:
    df = make_customers(
        {"OnlineSecurity": "Yes"},
        {"TechSupport": "Yes"},
        {"OnlineBackup": "Yes", "StreamingTV": "Yes"},  # add-ons but no support bundle
    )
    assert FeatureEngineer().transform(df)["has_support_bundle"].tolist() == [1, 1, 0]


def test_engineer_does_not_mutate_input(make_customers) -> None:
    df = make_customers({})
    before = df.copy()
    FeatureEngineer().transform(df)
    pd.testing.assert_frame_equal(df, before)


def test_engineer_reports_missing_columns(make_customers) -> None:
    with pytest.raises(ValueError, match="Contract"):
        FeatureEngineer().transform(make_customers({}).drop(columns=["Contract"]))


def test_pipeline_output_shape_names_and_no_nulls(random_customers) -> None:
    X, _ = split_xy(random_customers())
    pre = build_preprocessor().fit(X)
    out = pre.transform(X)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(X) and not out.isna().any().any()
    assert set(NUMERIC + BINARY) <= set(out.columns)
    assert "customerID" not in out.columns and "Churn" not in out.columns
    # one-hot columns exist, raw categorical text columns do not
    assert any(c.startswith("PaymentMethod_") for c in out.columns)
    assert not set(CATEGORICAL) & {c for c in out.columns if out[c].dtype == object}
    assert set(out.select_dtypes(exclude="number").columns) == set()


def test_numeric_columns_are_standardized_on_fit_data(random_customers) -> None:
    X, _ = split_xy(random_customers(n=500))
    out = build_preprocessor().fit_transform(X)
    assert np.allclose(out[NUMERIC].mean(), 0, atol=1e-9)
    assert np.allclose(out[NUMERIC].std(ddof=0), 1, atol=1e-9)


def test_missing_total_charges_is_imputed(random_customers) -> None:
    X, _ = split_xy(random_customers())
    pre = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row["TotalCharges"] = np.nan
    assert not pre.transform(row).isna().any().any()


@pytest.mark.filterwarnings("ignore:Found unknown categories")
def test_unseen_category_is_ignored(random_customers) -> None:
    X, _ = split_xy(random_customers())
    pre = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row["PaymentMethod"] = "Crypto"
    out = pre.transform(row)
    assert out.filter(like="PaymentMethod_").sum(axis=1).iloc[0] == 0
