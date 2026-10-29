from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

BASE_CUSTOMER = {
    "customerID": "X",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 10,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 50.0,
    "TotalCharges": 500.0,
    "Churn": 0,
}


@pytest.fixture
def make_customers() -> Callable[..., pd.DataFrame]:
    """Factory: make_customers({...overrides}, {...}) -> cleaned-schema frame, one row each."""

    def _make(*overrides: dict) -> pd.DataFrame:
        return pd.DataFrame([{**BASE_CUSTOMER, **o} for o in overrides])

    return _make


@pytest.fixture
def random_customers() -> Callable[..., pd.DataFrame]:
    """Factory for n random but valid cleaned-schema rows (seeded)."""

    def _make(n: int = 300, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        df = pd.DataFrame([BASE_CUSTOMER] * n)
        yes_no = ["Yes", "No"]
        for col in ["Partner", "Dependents", "OnlineSecurity", "TechSupport", "StreamingTV"]:
            df[col] = rng.choice(yes_no, n)
        df["Contract"] = rng.choice(["Month-to-month", "One year", "Two year"], n)
        df["PaymentMethod"] = rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)"], n
        )
        df["tenure"] = rng.integers(0, 73, n)
        df["MonthlyCharges"] = rng.uniform(20, 115, n).round(2)
        df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2)
        df["Churn"] = rng.integers(0, 2, n)
        df["customerID"] = [f"C{i}" for i in range(n)]
        return df

    return _make
