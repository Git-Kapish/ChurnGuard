"""Row-wise engineered features from PRD FR2.

Stateless on purpose: nothing is learned from data, so it cannot leak across splits.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from churnguard.data.validate import SERVICE_COLS

TENURE_BINS = [-1, 12, 24, 48, float("inf")]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49+"]
REQUIRED = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaymentMethod",
    *SERVICE_COLS,
]
ENGINEERED = [
    "tenure_bucket",
    "n_services",
    "avg_monthly_spend",
    "charge_delta",
    "is_month_to_month",
    "has_support_bundle",
    "auto_pay",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Append the FR2 engineered columns to a cleaned customer frame.

    n_services counts the six add-on services (OnlineSecurity, OnlineBackup,
    DeviceProtection, TechSupport, StreamingTV, StreamingMovies) that are "Yes".
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "FeatureEngineer":
        """No-op; present for sklearn Pipeline compatibility."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of X with the engineered columns added."""
        missing = [c for c in REQUIRED if c not in X.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        out = X.copy()
        # Plain numpy ops: this runs once per model in the calibrated ensemble on every request.
        bucket = np.searchsorted(TENURE_BINS[1:-1], out["tenure"].to_numpy(), side="left")
        out["tenure_bucket"] = np.array(TENURE_LABELS)[bucket]
        out["n_services"] = (out[SERVICE_COLS].to_numpy() == "Yes").sum(axis=1)
        out["avg_monthly_spend"] = out["TotalCharges"] / out["tenure"].clip(lower=1)
        out["charge_delta"] = out["MonthlyCharges"] - out["avg_monthly_spend"]
        out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)
        out["has_support_bundle"] = (
            (out["OnlineSecurity"] == "Yes") | (out["TechSupport"] == "Yes")
        ).astype(int)
        out["auto_pay"] = out["PaymentMethod"].str.contains("automatic", case=False).astype(int)
        return out
