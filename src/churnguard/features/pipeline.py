"""Preprocessing pipeline: feature engineering + impute/scale + one-hot.

Everything that learns from data (imputer medians, scaler stats, encoder categories) lives
inside this one Pipeline, so fitting it on training folds only keeps validation/test clean.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnguard.features.engineering import FeatureEngineer

NUMERIC = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "charge_delta",
    "n_services",
]
BINARY = ["SeniorCitizen", "is_month_to_month", "has_support_bundle", "auto_pay"]
CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_bucket",
]


def build_preprocessor() -> Pipeline:
    """Return an unfitted Pipeline(engineer -> ColumnTransformer) that outputs a DataFrame.

    customerID and Churn are dropped (remainder="drop"). Binary flags pass through unscaled.
    """
    columns = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC,
            ),
            ("bin", "passthrough", BINARY),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return Pipeline([("engineer", FeatureEngineer()), ("columns", columns)])
