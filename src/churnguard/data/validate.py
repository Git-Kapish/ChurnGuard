"""Schema, allowed-category and numeric-range checks for raw and cleaned data."""

import pandas as pd

YES_NO = {"Yes", "No"}
SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Allowed categories in the raw file.
RAW_CATEGORIES: dict[str, set[str]] = {
    "gender": {"Female", "Male"},
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": YES_NO | {"No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": YES_NO,
    "PaymentMethod": {
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    },
    "Churn": YES_NO,
    **{c: YES_NO | {"No internet service"} for c in SERVICE_COLS},
}

# After cleaning, the redundant "No ... service" categories are folded into "No".
CLEAN_CATEGORIES: dict[str, set[str]] = {
    **{k: v - {"No phone service", "No internet service"} for k, v in RAW_CATEGORIES.items()},
}
CLEAN_CATEGORIES.pop("Churn")

COLUMNS = [
    "customerID",
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
] + list(RAW_CATEGORIES.keys() - {"Churn"})
TENURE_RANGE = (0, 100)
MONTHLY_RANGE = (0.0, 250.0)


class ValidationError(ValueError):
    """Raised when a dataframe violates the expected schema."""


def _check_common(df: pd.DataFrame, categories: dict[str, set[str]]) -> list[str]:
    errors = [f"missing column: {c}" for c in COLUMNS if c not in df.columns]
    if errors:
        return errors
    if not df["customerID"].is_unique:
        errors.append("customerID is not unique")
    for col, allowed in categories.items():
        bad = set(df[col].dropna().unique()) - allowed
        if bad:
            errors.append(f"{col}: unexpected values {sorted(bad)}")
    if not df["SeniorCitizen"].isin([0, 1]).all():
        errors.append("SeniorCitizen must be 0/1")
    if not df["tenure"].between(*TENURE_RANGE).all():
        errors.append(f"tenure outside {TENURE_RANGE}")
    if not df["MonthlyCharges"].between(*MONTHLY_RANGE).all():
        errors.append(f"MonthlyCharges outside {MONTHLY_RANGE}")
    return errors


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Check the raw frame; blank TotalCharges is allowed only for tenure == 0."""
    errors = _check_common(df, RAW_CATEGORIES)
    if not errors:
        blank = df["TotalCharges"].astype(str).str.strip() == ""
        if (df.loc[blank, "tenure"] != 0).any():
            errors.append("blank TotalCharges on rows with tenure > 0")
    if errors:
        raise ValidationError("; ".join(errors))
    return df


def validate_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Check the cleaned frame: no nulls, numeric TotalCharges >= 0, binary target."""
    errors = _check_common(df, CLEAN_CATEGORIES)
    if not errors:
        if df.isna().any().any():
            errors.append("nulls present after cleaning")
        if not pd.api.types.is_numeric_dtype(df["TotalCharges"]) or (df["TotalCharges"] < 0).any():
            errors.append("TotalCharges must be numeric and >= 0")
        if not df["Churn"].isin([0, 1]).all():
            errors.append("Churn must be 0/1")
    if errors:
        raise ValidationError("; ".join(errors))
    return df
