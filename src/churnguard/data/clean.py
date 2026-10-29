"""Cleaning rules from PRD section 3. Never mutates the input frame."""

import pandas as pd

from churnguard.data.validate import SERVICE_COLS


def normalize_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Rules shared by training and inference (predict.py), on frames with or without Churn.

    - TotalCharges: coerce to numeric; a blank for a tenure-0 customer becomes 0. Other blanks
      stay NaN for the pipeline's median imputer.
    - "No phone service" / "No internet service" are folded into "No"; the parent
      flags (PhoneService, InternetService) are kept.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    new_customer = out["TotalCharges"].isna() & (out["tenure"] == 0)
    out.loc[new_customer, "TotalCharges"] = 0.0
    folded = ["MultipleLines", *SERVICE_COLS]
    out[folded] = out[folded].replace({"No phone service": "No", "No internet service": "No"})
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned training frame: normalized, no nulls in TotalCharges, Churn as 1/0."""
    out = normalize_customers(df)
    out["TotalCharges"] = out["TotalCharges"].fillna(0.0)
    out["Churn"] = (out["Churn"] == "Yes").astype(int)
    return out
