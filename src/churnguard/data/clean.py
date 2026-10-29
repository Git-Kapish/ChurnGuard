"""Cleaning rules from PRD section 3. Never mutates the input frame."""

import pandas as pd

from churnguard.data.validate import SERVICE_COLS


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy.

    - TotalCharges: coerce to numeric; blanks (tenure == 0 customers) become 0.
    - "No phone service" / "No internet service" are folded into "No"; the parent
      flags (PhoneService, InternetService) are kept.
    - Churn becomes 1/0.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    for col in ["MultipleLines", *SERVICE_COLS]:
        out[col] = out[col].replace({"No phone service": "No", "No internet service": "No"})
    out["Churn"] = (out["Churn"] == "Yes").astype(int)
    return out
