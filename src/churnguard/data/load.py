"""Load the raw Telco churn CSV (read-only)."""

from pathlib import Path

import pandas as pd

from churnguard.config import RAW_PATH


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV as-is. TotalCharges stays a string until cleaning."""
    return pd.read_csv(path)
