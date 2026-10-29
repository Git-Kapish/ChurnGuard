"""Paths and YAML config loading shared across the package."""

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml (e.g. 'model' or 'business') into a dict."""
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
