"""Paths and YAML config loading shared across the package."""

import os
from pathlib import Path
from typing import Any

import yaml

# CHURNGUARD_ROOT is set in Docker, where the package lives in site-packages and the repo
# files (configs/, models/, reports/, data/) are mounted elsewhere.
ROOT = Path(os.environ.get("CHURNGUARD_ROOT") or Path(__file__).resolve().parents[2])
RAW_PATH = ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml (e.g. 'model' or 'business') into a dict."""
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
