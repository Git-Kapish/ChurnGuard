"""Locations and loaders for the saved model artifact and its metadata."""

import json
from pathlib import Path
from typing import Any

import joblib

from churnguard.config import ROOT

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model_v1.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"


def load_bundle(path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load {"calibrated": CalibratedClassifierCV, "base": fitted Pipeline} from joblib.

    "calibrated" produces the probabilities used everywhere; "base" is the uncalibrated
    pipeline kept for SHAP (explain the base estimator, calibrate separately).

    Security: joblib is pickle-based, so only load artifacts this project built itself
    (models/ is written by `make train`); never load a model file from an untrusted source.
    """
    return joblib.load(path)


def load_metadata(path: Path = METADATA_PATH) -> dict[str, Any]:
    """Load metadata.json."""
    return json.loads(path.read_text(encoding="utf-8"))
