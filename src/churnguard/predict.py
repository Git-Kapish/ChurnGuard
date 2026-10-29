"""The single inference entrypoint. The API and the dashboard both score through this module.

    predict(customers)  ->  [{"churn_probability", "risk_tier", "will_target", "top_reasons"}]

Probabilities come from the calibrated model; reasons come from SHAP on the base estimator
(see explain/explainer.py). Preprocessing is the training pipeline itself plus the shared
`normalize_customers` rules, so there is no second copy of the feature logic.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from churnguard.data.clean import normalize_customers
from churnguard.explain.explainer import Explainer
from churnguard.models.artifact import METADATA_PATH, MODEL_PATH, load_bundle, load_metadata


def risk_tier(probability: float, tiers: dict[str, float]) -> str:
    """Low below tiers["medium"], High above tiers["high"], Medium in between (inclusive)."""
    if probability > tiers["high"]:
        return "High"
    return "Medium" if probability >= tiers["medium"] else "Low"


def _single_thread(calibrated: Any, base: Any) -> None:
    """Run every classifier single-threaded: thread start-up dominates a one-row predict."""
    pipelines = [c.estimator for c in calibrated.calibrated_classifiers_] + [base]
    for pipe in pipelines:
        if "n_jobs" in pipe.named_steps["clf"].get_params():
            pipe.named_steps["clf"].set_params(n_jobs=1)


class Predictor:
    """Loads the model artifact once and scores customers."""

    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH) -> None:
        self.metadata: dict[str, Any] = load_metadata(metadata_path)
        bundle = load_bundle(model_path)
        self.model = bundle["calibrated"]
        _single_thread(self.model, bundle["base"])
        self.explainer = Explainer(bundle["base"])
        self.features: list[str] = self.metadata["features"]
        self.threshold: float = self.metadata["threshold"]
        self.tiers: dict[str, float] = self.metadata["risk_tiers"]

    @property
    def model_version(self) -> str:
        """Version string recorded in metadata.json."""
        return self.metadata["model_version"]

    def prepare(self, customers: pd.DataFrame | list[dict]) -> pd.DataFrame:
        """Customers (frame or records) -> normalized frame with exactly the model's inputs."""
        df = pd.DataFrame(customers).reset_index(drop=True)
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        return normalize_customers(df[self.features])

    def predict(
        self, customers: pd.DataFrame | list[dict], explain: bool = True
    ) -> list[dict[str, Any]]:
        """Score customers in input order.

        customerID, if present, is echoed back. explain=False skips SHAP (reasons are []).
        """
        raw = pd.DataFrame(customers).reset_index(drop=True)
        X = self.prepare(raw)
        proba = self.model.predict_proba(X)[:, 1]
        reasons = self.explainer.reasons(X) if explain else [[] for _ in range(len(X))]
        ids = raw["customerID"] if "customerID" in raw.columns else [None] * len(X)
        return [
            {
                "customerID": None if pd.isna(cid) else str(cid),
                "churn_probability": float(p),
                "risk_tier": risk_tier(float(p), self.tiers),
                "will_target": bool(p >= self.threshold),
                "top_reasons": r,
            }
            for cid, p, r in zip(ids, proba, reasons, strict=True)
        ]


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Process-wide Predictor; the artifact is loaded on first use only."""
    return Predictor()


def predict(customers: pd.DataFrame | list[dict], explain: bool = True) -> list[dict[str, Any]]:
    """Score customers with the shared Predictor (see Predictor.predict)."""
    return get_predictor().predict(customers, explain=explain)
