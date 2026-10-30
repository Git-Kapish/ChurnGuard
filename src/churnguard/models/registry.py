"""Model factory: preprocessing + optional SMOTE + classifier as one leakage-safe pipeline."""

from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from churnguard.features.pipeline import build_preprocessor

MODEL_NAMES = ["dummy", "logistic_regression", "random_forest", "xgboost", "lightgbm"]
IMBALANCE = ["class_weight", "smote"]


class BalancedXGBClassifier(XGBClassifier):
    """XGBClassifier with sklearn-style class_weight="balanced".

    Weights are computed from the y seen in each fit, so every CV fold uses its own
    training labels only (a fixed scale_pos_weight would peek at validation-fold labels).
    """

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit with balanced sample weights, multiplied into any weights passed in."""
        weights = compute_sample_weight("balanced", y)
        if sample_weight is not None:
            weights = weights * sample_weight
        return super().fit(X, y, sample_weight=weights, **kwargs)


def _classifier(name: str, imbalance: str, params: dict[str, Any], seed: int):
    weighted = imbalance == "class_weight"
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logistic_regression":
        base = {"C": 1.0, "max_iter": 2000}
        return LogisticRegression(
            **{**base, **params}, class_weight="balanced" if weighted else None, random_state=seed
        )
    if name == "random_forest":
        base = {"n_estimators": 300, "min_samples_leaf": 3}
        return RandomForestClassifier(
            **{**base, **params},
            class_weight="balanced" if weighted else None,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "xgboost":
        base = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4}
        cls = BalancedXGBClassifier if weighted else XGBClassifier
        return cls(**{**base, **params}, n_jobs=-1, random_state=seed, eval_metric="logloss")
    if name == "lightgbm":
        base = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15}
        return LGBMClassifier(
            **{**base, **params},
            class_weight="balanced" if weighted else None,
            n_jobs=-1,
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
    raise ValueError(f"unknown model: {name}")


def build_estimator(
    name: str, imbalance: str = "class_weight", params: dict[str, Any] | None = None, seed: int = 42
) -> Pipeline:
    """Return an unfitted imblearn Pipeline for model `name`.

    The preprocessor steps are spliced in (imblearn forbids a nested Pipeline). With
    imbalance="smote", SMOTE runs after preprocessing and only on training folds
    (imblearn samplers are skipped at predict time).
    """
    if imbalance not in IMBALANCE:
        raise ValueError(f"imbalance must be one of {IMBALANCE}")
    steps = list(build_preprocessor().steps)
    if imbalance == "smote" and name != "dummy":
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("clf", _classifier(name, imbalance, params or {}, seed)))
    return Pipeline(steps)
