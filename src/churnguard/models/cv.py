"""Cross-validation on identical stratified folds."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from churnguard.evaluation.metrics import METRIC_NAMES, classification_scores

Folds = list[tuple[np.ndarray, np.ndarray]]


def make_folds(y: pd.Series, n_splits: int = 5, seed: int = 42) -> Folds:
    """Stratified shuffled folds; build once and reuse so every model sees the same splits."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def cross_validate(estimator, X: pd.DataFrame, y: pd.Series, folds: Folds) -> pd.DataFrame:
    """Fit a fresh clone per fold on the training part; score the held-out part.

    Returns one row per fold with the METRIC_NAMES columns.
    """
    rows = []
    for train_idx, val_idx in folds:
        model = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        rows.append(classification_scores(y.iloc[val_idx], proba))
    return pd.DataFrame(rows, columns=METRIC_NAMES)


def summarize(fold_scores: pd.DataFrame) -> dict[str, float]:
    """Flatten per-fold scores into {metric_mean, metric_std} (sample std, ddof=1)."""
    out: dict[str, float] = {}
    for col in fold_scores:
        out[f"{col}_mean"] = float(fold_scores[col].mean())
        out[f"{col}_std"] = float(fold_scores[col].std(ddof=1))
    return out
