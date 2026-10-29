"""Classification metrics shared by model selection (Block 3) and evaluation (Block 4)."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ["roc_auc", "pr_auc", "f1", "recall", "brier"]


def classification_scores(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Return ROC-AUC, PR-AUC (primary), F1, recall and Brier for positive-class probabilities.

    F1 and recall use `threshold`; the other three are threshold-free.
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "brier": brier_score_loss(y_true, proba),
    }
