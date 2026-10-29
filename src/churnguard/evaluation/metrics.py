"""Classification metrics shared by model selection (Block 3) and evaluation (Block 4)."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
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


def top_decile_lift(y_true: np.ndarray, proba: np.ndarray, fraction: float = 0.10) -> float:
    """Churn rate in the top `fraction` of customers by score, divided by the overall rate."""
    y = np.asarray(y_true)
    n_top = max(1, int(np.ceil(len(y) * fraction)))
    order = np.argsort(-np.asarray(proba), kind="stable")
    return float(y[order[:n_top]].mean() / y.mean())


def gain_lift_table(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Per-bin table (bin 1 = highest scores): churn rate, lift, cumulative gain and lift."""
    y = np.asarray(y_true)
    order = np.argsort(-np.asarray(proba), kind="stable")
    bins = np.array_split(y[order], n_bins)
    base = y.mean()
    rows, seen_n, seen_pos = [], 0, 0
    for i, chunk in enumerate(bins, start=1):
        seen_n += len(chunk)
        seen_pos += chunk.sum()
        rows.append(
            {
                "bin": i,
                "n": len(chunk),
                "churn_rate": chunk.mean(),
                "lift": chunk.mean() / base,
                "cum_gain": seen_pos / y.sum(),
                "cum_lift": (seen_pos / seen_n) / base,
            }
        )
    return pd.DataFrame(rows)


def threshold_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    """Precision, recall, F1 and confusion-matrix counts at a decision threshold."""
    y = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "threshold": float(threshold),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def bootstrap_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI of metric(y, proba) by resampling rows with replacement.

    Resamples that contain a single class are redrawn, since AUC-type metrics are undefined.
    """
    y, p = np.asarray(y_true), np.asarray(proba)
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    while len(stats) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        stats.append(metric(y[idx], p[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def slice_table(
    groups: pd.Series, y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> pd.DataFrame:
    """Metrics per group of `groups` (e.g. contract type). AUCs are NaN if a group has one class."""
    y, p = np.asarray(y_true), np.asarray(proba)
    rows = []
    for name in sorted(groups.astype(str).unique()):
        mask = (groups.astype(str) == name).to_numpy()
        yy, pp = y[mask], p[mask]
        both = yy.min() != yy.max()
        at = threshold_metrics(yy, pp, threshold)
        rows.append(
            {
                "group": name,
                "n": int(mask.sum()),
                "churn_rate": float(yy.mean()),
                "roc_auc": roc_auc_score(yy, pp) if both else float("nan"),
                "pr_auc": average_precision_score(yy, pp) if both else float("nan"),
                "precision": at["precision"],
                "recall": at["recall"],
            }
        )
    return pd.DataFrame(rows)
