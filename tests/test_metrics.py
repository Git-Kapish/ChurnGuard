import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from churnguard.evaluation.metrics import (
    METRIC_NAMES,
    bootstrap_ci,
    classification_scores,
    gain_lift_table,
    slice_table,
    threshold_metrics,
    top_decile_lift,
)


def test_classification_scores_match_hand_computation() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    s = classification_scores(y, p)
    assert list(s) == METRIC_NAMES
    assert s["roc_auc"] == pytest.approx(0.75)
    # ranked: .8(1) .4(0) .35(1) .1(0) -> precision 1.0 at recall .5, 2/3 at recall 1.0
    assert s["pr_auc"] == pytest.approx(0.5 * 1.0 + 0.5 * 2 / 3)
    # at 0.5 only the 0.8 is flagged: precision 1, recall .5
    assert s["recall"] == pytest.approx(0.5)
    assert s["f1"] == pytest.approx(2 / 3)
    assert s["brier"] == pytest.approx((0.01 + 0.16 + 0.65**2 + 0.04) / 4)


def test_threshold_changes_only_f1_and_recall() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    low, high = classification_scores(y, p, 0.3), classification_scores(y, p, 0.5)
    assert low["recall"] == 1.0 and high["recall"] == 0.5
    assert low["pr_auc"] == high["pr_auc"] and low["brier"] == high["brier"]


def test_pr_auc_helper_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    assert classification_scores(y, p)["pr_auc"] == pytest.approx(average_precision_score(y, p))


def test_top_decile_lift_perfect_and_uninformative() -> None:
    y = np.array([1] * 20 + [0] * 80)
    perfect = np.linspace(1, 0, 100)  # churners ranked first
    assert top_decile_lift(y, perfect) == pytest.approx(1.0 / 0.2)  # top 10 all churn -> 5x
    worst = perfect[::-1]
    assert top_decile_lift(y, worst) == 0.0


def test_gain_lift_table_values() -> None:
    y = np.array([1] * 20 + [0] * 80)
    t = gain_lift_table(y, np.linspace(1, 0, 100))
    assert t["n"].sum() == 100 and len(t) == 10
    assert t.loc[0, "lift"] == pytest.approx(5.0) and t.loc[2, "lift"] == 0.0
    assert t.loc[1, "cum_gain"] == pytest.approx(1.0)  # all 20 churners in the top 2 bins
    assert t.loc[9, "cum_gain"] == pytest.approx(1.0)
    assert t.loc[9, "cum_lift"] == pytest.approx(1.0)  # everyone contacted -> average


def test_threshold_metrics_known_confusion() -> None:
    m = threshold_metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.6, 0.1]), 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5


def test_bootstrap_ci_is_deterministic_and_brackets_the_estimate() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    ci = bootstrap_ci(y, p, average_precision_score, n_boot=300, seed=7)
    assert ci == bootstrap_ci(y, p, average_precision_score, n_boot=300, seed=7)
    assert ci[0] < average_precision_score(y, p) < ci[1]


def test_bootstrap_ci_narrows_with_more_data() -> None:
    def width(n: int) -> float:
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, n)
        p = np.clip(y * 0.3 + rng.random(n) * 0.7, 0, 1)
        lo, hi = bootstrap_ci(y, p, average_precision_score, n_boot=300, seed=3)
        return hi - lo

    assert width(2000) < width(200)


def test_bootstrap_ci_survives_rare_positives() -> None:
    y = np.array([1] + [0] * 29)  # many resamples will contain no positive and are redrawn
    p = np.linspace(1, 0, 30)
    lo, hi = bootstrap_ci(y, p, average_precision_score, n_boot=50, seed=0)
    assert 0 <= lo <= hi <= 1


def test_slice_table_groups_and_single_class_groups() -> None:
    groups = pd.Series(["a", "a", "a", "a", "b", "b"])
    y = np.array([1, 0, 1, 0, 1, 1])
    p = np.array([0.9, 0.1, 0.8, 0.3, 0.7, 0.6])
    t = slice_table(groups, y, p, 0.5).set_index("group")
    assert t.loc["a", "n"] == 4 and t.loc["a", "roc_auc"] == pytest.approx(1.0)
    assert t.loc["b", "churn_rate"] == 1.0 and np.isnan(t.loc["b", "roc_auc"])
    assert t.loc["b", "recall"] == 1.0
