import numpy as np
import pytest

from churnguard.evaluation.metrics import METRIC_NAMES, classification_scores


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
