import numpy as np
import pytest

from churnguard.evaluation.cost import (
    Business,
    best_threshold,
    best_top_k,
    compare_strategies,
    cumulative_expected_value,
    expected_value,
    plan_campaign,
    realized_value,
)

DEFAULT = Business(offer_cost=20, success_rate=0.3, clv=500, budget=3000)


def test_defaults_come_from_business_yaml() -> None:
    b = Business.from_config()
    assert b.breakeven_probability == pytest.approx(20 / (0.3 * 500))
    assert b.max_targets == 150


def test_value_formulas() -> None:
    assert expected_value(np.array([0.5]), DEFAULT)[0] == pytest.approx(0.5 * 0.3 * 500 - 20)
    assert list(realized_value(np.array([1, 0]), DEFAULT)) == [0.3 * 500 - 20, -20]


def test_targets_nobody_when_offer_cost_is_huge() -> None:
    p = np.array([0.9, 0.7, 0.4, 0.1])
    d = best_threshold(p, expected_value(p, Business(1e6, 0.3, 500, 1e9)))
    assert d.n_targeted == 0 and d.total_value == 0 and d.threshold == float("inf")


def test_targets_everyone_when_offer_cost_is_negligible() -> None:
    p = np.array([0.9, 0.7, 0.4, 0.01])
    d = best_threshold(p, expected_value(p, Business(1e-9, 0.3, 500, 1e9)))
    assert d.n_targeted == 4 and d.threshold == pytest.approx(0.01)


def test_threshold_lands_at_the_breakeven_probability() -> None:
    p = np.array([0.9, 0.5, 0.2, 0.1, 0.05])  # break-even is 20 / 150 = 0.1333
    d = best_threshold(p, expected_value(p, DEFAULT))
    assert d.n_targeted == 3 and d.threshold == 0.2
    assert d.total_value == pytest.approx((135 - 20) + (75 - 20) + (30 - 20))


def test_tied_scores_are_never_split() -> None:
    scores = np.array([0.5, 0.5, 0.1])
    values = np.array([10.0, -5.0, -100.0])
    d = best_threshold(scores, values)
    assert d.n_targeted == 2 and d.total_value == 5.0  # taking only the +10 would split the tie


def test_equal_value_prefers_the_smaller_campaign() -> None:
    d = best_threshold(np.array([0.9, 0.5, 0.1]), np.array([5.0, 0.0, 0.0]))
    assert d.n_targeted == 1


def test_top_k_is_capped_by_budget_and_stops_when_value_turns_negative() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    capped = best_top_k(scores, np.array([5.0, 5.0, 5.0, 5.0]), max_k=2)
    assert capped.n_targeted == 2 and capped.total_value == 10.0
    stops = best_top_k(scores, np.array([5.0, 5.0, -1.0, 5.0]), max_k=4)
    assert stops.n_targeted == 4 and stops.total_value == 14.0
    negative = best_top_k(scores, np.array([5.0, -9.0, 9.0, 9.0]), max_k=4)
    assert negative.n_targeted == 4 and negative.total_value == 14.0
    stops_early = best_top_k(scores, np.array([5.0, -9.0, 1.0, 1.0]), max_k=4)
    assert stops_early.n_targeted == 1 and stops_early.total_value == 5.0


def test_compare_strategies_known_values() -> None:
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.4, 0.2, 0.05])
    s = compare_strategies(y, p, DEFAULT, threshold=0.2)
    win, loss = 0.3 * 500 - 20, -20
    assert s["target_nobody"]["net_value"] == 0 and s["target_nobody"]["n_targeted"] == 0
    assert s["target_everyone"]["net_value"] == pytest.approx(2 * win + 3 * loss)
    assert s["threshold_0.5"]["net_value"] == pytest.approx(2 * win)
    assert s["chosen_threshold"]["n_targeted"] == 4
    assert s["chosen_threshold"]["net_value"] == pytest.approx(2 * win + 2 * loss)
    assert s["chosen_threshold"]["share_of_churners_reached"] == 1.0
    assert s["best_top_k_within_budget"]["net_value"] == pytest.approx(2 * win)


def test_campaign_is_limited_by_the_budget() -> None:
    p = np.array([0.1, 0.9, 0.6, 0.8, 0.7])
    b = Business(offer_cost=20, success_rate=0.3, clv=500, budget=40)  # pays for 2 offers
    plan = plan_campaign(p, b)
    assert plan.n_targeted == 2 and list(plan.targets) == [1, 3]  # the 0.9 and 0.8 customers
    assert plan.spend == 40
    assert plan.expected_net_value == pytest.approx((0.9 * 150 - 20) + (0.8 * 150 - 20))
    assert plan.expected_saves == pytest.approx((0.9 + 0.8) * 0.3)
    assert plan.realized_net_value is None and plan.churners_reached is None


def test_campaign_never_contacts_customers_below_breakeven() -> None:
    p = np.array([0.9, 0.5, 0.1, 0.05])  # break-even is 0.1333
    b = Business(offer_cost=20, success_rate=0.3, clv=500, budget=10_000)
    plan = plan_campaign(p, b)
    assert plan.n_targeted == 2 and plan.spend == 40


def test_campaign_with_outcomes_reports_actual_value() -> None:
    p = np.array([0.1, 0.9, 0.6, 0.8, 0.7])
    y = np.array([0, 1, 0, 0, 1])
    b = Business(offer_cost=20, success_rate=0.3, clv=500, budget=40)
    plan = plan_campaign(p, b, y)  # contacts customers 1 (churner) and 3 (not a churner)
    assert plan.churners_reached == 1
    assert plan.realized_net_value == pytest.approx((150 - 20) + (-20))


def test_campaign_edge_cases_zero_budget_and_nothing_worth_contacting() -> None:
    p = np.array([0.9, 0.5])
    none = plan_campaign(p, Business(20, 0.3, 500, budget=0))
    assert none.n_targeted == 0 and none.spend == 0 and none.expected_net_value == 0
    hopeless = plan_campaign(p, Business(offer_cost=1e6, success_rate=0.3, clv=500, budget=1e9))
    assert hopeless.n_targeted == 0


def test_cumulative_expected_value_peaks_at_the_breakeven_customer() -> None:
    p = np.array([0.05, 0.9, 0.5, 0.1])
    curve = cumulative_expected_value(p, DEFAULT)
    assert curve[-1] == pytest.approx(expected_value(p, DEFAULT).sum())
    assert int(np.argmax(curve)) + 1 == 2  # only 0.9 and 0.5 clear the 0.1333 break-even
    assert curve[0] == pytest.approx(0.9 * 150 - 20)


def test_budget_rule_does_not_exceed_cap_with_tied_probabilities() -> None:
    y = np.ones(10, dtype=int)
    p = np.full(10, 0.5)  # isotonic-style plateau: all tied
    b = Business(offer_cost=20, success_rate=0.3, clv=500, budget=60)  # pays for 3 offers
    s = compare_strategies(y, p, b, threshold=0.5)
    assert s["best_top_k_within_budget"]["n_targeted"] == 3
