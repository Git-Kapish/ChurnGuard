"""Business-cost targeting (PRD FR5). All outputs are assumption-driven simulation.

Per-customer value of making an offer, with success rate s, lifetime value V, offer cost c:
    model-expected:  p * s * V - c        (p = predicted churn probability)
    label-based:     y * s * V - c        (y = 1 if the customer really churned)
A customer is worth targeting when p > c / (s * V), the break-even probability.
"""

from dataclasses import dataclass

import numpy as np

from churnguard.config import load_config


@dataclass(frozen=True)
class Business:
    """Campaign assumptions from configs/business.yaml."""

    offer_cost: float
    success_rate: float
    clv: float
    budget: float

    @classmethod
    def from_config(cls) -> "Business":
        """Load the assumptions from configs/business.yaml."""
        cfg = load_config("business")
        return cls(
            offer_cost=cfg["retention_offer_cost"],
            success_rate=cfg["offer_success_rate"],
            clv=cfg["customer_lifetime_value"],
            budget=cfg["campaign_budget"],
        )

    @property
    def breakeven_probability(self) -> float:
        """Probability above which an offer has positive expected value."""
        return self.offer_cost / (self.success_rate * self.clv)

    @property
    def max_targets(self) -> int:
        """Customers the budget can pay for."""
        return int(self.budget // self.offer_cost) if self.offer_cost > 0 else 10**9


@dataclass(frozen=True)
class Decision:
    """A targeting rule's outcome: target everyone scoring >= threshold."""

    threshold: float
    n_targeted: int
    total_value: float


def expected_value(proba: np.ndarray, b: Business) -> np.ndarray:
    """Model-expected net value of targeting each customer."""
    return np.asarray(proba) * b.success_rate * b.clv - b.offer_cost


def realized_value(y_true: np.ndarray, b: Business) -> np.ndarray:
    """Label-based expected net value of targeting each customer (churners saved at rate s)."""
    return np.asarray(y_true) * b.success_rate * b.clv - b.offer_cost


def best_threshold(scores: np.ndarray, values: np.ndarray) -> Decision:
    """Threshold maximizing the total of `values` over customers scoring >= threshold.

    Cuts are only placed between distinct scores, so tied customers are never split.
    Ties in total value prefer the smaller campaign. threshold is inf for "target nobody".
    """
    scores, values = np.asarray(scores, dtype=float), np.asarray(values, dtype=float)
    order = np.argsort(-scores, kind="stable")
    s, v = scores[order], values[order]
    totals = np.concatenate([[0.0], np.cumsum(v)])  # totals[k] = value of the top-k customers
    allowed = np.ones(len(s) + 1, dtype=bool)
    allowed[1:-1] = s[:-1] > s[1:]  # a cut after k customers needs s[k-1] > s[k]
    totals = np.where(allowed, totals, -np.inf)
    k = int(np.argmax(totals))
    return Decision(float("inf") if k == 0 else float(s[k - 1]), k, float(totals[k]))


def best_top_k(scores: np.ndarray, values: np.ndarray, max_k: int) -> Decision:
    """Best number of top-scored customers to target, never more than max_k (the budget cap)."""
    scores, values = np.asarray(scores, dtype=float), np.asarray(values, dtype=float)
    order = np.argsort(-scores, kind="stable")
    totals = np.concatenate([[0.0], np.cumsum(values[order])])[: max_k + 1]
    k = int(np.argmax(totals))
    return Decision(float("inf") if k == 0 else float(scores[order][k - 1]), k, float(totals[k]))


@dataclass(frozen=True)
class CampaignPlan:
    """A budget-limited retention campaign over a scored customer pool (simulation only)."""

    order: np.ndarray  # pool row positions, highest churn probability first
    n_targeted: int
    spend: float
    expected_net_value: float  # model-expected: sum of p * s * V - c over the target list
    expected_saves: float  # model-expected number of churners saved: sum of p * s
    churners_reached: int | None  # actual churners on the list (needs labels)
    realized_net_value: float | None  # label-based net value of the list (needs labels)

    @property
    def targets(self) -> np.ndarray:
        """Pool row positions on the target list, in rank order."""
        return self.order[: self.n_targeted]


def plan_campaign(proba: np.ndarray, b: Business, y_true: np.ndarray | None = None) -> CampaignPlan:
    """Contact the highest-risk customers the budget allows, never one with negative value.

    The list length is min(budget / offer cost, customers whose model-expected value is
    positive), so a generous budget does not make us contact customers below break-even.
    """
    p = np.asarray(proba, dtype=float)
    order = np.argsort(-p, kind="stable")
    k = min(b.max_targets, int((expected_value(p, b) > 0).sum()))
    chosen = order[:k]
    realized = reached = None
    if y_true is not None:
        y = np.asarray(y_true)
        realized = float(realized_value(y[chosen], b).sum())
        reached = int(y[chosen].sum())
    return CampaignPlan(
        order=order,
        n_targeted=k,
        spend=k * b.offer_cost,
        expected_net_value=float(expected_value(p[chosen], b).sum()),
        expected_saves=float(p[chosen].sum() * b.success_rate),
        churners_reached=reached,
        realized_net_value=realized,
    )


def cumulative_expected_value(proba: np.ndarray, b: Business) -> np.ndarray:
    """Model-expected net value after contacting the top-1, top-2, ... customers by risk."""
    p = np.sort(np.asarray(proba, dtype=float))[::-1]
    return np.cumsum(expected_value(p, b))


def compare_strategies(
    y_true: np.ndarray, proba: np.ndarray, b: Business, threshold: float
) -> dict[str, dict[str, float]]:
    """Net value of the standard targeting rules, scored with the actual labels.

    Strategies: nobody, everyone, the default 0.5 threshold, the chosen threshold, and the
    best top-K within the budget. Also reports each rule's model-expected value.
    """
    y, p = np.asarray(y_true), np.asarray(proba)
    label_value, model_value = realized_value(y, b), expected_value(p, b)
    budget_rule = best_top_k(p, label_value, b.max_targets)
    top_k = np.zeros(len(y), dtype=bool)  # by rank, not threshold: tied scores must not exceed K
    top_k[np.argsort(-p, kind="stable")[: budget_rule.n_targeted]] = True
    masks = {
        "target_nobody": np.zeros(len(y), dtype=bool),
        "target_everyone": np.ones(len(y), dtype=bool),
        "threshold_0.5": p >= 0.5,
        "chosen_threshold": p >= threshold,
        "best_top_k_within_budget": top_k,
    }
    out = {}
    for name, mask in masks.items():
        out[name] = {
            "n_targeted": int(mask.sum()),
            "net_value": float(label_value[mask].sum()),
            "model_expected_value": float(model_value[mask].sum()),
            "churners_reached": int(y[mask].sum()),
            "share_of_churners_reached": float(y[mask].sum() / y.sum()),
        }
    return out
