"""Evaluation figures saved as PNG under reports/figures/."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay  # noqa: E402

from churnguard.evaluation.cost import Business, expected_value, realized_value  # noqa: E402

BLUE, RED, GREY = "#4c78a8", "#e45756", "#8a8a8a"


def _finish(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def reliability_plot(curves: dict[str, np.ndarray], y: np.ndarray, path: Path, title: str) -> Path:
    """Reliability curves (quantile bins) for several probability vectors on the same labels."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="perfectly calibrated")
    for (name, proba), color in zip(
        curves.items(), [RED, BLUE, "#54a24b", "#b279a2"], strict=False
    ):
        frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", color=color, label=name, ms=4)
    ax.set(xlabel="mean predicted probability", ylabel="observed churn rate", title=title)
    ax.legend(frameon=False, fontsize=8)
    _clean_axes(ax)
    return _finish(fig, path)


def profit_plot(
    y: np.ndarray, proba: np.ndarray, b: Business, threshold: float, path: Path, split: str
) -> Path:
    """Total net value vs threshold, label-based and model-expected, with reference rules."""
    grid = np.linspace(0, 1, 201)
    label_v, model_v = realized_value(y, b), expected_value(proba, b)
    label_curve = [label_v[proba >= t].sum() for t in grid]
    model_curve = [model_v[proba >= t].sum() for t in grid]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, label_curve, color=BLUE, label="actual outcomes")
    ax.plot(grid, model_curve, color=RED, alpha=0.8, label="model-expected")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axhline(label_v.sum(), color=BLUE, ls=":", lw=1, label="target everyone")
    ax.axvline(threshold, color="black", ls="--", lw=1, label=f"chosen threshold {threshold:.3f}")
    ax.axvline(0.5, color=GREY, ls="--", lw=1, label="default 0.5")
    ax.set(
        xlabel="probability threshold (target customers at or above)",
        ylabel="total net value",
        title=f"Net value vs threshold ({split} split)",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0,
        -0.2,
        f"Assumption-driven simulation: offer cost {b.offer_cost:g}, success rate "
        f"{b.success_rate:.0%}, customer lifetime value {b.clv:g}.",
        transform=ax.transAxes,
        fontsize=7,
        color=GREY,
    )
    _clean_axes(ax)
    return _finish(fig, path)


def roc_plot(y: np.ndarray, proba: np.ndarray, path: Path, title: str) -> Path:
    """ROC curve."""
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(
        y, proba, ax=ax, curve_kwargs={"color": BLUE}, plot_chance_level=True
    )
    ax.set_title(title)
    _clean_axes(ax)
    return _finish(fig, path)


def pr_plot(y: np.ndarray, proba: np.ndarray, path: Path, title: str) -> Path:
    """Precision-recall curve with the base-rate line."""
    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y, proba, ax=ax, color=BLUE, plot_chance_level=True)
    ax.set_title(title)
    _clean_axes(ax)
    return _finish(fig, path)


def lift_plot(table: pd.DataFrame, path: Path, title: str) -> Path:
    """Lift by decile (bars) and cumulative gain (line)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.bar(table["bin"], table["lift"], color=BLUE)
    a1.axhline(1, color=GREY, ls="--", lw=1)
    a1.set(xlabel="decile (1 = highest risk)", ylabel="lift vs average", title="Lift by decile")
    a2.plot([0, *table["bin"] / len(table)], [0, *table["cum_gain"]], "o-", color=BLUE, ms=4)
    a2.plot([0, 1], [0, 1], "--", color=GREY)
    a2.set(
        xlabel="share of customers contacted",
        ylabel="share of churners reached",
        title="Cumulative gain",
    )
    fig.suptitle(title)
    for ax in (a1, a2):
        _clean_axes(ax)
    return _finish(fig, path)
