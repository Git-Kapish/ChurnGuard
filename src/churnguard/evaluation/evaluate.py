"""Score the held-out TEST split once, using the saved model and the validation-chosen threshold.

This is the only place the test split is read. Re-running requires --force, and a forced rerun
is recorded in the output so a second look at the test set is never silent.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from churnguard.config import ROOT, load_config
from churnguard.data.split import load_split, split_xy
from churnguard.evaluation import plots
from churnguard.evaluation.cost import Business, compare_strategies
from churnguard.evaluation.metrics import (
    bootstrap_ci,
    gain_lift_table,
    slice_table,
    threshold_metrics,
    top_decile_lift,
)
from churnguard.features.engineering import TENURE_BINS, TENURE_LABELS
from churnguard.models import tracking
from churnguard.models.artifact import METADATA_PATH, MODEL_PATH, load_bundle, load_metadata

log = logging.getLogger("churnguard.evaluate")
REPORTS_DIR = ROOT / "reports"

# PRD section 7 targets: (name, direction, bound). "max" means the value must be <= bound.
PRD_TARGETS = {
    "roc_auc": (">=", 0.84),
    "pr_auc": (">=", 0.64),
    "recall": (">=", 0.70),
    "top_decile_lift": (">=", 2.8),
    "brier": ("<=", 0.15),
}


class AlreadyEvaluatedError(RuntimeError):
    """Raised when the test split has already been scored and --force was not given."""


def evaluate_model(
    proba: np.ndarray,
    y: np.ndarray,
    X: pd.DataFrame,
    threshold: float,
    business: Business,
    seed: int = 42,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Compute every reported number for one split. Pure: no file or model access."""
    at = threshold_metrics(y, proba, threshold)
    metrics = {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "log_loss": log_loss(y, proba),
        "top_decile_lift": top_decile_lift(y, proba),
        **{k: at[k] for k in ("threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn")},
    }
    ci_fns = {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
        "brier": brier_score_loss,
        "recall": lambda yy, pp: threshold_metrics(yy, pp, threshold)["recall"],
        "top_decile_lift": top_decile_lift,
    }
    ci = {k: bootstrap_ci(y, proba, fn, n_boot, seed=seed) for k, fn in ci_fns.items()}

    tenure_bucket = pd.cut(X["tenure"], TENURE_BINS, labels=TENURE_LABELS).astype(str)
    senior = X["SeniorCitizen"].map({0: "No", 1: "Yes"})
    slices = {
        name: slice_table(groups, y, proba, threshold).to_dict("records")
        for name, groups in {
            "contract": X["Contract"],
            "tenure_bucket": tenure_bucket,
            "senior_citizen": senior,
        }.items()
    }

    strategies = compare_strategies(y, proba, business, threshold)
    chosen, everyone = strategies["chosen_threshold"], strategies["target_everyone"]
    targets = {}
    for name, (direction, bound) in PRD_TARGETS.items():
        value = metrics[name]
        ok = value >= bound if direction == ">=" else value <= bound
        targets[name] = {"value": value, "target": f"{direction} {bound}", "met": bool(ok)}
    targets["net_value_vs_target_everyone"] = {
        "value": chosen["net_value"] - everyone["net_value"],
        "target": "> 0",
        "met": bool(chosen["net_value"] > everyone["net_value"]),
    }
    return {
        "n": len(y),
        "churn_rate": float(np.mean(y)),
        "metrics": metrics,
        "bootstrap_95ci": {k: list(v) for k, v in ci.items()},
        "slices": slices,
        "strategies": strategies,
        "prd_targets": targets,
    }


def render_report(result: dict[str, Any], split: str) -> str:
    """Markdown rendering of an evaluate_model result."""
    m, ci = result["metrics"], result["bootstrap_95ci"]

    def row(label: str, key: str) -> str:
        lo, hi = ci[key]
        return f"| {label} | {m[key]:.3f} | [{lo:.3f}, {hi:.3f}] |"

    lines = [
        f"# Evaluation on the {split} split",
        "",
        f"{result['n']} customers, churn rate {result['churn_rate']:.1%}. "
        f"Decision threshold {m['threshold']:.3f} was chosen on the validation split. "
        "Confidence intervals are 1000-sample bootstrap percentile intervals.",
        "",
        "| Metric | Value | 95% CI |",
        "|---|---|---|",
        row("PR-AUC (primary)", "pr_auc"),
        row("ROC-AUC", "roc_auc"),
        row("Brier score", "brier"),
        row("Recall at threshold", "recall"),
        row("Top-decile lift", "top_decile_lift"),
        f"| Precision at threshold | {m['precision']:.3f} | |",
        f"| F1 at threshold | {m['f1']:.3f} | |",
        f"| Log loss | {m['log_loss']:.3f} | |",
        "",
        f"Confusion matrix at the threshold: TN {m['tn']}, FP {m['fp']}, "
        f"FN {m['fn']}, TP {m['tp']}.",
        "",
        "## PRD targets",
        "",
        "| Metric | Value | Target | Met |",
        "|---|---|---|---|",
    ]
    for k, t in result["prd_targets"].items():
        lines.append(f"| {k} | {t['value']:.3f} | {t['target']} | {'yes' if t['met'] else 'NO'} |")
    lines += [
        "",
        "## Targeting rules (assumption-driven simulation, actual outcomes)",
        "",
        "| Strategy | Targeted | Net value | Share of churners reached |",
        "|---|---|---|---|",
    ]
    for k, s in result["strategies"].items():
        share = s["share_of_churners_reached"]
        lines.append(f"| {k} | {s['n_targeted']} | {s['net_value']:,.0f} | {share:.1%} |")
    for name, rows in result["slices"].items():
        lines += [
            "",
            f"## Slice: {name}",
            "",
            "| Group | n | Churn rate | ROC-AUC | PR-AUC | Precision | Recall |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['group']} | {r['n']} | {r['churn_rate']:.1%} | {r['roc_auc']:.3f} | "
                f"{r['pr_auc']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def run(
    force: bool = False,
    split: str = "test",
    reports_dir: Path = REPORTS_DIR,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    tracking_uri: str | None = None,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Evaluate the saved model on `split`, write reports/figures, update metadata.

    The test split is guarded by reports/test_metrics.json: if it exists, a rerun needs force.
    """
    record_path = reports_dir / f"{split}_metrics.json"
    if split == "test" and record_path.exists() and not force:
        raise AlreadyEvaluatedError(
            f"The test split was already scored ({record_path}). Re-running means looking at the "
            "test set again; pass --force only if that is deliberate."
        )
    cfg = load_config("model")
    meta, bundle = load_metadata(metadata_path), load_bundle(model_path)
    X, y = split_xy(load_split(split))
    y_arr = y.to_numpy()
    proba = bundle["calibrated"].predict_proba(X)[:, 1]

    result = evaluate_model(proba, y_arr, X, meta["threshold"], Business.from_config(), cfg["seed"])
    result["split"] = split
    result["forced_rerun"] = bool(force and record_path.exists())
    result["model_version"] = meta["model_version"]
    result["git_hash"] = tracking.git_hash()

    reports_dir.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (reports_dir / f"{split}_evaluation.md").write_text(
        render_report(result, split), encoding="utf-8"
    )

    fig = reports_dir / "figures"
    figures = [
        plots.roc_plot(y_arr, proba, fig / f"{split}_roc.png", f"ROC ({split} split)"),
        plots.pr_plot(y_arr, proba, fig / f"{split}_pr.png", f"Precision-recall ({split} split)"),
        plots.reliability_plot(
            {"calibrated model": proba},
            y_arr,
            fig / f"{split}_calibration.png",
            f"Reliability ({split} split)",
        ),
        plots.lift_plot(gain_lift_table(y_arr, proba), fig / f"{split}_lift.png", f"{split} split"),
        plots.profit_plot(
            y_arr,
            proba,
            Business.from_config(),
            meta["threshold"],
            fig / f"{split}_profit_vs_threshold.png",
            split,
        ),
    ]

    if split == "test":
        meta["test_metrics"] = {
            "metrics": result["metrics"],
            "bootstrap_95ci": result["bootstrap_95ci"],
            "prd_targets": result["prd_targets"],
        }
        metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if log_to_mlflow:
        tracking.setup(
            tracking_uri or cfg["mlflow"]["tracking_uri"], cfg["mlflow"]["experiment_name"]
        )
        with tracking.run(
            f"{split}-evaluation", {"split": split, "forced": result["forced_rerun"]}
        ):
            mlflow.log_metrics({f"{split}_{k}": float(v) for k, v in result["metrics"].items()})
            for f in figures:
                mlflow.log_artifact(str(f), "figures")
            mlflow.log_artifact(str(record_path))
    log.info("wrote %s", record_path)
    return result


def main() -> None:
    """CLI entrypoint (`make evaluate`, or `make evaluate ARGS=--force`)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rerun even though test was scored")
    args = parser.parse_args()
    try:
        result = run(force=args.force)
    except AlreadyEvaluatedError as exc:
        log.error("%s", exc)
        sys.exit(1)
    for name, t in result["prd_targets"].items():
        log.info(
            "%-32s %8.3f  target %-8s %s",
            name,
            t["value"],
            t["target"],
            "ok" if t["met"] else "MISSED",
        )


if __name__ == "__main__":
    main()
