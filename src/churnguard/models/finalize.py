"""Block 4 training stage: pick the winner, calibrate, choose the cost threshold, save.

Uses the train and validation splits only; the test split is reserved for evaluate.py.
Calibration is cross-fitted on the training split (5 folds), so the validation split is a
fair, untouched judge of isotonic vs sigmoid and of the threshold.
"""

import json
import logging
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split

from churnguard.config import ROOT, load_config
from churnguard.data.split import load_split, split_xy
from churnguard.evaluation import plots
from churnguard.evaluation.cost import (
    Business,
    best_threshold,
    best_top_k,
    compare_strategies,
    expected_value,
    realized_value,
)
from churnguard.evaluation.metrics import classification_scores, threshold_metrics, top_decile_lift
from churnguard.models import tracking
from churnguard.models.artifact import METADATA_PATH, MODEL_PATH
from churnguard.models.registry import build_estimator

log = logging.getLogger("churnguard.finalize")
REPORTS_DIR = ROOT / "reports"
LIBRARIES = ["scikit-learn", "xgboost", "lightgbm", "imbalanced-learn", "pandas", "numpy", "joblib"]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _scores(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {**classification_scores(y, proba), "top_decile_lift": top_decile_lift(y, proba)}


def select_winner(tuned: dict[str, Any], Xtr, ytr, Xv, yv, seed: int):  # noqa: ANN001, ANN201
    """Fit each tuned candidate on train; the best validation PR-AUC wins."""
    fitted, scores = {}, {}
    for name, info in tuned.items():
        model = build_estimator(name, info["imbalance"], info["best_params"], seed).fit(Xtr, ytr)
        fitted[name] = model
        scores[name] = _scores(yv, model.predict_proba(Xv)[:, 1])
        log.info("candidate %-20s val PR-AUC %.4f", name, scores[name]["pr_auc"])
    winner = max(scores, key=lambda n: scores[n]["pr_auc"])
    return winner, fitted[winner], scores


def run_final(
    sample: int | None = None,
    tuning_path: Path = REPORTS_DIR / "tuning_results.json",
    reports_dir: Path = REPORTS_DIR,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Select, calibrate, set the threshold and write the model artifact and metadata."""
    cfg = load_config("model")
    seed = cfg["seed"]
    business = Business.from_config()
    tracking.setup(tracking_uri or cfg["mlflow"]["tracking_uri"], cfg["mlflow"]["experiment_name"])
    tuned = json.loads(tuning_path.read_text(encoding="utf-8"))

    Xtr, ytr = split_xy(load_split("train"))
    Xv, yv = split_xy(load_split("val"))
    if sample:
        Xtr, _, ytr, _ = train_test_split(
            Xtr, ytr, train_size=sample, stratify=ytr, random_state=seed
        )
    yv_arr = yv.to_numpy()

    winner, base, candidates = select_winner(tuned, Xtr, ytr, Xv, yv, seed)
    info = tuned[winner]
    log.info("winner on validation: %s", winner)

    # Cross-fitted calibration on train; compare on validation.
    uncal = base.predict_proba(Xv)[:, 1]
    calibrators, cal_proba, cal_scores = {}, {}, {"uncalibrated": _scores(yv_arr, uncal)}
    for method in cfg["calibration"]["methods"]:
        cal = CalibratedClassifierCV(
            clone(base),
            method=method,
            cv=StratifiedKFold(cfg["cv"]["n_splits"], shuffle=True, random_state=seed),
            ensemble=True,
        ).fit(Xtr, ytr)
        calibrators[method] = cal
        cal_proba[method] = cal.predict_proba(Xv)[:, 1]
        cal_scores[method] = _scores(yv_arr, cal_proba[method])
        log.info(
            "%-9s val Brier %.4f  PR-AUC %.4f",
            method,
            cal_scores[method]["brier"],
            cal_scores[method]["pr_auc"],
        )
    method = min(cfg["calibration"]["methods"], key=lambda m: cal_scores[m]["brier"])
    model, proba = calibrators[method], cal_proba[method]

    # Cost-based threshold on validation (label-based value; model-expected shown for reference).
    decision = best_threshold(proba, realized_value(yv_arr, business))
    if not np.isfinite(decision.threshold):
        raise RuntimeError("Cost-optimal rule targets nobody under the configured assumptions.")
    model_rule = best_threshold(proba, expected_value(proba, business))
    budget_rule = best_top_k(proba, realized_value(yv_arr, business), business.max_targets)
    strategies = compare_strategies(yv_arr, proba, business, decision.threshold)
    at_threshold = threshold_metrics(yv_arr, proba, decision.threshold)
    log.info(
        "threshold %.3f (break-even %.3f) targets %d of %d validation customers",
        decision.threshold,
        business.breakeven_probability,
        decision.n_targeted,
        len(yv_arr),
    )

    fig_dir = reports_dir / "figures"
    figures = [
        plots.reliability_plot(
            {"uncalibrated": uncal, **cal_proba},
            yv_arr,
            fig_dir / "calibration_validation.png",
            "Reliability on validation split",
        ),
        plots.profit_plot(
            yv_arr,
            proba,
            business,
            decision.threshold,
            fig_dir / "profit_vs_threshold.png",
            "validation",
        ),
    ]

    metadata = {
        "model_version": "v1",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_name": winner,
        "imbalance": info["imbalance"],
        "params": info["best_params"],
        "calibration": method,
        "threshold": decision.threshold,
        "threshold_rule": "maximize label-based expected net value on the validation split",
        "breakeven_probability": business.breakeven_probability,
        "risk_tiers": cfg["risk_tiers"],
        "business_assumptions": {
            "note": "Assumptions, not measured facts; every profit figure is a simulation.",
            "retention_offer_cost": business.offer_cost,
            "offer_success_rate": business.success_rate,
            "customer_lifetime_value": business.clv,
            "campaign_budget": business.budget,
        },
        "features": list(Xtr.columns),
        "model_input_features": list(base["columns"].get_feature_names_out()),
        "metrics": {
            **_scores(yv_arr, proba),
            "recall_at_threshold": at_threshold["recall"],
            "precision_at_threshold": at_threshold["precision"],
            "split": "validation",
        },
        "validation": {
            "candidates": candidates,
            "calibration": cal_scores,
            "threshold_decision": decision.__dict__,
            "expected_value_threshold": model_rule.__dict__,
            "budget_top_k": budget_rule.__dict__,
            "strategies": strategies,
        },
        "train_rows": len(ytr),
        "git_hash": tracking.git_hash(),
        "git_dirty": tracking.git_dirty(),
        "data_sha256": tracking.data_checksum(),
        "library_versions": {
            "python": platform.python_version(),
            **{n: version(n) for n in LIBRARIES},
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"calibrated": model, "base": base}, model_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
    )
    (reports_dir / "validation_summary.md").write_text(render_summary(metadata), encoding="utf-8")

    with tracking.run(
        "final-model", {"model": winner, "calibration": method, **info["best_params"]}
    ):
        mlflow.log_metrics({f"val_{k}": v for k, v in metadata["metrics"].items() if k != "split"})
        mlflow.log_metric("threshold", decision.threshold)
        mlflow.log_artifacts(str(fig_dir), "figures")
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(str(model_path))
    log.info("saved %s and %s (%d figures)", model_path.name, metadata_path.name, len(figures))
    return metadata


def render_summary(meta: dict[str, Any]) -> str:
    """Markdown summary of validation-stage decisions (winner, calibration, threshold, value)."""
    v, a = meta["validation"], meta["business_assumptions"]
    lines = [
        "# Validation summary (model selection, calibration, threshold)",
        "",
        "All numbers on this page come from the validation split. Dollar figures are an "
        "**assumption-driven simulation** "
        f"(offer cost {a['retention_offer_cost']:g}, success rate {a['offer_success_rate']:.0%}, "
        f"lifetime value {a['customer_lifetime_value']:g}, budget {a['campaign_budget']:g}).",
        "",
        "## Tuned candidates (fit on train)",
        "",
        "| Model | PR-AUC | ROC-AUC | Brier |",
        "|---|---|---|---|",
    ]
    for n, s in v["candidates"].items():
        label = f"{n} (winner)" if n == meta["model_name"] else n
        lines.append(f"| {label} | {s['pr_auc']:.4f} | {s['roc_auc']:.4f} | {s['brier']:.4f} |")
    lines += [
        "",
        f"## Calibration (chosen: **{meta['calibration']}**, lowest validation Brier)",
        "",
        "| Version | Brier | PR-AUC | ROC-AUC |",
        "|---|---|---|---|",
    ]
    for n, s in v["calibration"].items():
        lines.append(f"| {n} | {s['brier']:.4f} | {s['pr_auc']:.4f} | {s['roc_auc']:.4f} |")
    d = v["threshold_decision"]
    lines += [
        "",
        "## Threshold",
        "",
        f"Chosen threshold **{meta['threshold']:.3f}** targets {d['n_targeted']} validation "
        f"customers for a net value of {d['total_value']:,.0f}. The break-even probability "
        f"c / (s * CLV) is {meta['breakeven_probability']:.3f}; maximizing model-expected value "
        f"gives {v['expected_value_threshold']['threshold']:.3f}.",
        "",
        "| Strategy | Targeted | Net value | Share of churners reached |",
        "|---|---|---|---|",
    ]
    for n, s in v["strategies"].items():
        reached = s["share_of_churners_reached"]
        lines.append(f"| {n} | {s['n_targeted']} | {s['net_value']:,.0f} | {reached:.1%} |")
    return "\n".join(lines) + "\n"
