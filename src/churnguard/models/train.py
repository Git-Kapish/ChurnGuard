"""Block 3 pipeline: CV model comparison -> imbalance study -> Optuna on the top models.

Uses the training split only. The validation split is reserved for Block 4 and the test
split is never loaded here.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from churnguard.config import ROOT, load_config
from churnguard.data.split import load_split, split_xy
from churnguard.explain.report import run as run_explain
from churnguard.models import tracking
from churnguard.models.cv import cross_validate, make_folds, summarize
from churnguard.models.finalize import run_final
from churnguard.models.registry import build_estimator
from churnguard.models.tune import tune

log = logging.getLogger("churnguard.train")
REPORTS_DIR = ROOT / "reports"


def _evaluate(
    name: str, imbalance: str, X, y, folds, seed: int, stage: str
) -> dict[str, float]:  # noqa: ANN001
    """CV one configuration and log it to MLflow; returns the mean/std summary."""
    start = time.time()
    scores = cross_validate(build_estimator(name, imbalance, seed=seed), X, y, folds)
    summary = summarize(scores)
    params = {"model": name, "imbalance": imbalance, "stage": stage, "n_splits": len(folds)}
    with tracking.run(f"{stage}-{name}-{imbalance}", params):
        tracking.log_cv(summary, scores)
    log.info(
        "%-10s %-19s %-12s PR-AUC %.4f  ROC-AUC %.4f  (%.1fs)",
        stage,
        name,
        imbalance,
        summary["pr_auc_mean"],
        summary["roc_auc_mean"],
        time.time() - start,
    )
    return summary


def run_pipeline(
    sample: int | None = None,
    n_trials: int | None = None,
    reports_dir: Path = REPORTS_DIR,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Run comparison, imbalance study and tuning; write the report and tuning results."""
    cfg = load_config("model")
    seed = cfg["seed"]
    n_trials = n_trials or cfg["tuning"]["n_trials"]
    tracking.setup(tracking_uri or cfg["mlflow"]["tracking_uri"], cfg["mlflow"]["experiment_name"])

    X, y = split_xy(load_split("train"))
    if sample:
        X, _, y, _ = train_test_split(X, y, train_size=sample, stratify=y, random_state=seed)
    folds = make_folds(y, cfg["cv"]["n_splits"], seed)
    log.info("train rows: %d, churn rate %.3f, folds: %d", len(y), y.mean(), len(folds))

    models = cfg["models"]
    comparison = {m: _evaluate(m, "class_weight", X, y, folds, seed, "compare") for m in models}

    candidates = [m for m in models if m != "dummy"]
    smote = {m: _evaluate(m, "smote", X, y, folds, seed, "imbalance") for m in candidates}
    min_gain = cfg["imbalance"]["smote_min_gain"]
    strategy, best = {}, {}
    for m in candidates:
        gain = smote[m]["pr_auc_mean"] - comparison[m]["pr_auc_mean"]
        strategy[m] = "smote" if gain > min_gain else "class_weight"
        best[m] = smote[m] if strategy[m] == "smote" else comparison[m]

    ranked = sorted(candidates, key=lambda m: best[m]["pr_auc_mean"], reverse=True)
    top = ranked[: cfg["tuning"]["n_top_models"]]
    tuned = {m: tune(m, strategy[m], X, y, folds, n_trials, seed) for m in top}

    result = {
        "comparison": comparison,
        "smote": smote,
        "strategy": strategy,
        "ranked": ranked,
        "tuned": {m: {"imbalance": strategy[m], "baseline": best[m], **tuned[m]} for m in top},
        "meta": {
            "n_rows": len(y),
            "n_splits": len(folds),
            "n_trials": n_trials,
            "seed": seed,
            "smote_min_gain": min_gain,
            "data_sha256": tracking.data_checksum(),
            "git_hash": tracking.git_hash(),
        },
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "tuning_results.json").write_text(
        json.dumps(result["tuned"], indent=2), encoding="utf-8"
    )
    (reports_dir / "model_comparison.md").write_text(render_report(result), encoding="utf-8")
    log.info("wrote %s", reports_dir / "model_comparison.md")
    return result


def _fmt(summary: dict[str, float], metric: str) -> str:
    return f"{summary[f'{metric}_mean']:.3f} ± {summary[f'{metric}_std']:.3f}"


def _table(rows: dict[str, dict[str, float]]) -> list[str]:
    head = "| Model | PR-AUC | ROC-AUC | F1 | Recall | Brier |"
    lines = [head, "|---|---|---|---|---|---|"]
    for name, s in rows.items():
        cells = [_fmt(s, m) for m in ("pr_auc", "roc_auc", "f1", "recall", "brier")]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def render_report(result: dict[str, Any]) -> str:
    """Render the comparison, imbalance study and tuning results as Markdown."""
    meta = result["meta"]
    out = [
        "# Model comparison (training split, cross-validation)",
        "",
        f"{meta['n_splits']}-fold stratified CV on {meta['n_rows']} training rows, seed "
        f"{meta['seed']}, identical folds for every model. Values are mean ± std across folds. "
        f"Data SHA256 `{meta['data_sha256'][:12]}`, git `{meta['git_hash']}`.",
        "",
        "PR-AUC is the primary metric. F1 and recall use a 0.5 threshold, which is not the "
        "business threshold (chosen in Block 4). Class-weighted models are not calibrated, so "
        "their Brier scores are pessimistic until calibration. The validation and test splits "
        "were not used here.",
        "",
        "## 1. Model comparison (class weights)",
        "",
        *_table(result["comparison"]),
        "",
        "## 2. Imbalance study: class weights vs SMOTE",
        "",
        f"SMOTE is applied inside the CV pipeline (training folds only). It is adopted for a "
        f"model only if it improves mean PR-AUC by more than {meta['smote_min_gain']}.",
        "",
        "| Model | Class weights PR-AUC | SMOTE PR-AUC | Difference | Used |",
        "|---|---|---|---|---|",
    ]
    for m, s in result["smote"].items():
        cw = result["comparison"][m]
        diff = s["pr_auc_mean"] - cw["pr_auc_mean"]
        out.append(
            f"| {m} | {_fmt(cw, 'pr_auc')} | {_fmt(s, 'pr_auc')} | {diff:+.4f} | "
            f"{result['strategy'][m]} |"
        )
    out += [
        "",
        f"## 3. Optuna tuning ({meta['n_trials']} trials each, top models by CV PR-AUC)",
        "",
        "Tuned scores are selected on the same CV folds, so they are optimistically biased; "
        "Block 4 compares the tuned models on the validation split.",
        "",
        "| Model | Imbalance | Default PR-AUC | Tuned PR-AUC | Tuned ROC-AUC | Tuned Brier |",
        "|---|---|---|---|---|---|",
    ]
    for m, t in result["tuned"].items():
        out.append(
            f"| {m} | {t['imbalance']} | {_fmt(t['baseline'], 'pr_auc')} | "
            f"{_fmt(t['summary'], 'pr_auc')} | {_fmt(t['summary'], 'roc_auc')} | "
            f"{_fmt(t['summary'], 'brier')} |"
        )
    out += ["", "Best parameters:", ""]
    for m, t in result["tuned"].items():
        out.append(f"- **{m}**: `{json.dumps(t['best_params'], sort_keys=True)}`")
    return "\n".join(out) + "\n"


def main() -> None:
    """CLI entrypoint (`make train`)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, help="use a stratified sample of N train rows")
    parser.add_argument("--trials", type=int, help="override Optuna trials per model")
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="reuse reports/tuning_results.json; only select, calibrate and save",
    )
    args = parser.parse_args()
    if not args.skip_search:
        run_pipeline(sample=args.sample, n_trials=args.trials)
    run_final(sample=args.sample)
    run_explain()  # SHAP figures and the SHAP-vs-permutation note for the model just saved


if __name__ == "__main__":
    main()
