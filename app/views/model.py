"""Model page: how good the model is, honestly, and how it was chosen."""

import pandas as pd
import streamlit as st

import common
from churnguard.models.artifact import METADATA_PATH
from common import REPORTS

METRIC_LABELS = {
    "pr_auc": "PR-AUC (primary)",
    "roc_auc": "ROC-AUC",
    "recall": "Recall at threshold",
    "top_decile_lift": "Top-decile lift",
    "brier": "Brier score",
}


def _read_markdown(name: str) -> str | None:
    path = REPORTS / name
    return path.read_text(encoding="utf-8") if path.exists() else None


def render() -> None:
    """Draw the Model page."""
    st.title("Model")
    common.require(METADATA_PATH, "Run `make train` first.")
    meta = common.predictor().metadata
    test = common.read_json(REPORTS / "test_metrics.json")

    st.write(
        f"**{meta['model_name']}** with **{meta['calibration']}** probability calibration, "
        f"trained on {meta['train_rows']:,} customers. A customer is flagged for an offer when "
        f"the churn probability is at least **{meta['threshold']:.1%}**, the threshold that "
        "maximized simulated net value on the validation split."
    )

    if test is None:
        st.info("Test-split results appear after `make evaluate` has run.")
    else:
        st.subheader("Held-out test results")
        st.caption(
            f"{test['n']} customers scored once, with 95% bootstrap confidence intervals. "
            "The threshold, calibration and model were all chosen before the test split was read."
        )
        cols = st.columns(len(METRIC_LABELS))
        for col, (key, label) in zip(cols, METRIC_LABELS.items(), strict=True):
            lo, hi = test["bootstrap_95ci"][key]
            col.metric(label, f"{test['metrics'][key]:.3f}")
            col.caption(f"95% CI [{lo:.3f}, {hi:.3f}]")

        money = "net_value_vs_target_everyone"  # a simulated amount, not a rate: no decimals
        targets = pd.DataFrame(
            [
                {
                    "Target": f"{k} (assumption-driven)" if k == money else k,
                    "Result": f"{v['value']:+,.0f}" if k == money else f"{v['value']:.3f}",
                    "Goal": v["target"],
                    "Met": "yes" if v["met"] else "NO",
                }
                for k, v in test["prd_targets"].items()
            ]
        )
        st.dataframe(targets, hide_index=True, use_container_width=True)
        st.caption(
            "ROC-AUC clears its target on the point estimate, but its confidence interval "
            "extends below the target, so treat that one as borderline."
        )

    st.subheader("Curves")
    a, b = st.columns(2)
    with a:
        common.show_figure("test_roc.png", "ROC curve (test split)")
        common.show_figure(
            "test_calibration.png", "Reliability: predicted vs observed churn (test)"
        )
    with b:
        common.show_figure("test_pr.png", "Precision-recall curve (test split)")
        common.show_figure("test_lift.png", "Lift and cumulative gain (test split)")

    st.subheader("Why the model predicts what it does")
    a, b = st.columns(2)
    with a:
        common.show_figure("shap_top10_bar.png", "Top drivers (validation split)")
    with b:
        common.show_figure("shap_beeswarm.png", "SHAP values per model column (validation)")
    notes = _read_markdown("explainability_notes.md")
    if notes:
        with st.expander("SHAP vs permutation importance"):
            st.markdown(notes)

    if test:
        st.subheader("Performance by slice (test split)")
        tabs = st.tabs(["Contract", "Tenure", "Senior citizen"])
        for tab, key in zip(tabs, ["contract", "tenure_bucket", "senior_citizen"], strict=True):
            with tab:
                df = pd.DataFrame(test["slices"][key]).rename(columns={"group": key})
                st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption(
            "Small groups give noisy numbers: two-year contracts have only a handful of "
            "churners, so a recall of zero there is weak evidence either way."
        )

    st.subheader("Model comparison")
    comparison = _read_markdown("model_comparison.md")
    if comparison:
        with st.expander("Cross-validation comparison, imbalance study and tuning", expanded=True):
            st.markdown(comparison)
    summary = _read_markdown("validation_summary.md")
    if summary:
        with st.expander("Validation summary: winner, calibration, threshold"):
            st.markdown(summary)
    common.show_figure("profit_vs_threshold.png", "Net value vs threshold (validation split)")
    st.caption(common.ASSUMPTION_NOTE)
    with st.expander("Run details"):
        st.json(
            {
                k: meta[k]
                for k in (
                    "model_version",
                    "trained_at",
                    "git_hash",
                    "git_dirty",
                    "data_sha256",
                    "library_versions",
                )
            }
        )
