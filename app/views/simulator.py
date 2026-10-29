"""Campaign simulator: who to contact under a budget, and what it is worth (an assumption)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common
from churnguard.api.csv_input import MAX_CSV_BYTES, CsvInputError
from churnguard.evaluation.cost import (
    Business,
    cumulative_expected_value,
    expected_value,
    plan_campaign,
    realized_value,
)
from churnguard.models.artifact import MODEL_PATH
from common import BLUE, GREY, RED

VALIDATION = "Validation sample (1,056 customers, outcomes known)"
UPLOAD = "Upload a CSV of customers"


def _assumptions() -> Business:
    """Sliders for the four business assumptions, starting from configs/business.yaml."""
    base = Business.from_config()
    c1, c2, c3, c4 = st.columns(4)
    budget = c1.slider("Campaign budget", 0, 20_000, int(base.budget), 250)
    cost = c2.slider("Offer cost per customer", 1, 100, int(base.offer_cost), 1)
    success = c3.slider("Offer success rate", 0.05, 1.0, float(base.success_rate), 0.05)
    clv = c4.slider("Customer lifetime value", 50, 2000, int(base.clv), 25)
    return Business(offer_cost=cost, success_rate=success, clv=clv, budget=budget)


def _pool() -> pd.DataFrame | None:
    source = st.radio("Customers to rank", [VALIDATION, UPLOAD], horizontal=True)
    if source == VALIDATION:
        return common.validation_pool()
    upload = st.file_uploader("CSV with the Telco customer columns (up to 1,000 rows)", type="csv")
    if upload is None:
        st.info("Upload a CSV to rank your own customers.")
        return None
    raw = upload.getvalue()
    if len(raw) > MAX_CSV_BYTES:
        st.error(f"File is larger than {MAX_CSV_BYTES:,} bytes.")
        return None
    try:
        return common.uploaded_pool(raw)
    except CsvInputError as exc:
        detail = exc.detail if isinstance(exc.detail, list) else [exc.detail]
        st.error("Could not use that CSV:\n\n" + "\n".join(f"- {d}" for d in detail))
        return None


def value_curve(proba: np.ndarray, b: Business, k: int) -> go.Figure:
    """Cumulative model-expected net value as more customers are contacted, highest risk first."""
    curve = cumulative_expected_value(proba, b)
    x = np.arange(1, len(curve) + 1)
    fig = go.Figure(
        go.Scatter(x=x, y=curve, mode="lines", line={"color": BLUE}, name="expected net value")
    )
    fig.add_hline(y=0, line_color=GREY, line_width=1)
    if 0 < k <= len(curve):
        fig.add_trace(
            go.Scatter(
                x=[k],
                y=[curve[k - 1]],
                mode="markers",
                name="this campaign",
                marker={"color": RED, "size": 11},
            )
        )
    fig.update_layout(
        title="Expected net value vs customers contacted (highest risk first)",
        xaxis_title="customers contacted",
        yaxis_title="expected net value (assumption-driven)",
        margin=dict(t=50, b=30),
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


def render() -> None:
    """Draw the Campaign simulator page."""
    st.title("Campaign simulator")
    common.require(MODEL_PATH, "Run `make train` first.")
    st.warning(common.ASSUMPTION_NOTE)

    b = _assumptions()
    st.caption(
        f"An offer pays off when churn probability exceeds **{b.breakeven_probability:.1%}** "
        f"(cost / (success rate x lifetime value)). The budget pays for at most "
        f"**{b.max_targets:,}** offers."
    )
    pool = _pool()
    if pool is None:
        return

    p = pool["churn_probability"].to_numpy()
    y = pool["churned"].to_numpy() if "churned" in pool else None
    plan = plan_campaign(p, b, y)
    ev_all = float(expected_value(p, b).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Customers to contact",
        f"{plan.n_targeted:,}",
        help="Highest risk first, limited "
        "by the budget and by customers whose expected value is positive.",
    )
    k2.metric("Spend", f"{plan.spend:,.0f}")
    k3.metric("Expected net value", f"{plan.expected_net_value:+,.0f}")
    k4.metric("Expected customers saved", f"{plan.expected_saves:.1f}")
    st.caption("All values above are model-expected and assumption-driven. They are not forecasts.")

    if y is not None:
        a, c = st.columns(2)
        a.metric("Actual churners on the list", f"{plan.churners_reached} of {int(y.sum())}")
        c.metric("Net value using actual outcomes", f"{plan.realized_net_value:+,.0f}")
        if pool is not None and len(pool) == 1056:
            st.caption(
                "Validation customers were used to pick the production threshold, so "
                "this is slightly optimistic. The test split is untouched here."
            )

    rows = [
        ("Contact nobody", 0, 0.0, 0.0 if y is not None else None),
        (
            "Contact everyone",
            len(p),
            ev_all,
            float(realized_value(y, b).sum()) if y is not None else None,
        ),
        (
            "This campaign (budget-limited)",
            plan.n_targeted,
            plan.expected_net_value,
            plan.realized_net_value,
        ),
    ]
    table = pd.DataFrame(
        rows, columns=["Strategy", "Customers", "Expected net value", "Net value (actual outcomes)"]
    )
    if y is None:
        table = table.drop(columns="Net value (actual outcomes)")
    st.dataframe(
        table.style.format({c: "{:+,.0f}" for c in table.columns[2:]}),
        hide_index=True,
        use_container_width=True,
    )

    st.plotly_chart(value_curve(p, b, plan.n_targeted), use_container_width=True)

    st.subheader("Target list")
    targets = pool.iloc[plan.targets].copy()
    targets.insert(0, "rank", np.arange(1, len(targets) + 1))
    targets["expected_net_value"] = expected_value(targets["churn_probability"].to_numpy(), b)
    shown = targets.drop(columns=["churned"], errors="ignore")
    st.dataframe(
        shown,
        hide_index=True,
        use_container_width=True,
        column_config={
            "churn_probability": st.column_config.ProgressColumn(
                "churn probability", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "expected_net_value": st.column_config.NumberColumn(
                "expected net value", format="%.0f"
            ),
            "top_reason": "top reason",
        },
    )
    st.download_button(
        "Download target list (CSV)",
        shown.to_csv(index=False),
        "campaign_targets.csv",
        "text/csv",
        disabled=shown.empty,
    )
