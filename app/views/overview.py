"""Overview page: how churn varies across the training customers (interactive charts)."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import common
from churnguard.data.validate import SERVICE_COLS
from common import BLUE, GREY, RED

EXPLORE = {
    "Contract": "Contract",
    "Tenure bucket": "tenure_bucket",
    "Internet service": "InternetService",
    "Payment method": "PaymentMethod",
    "Paperless billing": "PaperlessBilling",
    "Senior citizen": "Senior citizen",
    "Number of add-on services": "n_services",
    "Partner": "Partner",
    "Dependents": "Dependents",
}


def churn_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Customers and churn rate per value of `col`."""
    g = df.groupby(col, observed=True)["Churn"].agg(churn_rate="mean", customers="size")
    return g.reset_index()


def rate_bar(df: pd.DataFrame, col: str, title: str, order: list[str] | None = None) -> go.Figure:
    """Bar chart of churn rate per category, with the overall rate as a dashed line."""
    g = churn_by(df, col)
    g[col] = g[col].astype(str)
    fig = px.bar(
        g,
        x=col,
        y="churn_rate",
        text=g["churn_rate"].map("{:.0%}".format),
        hover_data={"customers": True, "churn_rate": ":.1%"},
        category_orders={col: order} if order else None,
        color_discrete_sequence=[BLUE],
    )
    fig.add_hline(
        y=df["Churn"].mean(),
        line_dash="dash",
        line_color=GREY,
        annotation_text="overall",
        annotation_position="top right",
    )
    fig.update_layout(
        title=title,
        yaxis_tickformat=".0%",
        yaxis_title="churn rate",
        xaxis_title=None,
        margin=dict(t=50, b=20),
    )
    return fig


def render() -> None:
    """Draw the Overview page."""
    st.title("Overview")
    st.caption(
        "Churn patterns in the 4,930 training-split customers (the validation and test splits "
        "are not used here). These are descriptive associations, not causal effects."
    )
    df = common.train_frame()

    with st.expander("Filter customers", expanded=False):
        c1, c2 = st.columns(2)
        contracts = c1.multiselect(
            "Contract", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique())
        )
        internet = c2.multiselect(
            "Internet service",
            sorted(df["InternetService"].unique()),
            default=sorted(df["InternetService"].unique()),
        )
    view = df[df["Contract"].isin(contracts) & df["InternetService"].isin(internet)]
    if view.empty:
        st.warning("No customers match the filter.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(view):,}")
    k2.metric("Churn rate", f"{view['Churn'].mean():.1%}")
    k3.metric("Median tenure", f"{view['tenure'].median():.0f} months")
    k4.metric("Avg monthly charges", f"${view['MonthlyCharges'].mean():.2f}")

    tab_contract, tab_tenure, tab_services, tab_explore = st.tabs(
        ["By contract", "By tenure", "By services", "Explore"]
    )
    with tab_contract:
        st.plotly_chart(
            rate_bar(
                view,
                "Contract",
                "Churn rate by contract type",
                ["Month-to-month", "One year", "Two year"],
            ),
            use_container_width=True,
        )
    with tab_tenure:
        st.plotly_chart(
            rate_bar(
                view,
                "tenure_bucket",
                "Churn rate by tenure (months)",
                ["0-12", "13-24", "25-48", "49+"],
            ),
            use_container_width=True,
        )
        window = view.assign(window=(view["tenure"] // 6) * 6)
        g = churn_by(window, "window")
        line = px.line(
            g,
            x="window",
            y="churn_rate",
            markers=True,
            hover_data={"customers": True, "churn_rate": ":.1%"},
            color_discrete_sequence=[RED],
        )
        line.update_layout(
            title="Churn rate by 6-month tenure window",
            yaxis_tickformat=".0%",
            xaxis_title="tenure window start (months)",
            yaxis_title="churn rate",
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(line, use_container_width=True)
    with tab_services:
        rows = []
        for col in SERVICE_COLS:
            for has in ("Yes", "No"):
                sub = view[view[col] == has]
                if len(sub):
                    rows.append(
                        {
                            "service": col,
                            "has service": has,
                            "churn_rate": sub["Churn"].mean(),
                            "customers": len(sub),
                        }
                    )
        fig = px.bar(
            pd.DataFrame(rows),
            x="service",
            y="churn_rate",
            color="has service",
            barmode="group",
            hover_data={"customers": True, "churn_rate": ":.1%"},
            color_discrete_map={"Yes": BLUE, "No": RED},
        )
        fig.update_layout(
            title="Churn rate with and without each add-on service",
            yaxis_tickformat=".0%",
            yaxis_title="churn rate",
            xaxis_title=None,
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(
            rate_bar(view, "InternetService", "Churn rate by internet service"),
            use_container_width=True,
        )
    with tab_explore:
        label = st.selectbox("Break churn down by", list(EXPLORE))
        col = EXPLORE[label]
        st.plotly_chart(
            rate_bar(view, col, f"Churn rate by {label.lower()}"), use_container_width=True
        )
        st.dataframe(
            churn_by(view, col).rename(columns={col: label}),
            hide_index=True,
            use_container_width=True,
        )
