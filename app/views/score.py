"""Score a customer: form -> probability gauge, plain-English reasons, SHAP waterfall."""

import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

import common
from churnguard.api.schemas import (
    AddOn,
    Contract,
    Customer,
    Gender,
    InternetService,
    MultipleLines,
    PaymentMethod,
    YesNo,
)
from churnguard.evaluation.cost import Business, expected_value
from churnguard.models.artifact import MODEL_PATH
from common import BLUE, GREEN, RED

ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
TYPICAL = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 55.0,
    "TotalCharges": 1320.0,
}
HIGH_RISK = {
    **TYPICAL,
    "Partner": "No",
    "tenure": 2,
    "InternetService": "Fiber optic",
    "OnlineBackup": "No",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.10,
    "TotalCharges": 178.20,
}
LOW_RISK = {
    **TYPICAL,
    "Dependents": "Yes",
    "tenure": 60,
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 60.0,
    "TotalCharges": 3600.0,
}
PRESETS = {
    "Typical customer": TYPICAL,
    "High-risk example": HIGH_RISK,
    "Low-risk example": LOW_RISK,
}


def _values(enum) -> list[str]:
    return [m.value for m in enum]


ADDON_LABELS = {
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
}


def _md(text: str) -> str:
    """Escape $ so Streamlit does not treat a pair of dollar amounts as inline math."""
    return text.replace("$", r"\$")


def gauge(probability: float, threshold: float, tiers: dict[str, float]) -> go.Figure:
    """Probability gauge with the risk-tier bands and the offer threshold marked."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Churn probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#333333", "thickness": 0.25},
                "steps": [
                    {"range": [0, tiers["medium"] * 100], "color": "#d9ead3"},
                    {"range": [tiers["medium"] * 100, tiers["high"] * 100], "color": "#fff2cc"},
                    {"range": [tiers["high"] * 100, 100], "color": "#f4cccc"},
                ],
                "threshold": {"line": {"color": BLUE, "width": 4}, "value": threshold * 100},
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=60, b=10, l=30, r=30))
    return fig


def waterfall(baseline: float, table, top: int = 8) -> go.Figure:
    """Waterfall from the average customer to this customer, in base-model log-odds."""
    head, tail = table.head(top), table.iloc[top:]
    labels = [
        "Average customer",
        *head["value"],
        *(["Other attributes"] if len(tail) else []),
        "This customer (before calibration)",
    ]
    values = [baseline, *head["impact"], *([tail["impact"].sum()] if len(tail) else []), None]
    measure = ["absolute", *["relative"] * (len(labels) - 2), "total"]
    text = [f"{baseline:+.2f}", *[f"{v:+.2f}" for v in values[1:-1]], ""]
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=measure,
            y=labels,
            x=values,
            text=text,
            increasing={"marker": {"color": RED}},
            decreasing={"marker": {"color": GREEN}},
            totals={"marker": {"color": BLUE}},
            connector={"line": {"color": "#bbbbbb"}},
        )
    )
    fig.update_layout(
        height=120 + 38 * len(labels),
        margin=dict(t=30, b=40, l=10, r=10),
        showlegend=False,
        xaxis_title="log-odds of churn (base model): red raises risk, green lowers it",
        yaxis={"autorange": "reversed"},
    )
    return fig


def _form(preset_name: str) -> dict | None:
    """Render the customer form; return the payload on submit."""
    p = PRESETS[preset_name]
    key = lambda f: f"{preset_name}|{f}"  # noqa: E731  (new preset -> fresh widgets)

    def pick(label: str, field: str, options: list[str], container) -> str:
        return container.selectbox(label, options, index=options.index(p[field]), key=key(field))

    with st.form("customer"):
        c1, c2, c3 = st.columns(3)
        payload = {
            "gender": pick("Gender", "gender", _values(Gender), c1),
            "SeniorCitizen": (
                1
                if c1.selectbox(
                    "Senior citizen", ["No", "Yes"], index=p["SeniorCitizen"], key=key("senior")
                )
                == "Yes"
                else 0
            ),
            "Partner": pick("Partner", "Partner", _values(YesNo), c1),
            "Dependents": pick("Dependents", "Dependents", _values(YesNo), c1),
            "tenure": int(
                c1.number_input("Tenure (months)", 0, 100, p["tenure"], key=key("tenure"))
            ),
            "PhoneService": pick("Phone service", "PhoneService", _values(YesNo), c1),
            "MultipleLines": pick("Multiple lines", "MultipleLines", _values(MultipleLines), c1),
            "InternetService": pick(
                "Internet service", "InternetService", _values(InternetService), c2
            ),
            **{f: pick(ADDON_LABELS[f], f, _values(AddOn), c2) for f in ADDONS},
            "Contract": pick("Contract", "Contract", _values(Contract), c3),
            "PaperlessBilling": pick("Paperless billing", "PaperlessBilling", _values(YesNo), c3),
            "PaymentMethod": pick("Payment method", "PaymentMethod", _values(PaymentMethod), c3),
            "MonthlyCharges": float(
                c3.number_input(
                    "Monthly charges ($)", 0.0, 250.0, p["MonthlyCharges"], 0.05, key=key("monthly")
                )
            ),
        }
        total = c3.number_input(
            "Total charges ($)", 0.0, 100000.0, p["TotalCharges"], 1.0, key=key("total")
        )
        blank = c3.checkbox("Total charges unknown (brand-new customer)", key=key("blank"))
        payload["TotalCharges"] = None if blank else float(total)
        submitted = st.form_submit_button("Score customer", type="primary")
    return payload if submitted else None


def render() -> None:
    """Draw the Score-a-customer page."""
    st.title("Score a customer")
    common.require(MODEL_PATH, "Run `make train` first.")
    predictor = common.predictor()
    st.caption(
        "Enter a customer's details. The probability comes from the calibrated model; the "
        "reasons are SHAP explanations of the underlying base model."
    )
    preset = st.radio("Start from", list(PRESETS), horizontal=True)
    payload = _form(preset)

    if payload is not None:
        try:
            record = Customer.model_validate(payload).model_dump(mode="json")
        except ValidationError as exc:
            st.error("; ".join(f"{e['loc'][0]}: {e['msg']}" for e in exc.errors()))
            return
        result = common.score([record])[0]
        baseline, table = common.attribute_table(record)
        st.session_state["scored"] = {
            "preset": preset,
            "result": result,
            "baseline": baseline,
            "table": table,
        }

    saved = st.session_state.get("scored")
    if not saved or saved["preset"] != preset:
        st.info("Press **Score customer** to see the result.")
        return

    result, p = saved["result"], saved["result"]["churn_probability"]
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(gauge(p, predictor.threshold, predictor.tiers), use_container_width=True)
        st.caption(
            f"Blue line: the {predictor.threshold:.1%} offer threshold. Bands: Low below "
            f"{predictor.tiers['medium']:.0%}, Medium up to {predictor.tiers['high']:.0%}, "
            "High above."
        )
    with right:
        st.metric("Risk tier", result["risk_tier"])
        if result["will_target"]:
            st.success("Recommended for a retention offer (probability at or above the threshold).")
        else:
            st.info("Not recommended for an offer (probability below the threshold).")
        business = Business.from_config()
        ev = float(expected_value([p], business)[0])
        st.metric("Expected net value of an offer", f"{ev:+,.0f}")
        st.caption(common.ASSUMPTION_NOTE)

    up = [r for r in result["top_reasons"] if r["direction"] == "increases_risk"]
    down = [r for r in result["top_reasons"] if r["direction"] == "decreases_risk"]
    a, b = st.columns(2)
    with a:
        st.subheader("Raises churn risk")
        for r in up or [{"text": "Nothing raises this customer's risk above average."}]:
            st.markdown(f"- :red[▲] {_md(r['text'].removesuffix(' raises churn risk'))}")
    with b:
        st.subheader("Lowers churn risk")
        for r in down or [{"text": "Nothing lowers this customer's risk below average."}]:
            st.markdown(f"- :green[▼] {_md(r['text'].removesuffix(' lowers churn risk'))}")

    st.subheader("How the score builds up")
    st.plotly_chart(waterfall(saved["baseline"], saved["table"]), use_container_width=True)
    st.caption(
        "The waterfall is in the base model's log-odds, before calibration. Calibration is a "
        "monotone adjustment, so a bar that raises the score also raises the calibrated "
        "probability, but the bar lengths are not percentage points."
    )
