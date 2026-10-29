"""Map model columns to the original customer attributes, and phrase each in plain English.

The model sees 36 transformed columns (one-hot pieces, engineered features). A reason like
"Contract_Two year = 0 lowers risk" is unreadable, so contributions are summed per group
and described using the customer's actual values.

Phrase functions take a row of the *engineered* frame (FeatureEngineer output), which keeps
every raw column and adds n_services, has_support_bundle, charge_delta, etc.
"""

from collections.abc import Callable

import pandas as pd

# group -> column-name prefixes; a column matches when it equals a prefix or starts with prefix_
GROUP_COLUMNS: dict[str, list[str]] = {
    "Tenure": ["tenure"],
    "Monthly charges": ["MonthlyCharges"],
    "Billing history": ["TotalCharges", "avg_monthly_spend", "charge_delta"],
    "Number of add-on services": ["n_services"],
    "Senior citizen": ["SeniorCitizen"],
    "Contract": ["is_month_to_month", "Contract"],
    "Security or support add-on": ["has_support_bundle"],
    "Payment method": ["auto_pay", "PaymentMethod"],
    "Gender": ["gender"],
    "Partner": ["Partner"],
    "Dependents": ["Dependents"],
    "Phone service": ["PhoneService"],
    "Multiple lines": ["MultipleLines"],
    "Internet service": ["InternetService"],
    "Online security": ["OnlineSecurity"],
    "Online backup": ["OnlineBackup"],
    "Device protection": ["DeviceProtection"],
    "Tech support": ["TechSupport"],
    "Streaming TV": ["StreamingTV"],
    "Streaming movies": ["StreamingMovies"],
    "Paperless billing": ["PaperlessBilling"],
}


def assign_groups(columns: list[str]) -> dict[str, str]:
    """Return {column: group}. Raises if a column matches no group or more than one."""
    out: dict[str, str] = {}
    for col in columns:
        hits = [
            g
            for g, prefixes in GROUP_COLUMNS.items()
            if any(col == p or col.startswith(p + "_") for p in prefixes)
        ]
        if len(hits) != 1:
            raise ValueError(f"column {col!r} matches groups {hits}; expected exactly one")
        out[col] = hits[0]
    return out


def _yes_no(col: str, yes: str, no: str) -> Callable[[pd.Series], str]:
    return lambda r: yes if r[col] == "Yes" else no


def _tenure(r: pd.Series) -> str:
    months = int(r["tenure"])
    return f"Tenure of {months} month{'' if months == 1 else 's'}"


def _billing(r: pd.Series) -> str:
    if pd.isna(r["TotalCharges"]):
        return "Total charges not available"
    delta = r["charge_delta"]
    side = "above" if delta >= 0 else "below"
    gap = f"current bill ${abs(delta):.2f} {side} average"
    return f"Total charges of ${r['TotalCharges']:,.0f} ({gap})"


def _addons(r: pd.Series) -> str:
    n = int(r["n_services"])
    return "No add-on services" if n == 0 else f"{n} add-on service{'' if n == 1 else 's'}"


def _senior(r: pd.Series) -> str:
    return "Being a senior citizen" if r["SeniorCitizen"] == 1 else "Not being a senior citizen"


def _support(r: pd.Series) -> str:
    if r["has_support_bundle"] == 1:
        return "Having online security or tech support"
    return "No online security or tech support"


def _internet(r: pd.Series) -> str:
    kind = r["InternetService"]
    return "No internet service" if kind == "No" else f"{kind} internet service"


# group -> function(engineered customer row) -> noun phrase; the reason adds "raises/lowers"
PHRASES: dict[str, Callable[[pd.Series], str]] = {
    "Tenure": _tenure,
    "Monthly charges": lambda r: f"Monthly charges of ${r['MonthlyCharges']:.2f}",
    "Billing history": _billing,
    "Number of add-on services": _addons,
    "Senior citizen": _senior,
    "Contract": lambda r: f"{r['Contract']} contract",
    "Security or support add-on": _support,
    "Payment method": lambda r: f"Paying by {str(r['PaymentMethod']).lower()}",
    "Gender": lambda r: f"Gender ({str(r['gender']).lower()})",
    "Partner": _yes_no("Partner", "Having a partner", "No partner"),
    "Dependents": _yes_no("Dependents", "Having dependents", "No dependents"),
    "Phone service": _yes_no("PhoneService", "Phone service", "No phone service"),
    "Multiple lines": _yes_no("MultipleLines", "Multiple phone lines", "No multiple phone lines"),
    "Internet service": _internet,
    "Online security": _yes_no("OnlineSecurity", "Having online security", "No online security"),
    "Online backup": _yes_no("OnlineBackup", "Having online backup", "No online backup"),
    "Device protection": _yes_no(
        "DeviceProtection", "Having device protection", "No device protection"
    ),
    "Tech support": _yes_no("TechSupport", "Having tech support", "No tech support"),
    "Streaming TV": _yes_no("StreamingTV", "Streaming TV", "No streaming TV"),
    "Streaming movies": _yes_no("StreamingMovies", "Streaming movies", "No streaming movies"),
    "Paperless billing": _yes_no("PaperlessBilling", "Paperless billing", "Paper billing"),
}
