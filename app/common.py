"""Shared loaders for the Streamlit pages. Heavy objects are cached once per server process."""

import json
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from churnguard.api.csv_input import parse_customers_csv
from churnguard.config import PROCESSED_DIR, ROOT
from churnguard.data.split import load_split
from churnguard.data.validate import SERVICE_COLS
from churnguard.features.engineering import TENURE_BINS, TENURE_LABELS
from churnguard.predict import Predictor, get_predictor

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
BLUE, RED, GREEN, GREY = "#4c78a8", "#e45756", "#54a24b", "#8a8a8a"
ASSUMPTION_NOTE = (
    "Assumption-driven simulation: offer cost, success rate and customer lifetime value are "
    "assumptions (configs/business.yaml), not measured facts. Treat money figures as "
    "illustrations of the method."
)
_SCORE_LOCK = threading.Lock()  # XGBoost/sklearn scoring from several sessions: one at a time


def require(path: Path, hint: str) -> None:
    """Stop the page with a clear message when a needed artifact is missing."""
    if not path.exists():
        st.error(f"Missing `{path.relative_to(ROOT)}`. {hint}")
        st.stop()


@st.cache_resource(show_spinner="Loading the model...")
def predictor() -> Predictor:
    """The shared Predictor (same object the API uses), loaded once per process."""
    return get_predictor()


def score(records: list[dict] | pd.DataFrame, explain: bool = True) -> list[dict[str, Any]]:
    """Score customers through predict.py (the single inference path)."""
    with _SCORE_LOCK:
        return predictor().predict(records, explain=explain)


def attribute_table(record: dict) -> tuple[float, pd.DataFrame]:
    """Per-attribute SHAP contributions for one customer (see Predictor.attribute_table)."""
    with _SCORE_LOCK:
        return predictor().attribute_table(record)


@st.cache_data(show_spinner=False)
def train_frame() -> pd.DataFrame:
    """Training split plus the columns the Overview charts group by."""
    require(PROCESSED_DIR / "train.parquet", "Run `make data` first.")
    df = load_split("train")
    df["tenure_bucket"] = pd.cut(df["tenure"], TENURE_BINS, labels=TENURE_LABELS).astype(str)
    df["n_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["Senior citizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def _pool_from_results(ids: list, results: list[dict], churn: pd.Series | None) -> pd.DataFrame:
    pool = pd.DataFrame(
        {
            "customerID": [i if i else f"row {n + 1}" for n, i in enumerate(ids)],
            "churn_probability": [r["churn_probability"] for r in results],
            "risk_tier": [r["risk_tier"] for r in results],
            "top_reason": [
                r["top_reasons"][0]["text"] if r["top_reasons"] else "" for r in results
            ],
        }
    )
    if churn is not None:
        pool["churned"] = churn.to_numpy()
    return pool


@st.cache_data(show_spinner="Scoring the validation customers (one-off, about 3 seconds)...")
def validation_pool() -> pd.DataFrame:
    """Validation split scored by the model, with known outcomes.

    Validation (not test) is used: the test split is reserved for the single final evaluation.
    """
    require(PROCESSED_DIR / "val.parquet", "Run `make data` first.")
    val = load_split("val")
    results = score(val.drop(columns=["Churn"]), explain=True)
    return _pool_from_results(val["customerID"].tolist(), results, val["Churn"])


@st.cache_data(show_spinner="Scoring the uploaded customers...")
def uploaded_pool(raw: bytes) -> pd.DataFrame:
    """Score an uploaded customers CSV (same validation rules as the API). Raises CsvInputError."""
    df, records = parse_customers_csv(raw)
    ids = df["customerID"].tolist() if "customerID" in df.columns else [""] * len(df)
    results = score(records, explain=True)
    churn = None
    if "Churn" in df.columns and df["Churn"].isin(["Yes", "No"]).all():
        churn = (df["Churn"] == "Yes").astype(int)
    return _pool_from_results(ids, results, churn)


def read_json(path: Path) -> dict | None:
    """Read a JSON file, or None if it does not exist."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def show_figure(name: str, caption: str) -> None:
    """Show reports/figures/<name> if present."""
    path = FIGURES / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{name} not found; run `make train` / `make evaluate`.")
