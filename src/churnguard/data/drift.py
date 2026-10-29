"""Population Stability Index (PSI) drift check between the training data and new customers.

PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%). Common reading:
< 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant shift (the flag threshold here).
This is a monitoring demonstration on a CSV, not a live service.
"""

import numpy as np
import pandas as pd

from churnguard.data.clean import normalize_customers

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
THRESHOLD = 0.2
N_BINS = 10
EPS = 1e-4  # share assigned to an empty bin so ln() stays finite


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI between two vectors of bin counts or shares (each is normalized to sum to 1)."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e, a = e / e.sum(), a / a.sum()
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(train: pd.Series, new: pd.Series, n_bins: int = N_BINS) -> float:
    """PSI with bins at the training quantiles; values outside the training range fall in the
    first or last bin."""
    edges = np.unique(np.quantile(train.dropna(), np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:  # near-constant feature: a single bin carries no information
        return 0.0
    inner = edges[1:-1]
    expected = np.bincount(
        np.searchsorted(inner, train.dropna(), side="right"), minlength=len(edges) - 1
    )
    actual = np.bincount(
        np.searchsorted(inner, new.dropna(), side="right"), minlength=len(edges) - 1
    )
    return psi(expected, actual)


def psi_categorical(train: pd.Series, new: pd.Series) -> float:
    """PSI over category shares; a category unseen in training counts as an empty expected bin."""
    cats = sorted(set(train.dropna().astype(str)) | set(new.dropna().astype(str)))
    expected = train.astype(str).value_counts().reindex(cats, fill_value=0)
    actual = new.astype(str).value_counts().reindex(cats, fill_value=0)
    return psi(expected.to_numpy(), actual.to_numpy())


def drift_report(
    train: pd.DataFrame, new: pd.DataFrame, threshold: float = THRESHOLD
) -> pd.DataFrame:
    """PSI per model input feature, largest first, with a `flag` column (PSI > threshold).

    Both frames are normalized the same way the model's inputs are (shared cleaning rules).
    """
    train, new = normalize_customers(train), normalize_customers(new)
    rows = [(f, "numeric", psi_numeric(train[f], new[f])) for f in NUMERIC_FEATURES]
    rows += [(f, "categorical", psi_categorical(train[f], new[f])) for f in CATEGORICAL_FEATURES]
    out = pd.DataFrame(rows, columns=["feature", "kind", "psi"])
    out["flag"] = out["psi"] > threshold
    return out.sort_values("psi", ascending=False).reset_index(drop=True)
