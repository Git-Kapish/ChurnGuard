"""SHAP explanations of the base (uncalibrated) estimator, aggregated to customer attributes.

Calibration pitfall (CLAUDE.md): CalibratedClassifierCV wraps five fitted copies of the model,
and SHAP cannot explain that wrapper. We explain the single uncalibrated base pipeline in
log-odds space and take probabilities from the calibrated model separately. Calibration is a
monotone increasing map, so "this factor raises the score" still means "raises churn risk",
but SHAP contributions are on the base model's log-odds scale, not on calibrated probabilities.
"""

import numpy as np
import pandas as pd
import shap
import xgboost
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from churnguard.explain.groups import PHRASES, assign_groups

N_INCREASING, N_DECREASING = 5, 3


class Explainer:
    """Wraps a fitted base pipeline: engineer -> columns -> classifier."""

    def __init__(self, base: Pipeline, background: pd.DataFrame | None = None) -> None:
        """`background` (raw customer frame) is only needed for a logistic-regression base."""
        self.engineer = base.named_steps["engineer"]
        self.columns = base.named_steps["columns"]
        self.clf = base.named_steps["clf"]
        self.feature_names: list[str] = list(self.columns.get_feature_names_out())
        self.column_group = assign_groups(self.feature_names)
        if isinstance(self.clf, LogisticRegression):
            if background is None:
                raise ValueError("a logistic-regression base needs a background frame")
            self._shap = shap.LinearExplainer(self.clf, self.transform(background))
        else:
            self._shap = shap.TreeExplainer(self.clf)
        self._xgb_bias: float | None = None
        if isinstance(self.clf, XGBClassifier):
            # The bias term of XGBoost's own contributions is constant across rows. Use it, not
            # TreeExplainer.expected_value: the two differ, and only this one makes
            # base + sum(contributions) equal the model's raw output.
            blank = xgboost.DMatrix(
                np.zeros((1, len(self.feature_names))), feature_names=self.feature_names
            )
            contribs = self.clf.get_booster().predict(blank, pred_contribs=True)
            self._xgb_bias = float(contribs[0, -1])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Raw (normalized) customers -> the model's feature matrix."""
        return self.columns.transform(self.engineer.transform(X))

    @property
    def base_value(self) -> float:
        """Log-odds baseline: base_value + sum of a row's SHAP values = the model's raw output."""
        if self._xgb_bias is not None:
            return self._xgb_bias
        value = np.asarray(self._shap.expected_value).ravel()
        return float(value[-1])

    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-column SHAP values (log-odds of churn), rows aligned with X."""
        Xt = self.transform(X)
        if isinstance(self.clf, XGBClassifier):
            # XGBoost's built-in TreeSHAP: same values as shap.TreeExplainer, ~30x faster per call
            dm = xgboost.DMatrix(Xt, feature_names=self.feature_names)
            values = self.clf.get_booster().predict(dm, pred_contribs=True)[:, :-1]
            return pd.DataFrame(values, index=X.index, columns=self.feature_names)
        values = self._shap.shap_values(Xt)
        if isinstance(values, list):  # older SHAP: one array per class
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:  # newer SHAP for forests: (rows, features, classes)
            values = values[:, :, -1]
        return pd.DataFrame(values, index=X.index, columns=self.feature_names)

    def group_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values summed per customer attribute (see groups.py); rows aligned with X."""
        sv = self.shap_values(X)
        return sv.T.groupby(self.column_group).sum().T

    def reasons(self, X: pd.DataFrame) -> list[list[dict]]:
        """Plain-English reasons per customer: top 5 raising risk, then top 3 lowering it."""
        engineered = self.engineer.transform(X)
        contributions = self.group_contributions(X)
        out = []
        for i in range(len(X)):
            row, impact = engineered.iloc[i], contributions.iloc[i]
            up = impact[impact > 0].sort_values(ascending=False).head(N_INCREASING)
            down = impact[impact < 0].sort_values().head(N_DECREASING)
            out.append(
                [_reason(g, v, row, "raises") for g, v in up.items()]
                + [_reason(g, v, row, "lowers") for g, v in down.items()]
            )
        return out


def _reason(group: str, impact: float, row: pd.Series, verb: str) -> dict:
    return {
        "feature": group,
        "text": f"{PHRASES[group](row)} {verb} churn risk",
        "direction": "increases_risk" if verb == "raises" else "decreases_risk",
        "impact": round(float(impact), 4),
    }
