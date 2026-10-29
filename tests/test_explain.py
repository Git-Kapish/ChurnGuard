import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from churnguard.config import PROCESSED_DIR
from churnguard.data.clean import normalize_customers
from churnguard.data.split import load_split, split_xy
from churnguard.explain import report as report_module
from churnguard.explain.explainer import N_DECREASING, N_INCREASING, Explainer
from churnguard.explain.groups import GROUP_COLUMNS, PHRASES, assign_groups
from churnguard.features.pipeline import build_preprocessor
from churnguard.models.artifact import MODEL_PATH, load_bundle

needs_model = pytest.mark.skipif(
    not (MODEL_PATH.exists() and (PROCESSED_DIR / "val.parquet").exists()),
    reason="run `make data train` first",
)
HIGH_RISK = {
    "Contract": "Month-to-month",
    "tenure": 2,
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "OnlineSecurity": "No",
    "TechSupport": "No",
    "MonthlyCharges": 89.1,
    "TotalCharges": 178.2,
}
LOW_RISK = {
    "Contract": "Two year",
    "tenure": 60,
    "InternetService": "DSL",
    "PaymentMethod": "Bank transfer (automatic)",
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes",
    "MonthlyCharges": 60.0,
    "TotalCharges": 3600.0,
}


@pytest.fixture(scope="module")
def explainer_and_val():
    bundle = load_bundle()
    Xv, yv = split_xy(load_split("val"))
    return Explainer(bundle["base"]), bundle, normalize_customers(Xv), yv


def test_every_model_column_belongs_to_exactly_one_group(random_customers) -> None:
    X, _ = split_xy(random_customers())
    names = list(build_preprocessor().fit(X).named_steps["columns"].get_feature_names_out())
    mapping = assign_groups(names)
    assert set(mapping) == set(names) and set(mapping.values()) <= set(GROUP_COLUMNS)
    assert set(PHRASES) == set(GROUP_COLUMNS)  # every group has a plain-English phrase


def test_unmapped_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="matches groups"):
        assign_groups(["mystery_column"])


def test_phrases_use_the_customers_actual_values(make_customers) -> None:
    from churnguard.features.engineering import FeatureEngineer

    row = (
        FeatureEngineer()
        .transform(
            make_customers(
                {"tenure": 1, "Contract": "One year", "OnlineBackup": "Yes", "TotalCharges": 100.0}
            )
        )
        .iloc[0]
    )
    assert PHRASES["Tenure"](row) == "Tenure of 1 month"
    assert PHRASES["Contract"](row) == "One year contract"
    assert PHRASES["Online backup"](row) == "Having online backup"
    assert PHRASES["Online security"](row) == "No online security"
    assert PHRASES["Number of add-on services"](row) == "1 add-on service"
    assert (
        PHRASES["Billing history"](row)
        == "Total charges of $100 (current bill $50.00 below average)"
    )
    row["TotalCharges"] = np.nan
    assert PHRASES["Billing history"](row) == "Total charges not available"


@needs_model
def test_shap_is_additive_and_matches_the_base_model(explainer_and_val) -> None:
    ex, bundle, Xv, _ = explainer_and_val
    sample = Xv.iloc[:200]
    sv = ex.shap_values(sample)
    margin = sv.sum(axis=1) + ex.base_value
    from_shap = 1 / (1 + np.exp(-margin))
    direct = bundle["base"].predict_proba(sample)[:, 1]
    assert np.allclose(from_shap, direct, atol=1e-4)


@needs_model
def test_fast_xgboost_path_matches_shap_tree_explainer(explainer_and_val) -> None:
    ex, _, Xv, _ = explainer_and_val
    sample = Xv.iloc[:100]
    slow = ex._shap.shap_values(ex.transform(sample))
    assert np.allclose(ex.shap_values(sample).to_numpy(), slow, atol=1e-4)


@needs_model
def test_group_contributions_preserve_the_total(explainer_and_val) -> None:
    ex, _, Xv, _ = explainer_and_val
    sample = Xv.iloc[:50]
    assert np.allclose(
        ex.group_contributions(sample).sum(axis=1), ex.shap_values(sample).sum(axis=1), atol=1e-6
    )


@needs_model
def test_reasons_structure_and_direction(explainer_and_val) -> None:
    ex, _, Xv, _ = explainer_and_val
    all_reasons = ex.reasons(Xv.iloc[:100])
    for reasons in all_reasons:
        up = [r for r in reasons if r["direction"] == "increases_risk"]
        down = [r for r in reasons if r["direction"] == "decreases_risk"]
        assert len(up) <= N_INCREASING and len(down) <= N_DECREASING
        assert all(r["impact"] > 0 for r in up) and all(r["impact"] < 0 for r in down)
        assert all(r["text"].endswith(("raises churn risk", "lowers churn risk")) for r in reasons)
        assert [r["impact"] for r in up] == sorted((r["impact"] for r in up), reverse=True)


@needs_model
def test_reasons_for_obvious_customers(explainer_and_val) -> None:
    ex, _, Xv, _ = explainer_and_val
    base = Xv.iloc[[0]].reset_index(drop=True)
    hi = base.assign(**HIGH_RISK)
    lo = base.assign(**LOW_RISK)
    hi_texts = [r["text"] for r in ex.reasons(hi)[0] if r["direction"] == "increases_risk"]
    lo_texts = [r["text"] for r in ex.reasons(lo)[0] if r["direction"] == "decreases_risk"]
    assert "Month-to-month contract raises churn risk" in hi_texts
    assert "Tenure of 2 months raises churn risk" in hi_texts
    assert "Two year contract lowers churn risk" in lo_texts
    assert "Tenure of 60 months lowers churn risk" in lo_texts


@needs_model
def test_calibrated_and_base_probabilities_rank_customers_the_same_way(explainer_and_val) -> None:
    """Why explaining the base model is valid: calibration must not reorder customers much."""
    _, bundle, Xv, _ = explainer_and_val
    cal = bundle["calibrated"].predict_proba(Xv)[:, 1]
    base = bundle["base"].predict_proba(Xv)[:, 1]
    assert spearmanr(cal, base)[0] > 0.98


@needs_model
def test_linear_base_requires_a_background_frame(explainer_and_val) -> None:
    from churnguard.models.registry import build_estimator

    _, _, Xv, yv = explainer_and_val
    lr = build_estimator("logistic_regression").fit(Xv, yv)
    with pytest.raises(ValueError, match="background"):
        Explainer(lr)
    ex = Explainer(lr, background=Xv.iloc[:100])
    sv = ex.shap_values(Xv.iloc[:20])
    margin = sv.sum(axis=1) + ex.base_value
    assert np.allclose(1 / (1 + np.exp(-margin)), lr.predict_proba(Xv.iloc[:20])[:, 1], atol=1e-4)


def test_permutation_importance_by_group_finds_the_informative_group() -> None:
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n = 600
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "sig_a": signal,
            "sig_b": signal * 0.5,  # same group as sig_a
            "noise": rng.normal(size=n),
        }
    )
    y = (signal + 0.3 * rng.normal(size=n) > 0).astype(int)
    clf = LogisticRegression().fit(X, y)
    groups = {"sig_a": "signal", "sig_b": "signal", "noise": "noise"}
    imp = report_module.permutation_importance_by_group(clf, X, y, groups, n_repeats=5, seed=1)
    assert imp.index[0] == "signal" and imp["signal"] > 0.1 and abs(imp["noise"]) < 0.02
    again = report_module.permutation_importance_by_group(clf, X, y, groups, n_repeats=5, seed=1)
    pd.testing.assert_series_equal(imp, again)


def test_compare_rankings_identical_and_reversed() -> None:
    a = pd.Series({"x": 3.0, "y": 2.0, "z": 1.0, "w": 0.5, "v": 0.1})
    same = report_module.compare_rankings(a, a)
    flipped = report_module.compare_rankings(a, a[::-1].reset_index(drop=True).set_axis(a.index))
    assert same["spearman"] == pytest.approx(1.0) and same["big_gaps"].empty
    assert flipped["spearman"] == pytest.approx(-1.0)


@needs_model
def test_report_run_writes_figures_and_note_without_touching_test(tmp_path, monkeypatch) -> None:
    loaded = []
    real = report_module.load_split
    monkeypatch.setattr(report_module, "load_split", lambda n: loaded.append(n) or real(n))
    out = report_module.run(reports_dir=tmp_path, sample=150)
    assert sorted(set(loaded)) == ["train", "val"]
    for name in ("shap_beeswarm.png", "shap_top10_bar.png"):
        assert (tmp_path / "figures" / name).stat().st_size > 5_000
    note = (tmp_path / "explainability_notes.md").read_text(encoding="utf-8")
    assert "Spearman" in note and "calibrated" in note.lower()
    assert len(out["top_shap"]) == 10
