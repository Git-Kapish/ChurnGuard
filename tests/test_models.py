import json

import numpy as np
import pytest
from xgboost import XGBClassifier

from churnguard.config import PROCESSED_DIR
from churnguard.data.split import split_xy
from churnguard.models import train as train_module
from churnguard.models.cv import cross_validate, make_folds, summarize
from churnguard.models.registry import (
    IMBALANCE,
    MODEL_NAMES,
    BalancedXGBClassifier,
    build_estimator,
)


@pytest.fixture
def xy(random_customers):
    return split_xy(random_customers(n=300, seed=3))


@pytest.mark.parametrize("imbalance", IMBALANCE)
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts_probabilities(xy, name: str, imbalance: str) -> None:
    X, y = xy
    model = build_estimator(name, imbalance).fit(X, y)
    proba = model.predict_proba(X.iloc[:10])
    assert proba.shape == (10, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_smote_sits_after_preprocessing_and_is_inside_pipeline(xy) -> None:
    X, y = xy
    model = build_estimator("logistic_regression", "smote")
    assert [n for n, _ in model.steps] == ["engineer", "columns", "smote", "clf"]
    model.fit(X, y)
    # the sampler is skipped at predict time: one output row per input row
    assert model.predict_proba(X.iloc[:7]).shape == (7, 2)


def test_unknown_model_or_strategy_rejected() -> None:
    with pytest.raises(ValueError):
        build_estimator("svm")
    with pytest.raises(ValueError):
        build_estimator("xgboost", "undersample")


def test_balanced_xgb_raises_minority_probability() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    y = (rng.random(600) < 0.1).astype(int)
    plain = XGBClassifier(n_estimators=30, random_state=0).fit(X, y)
    balanced = BalancedXGBClassifier(n_estimators=30, random_state=0).fit(X, y)
    assert balanced.predict_proba(X)[:, 1].mean() > plain.predict_proba(X)[:, 1].mean()


def test_folds_are_identical_stratified_and_partition_the_data(xy) -> None:
    _, y = xy
    a, b = make_folds(y, 5, 42), make_folds(y, 5, 42)
    assert all((x[1] == z[1]).all() for x, z in zip(a, b, strict=True))
    val_idx = np.concatenate([v for _, v in a])
    assert sorted(val_idx) == list(range(len(y)))
    for train_idx, val in a:
        assert not set(train_idx) & set(val)
        assert abs(y.iloc[val].mean() - y.mean()) < 0.05


def test_cross_validate_and_summarize(xy) -> None:
    X, y = xy
    folds = make_folds(y, 3, 42)
    scores = cross_validate(build_estimator("logistic_regression"), X, y, folds)
    assert len(scores) == 3 and not scores.isna().any().any()
    summary = summarize(scores)
    assert summary["pr_auc_mean"] == pytest.approx(scores["pr_auc"].mean())
    assert summary["pr_auc_std"] == pytest.approx(scores["pr_auc"].std(ddof=1))


@pytest.mark.skipif(not (PROCESSED_DIR / "train.parquet").exists(), reason="run `make data` first")
def test_pipeline_smoke_uses_only_train_split(tmp_path, monkeypatch) -> None:
    loaded = []
    real_load = train_module.load_split
    monkeypatch.setattr(train_module, "load_split", lambda n: loaded.append(n) or real_load(n))

    result = train_module.run_pipeline(
        sample=400,
        n_trials=2,
        reports_dir=tmp_path / "reports",
        tracking_uri=f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}",
    )

    assert loaded == ["train"]
    assert set(result["comparison"]) == {
        "dummy",
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
    }
    assert len(result["tuned"]) == 2 and set(result["tuned"]) <= set(result["ranked"][:2])
    assert result["comparison"]["dummy"]["roc_auc_mean"] == pytest.approx(0.5)
    report = (tmp_path / "reports" / "model_comparison.md").read_text()
    assert "Imbalance study" in report and "Optuna" in report
    tuned = json.loads((tmp_path / "reports" / "tuning_results.json").read_text())
    assert all("best_params" in v for v in tuned.values())
    assert any((tmp_path / "mlruns").iterdir())
