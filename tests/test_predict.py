import copy

import pandas as pd
import pytest

from churnguard import predict as predict_module
from churnguard.config import PROCESSED_DIR
from churnguard.data.split import load_split, split_xy
from churnguard.models.artifact import MODEL_PATH
from churnguard.predict import Predictor, get_predictor, risk_tier

TIERS = {"medium": 0.25, "high": 0.5}
needs_model = pytest.mark.skipif(
    not (MODEL_PATH.exists() and (PROCESSED_DIR / "val.parquet").exists()),
    reason="run `make data train` first",
)


@pytest.mark.parametrize(
    ("p", "tier"),
    [
        (0.0, "Low"),
        (0.2499, "Low"),
        (0.25, "Medium"),
        (0.5, "Medium"),
        (0.5001, "High"),
        (1.0, "High"),
    ],
)
def test_risk_tier_boundaries(p: float, tier: str) -> None:
    assert risk_tier(p, TIERS) == tier


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return get_predictor()


@pytest.fixture(scope="module")
def val_records() -> list[dict]:
    Xv, _ = split_xy(load_split("val"))
    return Xv.head(30).to_dict("records")


@needs_model
def test_prediction_shape_and_consistency(predictor, val_records) -> None:
    out = predictor.predict(val_records[:5])
    assert len(out) == 5
    for r in out:
        assert set(r) == {
            "customerID",
            "churn_probability",
            "risk_tier",
            "will_target",
            "top_reasons",
        }
        assert 0 <= r["churn_probability"] <= 1
        assert r["risk_tier"] == risk_tier(r["churn_probability"], predictor.tiers)
        assert r["will_target"] == (r["churn_probability"] >= predictor.threshold)
        assert 0 < len(r["top_reasons"]) <= 8
        assert {"text", "direction", "feature", "impact"} <= set(r["top_reasons"][0])


@needs_model
def test_batch_matches_one_by_one(predictor, val_records) -> None:
    batch = predictor.predict(val_records[:10])
    single = [predictor.predict([rec])[0] for rec in val_records[:10]]
    for b, s in zip(batch, single, strict=True):
        assert b["churn_probability"] == pytest.approx(s["churn_probability"], abs=1e-9)
        assert b["top_reasons"] == s["top_reasons"]


@needs_model
def test_module_level_predict_uses_the_shared_cached_predictor(val_records) -> None:
    assert get_predictor() is get_predictor()
    assert predict_module.predict(val_records[:1])[0]["churn_probability"] == pytest.approx(
        get_predictor().predict(val_records[:1])[0]["churn_probability"]
    )


@needs_model
def test_customer_id_is_echoed_and_optional(predictor, val_records) -> None:
    with_id = [{**val_records[0], "customerID": "ABC-1"}]
    assert predictor.predict(with_id)[0]["customerID"] == "ABC-1"
    assert predictor.predict(val_records[:1])[0]["customerID"] is None


@needs_model
def test_raw_dataset_vocabulary_scores_like_the_cleaned_one(predictor, val_records) -> None:
    rec = copy.deepcopy(val_records[0])
    rec.update(InternetService="No", PhoneService="No", MultipleLines="No")
    cleaned = {**rec, "OnlineSecurity": "No", "TechSupport": "No", "StreamingTV": "No"}
    raw = {
        **rec,
        "MultipleLines": "No phone service",
        "OnlineSecurity": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
    }
    cleaned["MultipleLines"] = "No"
    a, b = predictor.predict([cleaned])[0], predictor.predict([raw])[0]
    assert a["churn_probability"] == pytest.approx(b["churn_probability"], abs=1e-12)


@needs_model
def test_missing_total_charges_is_handled(predictor, val_records) -> None:
    new_customer = {**val_records[0], "tenure": 0, "TotalCharges": None}
    older = {**val_records[0], "tenure": 20, "TotalCharges": None}  # imputed by the pipeline
    for rec in (new_customer, older):
        assert 0 <= predictor.predict([rec])[0]["churn_probability"] <= 1


@needs_model
def test_missing_column_is_reported(predictor, val_records) -> None:
    rec = {k: v for k, v in val_records[0].items() if k != "Contract"}
    with pytest.raises(ValueError, match="Contract"):
        predictor.predict([rec])


@needs_model
def test_explain_false_skips_reasons_but_not_scores(predictor, val_records) -> None:
    fast = predictor.predict(val_records[:3], explain=False)
    full = predictor.predict(val_records[:3])
    assert all(r["top_reasons"] == [] for r in fast)
    assert [r["churn_probability"] for r in fast] == [r["churn_probability"] for r in full]


@needs_model
def test_accepts_a_dataframe_and_extra_columns(predictor) -> None:
    Xv, yv = split_xy(load_split("val"))
    df = Xv.head(4).assign(Churn=yv.head(4).to_numpy(), unrelated="x")  # extras are ignored
    assert len(predictor.predict(df)) == 4
    assert isinstance(pd.DataFrame(df), pd.DataFrame)


@needs_model
def test_probabilities_equal_the_calibrated_model_on_validation(predictor) -> None:
    """predict.py must not alter the score: same numbers as the artifact's own model."""
    Xv, _ = split_xy(load_split("val"))
    direct = predictor.model.predict_proba(predictor.prepare(Xv.head(40)))[:, 1]
    via = [r["churn_probability"] for r in predictor.predict(Xv.head(40), explain=False)]
    assert via == pytest.approx(list(direct), abs=1e-12)
