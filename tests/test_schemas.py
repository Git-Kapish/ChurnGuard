import pytest
from pydantic import ValidationError

from churnguard.api.schemas import MAX_BATCH, BatchRequest, Customer, Prediction


def test_valid_customer_parses_and_total_charges_is_optional(payload) -> None:
    del payload["TotalCharges"]
    c = Customer(**payload)
    assert c.TotalCharges is None and c.Contract == "Month-to-month"


def test_raw_vocabulary_is_accepted(payload) -> None:
    c = Customer(
        **{**payload, "MultipleLines": "No phone service", "TechSupport": "No internet service"}
    )
    assert c.MultipleLines == "No phone service"


@pytest.mark.parametrize(
    "override",
    [
        {"Contract": "Weekly"},
        {"gender": "Other"},
        {"SeniorCitizen": 2},
        {"tenure": -1},
        {"MonthlyCharges": 1000},
        {"TotalCharges": -5},
        {"surprise": 1},
    ],
)
def test_invalid_customer_rejected(payload, override: dict) -> None:
    with pytest.raises(ValidationError):
        Customer(**{**payload, **override})


def test_missing_field_rejected(payload) -> None:
    del payload["Contract"]
    with pytest.raises(ValidationError):
        Customer(**payload)


def test_batch_limits(payload) -> None:
    assert MAX_BATCH == 1000
    BatchRequest(customers=[payload] * MAX_BATCH)
    with pytest.raises(ValidationError):
        BatchRequest(customers=[payload] * (MAX_BATCH + 1))
    with pytest.raises(ValidationError):
        BatchRequest(customers=[])


def test_prediction_probability_bounds() -> None:
    ok = {"churn_probability": 0.4, "risk_tier": "Medium", "will_target": True, "top_reasons": []}
    Prediction(**ok)
    with pytest.raises(ValidationError):
        Prediction(**{**ok, "churn_probability": 1.2})
