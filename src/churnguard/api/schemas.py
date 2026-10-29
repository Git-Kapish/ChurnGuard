"""Pydantic v2 request/response models for the FastAPI service (endpoints arrive in Block 6).

Requests use the raw dataset vocabulary, so "No phone service" / "No internet service" are
accepted; predict.py folds them into "No" before scoring (same as data/clean.py).
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from churnguard.config import load_config
from churnguard.data.validate import MONTHLY_RANGE, TENURE_RANGE

MAX_BATCH = load_config("model")["api"]["max_batch_size"]


class Gender(StrEnum):
    FEMALE = "Female"
    MALE = "Male"


class YesNo(StrEnum):
    YES = "Yes"
    NO = "No"


class MultipleLines(StrEnum):
    YES = "Yes"
    NO = "No"
    NO_PHONE = "No phone service"


class InternetService(StrEnum):
    DSL = "DSL"
    FIBER = "Fiber optic"
    NONE = "No"


class AddOn(StrEnum):
    YES = "Yes"
    NO = "No"
    NO_INTERNET = "No internet service"


class Contract(StrEnum):
    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"


class PaymentMethod(StrEnum):
    BANK_TRANSFER = "Bank transfer (automatic)"
    CREDIT_CARD = "Credit card (automatic)"
    ELECTRONIC_CHECK = "Electronic check"
    MAILED_CHECK = "Mailed check"


class RiskTier(StrEnum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class Customer(BaseModel):
    """One customer, using the original Telco column names."""

    model_config = ConfigDict(extra="forbid")

    customerID: str | None = Field(None, max_length=64, description="Echoed back if given")
    gender: Gender
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=TENURE_RANGE[0], le=TENURE_RANGE[1], description="Months")
    PhoneService: YesNo
    MultipleLines: MultipleLines
    InternetService: InternetService
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Contract
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethod
    MonthlyCharges: float = Field(ge=MONTHLY_RANGE[0], le=MONTHLY_RANGE[1])
    TotalCharges: float | None = Field(
        None, ge=0, description="Blank for brand-new customers; imputed by the pipeline"
    )


class BatchRequest(BaseModel):
    """Up to MAX_BATCH customers."""

    model_config = ConfigDict(extra="forbid")

    customers: list[Customer] = Field(min_length=1, max_length=MAX_BATCH)


class Reason(BaseModel):
    """One plain-English explanation factor."""

    text: str
    direction: str = Field(pattern="^(increases|decreases)_risk$")


class Prediction(BaseModel):
    """Score for one customer."""

    customerID: str | None = None
    churn_probability: float = Field(ge=0, le=1)
    risk_tier: RiskTier
    will_target: bool = Field(description="Probability is at or above the cost-optimal threshold")
    top_reasons: list[Reason]


class BatchResponse(BaseModel):
    """Predictions in request order."""

    predictions: list[Prediction]


class Health(BaseModel):
    """Liveness plus the model version loaded."""

    status: str
    model_version: str


class ModelInfo(BaseModel):
    """Contents of models/metadata.json exposed by /model/info."""

    model_version: str
    trained_at: str
    threshold: float
    metrics: dict[str, float]
    features: list[str]
