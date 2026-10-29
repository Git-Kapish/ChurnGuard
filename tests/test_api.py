import io
import logging
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churnguard.api import main as api_main
from churnguard.api.main import MAX_CSV_BYTES, create_app
from churnguard.api.schemas import MAX_BATCH
from churnguard.config import PROCESSED_DIR
from churnguard.data.split import load_split, split_xy
from churnguard.models.artifact import MODEL_PATH
from churnguard.predict import Predictor

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and (PROCESSED_DIR / "val.parquet").exists()),
    reason="run `make data train` first",
)


@pytest.fixture(scope="module")
def loads():
    return []


@pytest.fixture(scope="module")
def client(loads):
    def factory() -> Predictor:
        loads.append(1)
        return Predictor()

    with TestClient(create_app(factory)) as c:
        yield c


@pytest.fixture(scope="module")
def customers():
    Xv, yv = split_xy(load_split("val"))
    return Xv.head(40).assign(Churn=yv.head(40).map({0: "No", 1: "Yes"}))


def body(customers, i: int = 0) -> dict:
    rec = customers.drop(columns=["Churn"]).iloc[i].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}


def to_csv_file(df: pd.DataFrame) -> dict:
    return {"file": ("customers.csv", df.to_csv(index=False), "text/csv")}


# --- health and info ---------------------------------------------------------------------


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "v1"}


def test_model_info(client) -> None:
    info = client.get("/model/info").json()
    assert info["model_version"] == "v1" and 0 < info["threshold"] < 1
    assert len(info["features"]) == 19 and "Churn" not in info["features"]
    assert info["metrics"]["pr_auc"] > 0.5 and "split" not in info["metrics"]
    assert info["test_metrics"] is None or info["test_metrics"]["roc_auc"] > 0.5
    assert "assumption" in info["note"].lower()
    assert {"medium", "high"} == set(info["risk_tiers"])


def test_docs_and_openapi_list_every_endpoint(client) -> None:
    assert client.get("/docs").status_code == 200
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {"/health", "/model/info", "/predict", "/predict/batch", "/predict/csv"} <= paths


# --- /predict ----------------------------------------------------------------------------


def test_predict_valid_payload(client, customers) -> None:
    r = client.post("/predict", json={**body(customers), "customerID": "ABC-1"})
    assert r.status_code == 200
    out = r.json()
    assert out["customerID"] == "ABC-1"
    assert 0 <= out["churn_probability"] <= 1
    assert out["risk_tier"] in {"Low", "Medium", "High"}
    assert isinstance(out["will_target"], bool)
    assert 0 < len(out["top_reasons"]) <= 8
    assert out["top_reasons"][0]["text"].endswith(("raises churn risk", "lowers churn risk"))


def test_predict_accepts_raw_vocabulary_and_missing_total_charges(client, customers) -> None:
    payload = {
        **body(customers),
        "MultipleLines": "No phone service",
        "TechSupport": "No internet service",
    }
    payload.pop("TotalCharges")
    payload["tenure"] = 0
    assert client.post("/predict", json=payload).status_code == 200


def test_predict_bad_enum_is_422(client, customers) -> None:
    r = client.post("/predict", json={**body(customers), "Contract": "Weekly"})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][-1] == "Contract"


def test_predict_missing_field_is_422(client, customers) -> None:
    payload = body(customers)
    del payload["tenure"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422 and r.json()["detail"][0]["type"] == "missing"


@pytest.mark.parametrize(
    "override",
    [
        {"tenure": -1},
        {"tenure": "soon"},
        {"MonthlyCharges": 9999},
        {"SeniorCitizen": 2},
        {"extra": 1},
    ],
)
def test_predict_invalid_values_are_422(client, customers, override) -> None:
    assert client.post("/predict", json={**body(customers), **override}).status_code == 422


def test_predict_empty_body_is_422(client) -> None:
    assert client.post("/predict", json={}).status_code == 422


# --- /predict/batch ----------------------------------------------------------------------


def test_batch_preserves_order_and_matches_single(client, customers) -> None:
    items = [body(customers, i) for i in range(5)]
    batch = client.post("/predict/batch", json={"customers": items}).json()["predictions"]
    singles = [client.post("/predict", json=it).json() for it in items]
    assert [b["churn_probability"] for b in batch] == pytest.approx(
        [s["churn_probability"] for s in singles]
    )


def test_batch_explain_false_skips_reasons(client, customers) -> None:
    r = client.post("/predict/batch?explain=false", json={"customers": [body(customers)]})
    assert r.json()["predictions"][0]["top_reasons"] == []


def test_batch_limit_is_enforced(client, customers) -> None:
    one = body(customers)
    ok = client.post("/predict/batch?explain=false", json={"customers": [one] * MAX_BATCH})
    assert ok.status_code == 200 and len(ok.json()["predictions"]) == MAX_BATCH
    too_many = client.post(
        "/predict/batch?explain=false", json={"customers": [one] * (MAX_BATCH + 1)}
    )
    assert too_many.status_code == 422
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_batch_with_one_bad_customer_is_rejected_whole(client, customers) -> None:
    items = [body(customers, 0), {**body(customers, 1), "PaymentMethod": "Cash"}]
    assert client.post("/predict/batch", json={"customers": items}).status_code == 422


# --- /predict/csv ------------------------------------------------------------------------


def test_csv_roundtrip_keeps_columns_and_adds_scores(client, customers) -> None:
    df = customers.astype({"tenure": object, "TotalCharges": object})
    df.loc[df.index[0], "tenure"] = "0"  # a brand-new customer with a blank TotalCharges
    df.loc[df.index[0], "TotalCharges"] = " "
    r = client.post("/predict/csv", files=to_csv_file(df))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "scored.csv" in r.headers["content-disposition"]
    out = pd.read_csv(io.StringIO(r.text))
    assert len(out) == len(df)
    assert list(df.columns) == list(out.columns[: len(df.columns)])
    assert {"churn_probability", "risk_tier", "will_target", "top_reasons"} <= set(out.columns)
    assert out["churn_probability"].between(0, 1).all()
    assert out["top_reasons"].str.contains("churn risk").all()


def test_csv_with_customer_ids_keeps_them(client, customers) -> None:
    df = customers.head(3).assign(customerID=["A-1", "B-2", "C-3"])
    out = pd.read_csv(io.StringIO(client.post("/predict/csv", files=to_csv_file(df)).text))
    assert out["customerID"].tolist() == ["A-1", "B-2", "C-3"]


def test_csv_scores_match_the_json_endpoint(client, customers) -> None:
    out = pd.read_csv(
        io.StringIO(client.post("/predict/csv", files=to_csv_file(customers.head(5))).text)
    )
    items = [body(customers, i) for i in range(5)]
    batch = client.post("/predict/batch", json={"customers": items}).json()["predictions"]
    assert out["churn_probability"].to_numpy() == pytest.approx(
        [b["churn_probability"] for b in batch], abs=1e-6
    )


def test_csv_bad_row_reports_its_line_number(client, customers) -> None:
    df = customers.head(5).copy()
    df.loc[df.index[2], "Contract"] = "Weekly"
    r = client.post("/predict/csv", files=to_csv_file(df))
    assert r.status_code == 422
    assert any(d.startswith("row 4:") and "Contract" in d for d in r.json()["detail"])


def test_csv_missing_column_is_422(client, customers) -> None:
    r = client.post("/predict/csv", files=to_csv_file(customers.head(3).drop(columns=["Contract"])))
    assert r.status_code == 422 and "Contract" in r.json()["detail"]


def test_csv_row_limit_size_limit_and_garbage(client, customers) -> None:
    big = pd.concat([customers.head(1)] * (MAX_BATCH + 1))
    assert client.post("/predict/csv?explain=false", files=to_csv_file(big)).status_code == 422
    huge = {"file": ("x.csv", b"a,b\n" + b"1,2\n" * (MAX_CSV_BYTES // 4 + 10), "text/csv")}
    assert client.post("/predict/csv", files=huge).status_code == 413
    empty = {"file": ("x.csv", b"", "text/csv")}
    assert client.post("/predict/csv", files=empty).status_code == 400
    header_only = to_csv_file(customers.head(0))
    assert client.post("/predict/csv", files=header_only).status_code == 422
    assert client.post("/predict/csv").status_code == 422  # no file at all


# --- service behaviour -------------------------------------------------------------------


def test_model_is_loaded_once_not_per_request(client, loads, customers) -> None:
    for i in range(3):
        client.post("/predict", json=body(customers, i))
    assert loads == [1]


def test_latency_header_and_request_log(client, customers, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="churnguard.api"):
        r = client.post("/predict", json=body(customers))
    assert float(r.headers["X-Process-Time-ms"]) > 0
    assert any(
        "POST /predict -> 200" in rec.message and " ms" in rec.message for rec in caplog.records
    )


def test_startup_fails_fast_when_the_model_is_missing() -> None:
    def broken() -> Predictor:
        raise FileNotFoundError("models/churn_model_v1.joblib - run `make train`")

    with pytest.raises(FileNotFoundError, match="make train"), TestClient(create_app(broken)):
        pass


def test_importing_the_module_does_not_load_the_model() -> None:
    assert api_main.app.state._state == {}  # lifespan has not run


def test_single_prediction_latency_is_reported_and_sane(client, customers) -> None:
    times = []
    for i in range(25):
        start = time.perf_counter()
        assert client.post("/predict", json=body(customers, i)).status_code == 200
        times.append((time.perf_counter() - start) * 1000)
    p95 = float(np.percentile(times[5:], 95))
    median = np.median(times[5:])
    print(f"\nsingle /predict (in-process HTTP): median {median:.0f} ms, p95 {p95:.0f} ms")
    assert p95 < 2000  # sanity bound only; the PRD target (100 ms) is reported in the block summary
