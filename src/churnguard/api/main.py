"""FastAPI service (PRD FR8). All scoring goes through churnguard.predict.

Run with `make api` (uvicorn churnguard.api.main:app --port 8000); docs at /docs.
"""

import logging
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from churnguard.api.csv_input import MAX_CSV_BYTES, CsvInputError, parse_customers_csv
from churnguard.api.schemas import (
    BatchRequest,
    BatchResponse,
    Customer,
    Health,
    ModelInfo,
    Prediction,
)
from churnguard.predict import Predictor, get_predictor

log = logging.getLogger("churnguard.api")
# Rejected before any parsing. Covers a 2 MB CSV upload plus multipart overhead, and a full
# 1,000-customer JSON batch (about 0.5 MB).
MAX_BODY_BYTES = MAX_CSV_BYTES + 100_000
DESCRIPTION = (
    "Scores telecom customers for churn risk. `will_target` is true when the probability is at "
    "or above a threshold chosen to maximize an **assumption-driven** simulated net value "
    "(offer cost, success rate and lifetime value are assumptions in configs/business.yaml, "
    "not measured facts). Reasons are SHAP explanations of the base model."
)


def create_app(predictor_factory: Callable[[], Predictor] = get_predictor) -> FastAPI:
    """Build the app. The predictor is created once at startup, so a missing model fails fast."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = predictor_factory()
        app.state.lock = threading.Lock()  # one scoring call at a time: it is CPU-bound anyway
        log.info("model %s loaded", app.state.predictor.model_version)
        yield

    if not logging.getLogger().handlers:  # under uvicorn: make our request log visible
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = FastAPI(title="ChurnGuard API", description=DESCRIPTION, version="1.0", lifespan=lifespan)

    @app.middleware("http")
    async def limit_size_and_log_latency(request: Request, call_next):
        start = time.perf_counter()
        size = request.headers.get("content-length", "")
        # shortcut: chunked uploads (no Content-Length) are not capped here; add a streaming
        # byte counter if the API is ever exposed beyond trusted callers.
        if size.isdigit() and int(size) > MAX_BODY_BYTES:
            response = JSONResponse(
                {"detail": f"request body larger than {MAX_BODY_BYTES} bytes"}, status_code=413
            )
        else:
            response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
        log.info(
            "%s %s -> %d in %.1f ms", request.method, request.url.path, response.status_code, ms
        )
        return response

    def score(request: Request, records: list[dict], explain: bool) -> list[dict[str, Any]]:
        try:
            with request.app.state.lock:
                return request.app.state.predictor.predict(records, explain=explain)
        except ValueError as exc:  # e.g. a column the model needs is missing
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        """Liveness plus the model version loaded."""
        return Health(status="ok", model_version=request.app.state.predictor.model_version)

    @app.get("/model/info", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        """Model metadata: metrics, threshold, features, training time."""
        meta = request.app.state.predictor.metadata
        numeric = {k: v for k, v in meta["metrics"].items() if isinstance(v, int | float)}
        test = meta.get("test_metrics", {}).get("metrics")
        return ModelInfo(
            model_version=meta["model_version"],
            model_name=meta["model_name"],
            calibration=meta["calibration"],
            trained_at=meta["trained_at"],
            threshold=meta["threshold"],
            risk_tiers=meta["risk_tiers"],
            metrics=numeric,
            test_metrics=test,
            features=meta["features"],
            note=meta["business_assumptions"]["note"] + " " + meta["threshold_rule"] + ".",
        )

    @app.post("/predict", response_model=Prediction)
    def predict_one(customer: Customer, request: Request) -> dict[str, Any]:
        """Score one customer."""
        return score(request, [customer.model_dump(mode="json")], explain=True)[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(body: BatchRequest, request: Request, explain: bool = True) -> dict:
        """Score up to 1,000 customers, in request order. `explain=false` skips reasons."""
        records = [c.model_dump(mode="json") for c in body.customers]
        return {"predictions": score(request, records, explain)}

    @app.post("/predict/csv")
    def predict_csv(
        request: Request, file: Annotated[UploadFile, File()], explain: bool = True
    ) -> Response:
        """Upload a CSV of customers (dataset columns); get it back with score columns added."""
        try:
            df, records = parse_customers_csv(file.file.read(MAX_CSV_BYTES + 1))
        except CsvInputError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        results = score(request, records, explain)
        out = df.copy()
        out["churn_probability"] = [r["churn_probability"] for r in results]
        out["risk_tier"] = [r["risk_tier"] for r in results]
        out["will_target"] = [r["will_target"] for r in results]
        out["top_reasons"] = [" | ".join(x["text"] for x in r["top_reasons"]) for r in results]
        return Response(
            content=out.to_csv(index=False),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="scored.csv"'},
        )

    return app


app = create_app()
