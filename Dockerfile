# syntax=docker/dockerfile:1
# One image, two services (see docker-compose.yml): the API (:8000) and the dashboard (:8501).

# --- builder: resolve and install pinned dependencies, then the package itself ---------------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
# requirements.lock pins every runtime dependency (training-only mlflow/optuna are excluded)
COPY requirements.lock .
RUN pip install -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# --- runtime: just the venv, the config, the model artifact and the apps ---------------------
FROM python:3.11-slim AS runtime
# libgomp: OpenMP runtime needed by XGBoost and LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHURNGUARD_ROOT=/app \
    NUMBA_CACHE_DIR=/tmp/numba
WORKDIR /app
COPY --chown=app:app configs ./configs
COPY --chown=app:app models ./models
COPY --chown=app:app reports ./reports
COPY --chown=app:app app ./app
COPY --chown=app:app .streamlit ./.streamlit
USER app
EXPOSE 8000 8501
CMD ["uvicorn", "churnguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
