# PRD: ChurnGuard — Customer Churn Prediction System

| | |
|---|---|
| **Version** | 1.0 |
| **Owner** | Kapish |
| **Build window** | 6 hours, built with Claude Code |
| **Stack** | Python 3.11, pandas, scikit-learn, XGBoost, SHAP, MLflow, FastAPI, Streamlit, Docker, pytest |
| **Status** | Ready to build |

---

## 1. Overview

ChurnGuard predicts which telecom subscribers are likely to cancel, explains why, and recommends **who to target with a retention offer given a budget**. It is an end-to-end ML system, not a notebook: data pipeline, tracked experiments, calibrated model, business-cost threshold, explainability, REST API, dashboard, Docker packaging, tests.

**Why this project (portfolio angle):** it shows the full ML lifecycle (EDA → modeling → evaluation → serving → monitoring hooks) plus business reasoning, which is what ML/SWE hiring panels look for beyond "I trained XGBoost and got 80% accuracy."

## 2. Goals and Non-Goals

### Goals
1. Reproducible training pipeline from raw CSV to a versioned model artifact.
2. Compare at least 4 models fairly with stratified cross-validation and tracked experiments.
3. Calibrated probabilities and a **cost-based decision threshold** (not default 0.5).
4. Per-customer explanations (SHAP) and global drivers of churn.
5. REST API for single and batch scoring, containerized.
6. Dashboard for exploration, scoring, and campaign targeting simulation.
7. Tests, README, and a model card.

### Non-Goals
- No live streaming or real-time feature store.
- No cloud deployment (local Docker only).
- No deep learning (tabular data, GBMs win here; stated in the model card).
- No auth or multi-user features.

## 3. Dataset

**Primary:** IBM Telco Customer Churn (7,043 rows, 21 columns, ~26.5% churn).

| Group | Columns |
|---|---|
| Identity | `customerID` (dropped from features) |
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Tenure/Account | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Billing | `MonthlyCharges`, `TotalCharges` |
| Target | `Churn` (Yes/No) |

**Known data issues to handle explicitly (and document):**
- `TotalCharges` has blank strings (~11 rows, all `tenure = 0`): coerce to numeric, impute 0 for new customers.
- "No internet service" / "No phone service" are redundant categories: normalize to "No" where the parent service is absent, keep the parent flag.
- Class imbalance (~73/27): stratified splits everywhere.

**Source:** `data/raw/telco_churn.csv`. A `scripts/download_data.py` fetches it from the public IBM sample repo; if unreachable, the README tells the user to drop the Kaggle copy in place. Raw data is never modified and never committed if the license is unclear (gitignore it, keep a checksum).

## 4. Users and Use Cases

| Persona | Need | Feature |
|---|---|---|
| Retention manager | Who do I call with a fixed budget? | Campaign simulator, ranked target list |
| Customer-care agent | Why is this customer at risk? | Single-customer score + SHAP reasons |
| Data scientist/reviewer | Can I trust and reproduce this? | MLflow runs, model card, tests |
| Engineer | Can I integrate it? | `/predict`, `/predict/batch` API |

## 5. Functional Requirements

### FR1: Data pipeline (`src/churnguard/data/`)
- Load, validate (schema + range checks via pandera or hand-rolled asserts), clean.
- Stratified split: **70 / 15 / 15** train / validation / test, fixed seed. Test set is touched exactly once, at the end.
- Output cleaned parquet to `data/processed/`.

### FR2: Feature engineering (`src/churnguard/features/`)
All transforms live inside an sklearn `Pipeline` / `ColumnTransformer` so train and serve are identical (no leakage, no skew).

| Feature | Definition |
|---|---|
| `tenure_bucket` | 0-12, 13-24, 25-48, 49+ months |
| `n_services` | Count of active add-on services |
| `avg_monthly_spend` | `TotalCharges / max(tenure, 1)` |
| `charge_delta` | `MonthlyCharges - avg_monthly_spend` (price-increase signal) |
| `is_month_to_month` | Contract flag |
| `has_support_bundle` | OnlineSecurity or TechSupport |
| `auto_pay` | Payment method contains "automatic" |

Numeric: median impute + scale (for linear models). Categorical: one-hot (trees may use the same encoding for simplicity). Leakage rule: fit every transformer on the training fold only.

### FR3: Modeling (`src/churnguard/models/`)
Models compared under identical 5-fold stratified CV on the train split:

| # | Model | Role |
|---|---|---|
| 1 | Dummy (prior) | Floor baseline |
| 2 | Logistic Regression (L2, `class_weight=balanced`) | Interpretable baseline |
| 3 | Random Forest | Bagging benchmark |
| 4 | XGBoost | Primary candidate |
| 5 | LightGBM | Alternative GBM |

- Hyperparameter search: Optuna (or RandomizedSearchCV) on the top 2 models, 30 trials each, optimizing **PR-AUC**.
- Imbalance: compare `class_weight` / `scale_pos_weight` vs SMOTE (inside the CV pipeline via imblearn) and report the result honestly, even if SMOTE does not help.
- Calibration: `CalibratedClassifierCV` (isotonic or sigmoid, chosen on validation) on the winner; report reliability curve and Brier score.
- Best model serialized with `joblib` to `models/churn_model_v1.joblib` plus `models/metadata.json` (feature list, threshold, metrics, git hash, data checksum, library versions).

### FR4: Evaluation (`src/churnguard/evaluation/`)
Reported on validation for selection, on the **test set once** for the final number.

- ROC-AUC, **PR-AUC (primary)**, F1, precision, recall, Brier score, log loss.
- Confusion matrix at the chosen threshold.
- Lift/gain chart and **top-decile lift**.
- Calibration curve.
- Error analysis: performance sliced by contract type, tenure bucket, and senior-citizen flag (fairness/robustness check, documented in the model card).
- Bootstrap 95% confidence intervals on test ROC-AUC and PR-AUC.

### FR5: Business-cost threshold (key differentiator)
Assumptions (configurable in `configs/business.yaml`):

| Parameter | Default | Meaning |
|---|---|---|
| `retention_offer_cost` | 20 | Cost of contacting/discounting one customer |
| `offer_success_rate` | 0.30 | Fraction of true churners saved by an offer |
| `customer_lifetime_value` | 500 | Value kept when a churner is saved |

Expected net value per targeted customer = `p * success_rate * CLV - offer_cost`. Choose the threshold (or top-K under a budget) that maximizes total expected net value on validation; show the profit-vs-threshold curve and compare against "target nobody", "target everyone", and "default 0.5 threshold". Clearly label this as an assumption-driven simulation.

### FR6: Explainability (`src/churnguard/explain/`)
- Global: SHAP summary (beeswarm) and top-10 mean |SHAP| bar chart saved to `reports/figures/`.
- Local: for any customer, top 5 risk-increasing and top 3 risk-decreasing factors in plain English (for example "Month-to-month contract raises risk").
- Sanity check: SHAP ranking compared with permutation importance; note disagreements.

### FR7: Experiment tracking
MLflow local tracking (`mlruns/`): log params, CV metrics, plots, model, and data checksum for every run. A `make mlflow` target opens the UI.

### FR8: REST API (`src/churnguard/api/`, FastAPI)

| Endpoint | Method | Behavior |
|---|---|---|
| `/health` | GET | Liveness + model version loaded |
| `/model/info` | GET | Metadata: metrics, threshold, features, trained-at |
| `/predict` | POST | One customer JSON → `{churn_probability, risk_tier, will_target, top_reasons[]}` |
| `/predict/batch` | POST | Up to 1,000 customers → list of the above |
| `/predict/csv` | POST | Upload CSV → scored CSV |

- Pydantic v2 request models with enum validation and clear 422 errors.
- Risk tiers: Low (< 0.25), Medium (0.25-0.5), High (> 0.5), configurable.
- Model loaded once at startup; request logging with latency; p95 latency under 100 ms for single prediction locally.

### FR9: Dashboard (`app/`, Streamlit)
Four pages:
1. **Overview:** churn rate, churn by contract/tenure/services (interactive charts).
2. **Model:** metrics table, ROC/PR/calibration/lift plots, model comparison table.
3. **Score a customer:** form → probability gauge + SHAP waterfall + reasons.
4. **Campaign simulator:** sliders for budget/offer cost/success rate/CLV → ranked target list, expected profit, download CSV.

The dashboard calls the model through the same `predict` module as the API (shared code, no duplicated logic).

### FR10: Packaging and ops
- `Dockerfile` (multi-stage, non-root) and `docker-compose.yml` running API (8000) and dashboard (8501).
- `Makefile`: `setup`, `data`, `train`, `evaluate`, `test`, `api`, `app`, `mlflow`, `docker-up`, `lint`.
- Pre-commit with ruff + black. GitHub Actions workflow: lint + tests on push.
- Drift hook: `scripts/drift_check.py` computes PSI per feature between train and a new CSV and flags PSI > 0.2 (a monitoring demonstration, not a live service).

## 6. Non-Functional Requirements

| Area | Requirement |
|---|---|
| Reproducibility | Fixed seeds; `make train` regenerates the same metrics (±0.002) on a clean clone |
| Leakage | Zero preprocessing fit outside train folds; verified by a test |
| Code quality | Type hints, docstrings on public functions, ruff clean |
| Test coverage | ≥ 80% on `src/`, including API tests via `TestClient` |
| Performance | Training the full comparison under 10 minutes on a laptop CPU |
| Security | No secrets in repo; API input size limits; non-root container |

## 7. Success Metrics

| Metric | Target | Notes |
|---|---|---|
| ROC-AUC (test) | ≥ 0.84 | Typical for this dataset is 0.83-0.86 |
| PR-AUC (test) | ≥ 0.64 | Primary selection metric |
| Recall at chosen threshold | ≥ 0.70 | Business prefers catching churners |
| Top-decile lift | ≥ 2.8x | |
| Brier score | ≤ 0.15 | Calibration quality |
| Net value vs "target everyone" | Strictly higher | Under default business assumptions |

If a target is missed, the report states the real number and why. Targets are not a reason to tune on the test set.

## 8. System Architecture

```
            ┌──────────────┐
 raw CSV ──▶│ Data pipeline│──▶ processed parquet
            └──────┬───────┘
                   ▼
        ┌─────────────────────┐        ┌──────────┐
        │ Feature + Model     │───────▶│  MLflow  │
        │ sklearn Pipeline    │ logs   └──────────┘
        └──────┬──────────────┘
               ▼  joblib + metadata.json
        ┌─────────────────────┐
        │ predict module      │◀──────────────┐
        │ (proba, threshold,  │               │
        │  SHAP reasons)      │               │
        └───────┬─────────────┘               │
                ▼                              ▼
         ┌────────────┐               ┌────────────────┐
         │  FastAPI   │               │   Streamlit    │
         │  :8000     │               │   :8501        │
         └────────────┘               └────────────────┘
              └────────── Docker Compose ──────────┘
```

## 9. Repository Layout

```
churnguard/
├── CLAUDE.md
├── PRD.md
├── README.md
├── Makefile
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── configs/
│   ├── model.yaml
│   └── business.yaml
├── data/
│   ├── raw/            # gitignored
│   └── processed/      # gitignored
├── models/             # versioned artifacts + metadata.json
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
│   ├── figures/
│   └── MODEL_CARD.md
├── scripts/
│   ├── download_data.py
│   └── drift_check.py
├── src/churnguard/
│   ├── data/           # load, validate, split
│   ├── features/       # transformers, pipeline
│   ├── models/         # train, tune, calibrate
│   ├── evaluation/     # metrics, plots, cost threshold
│   ├── explain/        # SHAP, reasons
│   ├── predict.py      # shared inference entrypoint
│   └── api/            # FastAPI app, schemas
├── app/                # Streamlit pages
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_leakage.py
    ├── test_metrics.py
    ├── test_cost_threshold.py
    └── test_api.py
```

## 10. 6-Hour Build Plan (full scope, no cuts)

The scope stays whole by running Claude Code on parallelizable modules and keeping each block's output verifiable.

| Block | Time | Deliverable | Done when |
|---|---|---|---|
| **0** | 0:00-0:20 | Repo scaffold, `pyproject.toml`, Makefile, `CLAUDE.md`, config files, data download | `make setup && make data` works |
| **1** | 0:20-1:10 | Data loading/validation/split + EDA notebook + data tests | Clean parquet written; cleaning issues documented; `test_data` green |
| **2** | 1:10-2:00 | Feature pipeline + leakage test | `test_features`, `test_leakage` green |
| **3** | 2:00-3:15 | 5-model CV comparison, imbalance study, Optuna on top 2, MLflow logging | Comparison table in MLflow and `reports/` |
| **4** | 3:15-4:00 | Calibration, test-set evaluation (once), plots, bootstrap CIs, slice analysis, cost-threshold optimizer + tests | `metadata.json` written; all figures saved |
| **5** | 4:00-4:30 | SHAP global and local explanations, plain-English reasons, `predict.py` | Reasons returned for sample customer |
| **6** | 4:30-5:10 | FastAPI endpoints + Pydantic schemas + API tests | `pytest tests/test_api.py` green; `/docs` works |
| **7** | 5:10-5:40 | Streamlit 4 pages | All pages run against the real model |
| **8** | 5:40-6:00 | Dockerfile/compose, CI workflow, drift script, README, MODEL_CARD | `docker compose up` serves both; README quickstart verified from a clean clone |

**Parallelization (Claude Code):** while Block 3 trains, write Block 6 schemas and Block 8 Docker files in a second session. Run long training with `run_in_background`.

**Risk buffers (inside the plan, not cuts):**
- If Optuna is slow: reduce to 20 trials per model, keep the study design identical.
- If SHAP on the calibrated wrapper is awkward: explain the uncalibrated base estimator, calibrate probabilities separately, and note it in the model card.
- If Docker build is slow: build the images during Block 7 in the background.

## 11. Testing Strategy

| Test | Asserts |
|---|---|
| `test_data` | Blank `TotalCharges` handled; no nulls post-clean; split sizes and stratification preserved |
| `test_features` | Engineered features match hand-computed values on fixture rows |
| `test_leakage` | Fitting the pipeline on train only gives identical transforms regardless of test content |
| `test_metrics` | Lift, PR-AUC, and bootstrap helpers match sklearn / known values |
| `test_cost_threshold` | Optimizer returns "target nobody" when offer cost exceeds any expected gain, and "target all" when it is negligible |
| `test_api` | 200 on valid payload, 422 on bad enum/missing field, batch limit enforced, `/health` correct |
| Smoke | `make train` end-to-end on a 500-row sample under 60 seconds |

## 12. Deliverables Checklist

- [ ] Reproducible pipeline (`make data train evaluate`)
- [ ] MLflow experiment with all runs
- [ ] `models/churn_model_v1.joblib` + `metadata.json`
- [ ] Evaluation figures (ROC, PR, calibration, lift, SHAP, profit curve)
- [ ] FastAPI service with OpenAPI docs
- [ ] Streamlit dashboard (4 pages)
- [ ] Docker Compose running both services
- [ ] Test suite ≥ 80% coverage, CI green
- [ ] `MODEL_CARD.md` (intended use, data, metrics with CIs, slices, limitations, ethical notes)
- [ ] README with architecture diagram, results table, screenshots, and quickstart

## 13. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Small dataset overfits during tuning | Inflated metrics | Nested-style discipline: tune on CV, select on validation, test once; report CIs |
| Test-set peeking | Invalid result | Test evaluation lives in one script that logs a single run; the Makefile refuses a second run without `--force` |
| Business numbers are invented | Misleading profit claims | Label as assumptions in UI, README, and model card; all in `business.yaml` |
| Dataset is synthetic-ish and clean | Limited real-world signal | State it in the model card; do not claim production readiness |
| Scope creep | Misses 6-hour window | Out-of-scope list in section 2 is binding |

## 14. Resume Framing (for later)

> **ChurnGuard**: End-to-end churn prediction system (XGBoost, SHAP, MLflow, FastAPI, Streamlit, Docker). Compared 5 models under stratified CV, calibrated probabilities, and chose a cost-optimal targeting threshold; achieved X ROC-AUC / Y PR-AUC with Z% higher simulated net value than blanket targeting. Served through a containerized REST API with per-customer explanations and 80%+ test coverage.

Fill X, Y, Z from the real test results only.

## 15. Open Decisions (defaults chosen; change before Block 0 if you disagree)

1. **Dataset:** IBM Telco (default). Alternative: a bank or SaaS churn dataset for variety.
2. **Tuning library:** Optuna (default) vs RandomizedSearchCV.
3. **Dashboard:** Streamlit (default) vs a React frontend (would not fit 6 hours alongside the rest).
