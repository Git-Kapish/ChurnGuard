# ChurnGuard: Claude Code Build Prompts

## Before you start (5 min, terminal)

```bash
mkdir churnguard && cd churnguard
git init
# copy PRD.md and CLAUDE.md into this folder
claude
```

Tip: use `/init` only if you want it to *append* to CLAUDE.md. Do not let it overwrite yours.
Consider Plan mode (Shift+Tab) for Block 0, then switch to normal mode.

Paste prompts one at a time. Wait for "done when" to be true before the next. Commit after each.

---

## Block 0: Scaffold (0:00-0:20)

```
Read PRD.md and CLAUDE.md fully. Then do Block 0 from PRD section 10:
- Create the repo layout from PRD section 9 (empty packages with __init__.py).
- pyproject.toml with pinned dependency ranges for everything in the CLAUDE.md stack, plus dev deps (pytest, pytest-cov, ruff, black, pre-commit, httpx).
- Makefile with every target listed in CLAUDE.md (stubs are fine for ones whose code doesn't exist yet, but they must fail clearly).
- configs/model.yaml and configs/business.yaml with the defaults from the PRD.
- .gitignore (data/, mlruns/, .venv, __pycache__, models/*.joblib kept out of git unless small), .pre-commit-config.yaml, README stub.
- scripts/download_data.py that fetches the IBM Telco churn CSV to data/raw/telco_churn.csv, writes a SHA256 checksum file, and prints a clear fallback message if the download fails.
Then run make setup and make data and show me the output. Commit.
```

## Block 1: Data + EDA (0:20-1:10)

```
Do Block 1. In src/churnguard/data/: load, validate (schema, allowed categories, numeric ranges), clean (TotalCharges blanks, "No internet/phone service" normalization), and stratified 70/15/15 split with seed 42, saving parquet to data/processed/. Write tests/test_data.py covering blank TotalCharges, no nulls after cleaning, split sizes and class ratio preservation.
Then create notebooks/01_eda.ipynb: churn rate, churn by contract/tenure/internet service/payment method, numeric distributions, correlations, and a short written "findings" cell. Use only the training split for EDA conclusions.
Run make lint && make test. Commit.
```

## Block 2: Features (1:10-2:00)

```
Do Block 2. Implement the feature engineering from PRD FR2 as sklearn transformers inside a ColumnTransformer/Pipeline in src/churnguard/features/. Include tenure_bucket, n_services, avg_monthly_spend, charge_delta, is_month_to_month, has_support_bundle, auto_pay.
Write tests/test_features.py (hand-computed values on fixture rows) and tests/test_leakage.py (fit on train only; transform output for a given row must not change when test rows change).
Run make lint && make test. Commit.
```

## Block 3: Models + tuning (2:00-3:15)

```
Do Block 3. In src/churnguard/models/: 5-fold stratified CV comparison of Dummy, Logistic Regression, Random Forest, XGBoost, LightGBM on the train split with identical folds. Log everything to MLflow (params, CV mean/std for ROC-AUC, PR-AUC, F1, recall, Brier, data checksum).
Then an imbalance study: class weights vs SMOTE (imblearn Pipeline inside CV). Then Optuna, 30 trials, on the top 2 models by CV PR-AUC. Save a comparison table to reports/model_comparison.md.
Run training in the background; while it runs, start on the FastAPI schemas in src/churnguard/api/schemas.py (Pydantic v2, enums for categoricals) without wiring endpoints yet.
Do NOT touch the test split. Commit.
```

## Block 4: Calibration, evaluation, cost threshold (3:15-4:00)

```
Do Block 4.
1. Calibrate the winner (isotonic vs sigmoid chosen on validation). Save reliability curve and Brier score.
2. Implement src/churnguard/evaluation/cost.py: expected net value = p * success_rate * CLV - offer_cost, configurable from configs/business.yaml. Find the best threshold and best top-K under a budget on VALIDATION. Compare vs target-nobody, target-everyone, and threshold 0.5. Save the profit-vs-threshold plot. Write tests/test_cost_threshold.py (target-nobody when cost is huge, target-all when negligible).
3. Write tests/test_metrics.py for lift, PR-AUC, bootstrap CI helpers.
4. Create the single evaluate script that scores the TEST set once: ROC-AUC, PR-AUC, F1, precision, recall, Brier, confusion matrix at chosen threshold, top-decile lift, bootstrap 95% CIs, and slices by contract type, tenure bucket, senior citizen. Make it refuse to run twice without --force.
5. Save models/churn_model_v1.joblib and models/metadata.json (features, threshold, metrics, git hash, data checksum, library versions). Save all figures to reports/figures/.
Run make lint && make test. Commit.
```

## Block 5: Explainability + predict module (4:00-4:30)

```
Do Block 5. In src/churnguard/explain/: SHAP global (beeswarm + top-10 bar) saved to reports/figures/, local explanations as plain-English reasons (top 5 risk-increasing, top 3 risk-decreasing), and a permutation-importance comparison note. Handle the calibrated-model case as described in CLAUDE.md pitfalls.
Then src/churnguard/predict.py: the single inference entrypoint returning probability, risk tier, will_target (using saved threshold), and reasons. Add tests. Commit.
```

## Block 6: API (4:30-5:10)

```
Do Block 6. FastAPI app in src/churnguard/api/ with /health, /model/info, /predict, /predict/batch (max 1000), /predict/csv per PRD FR8. Load the model once at startup, log request latency, use predict.py only. Write tests/test_api.py with TestClient: valid payload 200, bad enum 422, missing field 422, batch limit enforced, /health correct. Run make api and hit it with curl to confirm. Commit.
```

## Block 7: Dashboard (5:10-5:40)

```
Do Block 7. Streamlit app in app/ with the four pages from PRD FR9 (Overview, Model, Score a customer, Campaign simulator). Use predict.py and the cost module directly. Campaign simulator: sliders for budget, offer cost, success rate, CLV; ranked target list; expected profit; CSV download. Cache the model load. Run it and verify every page renders with the real model. Commit.
```

## Block 8: Packaging, docs, finish (5:40-6:00)

```
Do Block 8. Multi-stage non-root Dockerfile, docker-compose.yml (api :8000, app :8501), GitHub Actions workflow (ruff, black --check, pytest with coverage), scripts/drift_check.py (PSI per feature, flag > 0.2), reports/MODEL_CARD.md (intended use, data, metrics with CIs, slices, limitations, ethics, assumption caveats), and README.md (architecture diagram, results table with REAL numbers, quickstart, screenshots placeholders).
Verify: from a clean clone, make setup data train evaluate works, docker compose up serves both services, coverage >= 80%.
Then walk the PRD section 12 checklist and tell me honestly what is done, what is not, and which PRD targets were missed.
```

---

## Habits that keep you inside 6 hours
- Start Block 3's training before you review Block 2's code review notes, since it runs in the background.
- If a block overruns by more than 15 minutes, use the buffer options in PRD section 10 rather than skipping tests.
- Keep one Claude Code session per concern (main build, plus a second one for Docker/CI/README) once Block 3 starts.
- Don't open the test-set results until Block 4's evaluate step.
