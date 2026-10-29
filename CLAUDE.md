# CLAUDE.md: ChurnGuard

Read `PRD.md` first. It is the source of truth for scope, requirements, metrics, and the 6-hour block plan. If this file and the PRD conflict, ask before proceeding.

## Project in one line
End-to-end telecom churn prediction: pipeline → tracked models → calibrated, cost-optimal threshold → SHAP reasons → FastAPI + Streamlit → Docker.

## Stack
Python 3.11, pandas, scikit-learn, imbalanced-learn, XGBoost, LightGBM, Optuna, SHAP, MLflow, FastAPI, Pydantic v2, Streamlit, pytest, ruff, black, Docker.

## Commands (keep the Makefile in sync with these)
```
make setup        # venv + install deps (pyproject.toml)
make data         # download + validate + split
make train        # CV comparison, tuning, calibration, save model
make evaluate     # test-set evaluation (ONCE; needs --force to rerun)
make test         # pytest with coverage
make lint         # ruff + black --check
make api          # uvicorn on :8000
make app          # streamlit on :8501
make mlflow       # MLflow UI
make docker-up    # compose: api + app
```

## Hard rules
1. **Test set is touched once**, only by the evaluate step. Never use it for tuning, threshold choice, calibration choice, or feature decisions. Validation split does all of that.
2. **No leakage.** Every transformer lives inside an sklearn `Pipeline`/`ColumnTransformer` and is fit on training folds only. `tests/test_leakage.py` must stay green.
3. **One inference path.** API and dashboard both call `src/churnguard/predict.py`. Do not duplicate preprocessing or scoring logic.
4. **Config over constants.** Business numbers in `configs/business.yaml`, model/search settings in `configs/model.yaml`. Seeds fixed (`SEED=42`).
5. **Primary metric is PR-AUC.** Never report accuracy as a headline number.
6. **Report real numbers.** If a PRD target is missed, say so and explain. Never tune on test to hit a target.
7. **Business simulation is an assumption.** Label cost/CLV/success-rate outputs as assumption-driven everywhere (UI, README, model card).
8. **Raw data is read-only.** Never modify `data/raw/`. Never commit data files or secrets.
9. Do not add scope outside the PRD (no deep learning, cloud deploy, auth, streaming). Do not cut anything inside it.

## Code conventions
- Type hints on all functions; docstrings on public ones.
- Small modules under `src/churnguard/`; no logic in notebooks beyond EDA.
- Pydantic v2 models for API schemas; enums for categorical fields.
- Logging via `logging`, not `print`.
- Every module that does work gets a test, or a stated reason why not.
- Run `make lint && make test` before declaring any block done.

## Workflow per block
1. State the plan in a few lines (files to create, tests to write).
2. Implement, then write/run tests.
3. Run `make lint && make test`.
4. Report: what was built, test results, any deviation from the PRD.
5. Commit with a conventional message (`feat:`, `test:`, `docs:`) after each block.

For long training runs, run them in the background and continue with independent work (API schemas, Docker files, docs).

## Definition of done (project)
See PRD section 12 checklist. A block is done only when its "done when" condition in PRD section 10 is verifiably true.

## Known pitfalls to handle
- `TotalCharges` has blank strings for `tenure == 0` rows: coerce numeric, fill 0.
- "No internet service"/"No phone service" categories: normalize to "No".
- SHAP on `CalibratedClassifierCV`: explain the base estimator, calibrate probabilities separately, document it in the model card.
- SMOTE must be inside the CV pipeline (imblearn `Pipeline`), never applied before splitting.
- Docker image: non-root user, model artifact copied in, versions pinned.
