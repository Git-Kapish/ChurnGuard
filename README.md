# ChurnGuard

End-to-end telecom churn prediction: data pipeline, tracked models, calibrated probabilities,
a cost-based targeting threshold, SHAP explanations, a FastAPI service and a Streamlit dashboard,
packaged with Docker. See [PRD.md](PRD.md) for scope and [CLAUDE.md](CLAUDE.md) for build rules.

> **Status:** Block 0 (scaffold). Results, architecture diagram and screenshots arrive in Block 8.

## Quickstart

```bash
make setup      # venv (Python 3.11) + dependencies
make data       # download IBM Telco churn CSV + checksum
make test       # pytest with coverage
make lint       # ruff + black --check
```

On Windows without GNU make, use `mingw32-make` (or `choco install make`).

If `make data` cannot reach the download URL, place the Kaggle copy
(`WA_Fn-UseC_-Telco-Customer-Churn.csv`) at `data/raw/telco_churn.csv` and re-run `make data`
to record its checksum.

## Business simulation

Cost, success-rate and lifetime-value numbers in `configs/business.yaml` are **assumptions**,
not measured facts. Any profit figure derived from them is an assumption-driven simulation.

## TODO (Block 8)

Architecture diagram, results table with real numbers, screenshots, Docker instructions.
