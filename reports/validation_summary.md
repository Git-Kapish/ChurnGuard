# Validation summary (model selection, calibration, threshold)

All numbers on this page come from the validation split. Dollar figures are an **assumption-driven simulation** (offer cost 20, success rate 30%, lifetime value 500, budget 3000).

## Tuned candidates (fit on train)

| Model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| logistic_regression | 0.6355 | 0.8469 | 0.1687 |
| xgboost (winner) | 0.6557 | 0.8507 | 0.1639 |

## Calibration (chosen: **sigmoid**, lowest validation Brier)

| Version | Brier | PR-AUC | ROC-AUC |
|---|---|---|---|
| uncalibrated | 0.1639 | 0.6557 | 0.8507 |
| isotonic | 0.1351 | 0.6468 | 0.8479 |
| sigmoid | 0.1340 | 0.6538 | 0.8497 |

## Threshold

Chosen threshold **0.140** targets 561 validation customers for a net value of 27,480. The break-even probability c / (s * CLV) is 0.133; maximizing model-expected value gives 0.135.

| Strategy | Targeted | Net value | Share of churners reached |
|---|---|---|---|
| target_nobody | 0 | 0 | 0.0% |
| target_everyone | 1056 | 20,880 | 100.0% |
| threshold_0.5 | 256 | 18,880 | 57.1% |
| chosen_threshold | 561 | 27,480 | 92.1% |
| best_top_k_within_budget | 149 | 13,520 | 39.3% |
