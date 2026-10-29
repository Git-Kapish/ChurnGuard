# Model comparison (training split, cross-validation)

5-fold stratified CV on 4930 training rows, seed 42, identical folds for every model. Values are mean ± std across folds. Data SHA256 `16320c9c1ec7`, git `7a2c4c9`.

PR-AUC is the primary metric. F1 and recall use a 0.5 threshold, which is not the business threshold (chosen in Block 4). Class-weighted models are not calibrated, so their Brier scores are pessimistic until calibration. The validation and test splits were not used here.

## 1. Model comparison (class weights)

| Model | PR-AUC | ROC-AUC | F1 | Recall | Brier |
|---|---|---|---|---|---|
| dummy | 0.265 ± 0.001 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.195 ± 0.000 |
| logistic_regression | 0.662 ± 0.021 | 0.844 ± 0.012 | 0.625 ± 0.023 | 0.794 ± 0.021 | 0.165 ± 0.007 |
| random_forest | 0.657 ± 0.024 | 0.842 ± 0.014 | 0.634 ± 0.024 | 0.674 ± 0.034 | 0.144 ± 0.007 |
| xgboost | 0.657 ± 0.018 | 0.841 ± 0.016 | 0.630 ± 0.032 | 0.764 ± 0.038 | 0.159 ± 0.010 |
| lightgbm | 0.651 ± 0.023 | 0.838 ± 0.014 | 0.623 ± 0.023 | 0.732 ± 0.026 | 0.158 ± 0.010 |

## 2. Imbalance study: class weights vs SMOTE

SMOTE is applied inside the CV pipeline (training folds only). It is adopted for a model only if it improves mean PR-AUC by more than 0.005.

| Model | Class weights PR-AUC | SMOTE PR-AUC | Difference | Used |
|---|---|---|---|---|
| logistic_regression | 0.662 ± 0.021 | 0.655 ± 0.027 | -0.0069 | class_weight |
| random_forest | 0.657 ± 0.024 | 0.646 ± 0.023 | -0.0104 | class_weight |
| xgboost | 0.657 ± 0.018 | 0.656 ± 0.025 | -0.0012 | class_weight |
| lightgbm | 0.651 ± 0.023 | 0.651 ± 0.020 | -0.0003 | class_weight |

## 3. Optuna tuning (30 trials each, top models by CV PR-AUC)

Tuned scores are selected on the same CV folds, so they are optimistically biased; Block 4 compares the tuned models on the validation split.

| Model | Imbalance | Default PR-AUC | Tuned PR-AUC | Tuned ROC-AUC | Tuned Brier |
|---|---|---|---|---|---|
| logistic_regression | class_weight | 0.662 ± 0.021 | 0.662 ± 0.021 | 0.844 ± 0.012 | 0.165 ± 0.007 |
| xgboost | class_weight | 0.657 ± 0.018 | 0.670 ± 0.029 | 0.847 ± 0.016 | 0.161 ± 0.009 |

Best parameters:

- **logistic_regression**: `{"C": 0.9846738873614566}`
- **xgboost**: `{"colsample_bytree": 0.5375566124836215, "gamma": 2.8409719053591482, "learning_rate": 0.042132515671934254, "max_depth": 2, "min_child_weight": 7, "n_estimators": 493, "reg_lambda": 2.263418326583197, "subsample": 0.5011539821063576}`
