# Evaluation on the test split

1057 customers, churn rate 26.6%. Decision threshold 0.140 was chosen on the validation split. Confidence intervals are 1000-sample bootstrap percentile intervals.

| Metric | Value | 95% CI |
|---|---|---|
| PR-AUC (primary) | 0.673 | [0.613, 0.731] |
| ROC-AUC | 0.842 | [0.816, 0.869] |
| Brier score | 0.138 | [0.125, 0.151] |
| Recall at threshold | 0.861 | [0.820, 0.901] |
| Top-decile lift | 3.052 | [2.690, 3.390] |
| Precision at threshold | 0.454 | |
| F1 at threshold | 0.595 | |
| Log loss | 0.426 | |

Confusion matrix at the threshold: TN 485, FP 291, FN 39, TP 242.

## PRD targets

| Metric | Value | Target | Met |
|---|---|---|---|
| roc_auc | 0.842 | >= 0.84 | yes |
| pr_auc | 0.673 | >= 0.64 | yes |
| recall | 0.861 | >= 0.7 | yes |
| top_decile_lift | 3.052 | >= 2.8 | yes |
| brier | 0.138 | <= 0.15 | yes |
| net_value_vs_target_everyone | 4630.000 | > 0 | yes |

## Targeting rules (assumption-driven simulation, actual outcomes)

| Strategy | Targeted | Net value | Share of churners reached |
|---|---|---|---|
| target_nobody | 0 | 0 | 0.0% |
| target_everyone | 1057 | 21,010 | 100.0% |
| threshold_0.5 | 229 | 17,620 | 52.7% |
| chosen_threshold | 533 | 25,640 | 86.1% |
| best_top_k_within_budget | 148 | 13,240 | 38.4% |

## Slice: contract

| Group | n | Churn rate | ROC-AUC | PR-AUC | Precision | Recall |
|---|---|---|---|---|---|---|
| Month-to-month | 572 | 42.5% | 0.755 | 0.715 | 0.474 | 0.934 |
| One year | 229 | 14.4% | 0.730 | 0.270 | 0.300 | 0.455 |
| Two year | 256 | 2.0% | 0.814 | 0.143 | 0.000 | 0.000 |

## Slice: tenure_bucket

| Group | n | Churn rate | ROC-AUC | PR-AUC | Precision | Recall |
|---|---|---|---|---|---|---|
| 0-12 | 322 | 46.0% | 0.792 | 0.782 | 0.523 | 0.905 |
| 13-24 | 157 | 34.4% | 0.760 | 0.562 | 0.462 | 0.907 |
| 25-48 | 230 | 23.0% | 0.804 | 0.512 | 0.443 | 0.811 |
| 49+ | 348 | 7.5% | 0.838 | 0.328 | 0.216 | 0.615 |

## Slice: senior_citizen

| Group | n | Churn rate | ROC-AUC | PR-AUC | Precision | Recall |
|---|---|---|---|---|---|---|
| No | 896 | 23.4% | 0.841 | 0.658 | 0.424 | 0.829 |
| Yes | 161 | 44.1% | 0.779 | 0.709 | 0.553 | 0.958 |
