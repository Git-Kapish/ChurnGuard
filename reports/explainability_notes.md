# Explainability notes

Computed on the validation split (1056 customers) with the base (uncalibrated) XGBoost pipeline. SHAP values are in log-odds of the base model; the calibrated model supplies the probabilities shown to users. Calibration is a monotone map, so the direction of a factor is preserved, but SHAP magnitudes are not calibrated probability points.

Columns are grouped by customer attribute (all Contract columns together, and so on) for both methods, so the rankings compare like with like.

Spearman rank correlation between mean |SHAP| and permutation importance: **0.77**.

| Attribute | Mean abs SHAP | SHAP rank | PR-AUC drop when shuffled | Permutation rank |
|---|---|---|---|---|
| Contract | 0.674 | 1 | +0.0720 | 2 |
| Tenure | 0.512 | 2 | +0.1010 | 1 |
| Internet service | 0.354 | 3 | +0.0422 | 3 |
| Billing history | 0.288 | 4 | +0.0238 | 4 |
| Monthly charges | 0.173 | 5 | +0.0133 | 5 |
| Payment method | 0.172 | 6 | +0.0054 | 8 |
| Paperless billing | 0.158 | 7 | +0.0056 | 7 |
| Security or support add-on | 0.121 | 8 | +0.0083 | 6 |
| Streaming movies | 0.120 | 9 | +0.0033 | 10 |
| Dependents | 0.083 | 10 | -0.0019 | 20 |
| Multiple lines | 0.080 | 11 | -0.0040 | 21 |
| Online security | 0.065 | 12 | +0.0011 | 14 |
| Online backup | 0.063 | 13 | +0.0014 | 13 |
| Tech support | 0.044 | 14 | +0.0025 | 11 |
| Streaming TV | 0.042 | 15 | +0.0010 | 15 |
| Senior citizen | 0.041 | 16 | +0.0015 | 12 |
| Number of add-on services | 0.036 | 17 | -0.0005 | 19 |
| Phone service | 0.032 | 18 | +0.0044 | 9 |
| Gender | 0.020 | 19 | +0.0005 | 17 |
| Partner | 0.013 | 20 | -0.0003 | 18 |
| Device protection | 0.007 | 21 | +0.0007 | 16 |

## Disagreements

- **Dependents**: SHAP rank 10, permutation rank 20 (gap 10).
- **Device protection**: SHAP rank 21, permutation rank 16 (gap 5).
- **Multiple lines**: SHAP rank 11, permutation rank 21 (gap 10).
- **Phone service**: SHAP rank 18, permutation rank 9 (gap 9).
- **Senior citizen**: SHAP rank 16, permutation rank 12 (gap 4).

## Reading the comparison

- The same five attributes lead both rankings, so the main drivers agree.
- Shuffling Partner, Number of add-on services, Dependents, Multiple lines slightly *improved* held-out PR-AUC. The model uses these attributes (non-zero SHAP) but they add no reliable signal on unseen customers. Drops this small are also within permutation noise, so read them as 'no evidence of value', not as proof of harm.
- SHAP measures how much an attribute moves the model's own predictions; permutation importance measures how much held-out PR-AUC falls when it is scrambled. Correlated attributes can rank differently under the two, and shuffling one attribute can create unrealistic combinations that inflate its permutation score.
- Both describe this model, not causal effects on customer behavior.
