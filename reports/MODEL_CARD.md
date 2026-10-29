# Model card: ChurnGuard churn model v1

A calibrated gradient-boosted model that scores telecom subscribers for churn risk, with plain-English
reasons per customer and a budget-aware targeting threshold. This card describes what the model is,
what it was measured on, and where it should not be trusted. Dollar figures are an
**assumption-driven simulation**, not measured outcomes (see section 7).

## 1. Model details

| | |
|---|---|
| Task | Binary classification: will the customer churn (leave)? Output is a probability. |
| Model | XGBoost (`max_depth` 2, 493 trees, learning rate 0.042, subsample 0.50, column sample 0.54, `min_child_weight` 7, L2 regularization 2.26, `gamma` 2.84), class-balanced sample weights |
| Calibration | Sigmoid (Platt) calibration, cross-fitted over 5 folds of the training split; the final probability averages the 5 calibrated fold models |
| Inputs | 19 customer attributes (demographics, services, contract, billing); 7 engineered features |
| Decision rule | Offer a retention contact when probability >= **0.1403** (see section 7) |
| Version / date | `v1`, trained 2026-10-29 |
| Software | Python 3.11.15, scikit-learn 1.8.0, XGBoost 3.2.0, LightGBM 4.7.0, imbalanced-learn 0.14.2, pandas 2.3.3, numpy 2.4.6 |
| Artifacts | `models/churn_model_v1.joblib` (calibrated model plus the uncalibrated base pipeline used for explanations), `models/metadata.json` |
| License / owner | Portfolio project by Kapish. The dataset is IBM's public sample data. |

## 2. Intended use

**Intended:** decision support for a retention team. Rank customers by churn risk, explain each score in
plain English, and choose whom to contact given a budget. A human reviews the target list.

**Not intended:**
- Decisions that disadvantage an individual (denying service, raising a price, credit or eligibility).
  The model has not been assessed for that and the offers it supports are benefits, not penalties.
- Customers unlike the training data (a different country, product, or era). It was fit on one public
  sample dataset.
- Reading the reasons as causes. They describe what drives the model's score, not what would change a
  customer's mind.
- Production use without revalidation on current data, a drift check, and a fairness review.

## 3. Data

- **Source:** IBM Telco Customer Churn sample (7,043 customers, 21 columns, 26.5% churn). SHA256
  `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`. The raw file is never modified
  and is not committed.
- **Cleaning:** 11 blank `TotalCharges` values (all brand-new customers with tenure 0) became 0;
  "No phone service" and "No internet service" were folded into "No" (the parent service flag is kept).
- **Splits:** stratified 70 / 15 / 15 with seed 42: **4,930 train / 1,056 validation / 1,057 test**.
  Churn rate is 26.5%, 26.5% and 26.6%. Every transformer is fit on training data only; a test checks that
  a customer's features do not change when other customers change.
- **Caveat:** this is a public sample dataset with unusually tidy structure (few missing values, simple
  categories) and a single snapshot in time. Real billing systems are messier, and nothing here
  captures change over time.

## 4. How the model was chosen

1. **Five models, 5-fold stratified cross-validation on the training split, identical folds** (mean ± std):

   | Model | PR-AUC | ROC-AUC | Brier |
   |---|---|---|---|
   | Dummy (prior) | 0.265 ± 0.001 | 0.500 | 0.195 |
   | Logistic regression | 0.662 ± 0.021 | 0.844 ± 0.012 | 0.165 |
   | Random forest | 0.657 ± 0.024 | 0.842 ± 0.014 | 0.144 |
   | XGBoost | 0.657 ± 0.018 | 0.841 ± 0.016 | 0.159 |
   | LightGBM | 0.651 ± 0.023 | 0.838 ± 0.014 | 0.158 |

   The four real models are within about 0.01 PR-AUC of each other, which is inside fold-to-fold noise.
2. **Imbalance study:** class weights vs SMOTE (inside the CV pipeline). SMOTE was never better
   (differences -0.0003 to -0.0104 PR-AUC), so class weights were used throughout.
3. **Tuning:** Optuna, 30 trials, on the top two models by CV PR-AUC. Logistic regression gained nothing;
   XGBoost went from 0.657 to 0.670 ± 0.029 (selected on the same folds, so optimistic).
4. **Winner on the validation split:** XGBoost, PR-AUC 0.656 vs 0.636 for logistic regression.
5. **Calibration, judged on validation:** the uncalibrated class-weighted model over-predicts
   (Brier 0.164). Sigmoid 0.134 beat isotonic 0.135 by a small margin; sigmoid was kept.
6. **Threshold:** chosen on validation (section 7).

The primary metric throughout is PR-AUC; accuracy is never used as a headline number.

## 5. Performance on the held-out test split

The test split (1,057 customers, 26.6% churn) was scored **once**, after every choice above was fixed.
Intervals are 1,000-sample bootstrap 95% confidence intervals.

| Metric | Result | 95% CI | PRD target |
|---|---|---|---|
| PR-AUC (primary) | **0.673** | [0.613, 0.731] | >= 0.64: met |
| ROC-AUC | **0.842** | [0.816, 0.869] | >= 0.84: met on the point estimate only |
| Recall at the 0.140 threshold | 0.861 | [0.820, 0.901] | >= 0.70: met |
| Top-decile lift | 3.05 | [2.69, 3.39] | >= 2.8: met |
| Brier score | 0.138 | [0.125, 0.151] | <= 0.15: met |
| Precision at the threshold | 0.454 | | |
| F1 at the threshold | 0.595 | | |
| Log loss | 0.426 | | |

Confusion matrix at the threshold: 485 true negatives, 291 false positives, 39 false negatives,
242 true positives.

**How to read this honestly**
- Every point estimate meets its target, but the intervals are wide. The ROC-AUC lower bound (0.816) is
  below its 0.84 target and the PR-AUC lower bound (0.613) is below 0.64, so these are borderline passes,
  not comfortable ones.
- The threshold is deliberately low, so the model flags about half of all customers. Recall is high
  (86%) and precision is modest (45%): more than half of the offers go to customers who would have stayed.
  That is the right trade only because an offer is cheap relative to the value of a saved customer under
  the assumed economics.
- Validation numbers (used to choose the threshold, so slightly optimistic): PR-AUC 0.654, ROC-AUC 0.850,
  Brier 0.134, top-decile lift 2.78. On validation the lift target (2.8) was narrowly missed; the test
  split cleared it. That swing is a reminder of the sampling noise in a 1,000-customer split.

## 6. Performance by slice (test split)

Metrics at the 0.140 threshold. Small groups give noisy numbers.

| Slice | n | Churn rate | ROC-AUC | PR-AUC | Precision | Recall |
|---|---|---|---|---|---|---|
| **Contract** | | | | | | |
| Month-to-month | 572 | 42.5% | 0.755 | 0.715 | 0.474 | 0.934 |
| One year | 229 | 14.4% | 0.730 | 0.270 | 0.300 | 0.455 |
| Two year | 256 | 2.0% | 0.814 | 0.143 | 0.000 | 0.000 |
| **Tenure (months)** | | | | | | |
| 0-12 | 322 | 46.0% | 0.792 | 0.782 | 0.523 | 0.905 |
| 13-24 | 157 | 34.4% | 0.760 | 0.562 | 0.462 | 0.907 |
| 25-48 | 230 | 23.0% | 0.804 | 0.512 | 0.443 | 0.811 |
| 49+ | 348 | 7.5% | 0.838 | 0.328 | 0.216 | 0.615 |
| **Senior citizen** | | | | | | |
| No | 896 | 23.4% | 0.841 | 0.658 | 0.424 | 0.829 |
| Yes | 161 | 44.1% | 0.779 | 0.709 | 0.553 | 0.958 |

- Two-year contracts have only about 5 churners in the test split, so a recall of 0 there is weak
  evidence either way. One-year recall (0.455) is the weakest well-populated slice.
- The model ranks long-tenure and long-contract customers less well and catches fewer of their churners
  at this threshold. If retention of those customers matters, evaluate them separately.

## 7. Business simulation (assumption-driven)

The threshold and every dollar figure rest on three **assumed** numbers in `configs/business.yaml`:
offer cost 20, offer success rate 30%, customer lifetime value 500 (plus a campaign budget of 3,000).
They are illustrations, not measurements. Expected net value of contacting a customer is
`p x 0.30 x 500 - 20`, so an offer pays off above a break-even probability of 0.133; the chosen threshold
(0.1403) maximizes label-based net value on the validation split.

| Strategy (test split, actual outcomes) | Customers contacted | Net value | Churners reached |
|---|---|---|---|
| Contact nobody | 0 | 0 | 0% |
| Contact everyone | 1,057 | 21,010 | 100% |
| Default 0.5 threshold | 229 | 17,620 | 52.7% |
| **Chosen threshold (0.140)** | 533 | **25,640** | 86.1% |
| Best list within the 3,000 budget | 148 | 13,240 | 38.4% |

- The chosen threshold beats "contact everyone" by 4,630 (about 22%) under these assumptions, which meets
  the PRD target of "strictly higher".
- "Contact everyone" is itself profitable here, because the 26.6% churn rate is well above the 13.3%
  break-even. The model's value is in skipping the low-risk half, not in making blanket outreach viable.
- The budget caps the list at 150 offers, so budget-limited campaigns earn less in total than the
  unconstrained optimum. Change the assumptions and every figure changes with them.

## 8. Explainability

- **Reasons per customer** are SHAP values from the **uncalibrated base estimator**, summed per customer
  attribute (so "Contract" is one factor, not four one-hot columns) and phrased in plain English with the
  customer's actual values. The probability shown comes from the calibrated model.
- **Why the base model:** `CalibratedClassifierCV` wraps five fitted copies of the model and SHAP cannot
  explain that wrapper. Calibration is a monotone map, so a factor that raises the base score also raises
  the calibrated probability (the two rank customers almost identically, Spearman > 0.98), but SHAP
  magnitudes are in the base model's log-odds, not calibrated percentage points.
- **Top drivers** (mean |SHAP|): contract, tenure, internet service, billing history, monthly charges.
- **SHAP vs permutation importance** (grouped by attribute, on validation): rank correlation 0.77. The same
  five attributes lead both. Shuffling Partner, number of add-on services, Dependents and Multiple lines
  slightly *improved* held-out PR-AUC, so those attributes show no reliable value. See
  `reports/explainability_notes.md`.

## 9. Limitations

- **One dataset, one snapshot.** No time dimension, so drift and seasonality are untested. A PSI drift
  check (`scripts/drift_check.py`) is provided as a monitoring demonstration, not a live service.
- **Small test split.** 1,057 customers give wide intervals (section 5). Differences of a point or two of
  AUC between models are not meaningful.
- **Associations, not causes.** Fiber-optic customers churn more; the data cannot say whether price,
  product quality, or who buys fiber is responsible. Acting on a reason (for example discounting) is a
  hypothesis to test, not a proven lever.
- **Probabilities are approximately calibrated**, not exact: calibration was fit on training-fold
  predictions and checked on validation (Brier 0.134) and test (0.138).
- **Offer effects are assumed.** The model predicts churn, not whether an offer will work.
- **Latency target narrowly missed.** The PRD asks for p95 under 100 ms per single prediction. Measured
  server-side through the API: **median 99 ms, p95 132 ms** over 45 requests in the Linux Docker container
  (16 CPUs), and median 334 ms, p95 962 ms over 42 requests running natively on a Windows laptop
  (both noisy). The cause is the 5-model calibration ensemble, which re-runs preprocessing five times per
  request. A single-model calibration would be faster but changes the model, so it was not done after the
  test split was scored; it remains an option, at the price of one deliberate second look at the test
  split.

## 10. Ethical considerations

- **Protected-style attributes are model inputs.** The model uses `gender` and `SeniorCitizen`. Their
  contribution is small (gender ranks 19th of 21 attributes by mean |SHAP|, senior citizen 16th), but they
  are present. Recommendation: drop `gender` in any real deployment; it adds almost nothing.
- **Disparities seen on the test split.** Senior citizens are flagged more often and with higher recall
  (0.958 vs 0.829) and precision (0.553 vs 0.424), but the model ranks them less well (ROC-AUC 0.779 vs
  0.841). Because they churn at 44% vs 23%, much of the higher flag rate reflects real base-rate
  differences, but this was not decomposed.
- **Not assessed:** gender slices, intersections, demographic parity or equalized-odds formal tests, and
  any effect of the offers themselves on customers. Do not describe this model as fairness-audited.
- **Offers are benefits.** The intended action is a discount or call, which limits harm from a wrong
  prediction. Using the score to degrade service or deny options for flagged customers would be out of
  scope and would need a separate review.
- **Privacy.** Real customer data would be personal data; the scoring API accepts an optional customer ID
  and logs request latency only, never inputs.

## 11. Reproducibility and operations

- `make setup data train evaluate` rebuilds everything from the public CSV. Seeds are fixed (42); two
  full training runs produced identical CV scores and best parameters.
- `make evaluate` scores the test split once per checkout and refuses to run again without `--force`. The
  guard is a per-checkout marker (`reports/.test_scored`, gitignored), separate from the committed
  results, so a fresh clone can reproduce the evaluation once.
- **Clean-copy verification.** `make setup data train evaluate test lint` was run on a copy of the
  repository with no data, virtualenv, MLflow store or git history: all steps succeeded in about 12 minutes
  (training 3 min 9 s; 179 tests, 92% coverage). The retrained model reproduced the committed test metrics
  exactly (maximum difference 0.0 over all 13 metrics, identical bootstrap intervals, hyperparameters and
  threshold). **Disclosure:** that reproduction was a second reading of the test split, in a separate
  checkout, of the identical model; no decision was made or changed because of it.
- **Docker.** The container (Linux, pinned `requirements.lock`, CPU-only XGBoost) returns the same
  probabilities as the native run (largest difference 7e-18 on a scored CSV).
- Experiments are tracked in MLflow (local SQLite store under `mlruns/`, gitignored).
- `metadata.json` records the data checksum, library versions and git hash at training time. It notes the
  working tree was not clean when the model was trained (`git_dirty: true`), so the recorded hash points
  at the commit just before the final code.
