# Architecture — Customer Outcome Prediction and Explainable Scoring

**Project type**: ML Portfolio / Course project  
**Scale**: 2.26M rows, 151 columns, single machine  
**Philosophy**: Demonstrate strong ML engineering practices; no more infrastructure than the project requires.

---

## 1. Folder Structure

```text
money/                                   ← project root
│
├── README.md                            ← project overview, setup, how to run
├── requirements.txt                     ← pinned pip dependencies
├── pyproject.toml                       ← project metadata, optional tool config
│
├── configs/                             ← YAML configuration files (not code)
│   ├── data.yaml                        ← paths, column lists, exclusion lists
│   ├── features.yaml                    ← feature engineering spec
│   └── model.yaml                       ← hyperparameter search spaces, seeds
│
├── data/
│   ├── raw/                             ← symlink to kagglehub cache; never commit the 1.6 GB CSV
│   │   └── accepted_2007_to_2018Q4.csv  ← (optional symlink) actual file: kagglehub/datasets/wordsforthewise/lending-club/versions/3/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv
│   ├── interim/                         ← cleaned, target-encoded, split parquet files
│   │   ├── train.parquet
│   │   ├── dev.parquet
│   │   └── oot.parquet
│   └── processed/                       ← WoE-transformed, model-ready arrays
│       ├── X_train.parquet
│       ├── X_dev.parquet
│       └── X_oot.parquet
│
├── notebooks/                           ← exploratory and reporting notebooks
│   ├── 01_data_understanding.ipynb      ← EDA, distributions, missing patterns
│   ├── 02_data_quality.ipynb            ← cleaning decisions, anomaly detection
│   ├── 03_feature_engineering.ipynb     ← feature creation, transformation review
│   ├── 04_woe_iv.ipynb                  ← WoE/IV binning, variable selection
│   ├── 05_baseline.ipynb                ← Logistic Regression baseline
│   ├── 06_model_benchmark.ipynb         ← RF, XGBoost, LightGBM comparison
│   ├── 07_hyperparameter_tuning.ipynb   ← Optuna study, convergence plots
│   ├── 08_threshold_optimization.ipynb  ← precision-recall tradeoff analysis
│   └── 09_explainability.ipynb          ← SHAP global + local explanations
│
├── src/                                 ← importable Python package
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                    ← load_raw(), validate_schema(), drop_metadata_rows()
│   │   ├── cleaner.py                   ← encode_target(), exclude_leakage_cols(), clean_dtypes()
│   │   └── splitter.py                  ← chronological_split(df, train_end, dev_end)
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineer.py                  ← feature creation (ratios, date features, etc.)
│   │   ├── imputer.py                   ← fit/transform missing value strategy
│   │   └── woe.py                       ← WoEBinning class: fit(X_train, y_train), transform(X)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                     ← train_model(X, y, config) -> model artifact
│   │   ├── predict.py                   ← predict_proba(model, X) -> scores
│   │   └── experiment.py                ← Experiment class: log config, metrics, artifacts
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                   ← roc_auc, pr_auc, gini, ks_stat, confusion_matrix_report
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── shap_analysis.py             ← global_shap(), local_shap(), summary_plot()
│   └── utils/
│       ├── __init__.py
│       ├── config.py                    ← load_yaml_config()
│       └── logging.py                   ← get_logger()
│
├── tests/                               ← pytest unit tests
│   ├── conftest.py                      ← shared fixtures (small synthetic datasets)
│   ├── test_loader.py                   ← schema validation, metadata row detection
│   ├── test_cleaner.py                  ← target encoding, leakage column removal
│   ├── test_splitter.py                 ← chronological split boundaries, no leakage
│   ├── test_woe.py                      ← WoE fit/transform, train-only fitting
│   ├── test_metrics.py                  ← gini = 2*auc-1, ks calculation
│   └── test_pipeline.py                 ← end-to-end on mini dataset
│
├── experiments/                         ← experiment outputs (gitignored large files)
│   ├── optuna_study.db                  ← Optuna SQLite study
│   └── runs/
│       └── {experiment_id}/
│           ├── config.yaml
│           ├── metrics.json
│           └── feature_importance.csv
│
├── models/                              ← serialized model artifacts (gitignored)
│   ├── preprocessor.pkl                 ← fitted imputer + encoder + WoE pipeline
│   ├── feature_list.json                ← ordered list of features used at training
│   ├── woe_bins.json                    ← WoE bin definitions (reproducible)
│   └── final_model.pkl                  ← final trained model
│
├── reports/                             ← final output artifacts
│   ├── figures/                         ← ROC, PR, SHAP plots
│   └── final_evaluation.md             ← OOT evaluation report
│
└── docs/                                ← project documentation (this folder)
    ├── project_instruction.md           ← original brief (do not modify)
    ├── project_specification.md         ← Phase 1: problem formulation
    ├── data_dictionary.md               ← Phase 2: all 151 columns documented
    ├── leakage_audit.md                 ← Phase 2: leakage decisions
    ├── validation_strategy.md           ← Phase 3: split design
    ├── feature_engineering.md           ← Phase 4: FE spec (to be created)
    ├── woe_iv.md                        ← Phase 5: WoE/IV methodology (to be created)
    ├── model_experiments.md             ← Phase 7-10: experiment log (to be created)
    ├── explainability.md                ← Phase 13: SHAP findings (to be created)
    ├── final_results.md                 ← Phase 12: OOT evaluation (to be created)
    └── architecture.md                  ← this file
```

---

## 2. Data Flow

```
raw CSV (located at: kagglehub/datasets/wordsforthewise/lending-club/versions/3/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv)
  │
  ▼ src/data/loader.py
  Drop metadata rows (33 rows with null loan_status)
  Parse dtypes, dates
  │
  ▼ src/data/cleaner.py
  Encode target (loan_status → default 0/1, exclude Current/Late/Grace)
  Remove leakage columns (per leakage_audit.md)
  Remove identifiers (id, member_id, url)
  Remove constants (policy_code, deferral_term, hardship_length)
  │
  ▼ src/data/splitter.py
  Chronological split on issue_d:
    TRAIN  ← issue_d <= 2015-12-31
    DEV    ← 2016-01-01 <= issue_d <= 2016-12-31
    OOT    ← issue_d >= 2017-01-01
  Save to data/interim/*.parquet
  │
  ▼ src/features/engineer.py  (fitted on TRAIN only)
  Credit age = months between earliest_cr_line and issue_d
  Log transforms (annual_inc, loan_amnt, revol_bal, dti)
  Ratio features (revol_bal/annual_inc, loan_amnt/annual_inc)
  Missing indicators for columns with >10% missing
  │
  ▼ src/features/imputer.py  (fitted on TRAIN only)
  Median imputation for numerical
  Mode imputation for categorical
  │
  ▼ src/features/woe.py  (fitted on TRAIN folds only, never on DEV/OOT)
  WoE/IV binning
  Variable selection by IV threshold
  │
  ▼ src/models/train.py
  Cross-validation on TRAIN (5-fold stratified)
  Hyperparameter tuning with Optuna on TRAIN
  Final training on full TRAIN
  Evaluate on DEV (threshold optimization only)
  Final evaluation on OOT (once, after model frozen)
  │
  ▼ src/evaluation/metrics.py
  ROC-AUC, PR-AUC, Gini, KS, Confusion Matrix
  │
  ▼ src/explainability/shap_analysis.py
  Global SHAP (TreeExplainer on sample)
  Local SHAP (selected predictions)
```

---

## 3. Configuration Over Hardcoding

Column lists, split dates, hyperparameter search spaces, and thresholds live in `configs/*.yaml` — not hardcoded in source files. This allows reproducing experiments with different settings without touching code.

Example `configs/data.yaml`:
```yaml
raw_path: "kagglehub/datasets/wordsforthewise/lending-club/versions/3/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"
random_seed: 42

splits:
  train_end: "2015-12-31"
  dev_end: "2016-12-31"
  oot_start: "2017-01-01"

exclude_columns:
  leakage_post_outcome:
    - out_prncp
    - out_prncp_inv
    - total_pymnt
    - total_pymnt_inv
    - total_rec_prncp
    - total_rec_int
    - total_rec_late_fee
    - recoveries
    - collection_recovery_fee
    - last_pymnt_d
    - last_pymnt_amnt
    - next_pymnt_d
  leakage_credit_refresh:
    - last_fico_range_high
    - last_fico_range_low
    - last_credit_pull_d
  leakage_lc_underwriting:
    - grade
    - sub_grade
    - int_rate
    - installment
    - funded_amnt
    - funded_amnt_inv
  leakage_hardship:
    - hardship_flag
    - hardship_type
    - hardship_reason
    - hardship_status
    - deferral_term
    - hardship_amount
    - hardship_start_date
    - hardship_end_date
    - payment_plan_start_date
    - hardship_length
    - hardship_dpd
    - hardship_loan_status
    - orig_projected_additional_accrued_interest
    - hardship_payoff_balance_amount
    - hardship_last_payment_amount
  leakage_settlement:
    - debt_settlement_flag
    - debt_settlement_flag_date
    - settlement_status
    - settlement_date
    - settlement_amount
    - settlement_percentage
    - settlement_term
  identifiers:
    - id
    - member_id
    - url
  constants:
    - policy_code
  high_cardinality_text:
    - emp_title
    - desc
    - title
  other:
    - pymnt_plan

target_column: "loan_status"
target_positive: ["Charged Off", "Does not meet the credit policy. Status:Charged Off", "Default"]
target_negative: ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]
target_exclude: ["Current", "Late (31-120 days)", "In Grace Period", "Late (16-30 days)"]
```

---

## 4. What Is Deliberately Excluded

This is a portfolio project. The following are explicitly **not** included:

| Excluded | Reason |
|---|---|
| Docker / containerization | Not needed for a reproducible local ML project |
| MLflow or similar tracking | Experiments tracked manually via JSON + SQLite; Optuna covers tuning |
| REST API / serving layer | No inference serving requirement |
| Airflow / orchestration | No pipeline scheduling required |
| DVC / data versioning | Single fixed dataset version; git + .gitignore sufficient |
| Cloud infrastructure | Everything runs locally |
| Database (beyond Optuna SQLite) | Not needed |
| CI/CD | Out of scope |

---

## 5. Reproducibility Contract

Any reviewer with the raw CSV (in kagglehub/ cache, or re-downloaded via download.py) and `requirements.txt` should be able to:

1. Run `python src/data/loader.py` → produce `data/interim/*.parquet`
2. Run `python src/features/engineer.py` → produce `data/processed/*.parquet`
3. Run `python src/models/train.py --config configs/model.yaml` → produce `models/final_model.pkl`
4. Run `python src/evaluation/metrics.py --split oot` → reproduce OOT metrics

All random seeds, bin definitions, and feature lists are persisted as artifacts.

---

*This architecture is intentionally lean. Add complexity only when there is a clear technical requirement.*
