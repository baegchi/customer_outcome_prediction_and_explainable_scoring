# Credit Risk Scorecard — Lending Club Default Prediction

An end-to-end credit risk model that predicts loan default probability using only origination-time features, built on 1.35M resolved Lending Club loans (2007–2018). Designed as a portfolio project demonstrating the full ML engineering lifecycle from data ingestion through OOT evaluation.

---

## Results at a Glance

| Metric | CV (Development) | OOT (2017–2018) |
|---|---:|---:|
| **ROC-AUC** | 0.7225 | **0.6977** |
| **Gini** | 0.4450 | **0.3955** |
| **KS** | 36.20% | **28.60** |
| **PR-AUC** | 0.5140 | **0.3756** |

Best model: Optuna-tuned LightGBM on 36 WoE-transformed features.  
Recommended decision threshold: **t = 0.22–0.25** (catches 55–64% of defaults at 58–67% approval rate).

---

## What We Built

A **production-ready credit scorecard pipeline** that:

1. **Loads and cleans** 2.26M raw Lending Club records, removing 49 post-origination leakage columns
2. **Splits chronologically**: Train (≤2015) → Dev (2016) → OOT (2017–2018), with programmatic guards against data leakage
3. **Transforms features** via Weight of Evidence (WoE) binning with Information Value (IV) selection — 36 of 100 origination-time features survive IV ≥ 0.02
4. **Benchmarks 5 models** through identical stratified CV: Logistic Regression, Random Forest, XGBoost, default LightGBM, Optuna-tuned LightGBM
5. **Tunes hyperparameters** via Optuna TPE sampler with CV-inside-objective discipline
6. **Evaluates exactly once on OOT** — the holdout is never touched until the final evaluation
7. **Explains predictions** via SHAP (global importance + per-borrower decomposition)

---

## Why Each Major Decision Was Made

### Why exclude LC's own grade/sub_grade/int_rate?
These are **Lending Club's own credit assessment** — using them would build a model that predicts LC's decisions, not default risk. By excluding all 49 post-origination and LC-underwriting columns, we build an **independent** risk model that could be used at any lender, not just LC.

### Why WoE transformation instead of raw features?
WoE binning converts each feature into its **log-odds contribution**, producing monotonic, interpretable features that respect the scorecard tradition in banking. It also handles missing values, rare categories, and non-linearities automatically. The tradeoff: we lose feature interactions, which is why tree models barely outperform Logistic Regression on WoE features.

### Why chronological split instead of random?
Random train/test splits leak **future information** in time-series credit data. A loan originated in 2018 may share temporal patterns (macro conditions, LC policy changes) with its neighbors in the test set. Chronological splitting ensures the model is evaluated on genuinely unseen future data — the only honest test for a credit model.

### Why is the threshold 0.22 instead of 0.50?
The default rate is ~21% — most predicted probabilities are well below 0.50. At t=0.50, the model catches only **8% of defaults** (catastrophic for a lender). The threshold is set at 0.22 because the asymmetric loss ratio (charge-off cost ~5× lost margin) means missing a default is far costlier than rejecting a good borrower. See [docs/final_results.md](docs/final_results.md) §5 for the full analysis.

### Why does the tuned LightGBM only marginally beat Logistic Regression?
Because all models operate on **WoE-transformed features**, which are already monotonic univariate summaries. Tree models' power comes from modeling feature interactions — but WoE features have already compressed the information into single-feature log-odds. With a richer feature set (raw + WoE), trees would likely dominate more decisively.

---

## Project Structure

```
money/
├── configs/                      # Data paths and pipeline configuration
│   └── data.yaml
├── data/                         # Raw and interim data (not committed)
├── docs/                         # All documentation
│   ├── project_specification.md  # Problem definition, success criteria
│   ├── project_instruction.md    # 20-phase development roadmap
│   ├── validation_strategy.md    # Split design and maturity bias analysis
│   ├── leakage_audit.md          # 49 excluded columns with rationale
│   ├── model_experiments.md      # Model comparison and analysis
│   ├── final_results.md          # OOT evaluation, threshold, narrative
│   ├── explainability.md         # SHAP interpretation (global + local)
│   └── architecture.md           # System design
├── experiments/                  # Structured experiment records
│   ├── benchmark_summary.json    # All models' CV metrics
│   ├── optuna_best_config_*.yaml # Best hyperparameters
│   ├── optuna_study.db           # SQLite trial history
│   ├── oot_evaluation_results.json
│   └── runs/                     # Per-model fold-level metrics
├── models/                       # Serialized model artifacts
├── reports/                      # SHAP figures and data
│   ├── figures/                  # PNG visualizations
│   ├── shap_global_importance.*  # Feature importance tables
│   └── shap_local_explanations.json
├── src/
│   ├── data/                     # Loading, cleaning, splitting
│   │   ├── load.py               # Schema-validated CSV/GZ loader
│   │   ├── clean.py              # Leakage removal, target encoding
│   │   └── split.py              # Chronological TRAIN/DEV/OOT split
│   ├── features/
│   │   ├── build.py              # Feature engineering pipeline
│   │   └── woe.py                # WoE/IV transformation (optbinning)
│   ├── models/
│   │   ├── baseline.py           # Logistic Regression scorecard pipeline
│   │   ├── benchmark.py          # RF/XGB/LGBM unified experiment runner
│   │   ├── tune.py               # Optuna hyperparameter optimization
│   │   └── predict.py            # Inference pipeline
│   ├── evaluation/
│   │   ├── metrics.py            # ROC-AUC, Gini, KS, PR-AUC, F1, CM
│   │   └── threshold.py          # Threshold sweep and optimization
│   └── explainability/
│       └── shap_report.py        # Global + local SHAP analysis
└── tests/                        # 97 unit tests across 13 modules
    ├── test_load.py              # Data loading & schema validation
    ├── test_clean.py             # Cleaning & leakage prevention
    ├── test_split.py             # Chronological split integrity
    ├── test_features.py          # Feature engineering
    ├── test_woe.py               # WoE/IV correctness
    ├── test_metrics.py           # Metric calculations (incl. known KS)
    ├── test_baseline.py          # LR pipeline training & inference
    ├── test_benchmark.py         # Multi-model benchmark harness
    ├── test_tune.py              # Optuna integration
    ├── test_explainability.py    # SHAP report generation
    ├── test_threshold.py         # Threshold optimization
    ├── test_pipeline.py          # End-to-end prediction pipeline
    └── test_utils.py             # Configuration & logging utilities
```

---

## Honest Limitations

### 1. OOT Truncation Bias
The OOT set (2017–2018 vintages) only contains loans that resolved by the December 2018 dataset snapshot. Late-2018 originations that are still `Current` are excluded, biasing the OOT toward early defaults and early payoffs. OOT metrics are therefore **pessimistic** — a properly matured 2017–2018 vintage would likely produce different (probably better) metrics.

### 2. Only 5 Optuna Trials
The hyperparameter search was limited to 5 trials due to compute constraints. A proper search (50–100 trials) would likely find a better LightGBM configuration and potentially close more of the gap with proprietary scorecards. The current best config (num_leaves=17, reg_alpha=4.91) was found by trial 3 of 5 — it may not be near-optimal.

### 3. No Feature Interactions
All models operate on univariate WoE features. The pipeline does not engineer interaction terms (e.g., DTI × FICO), polynomial features, or aggregate bureau-depth features. A hybrid approach using both WoE and raw features would likely improve tree model performance substantially.

---

## Future Improvements

1. **Full Optuna budget** (50+ trials) with early stopping and learning curve analysis
2. **Raw + WoE hybrid features** for tree models, preserving WoE-only for the regulatory LR model
3. **Population Stability Index (PSI)** monitoring for detecting feature drift across vintages
4. **SHAP on production model** — run full TreeExplainer on the 36-feature tuned LightGBM across the entire OOT set
5. **Fairness audit** — test for disparate impact across `addr_state`, `home_ownership`, and other proxy-protected features
6. **Calibration analysis** — verify that predicted probabilities match observed default rates across deciles

---

## Running the Pipeline

```bash
# Environment setup
export PYTHONPATH="/DATA/suraj/pip_pkgs:/DATA/suraj/m1/money"

# Run tests (97 tests, ~30 seconds)
python3 -m pytest tests/ -v

# Run benchmark suite
python3 src/models/benchmark.py

# Run Optuna tuning
python3 src/models/tune.py

# Generate SHAP report
python3 src/explainability/shap_report.py
```

---

## Key Documentation

| Document | What It Covers |
|---|---|
| [Project Specification](docs/project_specification.md) | Problem definition, target encoding, success criteria |
| [Validation Strategy](docs/validation_strategy.md) | Why chronological split, maturity bias analysis |
| [Leakage Audit](docs/leakage_audit.md) | 49 excluded columns with category-by-category justification |
| [Model Experiments](docs/model_experiments.md) | Full 5-model comparison, tree complexity analysis |
| [Final Results](docs/final_results.md) | OOT evaluation, threshold recommendation, benchmark verification |
| [Explainability](docs/explainability.md) | SHAP global importance + 3 local examples, correlation ≠ causation |

---

*Built as a portfolio project demonstrating end-to-end ML engineering for credit risk — from data ingestion and leakage prevention through model selection, hyperparameter tuning, threshold optimization, and SHAP explainability.*
