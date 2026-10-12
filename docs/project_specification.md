# Project Specification — Customer Outcome Prediction and Explainable Scoring

**Dataset**: Lending Club Accepted Loans 2007–2018 Q4  
**Source**: Kaggle — `wordsforthewise/lending-club` v3  
**Discovery date**: 2026-09-04  
**Status**: Phase 1 — Requirements complete, pending implementation approval

---

## 1. Problem Context

Lending Club is a peer-to-peer lending marketplace. When a borrower applies for a personal loan, LC collects an application, pulls a credit bureau report, assigns its own risk grade, and lists the loan for investor funding. The outcome of each loan — whether the borrower fully repays or defaults — is recorded in `loan_status`.

The goal of this project is to build a **binary classification model** that predicts, at the time of loan origination, whether a loan will default. The model produces a **probability score** (0–1) that ranks loans from most likely to least likely to default. This score can be used by:

- A competing credit underwriter to approve/reject loans
- An investor to select which loans to fund
- A risk manager to set loan pricing

This is a **credit scorecard problem**: the output is a probability ranking, not a hardcoded yes/no decision. Threshold selection is a separate, downstream business decision.

---

## 2. What One Row Represents

Each row in the dataset is one **accepted loan application** issued by Lending Club between June 2007 and December 2018.

- Rows are unique: `id` is unique across all 2,260,701 rows.
- Each row has a single `loan_status` that records the final or current state of the loan at the dataset snapshot date (December 2018).
- `member_id` is 100% null — it cannot be used for borrower-level entity deduplication. Each row is treated as an independent loan.

---

## 3. Target Variable Definition

### 3.1 Raw loan_status distribution (from actual data)

| loan_status | Count | % of Total |
|---|---:|---:|
| Fully Paid | 1,076,751 | 47.63% |
| Current | 878,317 | 38.85% |
| Charged Off | 268,559 | 11.88% |
| Late (31-120 days) | 21,467 | 0.95% |
| In Grace Period | 8,436 | 0.37% |
| Late (16-30 days) | 4,349 | 0.19% |
| Does not meet the credit policy. Status:Fully Paid | 1,988 | 0.09% |
| Does not meet the credit policy. Status:Charged Off | 761 | 0.03% |
| Default | 40 | 0.002% |
| null | 33 | 0.001% |
| **Total** | **2,260,701** | **100%** |

### 3.2 Target encoding decision

**The "Current" category is the critical problem.** 878,317 rows (38.85%) represent loans that were still active at snapshot time (December 2018). Their final outcome is unknown — a "Current" loan from 2018 could eventually pay off or default. Including these in training would mean training on right-censored, unlabeled outcomes.

**Encoding rule:**

| Raw loan_status | Binary target | Reasoning |
|---|---|---|
| Fully Paid | **0** (non-default) | Loan repaid in full — definitive good outcome |
| Does not meet the credit policy. Status:Fully Paid | **0** | Same outcome; older policy cohort, status is resolved |
| Charged Off | **1** (default) | Loan principal declared unrecoverable — definitive bad outcome |
| Does not meet the credit policy. Status:Charged Off | **1** | Same outcome; older policy cohort |
| Default | **1** | 121+ days past due, pre-charge-off; effectively charged-off |
| Current | **EXCLUDE** | Outcome unknown at snapshot date; right-censored |
| Late (31-120 days) | **EXCLUDE** | Point-in-time delinquency — could resolve or charge off; ambiguous |
| In Grace Period | **EXCLUDE** | 1-15 days late — ambiguous outcome |
| Late (16-30 days) | **EXCLUDE** | 16-30 days late — ambiguous outcome |
| null (33 rows) | **EXCLUDE** | Metadata/footer rows embedded in CSV |

> **Note on the 33 null rows:** The `id` column contains the value
> `"Total amount funded in policy code 2: 521953170"` — a CSV footer row
> embedded in the file. These 33 rows with null `loan_status` are metadata
> and must be dropped during data loading before any analysis.

### 3.3 Resulting modeling dataset

| Class | Count | % of modeling set |
|---|---:|---:|
| **Non-default (0)** | 1,078,739 | **80.02%** |
| **Default (1)** | 269,360 | **19.98%** |
| **Total modeling rows** | **1,348,099** | **100%** |

- Excluded (Current + ambiguous + null): **912,602 rows** (40.37% of raw dataset)
- Class ratio: approximately **1:4** (default:non-default)

> **Design decision — Late/Grace exclusion:** The 34,252 Late and In Grace Period
> rows are excluded rather than labeled because their eventual outcome is uncertain.
> Including them as default introduces label noise (some will recover); including as
> non-default is worse. Exclusion produces cleaner labels. A sensitivity analysis
> should later test labeling Late (31-120 days) as default.

---

## 4. ML Problem Formulation

| Dimension | Definition |
|---|---|
| Problem type | Binary classification |
| Input X | Applicant and loan features known at origination time (post-leakage-removal) |
| Target y | `default` in {0, 1} as defined in §3.2 |
| Prediction output | Probability P(default=1 given X) in [0, 1] |
| Probability interpretation | Higher score = higher predicted risk of default |
| Positive class | default = 1 (Charged Off / Default) |
| Negative class | default = 0 (Fully Paid) |
| Threshold | Not fixed at 0.5; optimized post-training per business objective |
| Primary output | Probability score for ranking; classification is secondary |

The model is a **credit risk scorecard**, not a binary classifier. The primary objective is a well-calibrated ranking. The threshold is a business lever.

---

## 5. Success Criteria

### 5.1 Primary metrics (ranking quality — threshold-free)

| Metric | Benchmark | Why it matters |
|---|---|---|
| **ROC-AUC** | >= 0.65 (reproduce reported 0.6499) | P(model ranks random defaulter above random non-defaulter). Threshold-independent. Standard for credit scoring. |
| **Gini** | >= 0.30 (reproduce reported 0.2998) | Gini = 2 x ROC-AUC - 1. Common regulatory metric for credit scorecards. |
| **KS statistic** | >= 21 (reproduce reported 21.92) | Max separation between cumulative default and non-default distributions. Industry-standard credit metric. |
| **PR-AUC** | Report; compare to baseline prevalence (~0.20) | More informative than ROC-AUC under class imbalance. Not in original brief; added as rigorous supplement. |

> **On the reported numbers**: The brief claims ROC-AUC 64.99%, Gini 29.98%, KS 21.92
> on chronological OOT. These must be reproduced from data, not engineered backward.
> A Gini of ~30% is modest but realistic — LC has already pre-screened applicants,
> so range restriction reduces our discriminatory power vs. the full population.

### 5.2 Secondary metrics (threshold-dependent)

| Metric | Purpose |
|---|---|
| **Precision** | Of loans flagged as default, fraction that are actually default |
| **Recall** | Of actual defaults, fraction the model identifies |
| **F1** | Harmonic mean at chosen threshold |
| **Confusion matrix** | Absolute TP/FP/TN/FN counts |
| **Calibration** | Reliability diagram; Brier score. Ensures predicted probabilities match observed default rates. |

### 5.3 Minimum bar vs. naïve baseline

A model is considered materially useful if:
- ROC-AUC > 0.60 on OOT
- KS > 15 on OOT
- PR-AUC > 0.20 on OOT (the baseline prevalence)

---

## 6. Constraints

### 6.1 Dataset scale

- **2,260,701 rows x 151 columns** — full dataset  
- **1,348,099 rows** after target encoding (modeling set)
- Memory-aware operations required: use appropriate dtypes (int8/int16/float32)
- Cross-validation on 1M+ rows is expensive; use LightGBM/XGBoost with native CV and early stopping

### 6.2 Class imbalance

- 1:4 imbalance ratio (20% positive) is **moderate**, not severe
- Do NOT blindly apply SMOTE or aggressive oversampling on 1M+ rows
- Evaluate: class_weight='balanced', scale_pos_weight, is_unbalance, threshold optimization
- Primary metric for imbalance decisions: PR-AUC and KS (not accuracy)

### 6.3 Temporal validation — non-negotiable

- `issue_d` spans June 2007 – December 2018
- OOT data must never be used for training, feature selection, hyperparameter tuning, or threshold optimization
- All preprocessing (scalers, encoders, WoE bins) fitted only on training folds
- Full chronological split in `validation_strategy.md`

### 6.4 Data leakage — highest priority

- Dozens of post-outcome columns would not exist at prediction time
- LC's own underwriting outputs (grade, sub_grade, int_rate, installment) require explicit treatment (see `leakage_audit.md`)
- A model using post-outcome features will appear near-perfect and generalize to nothing

### 6.5 Reproducibility

- Random seed: `RANDOM_SEED = 42`
- All artifacts (model, preprocessor, WoE bins, feature list, metrics) persisted to `models/` and `experiments/`
- Optuna study saved to SQLite for resumability

### 6.6 Computational constraints

- Target: full training pipeline completable in <= 4 hours on 16 GB RAM single machine
- Profile memory before loading all 151 columns

---

## 7. What the Model Is Not

- Not a real-time decision system — no API or serving infrastructure required
- Not a Fair Lending model — protected-class analysis out of scope
- Not a causal model — SHAP explanations are correlational
- Not a survival model — censoring handled by exclusion, not Cox/survival framework

---

*See also: `data_dictionary.md`, `leakage_audit.md`, `validation_strategy.md`*
