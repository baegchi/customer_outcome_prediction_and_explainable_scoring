# Feature Engineering Specification — Customer Outcome Prediction

**Dataset**: Lending Club Accepted Loans 2007–2018 Q4  
**Based on**: `data_dictionary.md`, `leakage_audit.md`  
**Date**: 2026-09-04  
**Prediction moment**: Loan origination (all inputs must be known at that instant)

---

## 0. Design Principles

1. **No feature is created unless its motivation is grounded in credit-risk theory or the data dictionary.** No arbitrary polynomial terms, interaction explosions, or embedding layers.
2. **Every transformation that learns from data (imputation values, clip bounds, log offsets) must be fitted on the TRAIN set only and applied unchanged to DEV and OOT.**
3. **Leakage check is re-run after engineering.** A derived feature can introduce leakage even if its inputs are clean (e.g., dividing by `total_pymnt` would be leakage even though `loan_amnt` is not).
4. **Missing-value strategy is part of the feature definition.** It is not a separate, ad-hoc step.
5. Features are grouped into three tiers:
   - **Tier 1 — Direct raw features**: cleaned / type-corrected originals passed as-is to WoE
   - **Tier 2 — Derived numeric features**: new columns computed from raw columns
   - **Tier 3 — Missing-value indicators**: binary flags for informative missingness

---

## 1. Parsing and Type-Correction (Pre-Engineering Step)

These are not "features" — they are prerequisite type fixes applied before any engineering.

| Operation | Input | Output | Notes |
|---|---|---|---|
| Parse `issue_d` | object `"Mar-2016"` | datetime | `pd.to_datetime(df['issue_d'], format='%b-%Y')` |
| Parse `earliest_cr_line` | object `"Sep-2004"` | datetime | Same format |
| Parse `term` | object `" 36 months"` | int: 36 or 60 | Strip whitespace, extract int with regex |
| Parse `emp_length` | object `"10+ years"` | int 0–10 | See mapping table below; null stays null until imputation step |
| Parse `revol_util` | float `"56.4"` | float | Already float; clip to [0, 100] |
| Parse `int_rate` | float | float | **EXCLUDED** — not used, but type-parse if needed for audit |
| Drop footer rows | `id` contains non-numeric string | drop 33 rows | Filter: `df[pd.to_numeric(df['id'], errors='coerce').notna()]` |

**`emp_length` ordinal mapping:**
```
"< 1 year" → 0
"1 year"   → 1
"2 years"  → 2
...
"9 years"  → 9
"10+ years"→ 10
null       → null  (handled in imputation step below)
```

---

## 2. Tier 1 — Raw Features Passed to WoE (Cleaned Originals)

These columns require only cleaning/clipping — no new column is created. They go directly into the WoE pipeline.

| Feature | Clean operation | Missing strategy | Leakage Risk |
|---|---|---|---|
| `loan_amnt` | None (already float) | 0% missing — none | None |
| `term_months` | Parsed to int (36/60) above | 0% missing — none | None |
| `purpose` | Strip whitespace | 0% missing — none | None |
| `initial_list_status` | Map `w`→0, `f`→1 | 0% missing — none | None |
| `application_type` | Map `Individual`→0, `Joint App`→1 | 0% missing — none | None |
| `disbursement_method` | Map `Cash`→0, `DirectPay`→1 | 0% missing — none | None |
| `home_ownership` | Group `OTHER`,`NONE`,`ANY` → `OTHER` | 0% missing — none | None |
| `annual_inc` | Clip at 99.9th pct (train-fitted) | 0% missing | None |
| `verification_status` | Keep as-is | 0% missing — none | None |
| `addr_state` | Keep as-is (51 cats) | 0% missing — none | None |
| `dti` | Clip at 99th pct (train-fitted) | 0.08% → median impute | None |
| `delinq_2yrs` | Cap at 10 | 0% missing — none | None |
| `inq_last_6mths` | Cap at 10 | 0% missing — none | None |
| `open_acc` | Cap at 40 | 0% missing — none | None |
| `pub_rec` | Cap at 5 | 0% missing — none | None |
| `revol_bal` | Clip at 99.9th pct | 0% missing — none | None |
| `revol_util` | Clip to [0, 100] | 0.08% → median impute | None |
| `total_acc` | Cap at 80 | 0% missing — none | None |
| `acc_now_delinq` | Keep as-is | 0% missing — none | None |
| `delinq_amnt` | Clip at 99th pct | 0% missing — none | None |
| `collections_12_mths_ex_med` | Keep as-is | 0.01% → 0 impute | None |
| `chargeoff_within_12_mths` | Keep as-is | 0.01% → 0 impute | None |
| `pub_rec_bankruptcies` | Keep as-is | 0.06% → 0 impute | None |
| `tax_liens` | Keep as-is | 0.01% → 0 impute | None |
| `acc_now_delinq` | Keep as-is | 0% — none | None |
| `mort_acc` | Keep as-is | 2.21% → median impute | None |
| `avg_cur_bal` | Clip at 99th pct | 3.11% → median impute | None |
| `bc_util` | Clip to [0, 100] | 3.37% → median impute | None |
| `pct_tl_nvr_dlq` | Clip to [0, 100] | 3.12% → median impute | None |
| `percent_bc_gt_75` | Clip to [0, 100] | 3.34% → median impute | None |
| `num_bc_sats` | Cap at 30 | 2.59% → median impute | None |
| `num_sats` | Cap at 60 | 2.59% → median impute | None |
| `num_accts_ever_120_pd` | Cap at 10 | 3.11% → median impute | None |
| `num_tl_90g_dpd_24m` | Cap at 10 | 3.11% → median impute | None |
| `num_tl_30dpd` | Cap at 5 | 3.11% → 0 impute | None |
| `num_tl_120dpd_2m` | Cap at 5 | 6.80% → 0 impute | None |
| `tot_cur_bal` | Clip at 99.9th pct | 3.11% → median impute | None |
| `tot_hi_cred_lim` | Clip at 99.9th pct | 3.11% → median impute | None |

> **Cap/clip values are computed from the TRAIN set only and stored in the
> preprocessor artifact. The same values are applied to DEV and OOT.**

---

## 3. Tier 2 — Derived Features

### 3.1 `fico_mid`

| Property | Value |
|---|---|
| **Name** | `fico_mid` |
| **Formula** | `(fico_range_low + fico_range_high) / 2` |
| **Motivation** | LC reports FICO as a band (e.g., 675–679). The midpoint is a single clean scalar. Using both `fico_range_low` and `fico_range_high` introduces near-perfect collinearity (they are always 4 apart). The midpoint retains full information with no redundancy. |
| **Input columns** | `fico_range_low`, `fico_range_high` |
| **Leakage risk** | None — origination FICO |
| **dtype** | float32 |
| **Missing** | 0% — both inputs are 0% missing |
| **Notes** | Drop `fico_range_low` and `fico_range_high` from the model after creating `fico_mid` to avoid collinearity (keep them in the data frame for audit, exclude from feature list). |

---

### 3.2 `credit_age_months`

| Property | Value |
|---|---|
| **Name** | `credit_age_months` |
| **Formula** | `(issue_d.dt.to_period('M') - earliest_cr_line.dt.to_period('M')).apply(lambda x: x.n)` — months between earliest credit line and loan issue date |
| **Motivation** | Length of credit history is one of the five core FICO components. Longer history = more established borrower and lower default risk. The raw `earliest_cr_line` date has 754 cardinality and encodes absolute year (economic era). Converting to age removes the era confound and produces a credit-meaningful numeric. |
| **Input columns** | `earliest_cr_line` (parsed to datetime), `issue_d` (parsed to datetime) |
| **Leakage risk** | None — both columns are known at origination |
| **dtype** | int16 |
| **Missing** | `earliest_cr_line` is 0% missing; `issue_d` is 0% missing → 0% missing in output |
| **Notes** | Cap at 600 months (50 years) to handle data entry errors. Values < 0 (impossible — credit line after issue) indicate bad data; set to NaN and flag. |

---

### 3.3 `loan_to_income`

| Property | Value |
|---|---|
| **Name** | `loan_to_income` |
| **Formula** | `loan_amnt / annual_inc` |
| **Motivation** | Affordability ratio — how large is the loan relative to the borrower's income? Higher ratio = greater repayment burden = higher default risk. Standard credit bureau metric analogous to payment-to-income. |
| **Input columns** | `loan_amnt`, `annual_inc` |
| **Leakage risk** | None |
| **dtype** | float32 |
| **Missing** | 0% — both inputs are 0% missing. However, `annual_inc` could be 0 (data quality issue) → set to NaN if `annual_inc` <= 0, create `loan_to_income_invalid` flag. |
| **Notes** | Clip at 99.9th percentile (train-fitted). Expect values in range [0.01, 5.0] for most loans. |

---

### 3.4 `revol_bal_to_income`

| Property | Value |
|---|---|
| **Name** | `revol_bal_to_income` |
| **Formula** | `revol_bal / annual_inc` |
| **Motivation** | Revolving debt burden relative to income. Captures how much of a borrower's existing revolving balance relative to their ability to repay. Distinct from `revol_util` (which is balance/limit, not balance/income). |
| **Input columns** | `revol_bal`, `annual_inc` |
| **Leakage risk** | None |
| **dtype** | float32 |
| **Missing** | 0% (handle zero `annual_inc` same as above) |
| **Notes** | Clip at 99.9th pct (train-fitted). |

---

### 3.5 `log_annual_inc`

| Property | Value |
|---|---|
| **Name** | `log_annual_inc` |
| **Formula** | `np.log1p(annual_inc)` |
| **Motivation** | `annual_inc` is heavily right-skewed (range: near-0 to millions). Log-transform produces a near-normal distribution required for Logistic Regression to work well. Tree models don't require this, but it speeds convergence and helps WoE binning produce interpretable monotonic bins. |
| **Input columns** | `annual_inc` |
| **Leakage risk** | None |
| **dtype** | float32 |
| **Missing** | 0% |
| **Notes** | log1p handles zero values safely (log(0+1)=0). Do NOT use `np.log` — fails on zero income rows. |

---

### 3.6 `log_loan_amnt`

| Property | Value |
|---|---|
| **Name** | `log_loan_amnt` |
| **Formula** | `np.log(loan_amnt)` |
| **Motivation** | Right-skewed distribution (range: \$500–\$40,000). Log-transform improves Logistic Regression calibration and WoE bin quality. |
| **Input columns** | `loan_amnt` |
| **Leakage risk** | None |
| **dtype** | float32 |
| **Missing** | 0% — safe to use `np.log` (no zeros possible for loan amount) |

---

### 3.7 `log_revol_bal`

| Property | Value |
|---|---|
| **Name** | `log_revol_bal` |
| **Formula** | `np.log1p(revol_bal)` |
| **Motivation** | `revol_bal` ranges from 0 to millions, extremely right-skewed. |
| **Input columns** | `revol_bal` |
| **Leakage risk** | None |
| **dtype** | float32 |
| **Missing** | 0% |

---

### 3.8 `emp_length_years`

| Property | Value |
|---|---|
| **Name** | `emp_length_years` |
| **Formula** | Ordinal mapping of `emp_length` string to integer 0–10 (see parsing table in §1) |
| **Motivation** | `emp_length` is a string category; models need a numeric. The categories have a natural ordinal relationship (more employment = more stable). |
| **Input columns** | `emp_length` |
| **Leakage risk** | None |
| **dtype** | float32 (to allow NaN) |
| **Missing** | 6.50% → impute with **median from TRAIN** (typically 4 or 5 years); also create `emp_length_missing` indicator (Tier 3) |

---

### 3.9 `has_delinq_2y`

| Property | Value |
|---|---|
| **Name** | `has_delinq_2y` |
| **Formula** | `(delinq_2yrs > 0).astype(int)` |
| **Motivation** | `delinq_2yrs` is a count that is 0 for the vast majority of borrowers. The binary indicator captures the key signal (any recent delinquency vs. none) more cleanly for WoE binning than the raw count, which would have sparse bins at higher values. |
| **Input columns** | `delinq_2yrs` |
| **Leakage risk** | None |
| **dtype** | int8 |
| **Missing** | 0% |

---

### 3.10 `has_pub_rec`

| Property | Value |
|---|---|
| **Name** | `has_pub_rec` |
| **Formula** | `(pub_rec > 0).astype(int)` |
| **Motivation** | Same rationale as `has_delinq_2y`. Public records (bankruptcies, judgements, tax liens) are rare binary events; the count adds little over the binary flag. |
| **Input columns** | `pub_rec` |
| **Leakage risk** | None |
| **dtype** | int8 |
| **Missing** | 0% |

---

### 3.11 `has_bankruptcy`

| Property | Value |
|---|---|
| **Name** | `has_bankruptcy` |
| **Formula** | `(pub_rec_bankruptcies > 0).astype(int)` |
| **Motivation** | Bankruptcy is a distinct, highly specific derogatory event with known predictive power in credit models. Separating it from the generic `pub_rec` flag provides an independent signal. |
| **Input columns** | `pub_rec_bankruptcies` |
| **Leakage risk** | None |
| **dtype** | int8 |
| **Missing** | 0.06% → impute 0 before creating flag (0 is the correct default — no bankruptcy) |

---

### 3.12 `mths_since_last_delinq_imputed`

| Property | Value |
|---|---|
| **Name** | `mths_since_last_delinq_imputed` |
| **Formula** | `mths_since_last_delinq.fillna(999)` then `clip(upper=999)` |
| **Motivation** | 51.25% of rows have no recent delinquency (null). Imputing with 999 encodes "never or very long ago" as a large value, preserving the ordinal meaning (larger = better credit behaviour). This is standard credit scorecard practice. The raw missing indicator (`mths_since_last_delinq_missing`) captures whether the imputation was applied. |
| **Input columns** | `mths_since_last_delinq` |
| **Leakage risk** | None |
| **dtype** | float32 |
| **Missing** | 0% after imputation |

---

### 3.13 `mths_since_last_record_imputed`

| Property | Value |
|---|---|
| **Name** | `mths_since_last_record_imputed` |
| **Formula** | `mths_since_last_record.fillna(999).clip(upper=999)` |
| **Motivation** | 84.11% null = no public record. Same rationale as 3.12. |
| **Input columns** | `mths_since_last_record` |
| **Leakage risk** | None |
| **dtype** | float32 |
| **Missing** | 0% after imputation |

---

### 3.14 `mths_since_last_major_derog_imputed`

| Property | Value |
|---|---|
| **Name** | `mths_since_last_major_derog_imputed` |
| **Formula** | `mths_since_last_major_derog.fillna(999).clip(upper=999)` |
| **Motivation** | 74.31% null = no major derogatory. Same rationale. |
| **Input columns** | `mths_since_last_major_derog` |
| **Leakage risk** | None |
| **dtype** | float32 |
| **Missing** | 0% after imputation |

---

### 3.15 `mths_since_recent_bc_dlq_imputed`

| Property | Value |
|---|---|
| **Name** | `mths_since_recent_bc_dlq_imputed` |
| **Formula** | `mths_since_recent_bc_dlq.fillna(999).clip(upper=999)` |
| **Motivation** | 77.01% null = no bankcard delinquency. |
| **Input columns** | `mths_since_recent_bc_dlq` |
| **Leakage risk** | None |
| **dtype** | float32 |

---

### 3.16 `mths_since_recent_revol_delinq_imputed`

| Property | Value |
|---|---|
| **Name** | `mths_since_recent_revol_delinq_imputed` |
| **Formula** | `mths_since_recent_revol_delinq.fillna(999).clip(upper=999)` |
| **Motivation** | 67.25% null = no revolving delinquency. |
| **Input columns** | `mths_since_recent_revol_delinq` |
| **Leakage risk** | None |
| **dtype** | float32 |

---

### 3.17 `bureau_extended_missing`

| Property | Value |
|---|---|
| **Name** | `bureau_extended_missing` |
| **Formula** | `open_acc_6m.isna().astype(int)` |
| **Motivation** | The 38.31% missing block (open_acc_6m, open_act_il, open_il_12m, etc.) is not random missingness — it corresponds to pre-2014 vintages before LC added extended bureau fields. This indicator captures vintage/era information that is legitimately known at origination. It also serves as a surrogate for loan age. |
| **Input columns** | `open_acc_6m` (proxy for the entire 38.31% block) |
| **Leakage risk** | None — missingness pattern is purely a function of issue date, which is origination-time information |
| **dtype** | int8 |
| **Missing** | 0% |
| **Notes** | After creating this indicator, impute the 38.31% block with **0** (most of these counts/amounts are 0 for older vintages that pre-date these bureau fields). Do NOT use median imputation — median of the non-missing subgroup would distort the interpretation. |

---

### 3.18 `is_joint_app`

| Property | Value |
|---|---|
| **Name** | `is_joint_app` |
| **Formula** | `(application_type == 'Joint App').astype(int)` |
| **Motivation** | Clean binary flag; used to condition joint-specific feature construction (3.19, 3.20). |
| **Input columns** | `application_type` |
| **Leakage risk** | None |
| **dtype** | int8 |
| **Missing** | 0% |

---

### 3.19 `effective_annual_inc`

| Property | Value |
|---|---|
| **Name** | `effective_annual_inc` |
| **Formula** | `annual_inc_joint.where(is_joint_app == 1, other=annual_inc)` |
| **Motivation** | Joint applications report combined household income in `annual_inc_joint`. Individual applications report in `annual_inc`. A single income field is needed for downstream ratios (`loan_to_income`, `revol_bal_to_income`) to apply consistently to all loans. |
| **Input columns** | `annual_inc`, `annual_inc_joint`, `is_joint_app` |
| **Leakage risk** | None |
| **dtype** | float64 |
| **Missing** | Use `annual_inc` for all Individual loans (0% missing). `annual_inc_joint` has 94.66% null but these nulls are always Individual loans — after the conditional replacement, result is 0% missing. |

---

### 3.20 `effective_dti`

| Property | Value |
|---|---|
| **Name** | `effective_dti` |
| **Formula** | `dti_joint.where(is_joint_app == 1, other=dti)` |
| **Motivation** | Same rationale as `effective_annual_inc`. `dti_joint` is the combined household DTI for joint applications; `dti` for individual. |
| **Input columns** | `dti`, `dti_joint`, `is_joint_app` |
| **Leakage risk** | None |
| **dtype** | float32 |
| **Missing** | Same logic — 0% after conditional replacement. Clip at 99th pct (train-fitted). |

---

### 3.21 `issue_month`

| Property | Value |
|---|---|
| **Name** | `issue_month` |
| **Formula** | `issue_d.dt.month` (1–12) |
| **Motivation** | Seasonal borrowing patterns exist in consumer lending (holiday debt in Q4, tax-season payoffs in Q1-Q2). Month of issue can proxy these patterns independently of year. |
| **Input columns** | `issue_d` |
| **Leakage risk** | None. **Do NOT use `issue_d.dt.year`** — year encodes the train/OOT split boundary and would allow the model to learn "year 2015 = train distribution, year 2017 = OOT distribution" rather than genuine credit signals. |
| **dtype** | int8 |
| **Missing** | 0% |

---

### 3.22 `zip_3digit`

| Property | Value |
|---|---|
| **Name** | `zip_3digit` |
| **Formula** | `zip_code.str[:3]` (extract first 3 characters) |
| **Motivation** | `zip_code` is already masked as "123xx" format. Extracting the 3-digit prefix gives the geographic grouping LC uses. WoE encoding on 956 categories is feasible with rare-group collapsing. |
| **Input columns** | `zip_code` |
| **Leakage risk** | None |
| **dtype** | object (categorical) |
| **Missing** | 0% |
| **Notes** | Rare 3-digit zips (< 0.1% of TRAIN = ~700 rows) are collapsed to `"OTHER"` before WoE fitting. |

---

## 4. Tier 3 — Missing-Value Indicator Features

These binary flags capture informative missingness — cases where the absence of a value is itself predictive (e.g., no prior delinquency = good credit history).

| Feature name | Formula | When to create | Motivation |
|---|---|---|---|
| `emp_length_missing` | `emp_length.isna().astype(int)` | Before imputing `emp_length_years` | 6.5% missing; may indicate self-employed or gig workers who didn't disclose |
| `mths_since_last_delinq_missing` | `mths_since_last_delinq.isna().astype(int)` | Before imputing to 999 | 51.25% missing = no prior delinquency; strong positive signal |
| `mths_since_last_record_missing` | `mths_since_last_record.isna().astype(int)` | Before imputing to 999 | 84.11% missing = no public record; positive signal |
| `mths_since_last_major_derog_missing` | `mths_since_last_major_derog.isna().astype(int)` | Before imputing to 999 | 74.31% missing = no major derogatory; positive signal |
| `mths_since_recent_bc_dlq_missing` | `mths_since_recent_bc_dlq.isna().astype(int)` | Before imputing to 999 | 77.01% missing = no bankcard delinquency |
| `mths_since_recent_revol_delinq_missing` | `mths_since_recent_revol_delinq.isna().astype(int)` | Before imputing to 999 | 67.25% missing = no revolving delinquency |
| `mths_since_recent_inq_missing` | `mths_since_recent_inq.isna().astype(int)` | Before median impute | 13.07% missing; may indicate borrowers with no recent soft pulls |
| `il_util_missing` | `il_util.isna().astype(int)` | Before median impute | 47.28% missing; partially overlaps with `bureau_extended_missing` |

> **Note**: `bureau_extended_missing` (feature 3.17) already covers the full 38.31% block.
> Individual indicators for `open_acc_6m`, `open_act_il`, etc. are **not** needed
> because they are all co-missing (same rows). One indicator is sufficient.

---

## 5. Features Not Created — Exclusion Rationale

| Feature idea | Why excluded |
|---|---|
| `total_derog_combined` (sum of delinq + pub_rec + collections) | Combining features before WoE prevents the model from assigning different weights to each component. WoE will already capture each individually. |
| `fico_mid * revol_util` (interaction) | Interaction features inflate dimensionality without guaranteed benefit. Tree models (XGBoost, LightGBM) learn interactions implicitly. Add only if LR specifically needs it. |
| `log_revol_bal_to_income` | Already have `revol_bal_to_income`; log transformation applied to the ratio would double-transform. The ratio is already clipped. |
| `issue_year` | Encodes temporal split — leaks train/OOT boundary into feature space. |
| NLP features from `emp_title`, `desc`, `title` | Out of scope per data dictionary — too high-cardinality, require text preprocessing pipeline |
| sec_app features (when `is_joint_app == 0`) | Already handled: null for individual loans; include raw + `is_joint_app` as context |
| Policy-era dummies from `policy_code` | Constant (always 1.0) — no signal |

---

## 6. Final Feature List (Input to WoE Pipeline)

After all engineering, the following features enter the WoE binning step:

**Numeric (WoE-binned):** ~50 features including all Tier 1 clipped numerics + Tier 2 derived numerics

**Categorical (WoE group-encoded):** `purpose`, `home_ownership`, `verification_status`, `addr_state`, `zip_3digit`, `initial_list_status`, `disbursement_method`

**Binary (WoE-binned or passed directly):** `term_months` (36/60), `application_type` / `is_joint_app`, `has_delinq_2y`, `has_pub_rec`, `has_bankruptcy`, `bureau_extended_missing`, `issue_month`, all Tier 3 missing indicators

> Exact feature count after IV selection will be determined by `fit_woe()` output.
> Expected: 30–60 features survive IV >= 0.02 threshold.

---

## 7. Preprocessing Order (Must Be Strictly Followed)

```
1. Load raw CSV → drop 33 footer rows → parse dtypes/dates
2. Encode target (loan_status → 0/1, exclude rows)
3. Drop all EXCLUDED columns (per leakage_audit.md)
4. Chronological split → TRAIN / DEV / OOT
   ───────── TRAIN ONLY below this line ─────────
5. Fit clip/cap bounds on TRAIN (99th/99.9th pct values)
6. Fit median imputers on TRAIN for columns with random missingness
7. Fit emp_length ordinal mapping (confirm mapping, validate)
   ───────── Apply to TRAIN, DEV, OOT ──────────
8. Create Tier 3 missing indicators (no fitting needed — pure boolean)
9. Apply cap/clip using TRAIN-fitted bounds
10. Apply median imputation using TRAIN-fitted values
11. Create Tier 2 derived features (using fitted bounds where needed)
    ───────── TRAIN ONLY below this line ─────────
12. Fit WoE bins on TRAIN (see woe_iv.md)
    ───────── Apply to TRAIN, DEV, OOT ──────────
13. Transform TRAIN, DEV, OOT using fitted WoE bins
14. Apply IV filter (features with IV < 0.02 in TRAIN → dropped from all sets)
```

Steps 5–7 are **fitted on TRAIN** and applied via stored artifacts. Steps 8–11 are deterministic transformations requiring no fitting. Steps 12–14 are the WoE pipeline (documented separately in `woe_iv.md`).

---

*See `woe_iv.md` for the WoE/IV binning specification and implementation contract.*  
*See `data_dictionary.md` for source column documentation.*
