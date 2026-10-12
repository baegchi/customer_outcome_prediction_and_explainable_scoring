# Data Dictionary — Customer Outcome Prediction

**Dataset**: Lending Club Accepted Loans 2007–2018 Q4  
**Shape**: 2,260,701 rows × 151 columns  
**Dtypes**: 113 float64, 38 object  
**Discovery date**: 2026-09-04

All missing percentages, cardinalities, and statistics are computed from the actual data.

**Leakage risk key:**
- 🔴 **HIGH** — Post-outcome or post-origination field; must be excluded
- 🟡 **MEDIUM** — Requires careful reasoning; decision documented in `leakage_audit.md`
- 🟢 **NONE** — Known at origination; safe to use as feature

**Treatment key:**
- **EXCLUDE** — Do not include in feature matrix (leakage or zero-variance)
- **DROP** — Remove (identifier, free-text, constant)
- **USE** — Include in feature matrix after cleaning
- **SPLIT-ONLY** — Use for temporal split only, not as feature
- **TARGET** — The outcome variable

---

## Section 1 — Identifiers and Metadata

| Feature | Type | Meaning | Missing % | Cardinality | Leakage Risk | Treatment |
|---|---|---|---:|---:|---|---|
| `id` | object | Unique loan identifier. Top values include embedded metadata string "Total amount funded in policy code 2" — confirms 33 footer rows in CSV. All 2,260,701 values are unique. | 0.00% | 2,260,701 | 🟢 NONE (identifier only) | **DROP** — No predictive value. Use only to detect/remove footer rows during loading. |
| `member_id` | float64 | Borrower-level member identifier. **100% null** — LC stopped populating this field before this dataset version. Cannot be used for entity-level deduplication. | 100.00% | 0 | 🟢 NONE | **DROP** — Entirely missing. |
| `url` | object | URL to LC loan detail page. Near-unique (2,260,668 distinct). Format: `lendingclub.com/browse/loanDetail.action?loan_id={id}`. | 0.00% | 2,260,668 | 🟢 NONE (identifier only) | **DROP** — No predictive value. |
| `policy_code` | float64 | LC policy code. **Constant = 1.0** across all 2,260,668 non-null rows. The embedded footer row mentions "policy code 2" but no actual policy_code=2 rows exist in data. | 0.00% | 1 | 🟢 NONE | **DROP** — Zero variance. |

---

## Section 2 — Target Variable

| Feature | Type | Meaning | Missing % | Cardinality | Leakage Risk | Treatment |
|---|---|---|---:|---:|---|---|
| `loan_status` | object | Current status of loan at snapshot date (Dec 2018). Nine distinct values. Encoding per `project_specification.md`: Charged Off + DNTMCP:Charged Off + Default → 1; Fully Paid + DNTMCP:Fully Paid → 0; Current/Late/Grace → exclude. 33 nulls = footer rows. | 0.00% | 9 | — | **TARGET** — Encode to binary, then exclude from feature matrix. |

---

## Section 3 — Loan Application Terms (Known at Origination — Safe)

| Feature | Type | Meaning | Missing % | Cardinality | Leakage Risk | Treatment |
|---|---|---|---:|---:|---|---|
| `loan_amnt` | float64 | The listed loan amount requested by the borrower. May differ from `funded_amnt` if partially funded. This is the application figure. | 0.00% | continuous | 🟢 NONE | **USE** — Log-transform recommended (right-skewed). |
| `term` | object | Loan repayment term. Two values: `' 36 months'` and `' 60 months'` (note leading space). Strip whitespace. | 0.00% | 2 | 🟢 NONE | **USE** — Binary encode (36=0, 60=1). |
| `purpose` | object | Stated purpose of the loan. 14 categories including debt_consolidation (dominant), credit_card, home_improvement, etc. | 0.00% | 14 | 🟢 NONE | **USE** — One-hot or WoE encode. |
| `initial_list_status` | object | Initial listing status: `w` (whole loans, marketed to institutional investors) or `f` (fractional, marketed to retail). | 0.00% | 2 | 🟢 NONE | **USE** — Binary encode. |
| `application_type` | object | Whether application is Individual (95.2%) or Joint App (4.8%). Governs which joint fields are populated. | 0.00% | 2 | 🟢 NONE | **USE** — Binary encode. |
| `disbursement_method` | object | Cash or DirectPay. DirectPay is used when paying a creditor directly. | 0.00% | 2 | 🟢 NONE | **USE** — Binary encode. |
| `issue_d` | object | Month the loan was issued. Parsed as datetime (format: `%b-%Y`). 139 unique months from Jun-2007 to Dec-2018. | 0.00% | 139 | 🟢 NONE | **SPLIT-ONLY** — Use for chronological split. May also derive month-of-year seasonal feature, but avoid using raw year (would encode label info in time-correlated splits). |

---

## Section 4 — LC Underwriting Outputs (Conditional Exclude — see leakage_audit.md §5)

> **Decision: EXCLUDE all four.** These are LC's own credit risk assessment outputs,
> not raw applicant inputs. Including them means learning LC's model, not building ours.

| Feature | Type | Meaning | Missing % | Cardinality | Leakage Risk | Treatment |
|---|---|---|---:|---:|---|---|
| `grade` | object | LC-assigned letter credit grade (A–G). 7 unique values. Highly predictive of default — because it IS LC's default prediction. | 0.00% | 7 | 🟡 MEDIUM | **EXCLUDE** — LC's underwriting output. See `leakage_audit.md §5`. |
| `sub_grade` | object | LC-assigned sub-grade (A1–G5). 35 unique values. Finer version of `grade`. | 0.00% | 35 | 🟡 MEDIUM | **EXCLUDE** — Same as `grade`. |
| `int_rate` | float64 | Annual interest rate assigned by LC. Derived mechanically from `sub_grade`. | 0.00% | continuous | 🟡 MEDIUM | **EXCLUDE** — Derived from `sub_grade`; leaks LC's credit assessment. |
| `installment` | float64 | Monthly payment amount. Derived from `loan_amnt`, `int_rate`, and `term`. | 0.00% | continuous | 🟡 MEDIUM | **EXCLUDE** — Derived from `int_rate`; redundant leakage. |

---

## Section 5 — Post-Approval Funding (Exclude — Not Available at Application Time)

| Feature | Type | Meaning | Missing % | Cardinality | Leakage Risk | Treatment |
|---|---|---|---:|---:|---|---|
| `funded_amnt` | float64 | Actual amount funded (may differ from `loan_amnt` if partially funded). This is a post-approval figure. | 0.00% | continuous | 🔴 HIGH | **EXCLUDE** — Post-approval. Use `loan_amnt` instead. |
| `funded_amnt_inv` | float64 | Portion of the loan funded by investors. Post-approval allocation. | 0.00% | continuous | 🔴 HIGH | **EXCLUDE** — Post-approval. |

---

## Section 6 — Borrower Profile at Application (Known at Origination — Safe)

| Feature | Type | Meaning | Missing % | Cardinality | Leakage Risk | Treatment |
|---|---|---|---:|---:|---|---|
| `emp_length` | object | Employment length reported by borrower. 11 levels: `< 1 year`, `1 year`, ..., `9 years`, `10+ years`. | 6.50% | 11 | 🟢 NONE | **USE** — Ordinal encode (0–10). Impute missing as separate category. |
| `emp_title` | object | Employer name/title — free text. **512,694 unique values** — extremely high cardinality, prone to misspellings, synonyms. | 7.39% | 512,694 | 🟢 NONE | **DROP** — Unworkable as-is without NLP standardization which is out of scope. |
| `home_ownership` | object | Home ownership status: MORTGAGE, RENT, OWN, OTHER, NONE, ANY. 6 categories. | 0.00% | 6 | 🟢 NONE | **USE** — One-hot or WoE encode. Group rare categories (NONE, ANY, OTHER). |
| `annual_inc` | float64 | Self-reported annual income. Right-skewed; some extreme outliers. | 0.00% | continuous | 🟢 NONE | **USE** — Log-transform. Cap extreme outliers at 99.9th percentile. |
| `verification_status` | object | Income verification status: `Not Verified`, `Source Verified`, `Verified`. 3 categories. | 0.00% | 3 | 🟢 NONE | **USE** — Ordinal encode or WoE. |
| `zip_code` | object | Masked zip code (first 3 digits + "xx"). 956 unique. Can proxy geographic economic conditions. | 0.00% | 956 | 🟢 NONE | **USE** — WoE encode, grouping rare zips. Or map to state-level aggregates. |
| `addr_state` | object | US state. 51 unique (includes DC). | 0.00% | 51 | 🟢 NONE | **USE** — WoE encode by state. |
| `desc` | object | Borrower-written loan description. **94.42% missing**, 124,500 unique. | 94.42% | 124,500 | 🟢 NONE | **DROP** — Mostly missing; free text; out of scope. |
| `title` | object | Borrower-provided loan title. 63,154 unique values. Heavily overlaps with `purpose`. 1.03% missing. | 1.03% | 63,154 | 🟢 NONE | **DROP** — High cardinality; dominated by `purpose`. |

---

## Section 7 — Credit Bureau at Origination (Known at Origination — Safe)

These fields come from the hard credit bureau pull at application time.

| Feature | Type | Meaning | Missing % | Cardinality | Leakage Risk | Treatment |
|---|---|---|---:|---:|---|---|
| `dti` | float64 | Debt-to-income ratio (excluding mortgage). Strong credit signal. 0.08% missing. | 0.08% | continuous | 🟢 NONE | **USE** — Use raw or log-transform; cap extremes. |
| `delinq_2yrs` | float64 | Number of delinquencies (30+ days) in past 2 years. | 0.00% | low | 🟢 NONE | **USE** — Treat as count; WoE bin. |
| `earliest_cr_line` | object | Date of borrower's oldest credit account. 754 unique values. Use to derive credit age at origination. | 0.00% | 754 | 🟢 NONE | **USE** — Derive `credit_age_months = months(issue_d - earliest_cr_line)`. |
| `fico_range_low` | float64 | Lower bound of FICO score range at application time. This is the origination FICO. | 0.00% | continuous | 🟢 NONE | **USE** — Key credit signal. Consider using midpoint `(fico_range_low + fico_range_high) / 2`. |
| `fico_range_high` | float64 | Upper bound of FICO score range at application time. | 0.00% | continuous | 🟢 NONE | **USE** — See `fico_range_low`. |
| `inq_last_6mths` | float64 | Number of credit inquiries in past 6 months (indicator of credit-seeking behaviour). | 0.00% | low | 🟢 NONE | **USE** — Count feature; WoE bin. |
| `mths_since_last_delinq` | float64 | Months since most recent delinquency. **51.25% missing** — missing means no recent delinquency, which is informative. | 51.25% | continuous | 🟢 NONE | **USE** — Create missing indicator. Impute with large value (e.g., 999) to encode "never delinquent". |
| `mths_since_last_record` | float64 | Months since last public record (bankruptcy, tax lien, etc.). **84.11% missing** — missing means no public record. | 84.11% | continuous | 🟢 NONE | **USE** — Create missing indicator. Impute with large value. |
| `open_acc` | float64 | Number of open credit accounts. | 0.00% | low | 🟢 NONE | **USE** |
| `pub_rec` | float64 | Number of derogatory public records (bankruptcies, liens, judgements). | 0.00% | low | 🟢 NONE | **USE** |
| `revol_bal` | float64 | Total revolving credit balance. Right-skewed. | 0.00% | continuous | 🟢 NONE | **USE** — Log-transform. |
| `revol_util` | float64 | Revolving line utilization rate (balance / limit). 0.08% missing. | 0.08% | continuous | 🟢 NONE | **USE** — Strong credit signal. |
| `total_acc` | float64 | Total number of credit lines ever opened. | 0.00% | low | 🟢 NONE | **USE** |
| `collections_12_mths_ex_med` | float64 | Collections in past 12 months (excluding medical). 0.01% missing. | 0.01% | low | 🟢 NONE | **USE** — Median impute. |
| `mths_since_last_major_derog` | float64 | Months since most recent 90+ day delinquency. **74.31% missing** = no major derog. | 74.31% | continuous | 🟢 NONE | **USE** — Create indicator; impute with large value. |
| `chargeoff_within_12_mths` | float64 | Number of charge-offs in past 12 months. 0.01% missing. | 0.01% | low | 🟢 NONE | **USE** — Median impute. |
| `acc_now_delinq` | float64 | Number of accounts currently delinquent. | 0.00% | low | 🟢 NONE | **USE** |
| `delinq_amnt` | float64 | Amount past due on accounts currently delinquent. | 0.00% | continuous | 🟢 NONE | **USE** |
| `pub_rec_bankruptcies` | float64 | Number of public record bankruptcies. 0.06% missing. | 0.06% | low | 🟢 NONE | **USE** — Median impute. |
| `tax_liens` | float64 | Number of tax liens. 0.01% missing. | 0.01% | low | 🟢 NONE | **USE** — Median impute. |
| `pymnt_plan` | object | Whether borrower is on a payment plan ('n'/'y'). 620 'y' of 2.26M = 0.027%. | 0.00% | 2 | 🟡 MEDIUM | **EXCLUDE** — 620 'y' rows likely reflect mid-loan arrangements; see `leakage_audit.md §7`. Negligible information given rarity. |

---

## Section 8 — Extended Credit Bureau Features (38–47% Missing — Introduced Mid-Dataset)

The following block of ~16 columns has uniformly **38.31% missing**. This is not random — these fields were added to LC's bureau pull at a specific point in time, so older loans (pre-approximately 2014) lack them. This missingness is informative: "missing" = older loan vintage. Create a single binary indicator `bureau_extended_missing`.

| Feature | Type | Meaning | Missing % | Leakage Risk | Treatment |
|---|---|---|---:|---|---|
| `open_acc_6m` | float64 | Open trade lines in past 6 months | 38.31% | 🟢 NONE | **USE** — Create group missing indicator |
| `open_act_il` | float64 | Currently active installment trades | 38.31% | 🟢 NONE | **USE** |
| `open_il_12m` | float64 | Installment accounts opened in past 12 months | 38.31% | 🟢 NONE | **USE** |
| `open_il_24m` | float64 | Installment accounts opened in past 24 months | 38.31% | 🟢 NONE | **USE** |
| `mths_since_rcnt_il` | float64 | Months since most recent installment account opened. 40.25% missing. | 40.25% | 🟢 NONE | **USE** — Additional missingness beyond the 38% block |
| `total_bal_il` | float64 | Total current balance of installment accounts | 38.31% | 🟢 NONE | **USE** |
| `il_util` | float64 | Ratio of installment balance to high credit limit. **47.28% missing** — highest in block. | 47.28% | 🟢 NONE | **USE** |
| `open_rv_12m` | float64 | Revolving trades opened in past 12 months | 38.31% | 🟢 NONE | **USE** |
| `open_rv_24m` | float64 | Revolving trades opened in past 24 months | 38.31% | 🟢 NONE | **USE** |
| `max_bal_bc` | float64 | Maximum current balance of a bankcard account | 38.31% | 🟢 NONE | **USE** |
| `all_util` | float64 | Balance to credit limit for all trades | 38.32% | 🟢 NONE | **USE** |
| `total_rev_hi_lim` | float64 | Total revolving high credit/credit limit | 3.11% | 🟢 NONE | **USE** |
| `inq_fi` | float64 | Number of personal finance inquiries | 38.31% | 🟢 NONE | **USE** |
| `total_cu_tl` | float64 | Number of finance trades | 38.31% | 🟢 NONE | **USE** |
| `inq_last_12m` | float64 | Number of credit inquiries in past 12 months | 38.31% | 🟢 NONE | **USE** |

---

## Section 9 — Expanded Credit Bureau (2–7% Missing — Introduced ~2013)

| Feature | Type | Meaning | Missing % | Leakage Risk | Treatment |
|---|---|---|---:|---|---|
| `acc_open_past_24mths` | float64 | Accounts opened in past 24 months | 2.21% | 🟢 NONE | **USE** — Median impute |
| `avg_cur_bal` | float64 | Average current balance of all accounts | 3.11% | 🟢 NONE | **USE** |
| `bc_open_to_buy` | float64 | Total open to buy on revolving bankcards | 3.32% | 🟢 NONE | **USE** |
| `bc_util` | float64 | Ratio of total current balance to high credit limit for all bankcard accounts | 3.37% | 🟢 NONE | **USE** |
| `mo_sin_old_il_acct` | float64 | Months since oldest installment account | 6.15% | 🟢 NONE | **USE** |
| `mo_sin_old_rev_tl_op` | float64 | Months since oldest revolving account opened | 3.11% | 🟢 NONE | **USE** |
| `mo_sin_rcnt_rev_tl_op` | float64 | Months since most recently opened revolving account | 3.11% | 🟢 NONE | **USE** |
| `mo_sin_rcnt_tl` | float64 | Months since most recent account opened | 3.11% | 🟢 NONE | **USE** |
| `mort_acc` | float64 | Number of mortgage accounts | 2.21% | 🟢 NONE | **USE** |
| `mths_since_recent_bc` | float64 | Months since most recently opened bankcard | 3.25% | 🟢 NONE | **USE** |
| `mths_since_recent_bc_dlq` | float64 | Months since most recent bankcard delinquency. **77.01% missing** = no bankcard delinquency. | 77.01% | 🟢 NONE | **USE** — Create indicator; impute with large value |
| `mths_since_recent_inq` | float64 | Months since most recent inquiry. 13.07% missing. | 13.07% | 🟢 NONE | **USE** |
| `mths_since_recent_revol_delinq` | float64 | Months since most recent revolving delinquency. **67.25% missing**. | 67.25% | 🟢 NONE | **USE** — Create indicator; impute with large value |
| `num_accts_ever_120_pd` | float64 | Number of accounts ever 120+ days past due | 3.11% | 🟢 NONE | **USE** |
| `num_actv_bc_tl` | float64 | Number of currently active bankcard accounts | 3.11% | 🟢 NONE | **USE** |
| `num_actv_rev_tl` | float64 | Number of currently active revolving trades | 3.11% | 🟢 NONE | **USE** |
| `num_bc_sats` | float64 | Number of satisfactory bankcard accounts | 2.59% | 🟢 NONE | **USE** |
| `num_bc_tl` | float64 | Number of bankcard accounts | 3.11% | 🟢 NONE | **USE** |
| `num_il_tl` | float64 | Number of installment accounts | 3.11% | 🟢 NONE | **USE** |
| `num_op_rev_tl` | float64 | Number of open revolving accounts | 3.11% | 🟢 NONE | **USE** |
| `num_rev_accts` | float64 | Number of revolving accounts | 3.11% | 🟢 NONE | **USE** |
| `num_rev_tl_bal_gt_0` | float64 | Number of revolving trades with balance > 0 | 3.11% | 🟢 NONE | **USE** |
| `num_sats` | float64 | Number of satisfactory accounts | 2.59% | 🟢 NONE | **USE** |
| `num_tl_120dpd_2m` | float64 | Number of accounts 120+ days past due in past 2 months. 6.80% missing. | 6.80% | 🟢 NONE | **USE** |
| `num_tl_30dpd` | float64 | Number of accounts 30+ days past due in past 2 months | 3.11% | 🟢 NONE | **USE** |
| `num_tl_90g_dpd_24m` | float64 | Number of accounts 90+ days past due in past 24 months | 3.11% | 🟢 NONE | **USE** |
| `num_tl_op_past_12m` | float64 | Number of accounts opened in past 12 months | 3.11% | 🟢 NONE | **USE** |
| `pct_tl_nvr_dlq` | float64 | Percent of trades never delinquent | 3.12% | 🟢 NONE | **USE** |
| `percent_bc_gt_75` | float64 | Percent of bankcard accounts > 75% of limit | 3.34% | 🟢 NONE | **USE** |
| `tot_coll_amt` | float64 | Total collection amounts ever owed | 3.11% | 🟢 NONE | **USE** |
| `tot_cur_bal` | float64 | Total current balance of all accounts | 3.11% | 🟢 NONE | **USE** |
| `tot_hi_cred_lim` | float64 | Total high credit/credit limit | 3.11% | 🟢 NONE | **USE** |
| `total_bal_ex_mort` | float64 | Total credit balance excluding mortgage | 2.21% | 🟢 NONE | **USE** |
| `total_bc_limit` | float64 | Total bankcard high credit/credit limit | 2.21% | 🟢 NONE | **USE** |
| `total_il_high_credit_limit` | float64 | Total installment high credit/credit limit | 3.11% | 🟢 NONE | **USE** |

---

## Section 10 — Joint Application Fields (~95% Missing)

These fields are populated only for Joint App loans (4.78% of dataset). For Individual loans they are null. Do not impute — nulls encode application type.

| Feature | Type | Meaning | Missing % | Leakage Risk | Treatment |
|---|---|---|---:|---|---|
| `annual_inc_joint` | float64 | Joint annual income | 94.66% | 🟢 NONE | **USE** — Only for joint applications. Create `joint_annual_inc = annual_inc_joint if joint else annual_inc`. |
| `dti_joint` | float64 | Joint debt-to-income | 94.66% | 🟢 NONE | **USE** — Similar to above. |
| `verification_status_joint` | object | Joint income verification status. 3 categories. | 94.88% | 🟢 NONE | **USE** — For joint apps only. |
| `revol_bal_joint` | float64 | Joint revolving balance | 95.22% | 🟢 NONE | **USE** — For joint apps only. |
| `sec_app_fico_range_low` | float64 | Secondary applicant FICO lower bound | 95.22% | 🟢 NONE | **USE** — For joint apps only. |
| `sec_app_fico_range_high` | float64 | Secondary applicant FICO upper bound | 95.22% | 🟢 NONE | **USE** |
| `sec_app_earliest_cr_line` | object | Secondary applicant earliest credit line date | 95.22% | 🟢 NONE | **USE** — Derive credit age. |
| `sec_app_inq_last_6mths` | float64 | Secondary applicant inquiries in past 6 months | 95.22% | 🟢 NONE | **USE** |
| `sec_app_mort_acc` | float64 | Secondary applicant mortgage accounts | 95.22% | 🟢 NONE | **USE** |
| `sec_app_open_acc` | float64 | Secondary applicant open accounts | 95.22% | 🟢 NONE | **USE** |
| `sec_app_revol_util` | float64 | Secondary applicant revolving utilization. 95.30% missing. | 95.30% | 🟢 NONE | **USE** |
| `sec_app_open_act_il` | float64 | Secondary applicant active installment trades | 95.22% | 🟢 NONE | **USE** |
| `sec_app_num_rev_accts` | float64 | Secondary applicant revolving account count | 95.22% | 🟢 NONE | **USE** |
| `sec_app_chargeoff_within_12_mths` | float64 | Secondary applicant charge-offs in past 12 months | 95.22% | 🟢 NONE | **USE** |
| `sec_app_collections_12_mths_ex_med` | float64 | Secondary applicant collections in past 12 months | 95.22% | 🟢 NONE | **USE** |
| `sec_app_mths_since_last_major_derog` | float64 | Secondary applicant months since major derogatory. **98.41% missing** = nearly never populated even for joint apps. | 98.41% | 🟢 NONE | **USE** — Consider dropping if IV near-zero. |

---

## Section 11 — Post-Outcome Payment Fields (EXCLUDE — Hard Leakage)

> All fields below are populated as the loan progresses. They are **impossible** to know
> at origination time. See `leakage_audit.md §1` for detailed reasoning.

| Feature | Type | Meaning | Missing % | Leakage Risk | Treatment |
|---|---|---|---:|---|---|
| `out_prncp` | float64 | Outstanding principal balance. Zero for Fully Paid; near-zero for Charged Off after recovery. Encodes outcome directly. | 0.00% | 🔴 HIGH | **EXCLUDE** |
| `out_prncp_inv` | float64 | Investor portion of outstanding principal. | 0.00% | 🔴 HIGH | **EXCLUDE** |
| `total_pymnt` | float64 | Total payment received to date. Defaults pay less. Perfect predictor. | 0.00% | 🔴 HIGH | **EXCLUDE** |
| `total_pymnt_inv` | float64 | Investor portion of total payment. | 0.00% | 🔴 HIGH | **EXCLUDE** |
| `total_rec_prncp` | float64 | Total principal received. | 0.00% | 🔴 HIGH | **EXCLUDE** |
| `total_rec_int` | float64 | Total interest received. Defaults pay less interest. | 0.00% | 🔴 HIGH | **EXCLUDE** |
| `total_rec_late_fee` | float64 | Late fees received. Non-zero only for delinquent/defaulted loans. | 0.00% | 🔴 HIGH | **EXCLUDE** |
| `recoveries` | float64 | Recovery amount post charge-off. Non-zero for 185,432 defaulted loans; zero for all others by construction. | 0.00% | 🔴 HIGH | **EXCLUDE** |
| `collection_recovery_fee` | float64 | Collection agency fee on recoveries. | 0.00% | 🔴 HIGH | **EXCLUDE** |
| `last_pymnt_d` | object | Date of most recent payment. 0.11% null. 136 unique dates. | 0.11% | 🔴 HIGH | **EXCLUDE** |
| `last_pymnt_amnt` | float64 | Amount of most recent payment. | 0.00% | 🔴 HIGH | **EXCLUDE** |
| `next_pymnt_d` | object | Next scheduled payment date. 59.51% null — nulls are fully paid / charged off loans. Null pattern directly encodes outcome. | 59.51% | 🔴 HIGH | **EXCLUDE** |

---

## Section 12 — Post-Origination Credit Refresh (EXCLUDE — Leakage)

| Feature | Type | Meaning | Missing % | Leakage Risk | Treatment |
|---|---|---|---:|---|---|
| `last_fico_range_high` | float64 | FICO upper bound at most recent credit pull (post-origination). For defaults, this captures in-flight credit deterioration. | 0.00% | 🔴 HIGH | **EXCLUDE** — Use `fico_range_high` (origination FICO) instead. |
| `last_fico_range_low` | float64 | FICO lower bound at most recent credit pull. | 0.00% | 🔴 HIGH | **EXCLUDE** |
| `last_credit_pull_d` | object | Date of most recent credit pull. 141 unique dates. | 0.00% | 🔴 HIGH | **EXCLUDE** |

---

## Section 13 — Hardship Program Fields (EXCLUDE — Post-Origination Event)

> All populated post-origination when borrower enters financial distress program.
> See `leakage_audit.md §3`.

| Feature | Type | Meaning | Missing % | Leakage Risk | Treatment |
|---|---|---|---:|---|---|
| `hardship_flag` | object | 'Y' = borrower in hardship program. 0% missing but 'Y' encodes in-distress status. | 0.00% | 🔴 HIGH | **EXCLUDE** |
| `hardship_type` | object | Type of hardship arrangement. Only 1 unique value: "INTEREST ONLY-3 MONTHS DEFERRAL". | 99.52% | 🔴 HIGH | **EXCLUDE** |
| `hardship_reason` | object | Reason for hardship (9 categories). | 99.52% | 🔴 HIGH | **EXCLUDE** |
| `hardship_status` | object | Status of hardship plan (3 categories: COMPLETED, BROKEN, ACTIVE). | 99.52% | 🔴 HIGH | **EXCLUDE** |
| `deferral_term` | float64 | Deferral term months. **Constant = 3.0** for all non-null rows. | 99.52% | 🔴 HIGH | **EXCLUDE** |
| `hardship_amount` | float64 | Monthly payment during hardship. | 99.52% | 🔴 HIGH | **EXCLUDE** |
| `hardship_start_date` | object | Start date of hardship plan (27 unique dates). | 99.52% | 🔴 HIGH | **EXCLUDE** |
| `hardship_end_date` | object | End date of hardship plan (28 unique dates). | 99.52% | 🔴 HIGH | **EXCLUDE** |
| `payment_plan_start_date` | object | Start date of payment plan (27 unique dates). | 99.52% | 🔴 HIGH | **EXCLUDE** |
| `hardship_length` | float64 | Length of hardship in months. **Constant = 3.0**. | 99.52% | 🔴 HIGH | **EXCLUDE** |
| `hardship_dpd` | float64 | Days past due at hardship enrollment. Directly encodes delinquency at that moment. | 99.52% | 🔴 HIGH | **EXCLUDE** |
| `hardship_loan_status` | object | Loan status at hardship enrollment (5 categories). | 99.52% | 🔴 HIGH | **EXCLUDE** |
| `orig_projected_additional_accrued_interest` | float64 | Projected additional interest during hardship. **99.62% missing**. | 99.62% | 🔴 HIGH | **EXCLUDE** |
| `hardship_payoff_balance_amount` | float64 | Payoff balance during hardship. | 99.52% | 🔴 HIGH | **EXCLUDE** |
| `hardship_last_payment_amount` | float64 | Last payment amount during hardship. | 99.52% | 🔴 HIGH | **EXCLUDE** |

---

## Section 14 — Debt Settlement Fields (EXCLUDE — Post-Default Event)

| Feature | Type | Meaning | Missing % | Leakage Risk | Treatment |
|---|---|---|---:|---|---|
| `debt_settlement_flag` | object | 'Y' = loan in debt settlement. Settlement occurs only in default context. | 0.00% | 🔴 HIGH | **EXCLUDE** |
| `debt_settlement_flag_date` | object | Date settlement flag was set (83 unique dates). | 98.49% | 🔴 HIGH | **EXCLUDE** |
| `settlement_status` | object | Status of settlement: ACTIVE, COMPLETE, BROKEN. | 98.49% | 🔴 HIGH | **EXCLUDE** |
| `settlement_date` | object | Date of settlement agreement (90 unique dates). | 98.49% | 🔴 HIGH | **EXCLUDE** |
| `settlement_amount` | float64 | Amount settled. | 98.49% | 🔴 HIGH | **EXCLUDE** |
| `settlement_percentage` | float64 | Percentage of balance settled. | 98.49% | 🔴 HIGH | **EXCLUDE** |
| `settlement_term` | float64 | Term of settlement arrangement. | 98.49% | 🔴 HIGH | **EXCLUDE** |

---

## Summary — Column Disposition

| Disposition | Count |
|---|---:|
| **USE** (modelling features, after engineering) | ~75 |
| **EXCLUDE** (leakage) | ~45 |
| **DROP** (identifier/constant/text) | ~10 |
| **TARGET** | 1 |
| **SPLIT-ONLY** | 1 (`issue_d`) |
| **Total** | **151** |

> Exact feature count after IV selection will be lower. WoE/IV analysis will
> identify features with IV < 0.02 for removal.

---

*See `leakage_audit.md` for the full reasoning on each exclusion decision.*  
*See `feature_engineering.md` (to be created in Phase 4) for derived feature specifications.*
