# Leakage Audit — Customer Outcome Prediction

**Dataset**: Lending Club Accepted Loans 2007–2018 Q4  
**Audit date**: 2026-09-04  
**Auditor**: Senior ML Engineer review  
**Prediction moment**: Loan origination (the instant LC accepts the application and funds the loan)

---

## 0. Framing — When Is the Prediction Made?

The model predicts default **at origination time**: the moment a loan application is accepted, underwritten, and issued. At that instant:

- The borrower's application data exists
- A credit bureau report has been pulled
- **No payment has been made yet**
- **No outcome event has occurred yet**
- LC has already assigned its internal grade/rate (which is LC's own scoring output — see §3)

Any column whose value is determined by or correlated with events that happen **after origination** is leakage.

---

## 1. Post-Outcome Payment / Recovery Fields — HARD EXCLUDE

These fields are populated as the loan progresses through its lifecycle. They are **definitionally impossible** to know at origination time.

| Column | Leakage Mechanism | Decision |
|---|---|---|
| `out_prncp` | Outstanding principal balance — is zero for Fully Paid, positive for Current, near-zero for Charged Off after recovery. Encodes outcome directly. | **EXCLUDE** |
| `out_prncp_inv` | Investor's portion of outstanding principal — same leakage as above. | **EXCLUDE** |
| `total_pymnt` | Total payment received to date — defaults pay less than non-defaults. Perfect predictor if included. | **EXCLUDE** |
| `total_pymnt_inv` | Investor's portion of total payment — same leakage. | **EXCLUDE** |
| `total_rec_prncp` | Total principal recovered — zero for defaults that never paid. Encodes outcome. | **EXCLUDE** |
| `total_rec_int` | Total interest received — defaults pay less interest. Encodes outcome. | **EXCLUDE** |
| `total_rec_late_fee` | Late fees received — only non-zero for delinquent/defaulted loans. Directly encodes the outcome. | **EXCLUDE** |
| `recoveries` | Amount recovered after charge-off — non-zero only for charged-off loans. **This column is zero for all non-defaults by construction.** 185,432 rows have recoveries > 0, which aligns directly with the default population. | **EXCLUDE** |
| `collection_recovery_fee` | Fee on the recovery — same leakage as `recoveries`. | **EXCLUDE** |
| `last_pymnt_d` | Date of the most recent payment — null or very old for defaulted loans that stopped paying. Encodes default timing. | **EXCLUDE** |
| `last_pymnt_amnt` | Amount of the last payment — final payment amount before default differs from routine payment. Encodes outcome. | **EXCLUDE** |
| `next_pymnt_d` | Next scheduled payment date — null for charged-off and fully-paid loans (59.51% null). The null pattern is perfectly correlated with resolved loans. | **EXCLUDE** |

**Why this matters**: A model trained with any of these columns will learn trivial rules like "loans where `recoveries` > 0 are defaults". On a train/test random split this appears as near-perfect AUC. On OOT it collapses because future loans haven't generated these values yet.

---

## 2. Post-Origination Credit Bureau Refresh — HARD EXCLUDE

LC periodically re-pulls credit data during the loan life. These represent the borrower's creditworthiness **during** the loan, not at origination.

| Column | Leakage Mechanism | Decision |
|---|---|---|
| `last_fico_range_high` | FICO score at the most recent credit pull, not at origination. For defaults, FICO typically declined during the loan — using it means the model learns from post-origination deterioration. | **EXCLUDE** |
| `last_fico_range_low` | Lower bound of last-pulled FICO — same leakage as above. | **EXCLUDE** |
| `last_credit_pull_d` | Date of the most recent credit pull — provides post-origination timing information. Even as a feature (time since pull), it encodes loan lifecycle stage. | **EXCLUDE** |

**Note**: The original FICO at application time is available as `fico_range_low` and `fico_range_high` — these are **safe** and should be used.

---

## 3. Hardship Program Fields — HARD EXCLUDE

LC introduced a hardship program (interest-only deferral) for borrowers in financial distress. These fields are populated **only when a borrower enters the hardship program**, which occurs mid-loan due to emerging financial difficulty — i.e., post-origination and correlated with default risk.

| Column | Missing % | Leakage Mechanism | Decision |
|---|---|---|---|
| `hardship_flag` | 0% | Flag 'Y' indicates borrower is in hardship — borrower has already demonstrated inability to pay at the time this is set. | **EXCLUDE** |
| `hardship_type` | 99.52% | Type of hardship arrangement — post-event. | **EXCLUDE** |
| `hardship_reason` | 99.52% | Reason for hardship (e.g., UNEMPLOYMENT, NATURAL_DISASTER) — post-event. | **EXCLUDE** |
| `hardship_status` | 99.52% | Status of hardship plan (COMPLETED, BROKEN, ACTIVE) — post-event. | **EXCLUDE** |
| `deferral_term` | 99.52% | Term of deferral in months — constant at 3.0 for all hardship cases; post-event. | **EXCLUDE** |
| `hardship_amount` | 99.52% | Monthly payment during hardship — post-event. | **EXCLUDE** |
| `hardship_start_date` | 99.52% | Start date of hardship plan — post-event. | **EXCLUDE** |
| `hardship_end_date` | 99.52% | End date of hardship plan — post-event. | **EXCLUDE** |
| `payment_plan_start_date` | 99.52% | Start date of payment plan — post-event. | **EXCLUDE** |
| `hardship_length` | 99.52% | Length of hardship in months — constant at 3.0; post-event. | **EXCLUDE** |
| `hardship_dpd` | 99.52% | Days past due at hardship enrollment — directly encodes delinquency; post-event. | **EXCLUDE** |
| `hardship_loan_status` | 99.52% | Loan status at hardship enrollment — post-event loan status. | **EXCLUDE** |
| `orig_projected_additional_accrued_interest` | 99.62% | Projected accrued interest — post-event calculation. | **EXCLUDE** |
| `hardship_payoff_balance_amount` | 99.52% | Payoff balance during hardship — post-event. | **EXCLUDE** |
| `hardship_last_payment_amount` | 99.52% | Last payment during hardship — post-event. | **EXCLUDE** |

---

## 4. Debt Settlement Fields — HARD EXCLUDE

Debt settlement is a resolution option for defaulted or near-defaulted loans — the borrower pays a lump sum less than the outstanding balance. These fields are populated only after the default event.

| Column | Missing % | Leakage Mechanism | Decision |
|---|---|---|---|
| `debt_settlement_flag` | 0% | 'Y' means loan entered debt settlement — settlement only occurs in default context. | **EXCLUDE** |
| `debt_settlement_flag_date` | 98.49% | Date of settlement flag — post-default. | **EXCLUDE** |
| `settlement_status` | 98.49% | Status of settlement (ACTIVE, COMPLETE, BROKEN) — post-default. | **EXCLUDE** |
| `settlement_date` | 98.49% | Date of settlement agreement — post-default. | **EXCLUDE** |
| `settlement_amount` | 98.49% | Amount settled — post-default. | **EXCLUDE** |
| `settlement_percentage` | 98.49% | Percentage of balance settled — post-default. | **EXCLUDE** |
| `settlement_term` | 98.49% | Term of settlement — post-default. | **EXCLUDE** |

---

## 5. LC Underwriting Outputs — CONDITIONAL EXCLUDE (design decision required)

These four columns are produced by LC's internal underwriting process **after reviewing the application** but **before** funding. They are LC's own credit risk assessment.

| Column | What it is | Leakage argument | Keep argument |
|---|---|---|---|
| `grade` | LC's letter credit grade (A–G) | LC's model output — using it means our model is learning LC's decision, not borrower creditworthiness | Could argue it's "available at origination" since it's assigned before the loan is funded |
| `sub_grade` | LC's finer grade (A1–G5, 35 levels) | More granular version of the same leakage | Same as above |
| `int_rate` | Interest rate assigned by LC | Derived 1-to-1 from sub_grade — mechanically encodes grade | Same leakage |
| `installment` | Monthly payment amount | Derived from loan_amnt + int_rate + term — leaks int_rate | Same leakage |

**Decision: EXCLUDE all four.**

**Reasoning**: The purpose of this project is to build an **independent credit risk score**, not to replicate LC's internal model. If we include LC's own grade, we are answering "does LC's score predict default?" — which is trivially yes and defeats the purpose. The interesting question is: "can we predict default from raw applicant and credit bureau data, independently of LC's assessment?"

Excluding these also tests whether our independently-derived features capture the same signal LC captures. This is the intellectually honest and interview-defensible position.

> **Assumption flagged for confirmation**: This assumes the goal is an independent score.
> If the goal is instead to replicate LC's internal scoring exactly, grade/sub_grade/int_rate
> should be included as features (they would be the strongest predictors by far, and all
> downstream metrics would be higher). Please confirm.

---

## 6. Post-Approval Funding Fields — CONDITIONAL EXCLUDE

| Column | Leakage Mechanism | Decision |
|---|---|---|
| `funded_amnt` | Actual amount funded after approval — can differ from `loan_amnt` if partially funded. This is a post-application decision. | **EXCLUDE** (use `loan_amnt` instead) |
| `funded_amnt_inv` | Investor-funded portion — post-approval allocation detail. | **EXCLUDE** |

Note: `loan_amnt` (the borrower-requested amount) is safe and should be used.

---

## 7. Payment Plan Flag — EDGE CASE

| Column | Analysis | Decision |
|---|---|---|
| `pymnt_plan` | 620 loans are on a payment plan ('y'). Payment plans are typically negotiated when a borrower is in distress, mid-loan. However, in LC's data model this field may be set at origination for certain loan types. | **EXCLUDE** — The 620 'y' cases almost certainly correspond to post-origination arrangements. The information gain is minimal (620 of 1.35M rows). |

---

## 8. Identifier and Metadata Fields — DROP (Not Leakage, Just Useless)

| Column | Reason |
|---|---|
| `id` | Unique loan identifier — 2,260,701 unique values. No predictive value; also contains embedded metadata rows ("Total amount funded in policy code 2"). Drop after deduplication check. |
| `member_id` | 100% null. Drop. |
| `url` | Near-unique LC loan detail URL (2,260,668 unique). No predictive value. Drop. |
| `policy_code` | Constant 1.0 across all 2,260,668 non-null rows. Zero variance — drop. |

---

## 9. High-Cardinality Free-Text Fields — DROP (Not Leakage, but Unusable as-is)

| Column | Cardinality | Missing % | Reason |
|---|---|---|---|
| `emp_title` | 512,694 unique | 7.39% | Free-text employer name — impossible to encode cleanly. Standardization NLP is out of scope. Drop. |
| `desc` | 124,500 unique | 94.42% | Free-text loan description written by borrower. 94% missing. Out of scope for this project. Drop. |
| `title` | 63,154 unique | 1.03% | Borrower-provided loan title. Overlaps substantially with `purpose` (14 categories). Drop in favour of `purpose`. |

---

## 10. Safe Features — Retain (Known at Origination)

These fields are available at origination time with no leakage risk:

**Loan terms (application):**
`loan_amnt`, `term`, `purpose`, `initial_list_status`, `application_type`, `disbursement_method`

**Borrower profile (application):**
`emp_length`, `home_ownership`, `annual_inc`, `verification_status`, `addr_state`, `zip_code`

**Credit bureau at origination:**
`dti`, `delinq_2yrs`, `earliest_cr_line`, `fico_range_low`, `fico_range_high`, `inq_last_6mths`,
`mths_since_last_delinq`, `mths_since_last_record`, `open_acc`, `pub_rec`, `revol_bal`,
`revol_util`, `total_acc`, `collections_12_mths_ex_med`, `mths_since_last_major_derog`,
`chargeoff_within_12_mths`, `acc_now_delinq`, `tot_coll_amt`, `tot_cur_bal`,
`open_acc_6m`, `open_act_il`, `open_il_12m`, `open_il_24m`, `mths_since_rcnt_il`,
`total_bal_il`, `il_util`, `open_rv_12m`, `open_rv_24m`, `max_bal_bc`, `all_util`,
`total_rev_hi_lim`, `inq_fi`, `total_cu_tl`, `inq_last_12m`, `acc_open_past_24mths`,
`avg_cur_bal`, `bc_open_to_buy`, `bc_util`, `delinq_amnt`, `mo_sin_old_il_acct`,
`mo_sin_old_rev_tl_op`, `mo_sin_rcnt_rev_tl_op`, `mo_sin_rcnt_tl`, `mort_acc`,
`mths_since_recent_bc`, `mths_since_recent_bc_dlq`, `mths_since_recent_inq`,
`mths_since_recent_revol_delinq`, `num_accts_ever_120_pd`, `num_actv_bc_tl`,
`num_actv_rev_tl`, `num_bc_sats`, `num_bc_tl`, `num_il_tl`, `num_op_rev_tl`,
`num_rev_accts`, `num_rev_tl_bal_gt_0`, `num_sats`, `num_tl_120dpd_2m`, `num_tl_30dpd`,
`num_tl_90g_dpd_24m`, `num_tl_op_past_12m`, `pct_tl_nvr_dlq`, `percent_bc_gt_75`,
`pub_rec_bankruptcies`, `tax_liens`, `tot_hi_cred_lim`, `total_bal_ex_mort`,
`total_bc_limit`, `total_il_high_credit_limit`

**Joint application (secondary applicant — 4.78% of loans):**
`annual_inc_joint`, `dti_joint`, `verification_status_joint`, `revol_bal_joint`,
all `sec_app_*` fields — safe to use for joint loans; treat null as "not a joint application"

**Temporal (for splitting only, or as a derived feature):**
`issue_d`, `earliest_cr_line` — use to derive credit age at origination

---

## 11. Leakage Sanity Check Protocol

Before any model is trained, verify:

1. **Payment field check**: `total_pymnt`, `recoveries`, `out_prncp` are NOT in the feature matrix.
2. **FICO check**: Only `fico_range_low` / `fico_range_high` are used, NOT `last_fico_range_*`.
3. **Grade check**: `grade`, `sub_grade`, `int_rate`, `installment` are NOT in the feature matrix.
4. **Settlement/hardship check**: No `hardship_*` or `settlement_*` columns in feature matrix.
5. **Correlation check**: Compute Pearson correlation of each feature against the target — flag any feature with |r| > 0.5 for manual review.
6. **Perfect prediction check**: Train a decision tree of depth 1 on each feature individually. Any feature achieving AUC > 0.95 alone is almost certainly leakage.

---

*See also: `data_dictionary.md` for full column-level detail, `project_specification.md` for target definition.*
