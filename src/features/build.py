"""
Feature Engineering Module for Customer Outcome Prediction.
Implements domain-grounded feature transformations per docs/feature_engineering.md.
Each function corresponds to a distinct feature group, with docstrings detailing
the exact business and credit risk motivation from the specification.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Mapping definitions
# -----------------------------------------------------------------------------

EMP_LENGTH_MAP: Dict[str, float] = {
    "< 1 year": 0.0,
    "1 year": 1.0,
    "2 years": 2.0,
    "3 years": 3.0,
    "4 years": 4.0,
    "5 years": 5.0,
    "6 years": 6.0,
    "7 years": 7.0,
    "8 years": 8.0,
    "9 years": 9.0,
    "10+ years": 10.0,
}


# -----------------------------------------------------------------------------
# Feature Group 1: Pre-Engineering Parsing & Cleaning
# -----------------------------------------------------------------------------

def build_parsed_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-engineering type conversions and parsing.

    Motivation from docs/feature_engineering.md §1 & §2:
      - term: Raw categories are string (' 36 months', ' 60 months'). Extracting
        integer 36 or 60 provides an ordinal numeric representation.
      - emp_length: Raw values are strings ('< 1 year', '10+ years'). An ordinal
        mapping from 0 to 10 preserves the natural progression where longer employment
        signifies higher stability. Nulls are preserved for explicit imputation.
      - revol_util: Credit card utilization percentage. Valid domain is [0, 100%].
        Extreme outliers or data entry anomalies are clipped to [0, 100].

    Input columns:
      - term, emp_length, revol_util

    Outputs added/modified:
      - term_months (int/float)
      - emp_length_years (float)
      - revol_util (clipped float)
    """
    df = df.copy()

    # 1. Parse term to integer months
    if "term" in df.columns:
        if pd.api.types.is_numeric_dtype(df["term"]):
            df["term_months"] = df["term"].astype(float)
        else:
            df["term_months"] = (
                df["term"]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype(float)
            )

    # 2. Parse emp_length to ordinal years (0–10)
    if "emp_length" in df.columns:
        if pd.api.types.is_numeric_dtype(df["emp_length"]):
            df["emp_length_years"] = df["emp_length"].astype(float)
        else:
            clean_emp = df["emp_length"].astype(str).str.strip()
            df["emp_length_years"] = clean_emp.map(EMP_LENGTH_MAP)
            # Retain NaN for rows that were original NaNs
            df.loc[df["emp_length"].isna(), "emp_length_years"] = np.nan

    # 3. Clip revol_util to [0, 100]
    if "revol_util" in df.columns:
        df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce").clip(lower=0.0, upper=100.0)

    return df


# -----------------------------------------------------------------------------
# Feature Group 2: Credit Score & Bureau History Features
# -----------------------------------------------------------------------------

def build_credit_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Credit bureau score and account seasoning features.

    Motivation from docs/feature_engineering.md §3.1 & §3.2:
      - fico_mid: Lending Club reports origination FICO as a 4-point band
        (e.g., 675–679 via fico_range_low and fico_range_high). Using both features
        introduces near-perfect collinearity. The midpoint provides a single scalar
        capturing the full creditworthiness signal without redundancy.
      - credit_age_months: Length of credit history is one of the five core FICO
        components. Longer credit history indicates an established borrower with
        lower default risk. Converting earliest_cr_line to elapsed months at loan
        origination (issue_d) removes absolute calendar year confounds and produces
        a stationary credit-tenure numeric. Capped at 600 months (50 years);
        negative values (impossible credit line after issue) set to NaN.

    Input columns:
      - fico_range_low, fico_range_high, issue_d, earliest_cr_line

    Outputs added:
      - fico_mid (float32)
      - credit_age_months (float32)
    """
    df = df.copy()

    # 1. fico_mid
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        low = pd.to_numeric(df["fico_range_low"], errors="coerce")
        high = pd.to_numeric(df["fico_range_high"], errors="coerce")
        df["fico_mid"] = ((low + high) / 2.0).astype(np.float32)

    # 2. credit_age_months
    if "issue_d" in df.columns and "earliest_cr_line" in df.columns:
        issue_dt = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        # earliest_cr_line format can be '%b-%Y' or '%b-%y' or standard datetime
        earliest_dt = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
        # Fallback for 2-digit years if any
        if earliest_dt.isna().sum() > issue_dt.isna().sum():
            earliest_dt = earliest_dt.fillna(pd.to_datetime(df["earliest_cr_line"], errors="coerce"))

        # Calculate month difference: (y_issue - y_early)*12 + (m_issue - m_early)
        diff_months = (issue_dt.dt.year - earliest_dt.dt.year) * 12 + (issue_dt.dt.month - earliest_dt.dt.month)

        # Handle edge cases: negative values are invalid; cap at 600 months (50 years)
        diff_months = np.where(diff_months < 0, np.nan, diff_months)
        diff_months = np.where(diff_months > 600, 600.0, diff_months)

        df["credit_age_months"] = pd.Series(diff_months, index=df.index, dtype=np.float32)

    return df


# -----------------------------------------------------------------------------
# Feature Group 3: Affordability & Debt Burden Ratios
# -----------------------------------------------------------------------------

def build_affordability_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Affordability, leverage, and household debt-to-income ratios.

    Motivation from docs/feature_engineering.md §3.3, §3.4, §3.18, §3.19, §3.20:
      - is_joint_app: Binary flag indicating joint applications (4.8% of book)
        where underwriting accounts for secondary applicant finances.
      - effective_annual_inc: Harmonizes single and joint income. For joint applications,
        combined income is recorded in annual_inc_joint. A unified income metric ensures
        downstream affordability ratios reflect true household debt-service capacity.
      - effective_dti: Harmonizes individual dti with joint dti (dti_joint).
      - loan_to_income: Measures loan size relative to annual earnings. Higher loan
        burden directly amplifies debt servicing difficulty and default probability.
        Safely handles zero/negative income by setting ratio to NaN and flagging invalidity.
      - revol_bal_to_income: Ratio of existing revolving card debt to annual earnings.
        Captures existing leverage burden independent of line utilization.

    Input columns:
      - application_type, annual_inc, annual_inc_joint, dti, dti_joint, loan_amnt, revol_bal

    Outputs added:
      - is_joint_app (int8)
      - effective_annual_inc (float64)
      - effective_dti (float32)
      - loan_to_income (float32)
      - loan_to_income_invalid (int8)
      - revol_bal_to_income (float32)
    """
    df = df.copy()

    # 1. is_joint_app
    if "application_type" in df.columns:
        df["is_joint_app"] = (df["application_type"].astype(str).str.strip() == "Joint App").astype(np.int8)
    else:
        df["is_joint_app"] = np.int8(0)

    # 2. effective_annual_inc
    if "annual_inc" in df.columns:
        ann_inc = pd.to_numeric(df["annual_inc"], errors="coerce")
        if "annual_inc_joint" in df.columns:
            ann_inc_joint = pd.to_numeric(df["annual_inc_joint"], errors="coerce")
            df["effective_annual_inc"] = np.where(
                (df["is_joint_app"] == 1) & ann_inc_joint.notna(),
                ann_inc_joint,
                ann_inc,
            )
        else:
            df["effective_annual_inc"] = ann_inc
    else:
        df["effective_annual_inc"] = np.nan

    # 3. effective_dti
    if "dti" in df.columns:
        raw_dti = pd.to_numeric(df["dti"], errors="coerce")
        if "dti_joint" in df.columns:
            joint_dti = pd.to_numeric(df["dti_joint"], errors="coerce")
            df["effective_dti"] = np.where(
                (df["is_joint_app"] == 1) & joint_dti.notna(),
                joint_dti,
                raw_dti,
            ).astype(np.float32)
        else:
            df["effective_dti"] = raw_dti.astype(np.float32)

    # 4. loan_to_income (with zero-division & invalid checks)
    if "loan_amnt" in df.columns and "effective_annual_inc" in df.columns:
        l_amnt = pd.to_numeric(df["loan_amnt"], errors="coerce")
        inc = pd.to_numeric(df["effective_annual_inc"], errors="coerce")

        valid_inc_mask = (inc > 0) & inc.notna()
        df["loan_to_income"] = np.where(valid_inc_mask, l_amnt / inc, np.nan).astype(np.float32)
        df["loan_to_income_invalid"] = (~valid_inc_mask).astype(np.int8)

    # 5. revol_bal_to_income (with zero-division check)
    if "revol_bal" in df.columns and "effective_annual_inc" in df.columns:
        r_bal = pd.to_numeric(df["revol_bal"], errors="coerce")
        inc = pd.to_numeric(df["effective_annual_inc"], errors="coerce")

        valid_inc_mask = (inc > 0) & inc.notna()
        df["revol_bal_to_income"] = np.where(valid_inc_mask, r_bal / inc, np.nan).astype(np.float32)

    return df


# -----------------------------------------------------------------------------
# Feature Group 4: Log Transformations
# -----------------------------------------------------------------------------

def build_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Logarithmic scale transformations for right-skewed monetary variables.

    Motivation from docs/feature_engineering.md §3.5, §3.6, §3.7:
      - log_annual_inc: annual_inc spans multiple orders of magnitude ($0 to millions)
        with severe right skew. Log-transform (np.log1p) normalizes variance, improves
        linear calibration in Logistic Regression, and facilitates monotonic WoE binning.
        log1p ensures mathematical safety when income is 0 (log(0+1)=0).
      - log_loan_amnt: Compresses loan amount variance ($500 to $40,000) for stable
        scorecard binning and linear modeling.
      - log_revol_bal: Revolving credit balance is heavily right-skewed with a large spike
        at zero. np.log1p compresses high balances while mapping zero to zero.

    Input columns:
      - annual_inc, loan_amnt, revol_bal

    Outputs added:
      - log_annual_inc (float32)
      - log_loan_amnt (float32)
      - log_revol_bal (float32)
    """
    df = df.copy()

    # 1. log_annual_inc
    if "annual_inc" in df.columns:
        inc = pd.to_numeric(df["annual_inc"], errors="coerce").clip(lower=0.0)
        df["log_annual_inc"] = np.log1p(inc).astype(np.float32)

    # 2. log_loan_amnt
    if "loan_amnt" in df.columns:
        amnt = pd.to_numeric(df["loan_amnt"], errors="coerce").clip(lower=1.0)
        df["log_loan_amnt"] = np.log(amnt).astype(np.float32)

    # 3. log_revol_bal
    if "revol_bal" in df.columns:
        bal = pd.to_numeric(df["revol_bal"], errors="coerce").clip(lower=0.0)
        df["log_revol_bal"] = np.log1p(bal).astype(np.float32)

    return df


# -----------------------------------------------------------------------------
# Feature Group 5: Delinquency & Derogatory Binary Flags
# -----------------------------------------------------------------------------

def build_delinquency_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flags for past delinquency, public records, and bankruptcy events.

    Motivation from docs/feature_engineering.md §3.9, §3.10, §3.11:
      - has_delinq_2y: The vast majority of prime/near-prime borrowers have 0 recent
        delinquencies. Converting delinq_2yrs to a clean binary flag isolates the primary
        credit event signal (any past delinquency vs. none) without sparse high-count bins.
      - has_pub_rec: Public records (civil judgements, tax liens, bankruptcies) are rare
        negative events; binary presence provides cleaner discriminatory separation.
      - has_bankruptcy: Bankruptcy represents a distinct, severe credit failure event.
        Isolating pub_rec_bankruptcies into a clean binary flag provides strong independent
        default predictive power.

    Input columns:
      - delinq_2yrs, pub_rec, pub_rec_bankruptcies

    Outputs added:
      - has_delinq_2y (int8)
      - has_pub_rec (int8)
      - has_bankruptcy (int8)
    """
    df = df.copy()

    # 1. has_delinq_2y
    if "delinq_2yrs" in df.columns:
        d2 = pd.to_numeric(df["delinq_2yrs"], errors="coerce").fillna(0.0)
        df["has_delinq_2y"] = (d2 > 0).astype(np.int8)

    # 2. has_pub_rec
    if "pub_rec" in df.columns:
        pr = pd.to_numeric(df["pub_rec"], errors="coerce").fillna(0.0)
        df["has_pub_rec"] = (pr > 0).astype(np.int8)

    # 3. has_bankruptcy
    if "pub_rec_bankruptcies" in df.columns:
        bk = pd.to_numeric(df["pub_rec_bankruptcies"], errors="coerce").fillna(0.0)
        df["has_bankruptcy"] = (bk > 0).astype(np.int8)

    return df


# -----------------------------------------------------------------------------
# Feature Group 6: Delinquency Recency Imputations
# -----------------------------------------------------------------------------

def build_delinquency_recency_features(
    df: pd.DataFrame,
    delinquency_fill_value: float = 999.0,
) -> pd.DataFrame:
    """
    Imputation of delinquency and derogatory event recency variables.

    Motivation from docs/feature_engineering.md §3.12–§3.16:
      - mths_since_last_delinq_imputed: 51.25% of borrowers have no recorded delinquency.
        Imputing with 999.0 encodes "never or very long ago" as an extreme ordinal value,
        preserving monotonicity where larger numbers signify superior credit behaviour.
        This aligns with standard credit scorecard methodology.
      - mths_since_last_record_imputed: 84.11% null indicates no public record. Same rationale.
      - mths_since_last_major_derog_imputed: 74.31% null indicates no 90+ DPD derogatory.
      - mths_since_recent_bc_dlq_imputed: 77.01% null indicates no bankcard delinquency.
      - mths_since_recent_revol_delinq_imputed: 67.25% null indicates no revolving delinquency.

    Input columns:
      - mths_since_last_delinq, mths_since_last_record, mths_since_last_major_derog,
        mths_since_recent_bc_dlq, mths_since_recent_revol_delinq

    Outputs added:
      - mths_since_last_delinq_imputed (float32)
      - mths_since_last_record_imputed (float32)
      - mths_since_last_major_derog_imputed (float32)
      - mths_since_recent_bc_dlq_imputed (float32)
      - mths_since_recent_revol_delinq_imputed (float32)
    """
    df = df.copy()

    recency_cols = [
        "mths_since_last_delinq",
        "mths_since_last_record",
        "mths_since_last_major_derog",
        "mths_since_recent_bc_dlq",
        "mths_since_recent_revol_delinq",
    ]

    for col in recency_cols:
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").fillna(delinquency_fill_value)
            df[f"{col}_imputed"] = val.clip(upper=delinquency_fill_value).astype(np.float32)

    return df


# -----------------------------------------------------------------------------
# Feature Group 7: Missing-Value Indicators & Bureau Vintage Era
# -----------------------------------------------------------------------------

def build_missing_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Informative missing-value indicators and bureau vintage indicators.

    Motivation from docs/feature_engineering.md §3.17 & §4 (Tier 3):
      - bureau_extended_missing: The 38.31% missingness in extended bureau trades
        (open_acc_6m, open_act_il, etc.) represents an institutional shift around 2014
        when Lending Club expanded its bureau data pull. Captures vintage era legitimately
        known at loan origination.
      - emp_length_missing: 6.5% missingness; captures undisclosed employment status
        (self-employed, gig-workers, or unverified borrowers).
      - Delinquency recency indicators: Binary flags capturing the pristine credit
        signal of borrowers who have never experienced delinquencies or public records.

    Input columns:
      - open_acc_6m, emp_length, mths_since_last_delinq, mths_since_last_record,
        mths_since_last_major_derog, mths_since_recent_bc_dlq, mths_since_recent_revol_delinq,
        mths_since_recent_inq, il_util

    Outputs added:
      - bureau_extended_missing (int8)
      - emp_length_missing (int8)
      - mths_since_last_delinq_missing (int8)
      - mths_since_last_record_missing (int8)
      - mths_since_last_major_derog_missing (int8)
      - mths_since_recent_bc_dlq_missing (int8)
      - mths_since_recent_revol_delinq_missing (int8)
      - mths_since_recent_inq_missing (int8)
      - il_util_missing (int8)
    """
    df = df.copy()

    # 1. bureau_extended_missing (using open_acc_6m as proxy for the co-missing block)
    if "open_acc_6m" in df.columns:
        df["bureau_extended_missing"] = df["open_acc_6m"].isna().astype(np.int8)

    # 2. emp_length_missing
    if "emp_length" in df.columns:
        df["emp_length_missing"] = df["emp_length"].isna().astype(np.int8)

    # 3. Delinquency and bureau indicators
    indicator_sources = [
        "mths_since_last_delinq",
        "mths_since_last_record",
        "mths_since_last_major_derog",
        "mths_since_recent_bc_dlq",
        "mths_since_recent_revol_delinq",
        "mths_since_recent_inq",
        "il_util",
    ]

    for col in indicator_sources:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(np.int8)

    return df


# -----------------------------------------------------------------------------
# Feature Group 8: Temporal & Geographic Context Features
# -----------------------------------------------------------------------------

def build_temporal_and_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Seasonality and geographic origin indicators.

    Motivation from docs/feature_engineering.md §3.21 & §3.22:
      - issue_month: Captures cyclical seasonality in consumer borrowing (holiday
        debt spikes in Q4 vs. tax return debt repayments in Q1-Q2) independent of year.
        Strictly excludes issue_year to avoid leaking temporal split boundaries.
      - zip_3digit: Masked zip code prefix (e.g. '112xx' -> '112') grouping regional
        economic micro-climates for WoE binning.

    Input columns:
      - issue_d, zip_code

    Outputs added:
      - issue_month (int8)
      - zip_3digit (object)
    """
    df = df.copy()

    # 1. issue_month (1–12)
    if "issue_d" in df.columns:
        dt_issue = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        df["issue_month"] = dt_issue.dt.month.fillna(0).astype(np.int8)

    # 2. zip_3digit (3-character geographic cluster)
    if "zip_code" in df.columns:
        df["zip_3digit"] = df["zip_code"].astype(str).str.strip().str[:3]

    return df


# -----------------------------------------------------------------------------
# Master Feature Engineering Pipeline
# -----------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    drop_collinear_raw: bool = False,
) -> pd.DataFrame:
    """
    Execute all feature engineering groups in sequence per docs/feature_engineering.md.

    Motivation from docs/feature_engineering.md §0 & §7:
      Orchestrates the complete deterministic feature generation pipeline in strict sequence.
      Ensures informative missing indicators (Tier 3) are extracted before domain imputations,
      and constructs derived ratios and log-transforms from cleaned, leakage-free inputs.

    Pipeline sequence:
      1. Pre-engineering parsing (term, emp_length, revol_util)
      2. Credit score & bureau age features (fico_mid, credit_age_months)
      3. Affordability & leverage ratios (effective_annual_inc, loan_to_income, etc.)
      4. Logarithmic scale transforms (log_annual_inc, log_loan_amnt, log_revol_bal)
      5. Delinquency and derogatory flags (has_delinq_2y, has_pub_rec, has_bankruptcy)
      6. Delinquency recency imputations (mths_since_*_imputed)
      7. Informative missing indicators (Tier 3)
      8. Temporal & geographic context (issue_month, zip_3digit)

    Args:
        df: Cleaned DataFrame (post leakage removal).
        drop_collinear_raw: If True, drops raw components that are redundant
                           (e.g. fico_range_low, fico_range_high).

    Returns:
        DataFrame enriched with all engineered features.
    """
    df = df.copy()

    # 1. Missing indicators (before imputation alters missingness)
    df = build_missing_indicator_features(df)

    # 2. Pre-engineering parsing
    df = build_parsed_raw_features(df)

    # 3. Credit score features
    df = build_credit_score_features(df)

    # 4. Affordability ratios
    df = build_affordability_ratio_features(df)

    # 5. Log transforms
    df = build_log_transforms(df)

    # 6. Delinquency binary flags
    df = build_delinquency_flags(df)

    # 7. Delinquency recency imputations
    df = build_delinquency_recency_features(df)

    # 8. Temporal and geographic features
    df = build_temporal_and_geographic_features(df)

    if drop_collinear_raw:
        redundant_cols = [c for c in ["fico_range_low", "fico_range_high"] if c in df.columns]
        df = df.drop(columns=redundant_cols)

    return df


if __name__ == "__main__":
    from src.data.load import load_raw
    from src.data.clean import clean_data

    print("Loading raw sample (1,000 rows)...")
    df_raw = load_raw(nrows=1000)
    df_clean = clean_data(df_raw, drop_unresolved_target=True)

    print("Building engineered features...")
    df_feat = build_features(df_clean)
    print(f"Engineered feature matrix shape: {df_feat.shape}")

    # Inspect newly created features
    new_features = [
        "fico_mid", "credit_age_months", "effective_annual_inc",
        "loan_to_income", "revol_bal_to_income", "log_annual_inc",
        "has_delinq_2y", "has_pub_rec", "has_bankruptcy",
        "mths_since_last_delinq_imputed", "bureau_extended_missing",
        "issue_month", "zip_3digit",
    ]
    print("\nSample engineered values:")
    print(df_feat[new_features].head(5))
    print("\nSUCCESS: All feature engineering groups executed cleanly!")
