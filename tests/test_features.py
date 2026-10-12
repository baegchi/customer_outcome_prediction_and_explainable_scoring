"""
Unit tests for feature engineering (src/features/build.py).
Covers domain feature groups, docstring motivation validation,
and critical edge cases (zero-division, missing inputs, invalid dates, etc.).
"""

import pytest
import numpy as np
import pandas as pd

from src.features.build import (
    build_parsed_raw_features,
    build_credit_score_features,
    build_affordability_ratio_features,
    build_log_transforms,
    build_delinquency_flags,
    build_delinquency_recency_features,
    build_missing_indicator_features,
    build_temporal_and_geographic_features,
    build_features,
)


# -----------------------------------------------------------------------------
# Test Docstring Motivations
# -----------------------------------------------------------------------------

def test_feature_functions_contain_motivations_in_docstrings():
    """Verify that every feature builder function includes the motivation from docs/feature_engineering.md."""
    feature_functions = [
        build_parsed_raw_features,
        build_credit_score_features,
        build_affordability_ratio_features,
        build_log_transforms,
        build_delinquency_flags,
        build_delinquency_recency_features,
        build_missing_indicator_features,
        build_temporal_and_geographic_features,
        build_features,
    ]

    for fn in feature_functions:
        doc = fn.__doc__
        assert doc is not None, f"Function {fn.__name__} is missing a docstring!"
        assert "Motivation" in doc, f"Function {fn.__name__} docstring missing 'Motivation' section!"
        assert len(doc.strip()) > 50, f"Function {fn.__name__} docstring is too brief!"


# -----------------------------------------------------------------------------
# Edge Cases: Zero Division & Affordability Ratios
# -----------------------------------------------------------------------------

def test_ratio_zero_division_and_negative_income():
    """Test that zero or negative annual income does not trigger ZeroDivisionError and flags invalidity."""
    df = pd.DataFrame({
        "loan_amnt": [10000.0, 15000.0, 5000.0, 20000.0],
        "annual_inc": [0.0, -5000.0, np.nan, 50000.0],
        "revol_bal": [1000.0, 2000.0, 500.0, 10000.0],
        "application_type": ["Individual", "Individual", "Individual", "Individual"],
    })

    res = build_affordability_ratio_features(df)

    # loan_to_income checks
    assert np.isnan(res.loc[0, "loan_to_income"])  # 0 income -> NaN
    assert np.isnan(res.loc[1, "loan_to_income"])  # negative income -> NaN
    assert np.isnan(res.loc[2, "loan_to_income"])  # NaN income -> NaN
    assert res.loc[3, "loan_to_income"] == pytest.approx(20000.0 / 50000.0)

    # loan_to_income_invalid flag checks
    assert res.loc[0, "loan_to_income_invalid"] == 1
    assert res.loc[1, "loan_to_income_invalid"] == 1
    assert res.loc[2, "loan_to_income_invalid"] == 1
    assert res.loc[3, "loan_to_income_invalid"] == 0

    # revol_bal_to_income checks
    assert np.isnan(res.loc[0, "revol_bal_to_income"])
    assert np.isnan(res.loc[1, "revol_bal_to_income"])
    assert res.loc[3, "revol_bal_to_income"] == pytest.approx(10000.0 / 50000.0)


def test_joint_application_income_and_dti_hierarchy():
    """Test that joint application values override individual values when application_type is Joint App."""
    df = pd.DataFrame({
        "application_type": ["Individual", "Joint App", "Joint App"],
        "annual_inc": [60000.0, 50000.0, 40000.0],
        "annual_inc_joint": [np.nan, 120000.0, np.nan],
        "dti": [15.0, 20.0, 25.0],
        "dti_joint": [np.nan, 10.0, np.nan],
        "loan_amnt": [10000.0, 10000.0, 10000.0],
        "revol_bal": [5000.0, 5000.0, 5000.0],
    })

    res = build_affordability_ratio_features(df)

    # Row 0: Individual -> uses annual_inc (60k) and dti (15.0)
    assert res.loc[0, "effective_annual_inc"] == 60000.0
    assert res.loc[0, "effective_dti"] == pytest.approx(15.0)

    # Row 1: Joint App with joint data -> uses annual_inc_joint (120k) and dti_joint (10.0)
    assert res.loc[1, "effective_annual_inc"] == 120000.0
    assert res.loc[1, "effective_dti"] == pytest.approx(10.0)
    assert res.loc[1, "loan_to_income"] == pytest.approx(10000.0 / 120000.0)

    # Row 2: Joint App but joint data NaN -> falls back to annual_inc (40k) and dti (25.0)
    assert res.loc[2, "effective_annual_inc"] == 40000.0
    assert res.loc[2, "effective_dti"] == pytest.approx(25.0)


# -----------------------------------------------------------------------------
# Edge Cases: Date Parsing & Credit Age
# -----------------------------------------------------------------------------

def test_credit_age_months_edge_cases():
    """Test credit_age_months for normal dates, negative differences, caps, and NaNs."""
    df = pd.DataFrame({
        "issue_d": ["Jan-2015", "Jan-2015", "Jan-2015", "Jan-2015", "InvalidDate"],
        "earliest_cr_line": [
            "Jan-2005",     # 10 years = 120 months
            "Jan-1950",     # 65 years = 780 months -> should cap at 600
            "Jan-2020",     # Credit line after issue -> negative -> should be NaN
            np.nan,         # Missing earliest_cr_line -> NaN
            "Jan-2000",     # Invalid issue date -> NaN
        ],
    })

    res = build_credit_score_features(df)
    ages = res["credit_age_months"]

    assert ages.iloc[0] == pytest.approx(120.0)
    assert ages.iloc[1] == pytest.approx(600.0)  # capped at 600
    assert np.isnan(ages.iloc[2])                # negative set to NaN
    assert np.isnan(ages.iloc[3])                # missing earliest_cr_line
    assert np.isnan(ages.iloc[4])                # invalid issue_d


def test_fico_mid_calculation():
    """Test fico_mid scalar calculation and handling of NaNs."""
    df = pd.DataFrame({
        "fico_range_low": [675.0, 700.0, np.nan],
        "fico_range_high": [679.0, 704.0, 720.0],
    })

    res = build_credit_score_features(df)
    assert res.loc[0, "fico_mid"] == pytest.approx(677.0)
    assert res.loc[1, "fico_mid"] == pytest.approx(702.0)
    assert np.isnan(res.loc[2, "fico_mid"])


# -----------------------------------------------------------------------------
# Edge Cases: Log Transforms
# -----------------------------------------------------------------------------

def test_log_transforms_zero_and_negative_safety():
    """Test that log transforms handle zero values safely using log1p."""
    df = pd.DataFrame({
        "annual_inc": [0.0, 100000.0, -100.0],
        "loan_amnt": [1000.0, 40000.0, 0.0],
        "revol_bal": [0.0, 5000.0, -50.0],
    })

    res = build_log_transforms(df)

    # log1p(0) == 0.0
    assert res.loc[0, "log_annual_inc"] == pytest.approx(0.0)
    assert res.loc[1, "log_annual_inc"] == pytest.approx(np.log1p(100000.0))
    assert res.loc[2, "log_annual_inc"] == pytest.approx(0.0)  # clipped at 0

    assert res.loc[0, "log_revol_bal"] == pytest.approx(0.0)
    assert res.loc[1, "log_revol_bal"] == pytest.approx(np.log1p(5000.0))
    assert res.loc[2, "log_revol_bal"] == pytest.approx(0.0)  # clipped at 0

    # loan_amnt clip at 1.0 ensures log(1.0) == 0.0 for zero
    assert res.loc[2, "log_loan_amnt"] == pytest.approx(0.0)


# -----------------------------------------------------------------------------
# Edge Cases: Missing Inputs (Columns Absent)
# -----------------------------------------------------------------------------

def test_functions_handle_missing_columns_gracefully():
    """Verify that feature functions do not raise KeyError when expected columns are absent."""
    empty_df = pd.DataFrame({"dummy": [1, 2, 3]})

    # All builder functions should complete without crashing
    df1 = build_parsed_raw_features(empty_df)
    df2 = build_credit_score_features(df1)
    df3 = build_affordability_ratio_features(df2)
    df4 = build_log_transforms(df3)
    df5 = build_delinquency_flags(df4)
    df6 = build_delinquency_recency_features(df5)
    df7 = build_missing_indicator_features(df6)
    df8 = build_temporal_and_geographic_features(df7)

    assert "dummy" in df8.columns
    assert len(df8) == 3


# -----------------------------------------------------------------------------
# Edge Cases: Ordinal Mappings and Categorical Parsing
# -----------------------------------------------------------------------------

def test_emp_length_ordinal_parsing():
    """Test that employment length maps correctly to 0-10, preserving nulls."""
    df = pd.DataFrame({
        "emp_length": ["< 1 year", "1 year", "5 years", "10+ years", np.nan, "Unrecognized"],
        "term": [" 36 months", "60 months ", 36, "36 months", "unknown", " 60 months"],
        "revol_util": [50.0, 150.0, -10.0, np.nan, "65.5", 25.0],
    })

    res = build_parsed_raw_features(df)

    assert res.loc[0, "emp_length_years"] == 0.0
    assert res.loc[1, "emp_length_years"] == 1.0
    assert res.loc[2, "emp_length_years"] == 5.0
    assert res.loc[3, "emp_length_years"] == 10.0
    assert np.isnan(res.loc[4, "emp_length_years"])
    assert np.isnan(res.loc[5, "emp_length_years"])

    # term_months
    assert res.loc[0, "term_months"] == 36.0
    assert res.loc[1, "term_months"] == 60.0
    assert res.loc[2, "term_months"] == 36.0

    # revol_util clipped to [0, 100]
    assert res.loc[1, "revol_util"] == 100.0
    assert res.loc[2, "revol_util"] == 0.0
    assert res.loc[4, "revol_util"] == 65.5


# -----------------------------------------------------------------------------
# End-to-End Master Pipeline Integration
# -----------------------------------------------------------------------------

def test_end_to_end_build_features():
    """Test full build_features pipeline creating all specified features."""
    df = pd.DataFrame({
        "loan_amnt": [10000.0, 25000.0],
        "term": [" 36 months", " 60 months"],
        "emp_length": ["3 years", "10+ years"],
        "annual_inc": [60000.0, 100000.0],
        "revol_bal": [5000.0, 12000.0],
        "revol_util": [45.0, 85.0],
        "dti": [18.5, 22.0],
        "fico_range_low": [680.0, 720.0],
        "fico_range_high": [684.0, 724.0],
        "issue_d": ["Mar-2015", "Oct-2016"],
        "earliest_cr_line": ["Jun-2005", "Aug-1998"],
        "delinq_2yrs": [0.0, 2.0],
        "pub_rec": [0.0, 1.0],
        "pub_rec_bankruptcies": [0.0, 1.0],
        "mths_since_last_delinq": [np.nan, 14.0],
        "open_acc_6m": [np.nan, 2.0],
        "zip_code": ["941xx", "100xx"],
    })

    res = build_features(df, drop_collinear_raw=True)

    expected_new = [
        "fico_mid",
        "credit_age_months",
        "effective_annual_inc",
        "effective_dti",
        "loan_to_income",
        "revol_bal_to_income",
        "log_annual_inc",
        "log_loan_amnt",
        "log_revol_bal",
        "has_delinq_2y",
        "has_pub_rec",
        "has_bankruptcy",
        "mths_since_last_delinq_imputed",
        "mths_since_last_delinq_missing",
        "bureau_extended_missing",
        "issue_month",
        "zip_3digit",
    ]

    for feat in expected_new:
        assert feat in res.columns, f"Expected feature {feat} not generated!"

    # Collinear raw components dropped
    assert "fico_range_low" not in res.columns
    assert "fico_range_high" not in res.columns

    # Values verify correctly
    assert res.loc[0, "fico_mid"] == 682.0
    assert res.loc[1, "fico_mid"] == 722.0
    assert res.loc[0, "has_delinq_2y"] == 0
    assert res.loc[1, "has_delinq_2y"] == 1
    assert res.loc[0, "bureau_extended_missing"] == 1
    assert res.loc[1, "bureau_extended_missing"] == 0
    assert res.loc[0, "mths_since_last_delinq_imputed"] == 999.0
    assert res.loc[1, "mths_since_last_delinq_imputed"] == 14.0
    assert res.loc[0, "zip_3digit"] == "941"
    assert res.loc[1, "zip_3digit"] == "100"
