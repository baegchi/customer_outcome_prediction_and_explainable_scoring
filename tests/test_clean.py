"""
Unit tests for data cleaning, deduplication, target encoding,
missing value handling, and leakage column removal.
"""

import pytest
import numpy as np
import pandas as pd

from src.data.load import RAW_COLUMNS, FLOAT_COLUMNS
from src.data.clean import (
    ALL_LEAKAGE_COLUMNS,
    ALL_EXCLUDED_COLUMNS,
    LEAKAGE_POST_OUTCOME_COLS,
    LEAKAGE_CREDIT_REFRESH_COLS,
    LEAKAGE_LC_UNDERWRITING_COLS,
    LEAKAGE_HARDSHIP_COLS,
    LEAKAGE_SETTLEMENT_COLS,
    LEAKAGE_PAYMENT_PLAN_COLS,
    TARGET_POSITIVE,
    TARGET_NEGATIVE,
    TARGET_EXCLUDE,
    deduplicate_on_id,
    encode_target,
    remove_leakage_columns,
    assert_no_leakage,
    handle_missing_values,
    clean_data,
)


def create_full_synthetic_df(n_rows: int = 10) -> pd.DataFrame:
    """Generate synthetic DataFrame containing all 151 columns with sample values."""
    data = {}
    statuses = [
        "Fully Paid",
        "Charged Off",
        "Current",
        "Late (31-120 days)",
        "In Grace Period",
        "Default",
        "Does not meet the credit policy. Status:Fully Paid",
        "Does not meet the credit policy. Status:Charged Off",
        "Fully Paid",
        "Charged Off",
    ]

    for col in RAW_COLUMNS:
        if col in FLOAT_COLUMNS:
            data[col] = [float(i * 10 + 1) for i in range(n_rows)]
        else:
            if col == "id":
                data[col] = [f"{1000 + i}" for i in range(n_rows)]
            elif col == "loan_status":
                data[col] = [statuses[i % len(statuses)] for i in range(n_rows)]
            elif col == "term":
                data[col] = [" 36 months" if i % 2 == 0 else " 60 months" for i in range(n_rows)]
            elif col == "home_ownership":
                homes = ["MORTGAGE", "RENT", "OWN", "ANY", "NONE"]
                data[col] = [homes[i % len(homes)] for i in range(n_rows)]
            else:
                data[col] = [f"val_{col}_{i}" for i in range(n_rows)]

    return pd.DataFrame(data)


def test_deduplicate_on_id():
    """Test that duplicate IDs are properly identified and dropped."""
    df = create_full_synthetic_df(5)
    # Duplicate row 0 with modified value
    dup_row = df.iloc[[0]].copy()
    dup_row["loan_amnt"] = 99999.0
    df_with_dups = pd.concat([df, dup_row], ignore_index=True)
    assert len(df_with_dups) == 6

    deduped = deduplicate_on_id(df_with_dups, keep="first")
    assert len(deduped) == 5
    assert deduped["id"].nunique() == 5
    assert deduped.iloc[0]["loan_amnt"] == 1.0  # kept first occurrence


def test_encode_target_resolved_and_unresolved():
    """Test that target variable correctly encodes 0/1 and excludes unresolved categories."""
    df = create_full_synthetic_df(10)

    # 1. With drop_unresolved=True (default modeling set)
    encoded = encode_target(df, drop_unresolved=True)
    assert "target" in encoded.columns
    assert set(encoded["target"].unique()).issubset({0, 1})
    assert not any(status in encoded["loan_status"].values for status in TARGET_EXCLUDE)

    # 2. With drop_unresolved=False (all rows retained)
    encoded_all = encode_target(df, drop_unresolved=False)
    assert len(encoded_all) == len(df)
    # Unresolved statuses should have NaN target
    unresolved_mask = encoded_all["loan_status"].isin(TARGET_EXCLUDE)
    assert encoded_all.loc[unresolved_mask, "target"].isna().all()


def test_remove_leakage_columns():
    """Test that every excluded column and 'id' is removed."""
    df = create_full_synthetic_df(5)
    assert len(df.columns) == 151

    cleaned = remove_leakage_columns(df, drop_id=True)

    # Assert NONE of the excluded columns appear in the output frame
    assert not any(col in cleaned.columns for col in ALL_EXCLUDED_COLUMNS), (
        f"Found excluded columns in output: {[c for c in ALL_EXCLUDED_COLUMNS if c in cleaned.columns]}"
    )
    assert "id" not in cleaned.columns
    assert len(cleaned.columns) == 151 - len(ALL_EXCLUDED_COLUMNS) - 1  # 151 - 50 - 1 = 100


def test_assert_no_leakage_raises_on_leakage():
    """Test that assert_no_leakage catches prohibited columns and raises AssertionError."""
    df = create_full_synthetic_df(5)
    cleaned = remove_leakage_columns(df, drop_id=True)

    # Clean frame passes assertion
    assert_no_leakage(cleaned)

    # Infiltrate a post-outcome leakage column
    for leak_col in ["recoveries", "out_prncp", "last_fico_range_high", "grade", "hardship_flag"]:
        df_contaminated = cleaned.copy()
        df_contaminated[leak_col] = 100.0
        with pytest.raises(AssertionError) as excinfo:
            assert_no_leakage(df_contaminated)
        assert "DATA LEAKAGE DETECTED" in str(excinfo.value)
        assert leak_col in str(excinfo.value)


def test_handle_missing_values_patterns():
    """Test that missing value patterns per data_dictionary.md are correctly handled."""
    df = create_full_synthetic_df(5)

    # Inject NaNs into specific domain columns
    df.loc[0:2, "mths_since_last_delinq"] = np.nan
    df.loc[1:3, "open_acc_6m"] = np.nan
    df.loc[2:4, "collections_12_mths_ex_med"] = np.nan
    df["term"] = ["  36 months ", " 60 months", "36 months", "60 months ", " 36 months "]

    processed = handle_missing_values(df, create_indicators=True, delinquency_fill_value=999.0)

    # 1. Delinquency recency: missing filled with 999.0, indicator created
    assert "mths_since_last_delinq_missing" in processed.columns
    assert processed.loc[0:2, "mths_since_last_delinq_missing"].sum() == 3
    assert (processed.loc[0:2, "mths_since_last_delinq"] == 999.0).all()

    # 2. Bureau extended missing: indicator created, counts filled with 0
    assert "bureau_extended_missing" in processed.columns
    assert processed.loc[1:3, "bureau_extended_missing"].sum() == 3
    assert (processed.loc[1:3, "open_acc_6m"] == 0.0).all()

    # 3. Minor count columns: filled with 0
    assert (processed.loc[2:4, "collections_12_mths_ex_med"] == 0.0).all()

    # 4. Term whitespace stripped
    assert (processed["term"] == ["36 months", "60 months", "36 months", "60 months", "36 months"]).all()

    # 5. Home ownership consolidation
    assert not processed["home_ownership"].isin(["ANY", "NONE"]).any()


def test_end_to_end_clean_data():
    """Test full cleaning pipeline: deduping, target encoding, missing values, leakage removal."""
    df = create_full_synthetic_df(10)
    # Add a duplicate ID row
    dup = df.iloc[[0]].copy()
    df = pd.concat([df, dup], ignore_index=True)
    assert len(df) == 11

    cleaned = clean_data(df, deduplicate=True, drop_id=True, drop_leakage=True)

    # 1. Assert NONE of the excluded columns appear in the output frame
    for col in ALL_EXCLUDED_COLUMNS:
        assert col not in cleaned.columns, f"Prohibited column '{col}' found in output!"

    assert "id" not in cleaned.columns
    assert_no_leakage(cleaned)

    # 2. Assert target is encoded and resolved
    assert "target" in cleaned.columns
    assert set(cleaned["target"].unique()).issubset({0, 1})

    # 3. Assert deduplication worked
    assert len(cleaned) < 11
