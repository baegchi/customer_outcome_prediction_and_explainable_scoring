"""
Unit tests for chronological splitting and OOT leakage assertions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.split import (
    DEFAULT_TRAIN_END,
    DEFAULT_DEV_START,
    DEFAULT_DEV_END,
    DEFAULT_OOT_START,
    DEFAULT_OOT_END,
    parse_issue_date,
    chronological_split,
    assert_no_oot_leakage_in_split,
    save_splits_to_parquet,
)


def create_temporal_synthetic_df() -> pd.DataFrame:
    """Generate synthetic DataFrame with diverse issue_d dates across 2014–2018."""
    # 4 train dates (2014–2015), 3 dev dates (2016), 4 OOT dates (2017–2018)
    dates = [
        "Jan-2014", "Jun-2014", "Mar-2015", "Dec-2015",   # Train
        "Jan-2016", "Jun-2016", "Dec-2016",               # Dev
        "Jan-2017", "Jul-2017", "Mar-2018", "Nov-2018",   # OOT
    ]
    df = pd.DataFrame({
        "id": [f"id_{i}" for i in range(len(dates))],
        "issue_d": dates,
        "loan_amnt": [10000.0 * (i + 1) for i in range(len(dates))],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })
    return df


def test_parse_issue_date():
    """Test date parsing converts '%b-%Y' format to datetime correctly."""
    dates_series = pd.Series(["Jun-2007", "Dec-2015", "Mar-2018", "InvalidDate"])
    parsed = parse_issue_date(dates_series)
    assert pd.api.types.is_datetime64_any_dtype(parsed)
    assert parsed.iloc[0] == pd.Timestamp("2007-06-01")
    assert parsed.iloc[1] == pd.Timestamp("2015-12-01")
    assert parsed.iloc[2] == pd.Timestamp("2018-03-01")
    assert pd.isna(parsed.iloc[3])


def test_chronological_split_boundaries():
    """Test that chronological_split partitions rows strictly by issue_d."""
    df = create_temporal_synthetic_df()
    train_df, dev_df, oot_df = chronological_split(df, verbose=False)

    # Expected row counts: 4 train, 3 dev, 4 OOT
    assert len(train_df) == 4
    assert len(dev_df) == 3
    assert len(oot_df) == 4

    # Verify dates in train
    train_parsed = parse_issue_date(train_df["issue_d"])
    assert (train_parsed <= pd.Timestamp(DEFAULT_TRAIN_END)).all()

    # Verify dates in dev
    dev_parsed = parse_issue_date(dev_df["issue_d"])
    assert (dev_parsed >= pd.Timestamp(DEFAULT_DEV_START)).all()
    assert (dev_parsed <= pd.Timestamp(DEFAULT_DEV_END)).all()

    # Verify dates in OOT
    oot_parsed = parse_issue_date(oot_df["issue_d"])
    assert (oot_parsed >= pd.Timestamp(DEFAULT_OOT_START)).all()
    assert (oot_parsed <= pd.Timestamp(DEFAULT_OOT_END)).all()


def test_assert_no_oot_leakage_success():
    """Test that a clean split passes the assertion without error."""
    df = create_temporal_synthetic_df()
    train_df, dev_df, oot_df = chronological_split(df, verbose=False)

    # Must pass without raising AssertionError
    assert_no_oot_leakage_in_split(train_df, dev_df, oot_df)


def test_assert_no_oot_row_in_train():
    """Test that injecting an OOT-period row into train raises AssertionError."""
    df = create_temporal_synthetic_df()
    train_df, dev_df, oot_df = chronological_split(df, verbose=False)

    # Corrupt train set by injecting an OOT row from 2017
    leaked_oot_row = pd.DataFrame({
        "id": ["id_leaked_oot"],
        "issue_d": ["Jun-2017"],  # OOT period!
        "loan_amnt": [25000.0],
        "target": [1],
    })
    contaminated_train = pd.concat([train_df, leaked_oot_row], ignore_index=True)

    with pytest.raises(AssertionError) as excinfo:
        assert_no_oot_leakage_in_split(contaminated_train, dev_df, oot_df)

    assert "CRITICAL OOT LEAKAGE IN TRAIN" in str(excinfo.value)
    assert "Found 1 OOT-period rows in TRAIN partition!" in str(excinfo.value)


def test_assert_no_oot_row_in_dev():
    """Test that injecting an OOT-period row into dev raises AssertionError."""
    df = create_temporal_synthetic_df()
    train_df, dev_df, oot_df = chronological_split(df, verbose=False)

    # Corrupt dev set by injecting an OOT row from 2018
    leaked_oot_row = pd.DataFrame({
        "id": ["id_leaked_oot_2018"],
        "issue_d": ["Mar-2018"],  # OOT period!
        "loan_amnt": [30000.0],
        "target": [1],
    })
    contaminated_dev = pd.concat([dev_df, leaked_oot_row], ignore_index=True)

    with pytest.raises(AssertionError) as excinfo:
        assert_no_oot_leakage_in_split(train_df, contaminated_dev, oot_df)

    assert "CRITICAL OOT LEAKAGE IN DEV" in str(excinfo.value)
    assert "Found 1 OOT-period rows in DEV partition!" in str(excinfo.value)


def test_invalid_split_boundaries_raise_error():
    """Test that inverted/overlapping boundary parameters raise ValueError."""
    df = create_temporal_synthetic_df()

    with pytest.raises(ValueError) as excinfo:
        # train_end is after dev_start
        chronological_split(df, train_end="2016-06-30", dev_start="2016-01-01", verbose=False)

    assert "Invalid split boundaries" in str(excinfo.value)


def test_save_splits_to_parquet(tmp_path: Path):
    """Test that save_splits_to_parquet writes valid parquet files."""
    df = create_temporal_synthetic_df()
    train_df, dev_df, oot_df = chronological_split(df, verbose=False)

    out_dir = tmp_path / "interim"
    paths = save_splits_to_parquet(train_df, dev_df, oot_df, output_dir=out_dir)

    assert paths["train"].exists()
    assert paths["dev"].exists()
    assert paths["oot"].exists()

    # Load back and verify integrity
    loaded_train = pd.read_parquet(paths["train"])
    assert len(loaded_train) == len(train_df)
