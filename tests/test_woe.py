"""
Unit tests for WoE / IV pipeline (src/features/woe.py).
Verifies the exact implementation contract from docs/woe_iv.md:
  - Strict TRAIN-only fitting
  - Deterministic and immutable transformation (no refitting, no target access)
  - Null and unseen category handling
  - IV selection threshold filtering
  - Monotonicity checks
  - JSON serialization roundtrip
  - CV fold refitting behavior
"""

import copy
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from pandas.testing import assert_frame_equal

from src.features.woe import (
    fit_woe,
    transform_woe,
    save_woe_bins,
    load_woe_bins,
    compute_iv_table,
    validate_no_oot_leakage,
    BinArtifact,
)


@pytest.fixture
def sample_train_dev_data():
    """Create synthetic TRAIN and DEV datasets for WoE tests."""
    np.random.seed(42)
    n_train = 600
    n_dev = 200

    # Numerical features correlated with risk
    fico_train = np.random.normal(700, 40, n_train)
    dti_train = np.random.uniform(5, 40, n_train)
    # Binary feature
    has_delinq_train = np.random.binomial(1, 0.2, n_train)
    # Categorical feature
    state_train = np.random.choice(["CA", "NY", "TX", "FL", "RARE_ST"], size=n_train, p=[0.4, 0.3, 0.15, 0.1, 0.05])

    # Target probability influenced by features
    log_odds = -2.0 + (dti_train * 0.05) - ((fico_train - 700) * 0.03) + (has_delinq_train * 0.8)
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    y_train = np.random.binomial(1, prob, n_train)

    X_train = pd.DataFrame({
        "fico": fico_train,
        "dti": dti_train,
        "has_delinq": has_delinq_train,
        "state": state_train,
    })

    # DEV set with potential NaNs and an UNSEEN category
    fico_dev = np.random.normal(700, 40, n_dev)
    fico_dev[0:5] = np.nan  # inject missing
    dti_dev = np.random.uniform(5, 40, n_dev)
    has_delinq_dev = np.random.binomial(1, 0.2, n_dev).astype(float)
    has_delinq_dev[5:10] = np.nan  # inject missing
    state_dev_arr = np.random.choice(["CA", "NY", "TX", "NEW_UNSEEN_STATE"], size=n_dev, p=[0.4, 0.3, 0.2, 0.1])
    state_dev = pd.Series(state_dev_arr, dtype=object)
    state_dev.iloc[10:15] = np.nan  # inject missing

    X_dev = pd.DataFrame({
        "fico": fico_dev,
        "dti": dti_dev,
        "has_delinq": has_delinq_dev,
        "state": state_dev,
    })

    return X_train, pd.Series(y_train, name="target"), X_dev


def test_fit_woe_train_validation_guards(sample_train_dev_data):
    """Test that fit_woe raises ValueError on mismatched lengths, non-binary target, or missing features."""
    X_train, y_train, _ = sample_train_dev_data

    # 1. Non-binary target
    y_invalid = y_train.copy()
    y_invalid.iloc[0] = 5
    with pytest.raises(ValueError, match="invalid target values"):
        fit_woe(X_train, y_invalid, features=["fico", "dti"])

    # 2. Length mismatch
    with pytest.raises(ValueError, match="different lengths"):
        fit_woe(X_train.iloc[:-1], y_train, features=["fico", "dti"])

    # 3. Missing feature column
    with pytest.raises(ValueError, match="Features not found in X_train"):
        fit_woe(X_train, y_train, features=["fico", "non_existent_feature"])


def test_reproducibility_deterministic_and_no_refit(sample_train_dev_data):
    """
    CRITICAL TEST FROM BRIEF:
    Fitting on train and transforming dev must be 100% deterministic and must not refit or mutate bins.
    """
    X_train, y_train, X_dev = sample_train_dev_data
    features = ["fico", "dti", "has_delinq", "state"]

    # 1. Fit WoE on TRAIN
    bins = fit_woe(
        X_train=X_train,
        y_train=y_train,
        features=features,
        cat_features=["state"],
        binary_features=["has_delinq"],
        random_state=42,
    )

    # Make deep copy of bins to detect any mutation
    bins_before = copy.deepcopy(bins)

    # 2. First transformation of DEV
    dev_woe_1 = transform_woe(X_dev, bins, selected_only=False)

    # Assert bins artifact was NOT mutated during transformation
    validate_no_oot_leakage(bins_before, bins)

    # 3. Second transformation of DEV
    dev_woe_2 = transform_woe(X_dev, bins, selected_only=False)

    # Assert exactly deterministic bitwise identical outputs
    assert_frame_equal(dev_woe_1, dev_woe_2)
    assert dev_woe_1.shape == dev_woe_2.shape


def test_no_nulls_in_woe_output(sample_train_dev_data):
    """Test that transform_woe maps all NaN values to missing_woe, producing zero nulls."""
    X_train, y_train, X_dev = sample_train_dev_data
    features = ["fico", "dti", "has_delinq", "state"]

    bins = fit_woe(
        X_train=X_train,
        y_train=y_train,
        features=features,
        cat_features=["state"],
        binary_features=["has_delinq"],
    )

    # X_dev contains explicit NaNs in fico, has_delinq, and state
    assert X_dev["fico"].isna().sum() > 0
    assert X_dev["has_delinq"].isna().sum() > 0
    assert X_dev["state"].isna().sum() > 0

    dev_woe = transform_woe(X_dev, bins, selected_only=False)

    # Must contain 0 NaNs
    assert dev_woe.isna().sum().sum() == 0, (
        f"Found NaNs in WoE transformed DEV: {dev_woe.isna().sum()}"
    )


def test_unseen_category_handled(sample_train_dev_data):
    """Test that new categories in DEV not present in TRAIN map to unseen_woe without failing or creating NaN."""
    X_train, y_train, X_dev = sample_train_dev_data
    features = ["state"]

    bins = fit_woe(
        X_train=X_train,
        y_train=y_train,
        features=features,
        cat_features=["state"],
    )

    # DEV has "NEW_UNSEEN_STATE" which was not in TRAIN
    assert "NEW_UNSEEN_STATE" not in X_train["state"].values
    assert "NEW_UNSEEN_STATE" in X_dev["state"].values

    dev_woe = transform_woe(X_dev, bins, selected_only=False)

    unseen_mask = X_dev["state"] == "NEW_UNSEEN_STATE"
    expected_unseen_woe = bins["state"]["unseen_woe"]
    actual_unseen_woe = dev_woe.loc[unseen_mask, "state"]

    assert not actual_unseen_woe.isna().any()
    assert (actual_unseen_woe == expected_unseen_woe).all()


def test_iv_threshold_filter(sample_train_dev_data):
    """Test that features with IV < iv_threshold are filtered out when selected_only=True."""
    X_train, y_train, X_dev = sample_train_dev_data

    # Add pure binary noise feature (IV < 0.01)
    np.random.seed(99)
    X_train_noise = X_train.copy()
    X_train_noise["noise_feature"] = np.random.binomial(1, 0.5, len(X_train))
    X_dev_noise = X_dev.copy()
    X_dev_noise["noise_feature"] = np.random.binomial(1, 0.5, len(X_dev))

    features = ["fico", "noise_feature"]

    bins = fit_woe(
        X_train=X_train_noise,
        y_train=y_train,
        features=features,
        binary_features=["noise_feature"],
        iv_threshold=0.02,
    )

    # 1. Selected only: noise_feature must be dropped
    dev_filtered = transform_woe(X_dev_noise, bins, selected_only=True)
    assert "noise_feature" not in dev_filtered.columns
    assert "fico" in dev_filtered.columns

    # 2. Not selected only: all features retained
    dev_all = transform_woe(X_dev_noise, bins, selected_only=False)
    assert "noise_feature" in dev_all.columns
    assert "fico" in dev_all.columns


def test_woe_bin_monotonicity(sample_train_dev_data):
    """Test that numeric features fitted with monotonic_trend='auto' produce monotonic WoE values."""
    X_train, y_train, _ = sample_train_dev_data
    features = ["fico", "dti"]

    bins = fit_woe(
        X_train=X_train,
        y_train=y_train,
        features=features,
        monotonic_trend="auto",
    )

    for feat in features:
        meta = bins[feat]
        if meta["selected"] and meta["type"] == "numerical" and len(meta["splits"]) > 1:
            woe_vals = [meta["woe_map"][str(i)] for i in range(len(meta["splits"]) + 1)]
            diffs = np.diff(woe_vals)
            # Must be either non-decreasing or non-increasing
            is_increasing = (diffs >= -1e-6).all()
            is_decreasing = (diffs <= 1e-6).all()
            assert is_increasing or is_decreasing, f"Feature {feat} WoE is not monotonic: {woe_vals}"


def test_save_load_roundtrip(tmp_path: Path, sample_train_dev_data):
    """Test JSON serialization and deserialization produces an identical BinArtifact."""
    X_train, y_train, _ = sample_train_dev_data
    features = ["fico", "dti", "has_delinq", "state"]

    bins = fit_woe(
        X_train=X_train,
        y_train=y_train,
        features=features,
        cat_features=["state"],
        binary_features=["has_delinq"],
    )

    json_file = tmp_path / "woe_bins.json"
    save_woe_bins(bins, json_file)
    assert json_file.exists()

    loaded_bins = load_woe_bins(json_file)
    assert bins == loaded_bins


def test_compute_iv_table_and_suspicious_warning():
    """Test compute_iv_table returns sorted table and issues warning on IV >= 0.5."""
    synthetic_bins = {
        "good_feature": {
            "type": "numerical",
            "splits": [10.0],
            "woe_map": {"0": -0.2, "1": 0.3},
            "iv": 0.25,
            "selected": True,
            "rare_categories": [],
            "unseen_woe": 0.0,
            "missing_woe": 0.0,
            "n_bins": 2,
        },
        "weak_feature": {
            "type": "binary",
            "splits": [0.5],
            "woe_map": {"0": 0.01, "1": -0.01},
            "iv": 0.005,
            "selected": False,
            "rare_categories": [],
            "unseen_woe": 0.0,
            "missing_woe": 0.0,
            "n_bins": 2,
        },
        "suspicious_feature": {
            "type": "numerical",
            "splits": [5.0],
            "woe_map": {"0": -2.5, "1": 3.0},
            "iv": 0.85,  # >= 0.5
            "selected": True,
            "rare_categories": [],
            "unseen_woe": 0.0,
            "missing_woe": 0.0,
            "n_bins": 2,
        },
    }

    with pytest.warns(UserWarning, match="SUSPICIOUSLY HIGH IV"):
        iv_df = compute_iv_table(synthetic_bins)

    assert len(iv_df) == 3
    # Verify sorted descending by IV
    assert list(iv_df["feature"]) == ["suspicious_feature", "good_feature", "weak_feature"]
    assert iv_df.iloc[0]["iv"] == 0.85
    assert bool(iv_df.iloc[0]["selected"]) is True
    assert bool(iv_df.iloc[2]["selected"]) is False


def test_cv_fold_refit(sample_train_dev_data):
    """Test that WoE bins refit within CV folds produce distinct bin values on different data subsets."""
    X_train, y_train, _ = sample_train_dev_data
    features = ["fico", "dti"]

    # Fold 1: first half
    half = len(X_train) // 2
    bins_fold_1 = fit_woe(X_train.iloc[:half], y_train.iloc[:half], features=features)
    # Fold 2: second half
    bins_fold_2 = fit_woe(X_train.iloc[half:], y_train.iloc[half:], features=features)

    # Same keys
    assert set(bins_fold_1.keys()) == set(bins_fold_2.keys())
    # But different data produces different IV/WoE values (no cross-fold contamination)
    assert bins_fold_1["fico"]["iv"] != bins_fold_2["fico"]["iv"]
