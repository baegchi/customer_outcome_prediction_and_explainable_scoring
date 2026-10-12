"""
Unit tests for decision threshold optimization (src/evaluation/threshold.py).
Verifies:
  - Sweep analysis across probability cutoffs and monotonicity invariants
  - F1 score maximization
  - Youden's J statistic / KS divergence maximization
  - Target recall constraint satisfaction
  - Asymmetric business cost minimization
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.threshold import (
    optimize_threshold_cost,
    optimize_threshold_f1,
    optimize_threshold_target_recall,
    optimize_threshold_youden,
    sweep_thresholds,
)


@pytest.fixture
def sample_predictions():
    """Synthetic binary predictions with non-trivial risk distribution."""
    np.random.seed(42)
    n = 500
    y_true = np.random.binomial(1, 0.25, size=n)
    # Calibrated risk scores with separation
    noise = np.random.normal(0, 0.25, size=n)
    y_prob = np.clip(0.1 + 0.5 * y_true + noise, 0.01, 0.99)
    return y_true, y_prob


# =============================================================================
# 1. Sweep Thresholds Analysis
# =============================================================================

def test_sweep_thresholds_structure_and_monotonicity(sample_predictions):
    """Verify sweep_thresholds output structure and fundamental confusion matrix invariants."""
    y_true, y_prob = sample_predictions
    df_sweep = sweep_thresholds(y_true, y_prob, n_thresholds=20)

    assert len(df_sweep) == 20
    expected_cols = {
        "threshold",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1",
        "tpr",
        "fpr",
        "youden_j",
        "approval_rate",
        "default_rate_approved",
    }
    assert expected_cols.issubset(df_sweep.columns)

    # Invariants: as threshold increases, TP decreases (or stays same) and FN increases
    tps = df_sweep["tp"].values
    fps = df_sweep["fp"].values
    tns = df_sweep["tn"].values
    fns = df_sweep["fn"].values

    assert np.all(np.diff(tps) <= 0), "TP must be non-increasing with higher threshold"
    assert np.all(np.diff(fps) <= 0), "FP must be non-increasing with higher threshold"
    assert np.all(np.diff(fns) >= 0), "FN must be non-decreasing with higher threshold"
    assert np.all(np.diff(tns) >= 0), "TN must be non-decreasing with higher threshold"

    # Total applicants check
    total_applicants = len(y_true)
    for _, row in df_sweep.iterrows():
        assert row["tp"] + row["fp"] + row["tn"] + row["fn"] == total_applicants


# =============================================================================
# 2. F1 Score Maximization
# =============================================================================

def test_optimize_threshold_f1(sample_predictions):
    """Test F1 optimization finds global peak."""
    y_true, y_prob = sample_predictions
    res = optimize_threshold_f1(y_true, y_prob)

    assert res["objective"] == "max_f1"
    assert 0.0 < res["best_threshold"] < 1.0
    assert 0.0 < res["best_f1"] <= 1.0

    # Compare against full sweep to confirm it is the maximum
    df_sweep = sweep_thresholds(y_true, y_prob, n_thresholds=100)
    assert res["best_f1"] >= df_sweep["f1"].max() - 1e-6


# =============================================================================
# 3. Youden's J / KS Statistic Maximization
# =============================================================================

def test_optimize_threshold_youden(sample_predictions):
    """Verify Youden's J maximization and KS relationship."""
    y_true, y_prob = sample_predictions
    res = optimize_threshold_youden(y_true, y_prob)

    assert res["objective"] == "max_youden_j_ks"
    assert 0.0 < res["best_threshold"] < 1.0
    assert 0.0 < res["max_youden_j"] <= 1.0
    assert res["ks_percentage"] == pytest.approx(res["max_youden_j"] * 100.0)

    # Youden J = TPR - FPR
    assert res["max_youden_j"] == pytest.approx(res["tpr"] - res["fpr"], abs=1e-5)


# =============================================================================
# 4. Target Recall Constraint Satisfaction
# =============================================================================

def test_optimize_threshold_target_recall(sample_predictions):
    """Verify that target recall optimization satisfies minimum capture constraint."""
    y_true, y_prob = sample_predictions
    min_rec = 0.70
    res = optimize_threshold_target_recall(y_true, y_prob, min_recall=min_rec)

    assert res["target_met"] is True
    assert res["recall"] >= min_rec
    assert 0.0 < res["best_threshold"] < 1.0


# =============================================================================
# 5. Asymmetric Business Cost Minimization
# =============================================================================

def test_optimize_threshold_cost_directional_shift(sample_predictions):
    """
    Verify asymmetric cost sensitivity:
      - Penalizing FN heavily (e.g. 10x) forces a LOWER threshold (more conservative, declines more).
      - Penalizing FP heavily (e.g. 10x) forces a HIGHER threshold (approves more to preserve interest).
    """
    y_true, y_prob = sample_predictions

    # Case A: Heavy charge-off penalty (cost_fn=10, cost_fp=1)
    res_conservative = optimize_threshold_cost(y_true, y_prob, cost_fn=10.0, cost_fp=1.0)

    # Case B: Heavy lost-margin penalty (cost_fn=1, cost_fp=10)
    res_aggressive = optimize_threshold_cost(y_true, y_prob, cost_fn=1.0, cost_fp=10.0)

    assert res_conservative["best_threshold"] < res_aggressive["best_threshold"], (
        f"Conservative threshold ({res_conservative['best_threshold']}) should be lower than "
        f"aggressive threshold ({res_aggressive['best_threshold']})"
    )
    # Conservative threshold captures more bad loans (higher recall)
    assert res_conservative["recall"] >= res_aggressive["recall"]

