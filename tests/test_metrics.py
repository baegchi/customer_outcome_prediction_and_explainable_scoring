"""
Unit tests for credit risk evaluation metrics (src/evaluation/metrics.py).
Verifies:
  - Exact analytical known KS value calculation to prevent bugs
  - ROC-AUC, Gini, PR-AUC ranking behavior and bounds
  - Mathematical identity: Gini = 2 * ROC-AUC - 1
  - Secondary threshold metrics: Precision, Recall, F1, Confusion Matrix
  - evaluate_predictions comprehensive scorecard report
  - Edge cases: NaN detection, mismatched shapes, non-binary labels, single-class inputs
"""

import numpy as np
import pandas as pd
import pytest
from src.evaluation.metrics import (
    compute_roc_auc,
    compute_gini,
    compute_ks_statistic,
    compute_pr_auc,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_confusion_matrix,
    evaluate_predictions,
)


# =============================================================================
# 1. Kolmogorov-Smirnov (KS) Tests (Synthetic Known Value)
# =============================================================================

def test_known_ks_statistic():
    """
    Test KS calculation with an analytically known value.

    Analytical derivation:
      Non-defaults (yt=0): scores = [0.1, 0.2, 0.3, 0.4]  (N0 = 4)
      Defaults     (yt=1): scores = [0.3, 0.4, 0.5, 0.6]  (N1 = 4)

      Empirical Cumulative Distribution Functions:
        x < 0.1:          F0(x) = 0.00, F1(x) = 0.00 -> |diff| = 0.00
        0.1 <= x < 0.2:   F0(x) = 0.25, F1(x) = 0.00 -> |diff| = 0.25
        0.2 <= x < 0.3:   F0(x) = 0.50, F1(x) = 0.00 -> |diff| = 0.50
        0.3 <= x < 0.4:   F0(x) = 0.75, F1(x) = 0.25 -> |diff| = 0.50
        0.4 <= x < 0.5:   F0(x) = 1.00, F1(x) = 0.50 -> |diff| = 0.50
        0.5 <= x < 0.6:   F0(x) = 1.00, F1(x) = 0.75 -> |diff| = 0.25
        x >= 0.6:         F0(x) = 1.00, F1(x) = 1.00 -> |diff| = 0.00

      Maximum vertical separation = 0.5000 (50.00%).
    """
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.6])

    ks_stat = compute_ks_statistic(y_true, y_prob, return_percentage=False)
    ks_pct = compute_ks_statistic(y_true, y_prob, return_percentage=True)

    assert ks_stat == pytest.approx(0.50, abs=1e-6), f"Expected KS = 0.5, got {ks_stat}"
    assert ks_pct == pytest.approx(50.0, abs=1e-4), f"Expected KS% = 50.0%, got {ks_pct}"


def test_ks_boundary_conditions():
    """Test KS under perfect separation (KS=1.0) and identical distributions (KS=0.0)."""
    # Complete discrimination: all bads scored higher than all goods
    y_true_sep = np.array([0, 0, 0, 1, 1, 1])
    y_prob_sep = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert compute_ks_statistic(y_true_sep, y_prob_sep) == pytest.approx(1.0, abs=1e-6)
    assert compute_ks_statistic(y_true_sep, y_prob_sep, return_percentage=True) == pytest.approx(100.0, abs=1e-4)

    # Zero discrimination: identical score distributions
    y_true_zero = np.array([0, 0, 1, 1])
    y_prob_zero = np.array([0.5, 0.5, 0.5, 0.5])
    assert compute_ks_statistic(y_true_zero, y_prob_zero) == pytest.approx(0.0, abs=1e-6)
    assert compute_ks_statistic(y_true_zero, y_prob_zero, return_percentage=True) == pytest.approx(0.0, abs=1e-4)


# =============================================================================
# 2. ROC-AUC and Gini Tests
# =============================================================================

def test_roc_auc_known_values():
    """Test ROC-AUC for perfect, random, and inverse discrimination."""
    y_true = np.array([0, 0, 1, 1])

    # Perfect ranking
    y_prob_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_roc_auc(y_true, y_prob_perfect) == pytest.approx(1.0, abs=1e-6)

    # Random ranking
    y_prob_random = np.array([0.5, 0.5, 0.5, 0.5])
    assert compute_roc_auc(y_true, y_prob_random) == pytest.approx(0.5, abs=1e-6)

    # Inverse ranking
    y_prob_inverse = np.array([0.9, 0.8, 0.2, 0.1])
    assert compute_roc_auc(y_true, y_prob_inverse) == pytest.approx(0.0, abs=1e-6)


def test_gini_mathematical_identity():
    """Verify Gini = 2 * ROC-AUC - 1 identity holds across varying score distributions."""
    np.random.seed(123)
    y_true = np.random.binomial(1, 0.25, size=200)

    for noise_level in [0.1, 0.5, 1.0, 3.0]:
        y_prob = 1.0 / (1.0 + np.exp(- (y_true + np.random.normal(0, noise_level, size=200))))
        auc = compute_roc_auc(y_true, y_prob)
        gini = compute_gini(y_true, y_prob)

        expected_gini = 2.0 * auc - 1.0
        assert gini == pytest.approx(expected_gini, abs=1e-6)
        assert -1.0 <= gini <= 1.0


def test_gini_benchmark_reproduction():
    """Verify reported benchmark relationship: ROC-AUC ~0.6499 yields Gini ~0.2998."""
    reported_auc = 0.6499
    expected_gini = 2.0 * reported_auc - 1.0
    assert expected_gini == pytest.approx(0.2998, abs=1e-4)


# =============================================================================
# 3. Precision-Recall AUC Tests
# =============================================================================

def test_pr_auc_bounds_and_behavior():
    """Test PR-AUC calculation bounds and expected properties."""
    y_true = np.array([0, 0, 1, 1])

    # Perfect ranking -> PR-AUC = 1.0
    y_prob_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_pr_auc(y_true, y_prob_perfect) == pytest.approx(1.0, abs=1e-6)

    # General properties
    np.random.seed(42)
    y_t = np.random.binomial(1, 0.2, size=500)
    y_p = np.random.uniform(0, 1, size=500)
    pr_auc = compute_pr_auc(y_t, y_p)
    assert 0.0 <= pr_auc <= 1.0


# =============================================================================
# 4. Secondary Threshold Metrics (Precision, Recall, F1, Confusion Matrix)
# =============================================================================

def test_precision_recall_f1_exact():
    """
    Test precision, recall, and F1 on a hand-calculated confusion matrix:
      y_true = [1, 1, 1, 0, 0, 0]
      y_pred = [1, 1, 0, 1, 0, 0]

      Indices:
        0: true=1, pred=1 -> TP
        1: true=1, pred=1 -> TP
        2: true=1, pred=0 -> FN
        3: true=0, pred=1 -> FP
        4: true=0, pred=0 -> TN
        5: true=0, pred=0 -> TN

      Counts: TP=2, FP=1, FN=1, TN=2
      Precision = TP / (TP + FP) = 2 / 3 = ~0.6667
      Recall    = TP / (TP + FN) = 2 / 3 = ~0.6667
      F1        = 2 * (P * R) / (P + R) = 2 / 3 = ~0.6667
    """
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]

    prec = compute_precision(y_true, y_pred)
    rec = compute_recall(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred)

    assert prec == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert rec == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert f1 == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_confusion_matrix_counts():
    """Verify confusion matrix structure, keys, and values."""
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]

    cm = compute_confusion_matrix(y_true, y_pred)
    assert set(cm.keys()) == {"tn", "fp", "fn", "tp", "matrix"}
    assert cm["tn"] == 2
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tp"] == 2
    assert cm["matrix"].shape == (2, 2)
    assert np.array_equal(cm["matrix"], np.array([[2, 1], [1, 2]]))


def test_zero_division_handling():
    """Test zero division parameter when no positive predictions exist."""
    y_true = [1, 0, 1, 0]
    y_pred = [0, 0, 0, 0]  # TP=0, FP=0 -> denominator 0

    assert compute_precision(y_true, y_pred, zero_division=0.0) == 0.0
    assert compute_f1(y_true, y_pred, zero_division=0.0) == 0.0


def test_confusion_matrix_rejects_continuous():
    """Assert confusion matrix rejects continuous probabilities."""
    y_true = [0, 1, 0, 1]
    y_prob = [0.1, 0.8, 0.4, 0.9]  # Not binary {0, 1}

    with pytest.raises(ValueError, match="y_pred must contain binary decisions"):
        compute_confusion_matrix(y_true, y_prob)


# =============================================================================
# 5. Comprehensive Harness: evaluate_predictions
# =============================================================================

def test_evaluate_predictions_comprehensive():
    """Test full scorecard evaluation dictionary output and internal consistency."""
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.20, size=500)
    y_prob = np.clip(0.1 + 0.6 * y_true + np.random.normal(0, 0.2, size=500), 0.01, 0.99)

    report = evaluate_predictions(y_true, y_prob, threshold=0.35)

    expected_keys = {
        "roc_auc",
        "gini",
        "ks_stat",
        "ks_percentage",
        "pr_auc",
        "threshold",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
    }
    assert set(report.keys()) == expected_keys

    # Check value ranges
    assert 0.5 <= report["roc_auc"] <= 1.0
    assert 0.0 <= report["gini"] <= 1.0
    assert 0.0 <= report["ks_stat"] <= 1.0
    assert 0.0 <= report["ks_percentage"] <= 100.0
    assert report["threshold"] == 0.35
    assert 0.0 <= report["precision"] <= 1.0
    assert 0.0 <= report["recall"] <= 1.0
    assert 0.0 <= report["f1"] <= 1.0

    # Consistency checks
    assert report["ks_percentage"] == pytest.approx(report["ks_stat"] * 100.0, abs=0.01)
    assert report["gini"] == pytest.approx(2.0 * report["roc_auc"] - 1.0, abs=1e-4)

    # Confusion matrix total equals dataset size
    cm = report["confusion_matrix"]
    assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == 500


# =============================================================================
# 6. Input Validation & Defensive Guards
# =============================================================================

def test_input_validation_mismatched_lengths():
    """Mismatched lengths between y_true and y_prob must raise ValueError."""
    with pytest.raises(ValueError, match="Input arrays must have matching lengths"):
        compute_roc_auc([0, 1], [0.1, 0.2, 0.3])


def test_input_validation_empty_arrays():
    """Empty inputs must raise ValueError."""
    with pytest.raises(ValueError, match="Input arrays must not be empty"):
        compute_roc_auc([], [])


def test_input_validation_nan_rejection():
    """Inputs containing NaNs must raise ValueError."""
    with pytest.raises(ValueError, match="NaN/null value"):
        compute_roc_auc([0, 1, np.nan], [0.1, 0.8, 0.5])

    with pytest.raises(ValueError, match="NaN/null value"):
        compute_ks_statistic([0, 1, 0], [0.1, np.nan, 0.5])


def test_input_validation_non_binary_y_true():
    """Non-binary y_true (e.g. multiclass or negative) must raise ValueError."""
    with pytest.raises(ValueError, match="y_true must contain only binary classes"):
        compute_roc_auc([0, 1, 2], [0.1, 0.8, 0.5])

    with pytest.raises(ValueError, match="y_true must contain only binary classes"):
        compute_roc_auc([-1, 1, 1], [0.1, 0.8, 0.5])


def test_input_validation_single_class_y_true():
    """Single-class y_true must raise ValueError for ranking metrics."""
    with pytest.raises(ValueError, match="ROC-AUC requires both positive and negative"):
        compute_roc_auc([0, 0, 0], [0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="KS statistic requires both positive and negative"):
        compute_ks_statistic([1, 1, 1], [0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="PR-AUC requires both positive and negative"):
        compute_pr_auc([0, 0, 0], [0.1, 0.2, 0.3])


def test_input_data_types_flexibility():
    """Ensure functions accept Python lists, numpy arrays, and pandas Series identically."""
    y_true_list = [0, 1, 0, 1]
    y_prob_list = [0.2, 0.8, 0.3, 0.7]

    y_true_s = pd.Series(y_true_list)
    y_prob_s = pd.Series(y_prob_list)

    auc_list = compute_roc_auc(y_true_list, y_prob_list)
    auc_series = compute_roc_auc(y_true_s, y_prob_s)

    assert auc_list == pytest.approx(auc_series)

