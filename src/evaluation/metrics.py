"""
Credit Risk Scorecard Evaluation Metrics Module.
Implements primary threshold-free ranking metrics (ROC-AUC, Gini, KS, PR-AUC)
and secondary threshold-dependent operational metrics (Precision, Recall, F1, Confusion Matrix)
per docs/project_specification.md.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _prepare_inputs(
    y_true: Union[Sequence, np.ndarray, pd.Series],
    y_pred_or_prob: Union[Sequence, np.ndarray, pd.Series],
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and convert inputs to 1D numpy float/int arrays without NaNs."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred_or_prob).ravel()

    if len(yt) != len(yp):
        raise ValueError(f"Input arrays must have matching lengths: {len(yt)} vs {len(yp)}.")

    if len(yt) == 0:
        raise ValueError("Input arrays must not be empty.")

    # Check for NaN values
    valid_mask = ~np.isnan(yt) & ~np.isnan(yp)
    if not np.all(valid_mask):
        raise ValueError(
            f"Input arrays contain {np.sum(~valid_mask)} NaN/null value(s). "
            "Evaluation metrics require clean, non-null predictions."
        )

    # Check that y_true is strictly binary
    unique_vals = np.unique(yt)
    if not set(unique_vals).issubset({0, 1}):
        raise ValueError(
            f"y_true must contain only binary classes {0, 1}, but found: {unique_vals}."
        )

    return yt, yp


# -----------------------------------------------------------------------------
# Primary Ranking Metrics (Threshold-Independent)
# -----------------------------------------------------------------------------

def compute_roc_auc(
    y_true: Union[Sequence, np.ndarray, pd.Series],
    y_prob: Union[Sequence, np.ndarray, pd.Series],
) -> float:
    """
    Compute Receiver Operating Characteristic Area Under the Curve (ROC-AUC).

    Motivation from docs/project_specification.md §5.1:
      ROC-AUC represents the probability that a randomly selected defaulted borrower
      receives a higher predicted risk score than a randomly selected non-defaulted borrower.
      Because it is threshold-independent, it is the primary industry standard for
      evaluating discriminatory ranking power in retail credit underwriting.

    Args:
        y_true: Ground truth binary default labels (1 = default, 0 = non-default).
        y_prob: Predicted default probabilities or continuous risk scores.

    Returns:
        float: ROC-AUC score in [0.0, 1.0].
    """
    yt, yp = _prepare_inputs(y_true, y_prob)

    # If only 1 class is present, AUC is undefined
    if len(np.unique(yt)) < 2:
        raise ValueError("ROC-AUC requires both positive and negative classes in y_true.")

    return float(roc_auc_score(yt, yp))


def compute_gini(
    y_true: Union[Sequence, np.ndarray, pd.Series],
    y_prob: Union[Sequence, np.ndarray, pd.Series],
) -> float:
    """
    Compute the Gini Coefficient (Somers' D): Gini = 2 * ROC-AUC - 1.

    Motivation from docs/project_specification.md §5.1:
      The Gini coefficient linearly rescales ROC-AUC to [-1.0, 1.0], where 0.0
      represents random chance and 1.0 represents perfect discrimination.
      It is the regulatory reporting metric under Basel II/III Internal Ratings-Based (IRB)
      frameworks for scorecard validation.

    Args:
        y_true: Ground truth binary default labels.
        y_prob: Predicted default probabilities or continuous risk scores.

    Returns:
        float: Gini coefficient in [-1.0, 1.0].
    """
    auc = compute_roc_auc(y_true, y_prob)
    return float(2.0 * auc - 1.0)


def compute_ks_statistic(
    y_true: Union[Sequence, np.ndarray, pd.Series],
    y_prob: Union[Sequence, np.ndarray, pd.Series],
    return_percentage: bool = False,
) -> float:
    """
    Compute Kolmogorov-Smirnov (KS) statistic.

    Motivation from docs/project_specification.md §5.1:
      The KS statistic measures the maximum vertical distance between the cumulative
      distribution functions (CDFs) of predicted scores for defaulters (class 1)
      and non-defaulters (class 0):
          KS = max_score | F_default(score) - F_non_default(score) |
             = max_threshold (TPR(threshold) - FPR(threshold))
      A higher KS indicates greater separation between good and bad borrower populations,
      allowing credit risk teams to set cutoffs with high bad-capture rates.

    Args:
        y_true: Ground truth binary default labels.
        y_prob: Predicted default probabilities or continuous risk scores.
        return_percentage: If True, returns KS scaled to [0, 100] (standard in credit reporting).
                           If False (default), returns KS in [0.0, 1.0].

    Returns:
        float: KS statistic value.
    """
    yt, yp = _prepare_inputs(y_true, y_prob)

    if len(np.unique(yt)) < 2:
        raise ValueError("KS statistic requires both positive and negative classes in y_true.")

    scores_default = yp[yt == 1]
    scores_non_default = yp[yt == 0]

    # Two-sample Kolmogorov-Smirnov statistic
    ks_res = ks_2samp(scores_default, scores_non_default)
    ks_val = float(ks_res.statistic)

    if return_percentage:
        return float(ks_val * 100.0)
    return ks_val


def compute_pr_auc(
    y_true: Union[Sequence, np.ndarray, pd.Series],
    y_prob: Union[Sequence, np.ndarray, pd.Series],
) -> float:
    """
    Compute Area Under the Precision-Recall Curve (PR-AUC / Average Precision).

    Motivation from docs/project_specification.md §5.1:
      In imbalanced credit datasets (~20% default rate), ROC-AUC can be overly
      optimistic because a large pool of true negatives depresses the false positive rate.
      PR-AUC focuses directly on the minority positive class (defaults), evaluating
      the trade-off between precision (false alarm cost) and recall (bad debt capture).

    Args:
        y_true: Ground truth binary default labels.
        y_prob: Predicted default probabilities.

    Returns:
        float: PR-AUC score in [0.0, 1.0].
    """
    yt, yp = _prepare_inputs(y_true, y_prob)

    if len(np.unique(yt)) < 2:
        raise ValueError("PR-AUC requires both positive and negative classes in y_true.")

    return float(average_precision_score(yt, yp))


# -----------------------------------------------------------------------------
# Secondary Threshold-Dependent Metrics
# -----------------------------------------------------------------------------

def compute_precision(
    y_true: Union[Sequence, np.ndarray, pd.Series],
    y_pred: Union[Sequence, np.ndarray, pd.Series],
    zero_division: float = 0.0,
) -> float:
    """
    Compute classification Precision at a specific operating threshold: TP / (TP + FP).

    Motivation from docs/project_specification.md §5.2:
      Precision measures the fraction of flagged/declined applicants who would have
      actually defaulted. High precision minimizes "false alarms" that unnecessarily
      turn away creditworthy, revenue-generating borrowers.

    Args:
        y_true: Ground truth binary default labels.
        y_pred: Binary predicted decisions (1 = decline/flag, 0 = approve).
        zero_division: Value returned when denominator is zero (default 0.0).

    Returns:
        float: Precision in [0.0, 1.0].
    """
    yt, yp = _prepare_inputs(y_true, y_pred)
    return float(precision_score(yt, yp, zero_division=zero_division))


def compute_recall(
    y_true: Union[Sequence, np.ndarray, pd.Series],
    y_pred: Union[Sequence, np.ndarray, pd.Series],
    zero_division: float = 0.0,
) -> float:
    """
    Compute classification Recall (Sensitivity / Bad Capture Rate): TP / (TP + FN).

    Motivation from docs/project_specification.md §5.2:
      Recall measures the fraction of all defaulting loans that the scorecard successfully
      identifies. High recall directly prevents loan write-offs and bad debt losses.

    Args:
        y_true: Ground truth binary default labels.
        y_pred: Binary predicted decisions (1 = decline/flag, 0 = approve).
        zero_division: Value returned when denominator is zero (default 0.0).

    Returns:
        float: Recall in [0.0, 1.0].
    """
    yt, yp = _prepare_inputs(y_true, y_pred)
    return float(recall_score(yt, yp, zero_division=zero_division))


def compute_f1(
    y_true: Union[Sequence, np.ndarray, pd.Series],
    y_pred: Union[Sequence, np.ndarray, pd.Series],
    zero_division: float = 0.0,
) -> float:
    """
    Compute F1 score: 2 * (Precision * Recall) / (Precision + Recall).

    Motivation from docs/project_specification.md §5.2:
      F1 provides a balanced harmonic mean between bad debt prevention (recall)
      and business origination volume preservation (precision) at a chosen threshold.

    Args:
        y_true: Ground truth binary default labels.
        y_pred: Binary predicted decisions.
        zero_division: Value returned when denominator is zero (default 0.0).

    Returns:
        float: F1 score in [0.0, 1.0].
    """
    yt, yp = _prepare_inputs(y_true, y_pred)
    return float(f1_score(yt, yp, zero_division=zero_division))


def compute_confusion_matrix(
    y_true: Union[Sequence, np.ndarray, pd.Series],
    y_pred: Union[Sequence, np.ndarray, pd.Series],
) -> Dict[str, Any]:
    """
    Compute binary confusion matrix with explicit TP, FP, TN, FN counts.

    Motivation from docs/project_specification.md §5.2:
      Provides absolute borrower counts to translate statistical error rates into
      monetary business impact (e.g., charge-off loss vs. foregone interest).

    Args:
        y_true: Ground truth binary default labels.
        y_pred: Binary predicted decisions (0 or 1).

    Returns:
        dict with keys:
          - 'tn': True Negatives (approved good borrowers)
          - 'fp': False Positives (declined good borrowers)
          - 'fn': False Negatives (approved bad borrowers / defaults)
          - 'tp': True Positives (declined bad borrowers / defaults)
          - 'matrix': 2x2 numpy array [[tn, fp], [fn, tp]]
    """
    yt, yp = _prepare_inputs(y_true, y_pred)

    # Enforce that y_pred contains only 0 and 1
    if not set(np.unique(yp)).issubset({0, 1}):
        raise ValueError("y_pred must contain binary decisions (0 or 1) for confusion matrix.")

    cm = sk_confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "matrix": cm,
    }


# -----------------------------------------------------------------------------
# Comprehensive Evaluation Harness
# -----------------------------------------------------------------------------

def evaluate_predictions(
    y_true: Union[Sequence, np.ndarray, pd.Series],
    y_prob: Union[Sequence, np.ndarray, pd.Series],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Execute comprehensive scorecard evaluation across all primary and secondary metrics.

    Args:
        y_true: Ground truth binary labels (0/1).
        y_prob: Continuous predicted default probabilities in [0.0, 1.0].
        threshold: Operating decision threshold for converting probabilities to decisions.

    Returns:
        dict: Complete metrics report containing:
          - 'roc_auc': ROC Area Under Curve
          - 'gini': Gini coefficient (2*AUC - 1)
          - 'ks_stat': Kolmogorov-Smirnov statistic in [0.0, 1.0]
          - 'ks_percentage': Kolmogorov-Smirnov statistic in [0, 100]%
          - 'pr_auc': Precision-Recall Area Under Curve
          - 'threshold': Operating cutoff used for secondary metrics
          - 'precision': Precision at threshold
          - 'recall': Recall at threshold
          - 'f1': F1 score at threshold
          - 'confusion_matrix': Detailed TN, FP, FN, TP dictionary
    """
    yt, yp = _prepare_inputs(y_true, y_prob)

    # Continuous ranking metrics
    roc_auc = compute_roc_auc(yt, yp)
    gini = compute_gini(yt, yp)
    ks_stat = compute_ks_statistic(yt, yp, return_percentage=False)
    ks_pct = compute_ks_statistic(yt, yp, return_percentage=True)
    pr_auc = compute_pr_auc(yt, yp)

    # Discretize at operating threshold
    y_pred = (yp >= threshold).astype(int)

    precision = compute_precision(yt, y_pred)
    recall = compute_recall(yt, y_pred)
    f1 = compute_f1(yt, y_pred)
    cm = compute_confusion_matrix(yt, y_pred)

    return {
        "roc_auc": round(roc_auc, 6),
        "gini": round(gini, 6),
        "ks_stat": round(ks_stat, 6),
        "ks_percentage": round(ks_pct, 4),
        "pr_auc": round(pr_auc, 6),
        "threshold": float(threshold),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    np.random.seed(42)
    # Synthetic simulation of 1,000 credit applications
    y_t = np.random.binomial(1, 0.20, size=1000)
    # Model generates calibrated risk probabilities
    noise = np.random.normal(0, 0.25, size=1000)
    y_p = np.clip(0.1 + 0.5 * y_t + noise, 0.01, 0.99)

    report = evaluate_predictions(y_t, y_p, threshold=0.30)
    print("=" * 70)
    print("CREDIT RISK SCORECARD EVALUATION REPORT")
    print("=" * 70)
    print(f"ROC-AUC       : {report['roc_auc']:.4f}")
    print(f"Gini          : {report['gini']:.4f}")
    print(f"KS Statistic  : {report['ks_percentage']:.2f}% (stat = {report['ks_stat']:.4f})")
    print(f"PR-AUC        : {report['pr_auc']:.4f}")
    print("-" * 70)
    print(f"Threshold     : {report['threshold']:.2f}")
    print(f"Precision     : {report['precision']:.4f}")
    print(f"Recall        : {report['recall']:.4f}")
    print(f"F1 Score      : {report['f1']:.4f}")
    print(f"Confusion Mtx : TP={report['confusion_matrix']['tp']}, FP={report['confusion_matrix']['fp']}, "
          f"TN={report['confusion_matrix']['tn']}, FN={report['confusion_matrix']['fn']}")
    print("=" * 70)
