"""
Decision Threshold Optimization Module for Credit Risk Scorecards.
Performs:
  - Full threshold sweep across operating points with business KPIs
  - F1 score maximization (harmonic balance between precision and recall)
  - Youden's J statistic maximization (maximum KS vertical separation point: TPR - FPR)
  - Target recall optimization (guaranteeing minimum bad-debt capture rate)
  - Asymmetric business cost minimization (e.g. charge-off loss vs. customer acquisition cost)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from src.evaluation.metrics import _prepare_inputs


# -----------------------------------------------------------------------------
# Threshold Sweep Analysis
# -----------------------------------------------------------------------------

def sweep_thresholds(
    y_true: Union[Sequence, np.ndarray, pd.Series],
    y_prob: Union[Sequence, np.ndarray, pd.Series],
    thresholds: Optional[Sequence[float]] = None,
    n_thresholds: int = 100,
) -> pd.DataFrame:
    """
    Perform a comprehensive sweep across probability thresholds.
    Computes confusion matrix counts and business metrics at each cutoff point.

    Args:
        y_true: Ground truth binary default labels (0 = good, 1 = default).
        y_prob: Continuous predicted default probabilities in [0.0, 1.0].
        thresholds: Optional explicit list of thresholds to evaluate.
        n_thresholds: Number of evenly spaced threshold points if thresholds is None.

    Returns:
        pd.DataFrame: Table sorted by threshold containing:
          - threshold: operating cutoff
          - tp, fp, tn, fn: absolute borrower counts
          - precision, recall, f1: classification metrics
          - tpr, fpr, youden_j: ROC diagnostic curves
          - approval_rate: fraction of applicants approved (predicted 0)
          - default_rate_approved: actual default rate among approved loans
    """
    yt, yp = _prepare_inputs(y_true, y_prob)
    n_total = len(yt)
    n_pos = int(np.sum(yt == 1))
    n_neg = int(np.sum(yt == 0))

    if thresholds is None:
        thresh_arr = np.linspace(0.01, 0.99, n_thresholds)
    else:
        thresh_arr = np.sort(np.asarray(thresholds))

    records: List[Dict[str, Any]] = []

    for t in thresh_arr:
        t_val = float(t)
        # Class 1 = Flagged as Default / Declined
        # Class 0 = Approved
        y_pred = (yp >= t_val).astype(int)

        cm = sk_confusion_matrix(yt, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        youden_j = tpr - fpr

        # Business KPIs
        approved_count = tn + fn
        approval_rate = approved_count / n_total if n_total > 0 else 0.0
        default_rate_approved = fn / approved_count if approved_count > 0 else 0.0

        records.append({
            "threshold": round(t_val, 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "tpr": round(tpr, 6),
            "fpr": round(fpr, 6),
            "youden_j": round(youden_j, 6),
            "approval_rate": round(approval_rate, 6),
            "default_rate_approved": round(default_rate_approved, 6),
        })

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Optimization Objectives
# -----------------------------------------------------------------------------

def optimize_threshold_f1(
    y_true: Union[Sequence, np.ndarray, pd.Series],
    y_prob: Union[Sequence, np.ndarray, pd.Series],
    thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Find the threshold that maximizes the F1 score.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        thresholds: Optional search thresholds.

    Returns:
        dict: Best threshold and corresponding performance metrics.
    """
    df_sweep = sweep_thresholds(y_true, y_prob, thresholds=thresholds)
    best_row = df_sweep.loc[df_sweep["f1"].idxmax()]

    return {
        "objective": "max_f1",
        "best_threshold": float(best_row["threshold"]),
        "best_f1": float(best_row["f1"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "tp": int(best_row["tp"]),
        "fp": int(best_row["fp"]),
        "tn": int(best_row["tn"]),
        "fn": int(best_row["fn"]),
        "approval_rate": float(best_row["approval_rate"]),
    }


def optimize_threshold_youden(
    y_true: Union[Sequence, np.ndarray, pd.Series],
    y_prob: Union[Sequence, np.ndarray, pd.Series],
    thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Find the threshold that maximizes Youden's J statistic (J = TPR - FPR).
    In credit risk, this point corresponds to the cutoff that maximizes the
    Kolmogorov-Smirnov (KS) bad-capture divergence.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        thresholds: Optional search thresholds.

    Returns:
        dict: Best threshold, maximum Youden's J, and KS percentage.
    """
    df_sweep = sweep_thresholds(y_true, y_prob, thresholds=thresholds)
    best_row = df_sweep.loc[df_sweep["youden_j"].idxmax()]

    return {
        "objective": "max_youden_j_ks",
        "best_threshold": float(best_row["threshold"]),
        "max_youden_j": float(best_row["youden_j"]),
        "ks_percentage": round(float(best_row["youden_j"]) * 100.0, 4),
        "tpr": float(best_row["tpr"]),
        "fpr": float(best_row["fpr"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]),
        "approval_rate": float(best_row["approval_rate"]),
    }


def optimize_threshold_target_recall(
    y_true: Union[Sequence, np.ndarray, pd.Series],
    y_prob: Union[Sequence, np.ndarray, pd.Series],
    min_recall: float = 0.80,
    thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Find the highest threshold that satisfies a minimum bad-capture recall constraint,
    maximizing precision and loan origination volume while guaranteeing risk protection.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        min_recall: Required minimum recall / bad-capture fraction (e.g. 0.80 = 80%).
        thresholds: Optional search thresholds.

    Returns:
        dict: Optimal threshold meeting constraint.
    """
    df_sweep = sweep_thresholds(y_true, y_prob, thresholds=thresholds)
    qualifying = df_sweep[df_sweep["recall"] >= min_recall]

    if qualifying.empty:
        # Fallback to the point of maximum recall
        best_row = df_sweep.loc[df_sweep["recall"].idxmax()]
    else:
        # Pick highest threshold (maximizes precision and approvals while meeting recall)
        best_row = qualifying.loc[qualifying["threshold"].idxmax()]

    return {
        "objective": f"target_recall_{min_recall:.2f}",
        "best_threshold": float(best_row["threshold"]),
        "recall": float(best_row["recall"]),
        "precision": float(best_row["precision"]),
        "f1": float(best_row["f1"]),
        "min_recall_target": float(min_recall),
        "target_met": bool(best_row["recall"] >= min_recall),
        "approval_rate": float(best_row["approval_rate"]),
        "default_rate_approved": float(best_row["default_rate_approved"]),
    }


def optimize_threshold_cost(
    y_true: Union[Sequence, np.ndarray, pd.Series],
    y_prob: Union[Sequence, np.ndarray, pd.Series],
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Find the threshold that minimizes expected business cost under asymmetric losses:
      Cost(threshold) = cost_fn * FN(threshold) + cost_fp * FP(threshold)

    In credit underwriting:
      - False Negative (approving a bad loan): results in principal charge-off loss (~3-5x)
      - False Positive (rejecting a good loan): results in lost interest margin (~1x)

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        cost_fn: Monetary penalty for a False Negative (charge-off).
        cost_fp: Monetary penalty for a False Positive (lost customer margin).
        thresholds: Optional search thresholds.

    Returns:
        dict: Cost-optimal threshold and breakdown.
    """
    df_sweep = sweep_thresholds(y_true, y_prob, thresholds=thresholds)
    costs = (cost_fn * df_sweep["fn"]) + (cost_fp * df_sweep["fp"])
    df_sweep["total_cost"] = costs

    best_row = df_sweep.loc[df_sweep["total_cost"].idxmin()]

    return {
        "objective": "min_cost",
        "best_threshold": float(best_row["threshold"]),
        "min_total_cost": float(best_row["total_cost"]),
        "cost_fn_weight": float(cost_fn),
        "cost_fp_weight": float(cost_fp),
        "tp": int(best_row["tp"]),
        "fp": int(best_row["fp"]),
        "tn": int(best_row["tn"]),
        "fn": int(best_row["fn"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]),
        "approval_rate": float(best_row["approval_rate"]),
    }

