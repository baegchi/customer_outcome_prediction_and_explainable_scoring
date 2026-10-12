"""Evaluation module for credit risk scorecard assessment."""

from src.evaluation.metrics import (
    compute_confusion_matrix,
    compute_f1,
    compute_gini,
    compute_ks_statistic,
    compute_pr_auc,
    compute_precision,
    compute_recall,
    compute_roc_auc,
    evaluate_predictions,
)
from src.evaluation.threshold import (
    optimize_threshold_cost,
    optimize_threshold_f1,
    optimize_threshold_target_recall,
    optimize_threshold_youden,
    sweep_thresholds,
)

__all__ = [
    "compute_roc_auc",
    "compute_gini",
    "compute_ks_statistic",
    "compute_pr_auc",
    "compute_precision",
    "compute_recall",
    "compute_f1",
    "compute_confusion_matrix",
    "evaluate_predictions",
    "sweep_thresholds",
    "optimize_threshold_f1",
    "optimize_threshold_youden",
    "optimize_threshold_target_recall",
    "optimize_threshold_cost",
]
