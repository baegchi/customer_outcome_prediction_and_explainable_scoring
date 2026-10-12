"""Explainability package for credit risk scorecard and model interpretation."""

from src.explainability.shap_report import (
    compute_global_shap,
    explain_local_predictions,
    generate_shap_report,
    get_explainer,
)

__all__ = [
    "compute_global_shap",
    "explain_local_predictions",
    "generate_shap_report",
    "get_explainer",
]

