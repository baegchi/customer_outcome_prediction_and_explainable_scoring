"""
SHAP Explainability Module for Credit Risk Models.
Generates:
  - Global SHAP beeswarm summary plot
  - Global mean |SHAP| feature importance ranking table and bar plot
  - Local SHAP explanations (top positive and negative contributors) for selected OOT predictions
  - Structured JSON/CSV data records and image artifacts under reports/
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure custom packages on /DATA partition are discoverable
extra_pkgs = Path("/DATA/suraj/pip_pkgs")
if extra_pkgs.exists() and str(extra_pkgs) not in sys.path:
    sys.path.insert(0, str(extra_pkgs))

import matplotlib
matplotlib.use("Agg")  # Headless backend for artifact generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.features.woe import transform_woe
from src.models.baseline import BaselineScorecardPipeline

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Explainer Selection & SHAP Extraction Helpers
# -----------------------------------------------------------------------------

def get_explainer(
    model: Any,
    X_background: Optional[pd.DataFrame] = None,
) -> shap.Explainer:
    """
    Select and initialize the appropriate SHAP Explainer for a given model.

    Args:
        model: Trained classifier (or BaselineScorecardPipeline).
        X_background: Optional background dataset for linear/kernel explainers.

    Returns:
        shap.Explainer: Initialized SHAP explainer instance.
    """
    # If wrapped BaselineScorecardPipeline, unwrap underlying estimator
    estimator = model.model_ if isinstance(model, BaselineScorecardPipeline) else model

    # Check for tree-based models
    estimator_name = estimator.__class__.__name__.lower()
    is_tree = any(k in estimator_name for k in ["lgbm", "xgb", "forest", "tree", "gradientboosting"])

    if is_tree:
        try:
            return shap.TreeExplainer(estimator)
        except Exception:
            pass

    # Linear or generic explainer fallback
    if X_background is not None:
        bg = X_background.values if isinstance(X_background, pd.DataFrame) else X_background
        try:
            return shap.LinearExplainer(estimator, bg)
        except Exception:
            return shap.Explainer(estimator.predict_proba, bg)

    return shap.Explainer(estimator)


def _prepare_feature_data(model: Any, X: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
    """Prepare feature matrix and extract underlying estimator if pipeline."""
    if isinstance(model, BaselineScorecardPipeline):
        if model.woe_bins_ is None or model.selected_features_ is None or model.model_ is None:
            raise RuntimeError("BaselineScorecardPipeline must be fitted before computing SHAP.")
        # Transform X through pipeline's WoE bins
        X_woe = transform_woe(X, model.woe_bins_, selected_only=False)
        X_proc = X_woe[model.selected_features_]
        estimator = model.model_
        return X_proc, estimator

    return X, model


def _extract_positive_class_shap(shap_values: Any) -> np.ndarray:
    """
    Extract 2D array of SHAP values for the positive default class (class 1).
    Handles differences across SHAP versions and tree libraries.
    """
    if hasattr(shap_values, "values"):
        vals = shap_values.values
    else:
        vals = shap_values

    if isinstance(vals, list):
        # List of [class_0, class_1]
        arr = vals[1] if len(vals) > 1 else vals[0]
        return np.asarray(arr, dtype=np.float64)

    arr = np.asarray(vals, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr[:, :, 1]
    return arr


def _extract_base_value(explainer: Any) -> float:
    """Extract baseline expected value for positive class."""
    ev = getattr(explainer, "expected_value", 0.0)
    if isinstance(ev, (list, np.ndarray)):
        return float(ev[1] if len(ev) > 1 else ev[0])
    try:
        return float(ev)
    except (TypeError, ValueError):
        return 0.0


# -----------------------------------------------------------------------------
# Global SHAP Analysis & Visualizations
# -----------------------------------------------------------------------------

def compute_global_shap(
    model: Any,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    output_dir: Union[str, Path] = "reports",
    max_display: int = 15,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute global SHAP values, ranking table, and summary visualization artifacts.

    Args:
        model: Trained model or BaselineScorecardPipeline.
        X: Feature dataset (OOT or DEV partition).
        feature_names: Optional list of feature names.
        output_dir: Directory to save generated figures and tables.
        max_display: Maximum number of features to display in plots.
        verbose: Verbosity flag.

    Returns:
        dict: Summary containing feature ranking DataFrame and artifact paths.
    """
    out_path = Path(output_dir)
    fig_path = out_path / "figures"
    fig_path.mkdir(parents=True, exist_ok=True)

    X_proc, estimator = _prepare_feature_data(model, X)
    feats = feature_names or list(X_proc.columns)

    explainer = get_explainer(estimator, X_background=X_proc)
    raw_shap = explainer.shap_values(X_proc.values) if hasattr(explainer, "shap_values") else explainer(X_proc.values)
    shap_matrix = _extract_positive_class_shap(raw_shap)

    if shap_matrix.shape[1] != len(feats):
        raise ValueError(
            f"SHAP matrix shape {shap_matrix.shape} does not match feature count {len(feats)}."
        )

    # 1. Global Mean Absolute SHAP Ranking Table
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    df_importance = (
        pd.DataFrame({
            "feature": feats,
            "mean_abs_shap": np.round(mean_abs_shap, 6),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # Save tables
    csv_path = out_path / "shap_global_importance.csv"
    json_path = out_path / "shap_global_importance.json"
    df_importance.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df_importance.to_dict(orient="records"), f, indent=2)

    # 2. Mean |SHAP| Bar Chart
    top_df = df_importance.head(max_display).iloc[::-1]  # Bottom to top for horizontal bar
    fig, ax = plt.subplots(figsize=(10, max(5, int(len(top_df) * 0.4))), dpi=150)
    bars = ax.barh(top_df["feature"], top_df["mean_abs_shap"], color="#1f77b4", edgecolor="none", height=0.65)
    ax.set_title("Global Feature Importance — Mean |SHAP Value|", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Log-Odds of Default)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Add numeric labels to bars
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", ha="left", fontsize=9)

    plt.tight_layout()
    bar_plot_path = fig_path / "shap_mean_importance_bar.png"
    plt.savefig(bar_plot_path)
    plt.close(fig)

    # 3. Global Beeswarm / Summary Plot
    fig_summary = plt.figure(figsize=(10, max(5, int(min(len(feats), max_display) * 0.45))), dpi=150)
    try:
        shap.summary_plot(
            shap_matrix,
            X_proc,
            feature_names=feats,
            max_display=max_display,
            show=False,
        )
        plt.title("Global SHAP Summary Plot (Feature Impact on Default Risk)", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        beeswarm_plot_path = fig_path / "shap_summary_beeswarm.png"
        plt.savefig(beeswarm_plot_path)
    except Exception as e:
        logger.warning("Falling back to bar plot for summary due to: %s", e)
        beeswarm_plot_path = bar_plot_path
    finally:
        plt.close("all")

    if verbose:
        print(f"Global SHAP feature importance saved to: {csv_path}")
        print(f"Mean |SHAP| bar plot saved to: {bar_plot_path}")
        print(f"Summary beeswarm plot saved to: {beeswarm_plot_path}")

    return {
        "importance_df": df_importance,
        "importance_csv": str(csv_path),
        "importance_json": str(json_path),
        "bar_plot_path": str(bar_plot_path),
        "beeswarm_plot_path": str(beeswarm_plot_path),
        "shap_matrix": shap_matrix,
        "X_proc": X_proc,
    }


# -----------------------------------------------------------------------------
# Local SHAP Explanations for Individual Predictions
# -----------------------------------------------------------------------------

def explain_local_predictions(
    model: Any,
    X: pd.DataFrame,
    sample_indices: Optional[List[int]] = None,
    n_samples: int = 5,
    top_k: int = 5,
    feature_names: Optional[List[str]] = None,
    output_dir: Union[str, Path] = "reports",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate local SHAP explanations for selected individual OOT predictions.
    Identifies top positive risk contributors (pushing towards default)
    and top negative mitigating contributors (pushing towards non-default).

    Args:
        model: Trained model or BaselineScorecardPipeline.
        X: Feature dataset.
        sample_indices: Optional list of row indices to explain. If None, selects
                        a diverse subset: high-risk, low-risk, and borderline cases.
        n_samples: Number of samples to explain if sample_indices is None.
        top_k: Number of top positive/negative features to report.
        feature_names: Feature names list.
        output_dir: Output reports directory.
        verbose: Verbosity flag.

    Returns:
        list of dict: Structured explanation records for each selected sample.
    """
    out_path = Path(output_dir)
    fig_path = out_path / "figures"
    fig_path.mkdir(parents=True, exist_ok=True)

    X_proc, estimator = _prepare_feature_data(model, X)
    feats = feature_names or list(X_proc.columns)

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X if isinstance(model, BaselineScorecardPipeline) else X_proc)[:, 1]
    else:
        probs = estimator.predict_proba(X_proc)[:, 1]

    # Select representative sample indices if not provided
    if sample_indices is None:
        sorted_indices = np.argsort(probs)
        # Select highest risk (defaults), lowest risk (approved), and borderline (~0.5)
        n_sel = min(n_samples, len(X))
        high_idx = sorted_indices[-max(1, n_sel // 2):].tolist()
        low_idx = sorted_indices[:max(1, n_sel // 2)].tolist()
        mid_idx = sorted_indices[len(sorted_indices) // 2: len(sorted_indices) // 2 + 1].tolist()
        chosen_indices = list(dict.fromkeys(high_idx + mid_idx + low_idx))[:n_sel]
    else:
        chosen_indices = [idx for idx in sample_indices if idx < len(X)]

    explainer = get_explainer(estimator, X_background=X_proc)
    X_subset = X_proc.iloc[chosen_indices]
    raw_shap = explainer.shap_values(X_subset.values) if hasattr(explainer, "shap_values") else explainer(X_subset.values)
    shap_subset = _extract_positive_class_shap(raw_shap)
    base_val = _extract_base_value(explainer)

    explanations: List[Dict[str, Any]] = []

    for i, orig_idx in enumerate(chosen_indices):
        shap_row = shap_subset[i]
        orig_row = X.iloc[orig_idx] if hasattr(X, "iloc") else X_proc.iloc[orig_idx]
        prob_val = float(probs[orig_idx])

        # Combine feature name, original value, and SHAP value
        feat_contribs = []
        for feat_name, shap_v in zip(feats, shap_row):
            orig_v = orig_row[feat_name] if feat_name in orig_row else X_proc.iloc[orig_idx][feat_name]
            feat_contribs.append({
                "feature": feat_name,
                "original_value": (
                    round(float(orig_v), 4)
                    if isinstance(orig_v, (int, float, np.number)) and pd.notna(orig_v)
                    else str(orig_v)
                ),
                "shap_value": round(float(shap_v), 6),
            })

        # Separate positive (risk increasers) and negative (risk decreasers)
        pos_contribs = sorted(
            [c for c in feat_contribs if c["shap_value"] > 0],
            key=lambda x: x["shap_value"],
            reverse=True,
        )[:top_k]

        neg_contribs = sorted(
            [c for c in feat_contribs if c["shap_value"] < 0],
            key=lambda x: x["shap_value"],
        )[:top_k]

        # Top overall contributors for visualization
        top_local = sorted(feat_contribs, key=lambda x: abs(x["shap_value"]), reverse=True)[:top_k * 2]

        # Render local explanation bar plot
        fig, ax = plt.subplots(figsize=(9, max(4, int(len(top_local) * 0.4))), dpi=150)
        local_df = pd.DataFrame(top_local).iloc[::-1]
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in local_df["shap_value"]]

        labels = [f"{r['feature']} = {r['original_value']}" for _, r in local_df.iterrows()]
        bars = ax.barh(labels, local_df["shap_value"], color=colors, height=0.6)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(
            f"Local Explanation: Sample #{orig_idx} (P(Default) = {prob_val:.2%})",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("SHAP Value (Red = Higher Risk, Green = Lower Risk)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.6)

        plt.tight_layout()
        local_fig_path = fig_path / f"shap_local_sample_{orig_idx}.png"
        plt.savefig(local_fig_path)
        plt.close(fig)

        record = {
            "sample_index": int(orig_idx),
            "predicted_default_prob": round(prob_val, 6),
            "base_value": round(base_val, 6),
            "top_positive_contributors": pos_contribs,
            "top_negative_contributors": neg_contribs,
            "plot_path": str(local_fig_path),
        }
        explanations.append(record)

    # Save complete local explanations to JSON
    local_json_path = out_path / "shap_local_explanations.json"
    with open(local_json_path, "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2)

    if verbose:
        print(f"Generated {len(explanations)} local SHAP explanations into: {local_json_path}")

    return explanations


# -----------------------------------------------------------------------------
# End-to-End SHAP Report Workflow
# -----------------------------------------------------------------------------

def generate_shap_report(
    model: Any,
    X: pd.DataFrame,
    sample_indices: Optional[List[int]] = None,
    n_samples: int = 5,
    top_k: int = 5,
    output_dir: Union[str, Path] = "reports",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Execute full SHAP explainability workflow:
      1. Global SHAP computation & summary plots
      2. Mean |SHAP| feature ranking table and bar plot
      3. Local explanations for selected individual predictions
      4. Saves all image artifacts into reports/figures/ and data to reports/

    Args:
        model: Trained model instance or BaselineScorecardPipeline.
        X: Evaluation dataset (e.g. OOT test set).
        sample_indices: Optional list of sample indices for local explanations.
        n_samples: Number of representative samples to explain if sample_indices is None.
        top_k: Top K contributors per local explanation.
        output_dir: Output directory (default 'reports').
        verbose: Verbosity flag.

    Returns:
        dict: Complete dictionary of generated artifact file paths.
    """
    if verbose:
        print("\n" + "=" * 78)
        print("GENERATING SHAP EXPLAINABILITY REPORT")
        print("=" * 78)

    # 1. Global Explanations
    global_res = compute_global_shap(
        model=model,
        X=X,
        output_dir=output_dir,
        verbose=verbose,
    )

    # 2. Local Explanations
    local_exps = explain_local_predictions(
        model=model,
        X=X,
        sample_indices=sample_indices,
        n_samples=n_samples,
        top_k=top_k,
        output_dir=output_dir,
        verbose=verbose,
    )

    out_path = Path(output_dir)
    report_manifest = {
        "status": "success",
        "global_importance_csv": global_res["importance_csv"],
        "global_importance_json": global_res["importance_json"],
        "global_bar_plot": global_res["bar_plot_path"],
        "global_beeswarm_plot": global_res["beeswarm_plot_path"],
        "local_explanations_json": str(out_path / "shap_local_explanations.json"),
        "local_plots": [e["plot_path"] for e in local_exps],
        "n_local_samples": len(local_exps),
    }

    if verbose:
        print("=" * 78)
        print("SHAP EXPLAINABILITY ARTIFACTS GENERATED SUCCESSFULLY")
        print(f"  Global Importance Table : {report_manifest['global_importance_csv']}")
        print(f"  Global Bar Chart        : {report_manifest['global_bar_plot']}")
        print(f"  Global Beeswarm Plot    : {report_manifest['global_beeswarm_plot']}")
        print(f"  Local Explanations JSON : {report_manifest['local_explanations_json']}")
        print(f"  Local Plot Count        : {len(report_manifest['local_plots'])}")
        print("=" * 78 + "\n")

    return report_manifest


if __name__ == "__main__":
    np.random.seed(42)
    # Synthetic simulation of 500 OOT credit applications
    n = 500
    fico = np.random.normal(700, 35, size=n)
    dti = np.random.uniform(5, 38, size=n)
    delinq = np.random.binomial(1, 0.20, size=n)
    inq_last_6mths = np.random.poisson(1.0, size=n)
    annual_inc = np.random.lognormal(11.0, 0.5, size=n)

    prob = 1.0 / (1.0 + np.exp(- (-2.0 + dti * 0.04 - (fico - 700) * 0.025 + delinq * 0.6 + inq_last_6mths * 0.2)))
    y_oot = np.random.binomial(1, prob, size=n)

    df_oot = pd.DataFrame({
        "fico": fico,
        "dti": dti,
        "has_delinq": delinq,
        "inq_last_6mths": inq_last_6mths,
        "annual_inc": annual_inc,
        "issue_d": ["2017-06-01"] * n,  # Valid OOT vintage
        "target": y_oot,
    })

    # Train a quick LightGBM model on DEV to explain on OOT
    import lightgbm as lgb
    X_features = df_oot.drop(columns=["target", "issue_d"])
    clf = lgb.LGBMClassifier(n_estimators=50, max_depth=4, random_state=42, verbose=-1)
    clf.fit(X_features, y_oot)

    manifest = generate_shap_report(
        model=clf,
        X=X_features,
        n_samples=3,
        output_dir="reports",
        verbose=True,
    )

