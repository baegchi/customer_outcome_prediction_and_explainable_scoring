"""
Unit tests for SHAP explainability module (src/explainability/shap_report.py).
Verifies:
  - Explainer factory selection (TreeExplainer and LinearExplainer)
  - Global SHAP computation, ranking table, and plots (beeswarm and bar)
  - Local SHAP explanations (top positive/negative contributors per sample)
  - Compatibility with BaselineScorecardPipeline and tree models (LightGBM, XGBoost)
  - End-to-end generate_shap_report manifest and artifact generation
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

from src.explainability.shap_report import (
    compute_global_shap,
    explain_local_predictions,
    generate_shap_report,
    get_explainer,
)
from src.models.baseline import BaselineScorecardPipeline


@pytest.fixture
def explain_synthetic_data():
    """Create reproducible synthetic dataset for explainability tests."""
    np.random.seed(42)
    n = 200
    fico = np.random.normal(700, 30, size=n)
    dti = np.random.uniform(5, 35, size=n)
    delinq = np.random.binomial(1, 0.2, size=n)
    inq = np.random.poisson(1.0, size=n)

    prob = 1.0 / (1.0 + np.exp(- (-1.8 + dti * 0.04 - (fico - 700) * 0.025 + delinq * 0.6)))
    y = np.random.binomial(1, prob, size=n)

    df = pd.DataFrame({
        "fico": fico,
        "dti": dti,
        "has_delinq": delinq,
        "inq_last_6mths": inq,
        "target": y,
    })
    return df


# =============================================================================
# 1. Explainer Factory Tests
# =============================================================================

def test_get_explainer_types(explain_synthetic_data):
    """Verify get_explainer returns appropriate explainer per model type."""
    df = explain_synthetic_data
    X = df.drop(columns=["target"])
    y = df["target"]

    # Tree models
    lgb_model = lgb.LGBMClassifier(n_estimators=10, max_depth=3, random_state=42, verbose=-1)
    lgb_model.fit(X, y)
    exp_lgb = get_explainer(lgb_model)
    assert "tree" in exp_lgb.__class__.__name__.lower()

    # Linear models
    lr_model = LogisticRegression()
    lr_model.fit(X, y)
    exp_lr = get_explainer(lr_model, X_background=X)
    assert exp_lr is not None


# =============================================================================
# 2. Global SHAP Artifacts Tests
# =============================================================================

def test_compute_global_shap(explain_synthetic_data, tmp_path):
    """Verify global SHAP ranking table, bar plot, and beeswarm generation."""
    df = explain_synthetic_data
    X = df.drop(columns=["target"])
    y = df["target"]

    model = lgb.LGBMClassifier(n_estimators=15, max_depth=3, random_state=42, verbose=-1)
    model.fit(X, y)

    res = compute_global_shap(
        model=model,
        X=X,
        output_dir=tmp_path,
        max_display=5,
        verbose=False,
    )

    # Validate output structure
    assert "importance_df" in res
    assert len(res["importance_df"]) == X.shape[1]
    assert set(res["importance_df"]["feature"].values) == set(X.columns)

    # Check that CSV and JSON exist and are valid
    csv_path = Path(res["importance_csv"])
    json_path = Path(res["importance_json"])
    assert csv_path.exists()
    assert json_path.exists()

    csv_df = pd.read_csv(csv_path)
    assert len(csv_df) == X.shape[1]
    assert "mean_abs_shap" in csv_df.columns

    # Check images exist and have non-zero file size
    bar_plot = Path(res["bar_plot_path"])
    beeswarm_plot = Path(res["beeswarm_plot_path"])
    assert bar_plot.exists() and bar_plot.stat().st_size > 0
    assert beeswarm_plot.exists() and beeswarm_plot.stat().st_size > 0


# =============================================================================
# 3. Local SHAP Explanations Tests
# =============================================================================

def test_explain_local_predictions(explain_synthetic_data, tmp_path):
    """Verify local explanations for individual predictions."""
    df = explain_synthetic_data
    X = df.drop(columns=["target"])
    y = df["target"]

    model = lgb.LGBMClassifier(n_estimators=15, max_depth=3, random_state=42, verbose=-1)
    model.fit(X, y)

    sample_indices = [5, 25, 50]
    exps = explain_local_predictions(
        model=model,
        X=X,
        sample_indices=sample_indices,
        top_k=2,
        output_dir=tmp_path,
        verbose=False,
    )

    assert len(exps) == 3
    for rec in exps:
        assert rec["sample_index"] in sample_indices
        assert 0.0 <= rec["predicted_default_prob"] <= 1.0
        assert "top_positive_contributors" in rec
        assert "top_negative_contributors" in rec
        assert len(rec["top_positive_contributors"]) <= 2
        assert len(rec["top_negative_contributors"]) <= 2

        # Check local sample plot was created
        plot_path = Path(rec["plot_path"])
        assert plot_path.exists() and plot_path.stat().st_size > 0

    # Check JSON export
    local_json = tmp_path / "shap_local_explanations.json"
    assert local_json.exists()
    with open(local_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded) == 3


# =============================================================================
# 4. Baseline Pipeline Compatibility
# =============================================================================

def test_baseline_pipeline_shap_compatibility(explain_synthetic_data, tmp_path):
    """Verify SHAP explanation on a BaselineScorecardPipeline with WoE features."""
    df = explain_synthetic_data
    X = df.drop(columns=["target"])
    y = df["target"]

    pipeline = BaselineScorecardPipeline(max_bins=5, random_state=42)
    pipeline.fit(X, y)

    res = generate_shap_report(
        model=pipeline,
        X=X,
        n_samples=2,
        output_dir=tmp_path,
        verbose=False,
    )

    assert res["status"] == "success"
    assert Path(res["global_bar_plot"]).exists()
    assert Path(res["local_explanations_json"]).exists()
    assert len(res["local_plots"]) == 2


# =============================================================================
# 5. End-to-End Workflow Test
# =============================================================================

def test_generate_shap_report_end_to_end(explain_synthetic_data, tmp_path):
    """Test full generate_shap_report manifest structure."""
    df = explain_synthetic_data
    X = df.drop(columns=["target"])
    y = df["target"]

    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)

    manifest = generate_shap_report(
        model=model,
        X=X,
        n_samples=3,
        top_k=3,
        output_dir=tmp_path,
        verbose=False,
    )

    expected_keys = {
        "status",
        "global_importance_csv",
        "global_importance_json",
        "global_bar_plot",
        "global_beeswarm_plot",
        "local_explanations_json",
        "local_plots",
        "n_local_samples",
    }
    assert expected_keys.issubset(manifest.keys())
    assert manifest["status"] == "success"
    assert manifest["n_local_samples"] == 3
    assert len(manifest["local_plots"]) == 3

