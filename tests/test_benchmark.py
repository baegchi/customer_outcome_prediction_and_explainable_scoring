"""
Unit tests for tree model benchmark module (src/models/benchmark.py).
Verifies:
  - Estimator factory (Random Forest, XGBoost, LightGBM, Logistic Regression)
  - Unified experiment interface execution across models
  - Stratified K-Fold CV metrics evaluation
  - Structured experiment tracking in experiments/runs/ (config.yaml, metrics.json)
  - Consolidated benchmark summary ledger
  - Strict Out-Of-Time (OOT) data leakage rejection
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from src.models.benchmark import (
    DEFAULT_HYPERPARAMETERS,
    ModelConfig,
    build_estimator,
    run_benchmark_suite,
    run_experiment,
)


@pytest.fixture
def benchmark_synthetic_data():
    """Create a reproducible small synthetic dataset for fast benchmark testing."""
    np.random.seed(42)
    n = 300
    fico = np.random.normal(700, 30, size=n)
    dti = np.random.uniform(5, 35, size=n)
    delinq = np.random.binomial(1, 0.2, size=n)
    state = np.random.choice(["CA", "NY", "TX"], size=n)

    prob = 1.0 / (1.0 + np.exp(- (-1.8 + dti * 0.04 - (fico - 700) * 0.02 + delinq * 0.5)))
    y = np.random.binomial(1, prob, size=n)

    return pd.DataFrame({
        "fico": fico,
        "dti": dti,
        "has_delinq": delinq,
        "state": state,
        "issue_d": ["2016-03-01"] * n,  # Valid DEV vintage
        "target": y,
    })


# =============================================================================
# 1. Estimator Factory Tests
# =============================================================================

def test_estimator_factory_supported_models():
    """Verify build_estimator instantiates all 4 supported classifiers."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    import lightgbm as lgb

    lr = build_estimator("logistic_regression", {"C": 0.5})
    assert isinstance(lr, LogisticRegression)
    assert lr.C == 0.5

    rf = build_estimator("random_forest", {"n_estimators": 15, "max_depth": 4})
    assert isinstance(rf, RandomForestClassifier)
    assert rf.n_estimators == 15
    assert rf.max_depth == 4

    xgb_clf = build_estimator("xgboost", {"n_estimators": 15, "max_depth": 3})
    assert isinstance(xgb_clf, xgb.XGBClassifier)
    assert xgb_clf.n_estimators == 15
    assert xgb_clf.max_depth == 3

    lgb_clf = build_estimator("lightgbm", {"n_estimators": 15, "max_depth": 3})
    assert isinstance(lgb_clf, lgb.LGBMClassifier)
    assert lgb_clf.n_estimators == 15
    assert lgb_clf.max_depth == 3


def test_estimator_factory_unsupported_model():
    """Unsupported model type should raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_estimator("catboost_unsupported")


# =============================================================================
# 2. Individual Tree Model Experiment Runs
# =============================================================================

def test_run_experiment_random_forest(benchmark_synthetic_data, tmp_path):
    """Test Random Forest CV evaluation and artifact generation."""
    df = benchmark_synthetic_data
    cfg = ModelConfig(
        model_type="random_forest",
        hyperparameters={"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 5},
        n_splits=2,
        threshold=0.35,
        random_state=42,
    )

    exp_dir = tmp_path / "experiments"
    res = run_experiment(cfg, df, target_col="target", experiments_dir=exp_dir, verbose=False)

    # Validate output structure
    assert res["model_type"] == "random_forest"
    assert "mean_metrics" in res
    assert "std_metrics" in res
    assert "oof_metrics" in res
    assert len(res["folds"]) == 2

    # Check that metrics are within bounds
    assert 0.5 <= res["mean_metrics"]["roc_auc"] <= 1.0
    assert 0.0 <= res["mean_metrics"]["ks_percentage"] <= 100.0
    assert res["total_train_time_sec"] > 0.0

    # Validate saved files
    run_path = Path(res["run_directory"])
    assert (run_path / "config.yaml").exists()
    assert (run_path / "metrics.json").exists()


def test_run_experiment_xgboost(benchmark_synthetic_data, tmp_path):
    """Test XGBoost CV evaluation through common interface."""
    df = benchmark_synthetic_data
    cfg = ModelConfig(
        model_type="xgboost",
        hyperparameters={"n_estimators": 10, "max_depth": 2},
        n_splits=2,
        threshold=0.30,
        random_state=42,
    )

    exp_dir = tmp_path / "experiments"
    res = run_experiment(cfg, df, target_col="target", experiments_dir=exp_dir, verbose=False)

    assert res["model_type"] == "xgboost"
    assert "roc_auc" in res["mean_metrics"]
    assert "gini" in res["mean_metrics"]
    assert "ks_percentage" in res["mean_metrics"]
    assert "pr_auc" in res["mean_metrics"]
    assert res["total_train_time_sec"] > 0.0


def test_run_experiment_lightgbm(benchmark_synthetic_data, tmp_path):
    """Test LightGBM CV evaluation through common interface."""
    df = benchmark_synthetic_data
    cfg = ModelConfig(
        model_type="lightgbm",
        hyperparameters={"n_estimators": 10, "max_depth": 2, "num_leaves": 7},
        n_splits=2,
        threshold=0.30,
        random_state=42,
    )

    exp_dir = tmp_path / "experiments"
    res = run_experiment(cfg, df, target_col="target", experiments_dir=exp_dir, verbose=False)

    assert res["model_type"] == "lightgbm"
    assert "roc_auc" in res["mean_metrics"]
    assert "confusion_matrix" in res["oof_metrics"]
    assert res["total_train_time_sec"] > 0.0


def test_run_experiment_numeric_feature_mode(benchmark_synthetic_data, tmp_path):
    """Test execution under numeric feature mode (without WoE transformation)."""
    df = benchmark_synthetic_data
    cfg = ModelConfig(
        model_type="logistic_regression",
        feature_mode="numeric",
        n_splits=2,
        random_state=42,
    )

    exp_dir = tmp_path / "experiments"
    res = run_experiment(cfg, df, target_col="target", experiments_dir=exp_dir, verbose=False)

    assert res["config"]["feature_mode"] == "numeric"
    assert res["mean_metrics"]["roc_auc"] >= 0.5


# =============================================================================
# 3. Defensive OOT Protection Guard
# =============================================================================

def test_benchmark_oot_rejection():
    """Verify benchmark run_experiment raises ValueError on OOT dates (2017+)."""
    df_oot = pd.DataFrame({
        "fico": [700, 710, 680],
        "dti": [20, 15, 25],
        "issue_d": ["2016-12-01", "2017-01-01", "2017-05-01"],
        "target": [0, 1, 0],
    })
    cfg = ModelConfig(model_type="random_forest", n_splits=2)

    with pytest.raises(ValueError, match="OOT data leakage detected"):
        run_experiment(cfg, df_oot)


# =============================================================================
# 4. Multi-Model Benchmark Suite & Summary Ledger
# =============================================================================

def test_run_benchmark_suite_and_summary_ledger(benchmark_synthetic_data, tmp_path):
    """Verify run_benchmark_suite runs multiple models and populates benchmark_summary.json."""
    df = benchmark_synthetic_data
    exp_dir = tmp_path / "experiments"

    models_to_test = ["logistic_regression", "random_forest"]
    summary_df = run_benchmark_suite(
        df=df,
        target_col="target",
        models=models_to_test,
        n_splits=2,
        threshold=0.30,
        experiments_dir=exp_dir,
        verbose=False,
    )

    # Leaderboard DataFrame assertions
    assert len(summary_df) == 2
    assert set(summary_df["Model"].values) == set(models_to_test)
    assert "ROC-AUC" in summary_df.columns
    assert "Gini" in summary_df.columns
    assert "KS (%)" in summary_df.columns
    assert "PR-AUC" in summary_df.columns
    assert "Train Time (s)" in summary_df.columns

    # Verify summary ledger JSON
    summary_json_path = exp_dir / "benchmark_summary.json"
    assert summary_json_path.exists()

    with open(summary_json_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    assert len(ledger) == 2
    model_types_in_ledger = [item["model_type"] for item in ledger]
    assert "logistic_regression" in model_types_in_ledger
    assert "random_forest" in model_types_in_ledger

