"""
Unit tests for Optuna hyperparameter optimization (src/models/tune.py).
Verifies:
  - Domain-tuned search space sampling for LightGBM, XGBoost, and Random Forest
  - Optuna study optimization with Stratified K-Fold CV inside objective
  - Artifact export: optuna_best_config.yaml, optuna_trials.csv, optuna_study.db
  - Strict Out-Of-Time (OOT) data leakage rejection
  - Metric dispatcher and config serialization
"""

from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import pytest
import yaml

from src.models.tune import (
    TuningConfig,
    _compute_target_metric,
    define_search_space,
    run_tuning,
)


@pytest.fixture
def tuning_synthetic_data():
    """Create reproducible small synthetic dataset for fast tuning tests."""
    np.random.seed(42)
    n = 250
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
        "issue_d": ["2016-04-01"] * n,  # Valid DEV vintage
        "target": y,
    })


# =============================================================================
# 1. Search Space Sampling Tests
# =============================================================================

def test_define_search_space_lightgbm():
    """Verify LightGBM search space yields expected keys."""
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = define_search_space(trial, "lightgbm")

    expected_keys = {
        "n_estimators",
        "max_depth",
        "num_leaves",
        "learning_rate",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "verbose",
        "n_jobs",
    }
    assert expected_keys.issubset(params.keys())
    assert 50 <= params["n_estimators"] <= 250
    assert 3 <= params["max_depth"] <= 8
    assert 15 <= params["num_leaves"] <= 63


def test_define_search_space_xgboost():
    """Verify XGBoost search space yields expected keys."""
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = define_search_space(trial, "xgboost")

    expected_keys = {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "eval_metric",
        "n_jobs",
    }
    assert expected_keys.issubset(params.keys())


def test_define_search_space_unsupported():
    """Unsupported model type must raise ValueError."""
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    with pytest.raises(ValueError, match="Unsupported model_type"):
        define_search_space(trial, "invalid_model_type")


# =============================================================================
# 2. Metric Dispatcher & Config Serialization
# =============================================================================

def test_compute_target_metric():
    """Test metric computation helper."""
    yt = np.array([0, 0, 1, 1])
    yp = np.array([0.1, 0.2, 0.8, 0.9])

    assert _compute_target_metric(yt, yp, "roc_auc") == pytest.approx(1.0)
    assert _compute_target_metric(yt, yp, "gini") == pytest.approx(1.0)
    assert _compute_target_metric(yt, yp, "pr_auc") == pytest.approx(1.0)
    assert _compute_target_metric(yt, yp, "ks") == pytest.approx(100.0)

    with pytest.raises(ValueError, match="Unknown metric"):
        _compute_target_metric(yt, yp, "unknown_metric")


def test_tuning_config_serialization():
    """Verify roundtrip dictionary serialization of TuningConfig."""
    cfg = TuningConfig(model_type="xgboost", n_trials=10, seed=123)
    d = cfg.to_dict()
    assert d["model_type"] == "xgboost"
    assert d["n_trials"] == 10
    assert d["seed"] == 123

    restored = TuningConfig.from_dict(d)
    assert restored.model_type == "xgboost"
    assert restored.seed == 123


# =============================================================================
# 3. End-to-End Tuning Execution & Artifacts
# =============================================================================

def test_run_tuning_lightgbm(tuning_synthetic_data, tmp_path):
    """Test LightGBM tuning study execution, CV objective, and artifact generation."""
    df = tuning_synthetic_data
    db_path = tmp_path / "optuna_test.db"
    cfg = TuningConfig(
        model_type="lightgbm",
        objective_metric="roc_auc",
        n_trials=3,
        n_splits=2,
        seed=42,
        storage_url=f"sqlite:///{db_path}",
    )

    results = run_tuning(
        config=cfg,
        df=df,
        target_col="target",
        output_dir=tmp_path,
        verbose=False,
    )

    assert results["status"] == "success"
    assert results["model_type"] == "lightgbm"
    assert results["n_trials_completed"] == 3
    assert 0.5 <= results["best_score"] <= 1.0
    assert len(results["best_params"]) > 0

    # Verify artifacts exist on disk
    assert (tmp_path / "optuna_best_config_lightgbm.yaml").exists()
    assert (tmp_path / "optuna_best_config_lightgbm.json").exists()
    assert (tmp_path / "optuna_trials_lightgbm.csv").exists()
    assert (tmp_path / "optuna_study_summary_lightgbm.json").exists()
    assert db_path.exists()

    # Check YAML contents
    with open(tmp_path / "optuna_best_config_lightgbm.yaml", "r", encoding="utf-8") as f:
        loaded_yaml = yaml.safe_load(f)
    assert loaded_yaml["model_type"] == "lightgbm"
    assert loaded_yaml["total_trials"] == 3
    assert "best_hyperparameters" in loaded_yaml


def test_run_tuning_xgboost(tuning_synthetic_data, tmp_path):
    """Test XGBoost tuning study execution."""
    df = tuning_synthetic_data
    cfg = TuningConfig(
        model_type="xgboost",
        objective_metric="roc_auc",
        n_trials=2,
        n_splits=2,
        seed=42,
        storage_url=f"sqlite:///{tmp_path / 'xgb_test.db'}",
    )

    results = run_tuning(
        config=cfg,
        df=df,
        target_col="target",
        output_dir=tmp_path,
        verbose=False,
    )

    assert results["status"] == "success"
    assert results["model_type"] == "xgboost"
    assert results["n_trials_completed"] == 2
    assert (tmp_path / "optuna_best_config_xgboost.yaml").exists()


# =============================================================================
# 4. Out-Of-Time (OOT) Leakage Protection Guards
# =============================================================================

def test_tuning_oot_rejection():
    """Verify tuning raises ValueError when OOT records (2017+) are present."""
    df_oot = pd.DataFrame({
        "fico": [710, 690, 720],
        "dti": [20, 15, 25],
        "issue_d": ["2016-12-01", "2017-01-01", "2017-08-01"],
        "target": [0, 1, 0],
    })
    cfg = TuningConfig(model_type="lightgbm", n_trials=2)

    with pytest.raises(ValueError, match="OOT data leakage detected"):
        run_tuning(config=cfg, df=df_oot)

