"""
Unit tests for baseline scorecard pipeline (src/models/baseline.py).
Verifies:
  - BaselineScorecardPipeline fitting, probability prediction, and threshold classification
  - Stratified K-Fold CV evaluation with WoE refitted per fold
  - Comprehensive metrics computation using src/evaluation/metrics.py
  - Training time and inference time measurement
  - Pipeline serialization roundtrip (save & load) and prediction equivalence
  - Strict Out-Of-Time (OOT) data leakage prevention guards
  - End-to-end run_baseline workflow
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.baseline import (
    BaselineScorecardPipeline,
    assert_no_oot_data,
    evaluate_baseline_cv,
    train_baseline_pipeline,
    run_baseline,
)


@pytest.fixture
def synthetic_dev_data():
    """Create reproducible synthetic DEV partition dataset (vintage: 2016)."""
    np.random.seed(42)
    n = 600

    fico = np.random.normal(700, 35, size=n)
    dti = np.random.uniform(5, 38, size=n)
    delinq = np.random.binomial(1, 0.20, size=n)
    state = np.random.choice(["CA", "NY", "TX", "FL", "IL"], size=n)

    # Risk-generating process
    log_odds = -2.0 + (dti * 0.04) - ((fico - 700) * 0.025) + (delinq * 0.7)
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    y = np.random.binomial(1, prob, size=n)

    df = pd.DataFrame({
        "fico": fico,
        "dti": dti,
        "has_delinq": delinq,
        "state": state,
        "issue_d": ["2016-04-01"] * (n // 2) + ["2016-10-01"] * (n - n // 2),  # 2016 DEV dates
        "target": y,
    })
    return df


# =============================================================================
# 1. Pipeline Fitting, Prediction, and Thresholds
# =============================================================================

def test_baseline_pipeline_fit_predict(synthetic_dev_data):
    """Test basic fit, predict_proba, and predict functionality."""
    df = synthetic_dev_data
    X = df.drop(columns=["target"])
    y = df["target"]

    pipeline = BaselineScorecardPipeline(max_bins=5, random_state=42)
    pipeline.fit(X, y)

    # Check fitted state
    assert pipeline.woe_bins_ is not None
    assert pipeline.model_ is not None
    assert pipeline.selected_features_ is not None
    assert len(pipeline.selected_features_) > 0
    assert pipeline.training_time_ is not None
    assert pipeline.training_time_ > 0.0

    # Predict probabilities
    probs = pipeline.predict_proba(X)
    assert probs.shape == (len(df), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Predict binary decisions
    preds = pipeline.predict(X, threshold=0.5)
    assert preds.shape == (len(df),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_threshold_operating_decision(synthetic_dev_data):
    """Verify that predict(threshold) strictly mirrors (p >= threshold)."""
    df = synthetic_dev_data
    X = df.drop(columns=["target"])
    y = df["target"]

    pipeline = BaselineScorecardPipeline(max_bins=5, random_state=42)
    pipeline.fit(X, y)

    p1 = pipeline.predict_proba(X)[:, 1]

    for thresh in [0.2, 0.4, 0.6, 0.8]:
        preds = pipeline.predict(X, threshold=thresh)
        expected = (p1 >= thresh).astype(int)
        assert np.array_equal(preds, expected)


# =============================================================================
# 2. Serialization Roundtrip (Save & Load)
# =============================================================================

def test_pipeline_serialization_roundtrip(synthetic_dev_data, tmp_path):
    """Ensure pipeline artifact can be saved, reloaded, and produces identical outputs."""
    df = synthetic_dev_data
    X = df.drop(columns=["target"])
    y = df["target"]

    pipeline = BaselineScorecardPipeline(max_bins=5, random_state=42)
    pipeline.fit(X, y)

    artifact_path = tmp_path / "models" / "test_baseline.joblib"
    saved_path = pipeline.save(artifact_path)
    assert saved_path.exists()

    # Reload from disk
    loaded_pipeline = BaselineScorecardPipeline.load(artifact_path)

    # Check that predictions are identical
    orig_probs = pipeline.predict_proba(X)
    loaded_probs = loaded_pipeline.predict_proba(X)
    assert np.allclose(orig_probs, loaded_probs, atol=1e-7)

    orig_preds = pipeline.predict(X, threshold=0.35)
    loaded_preds = loaded_pipeline.predict(X, threshold=0.35)
    assert np.array_equal(orig_preds, loaded_preds)


def test_unfitted_pipeline_errors(synthetic_dev_data, tmp_path):
    """Calling predict/predict_proba/save on unfitted pipeline must raise RuntimeError."""
    df = synthetic_dev_data
    X = df.drop(columns=["target"])

    unfitted = BaselineScorecardPipeline()

    with pytest.raises(RuntimeError, match="not fitted"):
        unfitted.predict_proba(X)

    with pytest.raises(RuntimeError, match="not fitted"):
        unfitted.predict(X)

    with pytest.raises(RuntimeError, match="Cannot save unfitted"):
        unfitted.save(tmp_path / "unfitted.joblib")


# =============================================================================
# 3. Cross-Validation Evaluation & Metrics Logging
# =============================================================================

def test_cv_evaluation_metrics_and_timing(synthetic_dev_data):
    """Test Stratified K-Fold CV metrics calculation, aggregation, and timing."""
    df = synthetic_dev_data
    X = df.drop(columns=["target"])
    y = df["target"]

    n_splits = 3
    cv_res = evaluate_baseline_cv(
        X=X,
        y=y,
        n_splits=n_splits,
        threshold=0.30,
        pipeline_kwargs={"max_bins": 5},
        verbose=False,
    )

    # Check top-level keys
    expected_keys = {
        "n_splits",
        "n_samples",
        "threshold",
        "folds",
        "mean_metrics",
        "std_metrics",
        "oof_metrics",
        "total_train_time_sec",
        "total_infer_time_sec",
        "oof_predictions",
    }
    assert expected_keys.issubset(cv_res.keys())

    # Check folds structure
    assert len(cv_res["folds"]) == n_splits
    for f in cv_res["folds"]:
        assert "roc_auc" in f
        assert "gini" in f
        assert "ks_percentage" in f
        assert "pr_auc" in f
        assert "precision" in f
        assert "recall" in f
        assert "f1" in f
        assert "confusion_matrix" in f
        assert "train_time_sec" in f
        assert "infer_time_sec" in f
        assert f["train_time_sec"] > 0.0

    # Check mean metrics validity
    mean_m = cv_res["mean_metrics"]
    assert 0.5 <= mean_m["roc_auc"] <= 1.0
    assert 0.0 <= mean_m["gini"] <= 1.0
    assert 0.0 <= mean_m["ks_percentage"] <= 100.0
    assert 0.0 <= mean_m["pr_auc"] <= 1.0

    # Total times must be strictly positive
    assert cv_res["total_train_time_sec"] > 0.0
    assert cv_res["total_infer_time_sec"] > 0.0

    # Out-of-fold prediction array shape
    assert len(cv_res["oof_predictions"]) == len(df)


# =============================================================================
# 4. Out-Of-Time (OOT) Leakage Protection Guards
# =============================================================================

def test_oot_guard_raises_on_oot_data():
    """Verify that assert_no_oot_data raises ValueError when 2017+ records exist."""
    # Data containing 2017 OOT row
    df_oot = pd.DataFrame({
        "fico": [720, 680, 690],
        "issue_d": ["2016-05-01", "2016-11-01", "2017-03-01"],
        "target": [0, 1, 0],
    })

    with pytest.raises(ValueError, match="OOT data leakage detected"):
        assert_no_oot_data(df_oot)

    pipeline = BaselineScorecardPipeline()
    with pytest.raises(ValueError, match="OOT data leakage detected"):
        pipeline.fit(df_oot.drop(columns=["target"]), df_oot["target"])

    with pytest.raises(ValueError, match="OOT data leakage detected"):
        evaluate_baseline_cv(df_oot.drop(columns=["target"]), df_oot["target"])

    with pytest.raises(ValueError, match="OOT data leakage detected"):
        run_baseline(df_oot)


def test_oot_guard_passes_on_dev_and_train():
    """Verify that assert_no_oot_data passes on valid 2007-2016 records."""
    df_valid = pd.DataFrame({
        "fico": [700, 650, 720],
        "issue_d": ["2015-12-01", "2016-01-01", "2016-12-01"],
        "target": [0, 1, 0],
    })
    # Should execute without raising
    assert_no_oot_data(df_valid)


# =============================================================================
# 5. Full Pipeline Training & End-to-End Workflow
# =============================================================================

def test_train_baseline_pipeline_and_json_sidecar(synthetic_dev_data, tmp_path):
    """Verify train_baseline_pipeline saves both the joblib artifact and JSON metadata."""
    df = synthetic_dev_data
    X = df.drop(columns=["target"])
    y = df["target"]

    artifact_path = tmp_path / "artifacts" / "baseline_pipeline.joblib"
    pipeline, report = train_baseline_pipeline(
        X=X,
        y=y,
        artifact_path=artifact_path,
        threshold=0.35,
        pipeline_kwargs={"max_bins": 5},
        save_artifact=True,
        verbose=False,
    )

    assert artifact_path.exists()
    sidecar_json = artifact_path.with_suffix(".metrics.json")
    assert sidecar_json.exists()

    # Check report content
    assert report["roc_auc"] >= 0.5
    assert report["train_time_sec"] > 0.0
    assert "confusion_matrix" in report
    assert "selected_features" in report


def test_run_baseline_end_to_end(synthetic_dev_data, tmp_path):
    """Test full run_baseline function combining CV and full artifact training."""
    df = synthetic_dev_data
    artifact_path = tmp_path / "baseline_model.joblib"

    results = run_baseline(
        dev_df=df,
        target_col="target",
        artifact_path=artifact_path,
        n_splits=3,
        threshold=0.30,
        pipeline_kwargs={"max_bins": 5},
        verbose=False,
    )

    assert results["status"] == "success"
    assert "cv_results" in results
    assert "full_fit_results" in results
    assert Path(results["artifact_path"]).exists()
    assert results["n_samples"] == len(df)

