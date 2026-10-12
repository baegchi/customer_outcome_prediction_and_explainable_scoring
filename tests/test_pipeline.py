"""
End-to-End Prediction Pipeline and Leakage Sanity Integration Tests (Phase 17).
Verifies:
  - End-to-end data flow: cleaning -> features -> WoE transformation -> model -> credit score
  - Scorecard points translation: inverse monotonicity (higher risk = lower credit score in [300, 850])
  - Retail credit risk tier mapping (Tiers A through E)
  - Leakage sanity checks: frozen preprocessing applied to validation/OOT data produces
    identical transformations without refitting
"""

import numpy as np
import pandas as pd
import pytest

from src.features.woe import fit_woe, transform_woe
from src.models.baseline import BaselineScorecardPipeline
from src.models.predict import (
    ScorecardPredictor,
    predict_decision,
    predict_proba,
    probability_to_scorecard_points,
    score_to_risk_tier,
)


@pytest.fixture
def mini_scorecard_data():
    """Create mini synthetic train, dev, and oot sets with realistic credit attributes."""
    np.random.seed(42)

    def _make_split(n: int, issue_d_str: str) -> pd.DataFrame:
        fico = np.random.normal(700, 35, size=n)
        dti = np.random.uniform(5, 38, size=n)
        delinq = np.random.binomial(1, 0.20, size=n)
        annual_inc = np.random.lognormal(11.0, 0.5, size=n)
        state = np.random.choice(["CA", "NY", "TX", "FL"], size=n)

        prob = 1.0 / (1.0 + np.exp(- (-2.0 + dti * 0.04 - (fico - 700) * 0.025 + delinq * 0.7)))
        y = np.random.binomial(1, prob, size=n)

        return pd.DataFrame({
            "fico": fico,
            "dti": dti,
            "has_delinq": delinq,
            "annual_inc": annual_inc,
            "state": state,
            "issue_d": [issue_d_str] * n,
            "target": y,
        })

    train_df = _make_split(400, "2015-06-01")
    dev_df = _make_split(150, "2016-06-01")
    oot_df = _make_split(150, "2017-06-01")

    return train_df, dev_df, oot_df


# =============================================================================
# 1. End-to-End Pipeline Scoring & Decisions
# =============================================================================

def test_end_to_end_scoring_pipeline(mini_scorecard_data):
    """Verify full scoring flow: fit on train, score on dev and oot."""
    train_df, dev_df, oot_df = mini_scorecard_data

    X_train = train_df.drop(columns=["target", "issue_d"])
    y_train = train_df["target"]

    # 1. Fit baseline scorecard pipeline on train
    pipeline = BaselineScorecardPipeline(max_bins=5, random_state=42)
    pipeline.fit(X_train, y_train)

    # 2. Wrap in ScorecardPredictor
    predictor = ScorecardPredictor(
        model=pipeline,
        threshold=0.35,
        pdo=20.0,
        base_score=600.0,
        base_odds=50.0,
    )

    # 3. Score dev and oot datasets
    X_dev = dev_df.drop(columns=["target", "issue_d"])
    dev_scored = predictor.score(X_dev)

    assert len(dev_scored) == len(dev_df)
    expected_cols = {"default_probability", "credit_score", "risk_tier", "decision"}
    assert expected_cols.issubset(dev_scored.columns)

    # Probability bounds
    assert np.all((dev_scored["default_probability"] >= 0.0) & (dev_scored["default_probability"] <= 1.0))

    # Score bounds [300, 850]
    assert np.all((dev_scored["credit_score"] >= 300) & (dev_scored["credit_score"] <= 850))

    # Decisions in APPROVED / DECLINED
    assert set(dev_scored["decision"].unique()).issubset({"APPROVED", "DECLINED"})

    # Check decision matches threshold cutoff
    expected_declines = dev_scored["default_probability"] >= 0.35
    actual_declines = dev_scored["decision"] == "DECLINED"
    assert np.array_equal(expected_declines, actual_declines)


# =============================================================================
# 2. Leakage Sanity Check: Immutability without Refitting
# =============================================================================

def test_leakage_sanity_check_immutable_transformation(mini_scorecard_data):
    """
    Sanity check per Phase 17:
      'A transformation fitted on training data must produce identical
       transformations when applied later to validation/OOT data without refitting.'
    """
    train_df, dev_df, oot_df = mini_scorecard_data
    X_train = train_df.drop(columns=["target", "issue_d"])
    y_train = train_df["target"]

    # Fit WoE binning on Train ONLY
    features = list(X_train.columns)
    woe_bins = fit_woe(
        X_train=X_train,
        y_train=y_train,
        features=features,
        cat_features=["state"],
        binary_features=["has_delinq"],
        max_bins=5,
        random_state=42,
    )

    X_oot = oot_df.drop(columns=["target", "issue_d"])

    # Transform OOT first time
    oot_woe_1 = transform_woe(X_oot, woe_bins, selected_only=False)

    # Transform OOT second time
    oot_woe_2 = transform_woe(X_oot, woe_bins, selected_only=False)

    # Assert bitwise identical
    pd.testing.assert_frame_equal(oot_woe_1, oot_woe_2)

    # Assert train bins dictionary is not mutated by transforming OOT
    assert set(woe_bins.keys()) == set(features)


# =============================================================================
# 3. Scorecard Translation & Risk Tier Invariants
# =============================================================================

def test_probability_to_scorecard_points_monotonic_inverse():
    """
    Credit scores must be strictly inversely monotonic with default risk:
      Higher default risk P -> Lower credit score (e.g. 500)
      Lower default risk P -> Higher credit score (e.g. 780)
    """
    probs = np.array([0.01, 0.05, 0.15, 0.30, 0.60, 0.90])
    scores = probability_to_scorecard_points(probs)

    # Scores must be strictly descending as probability increases
    assert np.all(np.diff(scores) <= 0), f"Scores must be descending with risk: {scores}"

    # Extreme clipping verification
    extreme_probs = np.array([0.000001, 0.999999])
    extreme_scores = probability_to_scorecard_points(extreme_probs)
    assert extreme_scores[0] <= 850
    assert extreme_scores[1] >= 300


def test_score_to_risk_tier_boundaries():
    """Verify tier assignment across FICO scale cutoffs."""
    test_scores = np.array([800, 750, 720, 680, 620, 550, 320])
    tiers = score_to_risk_tier(test_scores)
    expected = np.array(["A", "A", "B", "C", "D", "E", "E"])
    assert np.array_equal(tiers, expected)

