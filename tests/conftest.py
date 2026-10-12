"""Shared pytest fixtures per docs/architecture.md."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_credit_data():
    """Create reproducible synthetic credit applicant DataFrame."""
    np.random.seed(42)
    n = 200
    fico = np.random.normal(700, 35, size=n)
    dti = np.random.uniform(5, 35, size=n)
    delinq = np.random.binomial(1, 0.2, size=n)
    state = np.random.choice(["CA", "NY", "TX", "FL"], size=n)

    prob = 1.0 / (1.0 + np.exp(- (-1.8 + dti * 0.04 - (fico - 700) * 0.025 + delinq * 0.6)))
    y = np.random.binomial(1, prob, size=n)

    return pd.DataFrame({
        "fico": fico,
        "dti": dti,
        "has_delinq": delinq,
        "state": state,
        "issue_d": ["2016-06-01"] * n,
        "target": y,
    })

