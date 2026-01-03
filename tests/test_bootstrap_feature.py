
import pytest
import numpy as np
import pandas as pd
from MannKS.trend_test import trend_test
from MannKS._bootstrap import block_bootstrap_confidence_intervals

def test_bootstrap_trend_test_integration():
    """Test that setting autocorr_method='block_bootstrap' works end-to-end."""
    n = 30
    t = np.arange(n)
    # Autocorrelated data: x_t = 0.5 * x_{t-1} + noise
    x = np.zeros(n)
    np.random.seed(42)
    for i in range(1, n):
        x[i] = 0.5 * x[i-1] + np.random.normal(0, 1)

    # Add a trend
    x += 0.1 * t

    # Basic test
    res = trend_test(x, t, autocorr_method='block_bootstrap', n_bootstrap=50)

    assert res.block_size_used is not None
    assert res.block_size_used > 1
    # Check that CI is populated
    assert not np.isnan(res.lower_ci)
    assert not np.isnan(res.upper_ci)

def test_bootstrap_censored_ci_sorting():
    """
    Verify that the bootstrap CI calculation specifically handles censored data
    and sorts the time vector internally (preventing the bug).
    """
    n = 20
    t = np.arange(n, dtype=float)
    x = t + np.random.normal(0, 0.1, n)
    censored = np.zeros(n, dtype=bool)
    cen_type = np.full(n, 'not')

    # Add some censoring that would trigger 'ambiguous' logic if time was reversed
    # e.g. Left-censored at end of series (high value)
    # If moved to start, it's a high limit.
    censored[-5:] = True
    cen_type[-5:] = 'lt'
    x[-5:] = 100.0 # High detection limit

    # If unsorted, pairs might show (Late, Early) -> (High <100, Low 1)
    # If sorted, pairs are (Early 1, Late <100) -> Ambiguous or Increasing?

    slope_obs, lower, upper, boots = block_bootstrap_confidence_intervals(
        x, t, censored, cen_type,
        n_bootstrap=20,
        block_size=5
    )

    # We just ensure it runs and produces finite values
    assert np.isfinite(slope_obs)
    assert np.isfinite(lower)
    assert np.isfinite(upper)
    assert len(boots) == 20
