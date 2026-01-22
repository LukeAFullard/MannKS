
import pytest
import numpy as np
import pandas as pd
from MannKS import trend_test, seasonal_trend_test
from MannKS._stats import _mk_score_and_var_censored, _sens_estimator_unequal_spacing
from MannKS._large_dataset import fast_sens_slope, stratified_seasonal_sampling

def test_audit_mk_score_optimization():
    """
    AUDIT: Verify O(N log N) MK Score implementation.
    The optimized MK score calculation must match the exact O(N^2) calculation
    for uncensored data.
    """
    np.random.seed(42)
    # N=6000 triggers the fast path (>5000)
    n = 6000
    x = np.random.normal(0, 1, n)
    t = np.arange(n)

    # 1. Fast Path (Default for N > 5000)
    s_fast, var_fast, _, _ = _mk_score_and_var_censored(
        x, t, np.zeros(n, bool), np.full(n, 'not')
    )

    # 2. Slow Path (Forced via lwp method to bypass fast path check,
    #    but using robust tie-breaking to match logic)
    s_slow, var_slow, _, _ = _mk_score_and_var_censored(
        x, t, np.zeros(n, bool), np.full(n, 'not'),
        mk_test_method='lwp', tie_break_method='robust'
    )

    assert s_fast == s_slow, f"MK Score Mismatch: Fast={s_fast}, Slow={s_slow}"
    assert var_fast == var_slow, f"Variance Mismatch: Fast={var_fast}, Slow={var_slow}"


def test_audit_stochastic_slope_accuracy():
    """
    AUDIT: Verify Stochastic Sen's Slope accuracy.
    The randomized slope estimator must be within acceptable error bounds (e.g. 5%)
    of the true Sen's slope.
    """
    np.random.seed(123)
    n = 1000
    t = np.arange(n)
    true_slope = 2.5
    # Add noise
    x = true_slope * t + np.random.normal(0, 50, n)

    # 1. Exact Calculation
    slopes_exact = _sens_estimator_unequal_spacing(x, t)
    slope_exact = np.median(slopes_exact)

    # 2. Fast Calculation (Forced)
    # Sampling 10,000 pairs out of ~500,000 possible
    slopes_fast = fast_sens_slope(x, t, max_pairs=10000, random_state=123)
    slope_fast = np.median(slopes_fast)

    error = abs(slope_fast - slope_exact) / abs(slope_exact)

    print(f"Exact Slope: {slope_exact}")
    print(f"Fast Slope: {slope_fast}")
    print(f"Error: {error:.4f}")

    assert error < 0.05, f"Stochastic slope error {error:.4f} exceeds 5% tolerance"


def test_audit_seasonal_stratification():
    """
    AUDIT: Verify Seasonal Stratified Sampling.
    Ensure that for large seasonal datasets, the sampling preserves the season
    structure and returns valid results.
    """
    # Create 10 years of hourly data (approx 87k points)
    # This forces 'aggregate' mode usually, but we will force 'fast' mode
    # to test stratification.
    dates = pd.date_range('2000-01-01', periods=24*365*2, freq='h') # 2 years, ~17k points
    n = len(dates)

    # Linear trend + Daily Seasonality
    values = 0.01 * np.arange(n) + 5 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 1, n)

    result = seasonal_trend_test(
        values, dates,
        season_type='hour', # 24 seasons
        large_dataset_mode='fast',
        max_per_season=100, # Should result in 24 * 100 = 2400 points used
        random_state=42
    )

    assert result.computation_mode == 'fast'
    assert result.pairs_used is not None

    # Check that pairs used is roughly what we expect
    # 24 seasons. Each has 100 points (stratified).
    # Pairs per season = 100*99/2 = 4950
    # Total pairs = 24 * 4950 = 118800
    assert result.pairs_used == 118800

    assert result.trend == 'increasing'
    assert result.p < 0.05


def test_audit_api_integration():
    """
    AUDIT: Verify API correctly exposes large dataset metadata.
    """
    n = 6000
    t = np.arange(n)
    x = t + np.random.normal(0, 1, n)

    res = trend_test(x, t)

    assert res.computation_mode == 'fast'
    # Pairs used should be <= default max_pairs (100000)
    # With N=6000, total pairs is ~18M.
    # It definitely uses sampling.
    assert res.pairs_used <= 100000
    assert res.approximation_error is not None
