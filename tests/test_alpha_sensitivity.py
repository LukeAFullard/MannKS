
import pytest
import numpy as np
import pandas as pd
from MannKS import trend_test, rolling_trend_test, segmented_trend_test

def generate_linear_data(n=100, slope=0.1, noise=1.0, seed=42):
    np.random.seed(seed)
    t = np.arange(n)
    x = slope * t + np.random.normal(0, noise, n)
    return x, t

def generate_segmented_data(n=100, breakpoint=50, slope1=0.2, slope2=-0.2, noise=1.0, seed=42):
    np.random.seed(seed)
    t = np.arange(n)
    x = np.zeros(n)
    x[:breakpoint] = slope1 * t[:breakpoint]
    x[breakpoint:] = x[breakpoint-1] + slope2 * (t[breakpoint:] - t[breakpoint-1])
    x += np.random.normal(0, noise, n)
    return x, t

def test_linear_trend_alpha_sensitivity():
    """
    Test that trend_test confidence intervals widen as alpha decreases (confidence increases).
    """
    x, t = generate_linear_data()

    # Test with standard alpha (95%)
    res_05 = trend_test(x, t, alpha=0.05)
    ci_width_05 = res_05.upper_ci - res_05.lower_ci

    # Test with stricter alpha (99%) -> Expect wider interval
    res_01 = trend_test(x, t, alpha=0.01)
    ci_width_01 = res_01.upper_ci - res_01.lower_ci

    # Test with looser alpha (90%) -> Expect narrower interval
    res_10 = trend_test(x, t, alpha=0.10)
    ci_width_10 = res_10.upper_ci - res_10.lower_ci

    # Assertions
    assert ci_width_01 > ci_width_05, f"CI for alpha=0.01 ({ci_width_01:.4f}) should be wider than alpha=0.05 ({ci_width_05:.4f})"
    assert ci_width_05 > ci_width_10, f"CI for alpha=0.05 ({ci_width_05:.4f}) should be wider than alpha=0.10 ({ci_width_10:.4f})"

def test_rolling_trend_alpha_sensitivity():
    """
    Test that rolling_trend_test confidence intervals widen as alpha decreases.
    """
    x, t = generate_linear_data(n=200) # Longer series for rolling

    # Run rolling test
    res_05 = rolling_trend_test(x, t, window=50, alpha=0.05)
    res_01 = rolling_trend_test(x, t, window=50, alpha=0.01)

    # Ensure we have matching windows
    assert len(res_05) == len(res_01)

    # Calculate average CI width
    avg_width_05 = (res_05['upper_ci'] - res_05['lower_ci']).mean()
    avg_width_01 = (res_01['upper_ci'] - res_01['lower_ci']).mean()

    assert avg_width_01 > avg_width_05, f"Avg rolling CI for alpha=0.01 ({avg_width_01:.4f}) should be wider than alpha=0.05 ({avg_width_05:.4f})"

def test_segmented_trend_alpha_sensitivity_ols():
    """
    Test that segmented_trend_test (standard OLS) confidence intervals for both
    slopes and breakpoints widen as alpha decreases.
    """
    x, t = generate_segmented_data(n=150, breakpoint=75, noise=2.0)

    # Use fixed n_breakpoints to ensure comparable models
    n_bp = 1

    res_05 = segmented_trend_test(x, t, n_breakpoints=n_bp, alpha=0.05, use_bagging=False)
    res_01 = segmented_trend_test(x, t, n_breakpoints=n_bp, alpha=0.01, use_bagging=False)

    # 1. Check Slopes
    # Compare the CI width of the first segment
    slope_width_05 = res_05.segments.iloc[0]['upper_ci'] - res_05.segments.iloc[0]['lower_ci']
    slope_width_01 = res_01.segments.iloc[0]['upper_ci'] - res_01.segments.iloc[0]['lower_ci']

    assert slope_width_01 > slope_width_05, f"Segment slope CI for alpha=0.01 ({slope_width_01:.4f}) should be wider than 0.05 ({slope_width_05:.4f})"

    # 2. Check Breakpoints (OLS method)
    bp_ci_05 = res_05.breakpoint_cis[0]
    bp_width_05 = bp_ci_05[1] - bp_ci_05[0]

    bp_ci_01 = res_01.breakpoint_cis[0]
    bp_width_01 = bp_ci_01[1] - bp_ci_01[0]

    # Ensure we actually have valid CIs
    if not np.isnan(bp_width_05) and not np.isnan(bp_width_01):
        assert bp_width_01 > bp_width_05, f"Breakpoint CI for alpha=0.01 ({bp_width_01:.4f}) should be wider than 0.05 ({bp_width_05:.4f})"

def test_segmented_trend_alpha_sensitivity_bagging():
    """
    Test that segmented_trend_test (Bagging) slope confidence intervals widen as alpha decreases.
    """
    x, t = generate_segmented_data(n=100, breakpoint=50, noise=3.0)
    n_bp = 1

    # Note on Bagging Breakpoint CIs:
    # Bagging involves random sampling. Comparing width between two separate stochastic runs
    # is not guaranteed to strictly follow alpha if n_bootstrap is small.
    # However, the SLOPES are calculated using the Sen's Slope on the determined segments.
    # Even if breakpoints shift slightly, the Sen's Slope CI logic (Mann-Kendall based)
    # is deterministic for a given segment and alpha.

    res_05 = segmented_trend_test(x, t, n_breakpoints=n_bp, alpha=0.05, use_bagging=True, n_bootstrap=20)
    res_01 = segmented_trend_test(x, t, n_breakpoints=n_bp, alpha=0.01, use_bagging=True, n_bootstrap=20)

    # Check Slope CIs for first segment
    slope_width_05 = res_05.segments.iloc[0]['upper_ci'] - res_05.segments.iloc[0]['lower_ci']
    slope_width_01 = res_01.segments.iloc[0]['upper_ci'] - res_01.segments.iloc[0]['lower_ci']

    assert slope_width_01 > slope_width_05, f"Bagging Segment slope CI for alpha=0.01 ({slope_width_01:.4f}) should be wider than 0.05 ({slope_width_05:.4f})"
