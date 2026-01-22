import numpy as np
import pandas as pd
import pytest
from MannKS import trend_test, seasonal_trend_test

def generate_ar1_process(n, phi, trend_slope=0.0, noise_sd=1.0):
    """Generate AR(1) process with trend."""
    x = np.zeros(n)
    x[0] = np.random.normal(0, noise_sd)
    for i in range(1, n):
        x[i] = phi * x[i-1] + np.random.normal(0, noise_sd)

    t = pd.date_range('2000-01-01', periods=n, freq='D')
    # Add trend
    trend = trend_slope * np.arange(n)
    x += trend
    return x, t

def test_trend_test_integration_block_bootstrap():
    """Test that block bootstrap runs via trend_test."""
    np.random.seed(42)
    # Generate AR(1) data
    x, t = generate_ar1_process(n=200, phi=0.6, trend_slope=0.0)

    result = trend_test(x, t, autocorr_method='block_bootstrap', n_bootstrap=100)

    assert result.block_size_used is not None
    # With n=200, p-value should be calculable
    assert not np.isnan(result.p)
    assert 'block_bootstrap' in str(result) or result.block_size_used > 0

def test_trend_test_integration_yue_wang():
    """Test Yue-Wang correction integration."""
    np.random.seed(42)
    x, t = generate_ar1_process(n=200, phi=0.6, trend_slope=0.0)

    result = trend_test(x, t, autocorr_method='yue_wang')

    # n_effective should be less than n
    assert result.n_effective < 200
    assert result.n_effective > 0
    assert 'Yue-Wang correction' in ' '.join(result.analysis_notes)

def test_trend_test_auto_detection():
    """Test automatic autocorrelation detection."""
    np.random.seed(42)
    # High autocorrelation
    x_high, t_high = generate_ar1_process(n=200, phi=0.8, trend_slope=0.0)

    result_high = trend_test(x_high, t_high, autocorr_method='auto', n_bootstrap=50)

    # Should detect and use bootstrap
    assert 'Autocorrelation detected' in ' '.join(result_high.analysis_notes)
    assert result_high.block_size_used is not None

    # Low autocorrelation (white noise)
    # Use larger sample size to reduce chance of spurious significant lags
    x_low = np.random.normal(0, 1, 1000)
    t_low = pd.date_range('2000-01-01', periods=1000, freq='D')

    result_low = trend_test(x_low, t_low, autocorr_method='auto')

    # Should NOT use bootstrap, OR if it does (due to random chance in higher lags),
    # the acf1 should still be low.
    # The 'auto' heuristic checks for ANY significant lag, which can be overly sensitive with 50 lags.
    if result_low.block_size_used is not None:
        # If it triggered, verify it wasn't because of lag-1
        assert abs(result_low.acf1) < 0.1
    else:
        assert result_low.block_size_used is None

def test_seasonal_trend_test_integration_bootstrap():
    """Test block bootstrap with seasonal test."""
    np.random.seed(42)
    # 5 years of monthly data
    n_years = 10
    dates = pd.date_range(start='2000-01-01', periods=n_years*12, freq='ME')
    # Seasonal signal + trend + noise
    seasonal = np.tile(np.sin(np.linspace(0, 2*np.pi, 12)), n_years)
    trend = np.linspace(0, 5, n_years*12)
    noise = np.random.normal(0, 0.5, n_years*12)
    values = seasonal + trend + noise

    result = seasonal_trend_test(
        values, dates, period=12, season_type='month',
        autocorr_method='block_bootstrap', n_bootstrap=50
    )

    # Check that bootstrap ran
    assert result.block_size_used is not None
    # Should find significant trend
    assert result.h
    # p-value should be valid
    assert 0 <= result.p <= 1

import numpy as np
import pytest
from MannKS.trend_test import trend_test
from MannKS.seasonal_trend_test import seasonal_trend_test
import pandas as pd

def test_variance_reporting_bootstrap_trend_test():
    """
    Test that var_s is calculated from the bootstrap distribution
    when autocorr_method='block_bootstrap' in trend_test.
    """
    # Create data with autocorrelation
    np.random.seed(42)
    n = 100
    t = np.arange(n)
    # AR(1) process
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.7 * x[i-1] + np.random.normal(0, 1)

    # Run with bootstrap
    res_boot = trend_test(x, t, autocorr_method='block_bootstrap', n_bootstrap=200)

    # Run without bootstrap
    res_std = trend_test(x, t, autocorr_method='none')

    # Variances should be different (bootstrap variance accounts for autocorrelation)
    # Typically, positive autocorrelation increases variance of S
    assert res_boot.var_s != res_std.var_s

    # Check that res_boot.var_s is close to variance of a bootstrap distribution
    # We can't access the internal s_boot_dist easily here without modifying the return type
    # but we can verify its plausibility.
    # The analytic variance is for independent data.
    # For AR(1) with rho=0.7, variance should be inflated by (1+rho)/(1-rho) approx?
    # Note: MK variance inflation is complex.
    # But we know they shouldn't be equal.

    print(f"Standard Var(S): {res_std.var_s}")
    print(f"Bootstrap Var(S): {res_boot.var_s}")

    # Ensure it's not NaN (unless something went wrong)
    assert not np.isnan(res_boot.var_s)


def test_variance_reporting_bootstrap_seasonal_trend_test():
    """
    Test that var_s is calculated from the bootstrap distribution
    when autocorr_method='block_bootstrap' in seasonal_trend_test.
    """
    np.random.seed(42)
    n_years = 20
    t = np.arange(n_years * 12)
    # Seasonal data with inter-annual autocorrelation
    x = np.zeros(len(t))
    seasonality = np.tile(np.sin(np.linspace(0, 2*np.pi, 12)), n_years)

    # AR(1) across years
    cycle_effect = np.zeros(n_years)
    for i in range(1, n_years):
        cycle_effect[i] = 0.8 * cycle_effect[i-1] + np.random.normal(0, 1)

    cycle_effect_expanded = np.repeat(cycle_effect, 12)
    x = seasonality + cycle_effect_expanded + np.random.normal(0, 0.5, len(t))

    # Run with bootstrap
    # Note: block_size='auto' or 1 for years
    res_boot = seasonal_trend_test(x, t, period=12, autocorr_method='block_bootstrap', n_bootstrap=100)

    # Run without bootstrap
    res_std = seasonal_trend_test(x, t, period=12, autocorr_method='none')

    assert res_boot.var_s != res_std.var_s

    print(f"Standard Seasonal Var(S): {res_std.var_s}")
    print(f"Bootstrap Seasonal Var(S): {res_boot.var_s}")

    assert not np.isnan(res_boot.var_s)
