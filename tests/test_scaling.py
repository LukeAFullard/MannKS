import pytest
import numpy as np
import pandas as pd
from MannKS import trend_test, seasonal_trend_test

def test_slope_scaling_datetime():
    """
    Tests that the slope is correctly scaled for different time units
    when using a datetime time vector.
    """
    x = np.array([1, 2, 3, 4, 5])
    t = pd.to_datetime(['2000-01-01', '2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01'])

    # Test scaling to 'year'
    result_year = trend_test(x, t, x_unit='m', slope_scaling='year')
    assert result_year.slope_units == 'm per year'
    assert np.isclose(result_year.slope, 1.0, atol=0.01) # Should be approx 1.0 m/year
    assert np.isclose(result_year.slope_per_second, result_year.slope / (365.25 * 24 * 60 * 60), atol=1e-9)

    # Test scaling to 'day'
    result_day = trend_test(x, t, x_unit='cm', slope_scaling='day')
    assert result_day.slope_units == 'cm per day'
    assert np.isclose(result_day.slope, 1.0 / 365.25, atol=0.001) # Should be approx 1/365.25 cm/day
    assert np.isclose(result_day.slope_per_second, result_day.slope / (24 * 60 * 60), atol=1e-9)

def test_slope_scaling_numeric_warning():
    """
    Tests that a UserWarning is issued when trying to scale a slope
    with a numeric time vector.
    """
    x = np.array([1, 2, 3, 4, 5])
    t = np.array([2000, 2001, 2002, 2003, 2004])

    with pytest.warns(UserWarning, match="Cannot apply `slope_scaling`"):
        result = trend_test(x, t, x_unit='m', slope_scaling='year')

    # The slope should be unscaled, and the units should reflect this
    assert result.slope == 1.0
    assert result.slope_units == 'm per unit of t'
    # For numeric time, slope_per_second is the same as the raw slope
    assert result.slope_per_second == 1.0

def test_invalid_scaling_unit():
    """
    Tests that an invalid scaling unit raises a warning and falls back gracefully.
    """
    x = np.array([1, 2, 3, 4, 5])
    t = pd.to_datetime(['2000-01-01', '2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01'])

    with pytest.warns(UserWarning, match="Slope scaling failed"):
        result = trend_test(x, t, x_unit='m', slope_scaling='invalid_unit')

    # Should fall back to units/sec
    assert result.slope_units == 'm per second'
    assert not np.isnan(result.slope)

def test_seasonal_slope_scaling():
    """
    Tests that slope scaling works correctly in seasonal_trend_test.
    """
    # Create 3 years of monthly data
    t = pd.date_range(start='2020-01-01', periods=36, freq='ME')

    # Trend: Increase by 1 unit per year
    # t is in seconds since epoch for calculation
    # 1 unit / year = 1 / (365.25 * 24 * 3600) units / sec
    seconds_per_year = 365.25 * 24 * 3600
    true_slope_per_sec = 10.0 / seconds_per_year

    # Use numeric timestamps for value generation to be precise
    t_num = t.astype(np.int64) // 10**9
    values = true_slope_per_sec * (t_num - t_num[0])

    # Add a tiny bit of noise to avoid perfect fit issues but keep trend clear
    np.random.seed(42)
    values += np.random.randn(36) * 0.001

    # 1. Test WITHOUT scaling (should be per second)
    res_raw = seasonal_trend_test(values, t, season_type='month', period=12)

    # 2. Test WITH scaling (per year)
    res_scaled = seasonal_trend_test(
        values, t, season_type='month', period=12,
        slope_scaling='year', x_unit='units'
    )

    # Check slope scaling
    # The calculated slope should be approx 10.0
    expected_slope = res_raw.slope * seconds_per_year
    assert np.isclose(res_scaled.slope, expected_slope, atol=1e-10)
    assert np.isclose(res_scaled.slope, 10.0, atol=0.1)

    # Check CIs are also scaled
    expected_lower = res_raw.lower_ci * seconds_per_year
    assert np.isclose(res_scaled.lower_ci, expected_lower, atol=1e-10)

    # Check slope_units string
    assert res_scaled.slope_units == "units per year"

def test_seasonal_scaling_variations():
    """
    Tests additional variations for seasonal slope scaling:
    - Datetime input without explicit scaling.
    - Numeric input (should not scale, warns if attempted).
    """
    # --- 1. Datetime Input, No Scaling ---
    t_dt = pd.date_range(start='2020-01-01', periods=24, freq='ME')
    values_dt = np.arange(24) # Simple increasing trend

    res_dt_raw = seasonal_trend_test(values_dt, t_dt, season_type='month', period=12)

    # Defaults to per second
    assert res_dt_raw.slope_units == "units per second"
    assert res_dt_raw.slope_per_second == res_dt_raw.slope

    # --- 2. Numeric Input, No Scaling ---
    # 2 years, monthly data represented numerically (0, 1, ..., 23)
    t_num = np.arange(24)
    values_num = t_num * 2.0 # Slope is 2.0 per unit time

    res_num_raw = seasonal_trend_test(values_num, t_num, season_type='month', period=12)

    assert res_num_raw.slope_units == "units per unit of t"
    assert np.isclose(res_num_raw.slope, 2.0, atol=1e-10)
    assert res_num_raw.slope_per_second == res_num_raw.slope # Not scaled

    # --- 3. Numeric Input, With Scaling (Should Warn) ---
    with pytest.warns(UserWarning, match="Cannot apply `slope_scaling`"):
        res_num_scaled = seasonal_trend_test(
            values_num, t_num, season_type='month', period=12,
            slope_scaling='year'
        )

    # Result should be identical to raw numeric result
    assert res_num_scaled.slope_units == "units per unit of t"
    assert np.isclose(res_num_scaled.slope, 2.0, atol=1e-10)
    assert res_num_scaled.slope == res_num_scaled.slope_per_second
