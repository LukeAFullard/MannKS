
import os
import pytest
import numpy as np
import pandas as pd
from MannKS import trend_test, seasonal_trend_test

def test_residual_plot_generation(tmp_path):
    """
    Test that residual diagnostic plots are correctly generated for both
    trend_test and seasonal_trend_test.
    """
    # 1. Test trend_test
    t = np.arange(20)
    x = 0.5 * t + np.random.normal(0, 1, 20)

    plot_path = tmp_path / "residuals_trend.png"

    trend_test(x, t, residual_plot_path=str(plot_path))

    assert plot_path.exists(), "Residual plot for trend_test was not created."
    assert plot_path.stat().st_size > 0, "Residual plot is empty."

    # 2. Test seasonal_trend_test
    # Create 2 years of monthly data
    dates = pd.date_range(start='2020-01-01', periods=24, freq='ME')
    values = np.linspace(0, 10, 24) + np.tile(np.arange(12), 2) # Trend + Seasonality

    seasonal_plot_path = tmp_path / "residuals_seasonal.png"

    seasonal_trend_test(
        values, dates,
        period=12, season_type='month',
        residual_plot_path=str(seasonal_plot_path)
    )

    assert seasonal_plot_path.exists(), "Residual plot for seasonal_trend_test was not created."
    assert seasonal_plot_path.stat().st_size > 0, "Seasonal residual plot is empty."

def test_residual_plot_censored(tmp_path):
    """
    Test residual plot generation with censored data.
    """
    t = np.arange(20)
    x = 0.5 * t + np.random.normal(0, 1, 20)
    # Censor some data (as strings)
    x_str = [f"<{val}" if i % 5 == 0 else val for i, val in enumerate(x)]

    plot_path = tmp_path / "residuals_censored.png"

    # trend_test handles the string conversion internally via _prepare_data
    # but we need to import prepare_censored_data if we want to pass a dataframe,
    # or just let trend_test handle the list of mixed types (if it supports it).
    # trend_test documentation says "x: a vector of data". prepare_censored_data is usually needed.
    # Let's rely on trend_test's internal call to _prepare_data which handles mixed lists if supported,
    # or better, use prepare_censored_data explicitly as that's the public API pattern.

    from MannKS import prepare_censored_data
    df = prepare_censored_data(x_str)

    trend_test(df, t, residual_plot_path=str(plot_path))

    assert plot_path.exists(), "Residual plot for censored data was not created."

def test_residual_plot_insufficient_data(tmp_path):
    """
    Test that plot generation handles insufficient data gracefully (no crash, no file or empty file depending on implementation).
    The current implementation returns early if n < 2.
    """
    t = [1]
    x = [1]
    plot_path = tmp_path / "residuals_fail.png"

    trend_test(x, t, residual_plot_path=str(plot_path))

    # Should NOT exist because it returns early before calling plot_residuals
    assert not plot_path.exists(), "Residual plot should not be created for insufficient data."
