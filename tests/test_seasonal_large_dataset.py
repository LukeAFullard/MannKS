
import pytest
import numpy as np
import pandas as pd
from MannKS import seasonal_trend_test

def test_seasonal_large_dataset_stratification():
    """
    Test that seasonal_trend_test correctly handles large datasets
    by using stratified sampling when required.
    """
    # Generate synthetic seasonal data
    # 12,000 hours = 500 days
    # This is > 10,000 points, triggering large dataset handling
    dates = pd.date_range(start='2000-01-01', periods=12000, freq='h')

    # Trend + Seasonality + Noise
    t = np.arange(len(dates))
    # Season is hour of day (0-23)
    # Note: seasonal_trend_test with numeric t and period=24 treats
    # t % 24 as the season index.

    # Strong trend, clear seasonality
    values = 0.01 * t + 10 * np.sin(2 * np.pi * (t % 24) / 24) + np.random.normal(0, 1, len(dates))

    t_numeric = np.arange(12000)

    # Run seasonal trend test
    # We set max_per_season=50 to force heavy subsampling
    result = seasonal_trend_test(
        x=values,
        t=t_numeric,
        period=24, # Hourly seasonality
        large_dataset_mode='fast',
        max_per_season=50,
        random_state=42
    )

    # Verify results
    assert result.computation_mode == 'fast'
    assert result.pairs_used is not None
    assert result.pairs_used > 0
    assert result.trend == 'increasing'

    # Check for stratified sampling note
    strat_note_found = any("stratified sampling" in note for note in result.analysis_notes)
    assert strat_note_found, "Analysis notes should contain 'stratified sampling'"

    # Check that pairs used is consistent with stratification
    # 24 seasons * (50 * 49 / 2) = 29400
    assert result.pairs_used == 29400

    # Check Confidence Intervals are valid
    assert not np.isnan(result.lower_ci), "Lower CI should not be NaN"
    assert not np.isnan(result.upper_ci), "Upper CI should not be NaN"
    assert result.lower_ci < result.upper_ci, "Lower CI should be less than Upper CI"
    assert result.lower_ci > 0, "Lower CI should be positive for increasing trend"

def test_seasonal_large_dataset_no_stratification_needed():
    """
    Test that seasonal_trend_test does NOT use stratified sampling
    if the dataset is small enough per season, even if large_dataset_mode='fast'.
    """
    # 24 seasons, 40 points per season = 960 points total
    # This is small, so even in 'fast' mode, it shouldn't stratify if max_per_season=50
    # But wait, stratified_seasonal_sampling is called if tier >= 2 (fast) AND n > 10000.
    # So if n < 10000, it shouldn't stratify regardless of max_per_season.

    n_points = 960
    t_numeric = np.arange(n_points)
    values = np.random.normal(0, 1, n_points)

    result = seasonal_trend_test(
        x=values,
        t=t_numeric,
        period=24,
        large_dataset_mode='fast', # User forces fast mode
        max_per_season=50,
        random_state=42
    )

    # Should be fast mode because user requested it
    assert result.computation_mode == 'fast'

    # But should NOT have stratified sampling note because n < 10000
    strat_note_found = any("stratified sampling" in note for note in result.analysis_notes)
    assert not strat_note_found, "Should not use stratified sampling for small datasets"

    # Check Confidence Intervals are valid
    assert not np.isnan(result.lower_ci)
    assert not np.isnan(result.upper_ci)
