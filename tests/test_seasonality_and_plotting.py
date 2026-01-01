import pytest
import numpy as np
import pandas as pd
import os
from MannKS.check_seasonality import check_seasonality
from MannKS.plotting import plot_seasonal_distribution
from MannKS import trend_test, seasonal_trend_test

@pytest.fixture
def seasonal_data():
    # 5 years of monthly data to ensure at least 5 points per season
    t = pd.to_datetime(pd.date_range(start='2020-01-01', periods=60, freq='ME'))
    x = 10 * np.sin(2 * np.pi * t.month / 12) + 50 + np.random.rand(60)
    return x, t

@pytest.fixture
def non_seasonal_data():
    # 5 years of monthly data
    t = pd.to_datetime(pd.date_range(start='2020-01-01', periods=60, freq='ME'))
    x = np.linspace(0, 10, 60) + np.random.rand(60)
    return x, t

def test_check_seasonality_detects_seasonality(seasonal_data):
    x, t = seasonal_data
    result = check_seasonality(x, t)
    assert result.is_seasonal
    assert result.p_value < 0.05
    assert len(result.seasons_tested) == 12
    assert len(result.seasons_skipped) == 0

def test_check_seasonality_rejects_non_seasonal(non_seasonal_data):
    x, t = non_seasonal_data
    result = check_seasonality(x, t)
    assert not result.is_seasonal
    assert result.p_value > 0.05
    assert len(result.seasons_tested) == 12
    assert len(result.seasons_skipped) == 0

def test_plot_seasonal_distribution(seasonal_data):
    x, t = seasonal_data
    plot_path = "test_plot.png"

    # Ensure the file does not exist before the test
    if os.path.exists(plot_path):
        os.remove(plot_path)

    returned_path = plot_seasonal_distribution(x, t, plot_path=plot_path)

    assert returned_path == plot_path
    assert os.path.exists(plot_path)

    # Clean up the created file
    os.remove(plot_path)

def test_trend_plotting():
    """
    Tests that the plotting functionality in trend_test and
    seasonal_trend_test creates a file.
    """
    t = pd.to_datetime(pd.date_range(start='2020-01-01', periods=20, freq='YE'))
    x = np.arange(20)

    # Test trend_test plotting
    original_plot_path = "trend_test_plot.png"
    if os.path.exists(original_plot_path):
        os.remove(original_plot_path)

    trend_test(x, t, plot_path=original_plot_path)
    assert os.path.exists(original_plot_path)
    os.remove(original_plot_path)

    # Test seasonal_trend_test plotting
    seasonal_plot_path = "seasonal_trend_test_plot.png"
    if os.path.exists(seasonal_plot_path):
        os.remove(seasonal_plot_path)

    seasonal_trend_test(x, t, plot_path=seasonal_plot_path)
    assert os.path.exists(seasonal_plot_path)
    os.remove(seasonal_plot_path)

# New tests for biweekly seasonality
def test_biweekly_seasonality():
    """
    Tests the biweekly seasonality functionality.
    """
    # Create a dataset with a known biweekly pattern
    t = pd.to_datetime(pd.date_range(start='2020-01-01', periods=104, freq='W'))
    x = np.array([i % 2 for i in range(104)]) # Alternating values every week, creating a biweekly pattern

    result = seasonal_trend_test(x, t, season_type='biweekly', period=26)
    # The alternating pattern [0, 1, 0, 1] aggregated by biweekly periods (period=26)
    # might end up with no trend, or weak trend depending on aggregation/alignment.
    # The key is that h should be False (no significant trend).
    assert not result.h
    # Classification might be "No Trend" or "As Likely as Not"
    assert 'No Trend' in result.classification or 'As Likely as Not' in result.classification

def test_biweekly_seasonality_53_week_year():
    """
    Tests the biweekly seasonality with a 53-week year.
    """
    t = pd.to_datetime(pd.date_range(start='2015-01-01', end='2015-12-31', freq='D'))
    x = np.arange(len(t))

    result = seasonal_trend_test(x, t, season_type='biweekly', period=27)
    assert result.trend == 'increasing'
    assert 'Increasing' in result.classification

# Edge case tests based on code review feedback
def test_day_of_year_seasonality_leap_year():
    """
    Tests 'day_of_year' seasonality, ensuring it handles leap years correctly.
    The 'period' for day_of_year is dynamic and should not be specified.
    """
    # Data spans a leap year (2020) and a common year (2021)
    t = pd.to_datetime(pd.date_range(start='2020-01-01', end='2021-12-31', freq='D'))
    x = np.arange(len(t))

    # The function should dynamically determine the number of seasons
    result = seasonal_trend_test(x, t, season_type='day_of_year')
    assert result.trend == 'increasing'
    assert 'Increasing' in result.classification

def test_week_of_year_seasonality_53_week_year():
    """
    Tests 'week_of_year' seasonality for a year with 53 weeks.
    """
    # 2015 was a 53-week year
    t = pd.to_datetime(pd.date_range(start='2015-01-01', end='2015-12-31', freq='D'))
    x = np.arange(len(t))

    result = seasonal_trend_test(x, t, season_type='week_of_year', period=53)
    assert result.trend == 'increasing'
    assert 'Increasing' in result.classification


# Parameterized test for robust season types
@pytest.mark.parametrize("season_type, period, freq, n_periods", [
    ('year', 1, 'YE', 20),
    ('month', 12, 'ME', 60),
    ('day_of_week', 7, 'D', 365 * 2),
    ('quarter', 4, 'QE', 40),
    ('hour', 24, 'h', 168 * 2),
    ('biweekly', 26, 'W', 104 * 2),
    ('minute', 60, 'min', 1440 * 2),
    ('second', 60, 's', 3600 * 2),
])
def test_general_season_types(season_type, period, freq, n_periods):
    """
    Tests the more straightforward season types in the season_map.
    Edge cases like 'day_of_year' and 'week_of_year' are tested separately.
    """
    t = pd.to_datetime(pd.date_range(start='2020-01-01', periods=n_periods, freq=freq))
    x = np.arange(len(t))

    result = seasonal_trend_test(x, t, season_type=season_type, period=period)
    assert result.trend == 'increasing'
    assert 'Increasing' in result.classification

def test_seasonal_trend_test_aggregation_methods():
    """
    Test the 'median' and 'middle' aggregation methods in seasonal_trend_test.
    """
    # Create a dataset with multiple observations per season-year
    dates = pd.to_datetime(['2020-01-10', '2020-01-20', '2021-01-15', '2022-01-05', '2022-01-25',
                            '2020-02-10', '2020-02-20', '2021-02-15', '2022-02-05', '2022-02-25',
                           '2023-01-10', '2023-01-20', '2024-01-15', '2025-01-05', '2025-01-25'])
    np.random.seed(0)
    values = np.arange(15) + np.random.normal(0, 2, 15) # Clear increasing trend with more noise

    # Test with no aggregation
    result_none = seasonal_trend_test(x=values, t=dates, period=12, season_type='month')
    assert result_none.trend == 'increasing'
    assert 'Increasing' in result_none.classification

    # Test with 'median' aggregation
    result_median = seasonal_trend_test(x=values, t=dates, period=12, season_type='month', agg_method='median')
    assert result_median.trend == 'increasing'
    assert 'Increasing' in result_median.classification

    # Test with 'middle' aggregation
    result_middle = seasonal_trend_test(x=values, t=dates, period=12, season_type='month', agg_method='middle')
    assert result_middle.trend == 'increasing'
    assert 'Increasing' in result_middle.classification


    # The scores and slopes should be different for each aggregation method
    assert result_none.s != result_median.s
    assert result_median.s != result_middle.s
    assert result_none.slope != result_median.slope
    assert result_median.slope != result_middle.slope

def test_check_seasonality_insufficient_data():
    """Test check_seasonality with insufficient data."""
    # Scenario: 3 seasons, each with only 1 data point.
    x = [1, 2, 3]
    t = pd.to_datetime(pd.date_range(start='2020-01-01', periods=3, freq='ME'))
    result = check_seasonality(x, t)
    assert np.isnan(result.h_statistic)
    assert np.isnan(result.p_value)
    assert not result.is_seasonal
    assert result.seasons_tested == []
    assert sorted(result.seasons_skipped) == [1, 2, 3]

def test_plot_seasonal_distribution_insufficient_data():
    """Test plot_seasonal_distribution with insufficient data."""
    x = [1]
    t = [pd.to_datetime('2020-01-01')]
    result = plot_seasonal_distribution(x, t, plot_path='test.png')
    assert result is None

def test_check_seasonality_insufficient_unique_values():
    """
    Test check_seasonality returns no trend if a season has enough points
    but not enough unique values.
    """
    # Create a dataset where one season has 5 identical points
    t = pd.to_datetime(pd.date_range(start='2020-01-01', periods=60, freq='ME'))
    # Convert to numpy array to make it mutable
    x = np.array(10 * np.sin(2 * np.pi * t.month / 12) + 50 + np.random.rand(60))

    # Corrupt the data for January (month=1) to have only one unique value
    x[t.month == 1] = 42
    # Make the rest of the data random to avoid seasonality
    x[t.month != 1] = np.random.rand(len(x[t.month != 1]))


    with pytest.warns(UserWarning, match="Season '1' has less than 2 unique values and will be skipped."):
        result = check_seasonality(x, t)
        assert not result.is_seasonal
        assert len(result.seasons_tested) == 11
        assert result.seasons_skipped == [1]

def test_check_seasonality_only_one_valid_season():
    """
    Test case where only one season has enough data for the test.
    The test should not run, but the tested/skipped lists should be correct.
    """
    # Create a dataset spanning 3 years
    t = pd.to_datetime(pd.date_range(start='2020-01-01', periods=36, freq='ME'))
    x = np.random.rand(36)

    # Make only January (month=1) have sufficient, valid data
    # All other months will have less than 3 samples or less than 2 unique values
    jan_mask = (t.month == 1)
    x[~jan_mask] = 1 # Force all other months to have only one unique value

    result = check_seasonality(x, t)

    assert np.isnan(result.h_statistic)
    assert np.isnan(result.p_value)
    assert not result.is_seasonal
    assert result.seasons_tested == [1]
    # Check that all other months were skipped
    assert all(s in result.seasons_skipped for s in range(2, 13))

def test_check_seasonality_aggregation():
    """
    Test the aggregation functionality in check_seasonality.
    """
    # Create a dataset with multiple observations per month
    dates = pd.to_datetime(['2020-01-10', '2020-01-20', '2021-01-15', '2022-01-05', '2022-01-25',
                            '2020-02-10', '2020-02-20', '2021-02-15', '2022-02-05', '2022-02-25'])
    # In the raw data, Feb has high values, creating a seasonal difference.
    values = [1, 1.1, 1.2, 1.3, 1.4,  # Jan
              5, 15, 1.3, 12, 9]      # Feb (one value is low)

    # Without aggregation, the difference between Jan and Feb should be significant.
    result_none = check_seasonality(x=values, t=dates, period=12, season_type='month')
    assert result_none.is_seasonal

    # With monthly aggregation, the median smooths the data.
    # Jan aggregated: [1.05, 1.2, 1.35]
    # Feb aggregated: [10, 1.3, 10.5]
    # The groups now overlap significantly, so the result should not be seasonal.
    result_agg = check_seasonality(x=values, t=dates, period=12, season_type='month',
                                   agg_method='median', agg_period='month')
    assert not result_agg.is_seasonal
