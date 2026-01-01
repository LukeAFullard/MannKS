import pytest
import numpy as np
import pandas as pd
from MannKS.trend_test import trend_test
from MannKS.seasonal_trend_test import seasonal_trend_test

@pytest.fixture
def unequal_linear_data():
    t = np.array([1, 2, 4, 7, 8, 10])
    x = 2 * t + 5 + np.random.rand(len(t))
    return t, x

@pytest.fixture
def unequal_seasonal_data():
    t = np.array([1, 2, 3, 13, 14, 15, 25, 26, 27])
    x = 10 * np.sin(2 * np.pi * t / 12) + np.random.rand(len(t))
    return t, x

def test_trend_test_unequal_spacing(unequal_linear_data):
    t, x = unequal_linear_data
    result = trend_test(x, t)

    assert result.trend == 'increasing'
    assert result.h
    assert result.slope == pytest.approx(2.0, abs=0.5)
    assert result.lower_ci <= result.slope <= result.upper_ci
    assert result.C > 0.95
    assert result.Cd < 0.05


def test_seasonal_trend_test_weekly_aggregation():
    # 5 weeks of data, with two observations on each Monday
    t = pd.to_datetime(np.array([
        '2023-01-02 10:00', '2023-01-02 14:00', # Week 1, Monday
        '2023-01-09 10:00', '2023-01-09 14:00', # Week 2, Monday
        '2023-01-16 10:00', '2023-01-16 14:00', # Week 3, Monday
        '2023-01-23 10:00', '2023-01-23 14:00', # Week 4, Monday
        '2023-01-30 10:00', '2023-01-30 14:00', # Week 5, Monday
    ]))
    x = np.array([10, 100, 20, 200, 30, 300, 40, 400, 50, 500])

    # Aggregation should find the median of each Monday's pair
    result = seasonal_trend_test(x, t, period=7, agg_method='median', season_type='day_of_week')

    assert result.trend == 'increasing'
    assert result.h
    # s for the single aggregated Monday series [55, 110, 165, 220, 275] is 10
    assert result.s == 10


def test_seasonal_trend_test_hourly_aggregation():
    # 5 days of data, with two observations at 10:00
    t = pd.to_datetime(np.array([
        '2023-01-01 10:00', '2023-01-01 10:30',
        '2023-01-02 10:00', '2023-01-02 10:30',
        '2023-01-03 10:00', '2023-01-03 10:30',
        '2023-01-04 10:00', '2023-01-04 10:30',
        '2023-01-05 10:00', '2023-01-05 10:30',
    ]))
    x = np.array([10, 100, 20, 200, 30, 300, 40, 400, 50, 500])

    result = seasonal_trend_test(x, t, period=24, agg_method='median', season_type='hour')

    assert result.trend == 'increasing'
    assert result.h
    # s for the single aggregated 10:00 series [55, 110, 165, 220, 275] is 10
    assert result.s == 10


def test_seasonal_trend_test_parameter_validation():
    t = pd.to_datetime(pd.date_range(start='2023-01-01', periods=10, freq='D'))
    x = np.arange(10)

    # Test that an incorrect period for a given season_type raises an error
    with pytest.raises(ValueError, match="For season_type='day_of_week', period must be 7."):
        seasonal_trend_test(x, t, period=12, season_type='day_of_week')

    # Test that an unknown season_type raises an error
    with pytest.raises(ValueError, match="Unknown season_type: 'invalid_season'."):
        seasonal_trend_test(x, t, period=7, season_type='invalid_season')

    # Test for day_of_year, which should not raise an error regardless of period
    t_leap = pd.to_datetime(pd.date_range(start='2020-01-01', periods=400, freq='D'))
    x_leap = np.arange(400)
    seasonal_trend_test(x_leap, t_leap, period=366, season_type='day_of_year')
    seasonal_trend_test(x_leap, t_leap, period=365, season_type='day_of_year')


def test_seasonal_trend_test_numeric_year_aggregation():
    # 5 years of data to ensure statistical significance
    t = np.array([1990.1, 1990.9, 1991.1, 1991.9, 1992.1, 1992.9, 1993.1, 1993.9, 1994.1, 1994.9])
    x = np.array([10, 100, 20, 200, 30, 300, 40, 400, 50, 500])

    result = seasonal_trend_test(x, t, period=1, agg_method='median')

    assert result.trend == 'increasing'
    assert result.h
    assert result.s == 10


def test_seasonal_trend_test_year_seasonality():
    t = pd.to_datetime(pd.date_range(start='2020-01-01', periods=10, freq='YE'))
    x = np.arange(10, 110, 10) # [10, 20, ..., 100]

    # Using 'year' should be equivalent to a non-seasonal trend test
    result = seasonal_trend_test(x, t, period=1, season_type='year')

    assert result.trend == 'increasing'
    assert result.h
    assert result.s == 45


def test_seasonal_trend_test_datetime_aggregation():
    # 5 years of data with two observations in Jan
    t = np.array([
        '2023-01-10', '2023-01-20',
        '2024-01-12', '2024-01-22',
        '2025-01-15', '2025-01-25',
        '2026-01-11', '2026-01-21',
        '2027-01-14', '2027-01-24',
    ], dtype='datetime64')
    x = np.array([10, 100, 20, 200, 30, 300, 40, 400, 50, 500])

    result = seasonal_trend_test(x, t, period=12, agg_method='median')

    assert result.trend == 'increasing'
    assert result.h
    assert result.s == 10 # For x_agg = [55, 110, 165, 220, 275]
    assert result.C > 0.95
    assert result.Cd < 0.05


@pytest.fixture
def seasonal_data_with_duplicates():
    # Two observations in the first 'month' of each of 5 'years'
    t = np.array([1.1, 1.9, 13.1, 13.9, 25.1, 25.9, 37.1, 37.9, 49.1, 49.9])
    # Clear increasing trend year-over-year
    x = np.array([10, 100, 20, 200, 30, 300, 40, 400, 50, 500])
    return t, x


def test_seasonal_trend_test_aggregation(seasonal_data_with_duplicates):
    t, x = seasonal_data_with_duplicates

    # Test 'median' aggregation
    result_med = seasonal_trend_test(x, t, period=12, agg_method='median')
    assert result_med.trend == 'increasing'
    assert result_med.s == 10

    # Test 'middle' aggregation
    result_mid = seasonal_trend_test(x, t, period=12, agg_method='middle')
    assert result_mid.trend == 'increasing'
    assert result_mid.s == 10

    # Test 'none' (default) aggregation
    result_none = seasonal_trend_test(x, t, period=12, agg_method='none')
    assert result_none.trend == 'increasing'
    assert result_none.s == 25


def test_trend_test_with_datetime64():
    t = np.array(['2023-01-01', '2023-01-02', '2023-01-04', '2023-01-07', '2023-01-08', '2023-01-10'], dtype='datetime64')
    x = 2 * np.arange(len(t)) + 5 + np.random.rand(len(t))
    result = trend_test(x, t)

    assert result.trend == 'increasing'
    assert result.h
    assert result.C > 0.95
    assert result.Cd < 0.05


def test_trend_test_with_datetime_objects():
    from datetime import datetime
    t = [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 4), datetime(2023, 1, 7), datetime(2023, 1, 8), datetime(2023, 1, 10)]
    x = 2 * np.arange(len(t)) + 5 + np.random.rand(len(t))
    result = trend_test(x, t)

    assert result.trend == 'increasing'
    assert result.h
    assert result.C > 0.95
    assert result.Cd < 0.05


def test_seasonal_trend_test_unequal_spacing(unequal_seasonal_data):
    t, x = unequal_seasonal_data
    result = seasonal_trend_test(x, t, period=12)
    # The trend might not be exactly 'no trend' (might be weak increasing/decreasing)
    # but it should definitely not be significant
    assert not result.h


def test_trend_test_no_trend():
    np.random.seed(0)
    t = np.linspace(0, 10, 20)
    x = np.full_like(t, 5.0)
    x += np.random.normal(0, 0.1, size=x.shape)

    result = trend_test(x, t)

    # With random noise, we expect no significant trend, but direction might be assigned
    assert not result.h
    # It might be 'increasing' or 'decreasing' depending on the random noise
    assert result.trend in ['increasing', 'decreasing', 'no trend']
    assert result.slope == pytest.approx(0.0, abs=0.1)
    assert result.lower_ci <= result.slope <= result.upper_ci


def test_seasonal_trend_test_with_trend():
    t = np.arange(0, 48, 1)
    seasonal_component = np.sin(2 * np.pi * t / 12)
    trend_component = 0.1 * t
    x = seasonal_component + trend_component

    result = seasonal_trend_test(x, t, period=12)

    assert result.trend == 'increasing'
    assert result.h
    assert result.slope == pytest.approx(0.1, abs=0.05)
    assert result.lower_ci <= result.slope <= result.upper_ci
    assert result.C > 0.95
    assert result.Cd < 0.05


def test_trend_test_with_nan():
    t = np.array([1, 2, 3, 4, 5, 6])
    x = np.array([1, 2, np.nan, 4, 5, 6])
    result = trend_test(x, t)
    assert result.trend == 'increasing'
    assert result.h
    assert result.slope == pytest.approx(1.0, abs=0.1)
    assert result.lower_ci <= result.slope <= result.upper_ci
    assert result.C > 0.95
    assert result.Cd < 0.05


def test_seasonal_trend_test_with_nan():
    t = np.arange(0, 48, 1).astype(float)
    x = 0.1 * t + np.sin(2 * np.pi * t / 12)
    x[5] = np.nan
    x[20] = np.nan

    result = seasonal_trend_test(x, t, period=12)

    assert result.trend == 'increasing'
    assert result.h
    assert result.slope == pytest.approx(0.1, abs=0.05)
    assert result.lower_ci <= result.slope <= result.upper_ci
    assert result.C > 0.95
    assert result.Cd < 0.05

def test_tied_timestamp_warning():
    """
    Tests that an analysis note is returned when tied timestamps are present.
    """
    t = np.array([1, 2, 2, 4, 5])
    x = np.arange(len(t))
    result = trend_test(x, t)
    assert 'tied timestamps present without aggregation' in result.analysis_notes

def test_empty_input():
    """
    Tests that the functions handle empty inputs without crashing.
    """
    result_orig = trend_test([], [])
    assert result_orig.trend == 'no trend'
    assert not result_orig.h
    assert np.isnan(result_orig.p)

    result_seasonal = seasonal_trend_test([], [])
    assert result_seasonal.trend == 'no trend'
    assert not result_seasonal.h
    assert np.isnan(result_seasonal.p)

def test_all_nan_input():
    """
    Tests that the functions handle inputs with only NaN values.
    """
    x = [np.nan, np.nan, np.nan]
    t = [1, 2, 3]
    result_orig = trend_test(x, t)
    assert result_orig.trend == 'no trend'
    assert not result_orig.h
    assert np.isnan(result_orig.p)

    result_seasonal = seasonal_trend_test(x, t)
    assert result_seasonal.trend == 'no trend'
    assert not result_seasonal.h
    assert np.isnan(result_seasonal.p)
