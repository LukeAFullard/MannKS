
import pytest
import numpy as np
import pandas as pd
from MannKS.rolling_trend import rolling_trend_test, compare_periods
from MannKS.trend_test import trend_test

def test_rolling_trend_numeric():
    t = np.arange(100)
    x = t * 0.5 + np.random.normal(0, 1, 100)

    # Simple numeric window
    df = rolling_trend_test(x, t, window=20, step=10, min_size=5)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'slope' in df.columns
    assert 'window_start' in df.columns

    # Check window size
    assert df['n_obs'].min() >= 5
    assert (df['window_end'] - df['window_start'] == 20).all()

def test_rolling_trend_datetime():
    t = pd.date_range(start='2020-01-01', periods=100, freq='D')
    x = np.arange(100) * 0.1 + np.random.normal(0, 0.1, 100)

    # Datetime window
    df = rolling_trend_test(x, t, window='30D', step='10D', min_size=5)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    # Check if window logic works for datetime
    assert isinstance(df.iloc[0]['window_start'], (pd.Timestamp, np.datetime64))

def test_rolling_trend_seasonal():
    # 2 years of monthly data
    t = pd.date_range(start='2020-01-01', periods=24, freq='ME')
    x = np.sin(np.arange(24)) + np.arange(24)*0.1 # Seasonal + Trend

    # Window needs to be large enough for seasonal test (usually > 2 seasons)
    # But here we just test execution
    df = rolling_trend_test(x, t, window='18ME', step='6ME', seasonal=True, period=12, min_size=12)

    assert isinstance(df, pd.DataFrame)
    # With 18 months, we might have enough for seasonal test depending on impl
    # 18 months = 1.5 cycles. seasonal test needs >= 2 obs per season?
    # Actually seasonal_trend_test needs at least 2 points total, but meaningful results need more.
    # Let's see if it crashes.

def test_rolling_trend_insufficient_data():
    t = np.arange(10)
    x = np.random.randn(10)

    # Window larger than data
    df = rolling_trend_test(x, t, window=20, step=5)
    # Should be empty or one window?
    # _generate_windows goes from min to max.
    # 0 to 20. mask sum will be 10. min_size default is 10.
    # Should get one result if n >= min_size.

    # Window smaller than min_size
    df_empty = rolling_trend_test(x, t, window=5, min_size=10)
    assert df_empty.empty

def test_compare_periods():
    t = np.arange(100)
    x = np.concatenate([t[:50]*0.1, t[50:]*(-0.1)]) # Up then Down

    res = compare_periods(x, t, breakpoint=50)

    assert res['slope_difference'] != 0
    assert res['before'].slope > 0
    assert res['after'].slope < 0
    assert res['significant_change']

def test_compare_periods_datetime():
    t = pd.date_range(start='2020-01-01', periods=100, freq='D')
    x = np.arange(100)
    breakpoint = pd.Timestamp('2020-02-20') # Roughly middle

    res = compare_periods(x, t, breakpoint=breakpoint)
    assert res['breakpoint'] == breakpoint
    assert isinstance(res['slope_difference'], float)

def test_rolling_input_validation():
    t = np.arange(10)
    x = np.arange(9) # Mismatch

    with pytest.raises(ValueError, match="same length"):
        rolling_trend_test(x, t, window=5)

    with pytest.raises(ValueError):
         rolling_trend_test(t, t, window='5D') # Numeric t with string window
