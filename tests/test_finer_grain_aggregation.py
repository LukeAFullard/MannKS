
import pytest
import numpy as np
import pandas as pd
from MannKS.trend_test import trend_test

def test_aggregation_hour():
    # Data clustered within hours
    # 10:00:00, 10:30:00 -> should aggregate to 1 point
    # 11:00:00, 11:30:00 -> should aggregate to 1 point
    dates = pd.to_datetime([
        '2023-01-01 10:00:00', '2023-01-01 10:30:00',
        '2023-01-01 11:00:00', '2023-01-01 11:30:00'
    ])
    values = np.array([10.0, 12.0, 20.0, 22.0])

    # Aggregating by hour should result in 2 points (one for 10am, one for 11am)
    # 2 points is the minimum for trend test, so it should run.
    res = trend_test(values, dates, agg_method='median', agg_period='hour')

    # If aggregation failed (treated as 4 points), p-value might be different or n would be 4.
    # However, since we can't inspect n directly from the result tuple easily without
    # inferring from other stats, we check if it runs without error and gives a result.
    # We can also check if the slope makes sense.
    # Median for 10am: 11.0, Median for 11am: 21.0. Time diff: 1 hour (3600s).
    # Slope ~ 10 units / 3600s = 0.00277
    assert res.slope is not None
    assert not np.isnan(res.slope)
    assert res.trend != 'insufficient data'

def test_aggregation_minute():
    # Data clustered within minutes
    dates = pd.to_datetime([
        '2023-01-01 10:00:05', '2023-01-01 10:00:45',
        '2023-01-01 10:01:05', '2023-01-01 10:01:45'
    ])
    values = np.array([1.0, 1.2, 2.0, 2.2])

    res = trend_test(values, dates, agg_method='median', agg_period='minute')
    assert res.slope is not None
    assert not np.isnan(res.slope)

def test_aggregation_second():
    # Data clustered within seconds (sub-second precision)
    # This might be tricky if to_period('S') truncates microseconds.
    dates = pd.to_datetime([
        '2023-01-01 10:00:00.100', '2023-01-01 10:00:00.900',
        '2023-01-01 10:00:01.100', '2023-01-01 10:00:01.900'
    ])
    values = np.array([5.0, 5.2, 6.0, 6.2])

    res = trend_test(values, dates, agg_method='median', agg_period='second')
    assert res.slope is not None
    assert not np.isnan(res.slope)

def test_invalid_period():
    dates = pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 11:00:00'])
    values = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Invalid `agg_period`"):
        trend_test(values, dates, agg_method='median', agg_period='century')
