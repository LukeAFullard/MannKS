
import numpy as np
import pandas as pd
from MannKS.trend_test import trend_test
from MannKS.seasonal_trend_test import seasonal_trend_test
from MannKS.preprocessing import prepare_censored_data
from MannKS._stats import _sens_estimator_censored

def test_trend_test_custom_multipliers():
    # Using non-monotonic data around the censored value
    x = np.array([1, 5, '<3', 2, 4, 6, 7, 8, 9, 10])
    t = np.arange(10)

    data = prepare_censored_data(x)

    # Calculate slopes with default multipliers
    default_slopes = _sens_estimator_censored(data['value'].to_numpy(), t, data['cen_type'].to_numpy())

    # Calculate slopes with custom multipliers
    custom_slopes = _sens_estimator_censored(data['value'].to_numpy(), t, data['cen_type'].to_numpy(), lt_mult=0.1)

    # The different multiplier for the censored value should result in a different set of slopes
    assert not np.array_equal(default_slopes, custom_slopes)

def test_seasonal_trend_test_custom_multipliers():
    # Using non-monotonic data and a period that groups data points into seasons
    x = np.array([1, 5, '<4', 2, 6, 10, '>8', 7, 9, 11, 12, 13])
    t = np.arange(12)

    data = prepare_censored_data(x)

    # Calculate slopes with default multipliers
    default_slopes = _sens_estimator_censored(data['value'].to_numpy(), t, data['cen_type'].to_numpy())

    # Calculate slopes with custom multipliers
    custom_slopes = _sens_estimator_censored(data['value'].to_numpy(), t, data['cen_type'].to_numpy(), lt_mult=0.1, gt_mult=1.5)

    # The different multipliers should result in a different set of slopes
    assert not np.array_equal(default_slopes, custom_slopes)
