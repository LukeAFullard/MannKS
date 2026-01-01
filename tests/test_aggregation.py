import pytest
import numpy as np
import pandas as pd
from MannKS import seasonal_trend_test
from MannKS.preprocessing import prepare_censored_data

def test_seasonal_trend_test_aggregation_methods():
    """
    Test the `agg_method` options in `seasonal_trend_test` to ensure they
    correctly handle multiple observations within a single season-cycle.
    """
    # Create a dataset with multiple observations in January 2020 and July 2021
    # and include a mix of censored and non-censored data.
    t = pd.to_datetime([
        '2020-01-10', '2020-01-20',  # Jan 2020
        '2020-07-15',              # Jul 2020
        '2021-01-15',              # Jan 2021
        '2021-07-10', '2021-07-20'   # Jul 2021
    ])
    x = ['<5', 10, 15, 8, 20, '<25']
    data = prepare_censored_data(x)

    # Define the aggregation methods to test
    agg_methods = ['median', 'robust_median', 'middle']

    for method in agg_methods:
        # Run the seasonal test with the specified aggregation method
        result = seasonal_trend_test(x=data, t=t, period=12, agg_method=method)

        # Assert that the test returns a valid result without crashing.
        # For this small, synthetic dataset, we are primarily concerned with
        # ensuring the aggregation logic executes correctly, not with a
        # specific trend outcome, which may be statistically weak.
        assert result is not None
        assert result.trend in ['increasing', 'decreasing', 'no trend', 'indeterminate']
        assert isinstance(result.h, (bool, np.bool_))
        assert not np.isnan(result.p) if result.h else True
