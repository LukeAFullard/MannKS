
import pytest
import pandas as pd
import numpy as np
from MannKS import trend_test

def test_median_thinning_not_implemented():
    """
    Test that 'lwp_median' IS now implemented and does NOT raise ValueError.
    """
    # Generate 2 years of monthly data
    t = pd.date_range(start='2020-01-01', periods=24, freq='ME')
    x = np.arange(24) # Simple increasing trend

    # Should not raise ValueError anymore
    trend_test(x, t, agg_method='lwp_median', agg_period='year')

def test_median_thinning_logic():
    """
    Verify the logic of median thinning.
    """
    # Generate 2 years of monthly data
    # Year 1: 0..11 -> Median is 5.5
    # Year 2: 12..23 -> Median is 17.5
    # Timestamps are end of month.
    t = pd.date_range(start='2020-01-01', periods=24, freq='ME')
    x = np.arange(24, dtype=float)

    # Run with annual aggregation using median
    result = trend_test(x, t, agg_method='lwp_median', agg_period='year')

    # The Mann-Kendall score for 2 points [5.5, 17.5] is +1
    assert result.s == 1
    assert result.trend == 'indeterminate' # Sample size 2 is small, h might be False

    # Verification of values is inferred from 's'.
    # If aggregation worked, n should be 2.
    # We can infer n from s variance or by checking analysis notes if sample size is small.
    # Or just use a larger sample size.

    # Let's try 5 years to get a significant trend
    t5 = pd.date_range(start='2020-01-01', periods=60, freq='ME')
    x5 = np.arange(60, dtype=float)

    result5 = trend_test(x5, t5, agg_method='lwp_median', agg_period='year')

    # We expect n=5. S should be 10 (comparisons: 4+3+2+1 = 10)
    assert result5.s == 10

    # Check that robust_median also works
    result5_robust = trend_test(x5, t5, agg_method='lwp_robust_median', agg_period='year')
    assert result5_robust.s == 10
