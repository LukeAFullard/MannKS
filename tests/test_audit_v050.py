
import pytest
import numpy as np
import pandas as pd
from MannKS import trend_test, seasonal_trend_test

def test_fast_mk_edge_case_n5001():
    """
    Audit Recommendation: n=5001 edge case (just over threshold).
    Verifies that the fast path is triggered and produces correct results.
    """
    np.random.seed(42)
    n = 5001
    t = np.arange(n)
    # Clear trend: x = 0.5*t + noise
    x = 0.5 * t + np.random.normal(0, 10, n)

    result = trend_test(x, t)

    assert result.computation_mode == 'fast'
    assert result.trend == 'increasing'
    assert result.h == True
    # Fast mode slope pairs should be capped at default max_pairs (100,000)
    # n=5001 => ~12.5M pairs.
    assert result.pairs_used is not None
    assert result.pairs_used <= 100000
    assert result.slope > 0.4 and result.slope < 0.6


def test_seasonal_stratification_n15000():
    """
    Audit Recommendation: Seasonal with n=15,000 (verify stratification note).
    """
    np.random.seed(42)
    n = 15000
    period = 12
    t = np.arange(n)

    # Create seasonal pattern
    # 15000 points, period 12 => ~1250 cycles
    seasonality = np.tile(np.arange(period), n // period + 1)[:n]

    # x = trend + seasonality + noise
    x = 0.01 * t + seasonality * 10 + np.random.normal(0, 5, n)

    # Must provide period for numeric time
    result = seasonal_trend_test(x, t, period=period)

    # Should trigger stratification because n > 10000
    # Note: result.computation_mode reflects the overall strategy ('fast' here)
    assert result.computation_mode == 'fast'
    assert result.trend == 'increasing'

    # Check if warnings/notes contain stratification info
    # The code adds to analysis_notes: "Large seasonal dataset: Used stratified sampling..."
    # Note: analysis_notes is a list of strings

    stratification_note_found = False
    for note in result.analysis_notes:
        if "stratified sampling" in note.lower():
            stratification_note_found = True
            break

    assert stratification_note_found, f"Stratification note not found in: {result.analysis_notes}"


def test_heavy_ties():
    """
    Audit Recommendation: Heavy ties (>50% tied values).
    """
    np.random.seed(42)
    n = 6000 # > 5000 to trigger fast path
    t = np.arange(n)

    # 80% of data is 0 to ensure > 50% tied pairs
    # (0.8*n)^2 / n^2 approx 0.64 ratio of tied pairs
    x = np.random.normal(0, 1, n)
    x[:int(0.8*n)] = 0

    # Expect warning about heavy ties
    with pytest.warns(UserWarning, match="Heavy ties detected"):
        result = trend_test(x, t)

    assert result.computation_mode == 'fast'
    # Result should be valid
    assert isinstance(result.slope, float)
