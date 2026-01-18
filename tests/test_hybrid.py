
import pytest
import numpy as np
import pandas as pd
from MannKS import segmented_trend_test

def test_hybrid_segmented_trend_basic():
    """
    Test the Hybrid segmented trend test on simple synthetic data.
    """
    np.random.seed(42)
    n = 50
    t = np.arange(n)
    # create data with 1 breakpoint at t=25
    # slope 1 then slope -1
    y = np.zeros(n)
    y[:25] = t[:25] * 1.0
    y[25:] = y[24] + (t[25:] - 24) * (-1.0)

    # Add noise
    y += np.random.normal(0, 0.1, n)

    # Run test
    # n_breakpoints=1 fixed
    result = segmented_trend_test(y, t, n_breakpoints=1, max_breakpoints=3)

    assert result.n_breakpoints == 1
    assert len(result.breakpoints) == 1
    # Breakpoint should be around 24-25
    assert 20 < result.breakpoints[0] < 30

    # Check segments
    assert len(result.segments) == 2
    seg1 = result.segments.iloc[0]
    seg2 = result.segments.iloc[1]

    # Slopes should be approx 1 and -1
    assert np.isclose(seg1['slope'], 1.0, atol=0.2)
    assert np.isclose(seg2['slope'], -1.0, atol=0.2)

def test_hybrid_segmented_trend_search():
    """
    Test that the Hybrid method can find the optimal number of breakpoints.
    """
    np.random.seed(42)
    n = 50
    t = np.arange(n)
    # Simple linear trend (0 breakpoints)
    y = t * 0.5 + np.random.normal(0, 0.1, n)

    result = segmented_trend_test(y, t, n_breakpoints=None, max_breakpoints=3)

    assert result.n_breakpoints == 0
    assert len(result.breakpoints) == 0
    assert len(result.segments) == 1
    assert np.isclose(result.segments.iloc[0]['slope'], 0.5, atol=0.1)

def test_hybrid_segmented_input_dataframe():
    """Test passing a dataframe."""
    df = pd.DataFrame({
        'val': np.arange(20),
        'time': np.arange(20)
    })

    result = segmented_trend_test(df[['val']], df['time'].values, n_breakpoints=0)
    assert result.n_breakpoints == 0
    assert np.isclose(result.segments.iloc[0]['slope'], 1.0, atol=0.1)
