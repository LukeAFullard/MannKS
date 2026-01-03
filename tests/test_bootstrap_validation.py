
import pytest
import numpy as np
import pandas as pd
from MannKS._bootstrap import moving_block_bootstrap, _moving_block_bootstrap_indices
from MannKS.trend_test import trend_test
from MannKS.seasonal_trend_test import seasonal_trend_test

def test_moving_block_bootstrap_validation():
    """Test that moving_block_bootstrap raises ValueError for invalid block sizes."""
    n = 10
    x = np.arange(n)

    # Valid block sizes should not raise
    try:
        moving_block_bootstrap(x, 1)
        moving_block_bootstrap(x, 5)
    except ValueError:
        pytest.fail("moving_block_bootstrap raised ValueError for valid block_size")

    # Invalid block sizes
    with pytest.raises(ValueError, match="Block size must be at least 1"):
        moving_block_bootstrap(x, 0)

    with pytest.raises(ValueError, match="Block size must be at least 1"):
        moving_block_bootstrap(x, -5)

def test_moving_block_bootstrap_indices_validation():
    """Test that _moving_block_bootstrap_indices raises ValueError for invalid block sizes."""
    n = 10

    # Valid block sizes
    try:
        _moving_block_bootstrap_indices(n, 1)
        _moving_block_bootstrap_indices(n, 5)
    except ValueError:
        pytest.fail("_moving_block_bootstrap_indices raised ValueError for valid block_size")

    # Invalid block sizes
    with pytest.raises(ValueError, match="Block size must be at least 1"):
        _moving_block_bootstrap_indices(n, 0)

    with pytest.raises(ValueError, match="Block size must be at least 1"):
        _moving_block_bootstrap_indices(n, -1)

def test_trend_test_block_size_validation():
    """Test that trend_test propagates the validation error."""
    # Create simple data
    x = np.random.rand(20)
    t = np.arange(20)

    # Should raise ValueError when block_size is invalid and bootstrap is enabled
    # Note: trend_test calls block_bootstrap_mann_kendall which calls _moving_block_bootstrap_indices
    with pytest.raises(ValueError, match="Block size must be at least 1"):
        trend_test(x, t, autocorr_method='block_bootstrap', block_size=0)

    with pytest.raises(ValueError, match="Block size must be at least 1"):
        trend_test(x, t, autocorr_method='block_bootstrap', block_size=-1)

def test_seasonal_trend_test_block_size_validation():
    """Test that seasonal_trend_test propagates the validation error."""
    # Create simple seasonal data
    dates = pd.date_range(start='2020-01-01', periods=24, freq='ME')
    x = np.random.rand(24)

    # Should raise ValueError when block_size is invalid and bootstrap is enabled
    with pytest.raises(ValueError, match="Block size must be at least 1"):
        seasonal_trend_test(x, dates, period=12, season_type='month',
                           autocorr_method='block_bootstrap', block_size=0)

    with pytest.raises(ValueError, match="Block size must be at least 1"):
        seasonal_trend_test(x, dates, period=12, season_type='month',
                           autocorr_method='block_bootstrap', block_size=-1)
