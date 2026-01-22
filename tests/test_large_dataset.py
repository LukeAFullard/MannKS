
import pytest
import numpy as np
import pandas as pd
from MannKS import trend_test, prepare_censored_data
from MannKS._large_dataset import fast_sens_slope, detect_size_tier

def test_fast_sens_slope_accuracy():
    """Verify fast mode matches full mode within error bounds."""
    np.random.seed(42)
    n = 1000
    t = np.arange(n)
    true_slope = 0.5
    x = true_slope * t + np.random.normal(0, 10, n)

    # Full calculation
    from MannKS._stats import _sens_estimator_unequal_spacing
    slopes_full = _sens_estimator_unequal_spacing(x, t)
    slope_full = np.median(slopes_full)

    # Fast calculation (using small n but forcing limit to test logic)
    # n=1000 -> 500k pairs. max_pairs=10000 -> subsampling.
    slopes_fast = fast_sens_slope(x, t, max_pairs=10000, random_state=42)
    slope_fast = np.median(slopes_fast)

    # Should be within 5% of true slope (small sample + noise)
    assert abs(slope_fast - true_slope) / true_slope < 0.05
    # Should be within 5% of full calculation
    assert abs(slope_fast - slope_full) / abs(slope_full) < 0.05


def test_tier_detection():
    """Test automatic tier selection."""
    # Small dataset
    info = detect_size_tier(1000)
    assert info['tier'] == 1
    assert info['strategy'] == 'full'

    # Medium dataset
    info = detect_size_tier(10000)
    assert info['tier'] == 2
    assert info['strategy'] == 'fast'

    # Large dataset
    info = detect_size_tier(250000)
    assert info['tier'] == 3
    assert info['strategy'] == 'aggregate'


def test_backwards_compatibility():
    """Ensure small datasets behave identically to v0.4.1 (Full Mode)."""
    np.random.seed(42)
    n = 100
    t = np.arange(n)
    x = 0.5 * t + np.random.normal(0, 1, n)

    # Should use full mode automatically
    result = trend_test(x, t)

    assert result.computation_mode == 'full'
    assert result.pairs_used is None
    assert result.trend == 'increasing'
    assert result.h == True


def test_user_mode_override():
    """Test user can override automatic detection."""
    n = 10000
    t = np.arange(n)
    x = t

    # Force full mode (might be slow but shouldn't crash with 10k due to chunking)
    result_full = trend_test(x, t, large_dataset_mode='full')
    assert result_full.computation_mode == 'full'

    # Force fast mode
    result_fast = trend_test(x, t, large_dataset_mode='fast')
    assert result_fast.computation_mode == 'fast'
    assert result_fast.pairs_used is not None


def test_reproducibility():
    """Test random_state ensures reproducible results."""
    n = 6000 # Just enough to trigger fast mode
    t = np.arange(n)
    x = 0.5 * t + np.random.normal(0, 1, n)

    result1 = trend_test(x, t, random_state=42)
    result2 = trend_test(x, t, random_state=42)
    result3 = trend_test(x, t, random_state=99)

    # Same seed = same results
    assert result1.slope == result2.slope
    assert result1.lower_ci == result2.lower_ci

    # Different seed = likely different (but close) results
    assert result1.slope != result3.slope
    assert abs(result1.slope - result3.slope) < 0.01 * abs(result1.slope)


def test_censored_fast_mode():
    """Test fast mode with censored data."""
    n = 6000
    t = np.arange(n)
    x = 0.5 * t + np.random.normal(0, 10, n)

    # Add censoring
    censored = x < 10
    x[censored] = 10

    x_str = [f'<{val}' if c else val for val, c in zip(x, censored)]
    data = prepare_censored_data(x_str)

    result = trend_test(data, t, random_state=42)

    assert result.computation_mode == 'fast'
    assert result.slope > 0.4  # Should still detect trend
    assert result.h == True
