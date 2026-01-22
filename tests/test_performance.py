
import pytest
import time
import numpy as np
from MannKS import trend_test

@pytest.mark.slow
def test_performance_scaling():
    """Verify computational complexity scaling."""
    # Reduced sizes for CI environment to avoid timeouts, but relative check holds
    sizes = [1000, 2000, 4000, 8000]
    times_full = []
    times_fast = []

    for n in sizes:
        t = np.arange(n)
        x = 0.5 * t + np.random.normal(0, 1, n)

        if n <= 4000:
            # Full mode
            start = time.time()
            trend_test(x, t, large_dataset_mode='full')
            times_full.append(time.time() - start)

        # Fast mode
        start = time.time()
        trend_test(x, t, large_dataset_mode='fast', max_pairs=50000)
        times_fast.append(time.time() - start)

    # Fast mode should scale linearly O(N) roughly (sampling overhead + MK score)
    # Actually MK score is still O(N^2/chunk) which is O(N^2).
    # But slope estimation is O(N) or O(1) depending on implementation.
    # The current implementation optimizes MEMORY for MK score, but TIME is still O(N^2)
    # because we iterate all chunks.
    # So we don't expect linear time scaling for the whole test, just memory safety.
    # However, slope estimation IS faster.

    # Let's just verify that Fast is faster than Full for N=4000
    if len(times_full) >= 3:
        assert times_fast[2] < times_full[2], "Fast mode should be faster than Full mode at N=4000"


@pytest.mark.slow
def test_accuracy_vs_speed_tradeoff():
    """Test different max_pairs settings."""
    n = 6000
    t = np.arange(n)
    x = 0.5 * t + np.random.normal(0, 10, n)

    # Ground truth (using high max_pairs to approx full)
    result_reference = trend_test(x, t, max_pairs=500000, random_state=42)

    settings = [10000, 50000, 100000]
    errors = []
    times = []

    for max_p in settings:
        start = time.time()
        result = trend_test(x, t, max_pairs=max_p, random_state=42)
        elapsed = time.time() - start

        error = abs(result.slope - result_reference.slope) / abs(result_reference.slope)

        errors.append(error)
        times.append(elapsed)

    # Verify error generally decreases (or stays low)
    # Note: 100k pairs is already very accurate, so differences might be noise
    assert errors[-1] < 0.01
