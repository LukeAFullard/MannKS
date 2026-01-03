import numpy as np
import pytest
from MannKS._autocorr import estimate_acf, effective_sample_size, should_apply_correction
from MannKS._bootstrap import moving_block_bootstrap, block_bootstrap_mann_kendall

def generate_ar1_process(n, phi, trend_slope=0.0, noise_sd=1.0):
    """Generate AR(1) process with trend."""
    x = np.zeros(n)
    x[0] = np.random.normal(0, noise_sd)

    for i in range(1, n):
        x[i] = phi * x[i-1] + np.random.normal(0, noise_sd)

    # Add trend
    t = np.arange(n)
    x += trend_slope * t

    return x, t

def test_estimate_acf_random_noise():
    """ACF of white noise should be low for lags > 0."""
    np.random.seed(42)
    n = 1000
    x = np.random.normal(0, 1, n)

    acf, sig_lag = estimate_acf(x, max_lag=10)

    assert acf[0] == 1.0
    # Lag 1 should be small
    assert abs(acf[1]) < 0.1
    # Usually no significant lag for white noise
    assert sig_lag is None or sig_lag > 5

def test_estimate_acf_ar1():
    """ACF of AR(1) should decay."""
    np.random.seed(42)
    x, _ = generate_ar1_process(n=1000, phi=0.8)

    acf, _ = estimate_acf(x, max_lag=10)

    assert acf[0] == 1.0
    assert 0.7 < acf[1] < 0.9  # Should be close to phi=0.8
    assert 0.5 < acf[2] < 0.8  # Should be around phi^2=0.64

def test_effective_sample_size():
    """ESS should be smaller than n for positive autocorrelation."""
    np.random.seed(42)
    x, _ = generate_ar1_process(n=1000, phi=0.5)

    n_eff, acf1 = effective_sample_size(x, method='yue')

    assert n_eff < 1000
    assert 0.4 < acf1 < 0.6

def test_moving_block_bootstrap_structure():
    """Test that moving block bootstrap preserves block structure."""
    x = np.arange(100) # 0, 1, 2, ... 99
    block_size = 10

    x_boot = moving_block_bootstrap(x, block_size)

    assert len(x_boot) == 100

    # Check that at least some adjacent elements are sequential (preserving structure)
    # diff should be 1 for contiguous blocks
    diffs = np.diff(x_boot)
    # In a pure block shuffle of 100 items with block 10, we have 10 blocks.
    # Within each block (9 pairs), diff is 1. So roughly 90% of diffs should be 1.
    seq_count = np.sum(diffs == 1)
    assert seq_count > 80

def test_bootstrap_mann_kendall_no_trend():
    """Test bootstrap MK on AR(1) data with no trend."""
    np.random.seed(123)
    x, t = generate_ar1_process(n=100, phi=0.6, trend_slope=0.0)
    censored = np.zeros(len(x), dtype=bool)
    cen_type = np.array(['not'] * len(x))

    p_boot, s_obs, s_dist = block_bootstrap_mann_kendall(
        x, t, censored, cen_type, block_size=10, n_bootstrap=200
    )

    # With no trend, p-value should typically be non-significant
    # This is probabilistic, but 0.6 AR1 usually creates false trends in standard MK.
    # Bootstrap p should be reasonably high.
    # Just check it runs and returns valid range.
    assert 0 <= p_boot <= 1.0
    assert len(s_dist) == 200

def test_should_apply_correction():
    np.random.seed(42)
    # High auto-corr
    x1, _ = generate_ar1_process(n=100, phi=0.8)
    needs_corr1, _, _ = should_apply_correction(x1)
    assert needs_corr1

    # White noise
    x2 = np.random.normal(0, 1, 100)
    needs_corr2, _, _ = should_apply_correction(x2)
    assert not needs_corr2
