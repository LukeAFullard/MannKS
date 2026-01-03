import numpy as np
from ._stats import _mk_score_and_var_censored, _sens_estimator_unequal_spacing, _sens_estimator_censored

def optimal_block_size(n, acf):
    """
    Calculate optimal block size using a correlation-length heuristic.

    This function estimates the optimal block length for the moving block bootstrap
    by identifying the "correlation length" of the data (the lag at which
    autocorrelation becomes negligible).

    Args:
        n: Sample size
        acf: Autocorrelation function

    Returns:
        block_size: Optimal block length
    """
    # Simple heuristic: block size should be roughly the correlation length
    # Correlation length: first lag where ACF < 0.1
    # If correlation never drops (strong long-range dependence), use the max lag checked.
    corr_length = len(acf)
    found = False
    for i in range(1, len(acf)):
        if np.abs(acf[i]) < 0.1:
            corr_length = i
            found = True
            break

    if not found:
        # If autocorrelation never drops, use a larger block size (e.g. sqrt(n))
        # to capture long-range dependence.
        corr_length = int(np.sqrt(n))

    # Use 2 * correlation length, bounded by sqrt(n)
    block_size = min(2 * corr_length, int(np.sqrt(n)))
    return max(block_size, 3)  # Minimum block size of 3


def _moving_block_bootstrap_indices(n, block_size):
    """
    Generate indices for one moving block bootstrap resample.

    Args:
        n: Length of data
        block_size: Block length

    Returns:
        indices: Array of indices for the bootstrap sample
    """
    if block_size < 1:
        raise ValueError(f"Block size must be at least 1. Got {block_size}.")

    if block_size >= n:
        return np.arange(n) # Cannot bootstrap if block size is entire series

    n_blocks = int(np.ceil(n / block_size))

    # Random starting positions for blocks
    # Max start index is n - block_size
    starts = np.random.randint(0, n - block_size + 1, size=n_blocks)

    # Build bootstrap sample indices
    indices = []
    for start in starts:
        indices.extend(range(start, start + block_size))

    return np.array(indices[:n])


def moving_block_bootstrap(x, block_size):
    """
    Generate one moving block bootstrap resample.

    Args:
        x: Data vector
        block_size: Block length

    Returns:
        x_boot: Bootstrap resample (same length as x)
    """
    n = len(x)
    indices = _moving_block_bootstrap_indices(n, block_size)

    # Handle list input
    if isinstance(x, list):
        return [x[i] for i in indices]

    return np.array(x)[indices]


def block_bootstrap_mann_kendall(x, t, censored, cen_type,
                                 block_size='auto', n_bootstrap=1000,
                                 tau_method='b', mk_test_method='robust'):
    """
    Block bootstrap Mann-Kendall test for autocorrelated data.

    Uses a Detrended Moving Block Bootstrap approach:
    1. Estimate and remove the trend (Sen's slope) to obtain residuals.
    2. Resample residuals in blocks to preserve autocorrelation structure under H0.
    3. Calculate Mann-Kendall statistics on the bootstrapped residuals.

    Note for Censored Data:
    Detrending censored data is heuristic (using the numeric value of the limit).
    While this preserves the 'null' behavior for the test statistic,
    the absolute values of the detrended series may effectively have randomized
    censoring limits. This is generally acceptable for the rank-based Mann-Kendall
    test which tests monotonic trend, but users should be aware of this approximation.

    Args:
        x, t, censored, cen_type: Standard inputs
        block_size: 'auto' or integer
        n_bootstrap: Number of bootstrap resamples
        tau_method, mk_test_method: Pass-through to _mk_score_and_var_censored

    Returns:
        p_boot: Bootstrap p-value
        s_obs: Observed S statistic
        s_boot_dist: Bootstrap distribution of S (for diagnostics)
    """
    n = len(x)

    # Calculate observed statistic
    s_obs, var_s_obs, _, _ = _mk_score_and_var_censored(
        x, t, censored, cen_type,
        tau_method=tau_method,
        mk_test_method=mk_test_method
    )

    # Detrend data under H0 (no trend)
    # We remove the global trend to test the null hypothesis of no trend
    # while preserving the autocorrelation structure.
    if np.any(censored):
        slopes = _sens_estimator_censored(x, t, cen_type)
    else:
        slopes = _sens_estimator_unequal_spacing(x, t)

    median_slope = np.nanmedian(slopes)
    if np.isnan(median_slope):
        median_slope = 0.0

    # Detrend: x_detrended = x - slope * t
    # Note: t might be large timestamps, so centering t is good practice
    t_centered = t - np.median(t)
    x_detrended = x - median_slope * t_centered

    # Determine block size
    # We estimate block size from the RESIDUALS (detrended data) because
    # the trend itself acts like infinite memory and would inflate the block size.
    if block_size == 'auto':
        from ._autocorr import estimate_acf
        acf, _ = estimate_acf(x_detrended)
        block_size = optimal_block_size(n, acf)

    # Bootstrap distribution
    s_boot_dist = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Generate bootstrap indices
        indices = _moving_block_bootstrap_indices(n, block_size)

        # Apply indices to detrended data AND censoring metadata
        x_boot = x_detrended[indices]
        censored_boot = censored[indices]
        cen_type_boot = cen_type[indices]

        # Calculate S for bootstrap sample
        # We use the original time vector t because Mann-Kendall depends on rank(t) vs rank(x)
        s_b, _, _, _ = _mk_score_and_var_censored(
            x_boot, t, censored_boot, cen_type_boot,
            tau_method=tau_method,
            mk_test_method=mk_test_method
        )
        s_boot_dist[b] = s_b

    # Two-sided p-value
    # How many bootstrap S are at least as extreme as observed S?
    p_boot = np.mean(np.abs(s_boot_dist) >= np.abs(s_obs))

    return p_boot, s_obs, s_boot_dist


def block_bootstrap_confidence_intervals(x, t, censored, cen_type,
                                        block_size='auto', n_bootstrap=1000,
                                        alpha=0.05,
                                        **kwargs):
    """
    Bootstrap confidence intervals for Sen's slope with autocorrelated data.

    Uses a 'Pairs Block Bootstrap' approach:
    1. Resamples blocks of (x, t) pairs to preserve both autocorrelation and
       the trend-time relationship.
    2. Calculates Sen's slope for each bootstrapped sample.
    3. Determines CI from the percentile distribution of slopes.

    This method is preferred over residual bootstrap for censored data because
    it avoids the bias introduced by 'reconstructing' censored values from residuals.

    Args:
        **kwargs: Additional arguments passed to _sens_estimator_censored
                  (e.g., method, lt_mult, gt_mult).

    Returns:
        slope: Sen's slope
        lower_ci, upper_ci: Bootstrap confidence intervals
        boot_slopes: Bootstrap distribution (for diagnostics)
    """
    n = len(x)

    # Calculate observed slope
    if np.any(censored):
        slopes = _sens_estimator_censored(x, t, cen_type, **kwargs)
    else:
        slopes = _sens_estimator_unequal_spacing(x, t)
    slope_obs = np.nanmedian(slopes)
    if np.isnan(slope_obs):
        slope_obs = 0.0

    # Determine block size
    # Even for pairs bootstrap, we want the block size to reflect the
    # correlation of the random error component, not the trend.
    if block_size == 'auto':
        from ._autocorr import estimate_acf
        t_centered = t - np.median(t)
        residuals = x - slope_obs * t_centered
        acf, _ = estimate_acf(residuals)
        block_size = optimal_block_size(n, acf)

    # Bootstrap distribution
    boot_slopes = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Generate bootstrap indices
        indices = _moving_block_bootstrap_indices(n, block_size)

        # Resample PAIRS (x, t) to preserve dependence structure and trend
        x_boot = x[indices]
        t_boot = t[indices]
        censored_boot = censored[indices]
        cen_type_boot = cen_type[indices]

        # Critical Fix: Sort by time for correct censored slope calculation.
        # The pairwise logic in _sens_estimator_censored assumes j > i implies later > earlier.
        # Pairs bootstrap can produce unsorted time vectors.
        sort_idx = np.argsort(t_boot)
        x_boot = x_boot[sort_idx]
        t_boot = t_boot[sort_idx]
        censored_boot = censored_boot[sort_idx]
        cen_type_boot = cen_type_boot[sort_idx]

        # Calculate slope for bootstrap sample
        if np.any(censored_boot):
            slopes_b = _sens_estimator_censored(x_boot, t_boot, cen_type_boot, **kwargs)
        else:
            slopes_b = _sens_estimator_unequal_spacing(x_boot, t_boot)

        boot_slope = np.nanmedian(slopes_b)
        boot_slopes[b] = boot_slope

    # Percentile confidence intervals
    lower_ci = np.percentile(boot_slopes, 100 * alpha / 2)
    upper_ci = np.percentile(boot_slopes, 100 * (1 - alpha / 2))

    return slope_obs, lower_ci, upper_ci, boot_slopes
