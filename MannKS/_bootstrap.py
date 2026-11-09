"""
Block bootstrap procedures for serially correlated data.

* Hypothesis test: the series is detrended with the chosen slope estimator,
  the residuals are resampled in circular blocks (Politis & Romano 1992) to
  build the null distribution of S while preserving short-range dependence,
  and the two-sided p-value is ``(1 + #{|S*| >= |S|}) / (B + 1)``.
* Confidence interval for the slope: circular block bootstrap of
  (time, value, censoring) triples ("pairs" bootstrap), percentile interval.

Block length (``block_size='auto'``) is twice the first lag at which the
autocorrelation of the *detrended* series falls below 0.1, bounded to
[3, sqrt(n)].  This is a simple heuristic, not an optimal block-length
selector.
"""
import numpy as np
from ._stats import (_mk_score_and_var_censored, _sens_estimator_adaptive,
                     _sens_estimator_censored_adaptive)


def _rng(random_state):
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def optimal_block_size(n, acf):
    """
    Heuristic block length: ``2 * L`` where L is the first lag with
    ``|acf| < 0.1`` (or sqrt(n) if none), bounded to [3, sqrt(n)].

    Args:
        n: Sample size
        acf: Autocorrelation function of the (detrended) series
    """
    corr_length = None
    for i in range(1, len(acf)):
        if np.abs(acf[i]) < 0.1:
            corr_length = i
            break
    if corr_length is None:
        corr_length = int(np.sqrt(n))
    block_size = min(2 * corr_length, int(np.sqrt(n)))
    return max(block_size, 3)


def _moving_block_bootstrap_indices(n, block_size, rng=None):
    """
    Indices for one circular block bootstrap resample of length n.
    Blocks may start anywhere and wrap around the end of the series, so every
    observation is equally likely to be drawn.
    """
    if block_size < 1:
        raise ValueError(f"Block size must be at least 1. Got {block_size}.")
    rng = _rng(rng)
    block_size = min(int(block_size), n)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks).astype(np.intp, copy=False)
    idx = (starts[:, None] + np.arange(block_size, dtype=np.intp)[None, :]) % n
    return idx.ravel()[:n].astype(np.intp, copy=False)


def moving_block_bootstrap(x, block_size, random_state=None):
    """One circular block bootstrap resample of ``x``."""
    n = len(x)
    indices = _moving_block_bootstrap_indices(n, block_size, random_state)
    if isinstance(x, list):
        return [x[i] for i in indices]
    return np.asarray(x)[indices]


def _slope(x, t, censored, cen_type, method='ats', lt_mult=0.5, gt_mult=1.1,
           max_pairs=None, random_state=None):
    """Point estimate of the trend slope with the requested estimator.
    ``max_pairs`` (fast mode) samples pairs instead of using all of them."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    if np.any(censored):
        if method == 'ats':
            from ._ats import ats_slope
            return ats_slope(t, x, censored, cen_type=cen_type, var_s=0.0,
                             max_pairs=max_pairs, random_state=random_state,
                             compute_intercept=False)['beta']
        m = 'drop_ambiguous' if method in ('unbiased', 'nan') else method
        slopes = _sens_estimator_censored_adaptive(x, t, np.asarray(cen_type), lt_mult, gt_mult, m,
                                                   max_pairs=max_pairs, random_state=random_state)
    else:
        slopes = _sens_estimator_adaptive(x, t, max_pairs=max_pairs, random_state=random_state)
    if len(slopes) == 0 or not np.any(~np.isnan(slopes)):
        return np.nan
    return np.nanmedian(slopes)


def _resample_seed(random_state, k):
    """Seed for pair sampling in bootstrap resample k (deterministic)."""
    base = random_state if isinstance(random_state, (int, np.integer)) else 0
    return int(base) + 1 + k


def detrended_block_size(x, t, censored, cen_type, method='ats', lt_mult=0.5, gt_mult=1.1,
                         max_pairs=None, random_state=None):
    """Block length chosen from the ACF of the detrended series."""
    from ._autocorr import estimate_acf
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    b = _slope(x, t, censored, cen_type, method, lt_mult, gt_mult, max_pairs, random_state)
    if not np.isfinite(b):
        b = 0.0
    acf, _ = estimate_acf(x - b * (t - np.median(t)))
    return optimal_block_size(len(x), acf)


def block_bootstrap_mann_kendall(x, t, censored, cen_type,
                                 block_size='auto', n_bootstrap=1000,
                                 tau_method='b', mk_test_method='robust',
                                 tie_break_method='robust', random_state=None,
                                 sens_slope_method='ats', lt_mult=0.5, gt_mult=1.1,
                                 max_pairs=None):
    """
    Block bootstrap Mann-Kendall test for autocorrelated data.

    1. Estimate the trend with ``sens_slope_method`` and detrend.
    2. Resample the residuals (with their censoring flags) in circular blocks.
    3. Compute S for each resample against the original time order.

    Note on censored data: detrending shifts each detection limit by the
    removed trend, so resampled censored residuals carry limits that vary over
    time.  This is an approximation; the rank-based S is affected only where
    such limits overlap uncensored residuals.

    Returns:
        p_boot: two-sided bootstrap p-value, (1 + #{|S*| >= |S|}) / (B + 1)
        s_obs: observed S
        s_boot_dist: bootstrap distribution of S
    """
    rng = _rng(random_state)
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    cen_type = np.asarray(cen_type)
    n = len(x)

    s_obs, _, _, _ = _mk_score_and_var_censored(
        x, t, censored, cen_type, tau_method=tau_method,
        mk_test_method=mk_test_method, tie_break_method=tie_break_method)

    b = _slope(x, t, censored, cen_type, sens_slope_method, lt_mult, gt_mult,
               max_pairs, random_state if isinstance(random_state, (int, np.integer)) else None)
    if not np.isfinite(b):
        b = 0.0
    x_detrended = x - b * (t - np.median(t))

    if block_size == 'auto':
        from ._autocorr import estimate_acf
        acf, _ = estimate_acf(x_detrended)
        block_size = optimal_block_size(n, acf)

    s_boot_dist = np.zeros(n_bootstrap)
    for k in range(n_bootstrap):
        idx = _moving_block_bootstrap_indices(n, block_size, rng)
        s_boot_dist[k] = _mk_score_and_var_censored(
            x_detrended[idx], t, censored[idx], cen_type[idx], tau_method=tau_method,
            mk_test_method=mk_test_method, tie_break_method=tie_break_method)[0]

    p_boot = (1 + np.sum(np.abs(s_boot_dist) >= np.abs(s_obs))) / (n_bootstrap + 1)
    return p_boot, s_obs, s_boot_dist


def block_bootstrap_confidence_intervals(x, t, censored, cen_type,
                                         block_size='auto', n_bootstrap=1000,
                                         alpha=0.05, random_state=None,
                                         method='ats', lt_mult=0.5, gt_mult=1.1,
                                         max_pairs=None):
    """
    Percentile confidence interval for the trend slope by circular block
    bootstrap of (time, value, censoring) triples.

    Each resample keeps observations together with their times and censoring
    status (so detection limits are never "reconstructed"); duplicated
    observations keep their multiplicity, pairs with equal times are ignored
    by the slope estimators.

    Args:
        method: slope estimator ('ats', 'lwp', 'drop_ambiguous').

    Returns:
        slope, lower_ci, upper_ci, boot_slopes
    """
    rng = _rng(random_state)
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    cen_type = np.asarray(cen_type)
    n = len(x)

    seed0 = random_state if isinstance(random_state, (int, np.integer)) else None
    slope_obs = _slope(x, t, censored, cen_type, method, lt_mult, gt_mult, max_pairs, seed0)

    if block_size == 'auto':
        block_size = detrended_block_size(x, t, censored, cen_type, method, lt_mult, gt_mult,
                                          max_pairs, seed0)

    boot_slopes = np.full(n_bootstrap, np.nan)
    for k in range(n_bootstrap):
        idx = np.sort(_moving_block_bootstrap_indices(n, block_size, rng))
        boot_slopes[k] = _slope(x[idx], t[idx], censored[idx], cen_type[idx], method, lt_mult, gt_mult,
                                max_pairs, _resample_seed(random_state, k) if max_pairs else None)

    ok = boot_slopes[np.isfinite(boot_slopes)]
    if len(ok) == 0:
        return slope_obs, np.nan, np.nan, boot_slopes
    lower_ci = np.percentile(ok, 100 * alpha / 2)
    upper_ci = np.percentile(ok, 100 * (1 - alpha / 2))
    return slope_obs, lower_ci, upper_ci, boot_slopes
