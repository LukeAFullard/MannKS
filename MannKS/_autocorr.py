import numpy as np
from scipy.stats import norm

def estimate_acf(x, max_lag=None):
    """
    Estimate autocorrelation function.

    Args:
        x: Data vector
        max_lag: Maximum lag to compute (default: min(n/4, 50))

    Returns:
        acf: Array of autocorrelation coefficients
        significant_lag: First lag where ACF crosses significance threshold
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 4, 50)

    x_centered = x - np.nanmean(x)

    # Handle NaN values if present by masking or simple omission for dot product
    # For robust calculation with NaNs, we should probably mask.
    # For now, assuming input x has been pre-processed/aggregated or we use nan-ignoring logic.
    # The Mann-Kendall test functions usually handle NaNs by removal or specialized logic.
    # Here, let's assume valid data or use masking.

    # Using masked arrays to handle potential NaNs safely
    x_masked = np.ma.masked_invalid(x_centered)

    if x_masked.count() < 2:
        return np.zeros(max_lag + 1), None

    c0 = np.ma.dot(x_masked, x_masked) / n

    if c0 == 0:
        return np.zeros(max_lag + 1), None

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0

    for lag in range(1, max_lag + 1):
        # Slice and calculate dot product
        # Ensure we are using the same centered mean
        c_lag = np.ma.dot(x_masked[:-lag], x_masked[lag:]) / n
        acf[lag] = c_lag / c0

    # Significance threshold (95% CI under H0: no autocorr)
    threshold = 1.96 / np.sqrt(n)

    # Find first significant lag
    significant_lag = None
    for lag in range(1, len(acf)):
        if np.abs(acf[lag]) > threshold:
            significant_lag = lag
            break

    return acf, significant_lag


def detrend(x, t, max_pairs=None, random_state=None):
    """
    Residuals after removing the Sen's slope trend, ``x - b * (t - median(t))``.
    Autocorrelation must be estimated on detrended data: a monotonic trend
    otherwise produces large positive sample autocorrelations.
    """
    from ._stats import _sens_estimator_adaptive
    from ._large_dataset import MAX_EXACT_N, DEFAULT_MAX_PAIRS
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(x) < 3:
        return x - np.nanmean(x)
    effective_max_pairs = max_pairs
    if effective_max_pairs is None and len(x) > MAX_EXACT_N:
        effective_max_pairs = DEFAULT_MAX_PAIRS
    slopes = _sens_estimator_adaptive(x, t, max_pairs=effective_max_pairs, random_state=random_state)
    b = np.nanmedian(slopes) if len(slopes) else 0.0
    if not np.isfinite(b):
        b = 0.0
    return x - b * (t - np.median(t))


def effective_sample_size(x, t=None, method='yue', max_pairs=None, random_state=None):
    """
    Effective sample size accounting for serial correlation.

    Args:
        x: Data vector (assumed to be in time order).
        t: Time vector.  If given, the series is detrended with Sen's slope
            before the autocorrelation is estimated (Yue & Wang 2004).
        method: 'yue' (Yue & Wang 2004 variance-inflation factor
            ``1 + 2 * sum_k (1 - k/n) rho_k``, using only lags whose sample
            autocorrelation is significant at the 5% level) or 'bayley'
            (AR(1) approximation ``n (1 - rho_1) / (1 + rho_1)``).
        max_pairs: Optional cap on pairwise slopes evaluated during detrending.
        random_state: Optional seed for sampled pairwise slopes.

    Returns:
        (n_eff, acf1): effective sample size, capped to the range [3, n] so
        that the correction can only inflate Var(S), and the lag-1
        autocorrelation of the (detrended) series.

    Note:
        Like all variance-correction methods, this under-corrects for strong
        autocorrelation in short series (e.g. AR(1) with rho = 0.6, n = 60
        gives a rejection rate of ~0.19 at alpha = 0.05 under H0).
    """
    x = np.asarray(x, dtype=float)
    if t is not None:
        t = np.asarray(t, dtype=float)
        keep = ~np.isnan(x)
        x = detrend(x[keep], t[keep], max_pairs=max_pairs, random_state=random_state)
    else:
        x = x[~np.isnan(x)]
    n = len(x)

    if n < 3:
        return n, 0.0

    acf, _ = estimate_acf(x, max_lag=min(n // 4, 50))
    acf1 = acf[1] if len(acf) > 1 else 0.0

    if method == 'yue':
        k = np.arange(1, len(acf), dtype=np.intp)
        rho = acf[1:]
        rho = np.where(np.abs(rho) > 1.96 / np.sqrt(n), rho, 0.0)
        factor = 1 + 2 * np.sum((1 - k / n) * rho)
        n_eff = n / factor if factor > 0 else n
    else:
        n_eff = n * (1 - acf1) / (1 + acf1)

    return float(min(max(n_eff, 3), n)), acf1


def should_apply_correction(x, t=None, threshold=0.1, max_pairs=None, random_state=None):
    """
    Decide whether serial-correlation correction is needed.

    The series is detrended first (when ``t`` is given).  Correction is
    recommended when the lag-1 autocorrelation of the detrended series exceeds
    both ``threshold`` and its 5% significance bound ``1.96 / sqrt(n)``.

    Returns:
        (needs_correction, acf1, n_eff)
    """
    x = np.asarray(x, dtype=float)
    if t is not None:
        t = np.asarray(t, dtype=float)
        keep = ~np.isnan(x)
        r = detrend(x[keep], t[keep], max_pairs=max_pairs, random_state=random_state)
    else:
        r = x[~np.isnan(x)]

    if len(r) < 3:
        return False, 0.0, len(r)

    acf, _ = estimate_acf(r)
    acf1 = acf[1] if len(acf) > 1 else 0.0
    n_eff, _ = effective_sample_size(r)

    needs_correction = abs(acf1) > max(threshold, 1.96 / np.sqrt(len(r)))
    return needs_correction, acf1, n_eff
