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


def effective_sample_size(x, method='yue'):
    """
    Calculate effective sample size accounting for autocorrelation.

    Args:
        x: Data vector
        method: 'yue' (Yue & Wang 2004) or 'bayley' (Bayley & Hammersley 1946)

    Returns:
        n_eff: Effective sample size
        acf1: Lag-1 autocorrelation
    """
    x = np.asarray(x)
    # Remove NaNs for ESS calculation
    x = x[~np.isnan(x)]
    n = len(x)

    if n < 3:
        return n, 0.0

    acf, _ = estimate_acf(x, max_lag=min(n//4, 50))
    acf1 = acf[1] if len(acf) > 1 else 0.0

    if method == 'yue':
        # Yue & Wang (2004) formula
        if np.abs(acf1) < 1e-10:
            return n, acf1

        # Sum of autocorrelations (with exponential decay assumption/actual ACF)
        # Yue & Wang typically use the actual sample ACF values up to some lag.
        # The report implementation uses sum of acf[k] * (n-k)/n
        rho_sum = 0
        for k in range(1, len(acf)):
            rho_sum += (1 - k/n) * acf[k]

        # Guard against negative n_eff (possible with strong negative autocorr)
        denom = 1 + 2 * rho_sum
        if denom <= 0:
             n_eff = n # Fallback
        else:
             n_eff = n / denom
    else:
        # Simple Bayley-Hammersley formula (often approx as (1-rho)/(1+rho))
        # Strictly Bayley-Hammersley is more complex sum, but this is the common AR(1) approx
        n_eff = n * (1 - acf1) / (1 + acf1)

    return max(n_eff, 3), acf1  # Ensure minimum of 3


def should_apply_correction(x, threshold=0.1):
    """
    Determine if autocorrelation correction is needed.

    Args:
        x: Data vector
        threshold: ACF threshold for correction (default 0.1)

    Returns:
        needs_correction: Boolean
        acf1: Lag-1 autocorrelation
        n_eff: Effective sample size
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]

    if len(x) < 3:
        return False, 0.0, len(x)

    acf, significant_lag = estimate_acf(x)
    acf1 = acf[1] if len(acf) > 1 else 0.0
    n_eff, _ = effective_sample_size(x)

    needs_correction = (np.abs(acf1) > threshold or significant_lag is not None)

    return needs_correction, acf1, n_eff
