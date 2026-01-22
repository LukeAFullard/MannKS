import numpy as np
import warnings
from scipy.stats import norm, rankdata, kendalltau

# --- Module-level Constants ---
DEFAULT_LT_MULTIPLIER = 0.5  # Half detection limit for left-censored
DEFAULT_GT_MULTIPLIER = 1.1  # 110% detection limit for right-censored
EPSILON = 1e-10

def _rle_lengths(a):
    """
    Calculates the lengths of runs of equal values in an array.
    Equivalent to R's `rle(x)$lengths`.
    """
    if len(a) == 0:
        return np.array([], dtype=int)
    y = a[1:] != a[:-1]
    i = np.append(np.where(y), len(a) - 1)
    return np.diff(np.append(-1, i))


def _get_min_positive_diff(arr):
    """Calculates the minimum positive difference between unique sorted values."""
    # Ensure array is unique and sorted, which is a prerequisite.
    if len(arr) <= 1:
        return 0.0
    diffs = np.diff(arr)
    pos_diffs = diffs[diffs > 0]
    return np.min(pos_diffs) if len(pos_diffs) > 0 else 0.0


def _mk_score_and_var_censored(x, t, censored, cen_type, tau_method='b', mk_test_method='robust', tie_break_method='robust'):
    """
    Calculates the Mann-Kendall S statistic and its variance for censored data.
    """
    x = np.asarray(x)
    t = np.asarray(t)
    censored = np.asarray(censored)
    cen_type = np.asarray(cen_type)

    # 1. Prepare inputs
    xx = x.copy()
    cx = censored.copy().astype(bool)

    if mk_test_method == 'lwp' and np.any(cen_type == 'gt'):
        gt_mask = cen_type == 'gt'
        max_gt_val = xx[gt_mask].max() + 0.1
        xx[gt_mask] = max_gt_val
        cx[gt_mask] = False

    # Time is treated as uncensored
    yy = rankdata(t, method='ordinal')
    cy = np.zeros_like(yy, dtype=bool)
    n = len(xx)

    MAX_SAFE_N = 46340
    if n > MAX_SAFE_N:
        # We can relax this warning since chunking (O(N) memory) and
        # fast path (O(N log N) time) handle large datasets efficiently.
        # However, for pure O(N^2) paths (e.g. censored data without chunking override),
        # this remains a relevant caution regarding computation time.
        pass

    if n < 2:
        return 0, 0, 0, 0

    # 3. Calculate delx and dely to break ties
    unique_xx = np.unique(xx)
    min_diff_x = _get_min_positive_diff(unique_xx)
    delx = min_diff_x / (1000.0 if tie_break_method == 'lwp' else 2.0) if min_diff_x > 0 else 1.0

    unique_yy = np.unique(yy)
    min_diff_y = _get_min_positive_diff(unique_yy)
    dely = min_diff_y / (1000.0 if tie_break_method == 'lwp' else 2.0) if min_diff_y > 0 else 1.0

    dupx = xx - delx * cx
    dupy = yy - dely * cy

    # 4. S-statistic calculation using chunking for large n
    chunk_size = 2000 # Tunable parameter
    kenS = 0
    tt = 0
    uu = 0

    use_fast_path = (n > 5000 and not np.any(cx) and mk_test_method != 'lwp')
    use_chunking = n > 5000 and not use_fast_path

    if use_fast_path:
        # FAST PATH: Uncensored large data using O(N log N) algorithm
        # Audit: Verified O(N log N) complexity implementation
        # We rely on scipy.stats.kendalltau which is O(N log N).
        # To replicate the existing behavior for tied timestamps (treating them as
        # ordinal/sequential based on array order), we sort x by t stable-ly
        # and compare against an ordinal time vector (0..N-1).

        # 1. Sort x by t (stable)
        # Note: yy is rankdata(t, 'ordinal') which is effectively 1..N permutation
        # matching the sort order. So sorting by yy is equivalent to sorting by t.
        sort_idx = np.argsort(yy, kind='stable')
        x_sorted = xx[sort_idx]
        t_ordinal = np.arange(n)

        # 2. Calculate Tau-b
        res = kendalltau(x_sorted, t_ordinal)
        tau_b = res.statistic
        if np.isnan(tau_b): tau_b = 0

        # 3. Recover S from Tau-b
        # Calculate ties in x
        _, c_x = np.unique(x_sorted, return_counts=True)
        tt = np.sum(c_x * (c_x - 1)) // 2

        # Ties in t_ordinal are 0 (by definition of ordinal ranking)
        uu = 0

        n_pairs = n * (n - 1) // 2

        # Audit validation for heavy ties
        if n_pairs > 0 and (float(tt) / float(n_pairs)) > 0.5:
            warnings.warn(
                "Heavy ties detected (>50% tied pairs). Fast MK score approximation "
                "may have minor rounding errors. Consider using large_dataset_mode='full' "
                "for exact results.",
                UserWarning
            )

        # For Tau-b, denom = sqrt((pairs-tt)(pairs-uu))
        # Since uu=0, denom = sqrt((pairs-tt)*pairs)
        denom_b = np.sqrt(float(n_pairs - tt) * float(n_pairs))

        kenS = int(round(tau_b * denom_b)) # v0.5.0 Audit: Verified S recovery from Tau-b

    elif use_chunking:
        # Loop over chunks of i (rows)
        for start_i in range(0, n, chunk_size):
            end_i = min(start_i + chunk_size, n)

            # chunk_dupx: shape (chunk_size,)
            chunk_dupx = dupx[start_i:end_i]
            chunk_dupy = dupy[start_i:end_i]
            chunk_cx = cx[start_i:end_i]
            chunk_cy = cy[start_i:end_i]

            # Broadcast against full arrays
            diffx = dupx[np.newaxis, :] - chunk_dupx[:, np.newaxis]
            diffy = dupy[np.newaxis, :] - chunk_dupy[:, np.newaxis]

            # Create mask for i < j
            row_indices = np.arange(start_i, end_i)[:, np.newaxis]
            col_indices = np.arange(n)[np.newaxis, :]
            triu_mask = row_indices < col_indices

            if not np.any(triu_mask):
                continue

            signyx = np.sign(diffy * diffx)

            diffcx = cx[np.newaxis, :].astype(int) - chunk_cx[:, np.newaxis].astype(int)
            cix = np.sign(diffcx) * np.sign(diffx)
            cix[cix <= 0] = 0
            signyx *= (1 - cix)

            xplus = (cx[np.newaxis, :].astype(int) + chunk_cx[:, np.newaxis].astype(int))
            xplus[xplus <= 1] = 0
            xplus[xplus > 1] = 1

            # Apply triu mask to select valid pairs
            valid_signyx = signyx * triu_mask
            valid_xplus = xplus * triu_mask

            kenS += np.sum(valid_signyx * (1 - valid_xplus))

            if tau_method != 'a':
                # Term 1: Ties in x
                term1 = (1 - np.abs(np.sign(diffx))) * triu_mask
                tt += np.sum(term1)

                # Term 2: cix sum
                cix_valid = cix * triu_mask
                tt += np.sum(cix_valid)

                # Term 3: tplus
                tplus = xplus * np.abs(np.sign(diffx))
                tplus_valid = tplus * triu_mask
                tt += np.sum(tplus_valid)

                # uu: ties in y
                diffcy = cy[np.newaxis, :].astype(int) - chunk_cy[:, np.newaxis].astype(int)
                ciy = np.sign(diffcy) * np.sign(diffy)
                ciy[ciy <= 0] = 0

                term1_y = (1 - np.abs(np.sign(diffy))) * triu_mask
                uu += np.sum(term1_y)

                ciy_valid = ciy * triu_mask
                uu += np.sum(ciy_valid)

                yplus = (cy[np.newaxis, :].astype(int) + chunk_cy[:, np.newaxis].astype(int))
                yplus[yplus <= 1] = 0
                yplus[yplus > 1] = 1
                uplus = yplus * np.abs(np.sign(diffy))
                uplus_valid = uplus * triu_mask
                uu += np.sum(uplus_valid)

    else:
        # Original vectorized implementation
        diffx = dupx[:, np.newaxis] - dupx
        diffy = dupy[:, np.newaxis] - dupy
        signyx = np.sign(diffy * diffx)

        diffcx = cx[:, np.newaxis].astype(int) - cx.astype(int)
        cix = np.sign(diffcx) * np.sign(diffx)
        cix[cix <= 0] = 0
        signyx *= (1 - cix)

        xplus = (cx[:, np.newaxis].astype(int) + cx.astype(int))
        xplus[xplus <= 1] = 0
        xplus[xplus > 1] = 1
        tplus = xplus * np.abs(np.sign(diffx))

        itot = np.sum(np.triu(signyx * (1 - xplus), k=1))
        kenS = itot

        if tau_method != 'a':
            # tt
            tt = (np.sum(1 - np.abs(np.sign(diffx))) - n) / 2.0
            tt += np.sum(cix) / 2.0
            tt += np.sum(tplus) / 2.0

            # uu
            diffcy = cy[:, np.newaxis].astype(int) - cy.astype(int)
            ciy = np.sign(diffcy) * np.sign(diffy)
            ciy[ciy <= 0] = 0
            uu = (np.sum(1 - np.abs(np.sign(diffy))) - n) / 2.0
            uu += np.sum(ciy) / 2.0
            yplus = (cy[:, np.newaxis].astype(int) + cy.astype(int))
            yplus[yplus <= 1] = 0
            yplus[yplus > 1] = 1
            uplus = yplus * np.abs(np.sign(diffy))
            uu += np.sum(uplus) / 2.0

    # 5. D (denominator) calculation for Tau
    J = n * (n - 1) / 2.0
    if tau_method == 'a':
        D = J
    else: # Default to Tau-b
        tau_denom = np.sqrt(J - tt) * np.sqrt(J - uu)
        D = tau_denom


    # 6. Variance Calculation (adapted from NADA::cenken)
    varS = n * (n - 1) * (2 * n + 5) / 18.0

    intg = np.arange(1, n + 1)

    dorder_x = np.argsort(dupx)
    dxx = dupx[dorder_x]
    dcx = cx[dorder_x]

    dorder_y = np.argsort(dupy)
    dyy = dupy[dorder_y]
    dcy = cy[dorder_y]

    # NADA STATISTICAL NOTE:
    # Term 1: delc - Correction for ties between any pair of values (censored
    #                or uncensored).
    # Term 2: deluc - Correction for ties between an uncensored value and a
    #                 censored value.
    # Term 3: delu - Correction for ties between two censored values.

    # delc: Correction for all ties.
    tmpx = dxx - intg * (1 - dcx) * delx
    tmpy = dyy - intg * (1 - dcy) * dely
    rxlng = _rle_lengths(rankdata(tmpx, method='ordinal'))
    rylng = _rle_lengths(rankdata(tmpy, method='ordinal'))

    def var_adj_term(lng):
        lng_vals, lng_counts = np.unique(lng, return_counts=True)
        t1 = np.sum(lng_counts * lng_vals * (lng_vals - 1) * (2 * lng_vals + 5))
        t2 = np.sum(lng_counts * lng_vals * (lng_vals - 1) * (lng_vals - 2))
        t3 = np.sum(lng_counts * lng_vals * (lng_vals - 1))
        return t1, t2, t3

    x1, x2, x3 = var_adj_term(rxlng)
    y1, y2, y3 = var_adj_term(rylng)

    term2 = (x2 * y2) / (9.0 * n * (n - 1) * (n - 2)) if n > 2 else 0
    term3 = (x3 * y3) / (2.0 * n * (n - 1))
    delc = (x1 + y1) / 18.0 - term2 - term3

    # deluc: Correction for ties between uncensored and censored values.
    x4 = x3
    y4 = y3
    tmpx_uc = intg * dcx - 1
    tmpx_uc[tmpx_uc < 0] = 0
    nrxlng_uc = np.sum(tmpx_uc)
    x1_uc = nrxlng_uc * 2 * 1 * (2 * 2 + 5)
    x2_uc = 0
    x3_uc = nrxlng_uc * 2 * 1

    tmpy_uc = intg * dcy - 1
    tmpy_uc[tmpy_uc < 0] = 0
    nrylng_uc = np.sum(tmpy_uc)
    y1_uc = nrylng_uc * 2 * 1 * (2 * 2 + 5)
    y2_uc = 0
    y3_uc = nrylng_uc * 2 * 1

    term2_uc = (x2_uc * y2_uc) / (9.0 * n * (n - 1) * (n - 2)) if n > 2 else 0
    term3_uc = (x3_uc * y3_uc) / (2.0 * n * (n - 1))
    deluc = (x1_uc + y1_uc) / 18.0 - term2_uc - term3_uc - (x4 + y4)

    # delu: Correction for ties between two censored values.
    dxx_u = dxx - intg * dcx * delx
    dyy_u = dyy - intg * dcy * dely
    rxlng_u = _rle_lengths(rankdata(dxx_u, method='ordinal'))
    rylng_u = _rle_lengths(rankdata(dyy_u, method='ordinal'))
    x1_u, x2_u, x3_u = var_adj_term(rxlng_u)
    y1_u, y2_u, y3_u = var_adj_term(rylng_u)

    term2_u = (x2_u * y2_u) / (9.0 * n * (n - 1) * (n - 2)) if n > 2 else 0
    term3_u = (x3_u * y3_u) / (2.0 * n * (n - 1))
    delu = (x1_u + y1_u) / 18.0 - term2_u - term3_u

    varS = varS - delc - deluc - delu

    if abs(D) > EPSILON:
        Tau = kenS / D
    else:
        Tau = 0
        warnings.warn("Denominator near zero in Tau calculation", UserWarning)

    return kenS, varS, D, Tau


def _z_score(s, var_s):
    if np.isnan(s) or np.isnan(var_s):
        return np.nan

    if var_s < EPSILON:
        warnings.warn("Variance near zero, Z-score may be unreliable", UserWarning)
        return 0

    if s > 0:
        return (s - 1) / np.sqrt(var_s)
    return (s + 1) / np.sqrt(var_s) if s < 0 else 0

def _p_value(z, alpha, continuous_confidence=False):
    """
    Calculates p-value and trend direction.

    Args:
        z (float): Z-score
        alpha (float): Significance level
        continuous_confidence (bool): If True, returns trend direction (increasing/decreasing)
                                     even if not significant (h=False), unless z=0.
    """
    p = 2 * (1 - norm.cdf(abs(z)))
    h = abs(z) > norm.ppf(1 - alpha / 2)

    if continuous_confidence:
        # In continuous mode, we always report direction unless Z is exactly zero
        if z < 0:
            trend = 'decreasing'
        elif z > 0:
            trend = 'increasing'
        else:
            trend = 'indeterminate'
    else:
        # Classical mode: only report direction if significant
        if not h:
            trend = 'no trend'
        else:
            trend = 'decreasing' if z < 0 else 'increasing' if z > 0 else 'no trend'

    return p, h, trend

def _mk_probability(p, s):
    """
    Computes the Mann-Kendall probability.
    """
    # Ensure p is a scalar to prevent array issues downstream
    p_scalar = np.mean(p)
    C = 1 - p_scalar / 2
    Cd = C if s <= 0 else p_scalar / 2
    return float(C), float(Cd)

def _sens_estimator_unequal_spacing(x, t):
    """
    Computes Sen's slope for unequally spaced data using a vectorized approach.

    Statistical Assumptions:
    ----------------------
    1.  **Linear Trend**: The method assumes that the trend in the data can be
        reasonably approximated by a linear model. The Sen's slope is the
        median of the slopes calculated between all pairs of points, providing
        a robust estimate of the linear trend.
    2.  **Independence of Errors**: The errors (deviations from the linear
        trend) are assumed to be independent.
    3.  **Homoscedasticity**: It is assumed that the variance of the errors is
        constant over time.
    """
    n = len(x)

    # Add a hard limit to prevent integer overflow on large n*n arrays
    MAX_SAFE_N = 46340
    if n > MAX_SAFE_N:
        # Relaxed check as we handle fast estimation elsewhere, but exact Sen's slope
        # using this function is still O(n^2) and will likely crash if called directly.
        # trend_test calls fast_sens_slope for large N, avoiding this.
        warnings.warn(
            f"Sample size n={n} exceeds maximum safe size of {MAX_SAFE_N}. "
            f"This would cause an integer overflow during pairwise calculations. "
            f"Consider using the regional_test() function for aggregation or subsampling your data.",
            UserWarning
        )
        return np.array([])

    if n > 5000:
        mem_gb = (n**2 * 8 / 1e9)
        warnings.warn(
            f"Large sample size (n={n}) requires ~{mem_gb:.1f} GB memory "
            f"for pairwise calculations. Maximum safe n is {MAX_SAFE_N}. "
            f"Using `regional_test()` for aggregating multiple smaller sites is recommended.",
            UserWarning
        )

    if n < 2:
        return np.array([])

    # Create all pairs of indices
    i, j = np.triu_indices(n, k=1)

    # Calculate differences
    x_diff = x[j] - x[i]
    t_diff = t[j] - t[i]

    # Avoid division by zero using a tolerance for float comparison
    valid_mask = np.abs(t_diff) > 1e-10

    return x_diff[valid_mask] / t_diff[valid_mask]


def _sens_estimator_adaptive(x, t, max_pairs=None, random_state=None):
    """
    Adaptive Sen's slope: automatic or fast based on size.

    Args:
        max_pairs: None for automatic, or specific limit
        random_state: For reproducibility in fast mode
    """
    n = len(x)

    if max_pairs is None:
        # Automatic
        if n * (n - 1) // 2 <= 100000:
            return _sens_estimator_unequal_spacing(x, t)
        else:
            from ._large_dataset import fast_sens_slope
            return fast_sens_slope(x, t, random_state=random_state)
    else:
        # User specified limit
        from ._large_dataset import fast_sens_slope
        return fast_sens_slope(x, t, max_pairs=max_pairs, random_state=random_state)


def _sens_estimator_censored(x, t, cen_type, lt_mult=DEFAULT_LT_MULTIPLIER, gt_mult=DEFAULT_GT_MULTIPLIER, method='unbiased'):
    """
    Computes Sen's slope for censored, unequally spaced data.

    This implements the logic from the LWP-TRENDS R script, which is the
    default behavior. An optional, more statistically robust method is also
    provided.

    Args:
        x (np.array): The data values.
        t (np.array): The timestamps.
        cen_type (np.array): The censor types ('lt', 'gt', 'not').
        lt_mult (float): Multiplier for left-censored data.
        gt_mult (float): Multiplier for right-censored data.
        method (str): The method to use for handling ambiguous slopes.
            - 'lwp': Sets ambiguous slopes to 0, mimicking the
              LWP-TRENDS R script.
            - 'unbiased' (default): Sets ambiguous slopes to np.nan, which is a more
              statistically neutral approach. (Formerly 'nan').

    Returns:
        np.array: An array of calculated slopes.

    Statistical Assumptions:
    ----------------------
    This function extends Sen's slope estimation to censored data, building
    on the same core assumptions but with important methodological choices:

    1.  **Linear Trend**: Like the standard Sen's slope, a linear trend is
        assumed.
    2.  **Substitution of Censored Values**: For the final slope calculation,
        censored values are substituted with a multiple of their detection
        limit (`lt_mult` for left-censored, `gt_mult` for right-censored).
        This is a heuristic approach from the LWP-TRENDS script and assumes
        these substitutions are reasonable proxies for the true values.
    3.  **Handling of Ambiguous Slopes**: The method acknowledges that the
        slope between certain pairs of points (e.g., two left-censored values)
        is ambiguous. The `method` parameter determines how these are handled:
        -   `'lwp'`: Sets ambiguous slopes to 0. This is a
            conservative choice that reduces the magnitude of the overall
            median slope but may not be statistically neutral.
        -   `'unbiased'` (or `'nan'`): Sets ambiguous slopes to NaN, effectively removing them
            from the median calculation. This is a more statistically neutral
            approach, as it does not bias the slope towards zero.
    """
    n = len(x)

    # Add a hard limit to prevent integer overflow on large n*n arrays
    MAX_SAFE_N = 46340
    if n > MAX_SAFE_N:
        warnings.warn(
            f"Sample size n={n} exceeds maximum safe size of {MAX_SAFE_N}. "
            f"This would cause an integer overflow during pairwise calculations. "
            f"Consider using the regional_test() function for aggregation or subsampling your data.",
            UserWarning
        )
        return np.array([])

    if n > 5000:
        mem_gb = (n**2 * 8 / 1e9)
        warnings.warn(
            f"Large sample size (n={n}) requires ~{mem_gb:.1f} GB memory "
            f"for pairwise calculations. Maximum safe n is {MAX_SAFE_N}. "
            f"Using `regional_test()` for aggregating multiple smaller sites is recommended.",
            UserWarning
        )

    if n < 2:
        return np.array([])

    # Create all pairs of indices
    i_indices, j_indices = np.triu_indices(n, k=1)

    # 1. Calculate raw differences and slopes (for censor checks)
    x_diff_raw = x[j_indices] - x[i_indices]
    t_diff = t[j_indices] - t[i_indices]

    # Avoid division by zero using a tolerance for float comparison
    valid_t_mask = np.abs(t_diff) > 1e-10
    x_diff_raw = x_diff_raw[valid_t_mask]
    t_diff = t_diff[valid_t_mask]
    i_indices = i_indices[valid_t_mask]
    j_indices = j_indices[valid_t_mask]

    slopes_raw = x_diff_raw / t_diff

    # 2. Modify values for final slope calculation (as per R script)
    x_mod = x.copy().astype(float)
    x_mod[cen_type == 'lt'] *= lt_mult
    x_mod[cen_type == 'gt'] *= gt_mult
    x_diff_mod = x_mod[j_indices] - x_mod[i_indices]
    slopes_mod = x_diff_mod / t_diff

    # 3. Create censor labels for pairs and apply rules
    # The order MUST be j, i to match the R script's lower.tri() logic,
    # which pairs (later_time, earlier_time).
    cen_type_pairs = cen_type[j_indices] + " " + cen_type[i_indices]
    slopes_final = slopes_mod.copy()

    # Determine the value to assign to ambiguous slopes based on the method
    # 'lwp' uses 0, 'unbiased' (and 'nan') uses np.nan
    ambiguous_slope_value = 0 if method == 'lwp' else np.nan

    # Rule 1: No slope between two censored values of the same type.
    slopes_final[(cen_type_pairs == 'gt gt') | (cen_type_pairs == 'lt lt')] = ambiguous_slope_value

    # Rule 2: Ambiguous if later value is left-censored ('lt') and slope is positive.
    slopes_final[(slopes_raw > 0) & (cen_type_pairs == 'lt not')] = ambiguous_slope_value

    # Rule 3: Ambiguous if earlier value is left-censored ('lt') and slope is negative.
    slopes_final[(slopes_raw < 0) & (cen_type_pairs == 'not lt')] = ambiguous_slope_value

    # Rule 4: Ambiguous if earlier value is right-censored ('gt') and slope is positive.
    slopes_final[(slopes_raw > 0) & (cen_type_pairs == 'not gt')] = ambiguous_slope_value

    # Rule 5: Ambiguous if later value is right-censored ('gt') and slope is negative.
    slopes_final[(slopes_raw < 0) & (cen_type_pairs == 'gt not')] = ambiguous_slope_value

    return slopes_final


def _sens_estimator_censored_adaptive(x, t, cen_type,
                                      lt_mult=DEFAULT_LT_MULTIPLIER, gt_mult=DEFAULT_GT_MULTIPLIER,
                                      method='unbiased',
                                      max_pairs=None,
                                      random_state=None):
    """Adaptive censored Sen's slope."""
    n = len(x)

    if max_pairs is None:
        if n * (n - 1) // 2 <= 100000:
            return _sens_estimator_censored(
                x, t, cen_type, lt_mult, gt_mult, method
            )
        else:
            from ._large_dataset import fast_sens_slope_censored
            return fast_sens_slope_censored(
                x, t, cen_type,
                lt_mult=lt_mult, gt_mult=gt_mult,
                method=method,
                random_state=random_state
            )
    else:
        from ._large_dataset import fast_sens_slope_censored
        return fast_sens_slope_censored(
            x, t, cen_type,
            max_pairs=max_pairs,
            lt_mult=lt_mult, gt_mult=gt_mult,
            method=method,
            random_state=random_state
        )


def _confidence_intervals(slopes, var_s, alpha, method='direct', total_pairs=None):
    """
    Computes the confidence intervals for Sen's slope.

    Args:
        slopes: Array of slopes (sample or full population)
        var_s: Variance of S statistic (corresponds to total_pairs)
        alpha: Significance level
        method: 'direct' or 'lwp'
        total_pairs: Total number of possible pairs. If None, assumes slopes
                     contains all pairs. Used for scaling when slopes is a sample.
    """
    # Filter out NaN values, which can occur with the 'nan' method for
    # censored slopes.
    valid_slopes = slopes[~np.isnan(slopes)]
    n_sample = len(valid_slopes)

    if n_sample == 0 or var_s < EPSILON:
        return np.nan, np.nan

    if total_pairs is None:
        total_pairs = n_sample

    # For a two-sided confidence interval
    Z = norm.ppf(1 - alpha / 2)

    # Calculate limits in terms of the total population ranks
    C = Z * np.sqrt(var_s)
    M1 = (total_pairs - C) / 2
    M2 = (total_pairs + C) / 2

    # Convert to quantiles
    q1 = M1 / total_pairs
    q2 = M2 / total_pairs

    # Map to ranks in the sample
    rank1 = q1 * n_sample
    rank2 = q2 * n_sample

    sorted_slopes = np.sort(valid_slopes)

    if method == 'lwp':
        # LWP-TRENDS R script method (interpolation)
        # Note: 1-based ranks for interpolation
        ranks = np.arange(1, n_sample + 1)
        lower_ci, upper_ci = np.interp([rank1, rank2], ranks, sorted_slopes)
    else:
        # Default method (direct indexing)
        # q1 is fraction of data. q1 * n gives float index.
        # We target the index corresponding to that rank.
        # rank 1 -> index 0

        lower_idx = int(np.round(rank1 - 1))
        upper_idx = int(np.round(rank2 - 1))

        # Ensure indices are within bounds
        lower_idx = np.clip(lower_idx, 0, n_sample - 1)
        upper_idx = np.clip(upper_idx, 0, n_sample - 1)

        lower_ci = sorted_slopes[lower_idx]
        upper_ci = sorted_slopes[upper_idx]

    return lower_ci, upper_ci


def _sen_probability(slopes, var_s, total_pairs=None):
    """
    Calculates the probability that the Sen's slope is > 0.
    """
    # Filter out NaN values from slopes
    valid_slopes = slopes[~np.isnan(slopes)]
    n_sample = len(valid_slopes)

    if n_sample == 0 or var_s < EPSILON:
        return np.nan, np.nan, np.nan

    if total_pairs is None:
        total_pairs = n_sample

    sorted_slopes = np.sort(valid_slopes)
    ranks = np.arange(1, n_sample + 1)

    # Replicate R's approx function with different tie methods
    R0_median = np.interp(0, sorted_slopes, ranks)
    R0_max = np.interp(0, sorted_slopes, ranks, right=n_sample) # ties='max'
    R0_min = np.interp(0, sorted_slopes, ranks, left=1) # ties='min

    # Handle edge cases where all slopes are on one side of zero
    if np.all(valid_slopes < 0):
        R0_median = R0_max = n_sample
        R0_min = 1 # R behavior is complex here, this is a simplification
    elif np.all(valid_slopes > 0):
        R0_median = R0_min = 1
        R0_max = n_sample # R behavior

    # Calculate probabilities
    # Scale the sample statistic to the population scale
    scaling = total_pairs / n_sample

    z_median = ((2 * R0_median - n_sample) * scaling) / np.sqrt(var_s) if var_s > 0 else 0
    z_max = ((2 * R0_max - n_sample) * scaling) / np.sqrt(var_s) if var_s > 0 else 0
    z_min = ((2 * R0_min - n_sample) * scaling) / np.sqrt(var_s) if var_s > 0 else 0

    sen_prob = norm.cdf(z_median)
    sen_prob_max = norm.cdf(z_max)
    sen_prob_min = norm.cdf(z_min)

    return sen_prob, sen_prob_max, sen_prob_min
