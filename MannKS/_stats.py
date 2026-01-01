import numpy as np
from scipy.stats import norm, rankdata

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
    This is a Python translation of the GetKendal function from the LWP-TRENDS
    R script, which is adapted from the NADA package (Helsel, 2012).

    Statistical Assumptions:
    ----------------------
    This function adapts the Mann-Kendall test for censored data and relies on
    similar core assumptions, but with specific considerations for censoring:

    1.  **Independence and Monotonic Trend**: Same as the standard Mann-Kendall
        test. The data should be serially independent, and the underlying
        trend should be monotonic.
    2.  **Correct Censoring Information**: The accuracy of the test depends on
        the correct classification of data as left-censored ('lt'),
        right-censored ('gt'), or non-censored ('not').
    3.  **Tie Correction**: The variance calculation includes a robust tie
        correction method that is essential for accuracy when censored data
        introduces ties. The method is adapted from the NADA R package and is
        designed to handle ties between censored-censored, censored-uncensored,
        and uncensored-uncensored data points.
    4.  **Handling of Right-Censored Data**: The method for handling
        right-censored data is determined by the `mk_test_method` parameter:
        -   `'robust'` (default): A non-parametric approach that treats
            right-censored values as having a rank greater than all observed
            values, without modifying their actual values.
        -   `'lwp'`: A heuristic from the LWP-TRENDS R script that replaces all
            right-censored values with a value slightly larger than the maximum
            observed right-censored value and treats them as non-censored.
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

    # Add a hard limit to prevent integer overflow on large n*n arrays
    MAX_SAFE_N = 46340
    if n > MAX_SAFE_N:
        raise ValueError(
            f"Sample size n={n} exceeds maximum safe size of {MAX_SAFE_N}. "
            f"This would cause an integer overflow during pairwise calculations. "
            f"Consider using the regional_test() function for aggregation or subsampling your data."
        )

    if n > 5000:
        import warnings
        mem_gb = (n**2 * 8 / 1e9)
        warnings.warn(
            f"Large sample size (n={n}) requires ~{mem_gb:.1f} GB memory "
            f"for pairwise calculations. Maximum safe n is {MAX_SAFE_N}. "
            f"Consider using regional_test() for aggregating multiple smaller sites.",
            UserWarning
        )

    if n < 2:
        return 0, 0, 0, 0

    # 3. Calculate delx and dely to break ties
    unique_xx = np.unique(xx)
    min_diff_x = _get_min_positive_diff(unique_xx)
    if min_diff_x > 0:
        if tie_break_method == 'lwp':
            delx = min_diff_x / 1000.0
        else: # robust
            delx = min_diff_x / 2.0
    else:
        # Default to 1.0 if no difference, to separate censored/uncensored
        delx = 1.0

    unique_yy = np.unique(yy)
    min_diff_y = _get_min_positive_diff(unique_yy)
    if min_diff_y > 0:
        if tie_break_method == 'lwp':
            dely = min_diff_y / 1000.0
        else: # robust
            dely = min_diff_y / 2.0
    else:
        dely = 1.0

    # 4. S-statistic calculation using vectorized outer products
    dupx = xx - delx * cx
    dupy = yy - dely * cy

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

    # 5. D (denominator) calculation for Tau
    J = n * (n - 1) / 2.0
    if tau_method == 'a':
        D = J
    else: # Default to Tau-b
        # tt: number of tied pairs in x
        tt = (np.sum(1 - np.abs(np.sign(diffx))) - n) / 2.0
        tt += np.sum(cix) / 2.0
        tt += np.sum(tplus) / 2.0

        # uu: number of tied pairs in y (time)
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
    # The following section calculates three correction terms for the variance
    # of S (varS), based on the methodology from the NADA R package by
    # Dennis Helsel. This is crucial for handling ties in censored data
    # correctly.
    #
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
    x2_uc = 0 # (2-2) = 0
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
        import warnings
        warnings.warn("Denominator near zero in Tau calculation", UserWarning)

    return kenS, varS, D, Tau


def _z_score(s, var_s):
    if var_s < EPSILON:
        import warnings
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
        raise ValueError(
            f"Sample size n={n} exceeds maximum safe size of {MAX_SAFE_N}. "
            f"This would cause an integer overflow during pairwise calculations. "
            f"Consider using the regional_test() function for aggregation or subsampling your data."
        )

    if n > 5000:
        import warnings
        mem_gb = (n**2 * 8 / 1e9)
        warnings.warn(
            f"Large sample size (n={n}) requires ~{mem_gb:.1f} GB memory "
            f"for pairwise calculations. Maximum safe n is {MAX_SAFE_N}. "
            f"Consider using regional_test() for aggregating multiple smaller sites.",
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


def _sens_estimator_censored(x, t, cen_type, lt_mult=DEFAULT_LT_MULTIPLIER, gt_mult=DEFAULT_GT_MULTIPLIER, method='nan'):
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
            - 'lwp' (default): Sets ambiguous slopes to 0, mimicking the
              LWP-TRENDS R script.
            - 'nan': Sets ambiguous slopes to np.nan, which is a more
              statistically neutral approach.

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
        -   `'lwp'` (default): Sets ambiguous slopes to 0. This is a
            conservative choice that reduces the magnitude of the overall
            median slope but may not be statistically neutral.
        -   `'nan'`: Sets ambiguous slopes to NaN, effectively removing them
            from the median calculation. This is a more statistically neutral
            approach, as it does not bias the slope towards zero.
    """
    n = len(x)

    # Add a hard limit to prevent integer overflow on large n*n arrays
    MAX_SAFE_N = 46340
    if n > MAX_SAFE_N:
        raise ValueError(
            f"Sample size n={n} exceeds maximum safe size of {MAX_SAFE_N}. "
            f"This would cause an integer overflow during pairwise calculations. "
            f"Consider using the regional_test() function for aggregation or subsampling your data."
        )

    if n > 5000:
        import warnings
        mem_gb = (n**2 * 8 / 1e9)
        warnings.warn(
            f"Large sample size (n={n}) requires ~{mem_gb:.1f} GB memory "
            f"for pairwise calculations. Maximum safe n is {MAX_SAFE_N}. "
            f"Consider using regional_test() for aggregating multiple smaller sites.",
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

def _confidence_intervals(slopes, var_s, alpha, method='direct'):
    """
    Computes the confidence intervals for Sen's slope.
    """
    # Filter out NaN values, which can occur with the 'nan' method for
    # censored slopes.
    valid_slopes = slopes[~np.isnan(slopes)]
    n = len(valid_slopes)

    if n == 0 or var_s < EPSILON:
        return np.nan, np.nan

    # For a two-sided confidence interval
    Z = norm.ppf(1 - alpha / 2)

    # Ranks of the lower and upper confidence limits (1-based)
    C = Z * np.sqrt(var_s)
    M1 = (n - C) / 2
    M2 = (n + C) / 2

    sorted_slopes = np.sort(valid_slopes)

    if method == 'lwp':
        # LWP-TRENDS R script method (interpolation)
        ranks = np.arange(1, n + 1)
        lower_ci, upper_ci = np.interp([M1, M2], ranks, sorted_slopes)
    else:
        # Default method (direct indexing)
        # Note: np.round uses "round half to even" which may differ from other
        # statistical software. This is a deliberate choice for consistency
        # with a standard, well-defined rounding method.
        lower_idx = int(np.round(M1 - 1))
        upper_idx = int(np.round(M2 - 1))

        # Ensure indices are within bounds
        if 0 <= lower_idx < n and 0 <= upper_idx < n:
            lower_ci = sorted_slopes[lower_idx]
            upper_ci = sorted_slopes[upper_idx]
        else:
            import warnings
            warnings.warn(
                f"Confidence interval calculation failed: calculated indices "
                f"({lower_idx}, {upper_idx}) were out of bounds for the {n} valid slopes. "
                f"This is typically due to insufficient data or extremely high variance.",
                UserWarning
            )
            lower_ci, upper_ci = np.nan, np.nan

    return lower_ci, upper_ci


def _sen_probability(slopes, var_s):
    """
    Calculates the probability that the Sen's slope is > 0.
    """
    # Filter out NaN values from slopes
    valid_slopes = slopes[~np.isnan(slopes)]
    n_slopes = len(valid_slopes)

    if n_slopes == 0 or var_s < EPSILON:
        return np.nan, np.nan, np.nan

    sorted_slopes = np.sort(valid_slopes)
    ranks = np.arange(1, n_slopes + 1)

    # Replicate R's approx function with different tie methods
    R0_median = np.interp(0, sorted_slopes, ranks)
    R0_max = np.interp(0, sorted_slopes, ranks, right=n_slopes) # ties='max'
    R0_min = np.interp(0, sorted_slopes, ranks, left=1) # ties='min

    # Handle edge cases where all slopes are on one side of zero
    if np.all(valid_slopes < 0):
        R0_median = R0_max = n_slopes
        R0_min = 1 # R behavior is complex here, this is a simplification
    elif np.all(valid_slopes > 0):
        R0_median = R0_min = 1
        R0_max = n_slopes # R behavior

    # Calculate probabilities
    z_median = (2 * R0_median - n_slopes) / np.sqrt(var_s) if var_s > 0 else 0
    z_max = (2 * R0_max - n_slopes) / np.sqrt(var_s) if var_s > 0 else 0
    z_min = (2 * R0_min - n_slopes) / np.sqrt(var_s) if var_s > 0 else 0

    sen_prob = norm.cdf(z_median)
    sen_prob_max = norm.cdf(z_max)
    sen_prob_min = norm.cdf(z_min)

    return sen_prob, sen_prob_max, sen_prob_min
