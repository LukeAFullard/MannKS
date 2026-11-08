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


def _tie_break_delta(min_diff, n, method):
    """
    Size of the tiny shift used to separate censored from uncensored values and
    to make uncensored values distinct inside the variance calculation
    (``delx``/``dely`` in NADA::cenken / LWP-TRENDS ``GetKendal``).

    The shift must satisfy ``n * delta < min_diff`` so that the cumulative
    shifts ``i * delta`` used in the variance terms can never move a value past
    a neighbouring distinct value.

    - ``'lwp'``: ``min_diff / 1000`` exactly as in the R code.
    - ``'robust'``: ``min_diff / (2 * (n + 1))``, which satisfies the
      constraint for any ``n``.
    """
    if min_diff <= 0:
        return 1.0
    if method == 'lwp':
        return min_diff / 1000.0
    return min_diff / (2.0 * (n + 1))


def _mk_score_and_var_censored(x, t, censored, cen_type, tau_method='b', mk_test_method='robust', tie_break_method='robust'):
    """
    Calculates the Mann-Kendall S statistic and its variance for censored data.

    This is a port of ``GetKendal`` from LWP-TRENDS (itself adapted from
    ``NADA::cenken``).  Censored comparisons that cannot be resolved score 0
    (Helsel 2012).  Tied time values score 0 and are accounted for in the
    variance, as are ties among values and among censored values.

    Returns:
        (S, Var(S), D, Tau) where D is the Tau denominator (Tau-a: n(n-1)/2,
        Tau-b: the tie-adjusted denominator).
    """
    x = np.asarray(x, dtype=float)
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

    # Time is treated as uncensored.  Equal times share a rank, so pairs with
    # tied times contribute 0 to S (as in LWP-TRENDS, which ranks by
    # season-year).
    yy = rankdata(t, method='dense').astype(float)
    cy = np.zeros_like(yy, dtype=bool)
    n = len(xx)

    if n < 2:
        return 0, 0, 0, 0

    # 3. Calculate delx and dely to break ties
    delx = _tie_break_delta(_get_min_positive_diff(np.unique(xx)), n, tie_break_method)
    dely = _tie_break_delta(_get_min_positive_diff(np.unique(yy)), n, tie_break_method)

    dupx = xx - delx * cx
    dupy = yy - dely * cy

    # 4. S-statistic calculation using chunking for large n
    chunk_size = 2000 # Tunable parameter
    kenS = 0
    tt = 0
    uu = 0

    use_fast_path = (n > 5000 and not np.any(cx) and mk_test_method != 'lwp')
    use_chunking = n > 2000 and not use_fast_path

    if use_fast_path:
        # FAST PATH: uncensored data, O(N log N).  scipy's Tau-b handles ties
        # in both x and t, and S is recovered exactly from Tau-b:
        #   S = tau_b * sqrt((n0 - n1) * (n0 - n2))
        res = kendalltau(xx, yy)
        tau_b = res.statistic
        if np.isnan(tau_b):
            tau_b = 0.0

        _, c_x = np.unique(xx, return_counts=True)
        _, c_y = np.unique(yy, return_counts=True)
        tt = int(np.sum(c_x * (c_x - 1)) // 2)
        uu = int(np.sum(c_y * (c_y - 1)) // 2)
        n_pairs = n * (n - 1) // 2
        denom_b = np.sqrt(float(n_pairs - tt) * float(n_pairs - uu))
        kenS = int(round(tau_b * denom_b))

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
    rxlng = _rle_lengths(rankdata(tmpx, method='average'))
    rylng = _rle_lengths(rankdata(tmpy, method='average'))

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
    # NADA/LWP: x4 <- nrxlng * (rxlng - 1), i.e. sum over tie groups of (length - 1).
    # Note: R's rank() averages ties, so tie runs are detected by rle(); the
    # equivalent here is rankdata(method='average').
    _lv, _lc = np.unique(rxlng, return_counts=True)
    x4 = np.sum(_lc * (_lv - 1))
    _lv, _lc = np.unique(rylng, return_counts=True)
    y4 = np.sum(_lc * (_lv - 1))
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
    rxlng_u = _rle_lengths(rankdata(dxx_u, method='average'))
    rylng_u = _rle_lengths(rankdata(dyy_u, method='average'))
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
    p = 2 * norm.sf(abs(z))
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
    p_scalar = float(p)
    C = 1 - p_scalar / 2
    Cd = C if s <= 0 else p_scalar / 2
    return float(C), float(Cd)

SENS_SLOPE_METHODS = ('ats', 'lwp', 'drop_ambiguous')
_DEPRECATED_SENS_SLOPE_METHODS = ('unbiased', 'nan')


def _normalize_sens_slope_method(method, stacklevel=3):
    """
    Map deprecated names of the drop-ambiguous-pairs estimator ('unbiased',
    'nan') to 'drop_ambiguous' with a FutureWarning; validate the name.
    """
    if method in _DEPRECATED_SENS_SLOPE_METHODS:
        warnings.warn(
            f"sens_slope_method='{method}' is deprecated and will be removed in a future "
            "release; use 'drop_ambiguous' (same estimator). Note that it is biased away "
            "from zero for censored data; the default 'ats' is recommended.",
            FutureWarning, stacklevel=stacklevel)
        return 'drop_ambiguous'
    if method not in SENS_SLOPE_METHODS:
        raise ValueError(f"Invalid `sens_slope_method`. Must be one of {list(SENS_SLOPE_METHODS)}.")
    return method


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

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if n <= 2000:
        i, j = np.triu_indices(n, k=1)
        x_diff = x[j] - x[i]
        t_diff = t[j] - t[i]
        # Pairs with (numerically) equal times have no defined slope.
        valid_mask = np.abs(t_diff) > 1e-10
        return x_diff[valid_mask] / t_diff[valid_mask]

    # Row-wise construction keeps peak memory at ~one copy of the slopes.
    out = []
    for i in range(n - 1):
        dt = t[i + 1:] - t[i]
        m = np.abs(dt) > 1e-10
        out.append((x[i + 1:][m] - x[i]) / dt[m])
    return np.concatenate(out)


def _total_pairs_if_sampled(n, max_pairs):
    """Number of possible pairs if a pair-sampling estimator will subsample, else None."""
    n_pairs = n * (n - 1) // 2
    if max_pairs is not None and n_pairs > max_pairs:
        return n_pairs
    return None


def _sens_estimator_adaptive(x, t, max_pairs=None, random_state=None):
    """
    Sen's slopes, exact or sampled.

    Args:
        max_pairs: None -> exact (all pairs).  An integer -> sample at most
            this many pairs when there are more possible pairs than that.
        random_state: Seed for pair sampling (a fixed default seed is used
            when None, so results are reproducible).
    """
    if max_pairs is None:
        return _sens_estimator_unequal_spacing(x, t)
    from ._large_dataset import fast_sens_slope
    return fast_sens_slope(x, t, max_pairs=max_pairs, random_state=random_state)


def _sens_estimator_censored(x, t, cen_type, lt_mult=DEFAULT_LT_MULTIPLIER, gt_mult=DEFAULT_GT_MULTIPLIER, method='drop_ambiguous'):
    """
    Computes Sen's slope for censored, unequally spaced data.

    This implements the pairwise-slope logic of the LWP-TRENDS R script
    (``GetInterObservationSlopes``).  For a less biased estimator with
    censored data use the Akritas-Theil-Sen estimator (``sens_slope_method='ats'``,
    the default in ``trend_test``/``seasonal_trend_test``).

    Args:
        x (np.array): The data values.
        t (np.array): The timestamps.
        cen_type (np.array): The censor types ('lt', 'gt', 'not').
        lt_mult (float): Multiplier for left-censored data.
        gt_mult (float): Multiplier for right-censored data.
        method (str): The method to use for handling ambiguous slopes.
            - 'lwp': Sets ambiguous slopes to 0, mimicking the
              LWP-TRENDS R script.
            - 'drop_ambiguous' (formerly 'unbiased'/'nan'): Drops ambiguous
              slopes (sets them to np.nan).  With a trend, the dropped pairs
              are mostly those of the opposite sign, so the median is pushed
              away from zero (roughly +17% at 20% censoring and +70% at 50%
              censoring in simulations).

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
        -   `'lwp'`: Sets ambiguous slopes to 0 (LWP-TRENDS behaviour).  This
            biases the median towards zero as censoring increases.
        -   `'drop_ambiguous'`: Sets ambiguous slopes to NaN, removing them
            from the median.  This biases the median *away* from zero.
        Both are heuristics; the Akritas-Theil-Sen estimator (`_ats.py`) is
        the recommended estimator for censored data.
    """
    n = len(x)
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if n < 2:
        return np.array([])
    code = _censor_codes(cen_type)
    if n <= 2000:
        i, j = np.triu_indices(n, k=1)
        slopes, valid = _censored_pair_slopes(x, t, code, i, j, lt_mult, gt_mult, method)
        return slopes[valid]
    out = []
    for i0 in range(n - 1):
        j = np.arange(i0 + 1, n)
        i = np.full(len(j), i0)
        slopes, valid = _censored_pair_slopes(x, t, code, i, j, lt_mult, gt_mult, method)
        out.append(slopes[valid])
    return np.concatenate(out)


def _censor_codes(cen_type):
    """Map censor labels to integer codes: 0 = not censored, 1 = 'lt', 2 = 'gt'."""
    cen_type = np.asarray(cen_type).astype(str)
    return np.where(cen_type == 'lt', 1, np.where(cen_type == 'gt', 2, 0)).astype(np.int8)


def _censored_pair_slopes(x, t, code, i, j, lt_mult=DEFAULT_LT_MULTIPLIER,
                          gt_mult=DEFAULT_GT_MULTIPLIER, method='drop_ambiguous'):
    """
    Censored pairwise slopes for index pairs (i, j), following LWP-TRENDS
    ``GetInterObservationSlopes``.  Each pair is oriented by time (later minus
    earlier), so the result does not depend on the order of the input rows.

    Returns (slopes, valid) where ``valid`` is False for pairs with
    (numerically) equal times.
    """
    dt = t[j] - t[i]
    valid = np.abs(dt) > 1e-10
    later_first = dt >= 0
    late = np.where(later_first, j, i)
    early = np.where(later_first, i, j)
    dt_abs = np.abs(dt)
    with np.errstate(divide='ignore', invalid='ignore'):
        slopes_raw = (x[late] - x[early]) / dt_abs
        x_mod = x.copy()
        x_mod[code == 1] *= lt_mult
        x_mod[code == 2] *= gt_mult
        slopes = (x_mod[late] - x_mod[early]) / dt_abs

    cl, ce = code[late], code[early]
    ambiguous = (
        ((cl == 1) & (ce == 1)) | ((cl == 2) & (ce == 2))          # 1. both censored, same side
        | ((slopes_raw > 0) & (cl == 1) & (ce == 0))               # 2. 'lt not', positive
        | ((slopes_raw < 0) & (cl == 0) & (ce == 1))               # 3. 'not lt', negative
        | ((slopes_raw > 0) & (cl == 0) & (ce == 2))               # 4. 'not gt', positive
        | ((slopes_raw < 0) & (cl == 2) & (ce == 0))               # 5. 'gt not', negative
    )
    slopes[ambiguous] = 0.0 if method == 'lwp' else np.nan
    return slopes, valid


def _sens_estimator_censored_adaptive(x, t, cen_type,
                                      lt_mult=DEFAULT_LT_MULTIPLIER, gt_mult=DEFAULT_GT_MULTIPLIER,
                                      method='drop_ambiguous',
                                      max_pairs=None,
                                      random_state=None):
    """Censored Sen's slopes, exact (max_pairs=None) or sampled."""
    if max_pairs is None:
        return _sens_estimator_censored(
            x, t, cen_type, lt_mult, gt_mult, method
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


def _valid_pair_count(slopes, total_pairs):
    """
    Number of usable (non-NaN) slopes in the population the slopes represent.

    When ``total_pairs`` is None the slopes are the full set of pairs.  When the
    slopes are a random sample of ``total_pairs`` pairs, the valid fraction
    observed in the sample is scaled up to the population.
    """
    n_all = len(slopes)
    n_valid = int(np.sum(~np.isnan(slopes)))
    if total_pairs is None or n_all == 0:
        return float(n_valid)
    return float(total_pairs) * n_valid / n_all


def _confidence_intervals(slopes, var_s, alpha, method='direct', total_pairs=None):
    """
    Confidence interval for Sen's slope (Gilbert 1987, section 16.5).

    The ranks of the limits among the N usable slopes are
    ``(N -/+ Z_{1-alpha/2} * sqrt(Var(S))) / 2``.

    Args:
        slopes: Array of slopes (all pairs, or a random sample of pairs).
            NaN entries (ambiguous censored pairs) are ignored.
        var_s: Variance of the Mann-Kendall S statistic.
        alpha: Significance level (two-sided interval of level 1 - alpha).
        method: 'direct' (nearest rank) or 'lwp' (linear interpolation
            between ranks, as in LWP-TRENDS).
        total_pairs: Number of possible pairs when ``slopes`` is a random
            sample; None when ``slopes`` holds every pair.
    """
    valid_slopes = slopes[~np.isnan(slopes)]
    n_sample = len(valid_slopes)

    if n_sample == 0 or var_s < EPSILON:
        return np.nan, np.nan

    n_pop = _valid_pair_count(slopes, total_pairs)

    Z = norm.ppf(1 - alpha / 2)
    C = Z * np.sqrt(var_s)
    M1 = (n_pop - C) / 2
    M2 = (n_pop + C) / 2

    # Map population ranks to ranks in the (possibly sampled) slope set.
    rank1 = M1 / n_pop * n_sample
    rank2 = M2 / n_pop * n_sample

    sorted_slopes = np.sort(valid_slopes)

    if method == 'lwp':
        # LWP-TRENDS: approx() over ranks 1..N; outside the range R returns NA
        # and falls back to min/max, which np.interp's clamping reproduces.
        ranks = np.arange(1, n_sample + 1)
        lower_ci, upper_ci = np.interp([rank1, rank2], ranks, sorted_slopes)
    else:
        lower_idx = int(np.clip(np.round(rank1 - 1), 0, n_sample - 1))
        upper_idx = int(np.clip(np.round(rank2 - 1), 0, n_sample - 1))
        lower_ci = sorted_slopes[lower_idx]
        upper_ci = sorted_slopes[upper_idx]

    return lower_ci, upper_ci


def _rank_of_zero(sorted_slopes, ties):
    """
    Rank of the value 0 among sorted slopes by linear interpolation, with tied
    slope values collapsed as in R's ``approx(..., ties=median|max|min)``.
    Returns NaN when 0 lies outside the range of the slopes.
    """
    vals, first, counts = np.unique(sorted_slopes, return_index=True, return_counts=True)
    lo_rank = first + 1.0
    hi_rank = first + counts.astype(float)
    if ties == 'max':
        r = hi_rank
    elif ties == 'min':
        r = lo_rank
    else:
        r = (lo_rank + hi_rank) / 2.0
    if 0 < vals[0] or 0 > vals[-1]:
        return np.nan
    if len(vals) == 1:
        return r[0]
    return float(np.interp(0.0, vals, r))


def _sen_probability(slopes, var_s, total_pairs=None):
    """
    Probability that the true slope is *less than zero*, from the rank of
    zero among the pairwise slopes (LWP-TRENDS ``Sen_Probability``,
    ``Sen_Probabilitymax``, ``Sen_Probabilitymin``).

    Returns (p_median_ties, p_max_ties, p_min_ties).  As in LWP-TRENDS, when
    every slope is negative (positive) the median-ties rank is set to the
    largest (smallest) rank; the max/min variants are NaN when 0 is outside
    the range of the slopes.
    """
    valid_slopes = slopes[~np.isnan(slopes)]
    n_sample = len(valid_slopes)

    if n_sample == 0 or var_s < EPSILON:
        return np.nan, np.nan, np.nan

    sorted_slopes = np.sort(valid_slopes)
    R0_median = _rank_of_zero(sorted_slopes, 'median')
    R0_max = _rank_of_zero(sorted_slopes, 'max')
    R0_min = _rank_of_zero(sorted_slopes, 'min')
    if np.all(valid_slopes < 0):
        R0_median = float(n_sample)
    elif np.all(valid_slopes > 0):
        R0_median = 1.0

    scaling = _valid_pair_count(slopes, total_pairs) / n_sample
    sd = np.sqrt(var_s)

    def prob(r0):
        if np.isnan(r0):
            return np.nan
        return float(norm.cdf((2 * r0 - n_sample) * scaling / sd))

    return prob(R0_median), prob(R0_max), prob(R0_min)
