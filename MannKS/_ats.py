# ats.py  -- Akritas-Theil-Sen implementation (single-file, dependency-light)
import numpy as np
import pandas as pd
from math import inf
from typing import Tuple, List, Optional, Callable
from random import randint

# ---------- Utilities: interval representation ----------
def make_intervals(y: np.ndarray,
                   censored: np.ndarray,
                   cen_type: Optional[np.ndarray] = None,
                   lod: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build intervals [a_i, b_i] for each observation.
    Arguments:
      y: numeric face values (for censored rows this should be the numeric reporting value)
      censored: boolean array (True when censored)
      cen_type: None or array of 'lt' / 'gt' / 'none' strings (if None, assume all censored are 'lt')
      lod: numeric detection limits associated with censored observations (len same as y)
    Returns:
      (lower, upper) arrays where lower[i] <= upper[i]; may include +-inf for left/right censor.
    """
    n = len(y)
    lower = np.empty(n, dtype=float)
    upper = np.empty(n, dtype=float)
    if cen_type is None:
        cen_type = np.array(['lt' if c else 'none' for c in censored])
    if lod is None:
        lod = np.array([y_i for y_i in y])

    for i in range(n):
        if not censored[i]:
            lower[i] = upper[i] = y[i]
        else:
            t = cen_type[i]
            d = lod[i]
            if t == 'lt' or t == '<' or t.lower() == 'left':
                lower[i] = -inf
                upper[i] = d
            elif t == 'gt' or t == '>' or t.lower() == 'right':
                lower[i] = d
                upper[i] = +inf
            else:
                # default to left-censor
                lower[i] = -inf
                upper[i] = d
    return lower, upper

# ---------- Pairwise interval comparison on residuals ----------
def S_of_beta(beta: float, x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> int:
    """
    Compute S(beta) = (#concordant) - (#discordant) using residual intervals:
      R_i(beta) = [lower_i - beta*x_i, upper_i - beta*x_i]
    Definitive comparisons only when intervals do not overlap.
    """
    n = len(x)
    lower_r = lower - beta * x
    upper_r = upper - beta * x

    concordant = 0
    discordant = 0
    # O(n^2) loop
    for i in range(n):
        for j in range(i+1, n):
            # check if r_i > r_j (lower_i > upper_j)
            if lower_r[i] > upper_r[j]:
                concordant += 1
            elif upper_r[i] < lower_r[j]:
                discordant += 1
            else:
                # tie/overlap -> do not count
                pass
    return concordant - discordant

# ---------- Root-finding to solve S(beta) = 0 ----------
def bracket_and_bisect_generic(score_func: Callable[[float], float],
                               slopes_hint: List[float],
                               max_expand=50, tol=1e-8,
                               maxiter=60) -> float:
    """
    Find beta* such that score_func(beta*) = 0 using bisection.
    Generic version that takes a score function and a list of slopes for initial bracket.
    """
    # Define the initial search bracket.
    if slopes_hint:
        low = np.percentile(slopes_hint, 5)
        high = np.percentile(slopes_hint, 95)
        if np.isclose(low, high):
            bracket_width = max(1.0, abs(low) * 0.5)
            low -= bracket_width
            high += bracket_width
    else:
        # Fallback if no slopes are available.
        low, high = -1e-5, 1e-5

    s_low = score_func(low)
    s_high = score_func(high)

    # Expand the bracket until the signs differ
    expand_factor = 1.6
    it = 0
    while s_low * s_high > 0 and it < max_expand:
        width = high - low
        low -= width * expand_factor
        high += width * expand_factor
        s_low = score_func(low)
        s_high = score_func(high)
        it += 1

    # Grid search fallback
    if s_low * s_high > 0:
        grid = np.linspace(low, high, num=201)
        s_vals = np.array([abs(score_func(g)) for g in grid])
        best_idx = np.argmin(s_vals)
        return float(grid[best_idx])

    # Bisection
    a, b = low, high
    sa, sb = s_low, s_high
    for _ in range(maxiter):
        m = (a + b) / 2.0
        sm = score_func(m)
        if sm == 0 or (b - a) / 2.0 < tol:
            return float(m)

        if sa * sm <= 0:
            b, sb = m, sm
        else:
            a, sa = m, sm

    return float((a + b) / 2.0)

def bracket_and_bisect(x: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                       beta0: Optional[float]=None, max_expand=50, tol=1e-8,
                       maxiter=60) -> float:
    """
    Backward compatible wrapper for the single-season case.
    """
    # Normalize x to improve numerical stability in slope calculation
    x_min = np.min(x)
    x_range = np.ptp(x) if np.ptp(x) > 0 else 1.0
    x_norm = (x - x_min) / x_range

    # Calculate slopes from all uncensored pairs to define the initial search space (on normalized data)
    detected_idx = np.where(np.isfinite(lower) & np.isfinite(upper) & (lower == upper))[0]
    slopes = []
    if len(detected_idx) >= 2:
        for i in range(len(detected_idx)):
            for j in range(i + 1, len(detected_idx)):
                xi, xj = x_norm[detected_idx[i]], x_norm[detected_idx[j]]
                if not np.isclose(xj, xi):
                    yi, yj = lower[detected_idx[i]], lower[detected_idx[j]]
                    slopes.append((yj - yi) / (xj - xi))

    def score_func(b):
        return S_of_beta(b, x_norm, lower, upper)

    beta_hat_norm = bracket_and_bisect_generic(score_func, slopes, max_expand, tol, maxiter)

    # De-normalize slope
    return beta_hat_norm / x_range


# ---------- Turnbull-style intercept (practical approach) ----------
def estimate_intercept_turnbull(residual_lower: np.ndarray, residual_upper: np.ndarray, tol=1e-6, max_iter=100) -> float:
    """
    Estimates the median of interval-censored data using a Turnbull-style approach.
    Correctly handles point masses (uncensored data) and infinite intervals (censored data).
    """
    # 1. Identify Unique Endpoints
    # We include all boundaries, including -inf and +inf if present.
    endpoints = np.unique(np.concatenate([residual_lower, residual_upper]))

    # 2. Construct Candidate Sets
    # We partition the real line into disjoint sets based on endpoints:
    # - Point sets [e_i, e_i] (for finite e_i)
    # - Open intervals (e_i, e_{i+1})
    candidate_sets = []

    for i in range(len(endpoints)):
        # Point set [e_i, e_i] - only for finite values
        if np.isfinite(endpoints[i]):
            candidate_sets.append((endpoints[i], endpoints[i]))

        # Open interval (e_i, e_{i+1})
        if i < len(endpoints) - 1:
            candidate_sets.append((endpoints[i], endpoints[i+1]))

    candidates = np.array(candidate_sets)
    if len(candidates) == 0:
        return 0.0

    # 3. Initialize P
    n_obs = len(residual_lower)
    n_sets = len(candidates)
    p = np.full(n_sets, 1.0 / n_sets)

    # 4. Pre-calculate Containment Matrix
    # A candidate set [c_L, c_R] is contained in observation [o_L, o_R] if:
    # o_L <= c_L AND c_R <= o_R

    L_col = residual_lower[:, np.newaxis]
    R_col = residual_upper[:, np.newaxis]
    C_start = candidates[:, 0]
    C_end = candidates[:, 1]

    is_contained = (C_start >= L_col) & (C_end <= R_col)

    # Filter out candidates that are never contained in any observation to save computation
    # (Though typically with endpoints derived from obs, most are relevant)
    valid_candidates_idx = np.where(is_contained.any(axis=0))[0]
    if len(valid_candidates_idx) == 0:
        return 0.0 # Should not happen

    p = p[valid_candidates_idx]
    p /= np.sum(p) # Renormalize
    is_contained = is_contained[:, valid_candidates_idx]
    candidates = candidates[valid_candidates_idx]
    n_sets = len(p)

    # 5. EM Loop
    for _ in range(max_iter):
        p_old = p.copy()

        # E-step
        numer = p * is_contained
        denoms = np.sum(numer, axis=1, keepdims=True)
        denoms[denoms == 0] = 1.0 # Avoid division by zero
        alpha = numer / denoms

        # M-step
        p = np.sum(alpha, axis=0) / n_obs

        if np.sum(np.abs(p - p_old)) < tol:
            break

    # 6. Find Median
    ecdf = np.cumsum(p)
    # Find first index where cumulative probability >= 0.5
    median_indices = np.where(ecdf >= 0.5)[0]
    if len(median_indices) == 0:
        median_idx = len(p) - 1
    else:
        median_idx = median_indices[0]

    chosen = candidates[median_idx]

    # Return representative value
    if chosen[0] == chosen[1]:
        return chosen[0] # Point mass
    else:
        # Interval (a, b)
        a, b = chosen[0], chosen[1]
        # Handle infinite bounds heuristically for plot/intercept purposes
        if np.isneginf(a): return b # Left tail -> return upper bound
        if np.isposinf(b): return a # Right tail -> return lower bound
        return (a + b) / 2.0


# ---------- Public wrapper (Non-Seasonal) ----------
def ats_slope(x: np.ndarray, y: np.ndarray, censored: np.ndarray,
              cen_type: Optional[np.ndarray] = None, lod: Optional[np.ndarray] = None,
              bootstrap_ci: bool = True, n_boot: int = 500,
              ci_alpha: float = 0.05) -> dict:
    """
    Compute ATS slope estimate and bootstrap CI.
    Returns dict with keys: beta, intercept, ci_lower, ci_upper, prop_censored, notes
    """
    lower, upper = make_intervals(y, censored, cen_type=cen_type, lod=lod)

    # Calculate beta using the (now internally normalized) bisection wrapper
    beta_hat = bracket_and_bisect(x, lower, upper, beta0=None)

    # Calculate residuals and estimate intercept using Turnbull method
    # r = y - beta * x.  Use x - x_min for numerical stability in residual calculation
    x_min = np.min(x)
    x_shifted = x - x_min

    r_lower = lower - beta_hat * x_shifted
    r_upper = upper - beta_hat * x_shifted
    intercept_shifted = estimate_intercept_turnbull(r_lower, r_upper)

    # The intercept we want is for the model y = I + beta * x
    # We found I_shifted such that y = I_shifted + beta * (x - x_min)
    # y = (I_shifted - beta * x_min) + beta * x
    intercept = intercept_shifted - beta_hat * x_min

    prop_cen = float(np.mean(censored))

    result = {'beta': beta_hat, 'intercept': intercept,
              'prop_censored': prop_cen, 'notes': []}

    # simple diagnostics: fraction of pairwise comparisons that were ties at final beta
    n = len(x)

    # Use normalized x for S_of_beta to be consistent with internal checks,
    # but strictly S_of_beta is scale-invariant if beta is scaled.
    # Here we just use original x and beta, as tie check is relative.
    lower_r = lower - beta_hat * x
    upper_r = upper - beta_hat * x
    ties = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i+1, n):
            total_pairs += 1
            if (lower_r[i] <= upper_r[j]) and (upper_r[i] >= lower_r[j]):
                # intervals overlap -> tie
                ties += 1
    result['pairwise_ties_frac'] = ties / total_pairs if total_pairs > 0 else np.nan

    # Bootstrap CI using residual resampling
    if bootstrap_ci and n >= 10:
        boot_betas = []
        rng = np.random.default_rng()

        # Calculate fitted values and residuals for the WHOLE dataset
        # fitted = intercept + beta * x
        fitted = intercept_shifted + beta_hat * x_shifted
        resid_lower = lower - fitted
        resid_upper = upper - fitted

        for b in range(n_boot):
            try:
                resid_idx = rng.integers(0, n, n)
                y_boot_lower = fitted + resid_lower[resid_idx]
                y_boot_upper = fitted + resid_upper[resid_idx]

                # bracket_and_bisect handles normalization internally
                beta_b = bracket_and_bisect(x, y_boot_lower, y_boot_upper)

                if np.isfinite(beta_b) and abs(beta_b) < (abs(beta_hat) + 1) * 100:
                     boot_betas.append(beta_b)
            except Exception:
                pass

        if len(boot_betas) >= max(20, int(0.8 * n_boot)):
            lo = np.quantile(boot_betas, ci_alpha/2)
            hi = np.quantile(boot_betas, 1 - ci_alpha/2)
            result['ci_lower'] = float(lo)
            result['ci_upper'] = float(hi)
            result['bootstrap_samples'] = len(boot_betas)
        else:
            result['ci_lower'] = None
            result['ci_upper'] = None
            result['notes'].append(
                f'bootstrap produced only {len(boot_betas)} valid samples '
                f'(need >= {max(20, int(0.8 * n_boot))})'
            )
    return result


# ---------- Public wrapper (Seasonal) ----------
def seasonal_ats_slope(x: np.ndarray, y: np.ndarray, censored: np.ndarray, seasons: np.ndarray,
                       cen_type: Optional[np.ndarray] = None, lod: Optional[np.ndarray] = None,
                       bootstrap_ci: bool = True, n_boot: int = 500,
                       ci_alpha: float = 0.05) -> dict:
    """
    Compute Stratified ATS slope estimate for seasonal data.
    The overall slope is the root of the sum of the individual seasonal score functions.
    Returns dict with keys: beta, intercept, ci_lower, ci_upper, prop_censored, notes
    """
    x_min = np.min(x)
    x_range = np.ptp(x) if np.ptp(x) > 0 else 1.0
    x_norm = (x - x_min) / x_range

    unique_seasons = np.unique(seasons)

    # Pre-process data for each season:
    # We store the normalized x, lower, upper arrays for each season.
    season_data = {}
    slopes_hint_norm = []

    # Process full dataset to get interval bounds
    full_lower, full_upper = make_intervals(y, censored, cen_type=cen_type, lod=lod)

    for s in unique_seasons:
        idx = np.where(seasons == s)[0]
        if len(idx) < 2:
            continue

        x_s = x_norm[idx]
        lower_s = full_lower[idx]
        upper_s = full_upper[idx]
        season_data[s] = (x_s, lower_s, upper_s)

        # Collect slopes for initial bracket hint (using normalized x)
        detected_idx = np.where(np.isfinite(lower_s) & np.isfinite(upper_s) & (lower_s == upper_s))[0]
        if len(detected_idx) >= 2:
             for i in range(len(detected_idx)):
                for j in range(i + 1, len(detected_idx)):
                    xi, xj = x_s[detected_idx[i]], x_s[detected_idx[j]]
                    if not np.isclose(xj, xi):
                        yi, yj = lower_s[detected_idx[i]], lower_s[detected_idx[j]]
                        slopes_hint_norm.append((yj - yi) / (xj - xi))

    # Define the global score function (on normalized beta)
    def global_score_func(beta):
        total_score = 0
        for s in season_data:
            x_s, lower_s, upper_s = season_data[s]
            total_score += S_of_beta(beta, x_s, lower_s, upper_s)
        return total_score

    # Find the stratified ATS slope (normalized)
    beta_hat_norm = bracket_and_bisect_generic(global_score_func, slopes_hint_norm)

    # De-normalize beta
    beta_hat = beta_hat_norm / x_range

    # Estimate Intercept: Median of residuals across the ENTIRE dataset
    # r = y - beta * x. Use x_shifted for stability.
    x_shifted = x - x_min
    r_lower = full_lower - beta_hat * x_shifted
    r_upper = full_upper - beta_hat * x_shifted
    intercept_shifted = estimate_intercept_turnbull(r_lower, r_upper)
    intercept = intercept_shifted - beta_hat * x_min

    prop_cen = float(np.mean(censored))
    result = {'beta': beta_hat, 'intercept': intercept,
              'prop_censored': prop_cen, 'notes': []}

    # Stratified Bootstrap
    n = len(x)
    if bootstrap_ci and n >= 10:
        boot_betas = []
        rng = np.random.default_rng()

        # Calculate fitted values and residuals for the WHOLE dataset
        fitted = intercept_shifted + beta_hat * x_shifted
        resid_lower = full_lower - fitted
        resid_upper = full_upper - fitted

        for b in range(n_boot):
            try:
                # Stratified resampling of residuals
                boot_season_data = {}

                for s in season_data:
                    x_s_norm, _, _ = season_data[s]

                    # Original indices for this season
                    idx_s = np.where(seasons == s)[0]
                    n_s = len(idx_s)

                    # Resample residuals within this season
                    resampled_indices = rng.choice(idx_s, size=n_s, replace=True)

                    # Construct bootstrapped intervals for this season
                    # Note: x_s_norm is already normalized
                    # fitted values must be grabbed corresponding to idx_s
                    fitted_s = fitted[idx_s]
                    resid_lower_s_resampled = resid_lower[resampled_indices]
                    resid_upper_s_resampled = resid_upper[resampled_indices]

                    y_boot_lower_s = fitted_s + resid_lower_s_resampled
                    y_boot_upper_s = fitted_s + resid_upper_s_resampled

                    boot_season_data[s] = (x_s_norm, y_boot_lower_s, y_boot_upper_s)

                def boot_score_func(beta):
                    total_score = 0
                    for s in boot_season_data:
                        x_b, l_b, u_b = boot_season_data[s]
                        total_score += S_of_beta(beta, x_b, l_b, u_b)
                    return total_score

                # Solve for beta_boot (normalized)
                beta_b_norm = bracket_and_bisect_generic(boot_score_func, slopes_hint_norm)

                # De-normalize
                beta_b = beta_b_norm / x_range

                if np.isfinite(beta_b) and abs(beta_b) < (abs(beta_hat) + 1) * 100:
                     boot_betas.append(beta_b)

            except Exception:
                pass

        if len(boot_betas) >= max(20, int(0.8 * n_boot)):
            lo = np.quantile(boot_betas, ci_alpha/2)
            hi = np.quantile(boot_betas, 1 - ci_alpha/2)
            result['ci_lower'] = float(lo)
            result['ci_upper'] = float(hi)
            result['bootstrap_samples'] = len(boot_betas)
        else:
            result['ci_lower'] = None
            result['ci_upper'] = None
            result['notes'].append(
                f'bootstrap produced only {len(boot_betas)} valid samples '
                f'(need >= {max(20, int(0.8 * n_boot))})'
            )

    return result
