"""
Akritas-Theil-Sen (ATS) slope estimator for censored data.

The ATS slope (Akritas, Murphy & LaValley 1995; Helsel 2012, NADA::cenken) is
the value beta at which Kendall's S between time and the residuals
``y - beta * t`` is zero.  With censored values the residuals are intervals,
and a pair of residuals is only scored when the intervals do not overlap.

Exact algorithm used here
-------------------------
For a pair of observations ordered in time (t_i < t_j) with value intervals
[L_i, U_i] and [L_j, U_j], the residual of j is definitely above the residual
of i when ``beta < a_ij = (L_j - U_i) / (t_j - t_i)`` and definitely below when
``beta > b_ij = (U_j - L_i) / (t_j - t_i)`` (a_ij <= b_ij).  Hence

    S(beta) = #{a_ij > beta} - #{b_ij < beta},

a non-increasing step function.  Without censoring a_ij = b_ij is the pairwise
slope and the ATS slope is exactly Sen's slope.  Sorting the a's and b's lets
S(beta) be evaluated in O(log N) time.

* The estimate is the midpoint of the set where S(beta) = 0 (as NADA::cenken).
* The confidence interval inverts the censored Mann-Kendall test, i.e. the
  limits are where S(beta) = +/- z_{1-alpha/2} * sqrt(Var(S)), the same
  construction that gives Sen's rank-based interval for uncensored data.
* As in NADA::cenken, a censored value is treated as lying just beyond its
  detection limit, so an uncensored value equal to the limit is resolved.
* Seasonal (stratified) ATS sums the within-season scores.
* For very many pairs, pairs are sampled (proportionally across seasons) and
  the target levels are rescaled accordingly.
"""
import numpy as np
from typing import Optional
from scipy.stats import norm

from ._stats import _mk_score_and_var_censored, _get_min_positive_diff, _iter_triu_blocks


# ---------- interval representation ----------
def make_intervals(y, censored, cen_type=None, lod=None, nudge=True):
    """
    Value intervals [lower, upper] for each observation.

    Left-censored values become (-inf, DL) and right-censored values (DL, +inf).
    With ``nudge`` the open end is implemented by moving the limit by a tiny
    amount (1/1000 of the smallest gap between distinct values), so that an
    uncensored value equal to the detection limit is resolved, as in NADA.
    """
    y = np.asarray(y, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    lod = y if lod is None else np.asarray(lod, dtype=float)
    if cen_type is None:
        cen_type = np.where(censored, 'lt', 'not')
    cen_type = np.asarray(cen_type).astype(str)
    is_gt = censored & np.isin(cen_type, ['gt', '>', 'right'])
    is_lt = censored & ~is_gt

    delta = 0.0
    if nudge and censored.any():
        md = _get_min_positive_diff(np.unique(y))
        delta = md / 1000.0 if md > 0 else 1e-9 * max(1.0, np.nanmax(np.abs(y)))

    lower = y.copy()
    upper = y.copy()
    lower[is_lt] = -np.inf
    upper[is_lt] = lod[is_lt] - delta
    lower[is_gt] = lod[is_gt] + delta
    upper[is_gt] = np.inf
    return lower, upper


def S_of_beta(beta, x, lower, upper):
    """
    Kendall's S between time ``x`` and the residual intervals at slope ``beta``
    (positive when residuals increase with time).  Tied times score 0.
    Reference O(n^2) implementation; the estimator uses ``_ATSScore``.
    """
    x = np.asarray(x, dtype=float)
    lo = lower - beta * x
    hi = upper - beta * x
    i, j = np.triu_indices(len(x), k=1)
    sx = np.sign(x[j] - x[i])
    up = lo[j] > hi[i]      # residual j definitely above residual i
    down = hi[j] < lo[i]    # residual j definitely below residual i
    return int(np.sum(sx * (up.astype(int) - down.astype(int))))


# ---------- exact score function ----------
def _pair_thresholds(x, lower, upper, i, j):
    dx = x[j] - x[i]
    keep = dx != 0
    i, j, dx = i[keep], j[keep], dx[keep]
    fwd = dx > 0
    a_idx = np.where(fwd, j, i)   # later observation
    b_idx = np.where(fwd, i, j)   # earlier observation
    d = np.abs(dx)
    with np.errstate(invalid='ignore'):
        a = (lower[a_idx] - upper[b_idx]) / d
        b = (upper[a_idx] - lower[b_idx]) / d
    return a, b


class _ATSScore:
    """Sorted pair thresholds giving S(beta) for one or more strata."""

    def __init__(self, x, lower, upper, groups=None, max_pairs=None, random_state=None):
        x = np.asarray(x, dtype=float)
        groups = np.zeros(len(x), dtype=int) if groups is None else np.asarray(groups)
        idx_by_group = [np.where(groups == g)[0] for g in np.unique(groups)]
        n_pairs = [len(ix) * (len(ix) - 1) // 2 for ix in idx_by_group]
        self.total_pairs = int(sum(n_pairs))
        sample = max_pairs is not None and self.total_pairs > max_pairs
        rng = np.random.default_rng(0 if random_state is None else random_state)

        A, B = [], []
        used = 0
        for ix, npairs in zip(idx_by_group, n_pairs):
            if npairs == 0:
                continue
            if sample:
                from ._large_dataset import _sample_pairs
                k = int(np.ceil(max_pairs * npairs / self.total_pairs))
                ii, jj = _sample_pairs(len(ix), k, rng)
                blocks = [(ii, jj)]
            else:
                blocks = _iter_triu_blocks(len(ix))
            for ii, jj in blocks:
                i, j = ix[ii], ix[jj]
                used += len(i)
                a, b = _pair_thresholds(x, lower, upper, i, j)
                A.append(a)
                B.append(b)
        self.A = np.sort(np.concatenate(A)) if A else np.array([])
        self.B = np.sort(np.concatenate(B)) if B else np.array([])
        # Factor converting population-scale levels to sample-scale levels.
        self.scale = used / self.total_pairs if (sample and self.total_pairs) else 1.0
        self._P = None

    def S(self, beta):
        beta = np.asarray(beta, dtype=float)
        return (len(self.A) - np.searchsorted(self.A, beta, side='right')) \
            - np.searchsorted(self.B, beta, side='left')

    def solve(self, level):
        """
        Midpoint of {beta : S(beta) crosses ``level``}: (lo + hi) / 2 with
        lo = inf{beta: S(beta) <= level} and hi = sup{beta: S(beta) >= level}.
        ``level`` is on the population scale.
        """
        level = level * self.scale
        P = self._candidates()
        if len(P) == 0:
            return np.nan
        span = max(1.0, np.abs(P).max())
        m = len(P)
        n_locs = 2 * m + 1

        # Locations in increasing order: left tail, P0, gap0, P1, ...,
        # P_last, right tail.  S is non-increasing in beta, so S at the
        # locations is non-increasing in k and both crossings are found by
        # bisection.
        def loc(k):
            if k == 0:
                return P[0] - span
            if k == n_locs - 1:
                return P[-1] + span
            if k % 2 == 1:
                return P[(k - 1) // 2]
            return (P[k // 2 - 1] + P[k // 2]) / 2.0

        def s_at(k):
            return int(self.S(loc(k)))

        # k_lo: first k with S <= level.
        lo_k, hi_k = 0, n_locs
        while lo_k < hi_k:
            mid = (lo_k + hi_k) // 2
            if s_at(mid) <= level:
                hi_k = mid
            else:
                lo_k = mid + 1
        k_lo = lo_k
        # k_hi: last k with S >= level.
        lo_k, hi_k = -1, n_locs - 1
        while lo_k < hi_k:
            mid = (lo_k + hi_k + 1) // 2
            if s_at(mid) >= level:
                lo_k = mid
            else:
                hi_k = mid - 1
        k_hi = lo_k
        if k_lo >= n_locs or k_hi < 0:
            return np.nan
        if k_lo == 0 or k_hi == n_locs - 1:
            return np.nan  # crossing in an unbounded tail: not identifiable
        # Point locations are odd indices (P[(k-1)//2]); gaps are even indices.
        lo = P[(k_lo - 1) // 2] if k_lo % 2 == 1 else P[k_lo // 2 - 1]
        hi = P[(k_hi - 1) // 2] if k_hi % 2 == 1 else P[k_hi // 2]
        return 0.5 * (lo + hi)

    def _candidates(self):
        """Sorted distinct finite thresholds (computed once, reused)."""
        if self._P is None:
            P = np.concatenate([self.A, self.B])
            P = P[np.isfinite(P)]
            P.sort(kind='stable')   # two sorted runs: merged in linear time
            if len(P):
                P = P[np.concatenate(([True], P[1:] != P[:-1]))]
            self._P = P
        return self._P


# ---------- intercept ----------
def estimate_intercept_turnbull(residual_lower, residual_upper, tol=1e-6, max_iter=200):
    """
    Median of interval-censored residuals using the Turnbull NPMLE (EM).
    For large n (>2000) a cheap approximation is used: the median of the
    finite interval end points.
    """
    lo = np.asarray(residual_lower, dtype=float)
    hi = np.asarray(residual_upper, dtype=float)
    n_obs = len(lo)
    if n_obs > 2000:
        rep = np.where(np.isfinite(hi), hi, lo)
        rep = np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2, rep)
        return float(np.median(rep))

    endpoints = np.unique(np.concatenate([lo, hi]))
    cands = []
    for k in range(len(endpoints)):
        if np.isfinite(endpoints[k]):
            cands.append((endpoints[k], endpoints[k]))
        if k < len(endpoints) - 1:
            cands.append((endpoints[k], endpoints[k + 1]))
    if not cands:
        return 0.0
    cands = np.array(cands)
    contained = (cands[:, 0] >= lo[:, None]) & (cands[:, 1] <= hi[:, None])
    keep = contained.any(axis=0)
    cands, contained = cands[keep], contained[:, keep]
    p = np.full(len(cands), 1.0 / len(cands))
    for _ in range(max_iter):
        num = p * contained
        den = num.sum(axis=1, keepdims=True)
        den[den == 0] = 1.0
        p_new = (num / den).sum(axis=0) / n_obs
        if np.abs(p_new - p).sum() < tol:
            p = p_new
            break
        p = p_new
    k = int(np.searchsorted(np.cumsum(p), 0.5))
    k = min(k, len(p) - 1)
    a, b = cands[k]
    if a == b:
        return float(a)
    if np.isneginf(a):
        return float(b)
    if np.isposinf(b):
        return float(a)
    return float((a + b) / 2.0)


# ---------- public wrappers ----------
def _ats_core(x, y, censored, cen_type, lod, groups, var_s, ci_alpha, max_pairs, random_state,
              compute_intercept=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    if cen_type is None:
        cen_type = np.where(censored, 'lt', 'not')
    cen_type = np.asarray(cen_type).astype(str)
    lower, upper = make_intervals(y, censored, cen_type=cen_type, lod=lod)

    score = _ATSScore(x, lower, upper, groups=groups, max_pairs=max_pairs, random_state=random_state)
    beta = score.solve(0.0)

    result = {'beta': beta, 'intercept': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
              'sen_probability': (np.nan, np.nan, np.nan),
              'prop_censored': float(np.mean(censored)) if len(censored) else np.nan,
              'pairs_used': len(score.A), 'total_pairs': score.total_pairs, 'notes': []}
    if np.isnan(beta):
        result['notes'].append('ATS slope not identifiable (insufficient resolvable pairs)')
        return result

    x_min = np.min(x)
    if compute_intercept:
        result['intercept'] = estimate_intercept_turnbull(lower - beta * (x - x_min),
                                                          upper - beta * (x - x_min)) - beta * x_min

    if score.scale != 1.0 and len(score.A):
        # Sampling error of the estimate: the sampled score at beta has
        # standard deviation ~ total * sqrt(mean(score^2) / K); the estimate
        # moves by about the beta-range where the score stays within
        # +/- 1.96 of those standard deviations.
        k = len(score.A)
        nonzero = (k - np.searchsorted(score.A, beta, side='right')) \
            + np.searchsorted(score.B, beta, side='left')
        sd = score.total_pairs * np.sqrt(max(nonzero, 1) / k / k)
        c_s = 1.959964 * sd
        result['approximation_error'] = float((score.solve(-c_s) - score.solve(c_s)) / 2.0)

    if var_s is None:
        g = np.zeros(len(x), dtype=int) if groups is None else np.asarray(groups)
        var_s = 0.0
        for gv in np.unique(g):
            m = g == gv
            if m.sum() > 1:
                var_s += _mk_score_and_var_censored(y[m], x[m], censored[m], cen_type[m])[1]
    if var_s is not None and var_s > 0 and np.isfinite(var_s):
        c = norm.ppf(1 - ci_alpha / 2) * np.sqrt(var_s)
        result['ci_lower'] = score.solve(c)
        result['ci_upper'] = score.solve(-c)
        sd = np.sqrt(var_s)
        s0 = float(score.S(0.0)) / score.scale
        s0_plus = float(score.S(np.nextafter(0.0, 1.0))) / score.scale
        s0_minus = float(score.S(np.nextafter(0.0, -1.0))) / score.scale
        # Probability that the slope is below zero (cf. _sen_probability).
        result['sen_probability'] = (float(norm.cdf(-s0 / sd)), float(norm.cdf(-s0_plus / sd)),
                                     float(norm.cdf(-s0_minus / sd)))
    return result


def ats_slope(x: np.ndarray, y: np.ndarray, censored: np.ndarray,
              cen_type: Optional[np.ndarray] = None, lod: Optional[np.ndarray] = None,
              var_s: Optional[float] = None, ci_alpha: float = 0.05,
              max_pairs: Optional[int] = None, random_state: Optional[int] = None,
              compute_intercept: bool = True) -> dict:
    """
    Akritas-Theil-Sen slope with a test-inversion confidence interval.

    Args:
        x: Time values.
        y: Values (detection limit for censored rows).
        censored: Boolean censoring flags.
        cen_type: 'lt' / 'gt' / 'not' per row (default: censored rows are 'lt').
        lod: Detection limits (default: ``y``).
        var_s: Var(S) of the Mann-Kendall statistic used for the interval
            (default: the censored Mann-Kendall variance of the data).
        ci_alpha: Two-sided significance level of the interval.
        max_pairs: Sample at most this many pairs (None: all pairs).
        random_state: Seed for pair sampling.
        compute_intercept: Estimate the intercept (Turnbull median of the
            residuals); set False when only the slope is needed.

    Returns:
        dict with beta, intercept, ci_lower, ci_upper, sen_probability
        (P(slope<0) with median/max/min tie handling), prop_censored, notes.
    """
    return _ats_core(x, y, censored, cen_type, lod, None, var_s, ci_alpha, max_pairs, random_state,
                     compute_intercept)


def seasonal_ats_slope(x: np.ndarray, y: np.ndarray, censored: np.ndarray, seasons: np.ndarray,
                       cen_type: Optional[np.ndarray] = None, lod: Optional[np.ndarray] = None,
                       var_s: Optional[float] = None, ci_alpha: float = 0.05,
                       max_pairs: Optional[int] = None, random_state: Optional[int] = None,
                       compute_intercept: bool = True) -> dict:
    """
    Stratified (seasonal) ATS slope: the root of the sum of the within-season
    Kendall scores, with a confidence interval from inverting the seasonal
    Kendall test (Var(S) summed over seasons).  See ``ats_slope``.
    """
    return _ats_core(x, y, censored, cen_type, lod, seasons, var_s, ci_alpha, max_pairs, random_state,
                     compute_intercept)
