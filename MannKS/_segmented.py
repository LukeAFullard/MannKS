
from contextlib import contextmanager
import numpy as np
import pandas as pd
import piecewise_regression
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, t as t_dist
import warnings

from ._autocorr import estimate_acf, effective_sample_size, should_apply_correction
from ._bootstrap import (
    optimal_block_size,
    _moving_block_bootstrap_indices,
    block_bootstrap_confidence_intervals,
)


_REL_RSS_FLOOR = 1e-12  # Relative (1 - R^2) floor on standardized (unit-std) x_ols
_VALID_CRITERIA = ('bic', 'mbic', 'aic')
_VALID_BOOTSTRAP_METHODS = ('block', 'pairs', 'residual')
_VALID_AUTOCORR_METHODS = ('none', 'auto', 'yue_wang', 'block_bootstrap')


def _is_valid_int(val):
    """Return True if val is an integer (Python int or numpy integer) and not a bool."""
    return isinstance(val, (int, np.integer)) and not isinstance(val, bool)


def _validate_alpha(alpha):
    """Validate that alpha is a finite numeric value strictly in (0, 1)."""
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.number))
        or not np.isfinite(alpha)
        or not (0.0 < float(alpha) < 1.0)
    ):
        raise ValueError(
            f"alpha must be a finite float strictly between 0 and 1 (0 < alpha < 1). Got {alpha!r}."
        )
    return float(alpha)


def _make_seed_sequence(random_state):
    """Create a numpy SeedSequence from random_state, or return None if unseeded."""
    if random_state is None:
        return None
    if isinstance(random_state, np.random.SeedSequence):
        return random_state
    if isinstance(random_state, np.random.Generator):
        return np.random.SeedSequence(int(random_state.integers(0, 2**32 - 1, dtype=np.int64)))
    if isinstance(random_state, np.random.RandomState):
        return np.random.SeedSequence(int(random_state.randint(0, 2**32 - 1, dtype=np.int64)))
    if _is_valid_int(random_state):
        return np.random.SeedSequence(int(random_state))
    return np.random.SeedSequence(random_state)


@contextmanager
def _isolated_legacy_numpy_seed(seed):
    """
    Temporarily seed legacy global np.random without polluting caller RNG state.
    Needed because piecewise_regression.Fit relies on legacy np.random calls.

    Note: Because this saves and restores process-global ``np.random`` state,
    concurrent calls across threads in the same Python process are not thread-safe.
    """
    state = np.random.get_state()
    try:
        if seed is None:
            fresh_seed = int(np.random.SeedSequence().generate_state(1)[0])
            np.random.seed(fresh_seed)
        else:
            np.random.seed(int(seed) % (2**32))
        yield
    finally:
        np.random.set_state(state)


def _compute_information_criteria(rss_scaled, x_std, n_samples, n_breakpoints, n_eff=None):
    """
    Compute original-scale BIC, modified BIC (``mbic``), and AIC from
    standardized RSS with a symmetric scale-free relative floor (``_REL_RSS_FLOOR``).

    ``mbic`` uses a stronger complexity penalty of ``(2 + 3 * k) * log(n)``
    (3 * log(n) per breakpoint, inspired by Zhang & Siegmund 2007, omitting the
    segment-length term ``sum(log(n_i / n))``).

    When ``n_eff`` is provided (ESS-adjusted scoring under autocorrelation),
    the log-likelihood term is weighted by ``n_eff`` while parameter penalties
    use ``log(n)``. This is a practical effective-sample-size heuristic rather
    than a formal autoregressive likelihood criterion; under strong positive
    AR(1) autocorrelation on short series some residual false-breakpoint rate
    can remain.
    """
    k_int = int(n_breakpoints)
    k_params = 2 + 2 * k_int
    mbic_params = 2 + 3 * k_int
    rss_eff = max(float(rss_scaled), n_samples * _REL_RSS_FLOOR)
    n_ll = float(n_samples) if n_eff is None else float(n_eff)
    log_term = n_ll * np.log(rss_eff / n_samples) + 2.0 * n_ll * np.log(x_std)
    bic = log_term + k_params * np.log(n_samples)
    mbic = log_term + mbic_params * np.log(n_samples)
    aic = log_term + 2.0 * k_params
    return float(bic), float(mbic), float(aic)


def _check_min_segment_size(t_sorted, bps, min_segment_size):
    """
    Verify that every segment defined by ``[b_i, b_{i+1})`` (last segment closed)
    has at least ``min_segment_size`` observations.

    Returns ``(is_valid, min_count)``.
    """
    if min_segment_size is None or len(bps) == 0:
        return True, len(t_sorted)
    bps_sorted = np.sort(np.asarray(bps, dtype=float))
    bounds = np.concatenate(([float(t_sorted[0])], bps_sorted, [float(t_sorted[-1])]))
    n_segs = len(bounds) - 1
    min_cnt = len(t_sorted)
    for j in range(n_segs):
        if j == n_segs - 1:
            cnt = int(np.sum((t_sorted >= bounds[j]) & (t_sorted <= bounds[j + 1])))
        else:
            cnt = int(np.sum((t_sorted >= bounds[j]) & (t_sorted < bounds[j + 1])))
        if cnt < min_cnt:
            min_cnt = cnt
    return (min_cnt >= int(min_segment_size)), min_cnt


def _compute_fit_distance_kwargs(t_sorted, k_int, min_segment_size):
    """
    Derive ``min_distance_to_edge`` and ``min_distance_between_breakpoints``
    for ``piecewise_regression.Fit`` from ``min_segment_size``.

    Returns ``(fit_kwargs, tightened)`` where ``tightened`` is True only when
    at least one constraint strictly exceeds piecewise_regression's defaults
    (0.02 for edge, 0.01 between breakpoints).
    """
    fit_kwargs = {}
    tightened = False
    n_samples = len(t_sorted)
    if min_segment_size is None or int(min_segment_size) < 2:
        return fit_kwargs, tightened
    t_span_val = float(t_sorted[-1] - t_sorted[0])
    ms = int(min_segment_size)
    if t_span_val <= 0.0 or n_samples < (k_int + 1) * ms:
        return fit_kwargs, tightened

    left_frac = float(t_sorted[ms - 1] - t_sorted[0]) / t_span_val
    right_frac = float(t_sorted[-1] - t_sorted[-ms]) / t_span_val
    edge_frac = min(left_frac, right_frac)
    max_edge = 0.4 / max(1, k_int)
    if 0.0 < edge_frac < max_edge:
        val = max(0.02, edge_frac)
        fit_kwargs['min_distance_to_edge'] = val
        if val > 0.02:
            tightened = True
    if k_int >= 2 and ms >= 2:
        # Exact span of ms consecutive observations: t[i + ms - 1] - t[i]
        between_frac = float(np.min(t_sorted[ms - 1:] - t_sorted[:-(ms - 1)])) / t_span_val
        if 0.0 < between_frac < max_edge:
            val = max(0.01, between_frac)
            fit_kwargs['min_distance_between_breakpoints'] = val
            if val > 0.01:
                tightened = True
    return fit_kwargs, tightened


def _estimate_ar1_rho_deterministic(t_centered, x_scaled, max_k=3):
    """
    Estimate lag-1 AR(1) autocorrelation deterministically (zero RNG, independent
    of ``random_state`` and Phase-1 optimizer convergence) and invariantly to
    piecewise-linear hinges.

    1. Fits a deterministic greedy piecewise-linear basis on an interior quantile
       grid (up to ``min(max_k, 3, n // 10)`` knots) followed by one coordinate-
       descent pass to remove linear/hinged trend structure without overfitting
       AR(1) wander.
    2. Computes lag-1 autocorrelation of MAD-winsorized first differences of
       residuals ``de = diff(resid)`` and inverts the exact AR(1) identity
       ``r_1(Delta e) = (rho - 1) / 2`` -> ``rho_hat = 1 + 2 * r_1(Delta e)``.
    """
    n = len(x_scaled)
    if n < 10:
        return 0.0
    n_grid = min(25, max(5, n - 6))
    grid = np.unique(np.quantile(t_centered, np.linspace(0.1, 0.9, n_grid)))
    k_lim = min(int(max_k), 3, max(0, n // 10))
    selected = []

    X = np.column_stack([np.ones(n), t_centered])
    coef, _, _, _ = np.linalg.lstsq(X, x_scaled, rcond=None)
    resid = x_scaled - X @ coef
    best_rss = float(np.dot(resid, resid))

    for _ in range(k_lim):
        best_kn = None
        best_r = resid
        for kn in grid:
            if any(abs(kn - s) < 1e-9 for s in selected):
                continue
            X_try = np.column_stack([X, np.maximum(0.0, t_centered - kn)])
            c, _, _, _ = np.linalg.lstsq(X_try, x_scaled, rcond=None)
            r = x_scaled - X_try @ c
            rss = float(np.dot(r, r))
            if rss < best_rss:
                best_rss = rss
                best_kn = kn
                best_r = r
        if best_kn is None:
            break
        selected.append(best_kn)
        X = np.column_stack([X, np.maximum(0.0, t_centered - best_kn)])
        resid = best_r

    # One deterministic coordinate-descent refinement pass (handles symmetric W-shapes)
    if len(selected) >= 2:
        for j in range(len(selected)):
            other = [selected[m] for m in range(len(selected)) if m != j]
            X_base = np.column_stack(
                [np.ones(n), t_centered] + [np.maximum(0.0, t_centered - kn) for kn in other]
            )
            best_kn = selected[j]
            for kn in grid:
                if any(abs(kn - s) < 1e-9 for s in other):
                    continue
                X_try = np.column_stack([X_base, np.maximum(0.0, t_centered - kn)])
                c, _, _, _ = np.linalg.lstsq(X_try, x_scaled, rcond=None)
                r = x_scaled - X_try @ c
                rss = float(np.dot(r, r))
                if rss < best_rss:
                    best_rss = rss
                    best_kn = kn
                    resid = r
            selected[j] = best_kn

    if best_rss <= n * _REL_RSS_FLOOR:
        return 0.0

    de = np.diff(resid)
    if len(de) < 3:
        return 0.0
    med = float(np.median(de))
    mad = float(np.median(np.abs(de - med)))
    if mad > 0.0:
        sc = 1.4826 * mad
        de = np.clip(de, med - 3.5 * sc, med + 3.5 * sc)
    dec = de - float(np.mean(de))
    denom = float(np.dot(dec, dec))
    if denom <= 0.0:
        return 0.0
    r1_de = float(np.dot(dec[:-1], dec[1:]) / denom)
    return float(np.clip(1.0 + 2.0 * r1_de, -0.95, 0.95))


def _compute_muggeo_se(t_centered, x_scaled, bps_c):
    """
    Compute Muggeo (2003) linearized Delta-method standard errors for
    breakpoints ``bps_c`` on the standardized scale.
    """
    n = len(t_centered)
    k = len(bps_c)
    df = n - (2 + 2 * k)
    if df <= 0 or k == 0:
        return [np.nan] * k
    X = np.column_stack(
        [np.ones(n), t_centered] + [np.maximum(0.0, t_centered - b) for b in bps_c]
    )
    coef, _, _, _ = np.linalg.lstsq(X, x_scaled, rcond=None)
    resid = x_scaled - X @ coef
    rss = float(np.dot(resid, resid))
    sigma2 = rss / df
    betas = coef[2:]
    Z_psi = [(-betas[j]) * (t_centered > bps_c[j]).astype(float) for j in range(k)]
    Z = np.column_stack([X] + Z_psi)
    try:
        cov = sigma2 * np.linalg.pinv(Z.T @ Z)
        return [
            float(np.sqrt(max(float(cov[2 + k + j, 2 + k + j]), 0.0)))
            for j in range(k)
        ]
    except Exception:
        return [np.nan] * k


def _eval_profile_ols(bps_c, t_centered, x_scaled):
    """Evaluate exact profile-OLS fit for given sorted standardized breakpoints."""
    n = len(t_centered)
    X = np.column_stack(
        [np.ones(n), t_centered] + [np.maximum(0.0, t_centered - b) for b in bps_c]
    )
    coef, _, _, _ = np.linalg.lstsq(X, x_scaled, rcond=None)
    y_pred = X @ coef
    resid = x_scaled - y_pred
    rss = float(np.dot(resid, resid))
    return rss, y_pred, resid


def _solve_profile_ols_k(
    t_centered,
    x_scaled,
    t_orig,
    t_mean,
    t_std,
    k_int,
    min_segment_size,
    prev_bps_c=None,
    extra_starts=None,
):
    """
    Deterministic multi-start profile-OLS solver for ``k_int`` breakpoints with
    native ``min_segment_size`` enforcement.

    Combines uniform & shifted quantile grids, sequential warm-starts from
    ``prev_bps_c`` (the ``k-1`` solution), and optional ``extra_starts`` (e.g.,
    from ``piecewise_regression.Fit``), polishing the minimum-RSS valid
    solution with Nelder-Mead + Powell.

    Returns ``(best_rss_scaled, best_bps_c)`` or ``(np.inf, None)`` if no valid
    solution exists.
    """
    n = len(t_centered)
    tc_min, tc_max = float(t_centered[0]), float(t_centered[-1])
    if tc_max <= tc_min or n <= 2 + 2 * k_int:
        return np.inf, None
    if min_segment_size is not None and n < (k_int + 1) * int(min_segment_size):
        return np.inf, None

    def obj(bps):
        bps_s = np.sort(np.asarray(bps, dtype=float))
        if not np.all(np.isfinite(bps_s)):
            return 1e30
        if bps_s[0] <= tc_min or bps_s[-1] >= tc_max:
            return 1e30
        if k_int > 1 and np.any(np.diff(bps_s) <= 1e-9):
            return 1e30
        if min_segment_size is not None:
            bps_orig = bps_s * t_std + t_mean
            ok, _ = _check_min_segment_size(t_orig, bps_orig, min_segment_size)
            if not ok:
                return 1e30
        X = np.column_stack(
            [np.ones(n), t_centered] + [np.maximum(0.0, t_centered - b) for b in bps_s]
        )
        coef, _, _, _ = np.linalg.lstsq(X, x_scaled, rcond=None)
        resid = x_scaled - X @ coef
        return float(np.dot(resid, resid))

    starts = []
    # 1. Deterministic uniform quantile grid
    starts.append(np.quantile(t_centered, np.linspace(0.0, 1.0, k_int + 2)[1:-1]))
    # 2. Shifted quantile grid
    if k_int <= 3:
        starts.append(np.quantile(t_centered, np.linspace(0.15, 0.85, k_int)))
    # 3. Exact observation-rank split grid (crucial when n is close to (k+1)*min_segment_size, e.g. n=10, k=1)
    if min_segment_size is not None and n >= (k_int + 1) * int(min_segment_size):
        ms = int(min_segment_size)
        rank_bps = []
        for j in range(1, k_int + 1):
            idx_lo = min(n - 2, max(0, int(round(j * n / (k_int + 1))) - 1))
            idx_hi = min(n - 1, idx_lo + 1)
            rank_bps.append(0.5 * (float(t_centered[idx_lo]) + float(t_centered[idx_hi])))
        starts.append(np.asarray(rank_bps, dtype=float))
    # 4. Sequential warm-starts from k-1 solution
    if prev_bps_c is not None and len(prev_bps_c) == k_int - 1:
        bnds = [tc_min] + list(prev_bps_c) + [tc_max]
        for seg_i in range(len(bnds) - 1):
            for frac in (0.33, 0.5, 0.67):
                cand = bnds[seg_i] + frac * (bnds[seg_i + 1] - bnds[seg_i])
                starts.append(np.sort(list(prev_bps_c) + [cand]))
    # 5. Extra starts (e.g., from piecewise_regression.Fit)
    if extra_starts:
        for es in extra_starts:
            arr = np.sort(np.asarray(es, dtype=float))
            if len(arr) == k_int and np.all(np.isfinite(arr)):
                starts.append(arr)

    best_rss = np.inf
    best_bps = None
    for sv in starts:
        sv_s = np.sort(np.asarray(sv, dtype=float))
        # Direct evaluation first (in case sv is already optimal)
        val0 = obj(sv_s)
        if val0 < best_rss and val0 < 1e29:
            best_rss = val0
            best_bps = sv_s.copy()
        try:
            res = minimize(
                obj,
                sv_s,
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-10, 'maxiter': 500},
            )
            if np.isfinite(res.fun) and res.fun < best_rss and res.fun < 1e29:
                cand_bps = np.sort(np.asarray(res.x, dtype=float))
                if obj(cand_bps) < 1e29:
                    best_rss = float(res.fun)
                    best_bps = cand_bps
        except Exception:
            continue

    # One Powell polish from the best Nelder-Mead solution
    if best_bps is not None:
        try:
            res2 = minimize(
                obj,
                best_bps,
                method='Powell',
                options={'xtol': 1e-11, 'ftol': 1e-14, 'maxiter': 500},
            )
            if np.isfinite(res2.fun) and res2.fun < best_rss and res2.fun < 1e29:
                cand_bps = np.sort(np.asarray(res2.x, dtype=float))
                if obj(cand_bps) < 1e29:
                    best_rss = float(res2.fun)
                    best_bps = cand_bps
        except Exception:
            pass

    return best_rss, best_bps


def _safe_davies_test(t, x_scaled, k=10):
    """
    Safe wrapper around piecewise_regression.davies.davies_test.
    Guards n >= 10, non-finite statistics, and clamps p-value to [0.0, 1.0].
    """
    if len(t) < 10:
        return np.nan
    try:
        from piecewise_regression.davies import davies_test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_val = davies_test(t, x_scaled, k=k)
        if p_val is None or not np.isfinite(p_val):
            return np.nan
        return float(min(max(p_val, 0.0), 1.0))
    except Exception:
        return np.nan


def _bootstrap_breakpoints(
    t,
    x,
    n_breakpoints,
    n_bootstrap=100,
    alpha_n=0.05,
    random_state=None,
    bootstrap_method='block',
    block_size='auto',
    inner_n_boot=10,
    y_hat=None,
    min_segment_size=None,
):
    """
    Bootstrap the breakpoint detection to find robust breakpoint locations.

    Args:
        t: Time vector (sorted).
        x: Data vector (numeric, optionally standardized).
        n_breakpoints: Number of breakpoints to find.
        n_bootstrap: Number of bootstrap iterations.
        alpha_n: Significance level (retained for backward compatibility).
        random_state: Seed for random number generator.
        bootstrap_method: Resampling scheme ('block', 'pairs', or 'residual').
        block_size: Block length for 'block'/'residual' ('auto' or int >= 1).
        inner_n_boot: Number of internal Muggeo restarts per outer bootstrap replicate.
        y_hat: Optional Phase-1 fitted values for residual ACF / residual bootstrap.
        min_segment_size: Optional minimum number of observations required per segment
            on the original time vector ``t`` (``>= 2`` or ``None``).

    Returns:
        all_breakpoints: List of lists. Each inner list contains sorted finite
            breakpoints found in one bootstrap iteration, or [] if that replicate failed.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    if not _is_valid_int(n_breakpoints) or int(n_breakpoints) < 1:
        raise ValueError("n_breakpoints must be an integer >= 1.")
    if not _is_valid_int(n_bootstrap) or int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be an integer >= 1.")
    if not _is_valid_int(inner_n_boot) or int(inner_n_boot) < 0:
        raise ValueError("inner_n_boot must be an integer >= 0.")
    if not isinstance(bootstrap_method, str) or bootstrap_method.lower() not in _VALID_BOOTSTRAP_METHODS:
        raise ValueError(f"bootstrap_method must be one of {_VALID_BOOTSTRAP_METHODS}.")
    bootstrap_method = bootstrap_method.lower()

    if isinstance(block_size, str) and block_size.lower() == 'auto':
        block_size_norm = 'auto'
    elif _is_valid_int(block_size) and int(block_size) >= 1:
        block_size_norm = int(block_size)
    else:
        raise ValueError("block_size must be 'auto' or an integer >= 1.")

    k_int = int(n_breakpoints)
    n_boot_int = int(n_bootstrap)
    inner_boot_int = int(inner_n_boot)
    n = len(x)
    t_min, t_max = float(np.min(t)), float(np.max(t))

    # Compute residuals for auto block-size selection or residual bootstrap
    if bootstrap_method in ('block', 'residual'):
        if y_hat is not None:
            y_hat_arr = np.asarray(y_hat, dtype=float)
        else:
            t_std = float(np.std(t))
            if float(np.ptp(t)) > 0.0 and t_std > 0.0:
                t_c = (t - float(np.mean(t))) / t_std
                p = np.polyfit(t_c, x, 1)
                y_hat_arr = np.polyval(p, t_c)
            else:
                y_hat_arr = np.full(n, float(np.mean(x)), dtype=float)
        resid = x - y_hat_arr
    else:
        y_hat_arr = None
        resid = None

    if bootstrap_method == 'pairs':
        blk = 1
    elif block_size_norm == 'auto':
        acf, _ = estimate_acf(resid)
        if len(acf) > 1 and np.isfinite(acf[1]) and abs(acf[1]) > 1.96 / np.sqrt(n):
            blk = optimal_block_size(n, acf)
        else:
            blk = 1
    else:
        blk = block_size_norm

    # Deterministic RNG and isolated per-replicate legacy seeds
    seed_seq = _make_seed_sequence(random_state)
    if seed_seq is not None:
        children = seed_seq.spawn(1 + n_boot_int)
        rng = np.random.default_rng(children[0])
        rep_seeds = [int(c.generate_state(1)[0]) for c in children[1:]]
    else:
        rng = np.random.default_rng(None)
        rep_seeds = [None] * n_boot_int

    all_breakpoints = []

    for b in range(n_boot_int):
        if bootstrap_method == 'pairs' or (bootstrap_method == 'block' and blk == 1):
            indices = rng.choice(n, n, replace=True).astype(np.intp, copy=False)
            t_boot = t[indices]
            x_boot = x[indices]
            sort_idx = np.argsort(t_boot).astype(np.intp, copy=False)
            t_boot = t_boot[sort_idx]
            x_boot = x_boot[sort_idx]
        elif bootstrap_method == 'block':
            indices = _moving_block_bootstrap_indices(n, blk, rng)
            t_boot = t[indices]
            x_boot = x[indices]
            sort_idx = np.argsort(t_boot).astype(np.intp, copy=False)
            t_boot = t_boot[sort_idx]
            x_boot = x_boot[sort_idx]
        else:  # 'residual'
            indices = _moving_block_bootstrap_indices(n, blk, rng)
            t_boot = t
            x_boot = y_hat_arr + resid[indices]

        # Guard against rank-deficient resampled time grids
        if len(np.unique(t_boot)) < 2 * k_int + 2:
            all_breakpoints.append([])
            continue

        t_boot_mean = float(np.mean(t_boot))
        t_boot_std = float(np.std(t_boot))
        if t_boot_std > 0.0:
            t_boot_fit = (t_boot - t_boot_mean) / t_boot_std
        else:
            all_breakpoints.append([])
            continue

        boot_fit_kwargs, _ = _compute_fit_distance_kwargs(t_boot, k_int, min_segment_size)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", UserWarning)
                try:
                    from scipy.linalg import LinAlgWarning
                    warnings.simplefilter("ignore", LinAlgWarning)
                except ImportError:
                    pass
                with _isolated_legacy_numpy_seed(rep_seeds[b]):
                    pw_fit = piecewise_regression.Fit(
                        t_boot_fit,
                        x_boot,
                        n_breakpoints=k_int,
                        n_boot=inner_boot_int,
                        verbose=False,
                        **boot_fit_kwargs,
                    )

            res = pw_fit.get_results()
            bm = getattr(pw_fit, 'best_muggeo', None)
            bf = getattr(bm, 'best_fit', None) if bm is not None else None
            if not (bool(res.get('converged', False)) and bf is not None):
                all_breakpoints.append([])
                continue

            estimates = getattr(bf, 'estimates', None) or res.get('estimates')
            current_iter_bps = []
            if estimates:
                for key, val in estimates.items():
                    if key.startswith('breakpoint'):
                        est_c = val.get('estimate', np.nan)
                        if est_c is not None and np.isfinite(est_c):
                            est = float(est_c) * t_boot_std + t_boot_mean
                            if np.isfinite(est) and t_min < est < t_max:
                                current_iter_bps.append(est)

            if len(current_iter_bps) == k_int:
                current_iter_bps.sort()
                # Enforce min_segment_size on the original time grid (H2)
                seg_ok, _ = _check_min_segment_size(t, current_iter_bps, min_segment_size)
                if seg_ok:
                    all_breakpoints.append(current_iter_bps)
                else:
                    all_breakpoints.append([])
            else:
                all_breakpoints.append([])

        except Exception:
            all_breakpoints.append([])
            continue

    return all_breakpoints


def find_bagged_breakpoints(
    t,
    x,
    n_breakpoints,
    n_bootstrap=100,
    random_state=None,
    bootstrap_method='block',
    block_size='auto',
    inner_n_boot=10,
    y_hat=None,
    min_segment_size=None,
):
    """
    Find robust breakpoints using Bagging and KDE per breakpoint index.

    1. Bootstrap data and find breakpoints (keeping each replicate ordered).
    2. For each breakpoint index (1st, 2nd, ...), estimate the density of valid
       finite samples strictly inside (t.min(), t.max()).
    3. Find the peak (mode) of that density. If per-index KDE modes invert
       order (multimodal bootstrap distributions), fall back to order-preserving
       column medians with a UserWarning.

    Args:
        t: Time vector.
        x: Data vector.
        n_breakpoints: Number of breakpoints to identify.
        n_bootstrap: Number of bootstrap iterations.
        random_state: Seed for random number generator.
        bootstrap_method: Resampling scheme ('block', 'pairs', or 'residual').
        block_size: Block length ('auto' or int >= 1).
        inner_n_boot: Internal Muggeo restarts per bootstrap replicate.
        y_hat: Optional Phase-1 fitted values.
        min_segment_size: Optional minimum number of observations per segment
            enforced on each bootstrap replicate.

    Returns:
        robust_breakpoints: Array of monotonically ordered robust breakpoint locations.
        all_breakpoints: List of lists of all bootstrap breakpoint samples.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    if n_breakpoints == 0:
        return np.array([]), []

    k_int = int(n_breakpoints)
    t_min, t_max = float(np.min(t)), float(np.max(t))

    all_bps_structured = _bootstrap_breakpoints(
        t,
        x,
        k_int,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        bootstrap_method=bootstrap_method,
        block_size=block_size,
        inner_n_boot=inner_n_boot,
        y_hat=y_hat,
        min_segment_size=min_segment_size,
    )

    robust_bps = []
    valid_cols = []

    # Process column-wise (per breakpoint index)
    for i in range(k_int):
        col_samples = np.array(
            [
                bps[i]
                for bps in all_bps_structured
                if len(bps) == k_int and np.isfinite(bps[i]) and t_min < bps[i] < t_max
            ],
            dtype=float,
        )
        valid_cols.append(col_samples)

        if len(col_samples) < 2:
            if len(col_samples) == 1:
                robust_bps.append(float(col_samples[0]))
            else:
                robust_bps.append(np.nan)
            continue

        # KDE for this column
        try:
            kde = gaussian_kde(col_samples)

            min_val, max_val = col_samples.min(), col_samples.max()
            padding = (max_val - min_val) * 0.2
            if padding == 0:
                padding = 1e-5

            t_grid = np.linspace(min_val - padding, max_val + padding, 500)
            density = kde(t_grid)

            peak_idx = int(np.argmax(density))
            mode_val = float(t_grid[peak_idx])
            if not (np.isfinite(mode_val) and t_min < mode_val < t_max):
                mode_val = float(np.median(col_samples))
            robust_bps.append(mode_val)

        except Exception:
            # Fallback to median if KDE fails (e.g. singular matrix due to identical values)
            robust_bps.append(float(np.median(col_samples)))

    robust_bps_arr = np.array(robust_bps, dtype=float)

    # Enforce strict monotonic ascending order when multiple breakpoints exist
    if k_int > 1 and np.all(np.isfinite(robust_bps_arr)) and not np.all(np.diff(robust_bps_arr) > 0):
        warnings.warn(
            "Bagged KDE modes were out of order (multimodal bootstrap distribution); "
            "falling back to order-preserving column medians.",
            UserWarning,
        )
        robust_bps_arr = np.array(
            [float(np.median(col)) if len(col) > 0 else np.nan for col in valid_cols],
            dtype=float,
        )
        if np.all(np.isfinite(robust_bps_arr)):
            robust_bps_arr = np.sort(robust_bps_arr)

    return robust_bps_arr, all_bps_structured


class HybridSegmentedTrend:
    """
    Hybrid Segmented Regression.

    Phase 1 (Structure Discovery): Uses standardized multi-start profile OLS
    (combining ``piecewise-regression`` with deterministic quantile and
    sequential ``k-1`` warm-starts) to find the number and location of
    breakpoints. Optionally uses bagging for robust breakpoint locations
    (conditional on the Phase-1 selected number of breakpoints; ``bic_`` and
    ``aic_`` remain Phase-1 OLS values).

    Phase 2 (Robust Estimation): Uses Mann-Kendall / Sen's Slope on the
    identified segments to estimate robust slopes and confidence intervals.

    Note: Because ``piecewise-regression`` relies on legacy global ``np.random``
    calls internally, concurrent calls to ``fit`` from multiple threads in the
    same Python process are not thread-safe.
    """

    def __init__(
        self,
        max_breakpoints=5,
        n_breakpoints=None,
        use_bagging=False,
        n_bootstrap=100,
        criterion='bic',
        random_state=None,
        bootstrap_method='block',
        block_size='auto',
        inner_n_boot=10,
        autocorr_penalty='auto',
        use_davies_test=False,
        autocorr_method='none',
        min_segment_size=5,
        ci_method='direct',
        mk_test_method='robust',
        tau_method='b',
        tie_break_method='robust',
        continuous_confidence=True,
        category_map=None,
    ):
        """
        Args:
            max_breakpoints (int): Maximum number of breakpoints to search for (if n_breakpoints is None).
            n_breakpoints (int, optional): Fixed number of breakpoints to fit.
            use_bagging (bool): Use bootstrap aggregating for breakpoint location
                (conditional on the Phase-1 selected ``n_breakpoints``).
            n_bootstrap (int): Number of bootstrap samples if bagging is used (must be >= 2).
            criterion (str): Model selection criterion ('bic', 'mbic', or 'aic'). Default 'bic'.
                ``'mbic'`` uses a stronger ``3 * log(n)`` complexity penalty per breakpoint.
            random_state (int, optional): Seed for random number generator.
            bootstrap_method (str): Resampling scheme for bagging ('block', 'pairs', or 'residual').
            block_size (str or int): Block size for block/residual bagging ('auto' or int >= 1).
            inner_n_boot (int): Internal Muggeo restarts per outer bootstrap replicate. Default 10.
            autocorr_penalty (str or bool): ESS adjustment heuristic for autocorrelation in Phase-1
                selection ('auto', 'none', True, or False). Default 'auto'.
            use_davies_test (bool): Opt-in Davies test gate (tested at ``alpha``) for 0 vs >=1
                breakpoints during automatic model selection. Default False.
            autocorr_method (str): Optional Phase-2 per-segment autocorrelation adjustment for
                slope confidence intervals ('none', 'auto', 'yue_wang', or 'block_bootstrap').
                Default 'none'.
            min_segment_size (int or None): Minimum number of observations required in every
                segment for a ``k >= 1`` candidate fit to be accepted (``>= 2`` or ``None``
                to disable). Default 5.
            ci_method (str): Method for Phase-2 Sen slope CIs ('direct' or 'lwp'). Default 'direct'.
            mk_test_method (str): Method for Phase-2 Mann-Kendall variance ('robust' or 'lwp').
                Default 'robust'.
            tau_method (str): Kendall's tau method ('a' or 'b'). Default 'b'.
            tie_break_method (str): Tie-breaking method for censored MK ('robust' or 'lwp').
                Default 'robust' (matching ``trend_test``).
            continuous_confidence (bool): Use continuous confidence direction in MK trend
                classification. Default True.
            category_map (dict, optional): Optional confidence category map passed to
                ``classify_trend``.
        """
        if not _is_valid_int(max_breakpoints) or int(max_breakpoints) < 0:
            raise ValueError("max_breakpoints must be a non-negative integer.")
        if n_breakpoints is not None:
            if not _is_valid_int(n_breakpoints) or int(n_breakpoints) < 0:
                raise ValueError("n_breakpoints must be None or a non-negative integer.")
            n_breakpoints = int(n_breakpoints)

        if not isinstance(criterion, str) or criterion.lower() not in _VALID_CRITERIA:
            raise ValueError(f"criterion must be one of {_VALID_CRITERIA}. Got {criterion!r}.")

        if use_bagging:
            if not _is_valid_int(n_bootstrap) or int(n_bootstrap) < 2:
                raise ValueError("n_bootstrap must be an integer >= 2 when use_bagging=True.")

        if not isinstance(bootstrap_method, str) or bootstrap_method.lower() not in _VALID_BOOTSTRAP_METHODS:
            raise ValueError(f"bootstrap_method must be one of {_VALID_BOOTSTRAP_METHODS}. Got {bootstrap_method!r}.")

        if isinstance(block_size, str) and block_size.lower() == 'auto':
            block_size_norm = 'auto'
        elif _is_valid_int(block_size) and int(block_size) >= 1:
            block_size_norm = int(block_size)
        else:
            raise ValueError("block_size must be 'auto' or an integer >= 1.")

        if not _is_valid_int(inner_n_boot) or int(inner_n_boot) < 0:
            raise ValueError("inner_n_boot must be an integer >= 0.")

        if isinstance(autocorr_penalty, bool):
            autocorr_penalty_norm = autocorr_penalty
        elif isinstance(autocorr_penalty, str) and autocorr_penalty.lower() in ('auto', 'none'):
            autocorr_penalty_norm = autocorr_penalty.lower()
        else:
            raise ValueError("autocorr_penalty must be 'auto', 'none', True, or False.")

        if not isinstance(autocorr_method, str) or autocorr_method.lower() not in _VALID_AUTOCORR_METHODS:
            raise ValueError(
                f"autocorr_method must be one of {_VALID_AUTOCORR_METHODS}. Got {autocorr_method!r}."
            )

        if min_segment_size is not None:
            if not _is_valid_int(min_segment_size) or int(min_segment_size) < 2:
                raise ValueError("min_segment_size must be None or an integer >= 2.")
            min_segment_size = int(min_segment_size)

        if not isinstance(ci_method, str) or ci_method.lower() not in ('direct', 'lwp'):
            raise ValueError(f"ci_method must be 'direct' or 'lwp'. Got {ci_method!r}.")
        if not isinstance(mk_test_method, str) or mk_test_method.lower() not in ('robust', 'lwp'):
            raise ValueError(f"mk_test_method must be 'robust' or 'lwp'. Got {mk_test_method!r}.")
        if not isinstance(tau_method, str) or tau_method.lower() not in ('a', 'b'):
            raise ValueError(f"tau_method must be 'a' or 'b'. Got {tau_method!r}.")
        if not isinstance(tie_break_method, str) or tie_break_method.lower() not in ('robust', 'lwp'):
            raise ValueError(f"tie_break_method must be 'robust' or 'lwp'. Got {tie_break_method!r}.")

        self.max_breakpoints = int(max_breakpoints)
        self.n_breakpoints = n_breakpoints
        self.use_bagging = bool(use_bagging)
        self.n_bootstrap = int(n_bootstrap) if _is_valid_int(n_bootstrap) else n_bootstrap
        self.criterion = criterion.lower()
        self.random_state = random_state
        self.bootstrap_method = bootstrap_method.lower()
        self.block_size = block_size_norm
        self.inner_n_boot = int(inner_n_boot)
        self.autocorr_penalty = autocorr_penalty_norm
        self.use_davies_test = bool(use_davies_test)
        self.autocorr_method = autocorr_method.lower()
        self.min_segment_size = min_segment_size
        self.ci_method = ci_method.lower()
        self.mk_test_method = mk_test_method.lower()
        self.tau_method = tau_method.lower()
        self.tie_break_method = tie_break_method.lower()
        self.continuous_confidence = bool(continuous_confidence)
        self.category_map = category_map

        self.breakpoints_ = None
        self.breakpoint_cis_ = None
        self.segments_ = None
        self.n_breakpoints_ = None
        self.bic_ = None
        self.aic_ = None
        self.mbic_ = None
        self.score_ = None
        self.selection_summary_ = None
        self.bootstrap_samples_ = None
        self.bootstrap_failure_rate_ = None
        self.computation_mode_ = 'hybrid'
        self.segment_notes_ = []

    def fit(self, t, x, censored=None, cen_type=None, lt_mult=0.5, gt_mult=1.1, alpha=0.05,
            large_dataset_mode='auto', max_pairs=None, sens_slope_method='ats'):
        """
        Fit breakpoints (OLS, piecewise-regression) and then estimate each
        segment's trend and slope robustly (Mann-Kendall + Sen's slope;
        Akritas-Theil-Sen for censored segments unless ``sens_slope_method`` is
        'lwp'/'drop_ambiguous').

        Segment confidence intervals are conditional on the estimated
        breakpoints: they do not include breakpoint-location uncertainty.
        """
        alpha = _validate_alpha(alpha)
        from ._stats import _rle_lengths

        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)

        # Sort data
        sort_idx = np.argsort(t).astype(np.intp, copy=False)
        t = t[sort_idx]
        x = x[sort_idx]

        if censored is not None:
            censored = np.asarray(censored, dtype=bool)[sort_idx]
            cen_type = np.asarray(cen_type)[sort_idx]
            # Warn if multiplicative substitution will invert censoring direction on negative limits (F-14)
            lt_neg = (cen_type == 'lt') & (x < 0)
            if np.any(lt_neg) and lt_mult != 1.0:
                warnings.warn(
                    f"Detected {int(np.sum(lt_neg))} negative left-censored value(s) (< DL with DL < 0). "
                    f"Multiplicative Phase-1 OLS substitution (lt_mult={lt_mult}) shifts negative limits "
                    f"upward (e.g., {lt_mult} * -5 = {lt_mult * -5:g} > -5), inverting the censoring direction.",
                    UserWarning,
                )
            gt_neg = (cen_type == 'gt') & (x < 0)
            if np.any(gt_neg) and gt_mult != 1.0:
                warnings.warn(
                    f"Detected {int(np.sum(gt_neg))} negative right-censored value(s) (> DL with DL < 0). "
                    f"Multiplicative Phase-1 OLS substitution (gt_mult={gt_mult}) shifts negative limits "
                    f"downward (e.g., {gt_mult} * -5 = {gt_mult * -5:g} < -5), inverting the censoring direction.",
                    UserWarning,
                )
            # OLS needs numeric x. Substitute censored values.
            x_ols = x.copy().astype(float)
            x_ols[cen_type == 'lt'] *= lt_mult
            x_ols[cen_type == 'gt'] *= gt_mult
        else:
            x_ols = x.astype(float, copy=False)

        n_samples = len(x_ols)

        # --- Phase 1 Censoring Diagnostics (F-07) ---
        if censored is not None and n_samples > 0:
            censored_frac = float(np.mean(censored))
            if censored_frac > 0.15:
                warnings.warn(
                    f"High proportion of censored data ({100.0 * censored_frac:.1f}% > 15%). "
                    f"Phase-1 OLS breakpoint detection uses substituted values "
                    f"(lt_mult={lt_mult}, gt_mult={gt_mult}), which may induce spurious "
                    f"breakpoints or over-splitting.",
                    UserWarning,
                )
            min_run = max(5, int(0.15 * n_samples))
            for ctype, label in (('lt', 'left-censored floor'), ('gt', 'right-censored ceiling')):
                is_c = (cen_type == ctype)
                if np.any(is_c):
                    lengths = _rle_lengths(is_c)
                    starts = np.concatenate(([np.intp(0)], np.cumsum(lengths[:-1], dtype=np.intp)))
                    true_runs = lengths[is_c[starts]]
                    max_run = int(np.max(true_runs)) if len(true_runs) > 0 else 0
                    if max_run >= min_run:
                        warnings.warn(
                            f"Detected a contiguous {label} of {max_run} observations "
                            f"(>= {min_run}); flat censoring shelves can induce spurious "
                            f"breakpoints in Phase-1 OLS.",
                            UserWarning,
                        )

        # --- Phase 1: Structure Discovery ---
        best_n = 0
        best_bps = []
        best_bp_cis = []
        best_bic = np.nan
        best_aic = np.nan
        best_mbic = np.nan
        best_score = np.nan
        best_resid_scaled = None
        best_y_pred_scaled = None
        x_scaled = None
        self.bootstrap_samples_ = None
        self.bootstrap_failure_rate_ = None
        self.computation_mode_ = 'hybrid'
        self.segment_notes_ = []

        selection_records = []

        if self.n_breakpoints is not None:
            n_range = [int(self.n_breakpoints)]
        else:
            n_range = list(range(self.max_breakpoints + 1))

        x_mean = float(np.mean(x_ols))
        x_std = float(np.std(x_ols))
        t_mean = float(np.mean(t))
        t_std = float(np.std(t))

        # Spawn deterministic seed sequences for Phase 1, optional bagging, and Phase 2
        fit_seed_seq = _make_seed_sequence(self.random_state)
        if fit_seed_seq is not None:
            phase1_seq, bag_seq, phase2_seq = fit_seed_seq.spawn(3)
        else:
            phase1_seq, bag_seq, phase2_seq = None, None, None

        if float(np.ptp(t)) == 0.0 or t_std == 0.0:
            warnings.warn("All time values are identical; cannot fit breakpoints.", UserWarning)
            for k in n_range:
                selection_records.append({
                    'n_breakpoints': int(k), 'bic': np.nan, 'aic': np.nan,
                    'mbic': np.nan, 'score': np.nan, 'rss': np.nan, 'sar': np.nan, 'converged': False,
                })
        elif float(np.ptp(x_ols)) == 0.0 or x_std == 0.0:
            if self.n_breakpoints is None or int(self.n_breakpoints) == 0:
                if self.n_breakpoints is None:
                    warnings.warn(
                        "Input series is constant (zero variance); skipping breakpoint search "
                        "and returning 0 breakpoints.",
                        UserWarning,
                    )
                selection_records.append({
                    'n_breakpoints': 0, 'bic': -np.inf, 'aic': -np.inf,
                    'mbic': -np.inf, 'score': -np.inf, 'rss': 0.0, 'sar': 0.0, 'converged': True,
                })
                best_n = 0
                best_bic = -np.inf
                best_aic = -np.inf
                best_mbic = -np.inf
                best_score = -np.inf
            else:
                k_req = int(self.n_breakpoints)
                warnings.warn(
                    f"Input series is constant (zero variance); cannot fit requested "
                    f"n_breakpoints={k_req}. Falling back to 0-breakpoint model.",
                    UserWarning,
                )
                selection_records.append({
                    'n_breakpoints': k_req, 'bic': np.nan, 'aic': np.nan,
                    'mbic': np.nan, 'score': np.nan, 'rss': np.nan, 'sar': np.nan, 'converged': False,
                })
        else:
            x_scaled = (x_ols - x_mean) / x_std
            t_centered = (t - t_mean) / t_std

            candidates = []
            k_seeds = (
                [int(c.generate_state(1)[0]) for c in phase1_seq.spawn(len(n_range))]
                if phase1_seq is not None
                else [None] * len(n_range)
            )
            prev_bps_c = None
            prev_rss_scaled = np.inf
            failed_k_warnings = []

            # For fixed n_breakpoints >= 2, build deterministic sequential warm-start chain 1..k-1
            if self.n_breakpoints is not None and int(self.n_breakpoints) >= 2:
                k_target = int(self.n_breakpoints)
                if (
                    n_samples > 2 + 2 * k_target
                    and (self.min_segment_size is None or n_samples >= (k_target + 1) * self.min_segment_size)
                ):
                    warm_bps = None
                    for k_warm in range(1, k_target):
                        _, warm_bps = _solve_profile_ols_k(
                            t_centered, x_scaled, t, t_mean, t_std,
                            k_warm, self.min_segment_size, prev_bps_c=warm_bps,
                        )
                    prev_bps_c = warm_bps

            for idx_k, k in enumerate(n_range):
                k_int = int(k)
                record = {
                    'n_breakpoints': k_int, 'bic': np.nan, 'aic': np.nan,
                    'mbic': np.nan, 'score': np.nan, 'rss': np.nan, 'sar': np.nan, 'converged': False,
                }
                try:
                    if k_int == 0:
                        p = np.polyfit(t_centered, x_scaled, 1)
                        y_pred_scaled = np.polyval(p, t_centered)
                        resid_scaled = x_scaled - y_pred_scaled
                        rss_scaled = float(np.sum(resid_scaled ** 2))
                        rss_orig = rss_scaled * (x_std ** 2)
                        bic, mbic, aic = _compute_information_criteria(
                            rss_scaled, x_std, n_samples, 0
                        )
                        record.update({
                            'bic': bic, 'aic': aic, 'mbic': mbic,
                            'rss': rss_orig, 'sar': rss_orig, 'converged': True,
                        })
                        candidates.append({
                            'k': 0,
                            'rss_scaled': rss_scaled,
                            'bic': bic,
                            'mbic': mbic,
                            'aic': aic,
                            'bps': [],
                            'bp_cis': [],
                            'y_pred_scaled': y_pred_scaled,
                            'resid_scaled': resid_scaled,
                            'record': record,
                        })
                        prev_rss_scaled = rss_scaled
                        prev_bps_c = []
                        selection_records.append(record)
                        # Short-circuit exact noiseless linear series during auto-select only
                        if self.n_breakpoints is None and rss_scaled <= n_samples * _REL_RSS_FLOOR:
                            break
                        continue

                    # Guard degrees of freedom for k >= 1
                    if n_samples <= 2 + 2 * k_int:
                        warnings.warn(
                            f"Insufficient observations (n={n_samples}) for "
                            f"n_breakpoints={k_int} (requires > {2 + 2 * k_int}).",
                            UserWarning,
                        )
                        selection_records.append(record)
                        continue

                    # Guard minimum total observations required for min_segment_size (F-09)
                    if (
                        self.min_segment_size is not None
                        and n_samples < (k_int + 1) * self.min_segment_size
                    ):
                        warnings.warn(
                            f"Insufficient observations (n={n_samples}) for "
                            f"n_breakpoints={k_int} with min_segment_size={self.min_segment_size} "
                            f"(requires >= {(k_int + 1) * self.min_segment_size}).",
                            UserWarning,
                        )
                        selection_records.append(record)
                        continue

                    fit_kwargs, tightened = _compute_fit_distance_kwargs(
                        t, k_int, self.min_segment_size
                    )

                    # 1. Stochastic Muggeo exploration via piecewise_regression.Fit
                    extra_starts = []
                    pw_best_rss = np.inf
                    pw_best_bps_c = None
                    try:
                        with _isolated_legacy_numpy_seed(k_seeds[idx_k]):
                            pw_fit = piecewise_regression.Fit(
                                t_centered, x_scaled, n_breakpoints=k_int,
                                n_boot=25, verbose=False, **fit_kwargs,
                            )
                        res = pw_fit.get_results()
                        bm = getattr(pw_fit, 'best_muggeo', None)
                        bf = getattr(bm, 'best_fit', None) if bm is not None else None
                        if bool(res.get('converged', False)) and bf is not None:
                            estimates = getattr(bf, 'estimates', None) or res.get('estimates') or {}
                            pw_bps_c = sorted(
                                float(val['estimate'])
                                for key, val in estimates.items()
                                if key.startswith('breakpoint')
                                and val.get('estimate') is not None
                                and np.isfinite(val['estimate'])
                            )
                            if len(pw_bps_c) == k_int:
                                pw_bps_orig = [b * t_std + t_mean for b in pw_bps_c]
                                seg_ok_pw, _ = _check_min_segment_size(
                                    t, pw_bps_orig, self.min_segment_size
                                )
                                if seg_ok_pw:
                                    extra_starts.append(pw_bps_c)
                                    rss_pw_eval, _, _ = _eval_profile_ols(
                                        pw_bps_c, t_centered, x_scaled
                                    )
                                    pw_best_rss = rss_pw_eval
                                    pw_best_bps_c = np.asarray(pw_bps_c, dtype=float)
                    except Exception:
                        pass

                    # 2. Deterministic multi-start + sequential warm-start profile-OLS solver
                    #    with native min_segment_size enforcement (H1, L1)
                    prof_rss, prof_bps_c = _solve_profile_ols_k(
                        t_centered,
                        x_scaled,
                        t,
                        t_mean,
                        t_std,
                        k_int,
                        self.min_segment_size,
                        prev_bps_c=prev_bps_c,
                        extra_starts=extra_starts,
                    )

                    # Pick whichever valid solution has strictly lower RSS
                    if prof_bps_c is not None and prof_rss <= pw_best_rss:
                        best_bps_c = prof_bps_c
                        rss_scaled = float(prof_rss)
                    elif pw_best_bps_c is not None:
                        best_bps_c = pw_best_bps_c
                        rss_scaled = float(pw_best_rss)
                    else:
                        best_bps_c = None
                        rss_scaled = np.inf

                    # Monotonicity guard: if RSS(k) > RSS(k-1), try inserting a knot at the
                    # midpoint of the highest-RSS segment of k-1
                    if (
                        best_bps_c is not None
                        and prev_bps_c is not None
                        and len(prev_bps_c) == k_int - 1
                        and rss_scaled > prev_rss_scaled + 1e-12
                    ):
                        _, _, prev_resid = (
                            _eval_profile_ols(prev_bps_c, t_centered, x_scaled)
                            if len(prev_bps_c) > 0
                            else (prev_rss_scaled, None, x_scaled - np.polyval(np.polyfit(t_centered, x_scaled, 1), t_centered))
                        )
                        bnds_c = [float(t_centered[0])] + list(prev_bps_c) + [float(t_centered[-1])]
                        seg_rss_vals = []
                        for sj in range(len(bnds_c) - 1):
                            if sj == len(bnds_c) - 2:
                                sm = (t_centered >= bnds_c[sj]) & (t_centered <= bnds_c[sj + 1])
                            else:
                                sm = (t_centered >= bnds_c[sj]) & (t_centered < bnds_c[sj + 1])
                            seg_rss_vals.append(float(np.sum(prev_resid[sm] ** 2)) if np.any(sm) else -1.0)
                        worst_seg = int(np.argmax(seg_rss_vals))
                        extra_mid = [
                            np.sort(list(prev_bps_c) + [bnds_c[worst_seg] + frac * (bnds_c[worst_seg + 1] - bnds_c[worst_seg])])
                            for frac in (0.25, 0.5, 0.75)
                        ]
                        retry_rss, retry_bps_c = _solve_profile_ols_k(
                            t_centered, x_scaled, t, t_mean, t_std,
                            k_int, self.min_segment_size, extra_starts=extra_mid,
                        )
                        if retry_bps_c is not None and retry_rss < rss_scaled:
                            best_bps_c = retry_bps_c
                            rss_scaled = float(retry_rss)

                    if best_bps_c is None or not np.isfinite(rss_scaled) or rss_scaled < 0:
                        constraint_note = (
                            f" (with min_segment_size={self.min_segment_size})"
                            if tightened
                            else ""
                        )
                        failed_k_warnings.append(
                            f"Breakpoint fit for n_breakpoints={k_int} did not converge"
                            f"{constraint_note}."
                        )
                        selection_records.append(record)
                        continue

                    best_bps_c = np.sort(np.asarray(best_bps_c, dtype=float))
                    cand_bps = [float(b * t_std + t_mean) for b in best_bps_c]

                    # Final min_segment_size verification on original time vector
                    seg_ok, min_seg_n = _check_min_segment_size(t, cand_bps, self.min_segment_size)
                    if not seg_ok:
                        failed_k_warnings.append(
                            f"Breakpoint fit for n_breakpoints={k_int} produced a segment "
                            f"with {min_seg_n} observation(s) (< min_segment_size="
                            f"{self.min_segment_size}); rejecting candidate."
                        )
                        selection_records.append(record)
                        continue

                    # Exact profile-OLS predictions & Muggeo (2003) Delta-method CIs
                    rss_scaled, y_pred_scaled, resid_scaled = _eval_profile_ols(
                        best_bps_c, t_centered, x_scaled
                    )
                    se_c_list = _compute_muggeo_se(t_centered, x_scaled, best_bps_c)
                    df = n_samples - (2 + 2 * k_int)
                    t_crit = float(t_dist.ppf(1 - alpha / 2, df)) if df > 0 else np.nan
                    cand_bp_cis = []
                    for est_val, se_c in zip(cand_bps, se_c_list):
                        se_val = float(se_c) * t_std if np.isfinite(se_c) else np.nan
                        if np.isfinite(est_val) and np.isfinite(se_val) and np.isfinite(t_crit):
                            cand_bp_cis.append((float(est_val - t_crit * se_val), float(est_val + t_crit * se_val)))
                        else:
                            cand_bp_cis.append((np.nan, np.nan))

                    bic, mbic, aic = _compute_information_criteria(
                        rss_scaled, x_std, n_samples, k_int
                    )
                    rss_orig = float(rss_scaled) * (x_std ** 2)
                    record.update({
                        'bic': bic, 'aic': aic, 'mbic': mbic,
                        'rss': rss_orig, 'sar': rss_orig, 'converged': True,
                    })
                    candidates.append({
                        'k': k_int,
                        'rss_scaled': float(rss_scaled),
                        'bic': bic,
                        'mbic': mbic,
                        'aic': aic,
                        'bps': cand_bps,
                        'bp_cis': cand_bp_cis,
                        'y_pred_scaled': y_pred_scaled,
                        'resid_scaled': resid_scaled,
                        'record': record,
                    })
                    prev_bps_c = best_bps_c
                    prev_rss_scaled = float(rss_scaled)
                except Exception as exc:
                    record['bic'] = np.nan
                    record['aic'] = np.nan
                    record['mbic'] = np.nan
                    record['score'] = np.nan
                    record['rss'] = np.nan
                    record['sar'] = np.nan
                    record['converged'] = False
                    warnings.warn(
                        f"Breakpoint fit for n_breakpoints={k_int} failed: {exc}",
                        UserWarning,
                    )

                selection_records.append(record)

            # Emit non-convergence warnings when a fixed k failed, or when no k >= 1 converged
            # during auto-search (avoids routine noise for excess k=4,5 on clean low-k series, L6).
            k_pos_converged = any(c['k'] > 0 for c in candidates)
            if failed_k_warnings and (self.n_breakpoints is not None or not k_pos_converged):
                for msg in failed_k_warnings:
                    warnings.warn(msg, UserWarning)

            # --- Score converged candidates (with deterministic ESS-BIC & optional Davies gate) ---
            if not candidates:
                best_n = 0
                best_bps = []
                best_bp_cis = []
                best_bic = np.nan
                best_aic = np.nan
                best_mbic = np.nan
                best_score = np.nan
                if self.n_breakpoints is not None and int(self.n_breakpoints) > 0:
                    warnings.warn(
                        f"Requested fixed n_breakpoints={int(self.n_breakpoints)} did not "
                        f"converge; falling back to 0-breakpoint model.",
                        UserWarning,
                    )
            else:
                n_eff = None
                ess_applied = False
                rho_ess_est = 0.0
                if (
                    self.n_breakpoints is None
                    and self.autocorr_penalty in ('auto', True)
                    and len(candidates) > 1
                    and n_samples >= 10
                ):
                    # Deterministic, hinge-invariant AR(1) rho estimator (H1, M1):
                    # independent of random_state and Phase-1 optimizer convergence.
                    max_k_cand = max(c['k'] for c in candidates)
                    rho_ess_est = _estimate_ar1_rho_deterministic(
                        t_centered, x_scaled, max_k=max(1, min(max_k_cand, 3))
                    )
                    acf_thresh = max(0.20, 2.2 / np.sqrt(n_samples))
                    if rho_ess_est > acf_thresh:
                        rho_clamped = min(rho_ess_est, 0.95)
                        n_eff = max(
                            3.0,
                            n_samples * (1.0 - rho_clamped) / (1.0 + rho_clamped),
                        )
                        ess_applied = True

                best_cand = None
                best_cand_score = np.inf
                for cand in candidates:
                    s_bic, s_mbic, s_aic = _compute_information_criteria(
                        cand['rss_scaled'], x_std, n_samples, cand['k'], n_eff=n_eff
                    )
                    score = {'bic': s_bic, 'mbic': s_mbic, 'aic': s_aic}[self.criterion]
                    cand['score'] = score
                    cand['record']['score'] = score
                    if np.isfinite(score) and score < best_cand_score:
                        best_cand_score = score
                        best_cand = cand

                # Opt-in Davies test gate (only during auto-select when k > 0 selected)
                davies_gated = False
                if (
                    self.use_davies_test
                    and self.n_breakpoints is None
                    and best_cand is not None
                    and best_cand['k'] > 0
                ):
                    p_davies = _safe_davies_test(t_centered, x_scaled, k=10)
                    if np.isfinite(p_davies) and p_davies >= alpha:
                        k0_cands = [c for c in candidates if c['k'] == 0]
                        if k0_cands:
                            best_cand = k0_cands[0]
                            davies_gated = True
                            warnings.warn(
                                f"Davies test for change in slope was not significant "
                                f"(p={p_davies:.4g} >= alpha={alpha}); gating selection "
                                f"to 0 breakpoints.",
                                UserWarning,
                            )

                if davies_gated:
                    for rec in selection_records:
                        rec['davies_gated'] = (rec['n_breakpoints'] == 0)

                if best_cand is not None:
                    best_n = best_cand['k']
                    best_bps = list(best_cand['bps'])
                    best_bp_cis = list(best_cand['bp_cis'])
                    best_bic = best_cand['bic']
                    best_aic = best_cand['aic']
                    best_mbic = best_cand['mbic']
                    best_score = best_cand['score']
                    best_resid_scaled = best_cand['resid_scaled']
                    best_y_pred_scaled = best_cand['y_pred_scaled']

                # Selected-model residual lag-1 ACF diagnostic warning (F-07)
                if (
                    best_cand is not None
                    and best_resid_scaled is not None
                    and best_cand['rss_scaled'] > n_samples * _REL_RSS_FLOOR
                    and n_samples >= 10
                ):
                    acf_sel, _ = estimate_acf(best_resid_scaled, max_lag=1)
                    rho_1_sel = (
                        float(acf_sel[1])
                        if len(acf_sel) > 1 and np.isfinite(acf_sel[1])
                        else 0.0
                    )
                    acf_thresh = max(0.2, 1.96 / np.sqrt(n_samples))
                    if abs(rho_1_sel) > acf_thresh:
                        if ess_applied:
                            penalty_note = (
                                f"ESS-adjusted criterion (n_eff={n_eff:.1f}, rho_d1={rho_ess_est:.3f}) "
                                f"was applied during model selection."
                            )
                        elif rho_1_sel < 0 and self.autocorr_penalty in ('auto', True):
                            penalty_note = (
                                "ESS-BIC only penalizes positive autocorrelation and was not applied for negative autocorrelation."
                            )
                        else:
                            penalty_note = (
                                "Consider autocorr_penalty='auto' or criterion='mbic' to guard against false breakpoints."
                            )
                        warnings.warn(
                            f"Significant residual lag-1 autocorrelation detected "
                            f"(rho_1={rho_1_sel:.3f}, threshold={acf_thresh:.3f}). "
                            f"Serial correlation can inflate false breakpoints and narrow "
                            f"confidence intervals. {penalty_note}",
                            UserWarning,
                        )

        self.selection_summary_ = pd.DataFrame(selection_records)

        # 2. Refine Locations with Bagging (conditional on Phase-1 selected best_n > 0)
        if self.use_bagging and best_n > 0 and x_scaled is not None:
            robust_bps, all_samples_structured = find_bagged_breakpoints(
                t,
                x_scaled,
                best_n,
                n_bootstrap=self.n_bootstrap,
                random_state=bag_seq if bag_seq is not None else None,
                bootstrap_method=self.bootstrap_method,
                block_size=self.block_size,
                inner_n_boot=self.inner_n_boot,
                y_hat=best_y_pred_scaled,
                min_segment_size=self.min_segment_size,
            )
            self.bootstrap_samples_ = all_samples_structured

            n_total = len(all_samples_structured)
            n_failed = sum(1 for bps in all_samples_structured if len(bps) != best_n)
            self.bootstrap_failure_rate_ = (n_failed / n_total) if n_total > 0 else 1.0
            if n_failed > 0:
                warnings.warn(
                    f"{n_failed}/{n_total} ({100.0 * self.bootstrap_failure_rate_:.1f}%) "
                    f"bagging bootstrap replicates failed to converge and were excluded.",
                    UserWarning,
                )

            t_min, t_max = float(t.min()), float(t.max())
            bag_valid = (
                len(robust_bps) == best_n
                and np.all(np.isfinite(robust_bps))
                and np.all((robust_bps > t_min) & (robust_bps < t_max))
            )
            if bag_valid:
                order = np.argsort(robust_bps).astype(np.intp, copy=False)
                cand_bag_bps = np.asarray(robust_bps, dtype=float)[order]
                bag_seg_ok, bag_min_n = _check_min_segment_size(
                    t, cand_bag_bps, self.min_segment_size
                )
                if not bag_seg_ok:
                    warnings.warn(
                        f"Bagged breakpoints produced a segment with {bag_min_n} observation(s) "
                        f"(< min_segment_size={self.min_segment_size}); falling back to Phase-1 "
                        f"OLS breakpoints.",
                        UserWarning,
                    )
                    bag_valid = False
            if bag_valid:
                bag_cis = []
                for i in range(best_n):
                    col_samples = [
                        bps[i]
                        for bps in all_samples_structured
                        if len(bps) == best_n and np.isfinite(bps[i]) and t_min < bps[i] < t_max
                    ]
                    if len(col_samples) >= 2:
                        low = float(np.percentile(col_samples, 100 * alpha / 2))
                        high = float(np.percentile(col_samples, 100 * (1 - alpha / 2)))
                        bag_cis.append((low, high))
                    else:
                        bag_cis.append((np.nan, np.nan))

                best_bps = cand_bag_bps
                best_bp_cis = [bag_cis[idx] for idx in order]
            elif not (
                len(robust_bps) == best_n
                and np.all(np.isfinite(robust_bps))
                and np.all((robust_bps > t_min) & (robust_bps < t_max))
            ):
                warnings.warn(
                    "Bagging failed to produce valid breakpoints; falling back to Phase-1 OLS breakpoints.",
                    UserWarning,
                )

        self.n_breakpoints_ = best_n
        self.breakpoints_ = np.array(best_bps, dtype=float)
        self.breakpoint_cis_ = best_bp_cis
        self.bic_ = best_bic
        self.aic_ = best_aic
        self.mbic_ = best_mbic
        self.score_ = best_score

        # --- Phase 2: Robust Estimation (MannKS) ---
        # Import here to avoid potential circular dependencies.
        from types import SimpleNamespace
        from scipy.stats import norm
        from ._stats import (
            _sens_estimator_adaptive,
            _sens_estimator_censored_adaptive,
            _mk_score_and_var_censored,
            _confidence_intervals,
            _z_score,
            _p_value,
            _mk_probability,
        )
        from .classification import classify_trend
        from ._large_dataset import detect_size_tier

        tier_info = detect_size_tier(len(t), user_mode=large_dataset_mode)
        for tier_warn in tier_info.get('warnings', []):
            warnings.warn(tier_warn, UserWarning)
        if max_pairs is None:
            max_pairs = tier_info['max_pairs']

        self.segments_ = []
        boundaries = np.concatenate(([t.min()], self.breakpoints_, [t.max()]))
        n_segs = len(boundaries) - 1
        seg_seeds = (
            [int(c.generate_state(1)[0]) for c in phase2_seq.spawn(n_segs)]
            if phase2_seq is not None
            else [None] * n_segs
        )
        any_sampled = False

        for i in range(n_segs):
            t_start = boundaries[i]
            t_end = boundaries[i+1]

            if i == n_segs - 1:
                mask = (t >= t_start) & (t <= t_end)
            else:
                mask = (t >= t_start) & (t < t_end)

            t_seg = t[mask]
            x_seg = x[mask]

            if len(x_seg) < 2:
                self.segments_.append({
                    'slope': np.nan, 'intercept': np.nan,
                    'lower_ci': np.nan, 'upper_ci': np.nan,
                    'n': len(x_seg),
                    's': np.nan, 'var_s': np.nan, 'z': np.nan, 'p': np.nan,
                    'h': False, 'Tau': np.nan, 'C': np.nan, 'Cd': np.nan,
                    'trend': 'no trend', 'classification': 'Insufficient Data',
                    'analysis_notes': '',
                })
                continue

            cen_seg = (np.asarray(censored, dtype=bool)[mask] if censored is not None
                       else np.zeros(len(x_seg), dtype=bool))
            cen_type_seg = (np.asarray(cen_type)[mask] if cen_type is not None
                            else np.full(len(x_seg), 'not', dtype=object))
            s, var_s, _, Tau = _mk_score_and_var_censored(
                x_seg, t_seg, cen_seg, cen_type_seg,
                tau_method=self.tau_method,
                mk_test_method=self.mk_test_method,
                tie_break_method=self.tie_break_method,
            )
            n_seg = len(x_seg)
            total_pairs = n_seg * (n_seg - 1) // 2
            sampled = max_pairs is not None and total_pairs > max_pairs
            if sampled:
                any_sampled = True
            seg_seed = seg_seeds[i]
            seg_notes = []

            seg_ac = self.autocorr_method
            if seg_ac == 'auto' and n_seg >= 3:
                needs_corr, acf1_seg, _ = should_apply_correction(
                    x_seg, t_seg, max_pairs=max_pairs, random_state=seg_seed
                )
                if needs_corr:
                    seg_ac = 'block_bootstrap' if n_seg >= 5 else 'yue_wang'
                    seg_notes.append(f"Autocorrelation detected (ACF1={acf1_seg:.3f}), applying {seg_ac}")
                else:
                    seg_ac = 'none'
            elif seg_ac == 'auto':
                seg_ac = 'none'

            if seg_ac == 'block_bootstrap' and n_seg < 5:
                warnings.warn(
                    f"Segment {i} has only {n_seg} observations (< 5); falling back from "
                    f"'block_bootstrap' to {'yue_wang' if n_seg >= 3 else 'unadjusted'} CI.",
                    UserWarning,
                )
                seg_ac = 'yue_wang' if n_seg >= 3 else 'none'

            var_factor = 1.0
            if seg_ac == 'block_bootstrap' and n_seg >= 5:
                from ._bootstrap import block_bootstrap_mann_kendall, detrended_block_size
                boot_n = (
                    int(self.n_bootstrap)
                    if _is_valid_int(self.n_bootstrap) and int(self.n_bootstrap) >= 2
                    else 100
                )
                if self.block_size == 'auto':
                    seg_block_size = detrended_block_size(
                        x_seg, t_seg, cen_seg, cen_type_seg,
                        method=sens_slope_method if cen_seg.any() else 'drop_ambiguous',
                        lt_mult=lt_mult, gt_mult=gt_mult,
                        max_pairs=max_pairs, random_state=seg_seed,
                    )
                else:
                    seg_block_size = int(self.block_size)
                p_val, _, s_boot_dist = block_bootstrap_mann_kendall(
                    x_seg, t_seg, cen_seg, cen_type_seg,
                    block_size=seg_block_size, n_bootstrap=boot_n,
                    tau_method=self.tau_method, mk_test_method=self.mk_test_method,
                    tie_break_method=self.tie_break_method, random_state=seg_seed,
                    sens_slope_method=sens_slope_method, lt_mult=lt_mult, gt_mult=gt_mult,
                    max_pairs=max_pairs,
                )
                z_val = float(norm.ppf(1.0 - p_val / 2.0) * np.sign(s)) if np.isfinite(p_val) else np.nan
                h_val = bool(p_val < alpha) if np.isfinite(p_val) else False
                var_s = float(np.var(s_boot_dist, ddof=1)) if len(s_boot_dist) > 1 else var_s
                if self.continuous_confidence:
                    trend_str = 'decreasing' if z_val < 0 else ('increasing' if z_val > 0 else 'indeterminate')
                else:
                    trend_str = ('decreasing' if z_val < 0 else 'increasing') if h_val else 'no trend'
            elif seg_ac == 'yue_wang' and n_seg >= 3:
                n_eff_seg, _ = effective_sample_size(
                    x_seg, t_seg, max_pairs=max_pairs, random_state=seg_seed
                )
                var_factor = n_seg / n_eff_seg
                var_s = var_s * var_factor
                z_val = _z_score(s, var_s)
                p_val, h_val, trend_str = _p_value(
                    z_val, alpha, continuous_confidence=self.continuous_confidence
                )
                seg_notes.append(f"Yue-Wang correction applied (n_eff={n_eff_seg:.1f})")
            else:
                z_val = _z_score(s, var_s)
                p_val, h_val, trend_str = _p_value(
                    z_val, alpha, continuous_confidence=self.continuous_confidence
                )

            # Variance for slope CIs ('lwp' recomputes uncensored variance, matching trend_test)
            var_s_ci = var_s
            if self.ci_method == 'lwp' and seg_ac != 'block_bootstrap':
                _, var_s_unc, _, _ = _mk_score_and_var_censored(
                    x_seg, t_seg,
                    np.zeros(n_seg, dtype=bool),
                    np.full(n_seg, 'not', dtype=object),
                    tau_method=self.tau_method,
                    mk_test_method=self.mk_test_method,
                    tie_break_method=self.tie_break_method,
                )
                var_s_ci = var_s_unc * var_factor

            C_val, Cd_val = _mk_probability(p_val, s)
            classification = classify_trend(
                SimpleNamespace(C=C_val, Cd=Cd_val, h=h_val, s=s, trend=trend_str),
                category_map=self.category_map,
            )

            if cen_seg.any() and sens_slope_method == 'ats':
                from ._ats import ats_slope
                ats = ats_slope(t_seg, x_seg, cen_seg, cen_type=cen_type_seg, var_s=var_s_ci,
                                ci_alpha=alpha, max_pairs=max_pairs, random_state=seg_seed)
                slope, intercept = ats['beta'], ats['intercept']
                lower_ci, upper_ci = ats['ci_lower'], ats['ci_upper']
                for note in ats.get('notes', []):
                    if note:
                        seg_notes.append(str(note))
                        self.segment_notes_.append(f"Segment {i}: {note}")
            else:
                if cen_seg.any():
                    method = 'lwp' if sens_slope_method == 'lwp' else 'drop_ambiguous'
                    slopes = _sens_estimator_censored_adaptive(
                        x_seg, t_seg, cen_type_seg, lt_mult, gt_mult, method=method,
                        max_pairs=max_pairs, random_state=seg_seed)
                else:
                    slopes = _sens_estimator_adaptive(
                        x_seg, t_seg, max_pairs=max_pairs, random_state=seg_seed)
                slope = np.nanmedian(slopes) if np.any(~np.isnan(slopes)) else np.nan
                lower_ci, upper_ci = _confidence_intervals(
                    slopes, var_s_ci, alpha=alpha, method=self.ci_method,
                    total_pairs=total_pairs if sampled else None,
                )
                intercept = (
                    float(np.nanmedian(x_seg) - slope * np.nanmedian(t_seg))
                    if np.isfinite(slope)
                    else np.nan
                )

            if seg_ac == 'block_bootstrap' and n_seg >= 5:
                boot_method = sens_slope_method if cen_seg.any() else 'drop_ambiguous'
                _, lower_ci, upper_ci, _ = block_bootstrap_confidence_intervals(
                    x_seg,
                    t_seg,
                    cen_seg,
                    cen_type_seg,
                    block_size=seg_block_size,
                    n_bootstrap=boot_n,
                    alpha=alpha,
                    random_state=seg_seed,
                    method=boot_method,
                    lt_mult=lt_mult,
                    gt_mult=gt_mult,
                    max_pairs=max_pairs,
                )

            self.segments_.append({
                'slope': slope,
                'intercept': intercept,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'n': len(x_seg),
                's': s,
                'var_s': var_s,
                'z': z_val,
                'p': p_val,
                'h': bool(h_val),
                'Tau': Tau,
                'C': C_val,
                'Cd': Cd_val,
                'trend': trend_str,
                'classification': classification,
                'analysis_notes': '; '.join(seg_notes),
            })

        self.computation_mode_ = 'fast' if any_sampled else 'hybrid'

    def predict(self, t):
        t = np.asarray(t)
        y_pred = np.zeros_like(t, dtype=float)
        y_pred[:] = np.nan

        if self.segments_ is None or len(self.segments_) == 0:
            return y_pred

        bps = self.breakpoints_
        if len(bps) == 0:
             seg = self.segments_[0]
             return seg['slope'] * t + seg['intercept']

        mask = t < bps[0]
        seg = self.segments_[0]
        y_pred[mask] = seg['slope'] * t[mask] + seg['intercept']

        for i in range(len(bps) - 1):
            mask = (t >= bps[i]) & (t < bps[i+1])
            seg = self.segments_[i+1]
            y_pred[mask] = seg['slope'] * t[mask] + seg['intercept']

        mask = t >= bps[-1]
        seg = self.segments_[-1]
        y_pred[mask] = seg['slope'] * t[mask] + seg['intercept']

        return y_pred
