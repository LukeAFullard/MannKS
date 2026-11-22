
from contextlib import contextmanager
import numpy as np
import pandas as pd
import piecewise_regression
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
        return np.random.SeedSequence(int(random_state.integers(0, 2**32 - 1)))
    if isinstance(random_state, np.random.RandomState):
        return np.random.SeedSequence(int(random_state.randint(0, 2**32 - 1)))
    if _is_valid_int(random_state):
        return np.random.SeedSequence(int(random_state))
    return np.random.SeedSequence(random_state)


@contextmanager
def _isolated_legacy_numpy_seed(seed):
    """
    Temporarily seed legacy global np.random without polluting caller RNG state.
    Needed because piecewise_regression.Fit relies on legacy np.random calls.
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
    Compute original-scale BIC, mBIC (Zhang & Siegmund 2007), and AIC from
    standardized RSS with a symmetric scale-free relative floor (_REL_RSS_FLOOR).

    When n_eff is provided (ESS-adjusted scoring under autocorrelation), the
    log-likelihood term is weighted by n_eff while parameter penalties use
    n_samples.
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
                all_breakpoints.append(current_iter_bps)
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

    Phase 1 (Structure Discovery): Uses OLS (piecewise-regression) to find
    the number and location of breakpoints. Optionally uses bagging for
    robust breakpoint locations (conditional on the Phase-1 selected number
    of breakpoints; ``bic_`` and ``aic_`` remain Phase-1 OLS values).

    Phase 2 (Robust Estimation): Uses Mann-Kendall / Sen's Slope on the
    identified segments to estimate robust slopes and confidence intervals.
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
    ):
        """
        Args:
            max_breakpoints (int): Maximum number of breakpoints to search for (if n_breakpoints is None).
            n_breakpoints (int, optional): Fixed number of breakpoints to fit.
            use_bagging (bool): Use bootstrap aggregating for breakpoint location
                (conditional on the Phase-1 selected ``n_breakpoints``).
            n_bootstrap (int): Number of bootstrap samples if bagging is used (must be >= 2).
            criterion (str): Model selection criterion ('bic', 'mbic', or 'aic'). Default 'bic'.
            random_state (int, optional): Seed for random number generator.
            bootstrap_method (str): Resampling scheme for bagging ('block', 'pairs', or 'residual').
            block_size (str or int): Block size for block/residual bagging ('auto' or int >= 1).
            inner_n_boot (int): Internal Muggeo restarts per outer bootstrap replicate. Default 10.
            autocorr_penalty (str or bool): ESS adjustment for autocorrelation in Phase-1 selection
                ('auto', 'none', True, or False). Default 'auto'.
            use_davies_test (bool): Opt-in Davies test gate for 0 vs >=1 breakpoints during
                automatic model selection. Default False.
            autocorr_method (str): Optional Phase-2 per-segment autocorrelation adjustment for
                slope confidence intervals ('none', 'auto', 'yue_wang', or 'block_bootstrap').
                Default 'none'.
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

    def fit(self, t, x, censored=None, cen_type=None, lt_mult=0.5, gt_mult=1.1, alpha=0.05,
            large_dataset_mode='auto', max_pairs=None, sens_slope_method='ats'):
        """
        Fit breakpoints (OLS, piecewise-regression) and then estimate each
        segment's slope robustly (Sen's slope; Akritas-Theil-Sen for censored
        segments unless ``sens_slope_method`` is 'lwp'/'drop_ambiguous').

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
                    'mbic': np.nan, 'sar': np.nan, 'converged': False,
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
                    'mbic': -np.inf, 'sar': 0.0, 'converged': True,
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
                    'mbic': np.nan, 'sar': np.nan, 'converged': False,
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

            for idx_k, k in enumerate(n_range):
                k_int = int(k)
                record = {
                    'n_breakpoints': k_int, 'bic': np.nan, 'aic': np.nan,
                    'mbic': np.nan, 'sar': np.nan, 'converged': False,
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
                            'sar': rss_orig, 'converged': True,
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
                        })
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

                    with _isolated_legacy_numpy_seed(k_seeds[idx_k]):
                        pw_fit = piecewise_regression.Fit(
                            t_centered, x_scaled, n_breakpoints=k_int, verbose=False
                        )

                    res = pw_fit.get_results()
                    bm = getattr(pw_fit, 'best_muggeo', None)
                    bf = getattr(bm, 'best_fit', None) if bm is not None else None
                    converged = bool(res.get('converged', False)) and bf is not None
                    rss_scaled = getattr(bf, 'residual_sum_squares', None) if converged else None

                    if (
                        not converged
                        or rss_scaled is None
                        or not np.isfinite(rss_scaled)
                        or rss_scaled < 0
                    ):
                        warnings.warn(
                            f"Breakpoint fit for n_breakpoints={k_int} did not converge.",
                            UserWarning,
                        )
                        selection_records.append(record)
                        continue

                    estimates = getattr(bf, 'estimates', None) or res.get('estimates')
                    bps_data = []
                    if estimates:
                        for key, val in estimates.items():
                            if key.startswith('breakpoint'):
                                est_c = val.get('estimate', np.nan)
                                est = (
                                    float(est_c) * t_std + t_mean
                                    if est_c is not None and np.isfinite(est_c)
                                    else np.nan
                                )
                                se_c = val.get('se')
                                se = (
                                    float(se_c) * t_std
                                    if se_c is not None and np.isfinite(se_c)
                                    else None
                                )
                                if np.isfinite(est) and se is not None and np.isfinite(se):
                                    df = n_samples - (2 + 2 * k_int)
                                    if df > 0:
                                        t_crit = t_dist.ppf(1 - alpha / 2, df)
                                        ci = (float(est - t_crit * se), float(est + t_crit * se))
                                    else:
                                        ci = (np.nan, np.nan)
                                else:
                                    ci_raw = val.get('confidence_interval', (np.nan, np.nan))
                                    if (
                                        isinstance(ci_raw, (list, tuple, np.ndarray))
                                        and len(ci_raw) == 2
                                    ):
                                        ci = (
                                            float(ci_raw[0]) * t_std + t_mean
                                            if ci_raw[0] is not None and np.isfinite(ci_raw[0])
                                            else np.nan,
                                            float(ci_raw[1]) * t_std + t_mean
                                            if ci_raw[1] is not None and np.isfinite(ci_raw[1])
                                            else np.nan,
                                        )
                                    else:
                                        ci = (np.nan, np.nan)
                                bps_data.append((est, ci))

                    if len(bps_data) != k_int or not all(np.isfinite(b[0]) for b in bps_data):
                        warnings.warn(
                            f"Breakpoint fit for n_breakpoints={k_int} returned invalid "
                            f"breakpoint estimates.",
                            UserWarning,
                        )
                        selection_records.append(record)
                        continue

                    bps_data.sort(key=lambda item: item[0])
                    cand_bps = [b[0] for b in bps_data]
                    cand_bp_cis = [b[1] for b in bps_data]

                    try:
                        y_pred_scaled = np.asarray(bf.get_predicted_yy(), dtype=float)
                        if len(y_pred_scaled) != n_samples:
                            y_pred_scaled = None
                    except Exception:
                        y_pred_scaled = None
                    resid_scaled = (x_scaled - y_pred_scaled) if y_pred_scaled is not None else None

                    bic, mbic, aic = _compute_information_criteria(
                        rss_scaled, x_std, n_samples, k_int
                    )
                    rss_orig = float(rss_scaled) * (x_std ** 2)
                    record.update({
                        'bic': bic, 'aic': aic, 'mbic': mbic,
                        'sar': rss_orig, 'converged': True,
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
                    })
                except Exception as exc:
                    record['bic'] = np.nan
                    record['aic'] = np.nan
                    record['mbic'] = np.nan
                    record['sar'] = np.nan
                    record['converged'] = False
                    warnings.warn(
                        f"Breakpoint fit for n_breakpoints={k_int} failed: {exc}",
                        UserWarning,
                    )

                selection_records.append(record)

            # --- Score converged candidates (with optional ESS-BIC & Davies gate) ---
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
                if (
                    self.n_breakpoints is None
                    and self.autocorr_penalty in ('auto', True)
                    and len(candidates) > 1
                    and n_samples >= 10
                ):
                    richest = max(candidates, key=lambda c: c['k'])
                    if (
                        richest['resid_scaled'] is not None
                        and richest['rss_scaled'] > n_samples * _REL_RSS_FLOOR
                    ):
                        acf_rich, _ = estimate_acf(richest['resid_scaled'], max_lag=1)
                        rho_1_rich = (
                            float(acf_rich[1])
                            if len(acf_rich) > 1 and np.isfinite(acf_rich[1])
                            else 0.0
                        )
                        acf_thresh = max(0.2, 1.96 / np.sqrt(n_samples))
                        if rho_1_rich > acf_thresh:
                            rho_clamped = min(rho_1_rich, 0.95)
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
                    if np.isfinite(score) and score < best_cand_score:
                        best_cand_score = score
                        best_cand = cand

                # Opt-in Davies test gate (only during auto-select when k > 0 selected)
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
                            warnings.warn(
                                f"Davies test for change in slope was not significant "
                                f"(p={p_davies:.4g} >= alpha={alpha}); gating selection "
                                f"to 0 breakpoints.",
                                UserWarning,
                            )

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
                                f"ESS-adjusted criterion (n_eff={n_eff:.1f}) was applied during model selection."
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
            if (
                len(robust_bps) == best_n
                and np.all(np.isfinite(robust_bps))
                and np.all((robust_bps > t_min) & (robust_bps < t_max))
            ):
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

                order = np.argsort(robust_bps).astype(np.intp, copy=False)
                best_bps = np.asarray(robust_bps, dtype=float)[order]
                best_bp_cis = [bag_cis[idx] for idx in order]
            else:
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
        from ._stats import (
            _sens_estimator_adaptive,
            _sens_estimator_censored_adaptive,
            _mk_score_and_var_censored,
            _confidence_intervals
        )

        from ._large_dataset import detect_size_tier
        if max_pairs is None:
            max_pairs = detect_size_tier(len(t), user_mode=large_dataset_mode)['max_pairs']

        self.segments_ = []
        boundaries = np.concatenate(([t.min()], self.breakpoints_, [t.max()]))
        n_segs = len(boundaries) - 1
        seg_seeds = (
            [int(c.generate_state(1)[0]) for c in phase2_seq.spawn(n_segs)]
            if phase2_seq is not None
            else [None] * n_segs
        )

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
                    'n': len(x_seg)
                })
                continue

            cen_seg = (np.asarray(censored, dtype=bool)[mask] if censored is not None
                       else np.zeros(len(x_seg), dtype=bool))
            cen_type_seg = (np.asarray(cen_type)[mask] if cen_type is not None
                            else np.full(len(x_seg), 'not', dtype=object))
            s, var_s, _, _ = _mk_score_and_var_censored(x_seg, t_seg, cen_seg, cen_type_seg)
            n_seg = len(x_seg)
            total_pairs = n_seg * (n_seg - 1) // 2
            sampled = max_pairs is not None and total_pairs > max_pairs
            seg_seed = seg_seeds[i]

            seg_ac = self.autocorr_method
            if seg_ac == 'auto' and n_seg >= 3:
                needs_corr, _, _ = should_apply_correction(
                    x_seg, t_seg, max_pairs=max_pairs, random_state=seg_seed
                )
                if needs_corr:
                    seg_ac = 'block_bootstrap' if n_seg >= 5 else 'yue_wang'
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

            if seg_ac == 'yue_wang' and n_seg >= 3:
                n_eff_seg, _ = effective_sample_size(
                    x_seg, t_seg, max_pairs=max_pairs, random_state=seg_seed
                )
                var_s = var_s * (n_seg / n_eff_seg)

            if cen_seg.any() and sens_slope_method == 'ats':
                from ._ats import ats_slope
                ats = ats_slope(t_seg, x_seg, cen_seg, cen_type=cen_type_seg, var_s=var_s,
                                ci_alpha=alpha, max_pairs=max_pairs, random_state=seg_seed)
                slope, intercept = ats['beta'], ats['intercept']
                lower_ci, upper_ci = ats['ci_lower'], ats['ci_upper']
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
                    slopes, var_s, alpha=alpha, total_pairs=total_pairs if sampled else None)
                intercept = np.median(x_seg - slope * t_seg) if np.isfinite(slope) else np.nan

            if seg_ac == 'block_bootstrap' and n_seg >= 5:
                boot_n = (
                    int(self.n_bootstrap)
                    if _is_valid_int(self.n_bootstrap) and int(self.n_bootstrap) >= 2
                    else 100
                )
                boot_method = sens_slope_method if cen_seg.any() else 'drop_ambiguous'
                _, lower_ci, upper_ci, _ = block_bootstrap_confidence_intervals(
                    x_seg,
                    t_seg,
                    cen_seg,
                    cen_type_seg,
                    block_size=self.block_size,
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
                'n': len(x_seg)
            })

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
