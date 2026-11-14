"""
Generalized Quantile-Kendall trend analysis for unequally spaced and censored time series.
"""
from collections import namedtuple, Counter
from typing import Union, Optional, Sequence, Dict
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

from .trend_test import trend_test
from .classification import classify_trend
from ._helpers import _prepare_data, _get_slope_scaling_factor
from ._datetime import _is_datetime_like, _to_numeric_time
from ._stats import (
    _z_score,
    _p_value,
    _mk_probability,
    _confidence_intervals,
    _mk_score_and_var_censored,
    _normalize_sens_slope_method,
)
from ._quantile import (
    _assign_blocks,
    _standardize_block,
    _sampling_diagnostic,
    _equalize_blocks,
    _mk_s_matrix,
    _normalize_start_freq,
    _egret_decimal_year,
    _auto_u_grid,
)

_Quantile_Trend_Tuple = namedtuple('Quantile_Trend_Test', [
    'results',
    'blocks',
    'matrix',
    'censored_matrix',
    'cen_type_matrix',
    'sampling_diagnostic',
    'field_significance',
    'n_ranks',
    'target_length',
    'block_frequency',
    'alpha',
    'transform',
    'standardization',
    'is_datetime',
    'computation_mode',
    'analysis_notes',
    'warnings',
])

class QuantileTrendResult(_Quantile_Trend_Tuple):
    """
    Structured result returned by :func:`quantile_trend_test`.

    Provides access to per-quantile trend results (``results``), block metadata
    (``blocks``), the standardized block-by-quantile data matrix (``matrix`` and
    optional censoring matrices), sampling drift diagnostics (``sampling_diagnostic``),
    optional permutation field significance (``field_significance``), and helper
    methods for inspecting and extracting quantile series.
    """

    @property
    def quantiles(self) -> pd.DataFrame:
        """DataFrame of per-quantile trend results (alias for ``self.results``)."""
        return self.results

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the per-quantile trend results DataFrame."""
        return self.results.copy()

    def significant(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """
        Filter quantile trend results to rows with two-sided ``p < alpha``.

        Parameters
        ----------
        alpha : float, optional
            Significance threshold in ``(0, 1)``. Defaults to ``self.alpha``.
        """
        if self.results.empty:
            return self.results.copy()
        thresh = self.alpha if alpha is None else float(alpha)
        return self.results[self.results['p'] < thresh].copy()

    def get_quantile_series(self, quantile_or_rank: Union[float, int]) -> pd.DataFrame:
        """
        Extract the block-level time series for a specific quantile rank or probability.

        Parameters
        ----------
        quantile_or_rank : int or float
            Either a 1-based integer rank (``>= 1``) matching ``results['rank']``,
            or a target probability in ``[0.0, 1.0]`` matched to the nearest
            entry in ``results['quantile']``.

        Returns
        -------
        pd.DataFrame
            Time series with columns ``'t'``, ``'value'``, ``'censored'``,
            ``'cen_type'`` (plus ``'rank'``, ``'quantile'``, and
            ``'plotting_position'`` when results are non-empty).
        """
        if self.matrix.empty or self.results.empty:
            return pd.DataFrame(columns=['t', 'value', 'censored', 'cen_type'])

        if isinstance(quantile_or_rank, (bool, np.bool_)):
            raise ValueError("`quantile_or_rank` cannot be a boolean; pass a rank int >= 1 or quantile float.")

        if isinstance(quantile_or_rank, (int, np.integer)) and quantile_or_rank >= 1:
            rank_target = int(quantile_or_rank)
            if rank_target not in self.results['rank'].values:
                raise ValueError(f"Rank {rank_target} not found in results (1..{self.n_ranks}).")
            idx = int(np.flatnonzero(self.results['rank'].to_numpy() == rank_target)[0])
        else:
            q_target = float(quantile_or_rank)
            if not np.isfinite(q_target) or q_target < 0.0 or q_target > 1.0:
                raise ValueError("Probability `quantile_or_rank` must be in [0.0, 1.0], or pass an integer rank >= 1.")
            idx = int(np.argmin(np.abs(self.results['quantile'].to_numpy() - q_target)))

        col_name = self.matrix.columns[idx]
        vals = self.matrix[col_name].to_numpy(dtype=float)
        if self.censored_matrix is not None:
            cens = self.censored_matrix[col_name].to_numpy(dtype=bool)
            ctyp = self.cen_type_matrix[col_name].to_numpy(dtype=object)
        else:
            cens = np.zeros(len(vals), dtype=bool)
            ctyp = np.full(len(vals), 'not', dtype=object)

        return pd.DataFrame({
            't': self.matrix.index,
            'value': vals,
            'censored': cens,
            'cen_type': ctyp,
            'rank': int(self.results['rank'].iloc[idx]),
            'quantile': float(self.results['quantile'].iloc[idx]),
            'plotting_position': float(self.results['plotting_position'].iloc[idx]),
        })

QuantileResult = QuantileTrendResult

_ALLOWED_MK_KWARGS = {
    'sens_slope_method', 'tau_method', 'mk_test_method', 'ci_method',
    'tie_break_method', 'lt_mult', 'gt_mult', 'autocorr_method',
    'block_size', 'n_bootstrap', 'large_dataset_mode', 'max_pairs',
}
_REJECTED_MK_KWARGS = {'agg_method', 'agg_period', 'residual_plot_path', 'min_size'}
_RESULTS_COLUMNS = [
    'rank', 'quantile', 'plotting_position', 'z_position', 'slope', 'lower_ci', 'upper_ci',
    'slope_pct', 'lower_ci_pct', 'upper_ci_pct', 'slope_per_second', 'lower_ci_per_second',
    'upper_ci_per_second', 'intercept', 'slope_units', 'p', 'h', 'z', 'C', 'Cd', 'Tau',
    's', 'var_s', 'trend', 'classification', 'n_blocks', 'prop_censored', 'acf1',
    'n_effective', 'analysis_notes',
]

def _resolve_quantile_grid(quantiles, block_sizes, target_length, standardization, max_quantiles, quantile_method='hazen'):
    n_med = int(np.round(np.median(block_sizes)))
    n_min = int(np.min(block_sizes))
    n_max = int(np.max(block_sizes))

    if isinstance(quantiles, (bool, np.bool_)):
        raise ValueError("`quantiles` cannot be a boolean; use 'auto', a positive integer, or a 1D probability sequence.")

    if isinstance(quantiles, str):
        if quantiles != 'auto':
            raise ValueError("String value for `quantiles` must be 'auto'.")
        if standardization == 'egret':
            if isinstance(target_length, (bool, np.bool_)) or (
                target_length != 'min'
                and not (isinstance(target_length, (int, np.integer)) and int(target_length) == n_min)
            ):
                raise ValueError(
                    "`standardization='egret'` requires `target_length='min'` (or an integer matching minimum block size)."
                )
            M = n_min
        elif isinstance(target_length, (bool, np.bool_)):
            raise ValueError("`target_length` must be 'min', 'median', 'max', or an integer >= 2.")
        elif target_length == 'min':
            M = n_min
        elif target_length == 'median':
            M = n_med
        elif target_length == 'max':
            M = n_max
        elif isinstance(target_length, (int, np.integer)):
            M = int(target_length)
        else:
            raise ValueError("`target_length` must be 'min', 'median', 'max', or an integer >= 2.")

        if M < 2:
            raise ValueError(f"Target quantile length must be at least 2, got {M}.")
        if M > max_quantiles:
            raise ValueError(f"Target length M={M} exceeds `max_quantiles` ({max_quantiles}).")
        ranks = np.arange(1, M + 1, dtype=np.intp)
        # 'egret' keeps order statistics; label them with their Hazen probabilities.
        u_grid = _auto_u_grid(M, 'hazen' if standardization == 'egret' else quantile_method)
        p_grid = ranks.astype(float) / (M + 1.0)
        z_grid = norm.ppf(p_grid)
        return M, ranks, u_grid, p_grid, z_grid, M
    elif isinstance(quantiles, (int, np.integer)):
        M = int(quantiles)
        if M < 1 or M > max_quantiles:
            raise ValueError(f"Integer `quantiles` must be between 1 and {max_quantiles}.")
        if standardization == 'egret':
            raise ValueError("`standardization='egret'` requires `quantiles='auto'`.")
        ranks = np.arange(1, M + 1, dtype=np.intp)
        u_grid = ranks.astype(float) / (M + 1.0)
        p_grid = u_grid.copy()
        z_grid = norm.ppf(p_grid)
        return M, ranks, u_grid, p_grid, z_grid, M
    else:
        if standardization == 'egret':
            raise ValueError("`standardization='egret'` requires `quantiles='auto'`.")
        q_arr = np.asarray(quantiles, dtype=float)
        if q_arr.ndim != 1 or len(q_arr) == 0 or np.any(np.isnan(q_arr)) or np.any(q_arr < 0.0) or np.any(q_arr > 1.0):
            raise ValueError("`quantiles` sequence must be a non-empty 1D sequence in [0.0, 1.0].")
        u_grid = np.sort(q_arr)
        if len(np.unique(u_grid)) != len(u_grid):
            raise ValueError("`quantiles` sequence must not contain duplicate probabilities.")
        M = len(u_grid)
        if M > max_quantiles:
            raise ValueError(f"Length of `quantiles` exceeds `max_quantiles` ({max_quantiles}).")
        ranks = np.arange(1, M + 1, dtype=np.intp)
        p_min_bound = 1.0 / (n_med + 1.0)
        p_max_bound = n_med / (n_med + 1.0)
        p_grid = np.clip(u_grid, p_min_bound, p_max_bound)
        z_grid = norm.ppf(p_grid)
        return M, ranks, u_grid, p_grid, z_grid, None

def _fast_uncensored_matrix_trends(M_matrix, t_numeric, is_datetime, alpha, slope_scaling, x_unit, continuous_confidence, category_map, ci_method='direct', tau_method='b'):
    B, K = M_matrix.shape
    s_scores, var_s_vec, tau_vec = _mk_s_matrix(M_matrix, t_numeric, tau_method=tau_method)
    i_idx, j_idx = np.triu_indices(B, k=1)
    dt = t_numeric[j_idx] - t_numeric[i_idx]
    valid_t = dt != 0.0
    ii, jj = i_idx[valid_t], j_idx[valid_t]
    dt_valid = dt[valid_t]

    if len(dt_valid) == 0:
        pairwise_slopes = np.empty((0, K), dtype=float)
        slopes_raw = np.full(K, np.nan, dtype=float)
        intercepts = np.full(K, np.nan, dtype=float)
    else:
        pairwise_slopes = (M_matrix[jj, :] - M_matrix[ii, :]) / dt_valid[:, np.newaxis]
        slopes_raw = np.median(pairwise_slopes, axis=0)
        t_med = float(np.median(t_numeric))
        y_meds = np.median(M_matrix, axis=0)
        intercepts = y_meds - slopes_raw * t_med

    slope_factor = 1.0
    if slope_scaling and is_datetime:
        slope_factor = _get_slope_scaling_factor(slope_scaling)
        slope_units = f"{x_unit} per {slope_scaling.lower()}"
    elif is_datetime:
        slope_units = f"{x_unit} per second"
    else:
        slope_units = f"{x_unit} per unit of t"

    rows = []
    MockRes = namedtuple('MockRes', ['C', 'trend'])

    for k in range(K):
        s_k = float(s_scores[k])
        var_s_k = float(var_s_vec[k])
        tau_k = float(tau_vec[k])
        z_k = _z_score(s_k, var_s_k)
        p_k, h_k, trend_k = _p_value(z_k, alpha, continuous_confidence=continuous_confidence)
        C_k, Cd_k = _mk_probability(p_k, s_k)
        lo_raw, hi_raw = _confidence_intervals(pairwise_slopes[:, k], var_s_k, alpha, method=ci_method)

        class_str = classify_trend(MockRes(C=C_k, trend=trend_k), category_map=category_map) if continuous_confidence else (trend_k.title() if trend_k != 'no trend' else 'No Trend')

        rows.append({
            'slope_per_second': float(slopes_raw[k]),
            'lower_ci_per_second': float(lo_raw),
            'upper_ci_per_second': float(hi_raw),
            'slope': float(slopes_raw[k] * slope_factor),
            'lower_ci': float(lo_raw * slope_factor),
            'upper_ci': float(hi_raw * slope_factor),
            'intercept': float(intercepts[k]),
            'slope_units': slope_units,
            'p': float(p_k),
            'h': bool(h_k),
            'z': float(z_k),
            'C': float(C_k),
            'Cd': float(Cd_k),
            'Tau': float(tau_k),
            's': float(s_k),
            'var_s': float(var_s_k),
            'trend': trend_k,
            'classification': class_str,
            'n_blocks': B,
            'prop_censored': 0.0,
            'acf1': 0.0,
            'n_effective': float(B),
            'analysis_notes': [],
        })
    return rows

def _compute_field_significance(M_matrix, t_numeric, var_s_vec, z_obs_vec, alpha, n_permutations, random_state):
    rng = np.random.default_rng(random_state)
    B, K = M_matrix.shape
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    valid_var = (var_s_vec > 1e-10) & np.isfinite(var_s_vec)
    sqrt_var = np.where(valid_var, np.sqrt(np.maximum(var_s_vec, 1e-10)), np.inf)

    abs_z_obs = np.abs(z_obs_vec)
    n_sig_obs = int(np.nansum(abs_z_obs > z_crit))
    max_z_obs = float(np.nanmax(abs_z_obs)) if np.any(np.isfinite(abs_z_obs)) else 0.0

    has_nan = bool(np.isnan(M_matrix).any())
    i_idx, j_idx = np.triu_indices(B, k=1)
    dt = t_numeric[j_idx] - t_numeric[i_idx]
    valid_t = dt != 0.0
    ii, jj = i_idx[valid_t], j_idx[valid_t]
    dt_sign = np.sign(dt[valid_t])[:, np.newaxis]

    count_exceed = 0
    max_z_exceed = 0

    for _ in range(n_permutations):
        perm_idx = rng.permutation(B)
        M_perm = M_matrix[perm_idx, :]
        if has_nan:
            dM = M_perm[jj, :] - M_perm[ii, :]
            s_perm = np.nansum(dt_sign * np.sign(dM), axis=0)
        else:
            s_perm, _, _ = _mk_s_matrix(M_perm, t_numeric)
        z_perm = np.where(s_perm > 0, (s_perm - 1.0) / sqrt_var, np.where(s_perm < 0, (s_perm + 1.0) / sqrt_var, 0.0))
        abs_z_perm = np.abs(z_perm)
        if int(np.nansum(abs_z_perm > z_crit)) >= n_sig_obs:
            count_exceed += 1
        max_perm = float(np.nanmax(abs_z_perm)) if np.any(np.isfinite(abs_z_perm)) else 0.0
        if max_perm >= max_z_obs:
            max_z_exceed += 1

    return {
        'n_significant': n_sig_obs,
        'p_count': float((1 + count_exceed) / (n_permutations + 1)),
        'max_abs_z': max_z_obs,
        'p_max_z': float((1 + max_z_exceed) / (n_permutations + 1)),
        'n_permutations': int(n_permutations),
    }

def quantile_trend_test(
    x, t,
    block_frequency='YE',
    block_origin=None,
    months=None,
    quantiles='auto',
    target_length='min',
    standardization='interpolate',
    quantile_method='hazen',
    block_time='center',
    min_size_per_block=10,
    min_blocks=5,
    min_coverage=0.9,
    require_complete=False,
    equalize='none',
    censor_quantile_method='propagate',
    transform='none',
    log_offset=None,
    alpha=0.05,
    hicensor=False,
    category_map=None,
    continuous_confidence=True,
    slope_scaling='year',
    x_unit='units',
    engine='trend_test',
    field_significance=False,
    n_permutations=999,
    max_quantiles=2000,
    random_state=None,
    plot_path=None,
    **mk_kwargs,
) -> QuantileTrendResult:
    """
    Generalized Quantile-Kendall non-parametric trend test across distribution quantiles.

    Evaluates monotonic trends at multiple quantiles across temporal blocks
    (e.g., annual, seasonal, or multi-year epochs) to identify non-uniform distribution
    shifts such as extremes warming or drying faster than medians.

    The method standardizes sample sizes across blocks, extracts order statistics
    (or interpolated quantiles), and conducts Mann-Kendall trend tests and
    Sen's / Akritas-Theil-Sen slope estimation for each quantile level.

    Parameters
    ----------
    x : array-like or DataFrame
        Observed values. For censored data, provide a DataFrame with columns
        ``'value'``, ``'censored'``, and ``'cen_type'`` (or as returned by
        ``prepare_censored_data``).
    t : array-like
        Observation times (datetime-like or numeric). Timezone-aware timestamps
        are analysed in their own local wall-clock time: block boundaries, the
        ``months`` filter and all reported block times are local (e.g. a calendar
        year starts at local midnight on 1 January), and are returned tz-naive.
    block_frequency : str or float, default 'YE'
        Duration of time blocks. For datetime-like ``t``, a pandas offset string
        such as ``'YE'`` (annual calendar year), ``'5YE'`` (5-year epochs),
        ``'QE'`` (quarterly), or ``'ME'`` (monthly). For numeric ``t``, a positive float
        specifying block width.
    block_origin : datetime-like, numeric, or None, default None
        Fixed anchor point for block boundaries. If None, calendar boundaries
        or data minimum are used. Calendar-anchored frequencies (``'YE'``,
        ``'YS-OCT'``, ``'QE'``, ``'ME'`` ...) can only start a block on their own
        anchor, so an origin between anchors only selects the phase of multi-period
        blocks (e.g. which years start a ``'5YE'`` block) and a ``UserWarning``
        reports the edges actually used; no observation is ever dropped. A
        timezone-aware origin is converted to the timezone of ``t``.
    months : sequence of int or None, default None
        Months to include within each annual block (1-12). If specified, filters
        observations to selected seasons before block analysis.
    quantiles : {'auto'} or int or array-like, default 'auto'
        Quantile grid configuration:
        - ``'auto'``: ``M`` ranks, with ``M`` set by ``target_length``. The rank
          probabilities are those of ``quantile_method`` (``(j - 0.5) / M`` for
          ``'hazen'``, ``(j - 1) / (M - 1)`` for ``'linear'``), so a block of
          exactly ``M`` observations is returned as its own order statistics.
        - int: Integer number of evenly spaced quantiles.
        - array-like: Explicit sequence of distinct non-exceedance probabilities in ``[0.0, 1.0]``.
    target_length : {'min', 'median', 'max'} or int, default 'min'
        Standard sample size across blocks when ``quantiles='auto'``:
        - ``'min'``: Uses the minimum valid block size (no block is up-sampled).
        - ``'median'``: Uses the median valid block size.
        - ``'max'``: Uses the maximum valid block size.
        - int: Explicit integer sample size.
        When ``standardization='egret'``, ``target_length`` must be ``'min'``
        (or an integer matching the minimum valid block size).
    standardization : {'interpolate', 'egret'}, default 'interpolate'
        Method for standardizing unequal block lengths:
        - ``'interpolate'``: Quantile interpolation across block order statistics via ``numpy.quantile``.
        - ``'egret'``: Exact USGS EGRET leap-year rank-reduction method (365 ranks).
    quantile_method : str, default 'hazen'
        Percentile calculation method passed to ``numpy.quantile`` (e.g. ``'hazen'``, ``'linear'``, ``'weibull'``).
    block_time : {'center', 'egret'}, default 'center'
        Time coordinate given to each block in the trend tests:
        - ``'center'``: Calendar midpoint of the block ``[block_start, block_end)``.
        - ``'egret'``: USGS EGRET ``midDecYear`` — the mean of the EGRET decimal
          year (``year + (day_of_year - 0.5) / days_in_year``) of the first and
          last observation in the block. Slopes are then per (decimal) year,
          exactly as in EGRET. Requires datetime ``t`` and ``slope_scaling='year'``.
          Together with ``standardization='egret'``, ``transform='log'``,
          and ``require_complete=True`` this reproduces EGRET's
          ``makeSortQ`` / ``trendSortQ`` output (see ``validation/43_Quantile_Kendall``).
    min_size_per_block : int, default 10
        Minimum number of observations required to retain a block.
    min_blocks : int, default 5
        Minimum number of valid blocks required to perform trend analysis.
    min_coverage : float, default 0.9
        Minimum fraction of the nominal block duration spanned by the block's
        observations, from first to last (0.0 to 1.0). Blocks below this threshold
        are rejected. This is a span, not a density: it does not detect gaps
        between the first and last observation (use ``min_size_per_block``,
        ``require_complete`` or ``equalize`` for that), and it is uninformative for
        non-contiguous ``months`` selections.
    require_complete : bool, default False
        If True, blocks must contain at least ``round(expected_duration / global_median_step)``
        observations (e.g., all 365/366 daily observations per year).
    equalize : {'none', 'thin', 'aggregate'} or None, default 'none'
        Subsampling or aggregation strategy to equalize sample counts across blocks
        and guard against sampling-frequency drift artifacts:
        - ``'none'`` or ``None``: No equalization (raw observations).
        - ``'thin'``: Deterministic systematic subsampling (evenly spaced indices via
          ``linspace``) down to the minimum valid block sample size.
        - ``'aggregate'``: Computes censored-aware ``'robust_median'`` within monthly
          sub-intervals (multi-month blocks), daily/hourly/minute sub-intervals (shorter
          datetime blocks), or 12 sub-intervals (numeric time). Every sub-interval is
          first thinned to a common number of observations ``k`` (the smallest count
          in any populated sub-interval of any valid block, reported as
          ``sampling_diagnostic['aggregate_support']``), because medians of different
          numbers of observations are not comparable between blocks. When sampling
          frequency changed over the record ``k`` is typically 1 and the mode reduces
          to stratified thinning (one observation per sub-interval).
    censor_quantile_method : {'propagate', 'substitute', 'error'}, default 'propagate'
        Handling of censored values during quantile standardization:
        - ``'propagate'``: Tracks censoring flags through order statistics and applies
          Akritas-Theil-Sen (ATS) slope estimation by default. With ``transform='log'``,
          only ``sens_slope_method='ats'`` is supported.
        - ``'substitute'``: Substitutes censored values in raw units using ``lt_mult`` / ``gt_mult``.
          Not recommended when detection limits change over the record: the
          substituted values then step with the limit and create spurious trends
          in the low quantiles (use ``'propagate'``, optionally with ``hicensor``).
        - ``'error'``: Raises an exception if censoring is present.
    transform : {'none', 'log', 'pct_of_median'} or None, default 'none'
        Data transformation applied prior to trend testing:
        - ``'none'`` or ``None``: Raw values; reports slope and percentage slope relative to median.
        - ``'log'``: Natural logarithm; ``slope`` is in natural log units per time unit and
          ``slope_pct`` is exponentiated as ``100 * (exp(slope) - 1)`` % per time unit
          (``slope_units`` reads ``'ln(<x_unit>) per <time unit>'``).
        - ``'pct_of_median'``: Normalized as percentage of each quantile's cross-block median.
    log_offset : float or None, default None
        Constant added to values before log-transformation if non-positive values exist.
    alpha : float, default 0.05
        Significance level for hypothesis tests and confidence intervals.
    hicensor : bool or float, default False
        High-censoring rule: values below the highest left-censoring threshold are censored at that threshold.
    category_map : dict or None, default None
        Custom mapping dictionary for qualitative trend classifications.
    continuous_confidence : bool, default True
        Whether to assign continuous confidence classifications (e.g. 'Likely Increasing').
    slope_scaling : str or None, default 'year'
        Time unit for reporting slope rates: ``'year'``, ``'month'``, ``'week'``, ``'day'``,
        ``'hour'``, ``'minute'``, ``'second'``, or None (unscaled, per second for datetime ``t``).
    x_unit : str or None, default 'units'
        Measurement unit of the variable for labeling (e.g. ``'mg/L'``, ``'m3/s'``).
        If None, defaults to ``'units'``.
    engine : {'trend_test', 'fast'}, default 'trend_test'
        Computational engine for trend estimation across quantiles.
    field_significance : bool, default False
        Whether to compute simultaneous field significance across all quantiles via permutation.
    n_permutations : int, default 999
        Number of block permutations for field significance testing.
    max_quantiles : int, default 2000
        Maximum allowable quantiles when generating automatic grids.
    random_state : int, Generator, or None, default None
        Seed or random number generator for permutation tests (``field_significance=True``)
        and block-bootstrap trend tests.
    plot_path : str, Path, or None, default None
        Optional file path to automatically render and save a Quantile-Kendall diagnostic plot.
    **mk_kwargs : dict
        Additional keyword arguments forwarded to the Mann-Kendall trend testing engine.

    Returns
    -------
    QuantileTrendResult
        A named tuple containing:
        - ``results``: DataFrame of per-quantile trend statistics (rank, quantile, tau, slope, p, etc.).
        - ``blocks``: DataFrame detailing block start, end, center, pre-equalization count (``n_obs``),
          censored count, coverage, and validity.
        - ``matrix``: DataFrame of standardized quantile values (blocks x quantiles).
        - ``censored_matrix``: Boolean DataFrame of censoring status.
        - ``cen_type_matrix``: String DataFrame of censoring types ('not', 'lt', 'gt').
        - ``sampling_diagnostic``: Dict with tau, p-value, sample size ratio, and drift flag
          (evaluated on the block sizes used for quantile standardization).
        - ``field_significance``: Dict of permutation test results if field_significance=True, else None.
        - ``n_ranks``: Number of evaluated quantiles.
        - ``target_length``: Resolved standardization sample size.
        - ``block_frequency``: Block frequency used.
        - ``alpha``: Significance level used.
        - ``transform``: Transformation used.
        - ``standardization``: Standardization method used.
        - ``is_datetime``: Boolean indicating if time index was datetime-like.
        - ``computation_mode``: Mode string ('exact', 'fast', or 'insufficient').
        - ``analysis_notes``: List of analysis and diagnostic notes.
        - ``warnings``: List of warnings raised during execution.
    """
    mk_kwargs = dict(mk_kwargs)
    rejected = _REJECTED_MK_KWARGS.intersection(mk_kwargs.keys())
    if rejected:
        raise ValueError(f"Arguments {sorted(rejected)} are not supported in `quantile_trend_test`.")
    unknown = set(mk_kwargs.keys()) - _ALLOWED_MK_KWARGS
    if unknown:
        raise ValueError(f"Unknown keyword argument(s): {sorted(unknown)}")

    if 'sens_slope_method' in mk_kwargs:
        mk_kwargs['sens_slope_method'] = _normalize_sens_slope_method(mk_kwargs['sens_slope_method'], stacklevel=2)
    if mk_kwargs.get('tau_method', 'b') not in ('a', 'b'):
        raise ValueError("Invalid `tau_method`. Must be one of ['a', 'b'].")
    if mk_kwargs.get('mk_test_method', 'robust') not in ('robust', 'lwp'):
        raise ValueError("Invalid `mk_test_method`. Must be one of ['robust', 'lwp'].")
    if mk_kwargs.get('ci_method', 'direct') not in ('direct', 'lwp'):
        raise ValueError("Invalid `ci_method`. Must be one of ['direct', 'lwp'].")
    if mk_kwargs.get('tie_break_method', 'robust') not in ('robust', 'lwp'):
        raise ValueError("Invalid `tie_break_method`. Must be one of ['robust', 'lwp'].")
    if mk_kwargs.get('autocorr_method', 'none') not in ('none', 'auto', 'block_bootstrap', 'yue_wang'):
        raise ValueError("Invalid `autocorr_method`. Must be one of ['none', 'auto', 'block_bootstrap', 'yue_wang'].")

    if equalize is None:
        equalize = 'none'
    if transform is None:
        transform = 'none'

    if not 0 < alpha < 1:
        raise ValueError(f"Significance level `alpha` must be between 0 and 1. Got {alpha}.")
    if standardization not in ('interpolate', 'egret'):
        raise ValueError("`standardization` must be 'interpolate' or 'egret'.")
    if equalize not in ('none', 'thin', 'aggregate'):
        raise ValueError("`equalize` must be 'none', 'thin', or 'aggregate'.")
    if censor_quantile_method not in ('propagate', 'substitute', 'error'):
        raise ValueError("`censor_quantile_method` must be 'propagate', 'substitute', or 'error'.")
    if transform not in ('none', 'log', 'pct_of_median'):
        raise ValueError("`transform` must be 'none', 'log', or 'pct_of_median'.")
    if block_time not in ('center', 'egret'):
        raise ValueError("`block_time` must be 'center' or 'egret'.")
    if engine not in ('trend_test', 'fast'):
        raise ValueError("`engine` must be 'trend_test' or 'fast'.")
    if isinstance(min_size_per_block, (bool, np.bool_)) or not isinstance(min_size_per_block, (int, np.integer)) or int(min_size_per_block) < 1:
        raise ValueError("`min_size_per_block` must be an integer >= 1.")
    if isinstance(min_blocks, (bool, np.bool_)) or not isinstance(min_blocks, (int, np.integer)) or int(min_blocks) < 1:
        raise ValueError("`min_blocks` must be an integer >= 1.")
    try:
        _cov_ok = (not isinstance(min_coverage, (bool, np.bool_))) and 0.0 <= float(min_coverage) <= 1.0
    except (TypeError, ValueError):
        _cov_ok = False
    if not _cov_ok:
        raise ValueError("`min_coverage` must be a number between 0.0 and 1.0.")
    try:
        np.quantile(np.array([0.0, 1.0]), 0.5, method=quantile_method)
    except (ValueError, TypeError, KeyError):
        raise ValueError(
            f"Invalid `quantile_method` {quantile_method!r}. Must be a `numpy.quantile` method "
            "(e.g. 'hazen', 'linear', 'weibull', 'median_unbiased')."
        )
    if isinstance(max_quantiles, (bool, np.bool_)) or not isinstance(max_quantiles, (int, np.integer)) or int(max_quantiles) < 1:
        raise ValueError("`max_quantiles` must be an integer >= 1.")
    if isinstance(n_permutations, (bool, np.bool_)) or not isinstance(n_permutations, (int, np.integer)) or int(n_permutations) < 1:
        raise ValueError("`n_permutations` must be an integer >= 1.")
    if log_offset is not None:
        if isinstance(log_offset, (bool, np.bool_)):
            raise ValueError("`log_offset` must be a finite number or None.")
        try:
            log_offset_val = float(log_offset)
        except (TypeError, ValueError):
            raise ValueError("`log_offset` must be a finite number or None.")
        if not np.isfinite(log_offset_val):
            raise ValueError("`log_offset` must be finite.")

    validated_months = None
    if months is not None:
        if isinstance(months, (str, bytes)) or not hasattr(months, '__iter__'):
            raise ValueError("`months` must be a non-empty sequence of integers between 1 and 12.")
        try:
            months_list = list(months)
        except TypeError:
            raise ValueError("`months` must be a non-empty sequence of integers between 1 and 12.")
        if len(months_list) == 0:
            raise ValueError("`months` must be a non-empty sequence of integers between 1 and 12.")
        validated_months = []
        for m in months_list:
            if isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)):
                raise ValueError(f"All entries in `months` must be integers between 1 and 12. Got {m!r}.")
            m_int = int(m)
            if not 1 <= m_int <= 12:
                raise ValueError(f"All entries in `months` must be integers between 1 and 12. Got {m_int}.")
            validated_months.append(m_int)
        months = validated_months

    eff_x_unit = 'units' if x_unit is None else str(x_unit)

    x_arr = np.asarray(x) if not isinstance(x, pd.DataFrame) else x
    t_arr = np.asarray(t)
    if getattr(x_arr, 'ndim', 1) == 0 or t_arr.ndim == 0:
        raise ValueError("`x` and `t` must be array-like sequences, not scalars.")
    if len(x_arr) != len(t_arr):
        raise ValueError(f"Input vectors `x` and `t` must have the same length.")

    module_notes = []
    module_warnings = []
    warning_categories = {}   # message -> original warning category
    captured_logs = []        # the recorder of `_run`, so warnings survive an exception

    def _drain(log):
        for w in log:
            msg = str(w.message)
            module_warnings.append(msg)
            warning_categories.setdefault(msg, w.category)

    # Timezone-aware timestamps are analysed in local wall-clock time, so that
    # "calendar year", "water year" and `months` mean what they mean locally.
    t_work = t
    data_tz = None
    if _is_datetime_like(t_arr):
        try:
            t_idx = pd.DatetimeIndex(pd.Index(t))
        except (TypeError, ValueError):
            t_idx = None
        if t_idx is not None and t_idx.tz is not None:
            data_tz = t_idx.tz
            t_work = t_idx.tz_localize(None)
            module_notes.append(
                f"timezone-aware timestamps analysed in local wall-clock time ({data_tz}); "
                "block boundaries and block times are tz-naive local times"
            )
    origin_work = block_origin
    if block_origin is not None and _is_datetime_like(t_arr):
        try:
            origin_ts = pd.Timestamp(block_origin)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid `block_origin` {block_origin!r} for datetime data.")
        if origin_ts.tzinfo is not None:
            origin_ts = (origin_ts.tz_convert(data_tz) if data_tz is not None else origin_ts.tz_convert('UTC')).tz_localize(None)
        origin_work = origin_ts

    def _empty_result(is_dt, blocks_df=None):
        return QuantileTrendResult(
            results=pd.DataFrame(columns=_RESULTS_COLUMNS),
            blocks=blocks_df if blocks_df is not None else pd.DataFrame(),
            matrix=pd.DataFrame(),
            censored_matrix=None,
            cen_type_matrix=None,
            sampling_diagnostic={'tau_n_vs_time': np.nan, 'p_n_vs_time': np.nan, 'n_min': 0, 'n_max': 0, 'ratio_max_min': np.nan, 'flag': False},
            field_significance=None,
            n_ranks=0,
            target_length=None,
            block_frequency=block_frequency,
            alpha=alpha,
            transform=transform,
            standardization=standardization,
            is_datetime=is_dt,
            computation_mode='insufficient',
            analysis_notes=list(dict.fromkeys(module_notes)),
            warnings=list(dict.fromkeys(module_warnings)),
        )

    def _run():
        nonlocal module_warnings, module_notes
        with warnings.catch_warnings(record=True) as outer_w_log:
            warnings.simplefilter("always")
            captured_logs.append(outer_w_log)
    
            data_filtered, is_datetime = _prepare_data(x, t_work, hicensor)
            data_filtered = data_filtered.sort_values('t', kind='stable').reset_index(drop=True)
    
            if block_time == 'egret':
                if not is_datetime:
                    raise ValueError("`block_time='egret'` requires datetime-like `t`.")
                if slope_scaling != 'year':
                    raise ValueError("`block_time='egret'` reports slopes per decimal year and requires `slope_scaling='year'`.")

            if len(data_filtered) < 2:
                warnings.warn("Insufficient data for Quantile-Kendall analysis.", UserWarning)
                _drain(outer_w_log)
                return _empty_result(is_datetime)
    
            if months is not None:
                if not is_datetime:
                    warnings.warn("Parameter `months` is ignored when `t` is not datetime-like.", UserWarning)
                else:
                    norm_freq = _normalize_start_freq(block_frequency)
                    if 'YS' in norm_freq:
                        month_map = {
                            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                        }
                        start_month = 1
                        if '-' in norm_freq:
                            m_str = norm_freq.split('-')[-1][:3].upper()
                            start_month = month_map.get(m_str, 1)
                        end_month = 12 if start_month == 1 else start_month - 1
                        unique_months = set(months)
                        if len(unique_months) < 12 and start_month in unique_months and end_month in unique_months:
                            warnings.warn(f"Cross-year `months={list(months)}` used with calendar-year block frequency.", UserWarning)
    
            valid_groups, blocks_df, block_centers = _assign_blocks(
                data=data_filtered, block_frequency=block_frequency, block_origin=origin_work,
                months=months, min_size_per_block=min_size_per_block, min_coverage=min_coverage,
                require_complete=require_complete, is_datetime=is_datetime,
            )
    
            n_valid_blocks = len(valid_groups)
            if n_valid_blocks < max(2, min_blocks):
                msg = f"Insufficient valid blocks ({n_valid_blocks}) for Quantile-Kendall analysis."
                warnings.warn(msg, UserWarning)
                module_notes.append(msg)
                _drain(outer_w_log)
                return _empty_result(is_datetime, blocks_df=blocks_df)
    
            if n_valid_blocks < 10:
                module_notes.append(f"block count ({n_valid_blocks}) below recommended minimum (10)")
    
            egret_times = None
            if block_time == 'egret':
                first_obs = [g['t_original'].iloc[0] for g in valid_groups]
                last_obs = [g['t_original'].iloc[-1] for g in valid_groups]
                egret_times = 0.5 * (_egret_decimal_year(first_obs) + _egret_decimal_year(last_obs))

            block_sizes = np.array([len(g) for g in valid_groups], dtype=np.intp)
            samp_diag = _sampling_diagnostic(block_centers, block_sizes)
            if samp_diag['flag'] and equalize == 'none':
                drift_msg = f"Potential sampling-frequency drift across blocks (N_min={samp_diag['n_min']}, N_max={samp_diag['n_max']})."
                warnings.warn(drift_msg, UserWarning)
                module_notes.append(drift_msg)
    
            if equalize != 'none':
                eq_info = {}
                valid_groups = _equalize_blocks(valid_groups, equalize, is_datetime=is_datetime, info=eq_info)
                block_sizes = np.array([len(g) for g in valid_groups], dtype=np.intp)
                samp_diag = _sampling_diagnostic(block_centers, block_sizes)
                if equalize == 'aggregate':
                    k_sup = int(eq_info.get('aggregate_support', 1))
                    samp_diag['aggregate_support'] = k_sup
                    module_notes.append(
                        f"equalize='aggregate': each sub-interval value is the robust median of "
                        f"{k_sup} observation(s) (common support across all blocks)"
                        + ("; equivalent to stratified thinning" if k_sup == 1 else "")
                    )
                if samp_diag['flag']:
                    module_notes.append(
                        f"Residual sampling-frequency imbalance remains after equalize='{equalize}' "
                        f"(N_min={samp_diag['n_min']}, N_max={samp_diag['n_max']})."
                    )
    
            M, ranks, u_grid, p_grid, z_grid, target_len_resolved = _resolve_quantile_grid(
                quantiles=quantiles, block_sizes=block_sizes, target_length=target_length,
                standardization=standardization, max_quantiles=max_quantiles,
                quantile_method=quantile_method,
            )
    
            lt_mult = float(mk_kwargs.get('lt_mult', 0.5))
            gt_mult = float(mk_kwargs.get('gt_mult', 1.1))
    
            val_rows = np.empty((n_valid_blocks, M), dtype=float)
            cens_rows = np.zeros((n_valid_blocks, M), dtype=bool)
            type_rows = np.full((n_valid_blocks, M), 'not', dtype=object)
    
            std_info = {}
            for b_idx, grp in enumerate(valid_groups):
                sv, sc, st = _standardize_block(
                    group_df=grp, M=M, standardization=standardization, quantile_method=quantile_method,
                    censor_quantile_method=censor_quantile_method, lt_mult=lt_mult, gt_mult=gt_mult, u_grid=u_grid,
                    info=std_info,
                )
                val_rows[b_idx, :] = sv
                cens_rows[b_idx, :] = sc
                type_rows[b_idx, :] = st
    
            has_any_censored = bool(cens_rows.any())
            n_mixed = int(std_info.get('n_mixed_brackets', 0))
            if n_mixed:
                mixed_msg = (
                    f"{n_mixed} standardized quantile value(s) are interpolated between a left-censored ('<') "
                    "and a right-censored ('>') observation. Their true value is unbounded on both sides; they "
                    "are flagged 'lt' by convention, which is not a valid bound. Interpret those quantiles "
                    "with caution (see `cen_type_matrix`)."
                )
                warnings.warn(mixed_msg, UserWarning)
                module_notes.append(mixed_msg)
            col_labels = [f"r{int(r)}" for r in ranks]
            index_centers = pd.Index(block_centers, name='block_center')
    
            matrix_df = pd.DataFrame(val_rows, index=index_centers, columns=col_labels)
            censored_df = pd.DataFrame(cens_rows, index=index_centers, columns=col_labels) if has_any_censored else None
            cen_type_df = pd.DataFrame(type_rows, index=index_centers, columns=col_labels) if has_any_censored else None
    
            work_matrix = val_rows.copy()
            col_medians_raw = np.nanmedian(val_rows, axis=0)
    
            if transform == 'log':
                if has_any_censored and censor_quantile_method == 'propagate' and mk_kwargs.get('sens_slope_method', 'ats') != 'ats':
                    raise ValueError(
                        "`transform='log'` with `censor_quantile_method='propagate'` requires `sens_slope_method='ats'` "
                        "(or `censor_quantile_method='substitute'`), because heuristic multipliers (`lt_mult`/`gt_mult`) "
                        "are invalid on negative log-transformed values."
                    )
                if log_offset is not None:
                    work_matrix = work_matrix + float(log_offset)
                if np.any(work_matrix <= 0):
                    raise ValueError("`transform='log'` requires all standardized block values to be > 0.")
                work_matrix = np.log(work_matrix)
            elif transform == 'pct_of_median':
                has_non_pos = False
                for k in range(M):
                    med_k = col_medians_raw[k]
                    if pd.notna(med_k) and med_k > 0.0 and np.isfinite(med_k):
                        work_matrix[:, k] = (work_matrix[:, k] / med_k) * 100.0
                    else:
                        work_matrix[:, k] = np.nan
                        has_non_pos = True
                if has_non_pos:
                    module_notes.append("transform='pct_of_median' encountered non-positive quantile median; set to NaN.")
    
            eff_slope_scaling = slope_scaling if (is_datetime and egret_times is None) else None
            if eff_slope_scaling:
                try:
                    _get_slope_scaling_factor(eff_slope_scaling)
                except (ValueError, TypeError) as e:
                    warnings.warn(f"Slope scaling failed: {e}", UserWarning)
                    eff_slope_scaling = None

            _fast_incompatible = (
                mk_kwargs.get('autocorr_method', 'none') != 'none'
                or mk_kwargs.get('mk_test_method', 'robust') == 'lwp'
                or mk_kwargs.get('tie_break_method', 'robust') == 'lwp'
                or mk_kwargs.get('large_dataset_mode', 'auto') in ('fast', 'aggregate')
                or mk_kwargs.get('max_pairs') is not None
                or (mk_kwargs.get('large_dataset_mode', 'auto') == 'auto' and n_valid_blocks > 5000)
            )
            use_fast_engine = (
                engine == 'fast'
                and not has_any_censored
                and not _fast_incompatible
                and not np.isnan(work_matrix).any()
            )
            if engine == 'fast' and not use_fast_engine:
                reasons = []
                if has_any_censored:
                    reasons.append("censored quantiles")
                if np.isnan(work_matrix).any():
                    reasons.append("NaN quantile columns")
                if _fast_incompatible:
                    reasons.append("options not supported by the fast engine")
                module_notes.append(
                    "engine='fast' not applicable (" + ", ".join(reasons) + "); used engine='trend_test'"
                )
    
            if egret_times is not None:
                t_numeric_blocks = egret_times
                t_for_trend = egret_times
                trend_is_datetime = False
            else:
                t_numeric_blocks = _to_numeric_time(block_centers)
                t_for_trend = block_centers
                trend_is_datetime = is_datetime
            col_warning_counts = Counter()
            results_list = []
    
            if use_fast_engine:
                comp_mode = 'fast'
                fast_rows = _fast_uncensored_matrix_trends(
                    M_matrix=work_matrix, t_numeric=t_numeric_blocks, is_datetime=trend_is_datetime,
                    alpha=alpha, slope_scaling=eff_slope_scaling, x_unit=eff_x_unit,
                    continuous_confidence=continuous_confidence, category_map=category_map,
                    ci_method=mk_kwargs.get('ci_method', 'direct'),
                    tau_method=mk_kwargs.get('tau_method', 'b'),
                )
                for k, row_dict in enumerate(fast_rows):
                    row_dict['rank'] = int(ranks[k])
                    row_dict['quantile'] = float(u_grid[k])
                    row_dict['plotting_position'] = float(p_grid[k])
                    row_dict['z_position'] = float(z_grid[k])
                    results_list.append(row_dict)
            else:
                comp_mode = 'exact'
                for k in range(M):
                    y_col = work_matrix[:, k]
                    c_col = cens_rows[:, k]
                    ct_col = type_rows[:, k]
    
                    x_input = pd.DataFrame({'value': y_col, 'censored': c_col, 'cen_type': ct_col}) if (has_any_censored and c_col.any()) else y_col
    
                    with warnings.catch_warnings(record=True) as col_w:
                        warnings.simplefilter("always")
                        res_k = trend_test(
                            x=x_input, t=t_for_trend, alpha=alpha, hicensor=False,
                            category_map=category_map, continuous_confidence=continuous_confidence,
                            slope_scaling=eff_slope_scaling, x_unit=eff_x_unit, min_size=None,
                            random_state=random_state, **mk_kwargs,
                        )
                        for w_msg in res_k.warnings:
                            col_warning_counts[w_msg] += 1
    
                    results_list.append({
                        'rank': int(ranks[k]), 'quantile': float(u_grid[k]),
                        'plotting_position': float(p_grid[k]), 'z_position': float(z_grid[k]),
                        'slope': float(res_k.slope) if pd.notna(res_k.slope) else np.nan,
                        'lower_ci': float(res_k.lower_ci) if pd.notna(res_k.lower_ci) else np.nan,
                        'upper_ci': float(res_k.upper_ci) if pd.notna(res_k.upper_ci) else np.nan,
                        'slope_per_second': float(res_k.slope_per_second) if pd.notna(res_k.slope_per_second) else np.nan,
                        'lower_ci_per_second': float(res_k.lower_ci_per_second) if pd.notna(res_k.lower_ci_per_second) else np.nan,
                        'upper_ci_per_second': float(res_k.upper_ci_per_second) if pd.notna(res_k.upper_ci_per_second) else np.nan,
                        'intercept': float(res_k.intercept) if pd.notna(res_k.intercept) else np.nan,
                        'slope_units': res_k.slope_units, 'p': float(res_k.p) if pd.notna(res_k.p) else np.nan,
                        'h': bool(res_k.h), 'z': float(res_k.z) if pd.notna(res_k.z) else np.nan,
                        'C': float(res_k.C) if pd.notna(res_k.C) else np.nan,
                        'Cd': float(res_k.Cd) if pd.notna(res_k.Cd) else np.nan,
                        'Tau': float(res_k.Tau) if pd.notna(res_k.Tau) else np.nan,
                        's': float(res_k.s) if pd.notna(res_k.s) else np.nan,
                        'var_s': float(res_k.var_s) if pd.notna(res_k.var_s) else np.nan,
                        'trend': res_k.trend, 'classification': res_k.classification,
                        'n_blocks': int(np.sum(~np.isnan(y_col))), 'prop_censored': float(res_k.prop_censored),
                        'acf1': float(res_k.acf1) if pd.notna(res_k.acf1) else np.nan,
                        'n_effective': float(res_k.n_effective) if pd.notna(res_k.n_effective) else np.nan,
                        'analysis_notes': res_k.analysis_notes,
                    })
    
            if egret_times is not None:
                scale_unit_label = "year"
            else:
                scale_unit_label = (
                    eff_slope_scaling.lower()
                    if (eff_slope_scaling and is_datetime)
                    else ("second" if is_datetime else "unit of t")
                )
            _sec_per_year = 365.25 * 86400.0
            for k, row_dict in enumerate(results_list):
                if egret_times is not None:
                    # The trend engines saw numeric decimal years: slopes are per year.
                    row_dict['slope_units'] = f"{eff_x_unit} per year"
                    for src, dst in (('slope', 'slope_per_second'), ('lower_ci', 'lower_ci_per_second'),
                                     ('upper_ci', 'upper_ci_per_second')):
                        v = row_dict[src]
                        row_dict[dst] = float(v / _sec_per_year) if pd.notna(v) else np.nan
                slp = row_dict['slope']
                lo = row_dict['lower_ci']
                hi = row_dict['upper_ci']
                if transform == 'log':
                    row_dict['slope_pct'] = float(100.0 * (np.exp(slp) - 1.0)) if pd.notna(slp) else np.nan
                    row_dict['lower_ci_pct'] = float(100.0 * (np.exp(lo) - 1.0)) if pd.notna(lo) else np.nan
                    row_dict['upper_ci_pct'] = float(100.0 * (np.exp(hi) - 1.0)) if pd.notna(hi) else np.nan
                    # `slope` and its CI stay in natural-log units; `slope_pct` is % per time unit.
                    row_dict['slope_units'] = f"ln({eff_x_unit}) per {scale_unit_label}"
                elif transform == 'pct_of_median':
                    med_k = col_medians_raw[k]
                    if pd.notna(med_k) and med_k > 0.0 and np.isfinite(med_k):
                        row_dict['slope_pct'] = slp
                        row_dict['lower_ci_pct'] = lo
                        row_dict['upper_ci_pct'] = hi
                    else:
                        row_dict['slope_pct'] = np.nan
                        row_dict['lower_ci_pct'] = np.nan
                        row_dict['upper_ci_pct'] = np.nan
                    row_dict['slope_units'] = f"% of median per {scale_unit_label}"
                else:
                    med_k = col_medians_raw[k]
                    if pd.notna(med_k) and med_k > 0.0 and np.isfinite(med_k):
                        row_dict['slope_pct'] = float(100.0 * slp / med_k) if pd.notna(slp) else np.nan
                        row_dict['lower_ci_pct'] = float(100.0 * lo / med_k) if pd.notna(lo) else np.nan
                        row_dict['upper_ci_pct'] = float(100.0 * hi / med_k) if pd.notna(hi) else np.nan
                    else:
                        row_dict['slope_pct'] = np.nan
                        row_dict['lower_ci_pct'] = np.nan
                        row_dict['upper_ci_pct'] = np.nan
    
            results_df = pd.DataFrame(results_list, columns=_RESULTS_COLUMNS)
    
            field_sig_dict = None
            if field_significance:
                if has_any_censored:
                    warnings.warn(
                        "`field_significance=True` permutes numeric standardized quantile values "
                        "without censored-interval scoring; interpret field p-values cautiously "
                        "when censored quantiles are present.",
                        UserWarning,
                    )
                if mk_kwargs.get('autocorr_method', 'none') != 'none':
                    warnings.warn(
                        "`field_significance=True` assumes exchangeable blocks under permutation; "
                        "autocorrelation adjustments are not preserved across permutations.",
                        UserWarning,
                    )
                field_sig_dict = _compute_field_significance(
                    M_matrix=work_matrix, t_numeric=t_numeric_blocks,
                    var_s_vec=results_df['var_s'].to_numpy(dtype=float),
                    z_obs_vec=results_df['z'].to_numpy(dtype=float),
                    alpha=alpha, n_permutations=n_permutations, random_state=random_state,
                )
    
            _drain(outer_w_log)
            for msg, count in col_warning_counts.items():
                agg_msg = f"In {count} of {M} quantiles: {msg}" if M > 1 else msg
                module_warnings.append(agg_msg)
    
        module_warnings = list(dict.fromkeys(module_warnings))
        final_result = QuantileTrendResult(
            results=results_df, blocks=blocks_df, matrix=matrix_df,
            censored_matrix=censored_df, cen_type_matrix=cen_type_df,
            sampling_diagnostic=samp_diag, field_significance=field_sig_dict,
            n_ranks=M, target_length=target_len_resolved, block_frequency=block_frequency,
            alpha=alpha, transform=transform, standardization=standardization,
            is_datetime=is_datetime, computation_mode=comp_mode,
            analysis_notes=list(dict.fromkeys(module_notes)),
            warnings=list(dict.fromkeys(module_warnings)),
        )
    
    
        return final_result

    try:
        res = _run()
    except Exception:
        # Do not swallow warnings raised before the failure.
        if captured_logs:
            seen = set(module_warnings)
            for w in captured_logs[0]:
                if str(w.message) not in seen:
                    seen.add(str(w.message))
                    warnings.warn(str(w.message), w.category, stacklevel=2)
        raise
    for w_str in res.warnings:
        warnings.warn(w_str, warning_categories.get(w_str, UserWarning), stacklevel=2)

    if plot_path is not None:
        from .plotting import plot_quantile_trend
        plot_quantile_trend(res, save_path=plot_path)

    return res

_REGIME_SUMMARY_COLUMNS = [
    'regime', 'p_min', 'p_max', 'n_quantiles', 'median_slope',
    'median_slope_pct', 'prop_significant', 'modal_classification',
]


def summarize_quantile_regimes(
    result: Union[QuantileTrendResult, pd.DataFrame],
    regimes: Optional[Dict[str, tuple]] = None,
) -> pd.DataFrame:
    """
    Summarize quantile trend test results across probability bands (regimes).

    Parameters
    ----------
    result : QuantileTrendResult or pd.DataFrame
        Output object returned by :func:`quantile_trend_test`, or its results DataFrame.
    regimes : dict of str -> (float, float), optional
        Mapping of regime names to ``(p_min, p_max)`` bounds on ``plotting_position``.
        Defaults to ``{'low': (0.0, 0.25), 'mid': (0.25, 0.75), 'high': (0.75, 1.0)}``.
        Intervals are half-open ``[p_min, p_max)`` except when ``p_max >= 1.0``,
        where ``p_max`` is included.

    Returns
    -------
    pd.DataFrame
        Summary table with columns ``regime``, ``p_min``, ``p_max``, ``n_quantiles``,
        ``median_slope``, ``median_slope_pct``, ``prop_significant``, and
        ``modal_classification``.
    """
    if regimes is None:
        regimes = {'low': (0.0, 0.25), 'mid': (0.25, 0.75), 'high': (0.75, 1.0)}
    df = result if isinstance(result, pd.DataFrame) else result.results
    if df.empty:
        return pd.DataFrame(columns=_REGIME_SUMMARY_COLUMNS)

    summary_rows = []
    q_col = df['plotting_position']
    for name, (lo, hi) in regimes.items():
        mask = (q_col >= lo) & (q_col <= hi) if hi >= 1.0 else (q_col >= lo) & (q_col < hi)
        sub = df[mask]
        if sub.empty:
            continue
        modes = sub['classification'].mode()
        modal_class = modes.iloc[0] if not modes.empty else 'Insufficient Data'
        summary_rows.append({
            'regime': name, 'p_min': float(lo), 'p_max': float(hi),
            'n_quantiles': int(len(sub)), 'median_slope': float(sub['slope'].median()),
            'median_slope_pct': float(sub['slope_pct'].median()), 'prop_significant': float(sub['h'].mean()),
            'modal_classification': modal_class,
        })
    return pd.DataFrame(summary_rows, columns=_REGIME_SUMMARY_COLUMNS)
