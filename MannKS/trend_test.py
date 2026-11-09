"""
Mann-Kendall trend test with Sen's / Akritas-Theil-Sen slope for unequally
spaced (and optionally censored) time series.
"""
from collections import namedtuple
import numpy as np
import pandas as pd
import warnings
from scipy.stats import norm
from ._stats import (_z_score, _p_value, _confidence_intervals, _mk_probability,
                     _mk_score_and_var_censored, _sen_probability,
                     _sens_estimator_adaptive, _sens_estimator_censored_adaptive,
                     _total_pairs_if_sampled, _normalize_sens_slope_method)
from ._ats import ats_slope
from ._helpers import (_prepare_data, _aggregate_by_group, _value_for_time_increment)
from ._large_dataset import detect_size_tier
from .plotting import plot_trend, plot_residuals
from .analysis_notes import get_analysis_note, get_sens_slope_analysis_note
from .classification import classify_trend


from typing import Union, Optional

_RESULT_FIELDS = [
    'trend', 'h', 'p', 'z', 'Tau', 's', 'var_s', 'slope', 'intercept',
    'lower_ci', 'upper_ci', 'C', 'Cd', 'classification', 'analysis_notes',
    'sen_probability', 'sen_probability_max', 'sen_probability_min',
    'prop_censored', 'prop_unique', 'n_censor_levels',
    'slope_per_second', 'lower_ci_per_second', 'upper_ci_per_second',
    'scaled_slope', 'slope_units',
    'acf1', 'n_effective', 'block_size_used', 'warnings',
    'computation_mode', 'pairs_used', 'approximation_error'
]

_PERIOD_MAP = {
    'year': 'Y', 'month': 'M', 'quarter': 'Q',
    'week': 'W', 'day': 'D',
    'hour': 'h', 'minute': 'min', 'second': 's'
}


def trend_test(
    x: Union[np.ndarray, pd.DataFrame],
    t: np.ndarray,
    alpha: float = 0.05,
    hicensor: Union[bool, float] = False,
    plot_path: Optional[str] = None,
    residual_plot_path: Optional[str] = None,
    lt_mult: float = 0.5,
    gt_mult: float = 1.1,
    sens_slope_method: str = 'ats',
    tau_method: str = 'b',
    agg_method: str = 'none',
    agg_period: Optional[str] = None,
    min_size: Optional[int] = 10,
    mk_test_method: str = 'robust',
    ci_method: str = 'direct',
    tie_break_method: str = 'robust',
    category_map: Optional[dict] = None,
    continuous_confidence: bool = True,
    x_unit: str = "units",
    slope_scaling: Optional[str] = None,
    seasonal_coloring: bool = False,
    autocorr_method: str = 'none',
    block_size: Union[str, int] = 'auto',
    n_bootstrap: int = 1000,
    large_dataset_mode: str = 'auto',
    max_pairs: Optional[int] = None,
    random_state: Optional[int] = None
) -> namedtuple:
    """
    Mann-Kendall trend test with a robust slope estimate.

    The data are sorted by time internally, so the input order does not
    matter.  Pairs of observations with equal times score 0 in S.

    Parameters
    ----------
    x : array-like or DataFrame
        Values, or the DataFrame returned by ``prepare_censored_data``
        (columns ``value``, ``censored``, ``cen_type``).
    t : array-like
        Times (datetime-like or numeric).
    alpha : float, default 0.05
        Significance level for ``h`` and for the two-sided (1 - alpha)
        confidence interval of the slope.  LWP-TRENDS reports 90% intervals,
        i.e. ``alpha=0.1``.
    hicensor : bool or float
        High-censor rule (LWP-TRENDS ``ApplyHiCensor``): values below the
        highest left-censoring limit (or below ``min(that limit, hicensor)``
        when numeric) are set censored at that limit.
    sens_slope_method : {'ats', 'lwp', 'drop_ambiguous'}, default 'ats'
        Slope estimator for censored data (uncensored data always use the
        exact Sen's slope).

        - ``'ats'``: Akritas-Theil-Sen estimator (as NADA::cenken), with a
          confidence interval obtained by inverting the censored Kendall test.
          Recommended.
        - ``'lwp'``: LWP-TRENDS pairwise slopes with ambiguous censored pairs
          set to 0 (biased towards zero as censoring increases).
        - ``'drop_ambiguous'``: pairwise slopes with ambiguous pairs dropped
          (biased *away* from zero as censoring increases).  The former names
          ``'unbiased'`` and ``'nan'`` are deprecated aliases.
    tau_method : {'b', 'a'}
        Kendall's tau variant reported (LWP-TRENDS reports tau-a).
    agg_method, agg_period
        Temporal aggregation.  ``agg_method='lwp'`` keeps, per ``agg_period``
        increment, the observation closest to the increment mid-point
        (LWP-TRENDS ``UseMidObs=TRUE``); ``'lwp_median'`` takes the median
        (``UseMidObs=FALSE``).  Other options: 'median', 'robust_median',
        'middle', 'middle_lwp'.  With ``agg_method != 'none'`` and no
        ``agg_period``, observations with identical timestamps are aggregated.
    mk_test_method : {'robust', 'lwp'}
        ``'lwp'`` treats all right-censored values as tied above the largest
        right-censoring limit, as LWP-TRENDS does.
    ci_method : {'direct', 'lwp'}
        Rank-based Sen's slope interval: nearest rank ('direct') or
        interpolation with the variance computed ignoring censoring
        ('lwp', as LWP-TRENDS).
    tie_break_method : {'robust', 'lwp'}
        Size of the internal shift used to separate tied values in Var(S);
        both give identical results except in pathological floating-point
        cases ('lwp' uses exactly the LWP-TRENDS constant).
    autocorr_method : {'none', 'auto', 'block_bootstrap', 'yue_wang'}
        Serial correlation handling.  'yue_wang' inflates Var(S) using the
        significant autocorrelations of the Sen-detrended series; 'block_bootstrap'
        uses a detrended circular block bootstrap for the p-value and a block
        bootstrap percentile interval for the slope; 'auto' applies the block
        bootstrap when the detrended lag-1 autocorrelation is significant.
    large_dataset_mode : {'auto', 'full', 'fast', 'aggregate'}
        'full' always computes every pair (exact).  'fast' (automatic for
        5000 < n <= 50000) samples ``max_pairs`` pairs for the slope; the
        Mann-Kendall test itself is always exact.  'aggregate' (automatic
        above 50000) computes as 'fast' and warns that the data should be
        aggregated (with ``agg_method``/``agg_period``); it does not
        aggregate by itself.
    max_pairs : int, optional
        Number of pairs sampled for the slope in fast mode (default 100,000).
    random_state : int, optional
        Seed for pair sampling and bootstraps.  Results are reproducible by
        default (a fixed seed is used for pair sampling when None).

    Returns
    -------
    namedtuple ``Mann_Kendall_Test`` with, among others:
        trend, h, p, z, Tau, s, var_s, slope, intercept, lower_ci, upper_ci,
        C (confidence in the trend direction, 1 - p/2), Cd (confidence the
        trend is decreasing), classification, analysis_notes,
        sen_probability (probability the true slope is below zero, LWP-TRENDS
        ``Sen_Probability``), prop_censored / prop_unique / n_censor_levels
        (of the data before aggregation), slope_per_second, scaled_slope,
        slope_units.  The intercept is defined at t = 0 (the Unix epoch for
        datetime ``t``) on the unscaled time axis.
    """

    # --- Basic Input Validation ---
    x_arr = np.asarray(x) if not isinstance(x, pd.DataFrame) else x
    t_arr = np.asarray(t)
    if len(x_arr) != len(t_arr):
        raise ValueError(f"Input vectors `x` and `t` must have the same length. Got {len(x_arr)} and {len(t_arr)}.")
    if not 0 < alpha < 1:
        raise ValueError(f"Significance level `alpha` must be between 0 and 1. Got {alpha}.")

    res = namedtuple('Mann_Kendall_Test', _RESULT_FIELDS)

    sens_slope_method = _normalize_sens_slope_method(sens_slope_method)
    valid_tau_methods = ['a', 'b']
    if tau_method not in valid_tau_methods:
        raise ValueError(f"Invalid `tau_method`. Must be one of {valid_tau_methods}.")
    valid_agg_methods = ['none', 'median', 'robust_median', 'middle', 'middle_lwp', 'lwp', 'lwp_median', 'lwp_robust_median']
    if agg_method not in valid_agg_methods:
        raise ValueError(f"Invalid `agg_method`. Must be one of {valid_agg_methods}.")
    valid_mk_test_methods = ['robust', 'lwp']
    if mk_test_method not in valid_mk_test_methods:
        raise ValueError(f"Invalid `mk_test_method`. Must be one of {valid_mk_test_methods}.")
    valid_ci_methods = ['direct', 'lwp']
    if ci_method not in valid_ci_methods:
        raise ValueError(f"Invalid `ci_method`. Must be one of {valid_ci_methods}.")
    valid_tie_break_methods = ['robust', 'lwp']
    if tie_break_method not in valid_tie_break_methods:
        raise ValueError(f"Invalid `tie_break_method`. Must be one of {valid_tie_break_methods}.")
    valid_autocorr_methods = ['none', 'auto', 'block_bootstrap', 'yue_wang']
    if autocorr_method not in valid_autocorr_methods:
        raise ValueError(f"Invalid `autocorr_method`. Must be one of {valid_autocorr_methods}.")

    analysis_notes = []
    captured_warnings = []

    def _insufficient(classification, notes, warn_log):
        for w in warn_log:
            captured_warnings.append(str(w.message))
        return res('no trend', False, np.nan, 0, 0, 0, 0, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan, classification, notes,
                   np.nan, np.nan, np.nan, 0, 0, 0, np.nan, np.nan, np.nan, np.nan, '',
                   np.nan, np.nan, None, captured_warnings,
                   'insufficient', None, None)

    with warnings.catch_warnings(record=True) as w_log:
        warnings.simplefilter("always")

        data_filtered, is_datetime = _prepare_data(x, t, hicensor)
        data_filtered = data_filtered.sort_values('t', kind='stable').reset_index(drop=True)
        raw_data = data_filtered

        note = get_analysis_note(data_filtered, values_col='value', censored_col='censored')
        analysis_notes.append(note)

        n = len(data_filtered)
        if n < 2:
            return _insufficient('insufficient data', analysis_notes, w_log)

        if min_size is not None and n < min_size:
            analysis_notes.append(f'sample size ({n}) below minimum ({min_size})')

        # --- Temporal aggregation ---
        lwp_methods = ['lwp', 'lwp_median', 'lwp_robust_median']
        using_period_agg = (agg_period is not None) or (agg_method in lwp_methods)

        if using_period_agg:
            if not is_datetime:
                if agg_period is not None:
                    raise ValueError("`agg_period` can only be used with datetime-like inputs for `t`.")
                raise ValueError(f"`agg_method='{agg_method}'` can only be used with datetime-like inputs for `t`.")

            t_datetime = pd.to_datetime(data_filtered['t_original'])
            if getattr(t_datetime.dt, 'tz', None) is not None:
                t_datetime = t_datetime.dt.tz_convert(None)
            effective_period = agg_period
            if effective_period is None and agg_method in lwp_methods:
                effective_period = 'year'
            if effective_period not in _PERIOD_MAP:
                raise ValueError(f"Invalid `agg_period`: {effective_period}. "
                                 f"Must be one of {list(_PERIOD_MAP.keys())}.")
            period_freq = _PERIOD_MAP[effective_period]
            group_key = t_datetime.dt.to_period(period_freq)

            if data_filtered['censored'].any() and agg_method in ['median', 'lwp_median']:
                analysis_notes.append(f"'{agg_method}' aggregation used with censored data")

            if agg_method == 'none':
                pass
            elif agg_method == 'lwp':
                data_filtered = _value_for_time_increment(data_filtered, group_key, period_freq)
            else:
                helper_method = agg_method.replace('lwp_', '')
                data_filtered = data_filtered.assign(period_group=group_key.values)
                data_filtered = pd.DataFrame(
                    [_aggregate_by_group(g, helper_method, is_datetime, period=period_freq, return_dict=True)
                     for _, g in data_filtered.groupby('period_group', sort=True)]
                ).drop(columns=['period_group'], errors='ignore')

        elif len(data_filtered['t']) != len(np.unique(data_filtered['t'])):
            if agg_method == 'none':
                analysis_notes.append('tied timestamps present without aggregation')
            else:
                if data_filtered['censored'].any() and agg_method not in ['robust_median']:
                    analysis_notes.append(f"'{agg_method}' aggregation used with censored data")
                data_filtered = pd.DataFrame(
                    [_aggregate_by_group(g, agg_method, is_datetime, return_dict=True)
                     for _, g in data_filtered.groupby('t')]
                )

        data_filtered = data_filtered.sort_values('t', kind='stable').reset_index(drop=True)
        x_filtered = data_filtered['value'].to_numpy(dtype=float)
        t_filtered = data_filtered['t'].to_numpy(dtype=float)
        censored_filtered = data_filtered['censored'].to_numpy(dtype=bool)
        cen_type_filtered = data_filtered['cen_type'].to_numpy()

        n_filtered = len(x_filtered)
        # Mode chosen on the data actually analysed (after aggregation).
        tier_info_filtered = detect_size_tier(n_filtered, user_mode=large_dataset_mode)
        for w in tier_info_filtered['warnings']:
            warnings.warn(w, UserWarning)
        pair_cap = max_pairs if max_pairs is not None else tier_info_filtered['max_pairs']

        note = get_analysis_note(data_filtered, values_col='value', censored_col='censored',
                                 post_aggregation=True)
        analysis_notes.append(note)

        if n_filtered < 2:
            return _insufficient('insufficient data post-aggregation', analysis_notes, w_log)

        # --- Autocorrelation handling ---
        acf1 = 0.0
        n_eff = float(n_filtered)
        block_size_used = None
        var_factor = 1.0

        if autocorr_method == 'auto':
            from ._autocorr import should_apply_correction
            needs_correction, acf1, n_eff = should_apply_correction(
                x_filtered, t_filtered, max_pairs=pair_cap, random_state=random_state
            )
            if needs_correction:
                analysis_notes.append(f'Autocorrelation detected (ACF1={acf1:.3f}), applying block bootstrap')
                autocorr_method = 'block_bootstrap'
        elif autocorr_method in ('block_bootstrap', 'yue_wang'):
            from ._autocorr import effective_sample_size
            n_eff, acf1 = effective_sample_size(
                x_filtered, t_filtered, max_pairs=pair_cap, random_state=random_state
            )

        s, var_s, D, Tau = _mk_score_and_var_censored(
            x_filtered, t_filtered, censored_filtered, cen_type_filtered,
            tau_method=tau_method, mk_test_method=mk_test_method,
            tie_break_method=tie_break_method
        )

        if autocorr_method == 'block_bootstrap':
            from ._bootstrap import block_bootstrap_mann_kendall, detrended_block_size
            if block_size == 'auto':
                block_size_used = detrended_block_size(
                    x_filtered, t_filtered, censored_filtered, cen_type_filtered,
                    method=sens_slope_method, lt_mult=lt_mult, gt_mult=gt_mult,
                    max_pairs=pair_cap, random_state=random_state)
            else:
                block_size_used = int(block_size)

            p, _, s_boot_dist = block_bootstrap_mann_kendall(
                x_filtered, t_filtered, censored_filtered, cen_type_filtered,
                block_size=block_size_used, n_bootstrap=n_bootstrap,
                tau_method=tau_method, mk_test_method=mk_test_method,
                tie_break_method=tie_break_method, random_state=random_state,
                sens_slope_method=sens_slope_method, lt_mult=lt_mult, gt_mult=gt_mult,
                max_pairs=pair_cap
            )
            z = norm.ppf(1 - p / 2) * np.sign(s)
            h = p < alpha
            var_s = np.var(s_boot_dist, ddof=1)
            if continuous_confidence:
                trend = 'decreasing' if z < 0 else ('increasing' if z > 0 else 'indeterminate')
            else:
                trend = ('decreasing' if z < 0 else 'increasing') if h else 'no trend'

        elif autocorr_method == 'yue_wang':
            var_factor = n_filtered / n_eff
            var_s = var_s * var_factor
            z = _z_score(s, var_s)
            p, h, trend = _p_value(z, alpha, continuous_confidence=continuous_confidence)
            analysis_notes.append(f'Yue-Wang correction applied (n_eff={n_eff:.1f})')

        else:
            z = _z_score(s, var_s)
            p, h, trend = _p_value(z, alpha, continuous_confidence=continuous_confidence)

        # Variance for the slope interval.  'lwp' (LWP-TRENDS) recomputes it
        # ignoring censoring; any autocorrelation inflation is kept.
        var_s_ci = var_s
        if ci_method == 'lwp':
            _, var_s_unc, _, _ = _mk_score_and_var_censored(
                x_filtered, t_filtered, np.zeros(n_filtered, dtype=bool),
                np.full(n_filtered, 'not', dtype=object),
                tau_method=tau_method, mk_test_method=mk_test_method,
                tie_break_method=tie_break_method
            )
            var_s_ci = var_s_unc * var_factor
            if autocorr_method == 'block_bootstrap':
                var_s_ci = var_s

        C, Cd = _mk_probability(p, s)

        # --- Slope ---
        slope, intercept, lower_ci, upper_ci = np.nan, np.nan, np.nan, np.nan
        sen_prob, sen_prob_max, sen_prob_min = np.nan, np.nan, np.nan
        slopes = np.array([])
        sampled_total = None
        pairs_used = None
        ats_approx_error = None
        has_censored = bool(np.any(censored_filtered))

        if sens_slope_method == 'ats' and has_censored:
            ats_results = ats_slope(
                x=t_filtered, y=x_filtered, censored=censored_filtered,
                cen_type=cen_type_filtered, lod=x_filtered, var_s=var_s_ci,
                ci_alpha=alpha, max_pairs=pair_cap, random_state=random_state
            )
            slope = ats_results['beta']
            intercept = ats_results['intercept']
            lower_ci, upper_ci = ats_results['ci_lower'], ats_results['ci_upper']
            sen_prob, sen_prob_max, sen_prob_min = ats_results['sen_probability']
            analysis_notes.extend(ats_results['notes'])
            if pair_cap is not None:
                pairs_used = ats_results['pairs_used']
                ats_approx_error = ats_results.get('approximation_error')
        else:
            pair_method = 'lwp' if sens_slope_method == 'lwp' else 'drop_ambiguous'
            if has_censored:
                slopes = _sens_estimator_censored_adaptive(
                    x_filtered, t_filtered, cen_type_filtered,
                    lt_mult=lt_mult, gt_mult=gt_mult, method=pair_method,
                    max_pairs=pair_cap, random_state=random_state
                )
            else:
                slopes = _sens_estimator_adaptive(
                    x_filtered, t_filtered, max_pairs=pair_cap, random_state=random_state
                )
            sampled_total = _total_pairs_if_sampled(n_filtered, pair_cap)
            if pair_cap is not None:
                pairs_used = len(slopes)

            slope = np.nanmedian(slopes) if np.any(~np.isnan(slopes)) else np.nan
            analysis_notes.append(get_sens_slope_analysis_note(slopes, t_filtered, cen_type_filtered))
            if not np.isnan(slope):
                intercept = np.nanmedian(x_filtered) - np.nanmedian(t_filtered) * slope

            lower_ci, upper_ci = _confidence_intervals(
                slopes, var_s_ci, alpha, method=ci_method, total_pairs=sampled_total)
            sen_prob, sen_prob_max, sen_prob_min = _sen_probability(
                slopes, var_s_ci, total_pairs=sampled_total)

        if autocorr_method == 'block_bootstrap':
            from ._bootstrap import block_bootstrap_confidence_intervals
            _, lower_ci, upper_ci, _ = block_bootstrap_confidence_intervals(
                x_filtered, t_filtered, censored_filtered, cen_type_filtered,
                block_size=block_size_used, n_bootstrap=n_bootstrap, alpha=alpha,
                random_state=random_state,
                method=sens_slope_method if has_censored else 'drop_ambiguous',
                lt_mult=lt_mult, gt_mult=gt_mult, max_pairs=pair_cap
            )

        # --- Slope scaling ---
        slope_per_second = slope
        scaled_slope = slope
        slope_factor = 1.0
        scaled_lower_ci = lower_ci
        scaled_upper_ci = upper_ci

        if slope_scaling and pd.notna(slope):
            if is_datetime:
                from ._helpers import _get_slope_scaling_factor
                try:
                    factor = _get_slope_scaling_factor(slope_scaling)
                    scaled_slope = slope * factor
                    scaled_lower_ci = lower_ci * factor
                    scaled_upper_ci = upper_ci * factor
                    slope_factor = factor
                    slope_units = f"{x_unit} per {slope_scaling.lower()}"
                except (ValueError, TypeError) as e:
                    warnings.warn(f"Slope scaling failed: {e}", UserWarning)
                    slope_units = f"{x_unit} per second"
            else:
                warnings.warn(
                    "Cannot apply `slope_scaling` to a numeric (non-datetime) "
                    "time vector `t`. The slope's unit is inherited from `t`.",
                    UserWarning
                )
                slope_units = f"{x_unit} per unit of t"
        elif is_datetime:
            slope_units = f"{x_unit} per second"
        else:
            slope_units = f"{x_unit} per unit of t"

        # Data summaries refer to the data before aggregation (as LWP-TRENDS).
        raw_cens = raw_data['censored'].to_numpy(dtype=bool)
        prop_censored = float(np.mean(raw_cens))
        prop_unique = raw_data['value'].nunique() / n
        n_censor_levels = int(raw_data.loc[raw_cens, 'value'].nunique())

        computation_mode = tier_info_filtered['strategy']
        approximation_error = None
        if sampled_total is not None and len(slopes) > 0:
            from ._large_dataset import sampled_median_error
            approximation_error = sampled_median_error(slopes)
        elif ats_approx_error is not None:
            approximation_error = ats_approx_error

        for w in w_log:
            captured_warnings.append(str(w.message))

    if approximation_error is not None:
        approximation_error = approximation_error * slope_factor  # same units as `slope`

    results = res(trend, h, p, z, Tau, s, var_s, scaled_slope, intercept, scaled_lower_ci, scaled_upper_ci, C, Cd,
                  '', [], sen_prob, sen_prob_max, sen_prob_min,
                  prop_censored, prop_unique, n_censor_levels,
                  slope_per_second, lower_ci, upper_ci,
                  scaled_slope, slope_units,
                  acf1, n_eff, block_size_used, captured_warnings,
                  computation_mode, pairs_used, approximation_error)

    if continuous_confidence:
        classification = classify_trend(results, category_map=category_map)
    else:
        classification = results.trend.title() if results.trend != 'no trend' else 'No Trend'

    final_notes = list(dict.fromkeys(note for note in analysis_notes if note != 'ok'))
    final_results = results._replace(classification=classification, analysis_notes=final_notes)

    for w_str in captured_warnings:
        warnings.warn(w_str, UserWarning)

    if plot_path:
        plot_trend(data_filtered, final_results, plot_path, alpha, seasonal_coloring=seasonal_coloring)
    if residual_plot_path:
        plot_residuals(data_filtered, final_results, residual_plot_path)

    return final_results
