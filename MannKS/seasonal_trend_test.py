"""
Seasonal Mann-Kendall trend test (Hirsch, Slack & Smith 1982) with a seasonal
Sen's / Akritas-Theil-Sen slope, for unequally spaced and censored data.
"""
from collections import namedtuple
import numpy as np
import pandas as pd
import warnings
from scipy.stats import norm
from ._stats import (_z_score, _p_value, _confidence_intervals, _mk_probability,
                     _mk_score_and_var_censored, _sen_probability,
                     _sens_estimator_adaptive, _sens_estimator_censored_adaptive,
                     _normalize_sens_slope_method)
from ._ats import seasonal_ats_slope
from ._datetime import (_get_season_func, _get_cycle_identifier, _infer_period)
from ._helpers import (_prepare_data, _aggregate_by_group, _value_for_time_increment)
from ._large_dataset import detect_size_tier
from .plotting import plot_trend, plot_residuals
from .analysis_notes import get_analysis_note, get_sens_slope_analysis_note
from .classification import classify_trend


from typing import Union, Optional

# Increment used for mid-observation ('lwp') aggregation for each season type.
_SEASON_TO_PERIOD = {
    'year': 'Y', 'month': 'M', 'quarter': 'Q', 'week_of_year': 'W',
    'day_of_week': 'D', 'day_of_year': 'D', 'hour': 'h', 'minute': 'min', 'second': 's',
    'biweekly': None,
}

# All seasons that should be present, per season type (None: not fixed).
_EXPECTED_SEASONS = {
    'month': range(1, 13), 'quarter': range(1, 5), 'day_of_week': range(0, 7),
    'hour': range(0, 24), 'minute': range(0, 60), 'second': range(0, 60),
}

_RESULT_FIELDS = [
    'trend', 'h', 'p', 'z', 'Tau', 's', 'var_s', 'slope', 'intercept',
    'lower_ci', 'upper_ci', 'C', 'Cd', 'classification', 'analysis_notes',
    'sen_probability', 'sen_probability_max', 'sen_probability_min',
    'prop_censored', 'prop_unique', 'n_censor_levels',
    'slope_per_second', 'scaled_slope', 'slope_units',
    'lower_ci_per_second', 'upper_ci_per_second',
    'acf1', 'n_effective', 'block_size_used', 'warnings',
    'computation_mode', 'pairs_used', 'approximation_error'
]


def _seasonal_slope(data, seasons, sens_slope_method, lt_mult, gt_mult,
                    pair_cap=None, random_state=None):
    """Seasonal slope point estimate (used by the bootstrap)."""
    x = data['value'].to_numpy(dtype=float)
    t = data['t'].to_numpy(dtype=float)
    c = data['censored'].to_numpy(dtype=bool)
    ct = data['cen_type'].to_numpy()
    if sens_slope_method == 'ats' and c.any():
        return seasonal_ats_slope(t, x, c, seasons, cen_type=ct, var_s=0.0,
                                  max_pairs=pair_cap, random_state=random_state,
                                  compute_intercept=False)['beta']
    method = 'lwp' if sens_slope_method == 'lwp' else 'drop_ambiguous'
    out = []
    for s in np.unique(seasons):
        m = seasons == s
        if m.sum() < 2:
            continue
        if c[m].any():
            out.append(_sens_estimator_censored_adaptive(x[m], t[m], ct[m], lt_mult, gt_mult, method))
        else:
            out.append(_sens_estimator_adaptive(x[m], t[m]))
    if not out:
        return np.nan
    allv = np.concatenate(out)
    return np.nanmedian(allv) if np.any(~np.isnan(allv)) else np.nan


def seasonal_trend_test(
    x: Union[np.ndarray, pd.DataFrame],
    t: np.ndarray,
    period: Optional[int] = None,
    alpha: float = 0.05,
    agg_method: str = 'none',
    agg_period: Optional[str] = None,
    season_type: str = 'month',
    hicensor: Union[bool, float] = False,
    plot_path: Optional[str] = None,
    residual_plot_path: Optional[str] = None,
    lt_mult: float = 0.5,
    gt_mult: float = 1.1,
    sens_slope_method: str = 'ats',
    tau_method: str = 'b',
    min_size_per_season: Optional[int] = 5,
    mk_test_method: str = 'robust',
    ci_method: str = 'direct',
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
    max_per_season: Optional[int] = None,
    random_state: Optional[int] = None,
    tie_break_method: str = 'robust'
) -> namedtuple:
    """
    Seasonal Mann-Kendall trend test with a seasonal slope estimate.

    S and Var(S) are summed over seasons (Hirsch, Slack & Smith 1982, as in
    LWP-TRENDS ``SeasonalKendall``; no inter-season covariance term).  The
    slope is the median of the within-season pairwise slopes (seasonal Sen) or
    the stratified Akritas-Theil-Sen estimate for censored data.

    Parameters (see ``trend_test`` for the shared ones)
    ----------
    period : int, optional
        Number of seasons per cycle.  Inferred from ``season_type`` for
        datetime ``t``; required for numeric ``t``, which must then be
        expressed in season units (e.g. months, with ``period=12``): the
        season of an observation is ``floor(t - min(t)) % period``.
    season_type : str
        For datetime ``t``: 'month', 'quarter', 'day_of_week', 'week_of_year',
        'biweekly', 'day_of_year', 'hour', 'minute', 'second' or 'year'.
    agg_period : str, optional
        Not used: aggregation always groups by season and cycle (as defined
        by ``season_type``/``period``).  Kept for backward compatibility.
    agg_method : str
        One value per season and cycle: 'lwp' (observation closest to the
        mid-point of the increment, LWP-TRENDS ``UseMidObs=TRUE``), 'median'
        / 'lwp_median', 'robust_median' / 'lwp_robust_median', 'middle',
        'middle_lwp', or 'none'.
    sens_slope_method : {'ats', 'lwp', 'drop_ambiguous'}, default 'ats'
        Slope estimator for censored data; see ``trend_test``.
    autocorr_method : {'none', 'block_bootstrap'}
        'block_bootstrap' resamples whole cycles (e.g. years) in circular
        blocks: detrended within seasons for the p-value, and as observed
        (pairs bootstrap) for the slope's percentile interval.
    tie_break_method : {'robust', 'lwp'}
        See ``trend_test``.

    Returns
    -------
    namedtuple ``Seasonal_Mann_Kendall_Test``; fields as in ``trend_test``.
    """

    x_arr = np.asarray(x) if not isinstance(x, pd.DataFrame) else x
    t_arr = np.asarray(t)
    if len(x_arr) != len(t_arr):
        raise ValueError(f"Input vectors `x` and `t` must have the same length. Got {len(x_arr)} and {len(t_arr)}.")
    if not 0 < alpha < 1:
        raise ValueError(f"Significance level `alpha` must be between 0 and 1. Got {alpha}.")

    res = namedtuple('Seasonal_Mann_Kendall_Test', _RESULT_FIELDS)

    agg_aliases = {'lwp_median': 'median', 'lwp_robust_median': 'robust_median'}
    valid_agg_methods = ['none', 'median', 'robust_median', 'middle', 'middle_lwp', 'lwp',
                         'lwp_median', 'lwp_robust_median']
    if agg_method not in valid_agg_methods:
        raise ValueError(f"Invalid `agg_method`. Must be one of {valid_agg_methods}.")
    agg_method = agg_aliases.get(agg_method, agg_method)
    sens_slope_method = _normalize_sens_slope_method(sens_slope_method)
    if tau_method not in ['a', 'b']:
        raise ValueError("Invalid `tau_method`. Must be one of ['a', 'b'].")
    if mk_test_method not in ['robust', 'lwp']:
        raise ValueError("Invalid `mk_test_method`. Must be one of ['robust', 'lwp'].")
    if ci_method not in ['direct', 'lwp']:
        raise ValueError("Invalid `ci_method`. Must be one of ['direct', 'lwp'].")
    if tie_break_method not in ['robust', 'lwp']:
        raise ValueError("Invalid `tie_break_method`. Must be one of ['robust', 'lwp'].")
    valid_autocorr_methods = ['none', 'block_bootstrap']
    if autocorr_method not in valid_autocorr_methods:
        # Variance-correction methods are not defined here for the seasonal
        # test; the cycle block bootstrap is the supported approach.
        raise ValueError(f"Invalid `autocorr_method` for seasonal test. Must be one of {valid_autocorr_methods}.")

    analysis_notes = []
    captured_warnings = []
    sens_slope_notes = []

    with warnings.catch_warnings(record=True) as w_log:
        warnings.simplefilter("always")

        n_raw = len(x_arr)
        tier_info = detect_size_tier(n_raw, user_mode=large_dataset_mode, force_tier=None)
        for w in tier_info['warnings']:
            warnings.warn(w, UserWarning)

        data_filtered, is_datetime = _prepare_data(x, t, hicensor)
        data_filtered = data_filtered.sort_values('t', kind='stable').reset_index(drop=True)
        raw_data = data_filtered

        analysis_notes.append(get_analysis_note(data_filtered, values_col='value', censored_col='censored'))

        if period is None:
            if is_datetime:
                period = _infer_period(season_type)
            else:
                raise ValueError("The `period` parameter must be specified for numeric (non-datetime) time inputs.")

        if is_datetime:
            season_func = _get_season_func(season_type, period)

        if len(data_filtered) < 2:
            for w in w_log:
                captured_warnings.append(str(w.message))
            return res('no trend', False, np.nan, 0, 0, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                       'insufficient data', analysis_notes,
                       np.nan, np.nan, np.nan, 0, 0, 0, np.nan, np.nan, '',
                       np.nan, np.nan, np.nan, np.nan, None, captured_warnings,
                       'insufficient', None, None)

        def assign_seasons(df):
            if is_datetime and season_type != 'year':
                t_pd = pd.to_datetime(df['t_original'])
                return np.asarray(season_func(t_pd)), np.asarray(_get_cycle_identifier(t_pd, season_type))
            if not is_datetime:
                t_norm = df['t'].to_numpy(dtype=float) - raw_data['t'].min()
                return (np.floor(t_norm) % period).astype(int), np.floor(t_norm / period)
            t_pd = pd.to_datetime(df['t_original'])
            return np.ones(len(df), dtype=int), np.asarray(_get_cycle_identifier(t_pd, season_type))

        if not is_datetime:
            t_vals = data_filtered['t'].to_numpy(dtype=float)
            span = t_vals.max() - t_vals.min()
            spacing = np.median(np.diff(np.unique(t_vals))) if len(np.unique(t_vals)) > 1 else 1.0
            if span < 2 * period or (period > 1 and spacing < 0.5):
                msg = ("Numeric `t` should be expressed in season units (season = floor(t - min(t)) % period); "
                       f"the data span {span:g} units for period={period}"
                       f" with median spacing {spacing:g}. Check that `t` and `period` are consistent "
                       "(e.g. months with period=12, not decimal years).")
                warnings.warn(msg, UserWarning)
                analysis_notes.append('numeric time may not be in season units')

        # --- Aggregation to one value per season and cycle ---
        if agg_method != 'none':
            seasons_agg, cycles_agg = assign_seasons(data_filtered)
            data_filtered = data_filtered.assign(cycle=cycles_agg, season=seasons_agg)
            if data_filtered['censored'].any() and agg_method not in ['robust_median', 'lwp']:
                analysis_notes.append(
                    f"WARNING: '{agg_method}' aggregation with censored data may produce "
                    f"biased results. Consider using agg_method='robust_median'."
                )
            period_alias = _SEASON_TO_PERIOD.get(season_type) if is_datetime else None
            if agg_method == 'lwp':
                group_key = pd.Series(list(zip(data_filtered['cycle'], data_filtered['season'])),
                                      index=data_filtered.index)
                data_filtered = _value_for_time_increment(data_filtered, group_key, period_alias)
            else:
                data_filtered = pd.concat(
                    [_aggregate_by_group(g, agg_method, is_datetime, period=period_alias)
                     for _, g in data_filtered.groupby(['cycle', 'season'], sort=True)],
                    ignore_index=True)
            data_filtered = data_filtered.sort_values('t', kind='stable').reset_index(drop=True)

        seasons, cycles = assign_seasons(data_filtered)
        data_filtered = data_filtered.assign(season=seasons, cycle=cycles)
        if is_datetime and season_type != 'year':
            season_range = np.unique(seasons)
            expected_seasons = _EXPECTED_SEASONS.get(season_type)
        elif not is_datetime:
            season_range = range(int(period))
            expected_seasons = range(int(period))
        else:
            season_range = [1]
            expected_seasons = None

        n_filtered = len(data_filtered)
        tier_info_filtered = detect_size_tier(n_filtered, user_mode=large_dataset_mode)

        # Stratified subsampling for very large data sets
        if tier_info_filtered['tier'] >= 2 and n_filtered > 10000:
            from ._large_dataset import stratified_seasonal_sampling
            max_per_season_val = max_per_season if max_per_season else 1000
            if data_filtered['season'].value_counts().max() > max_per_season_val:
                data_filtered = stratified_seasonal_sampling(
                    data_filtered, season_col='season',
                    max_per_season=max_per_season_val, random_state=random_state
                ).reset_index(drop=True)
                analysis_notes.append(
                    f"Large seasonal dataset: Used stratified sampling (max {max_per_season_val} obs/season)")
                n_filtered = len(data_filtered)

        seasons = data_filtered['season'].to_numpy()
        analysis_notes.append(get_analysis_note(
            data_filtered, values_col='value', censored_col='censored', is_seasonal=True,
            post_aggregation=True, season_col='season', expected_seasons=expected_seasons))

        if n_filtered < 10:
            analysis_notes.append('Insufficient data to complete Seasonal Mann Kendall')

        if min_size_per_season is not None:
            season_counts = data_filtered.groupby('season').size()
            if not season_counts.empty and season_counts.min() < min_size_per_season:
                analysis_notes.append(
                    f'minimum season size ({season_counts.min()}) below minimum ({min_size_per_season})')

        # --- Per-season Mann-Kendall statistics ---
        def season_stats(df, censored_override=False):
            s_tot, v_tot, tau_w, d_tot = 0.0, 0.0, 0.0, 0.0
            for i in season_range:
                sd = df[df['season'] == i]
                if len(sd) > 1:
                    cens = np.zeros(len(sd), bool) if censored_override else sd['censored'].to_numpy()
                    ctyp = np.full(len(sd), 'not', dtype=object) if censored_override else sd['cen_type'].to_numpy()
                    s_i, v_i, d_i, tau_i = _mk_score_and_var_censored(
                        sd['value'].to_numpy(), sd['t'].to_numpy(), cens, ctyp,
                        tau_method=tau_method, mk_test_method=mk_test_method,
                        tie_break_method=tie_break_method)
                    s_tot += s_i
                    v_tot += v_i
                    if d_i > 0:
                        tau_w += tau_i * d_i
                        d_tot += d_i
            return s_tot, v_tot, (tau_w / d_tot if d_tot > 0 else 0.0)

        s, var_s, Tau = season_stats(data_filtered)
        n_pairs_by_season = {i: int((seasons == i).sum()) * (int((seasons == i).sum()) - 1) // 2
                             for i in season_range}
        total_possible_pairs = int(sum(n_pairs_by_season.values()))
        pair_cap = max_pairs if max_pairs is not None else tier_info_filtered['max_pairs']
        sampling = pair_cap is not None and total_possible_pairs > pair_cap

        block_size_used = None
        rng = np.random.default_rng(random_state)

        if autocorr_method == 'block_bootstrap':
            from ._bootstrap import optimal_block_size, _moving_block_bootstrap_indices
            from ._autocorr import estimate_acf

            sorted_cycles = np.sort(data_filtered['cycle'].unique())
            # Detrend within each season with the seasonal slope estimate.
            beta = _seasonal_slope(data_filtered, seasons, sens_slope_method, lt_mult, gt_mult,
                                   pair_cap, random_state)
            if not np.isfinite(beta):
                beta = 0.0
            data_detrended = data_filtered.copy()
            data_detrended['value'] = data_filtered['value'] - beta * (data_filtered['t'] - data_filtered['t'].median())

            if block_size == 'auto':
                cycle_means = data_detrended.groupby('cycle')['value'].mean()
                if len(cycle_means) > 2:
                    acf_cycle, _ = estimate_acf(cycle_means.values)
                    blk_len = optimal_block_size(len(cycle_means), acf_cycle)
                else:
                    blk_len = 1
            else:
                blk_len = int(block_size)
            block_size_used = blk_len

            groups = {c: data_detrended[data_detrended['cycle'] == c] for c in sorted_cycles}
            groups_obs = {c: data_filtered[data_filtered['cycle'] == c] for c in sorted_cycles}
            s_boot = np.zeros(n_bootstrap)
            boot_slopes = np.full(n_bootstrap, np.nan)
            for b in range(n_bootstrap):
                idx = _moving_block_bootstrap_indices(len(sorted_cycles), blk_len, rng)
                # Null distribution: detrended cycles in bootstrap order.
                boot = pd.concat([groups[sorted_cycles[k]] for k in idx], ignore_index=True)
                boot['t'] = np.arange(len(boot), dtype=float)
                s_b, _, _ = season_stats(boot)
                s_boot[b] = s_b
                # Pairs bootstrap for the slope: resampled cycles keep their times.
                bobs = pd.concat([groups_obs[sorted_cycles[k]] for k in np.sort(idx)], ignore_index=True)
                boot_slopes[b] = _seasonal_slope(bobs, bobs['season'].to_numpy(), sens_slope_method,
                                                 lt_mult, gt_mult, pair_cap, random_state)

            p = (1 + np.sum(np.abs(s_boot) >= np.abs(s))) / (n_bootstrap + 1)
            z = norm.ppf(1 - p / 2) * np.sign(s)
            h = p < alpha
            var_s = np.var(s_boot, ddof=1)
            trend = _p_value(z, alpha, continuous_confidence=continuous_confidence)[2]
            if not continuous_confidence:
                trend = ('decreasing' if z < 0 else 'increasing') if h else 'no trend'
        else:
            z = _z_score(s, var_s)
            p, h, trend = _p_value(z, alpha, continuous_confidence=continuous_confidence)

        var_s_for_ci = var_s
        if ci_method == 'lwp' and autocorr_method != 'block_bootstrap':
            _, var_s_for_ci, _ = season_stats(data_filtered, censored_override=True)

        # --- Slope ---
        slope, intercept, lower_ci, upper_ci = np.nan, np.nan, np.nan, np.nan
        sen_prob, sen_prob_max, sen_prob_min = np.nan, np.nan, np.nan
        all_slopes_arr = np.array([])
        pairs_used = None
        has_censored = bool(data_filtered['censored'].any())

        if sens_slope_method == 'ats' and has_censored:
            overall_ats = seasonal_ats_slope(
                x=data_filtered['t'].to_numpy(), y=data_filtered['value'].to_numpy(),
                censored=data_filtered['censored'].to_numpy(), seasons=seasons,
                cen_type=data_filtered['cen_type'].to_numpy(), lod=data_filtered['value'].to_numpy(),
                var_s=var_s_for_ci, ci_alpha=alpha, max_pairs=pair_cap, random_state=random_state
            )
            slope = overall_ats['beta']
            intercept = overall_ats['intercept']
            lower_ci, upper_ci = overall_ats['ci_lower'], overall_ats['ci_upper']
            sen_prob, sen_prob_max, sen_prob_min = overall_ats['sen_probability']
            sens_slope_notes.extend(overall_ats['notes'])
            if pair_cap is not None:
                pairs_used = overall_ats['pairs_used']
        else:
            pair_method = 'lwp' if sens_slope_method == 'lwp' else 'drop_ambiguous'
            all_slopes = []
            for i in season_range:
                sd = data_filtered[data_filtered['season'] == i]
                n_i = len(sd)
                if n_i < 2:
                    continue
                # Sample pairs in proportion to each season's number of pairs so
                # that the pooled sample weights seasons as the full set does.
                cap_i = None
                if sampling:
                    cap_i = max(1, int(np.ceil(pair_cap * n_pairs_by_season[i] / total_possible_pairs)))
                sx, st = sd['value'].to_numpy(), sd['t'].to_numpy()
                sct = sd['cen_type'].to_numpy()
                if sd['censored'].any():
                    sl = _sens_estimator_censored_adaptive(sx, st, sct, lt_mult=lt_mult, gt_mult=gt_mult,
                                                           method=pair_method, max_pairs=cap_i,
                                                           random_state=random_state)
                else:
                    sl = _sens_estimator_adaptive(sx, st, max_pairs=cap_i, random_state=random_state)
                note = get_sens_slope_analysis_note(sl, st, sct)
                if note != "ok" and note not in sens_slope_notes:
                    sens_slope_notes.append(note)
                all_slopes.append(sl)

            if all_slopes:
                all_slopes_arr = np.concatenate(all_slopes)
            if len(all_slopes_arr) and np.any(~np.isnan(all_slopes_arr)):
                total_pairs_arg = total_possible_pairs if sampling else None
                if pair_cap is not None:
                    pairs_used = len(all_slopes_arr)
                slope = np.nanmedian(all_slopes_arr)
                intercept = np.nanmedian(data_filtered['value']) - np.nanmedian(data_filtered['t']) * slope
                lower_ci, upper_ci = _confidence_intervals(
                    all_slopes_arr, var_s_for_ci, alpha, method=ci_method, total_pairs=total_pairs_arg)
                sen_prob, sen_prob_max, sen_prob_min = _sen_probability(
                    all_slopes_arr, var_s_for_ci, total_pairs=total_pairs_arg)

        if autocorr_method == 'block_bootstrap':
            ok = boot_slopes[np.isfinite(boot_slopes)]
            if len(ok):
                lower_ci = np.percentile(ok, 100 * alpha / 2)
                upper_ci = np.percentile(ok, 100 * (1 - alpha / 2))

        analysis_notes.extend(sens_slope_notes)

        C, Cd = _mk_probability(p, s)

        raw_cens = raw_data['censored'].to_numpy(dtype=bool)
        prop_censored = float(np.mean(raw_cens))
        prop_unique = raw_data['value'].nunique() / len(raw_data)
        n_censor_levels = int(raw_data.loc[raw_cens, 'value'].nunique())

        computation_mode = tier_info_filtered['strategy']
        approximation_error = None
        if sampling and len(all_slopes_arr) > 0:
            valid = all_slopes_arr[~np.isnan(all_slopes_arr)]
            if len(valid):
                iqr = np.percentile(valid, 75) - np.percentile(valid, 25)
                approximation_error = 1.96 * iqr / np.sqrt(len(valid))

        # --- Slope scaling ---
        slope_per_second = slope
        lower_ci_per_second = lower_ci
        upper_ci_per_second = upper_ci
        scaled_slope = slope
        scaled_lower_ci = lower_ci
        scaled_upper_ci = upper_ci

        if slope_scaling and pd.notna(slope):
            if is_datetime:
                from ._helpers import _get_slope_scaling_factor
                try:
                    factor = _get_slope_scaling_factor(slope_scaling)
                    scaled_slope = slope * factor
                    scaled_lower_ci = lower_ci * factor if pd.notna(lower_ci) else lower_ci
                    scaled_upper_ci = upper_ci * factor if pd.notna(upper_ci) else upper_ci
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

        for w in w_log:
            captured_warnings.append(str(w.message))

    results = res(trend, h, p, z, Tau, s, var_s, scaled_slope, intercept, scaled_lower_ci, scaled_upper_ci, C, Cd,
                  '', [], sen_prob, sen_prob_max, sen_prob_min,
                  prop_censored, prop_unique, n_censor_levels,
                  slope_per_second, scaled_slope, slope_units,
                  lower_ci_per_second, upper_ci_per_second,
                  np.nan, np.nan, block_size_used, captured_warnings,
                  computation_mode, pairs_used, approximation_error)

    if continuous_confidence:
        classification = classify_trend(results, category_map=category_map)
    else:
        classification = results.trend.title() if results.trend != 'no trend' else 'No Trend'

    final_notes = [note for note in analysis_notes if note != 'ok']
    final_results = results._replace(classification=classification, analysis_notes=final_notes)

    for w_str in captured_warnings:
        warnings.warn(w_str, UserWarning)

    if plot_path:
        plot_trend(data_filtered, final_results, plot_path, alpha, seasonal_coloring=seasonal_coloring)
    if residual_plot_path:
        plot_residuals(data_filtered, final_results, residual_plot_path)

    return final_results
