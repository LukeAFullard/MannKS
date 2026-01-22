"""
This script provides a modified version of the Seasonal Mann-Kendall test
and Sen's slope estimator to handle unequally spaced time series data.
"""
from collections import namedtuple
import numpy as np
import pandas as pd
from pandas import DataFrame
import warnings
from scipy.stats import norm
from ._stats import (_z_score, _p_value,
                   _sens_estimator_unequal_spacing, _confidence_intervals,
                   _mk_probability, _mk_score_and_var_censored,
                   _sens_estimator_censored, _sen_probability,
                   _sens_estimator_adaptive, _sens_estimator_censored_adaptive)
from ._ats import ats_slope, seasonal_ats_slope
from ._datetime import (_get_season_func, _get_cycle_identifier, _get_time_ranks, _infer_period)
from ._helpers import (_prepare_data, _aggregate_by_group, _value_for_time_increment)
from ._large_dataset import detect_size_tier
from .plotting import plot_trend, plot_residuals
from .analysis_notes import get_analysis_note, get_sens_slope_analysis_note
from .classification import classify_trend


from typing import Union, Optional

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
    sens_slope_method: str = 'unbiased',
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
    random_state: Optional[int] = None
) -> namedtuple:
    """
    Seasonal Mann-Kendall trend test with Sen's slope for time series data.

    New in v0.5.0: Large Dataset Support
    ------------------------------------
    For seasonal data with many observations, automatic optimizations maintain
    both statistical validity and seasonal balance.

    Key difference from trend_test:
        Seasonal tests require balanced representation across all seasons.
        Fast mode uses stratified sampling to ensure each season contributes
        proportionally to the final result.

    Parameters
    ----------
    large_dataset_mode : str, default 'auto'
        Controls algorithm selection for large datasets:
        - 'auto': Automatic based on sample size (recommended)
        - 'full': Force exact calculations (may be slow/crash for large n)
        - 'fast': Force fast approximations
        - 'aggregate': Force aggregation workflow

    max_pairs : int, optional
        Maximum number of pairs to sample in fast mode. Default is 100,000.
        Higher values increase accuracy but also computation time.

    max_per_season : int, optional
        In fast mode, maximum observations to use per season. Default is 1000.
        Total dataset size after stratification approx max_per_season * n_seasons.
        Example: 12 months * 1000 obs/month = 12,000 total observations used.

    random_state : int, optional
        Random seed for reproducible results in fast mode. Set this for
        deterministic output when using fast approximations.
    """

    # --- Basic Input Validation ---
    x_arr = np.asarray(x) if not isinstance(x, pd.DataFrame) else x
    t_arr = np.asarray(t)
    if len(x_arr) != len(t_arr):
        raise ValueError(f"Input vectors `x` and `t` must have the same length. Got {len(x_arr)} and {len(t_arr)}.")
    if not 0 < alpha < 1:
        raise ValueError(f"Significance level `alpha` must be between 0 and 1. Got {alpha}.")

    res = namedtuple('Seasonal_Mann_Kendall_Test', [
        'trend', 'h', 'p', 'z', 'Tau', 's', 'var_s', 'slope', 'intercept',
        'lower_ci', 'upper_ci', 'C', 'Cd', 'classification', 'analysis_notes',
        'sen_probability', 'sen_probability_max', 'sen_probability_min',
        'prop_censored', 'prop_unique', 'n_censor_levels',
        'slope_per_second', 'scaled_slope', 'slope_units',
        'lower_ci_per_second', 'upper_ci_per_second',
        'acf1', 'n_effective', 'block_size_used', 'warnings',
        'computation_mode', 'pairs_used', 'approximation_error'
    ])

    # --- Method String Validation ---
    valid_agg_methods = ['none', 'median', 'robust_median', 'middle', 'middle_lwp', 'lwp']
    if agg_method not in valid_agg_methods:
        raise ValueError(f"Invalid `agg_method`. Must be one of {valid_agg_methods}.")

    valid_sens_slope_methods = ['unbiased', 'nan', 'lwp', 'ats']
    if sens_slope_method not in valid_sens_slope_methods:
        raise ValueError(f"Invalid `sens_slope_method`. Must be one of {valid_sens_slope_methods}.")

    valid_tau_methods = ['a', 'b']
    if tau_method not in valid_tau_methods:
        raise ValueError(f"Invalid `tau_method`. Must be one of {valid_tau_methods}.")

    valid_mk_test_methods = ['robust', 'lwp']
    if mk_test_method not in valid_mk_test_methods:
        raise ValueError(f"Invalid `mk_test_method`. Must be one of {valid_mk_test_methods}.")

    valid_ci_methods = ['direct', 'lwp']
    if ci_method not in valid_ci_methods:
        raise ValueError(f"Invalid `ci_method`. Must be one of {valid_ci_methods}.")

    valid_autocorr_methods = ['none', 'block_bootstrap']
    if autocorr_method not in valid_autocorr_methods:
        # Note: 'auto' and 'yue_wang' methods are not fully supported for the seasonal test
        # in this version. Block bootstrap is the recommended approach.
        raise ValueError(f"Invalid `autocorr_method` for seasonal test. Must be one of {valid_autocorr_methods}.")

    analysis_notes = []
    captured_warnings = []

    with warnings.catch_warnings(record=True) as w_log:
        warnings.simplefilter("always")

        # Detect size tier before filtering
        n_raw = len(np.asarray(x) if not isinstance(x, pd.DataFrame) else x)
        tier_info = detect_size_tier(
            n_raw,
            user_mode=large_dataset_mode,
            force_tier=None
        )

        # Add warnings from tier detection
        for w in tier_info['warnings']:
            warnings.warn(w, UserWarning)

        data_filtered, is_datetime = _prepare_data(x, t, hicensor)

        # Ensure data is sorted by time. This is critical for:
        # 1. Correct application of censor rules in Sen's slope (which assume j > i implies t[j] > t[i]).
        # 2. Consistent results regardless of input order.
        if is_datetime:
            data_filtered = data_filtered.sort_values(by='t_original')
        else:
            data_filtered = data_filtered.sort_values(by='t')

        note = get_analysis_note(data_filtered, values_col='value', censored_col='censored')
        analysis_notes.append(note)

        # --- Infer or Validate Period ---
        if period is None:
            if is_datetime:
                # Try to infer period from season_type
                # If inference returns None (e.g. week_of_year), period stays None.
                # _get_season_func handles period=None by skipping validation.
                period = _infer_period(season_type)
            else:
                raise ValueError("The `period` parameter must be specified for numeric (non-datetime) time inputs.")

        if is_datetime:
            season_func = _get_season_func(season_type, period)

        if len(data_filtered) < 2:
            # Manually collect warnings before returning
            for w in w_log:
                 captured_warnings.append(str(w.message))
            return res('no trend', False, np.nan, 0, 0, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                    'insufficient data', analysis_notes,
                    np.nan, np.nan, np.nan, 0, 0, 0, np.nan, np.nan, '',
                    np.nan, np.nan, np.nan, np.nan, None, captured_warnings,
                    'insufficient', None, None)

        # --- Aggregation Logic ---
        if agg_method != 'none':
            if is_datetime:
                t_pd = pd.to_datetime(data_filtered['t_original'])
                cycles = _get_cycle_identifier(t_pd, season_type)
                seasons_agg = season_func(t_pd) if season_type != 'year' else np.ones(len(t_pd))
            else:
                # period is guaranteed to be not None here due to check above
                t_numeric_agg = data_filtered['t'].to_numpy()
                t_normalized = t_numeric_agg - t_numeric_agg[0]
                cycles = np.floor(t_normalized / period)
                seasons_agg = np.floor(t_normalized % period)
            data_filtered['cycle'] = cycles
            data_filtered['season'] = seasons_agg

        if agg_method == 'lwp':
            # The 'lwp' method uses a specific aggregation that chooses one value per time increment.

            # Define mapping from season_type to pandas offset alias
            SEASON_TO_OFFSET = {
                'year': 'Y', 'month': 'M', 'quarter': 'Q',
                'day': 'D', 'hour': 'H', 'minute': 'T', 'second': 'S',
                'week': 'W', 'week_of_year': 'W', 'day_of_year': 'D'
            }
            period_alias = SEASON_TO_OFFSET.get(season_type, 'M')

            # Handle numeric data conversion if needed
            # We ensure t_original is datetime-like for the helper function
            if not is_datetime:
                 data_filtered['t_original'] = pd.to_datetime(data_filtered['t_original'], unit='s', origin='unix')

            # Construct group key from the cycle/season columns calculated above
            # This matches the (cycle, season) grouping used for other methods
            group_key = pd.Series(list(zip(data_filtered['cycle'], data_filtered['season'])), index=data_filtered.index)

            data_filtered = _value_for_time_increment(data_filtered, group_key, period_alias)

        elif agg_method != 'none':
            if data_filtered['censored'].any() and agg_method not in ['robust_median', 'lwp']:
                 analysis_notes.append(
                    f"WARNING: '{agg_method}' aggregation with censored data may produce "
                    f"biased results. Consider using agg_method='robust_median'."
                )

            # We reuse the cycle/season calculation from the initial block
            # Only recalculate if explicitly needed, but here we can just use the existing columns.
            # However, to be safe and match previous logic strictly,
            # we will use the columns we just set.

            data_filtered['season_agg'] = data_filtered['season']

            agg_data_list = [
                _aggregate_by_group(group, agg_method, is_datetime)
                for _, group in data_filtered.groupby(['cycle', 'season_agg'])
            ]
            data_filtered = pd.concat(agg_data_list, ignore_index=True)


        # --- Trend Analysis ---
        if is_datetime and season_type != 'year':
            t_pd = pd.to_datetime(data_filtered['t_original'])
            seasons = season_func(t_pd)
            cycles = _get_cycle_identifier(t_pd, season_type)
            season_range = np.unique(seasons)
        elif not is_datetime:
            # period is guaranteed to be not None here
            t_normalized = data_filtered['t'] - data_filtered['t'].min()
            seasons = (np.floor(t_normalized) % period).astype(int)
            cycles = np.floor(t_normalized / period)
            season_range = range(int(period))
        else: # is_datetime and season_type == 'year'
            seasons = np.ones(len(data_filtered))
            cycles = _get_cycle_identifier(pd.to_datetime(data_filtered['t_original']), season_type) if is_datetime else np.zeros(len(data_filtered))
            season_range = [1]


        data_filtered['season'] = seasons
        data_filtered['cycle'] = cycles

        # Re-check size after aggregation
        n_filtered = len(data_filtered)
        tier_info_filtered = detect_size_tier(
            n_filtered,
            user_mode=large_dataset_mode
        )

        # SEASONAL-SPECIFIC: Stratified sampling if needed
        # We check if tier is fast (2) or aggregate (3) which defaults to fast if not aggregated yet
        if tier_info_filtered['tier'] >= 2 and n_filtered > 10000 and sens_slope_method != 'ats':
             from ._large_dataset import stratified_seasonal_sampling

             max_per_season_val = max_per_season if max_per_season else 1000

             # Only stratify if we are actually above the effective threshold
             # Check if any season is huge
             season_counts = data_filtered['season'].value_counts()
             if season_counts.max() > max_per_season_val:
                 data_filtered = stratified_seasonal_sampling(
                     data_filtered,
                     season_col='season',
                     max_per_season=max_per_season_val,
                     random_state=random_state
                 )

                 analysis_notes.append(
                     f"Large seasonal dataset: Used stratified sampling "
                     f"(max {max_per_season_val} obs/season)"
                 )

                 # Re-calculate size info after stratification
                 n_filtered = len(data_filtered)
                 # We still want to use 'fast' strategies if we were originally large
                 # But max_pairs might be adjusted implicitly by the data reduction

        note = get_analysis_note(data_filtered, values_col='value', censored_col='censored',
                                 is_seasonal=True, post_aggregation=True, season_col='season')
        analysis_notes.append(note)


        # Sample size validation per season
        if min_size_per_season is not None:
            season_counts = data_filtered.groupby('season').size()
            if not season_counts.empty:
                min_season_n = season_counts.min()
                if min_season_n < min_size_per_season:
                    analysis_notes.append(f'minimum season size ({min_season_n}) below minimum ({min_size_per_season})')


        # --- Bootstrap Logic for Seasonality ---
        block_size_used = None
        total_possible_pairs = 0

        if autocorr_method == 'block_bootstrap':
            # Bootstrap approach:
            # 1. Identify unique cycles (e.g. years)
            # 2. Resample cycles with replacement (block bootstrap where block = 1 cycle)
            # 3. For each resample, calculate the Seasonal MK Statistic (S)
            # 4. Construct distribution of S

            unique_cycles = data_filtered['cycle'].unique()
            n_cycles = len(unique_cycles)

            # Calculate observed S
            s_obs = 0
            for i in season_range:
                season_mask = data_filtered['season'] == i
                season_data = data_filtered[season_mask]
                n = len(season_data)
                if n > 1:
                    s_i, _, _, _ = _mk_score_and_var_censored(
                        season_data['value'], season_data['t'], season_data['censored'],
                        season_data['cen_type'], tau_method=tau_method, mk_test_method=mk_test_method
                    )
                    s_obs += s_i
                    total_possible_pairs += n * (n - 1) // 2

            # Bootstrap
            s_boot_dist = np.zeros(n_bootstrap)

            # Strategy: Null hypothesis is "no trend".
            # Shuffling whole cycles (years) destroys trend but preserves seasonality and
            # within-year autocorrelation. To preserve serial correlation between years,
            # moving block bootstrap on the CYCLES is used.

            # Sort data by cycle then season
            data_sorted = data_filtered.sort_values(['cycle', 'season'])

            # Data must be detrended to test H0.
            # Observed seasonal slopes are used for detrending.

            data_detrended = data_sorted.copy()
            for i in season_range:
                mask = data_sorted['season'] == i
                season_subset = data_sorted[mask]
                if len(season_subset) > 1:
                    # Calculate simple Theil-Sen for this season
                    if np.any(season_subset['censored']):
                         slopes = _sens_estimator_censored_adaptive(
                            season_subset['value'].values, season_subset['t'].values, season_subset['cen_type'].values,
                            method=sens_slope_method, lt_mult=lt_mult, gt_mult=gt_mult,
                            max_pairs=max_pairs if max_pairs else tier_info_filtered['max_pairs'],
                            random_state=random_state
                         )
                    else:
                         slopes = _sens_estimator_adaptive(
                            season_subset['value'].values, season_subset['t'].values,
                            max_pairs=max_pairs if max_pairs else tier_info_filtered['max_pairs'],
                            random_state=random_state
                         )

                    slope_i = np.nanmedian(slopes)
                    if np.isnan(slope_i): slope_i = 0

                    # Remove trend
                    # t_center = np.median(season_subset['t'])
                    data_detrended.loc[mask, 'value'] = data_detrended.loc[mask, 'value'] - slope_i * (data_detrended.loc[mask, 't'] - data_detrended.loc[mask, 't'].iloc[0])

            # Moving block bootstrap on CYCLES
            # Treat each cycle as a "point" in the block bootstrap
            from ._bootstrap import moving_block_bootstrap, optimal_block_size
            from ._autocorr import estimate_acf

            # Default block size for seasonal is 1 (year/cycle) if 'auto', or user specified
            if block_size == 'auto':
                # Estimate ACF of annual (cycle) averages to determine optimal block size
                # We aggregate by cycle to get a single value per year/cycle
                # This is a robust way to estimate inter-annual dependence
                cycle_means = data_filtered.groupby('cycle')['value'].mean()
                if len(cycle_means) > 2:
                    acf_cycle, _ = estimate_acf(cycle_means.values)
                    blk_len = optimal_block_size(len(cycle_means), acf_cycle)
                else:
                    blk_len = 2 # Default fallback for very short series
            else:
                blk_len = int(block_size)

            block_size_used = blk_len

            sorted_cycles = np.sort(unique_cycles)

            for b in range(n_bootstrap):
                # Bootstrap the cycle INDICES
                # Note: We need a sequence of cycles.
                # If cycles are missing (gaps), this is complex. Assuming roughly continuous.
                # We map cycles to 0..N-1
                cycle_indices = np.arange(len(sorted_cycles))
                boot_indices = moving_block_bootstrap(cycle_indices, blk_len)

                # Construct bootstrap sample.
                # The bootstrap test for H0 (No trend) requires destroying the trend
                # while preserving autocorrelation structure. This is achieved by
                # resampling blocks of detrended data.

                # Reconstruct data
                boot_data_list = []
                for i, idx in enumerate(boot_indices):
                    cycle_val = sorted_cycles[idx]
                    cycle_data = data_detrended[data_detrended['cycle'] == cycle_val].copy()

                    # Values from bootstrapped blocks are treated as a sequential time series.
                    # This effectively tests the permuted values against the original time order
                    # (represented by monotonic increase).
                    boot_data_list.append(cycle_data)

                boot_data = pd.concat(boot_data_list)

                # Define "Time" for this bootstrap sample to ensure monotonic sequence
                # matching the block order, required for the MK test.
                # Seasonality must be preserved (which is inherent in the cycle_data),
                # but 't' must increase monotonically.

                # Create synthetic time index
                boot_data['t_boot'] = np.arange(len(boot_data))

                s_b = 0
                for i_season in season_range:
                    mask = boot_data['season'] == i_season
                    subset = boot_data[mask]
                    if len(subset) > 1:
                        s_i, _, _, _ = _mk_score_and_var_censored(
                            subset['value'], subset['t_boot'], subset['censored'],
                            subset['cen_type'], tau_method=tau_method, mk_test_method=mk_test_method
                        )
                        s_b += s_i
                s_boot_dist[b] = s_b

            # Calculate P-value
            p_boot = np.mean(np.abs(s_boot_dist) >= np.abs(s_obs))

            # Override standard results
            s = s_obs
            p = p_boot
            # Back-calculate Z
            p_safe = np.clip(p, 1e-10, 1 - 1e-10)
            z = norm.ppf(1 - p_safe/2) * np.sign(s)
            h = p < alpha

            # Use empirical variance from bootstrap distribution
            var_s = np.var(s_boot_dist, ddof=1)

            # For reference (and for Tau calc), we still need the analytic components
            # although var_s will be overwritten by the bootstrap estimate above.
            s_analytic, var_s_analytic, denom = 0, 0, 0
            tau_weighted_sum = 0
            denom_sum = 0
            for i in season_range:
                season_mask = data_filtered['season'] == i
                season_data = data_filtered[season_mask]
                n = len(season_data)
                if n > 1:
                    _, var_s_season, d_season, tau_season = _mk_score_and_var_censored(
                        season_data['value'], season_data['t'], season_data['censored'],
                        season_data['cen_type'], tau_method=tau_method, mk_test_method=mk_test_method
                    )
                    var_s_analytic += var_s_season
                    if d_season > 0:
                        tau_weighted_sum += tau_season * d_season
                        denom_sum += d_season

        # --- Standard Analytic Calculation ---
        else:
            s, var_s, denom = 0, 0, 0
            all_slopes = []
            seasonal_slopes = []
            tau_weighted_sum = 0
            denom_sum = 0
            sens_slope_notes = set()
            total_possible_pairs = 0

            for i in season_range:
                season_mask = data_filtered['season'] == i
                season_data = data_filtered[season_mask]
                n = len(season_data)

                if n > 1:
                    s_season, var_s_season, d_season, tau_season = _mk_score_and_var_censored(
                        season_data['value'], season_data['t'], season_data['censored'],
                        season_data['cen_type'], tau_method=tau_method, mk_test_method=mk_test_method
                    )
                    s += s_season
                    var_s += var_s_season
                    if d_season > 0:
                        tau_weighted_sum += tau_season * d_season
                        denom_sum += d_season

                    # Accumulate total pairs for CI scaling
                    total_possible_pairs += n * (n - 1) // 2

        # Sen's slope calculation (Same as before)
        slope_data = data_filtered
        var_s_for_ci = var_s

        # LWP-TRENDS Compatibility Mode:
        if ci_method == 'lwp':
            var_s_ci_accum = 0
            for i in season_range:
                season_mask = slope_data['season'] == i
                season_data = slope_data[season_mask]
                n = len(season_data)
                if n > 1:
                    season_censored = np.zeros_like(season_data['censored'], dtype=bool)
                    season_cen_type = np.full_like(season_data['cen_type'], 'not')

                    _, var_s_unc, _, _ = _mk_score_and_var_censored(
                        season_data['value'], season_data['t'], season_censored,
                        season_cen_type, tau_method=tau_method, mk_test_method=mk_test_method
                    )
                    var_s_ci_accum += var_s_unc
            var_s_for_ci = var_s_ci_accum

        if sens_slope_method == 'ats':
            # Use Stratified ATS: Sum of within-season scores.
            overall_ats = seasonal_ats_slope(
                x=slope_data['t'].to_numpy(),
                y=slope_data['value'].to_numpy(),
                censored=slope_data['censored'].to_numpy(),
                seasons=slope_data['season'].to_numpy(),
                cen_type=slope_data['cen_type'].to_numpy(),
                lod=slope_data['value'].to_numpy(),
                bootstrap_ci=True,
                ci_alpha=alpha
            )
            slope = overall_ats['beta']
            intercept = overall_ats['intercept']
            lower_ci = overall_ats.get('ci_lower', np.nan)
            upper_ci = overall_ats.get('ci_upper', np.nan)
            sen_prob, sen_prob_max, sen_prob_min = np.nan, np.nan, np.nan
            if overall_ats.get('notes'):
                sens_slope_notes.update(overall_ats['notes'])

        else: # For 'lwp' or 'unbiased' (nan) methods
            all_slopes = []
            sens_slope_notes = set()
            for i in season_range:
                season_mask = slope_data['season'] == i
                season_data = slope_data[season_mask]
                n = len(season_data)
                if n > 1:
                    season_x = season_data['value'].to_numpy()
                    season_t = season_data['t'].to_numpy()
                    season_censored = season_data['censored'].to_numpy()
                    season_cen_type = season_data['cen_type'].to_numpy()
                    if np.any(season_censored):
                        slopes = _sens_estimator_censored_adaptive(
                            season_x, season_t, season_cen_type,
                            lt_mult=lt_mult, gt_mult=gt_mult, method=sens_slope_method,
                            max_pairs=max_pairs if max_pairs else tier_info_filtered['max_pairs'],
                            random_state=random_state
                        )
                    else:
                        slopes = _sens_estimator_adaptive(
                            season_x, season_t,
                            max_pairs=max_pairs if max_pairs else tier_info_filtered['max_pairs'],
                            random_state=random_state
                        )
                    note = get_sens_slope_analysis_note(slopes, season_t, season_cen_type)
                    if note != "ok":
                        sens_slope_notes.add(note)
                    all_slopes.extend(slopes)

            if sens_slope_notes:
                analysis_notes.extend(list(sens_slope_notes))

            if not all_slopes:
                slope, intercept, lower_ci, upper_ci = np.nan, np.nan, np.nan, np.nan
                sen_prob, sen_prob_max, sen_prob_min = np.nan, np.nan, np.nan
            else:
                all_slopes_arr = np.asarray(all_slopes)
                slope = np.nanmedian(all_slopes_arr)
                intercept = np.nanmedian(data_filtered['value']) - np.nanmedian(data_filtered['t']) * slope if pd.notna(slope) else np.nan

                # Pass total_possible_pairs (accumulated from seasons) to correctly scale ranks
                # when all_slopes_arr contains a subsample of slopes.
                # If block_bootstrap was used, total_possible_pairs is not calculated above,
                # but this block is under `else` of block_bootstrap (or inside it but we are in the main else block here).
                # Wait, the total_possible_pairs calculation was inside the `else` (Standard Analytic) block.
                # So it is available here.

                lower_ci, upper_ci = _confidence_intervals(
                    all_slopes_arr, var_s_for_ci, alpha, method=ci_method, total_pairs=total_possible_pairs
                )
                sen_prob, sen_prob_max, sen_prob_min = _sen_probability(
                    all_slopes_arr, var_s_for_ci, total_pairs=total_possible_pairs
                )

        if sens_slope_notes:
            analysis_notes.extend(list(sens_slope_notes))

        if autocorr_method != 'block_bootstrap':
            # Calculate standard Tau if not bootstrapped
            # (If bootstrapped, we derived p/z/h directly)
            Tau = tau_weighted_sum / denom_sum if denom_sum > 0 else 0
            z = _z_score(s, var_s)
            p, h, trend = _p_value(z, alpha, continuous_confidence=continuous_confidence)
        else:
            # For bootstrap, Tau is hard to define simply, keep analytic or approximate?
            # Let's keep analytic Tau for now as a descriptive stat
            Tau = tau_weighted_sum / denom_sum if denom_sum > 0 else 0
            trend = _p_value(z, alpha, continuous_confidence=continuous_confidence)[2] # Get trend string

        C, Cd = _mk_probability(p, s)

        # Calculate metadata fields
        n_total = len(data_filtered)
        prop_censored = np.sum(data_filtered['censored']) / n_total if n_total > 0 else 0
        prop_unique = len(data_filtered['value'].unique()) / n_total if n_total > 0 else 0
        censored_values = data_filtered['value'][data_filtered['censored']]
        n_censor_levels = len(censored_values.unique()) if not censored_values.empty else 0

        # Calculate large dataset metadata
        computation_mode = tier_info_filtered['strategy']

        # Determine pairs used and approx error if applicable
        if computation_mode == 'fast' and sens_slope_method != 'ats':
             # Note: For seasonal, this is tricky because we accumulate slopes list
             # If we used adaptive slope estimation
             if 'all_slopes_arr' in locals() and len(all_slopes_arr) > 0:
                 pairs_used = len(all_slopes_arr)
                 # Estimate approximation error (IQR / sqrt(K))
                 iqr = np.percentile(all_slopes_arr, 75) - np.percentile(all_slopes_arr, 25)
                 approximation_error = 1.96 * iqr / np.sqrt(pairs_used)
             else:
                 pairs_used = None
                 approximation_error = None
        else:
             pairs_used = None
             approximation_error = None


        # --- Slope Scaling ---
        slope_per_second = slope
        lower_ci_per_second = lower_ci
        upper_ci_per_second = upper_ci
        scaled_slope = slope
        slope_units = ""
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
                    slope_units = f"{x_unit} per second" # Fallback
            else:
                warnings.warn(
                    "Cannot apply `slope_scaling` to a numeric (non-datetime) "
                    "time vector `t`. The slope's unit is inherited from `t`.",
                    UserWarning
                )
                slope_units = f"{x_unit} per unit of t" # Clarify for numeric time
        elif is_datetime:
            slope_units = f"{x_unit} per second"
        else: # Numeric time without scaling
            slope_units = f"{x_unit} per unit of t"

        # Collect warnings
        for w in w_log:
                captured_warnings.append(str(w.message))

    results = res(trend, h, p, z, Tau, s, var_s, scaled_slope, intercept, scaled_lower_ci, scaled_upper_ci, C, Cd,
                  '', [], sen_prob, sen_prob_max, sen_prob_min,
                  prop_censored, prop_unique, n_censor_levels,
                  slope_per_second, scaled_slope, slope_units,
                  lower_ci_per_second, upper_ci_per_second,
                  np.nan, np.nan, block_size_used, captured_warnings,
                  computation_mode, pairs_used, approximation_error) # acf1, n_eff not calc for seasonal

    # Final Classification and Notes
    if continuous_confidence:
        classification = classify_trend(results, category_map=category_map)
    else:
        # Classical behavior: Just capitalize the trend direction
        classification = results.trend.title() if results.trend != 'no trend' else 'No Trend'

    final_notes = [note for note in analysis_notes if note != 'ok']
    final_results = results._replace(classification=classification, analysis_notes=final_notes)

    # Re-issue warnings so they appear in stderr as well
    for w_str in captured_warnings:
        warnings.warn(w_str, UserWarning)

    if plot_path:
        plot_trend(data_filtered, final_results, plot_path, alpha, seasonal_coloring=seasonal_coloring)

    if residual_plot_path:
        plot_residuals(data_filtered, final_results, residual_plot_path)

    return final_results
