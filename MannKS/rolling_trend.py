import numpy as np
import pandas as pd
from typing import Union, Optional
import warnings
from .trend_test import trend_test
from .seasonal_trend_test import seasonal_trend_test
from ._datetime import _is_datetime_like

# Robust import for to_offset
try:
    from pandas.tseries.offsets import to_offset
except ImportError:
    try:
        from pandas.tseries.frequencies import to_offset
    except ImportError:
        to_offset = None # Will fail at runtime if needed

def rolling_trend_test(
    x: Union[np.ndarray, pd.DataFrame],
    t: np.ndarray,
    window: Union[str, int, float],
    step: Optional[Union[str, int, float]] = None,
    min_size: int = 10,
    alpha: float = 0.05,
    seasonal: bool = False,
    period: int = 12,
    season_type: str = 'month',
    slope_scaling: Optional[str] = None,
    x_unit: str = "units",
    continuous_confidence: bool = True,
    large_dataset_mode: str = 'auto',
    max_pairs: Optional[int] = None,
    random_state: Optional[int] = None,
    max_windows: int = 10000,
    **kwargs
) -> pd.DataFrame:
    """
    Calculate rolling Sen's slope over moving time windows.

    Args:
        x: Data vector (array or DataFrame column)
        t: Time vector (array of timestamps or numeric)
        window: Window size.
                - For datetime t: string (e.g., '365D', '10YE') or Timedelta-compatible string.
                  Note: Anchored offsets (e.g., 'YE', 'ME') snap to the next anchor point,
                  which may result in variable window durations (e.g., '1YE' from Jan 31
                  goes to Dec 31 of the same year). For fixed duration, use 'D' (days).
                - For numeric t: integer or float (e.g., 10, 5.5)
        step: Step size for sliding the window.
              - For datetime t: string (e.g., '1Y', '6M').
                Default for Timedeltas (e.g. '365D') is window/2.
                Default for DateOffsets (e.g. '1YE') is the full window size (non-overlapping)
                because offsets cannot be divided.
              - For numeric t: integer or float. Default is window/2.
        min_size: Minimum number of observations required in a window to calculate a trend.
        alpha: Significance level for the Mann-Kendall test (default 0.05).
        seasonal: If True, uses `seasonal_trend_test` instead of `trend_test`.
        period: The seasonal period (e.g., 12 for monthly data). Used if seasonal=True.
        season_type: The type of seasonality (e.g., 'month'). Used if seasonal=True.
        slope_scaling: Time unit for scaling the Sen's slope (e.g., 'year').
        x_unit: Unit of measurement for x (e.g., 'mg/L').
        continuous_confidence: If True (default for rolling), uses continuous confidence (C)
                               interpretation. If False, uses classical p-value testing.
        large_dataset_mode : str, default 'auto'
            Controls algorithm selection for large datasets:
            - 'auto': Automatic based on sample size (recommended)
            - 'full': Force exact calculations (may be slow/crash for large n)
            - 'fast': Force fast approximations
            - 'aggregate': As 'fast', with a warning recommending aggregation
        max_pairs : int, optional
            Maximum number of pairs to sample in fast mode.
        random_state : int, optional
            Random seed for reproducible results in fast mode.
        max_windows : int, default 10000
            Maximum number of sliding windows allowed before raising a ValueError.
        **kwargs: Additional arguments passed to `trend_test` or `seasonal_trend_test`.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a time window, containing:
            - window_start, window_end: The time bounds of the window.
            - window_center: The midpoint of the window (useful for plotting).
            - n_obs: Number of observations in the window.
            - slope: The Sen's slope.
            - lower_ci, upper_ci: Confidence intervals for the slope.
            - p_value: The p-value of the trend test.
            - h: Boolean indicating if the trend is significant (at alpha).
            - classification: The trend classification string.
            - C: Confidence in the trend direction.
            - Cd: Directional confidence.
            - tau: Kendall's Tau.
            - s: Mann-Kendall score.
            - intercept: The intercept of the trend line.
            - slope_per_second: Unscaled Sen's slope (e.g., units/second for datetime).
            - lower_ci_per_second: Unscaled lower CI.
            - upper_ci_per_second: Unscaled upper CI.

    Note on Edge Handling:
        The function uses an asymmetric approach to window generation:
        - Start: Truncated. The first window begins strictly at t[0]. No partial windows are
          created before the data start.
        - End: Adaptive. The window slides past the end of the data, using available
          points in the tail until the sample size drops below `min_size`.

    Statistical Note:
        Rolling window analysis involves multiple hypothesis tests on overlapping
        data segments. The results (p-values, slopes) from adjacent windows are
        highly autocorrelated and not statistically independent. The results should
        be interpreted as a descriptive time series of the trend strength/direction,
        not as a set of independent statistical findings.
    """
    # Input validation
    x_arr = np.asarray(x) if not isinstance(x, pd.DataFrame) else x
    t_arr = np.asarray(t)

    if len(x_arr) != len(t_arr):
        raise ValueError("x and t must have same length")

    is_datetime = _is_datetime_like(t_arr)

    # Parse window and step
    window_val = None
    step_val = None

    if is_datetime:
        t_series = pd.to_datetime(t_arr)

        if to_offset is None:
             raise ImportError("Could not import to_offset from pandas. Please ensure pandas is installed correctly.")

        # Try parsing as Timedelta first (fixed duration)
        try:
            window_val = pd.Timedelta(window)
        except ValueError:
            # Try parsing as DateOffset (variable duration like 'Y', 'M')
            try:
                window_val = to_offset(window)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid window '{window}' for datetime data. Use a pandas frequency string (e.g., '365D', '10YE').")

        if step is not None:
            try:
                step_val = pd.Timedelta(step)
            except ValueError:
                try:
                    step_val = to_offset(step)
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid step '{step}' for datetime data.")
        else:
            # Default step: half window
            # Division is tricky with DateOffsets.
            if isinstance(window_val, pd.Timedelta):
                step_val = window_val / 2
            else:
                # Fallback for Offsets: Just use the offset itself as step (non-overlapping)
                step_val = window_val

    else:
        t_series = pd.Series(t_arr)
        try:
            window_val = float(window)
        except (ValueError, TypeError):
             raise ValueError(f"Invalid window '{window}' for numeric data. Must be a number.")

        if step is not None:
             try:
                step_val = float(step)
             except (ValueError, TypeError):
                raise ValueError(f"Invalid step '{step}' for numeric data. Must be a number.")
        else:
            step_val = window_val / 2

    # Generate windows
    windows = _generate_windows(t_series, window_val, step_val, is_datetime, max_windows=max_windows)

    # Calculate trends for each window
    results = []
    t_index = pd.Index(t_series)
    use_bisect = bool(t_index.is_monotonic_increasing) and not t_index.hasnans

    for win_start, win_end in windows:
        if use_bisect:
            i_start = int(t_index.searchsorted(win_start, side='left'))
            i_end = int(t_index.searchsorted(win_end, side='left'))
            n_win = i_end - i_start
            if n_win < min_size:
                continue
            if isinstance(x, pd.DataFrame):
                x_window = x.iloc[i_start:i_end].reset_index(drop=True)
            else:
                x_window = x_arr[i_start:i_end]
            t_window = t_arr[i_start:i_end]
        else:
            # Select data in window [start, end)
            mask = np.asarray((t_series >= win_start) & (t_series < win_end), dtype=bool)
            n_win = int(mask.sum())
            if n_win < min_size:
                continue
            if isinstance(x, pd.DataFrame):
                x_window = x[mask].reset_index(drop=True)
            else:
                x_window = x_arr[mask]
            t_window = t_arr[mask]

        # Run trend test
        try:
            common_kwargs = {
                'alpha': alpha,
                'slope_scaling': slope_scaling,
                'x_unit': x_unit,
                'continuous_confidence': continuous_confidence,
                'large_dataset_mode': large_dataset_mode,
                'max_pairs': max_pairs,
                'random_state': random_state,
                **kwargs
            }

            if seasonal:
                # seasonal_trend_test uses min_size_per_season, not min_size
                if 'min_size' in common_kwargs:
                    del common_kwargs['min_size']

                result = seasonal_trend_test(
                    x=x_window,
                    t=t_window,
                    period=period,
                    season_type=season_type,
                    **common_kwargs
                )
            else:
                # trend_test accepts min_size. We set it to None because we already filtered
                # at the window level.
                common_kwargs['min_size'] = None
                result = trend_test(
                    x=x_window,
                    t=t_window,
                    **common_kwargs
                )

            # Calculate window center
            if is_datetime:
                # Midpoint calculation
                win_center = win_start + (win_end - win_start) / 2
            else:
                win_center = (win_start + win_end) / 2

            results.append({
                'window_start': win_start,
                'window_end': win_end,
                'window_center': win_center,
                'n_obs': n_win,
                'slope': result.slope,
                'lower_ci': result.lower_ci,
                'upper_ci': result.upper_ci,
                'p_value': result.p,
                'h': result.h,
                'classification': result.classification,
                'C': result.C,
                'Cd': result.Cd,
                'tau': result.Tau,
                's': result.s,
                'intercept': result.intercept,
                'slope_per_second': result.slope_per_second,
                'lower_ci_per_second': result.lower_ci_per_second,
                'upper_ci_per_second': result.upper_ci_per_second,
                'warnings': result.warnings
            })

        except Exception as e:
            warnings.warn(f"Failed to calculate trend for window {win_start} to {win_end}: {e}")
            continue

    if not results:
        return pd.DataFrame(columns=[
            'window_start', 'window_end', 'window_center', 'n_obs',
            'slope', 'lower_ci', 'upper_ci', 'p_value', 'h',
            'classification', 'C', 'Cd', 'tau', 's',
            'intercept', 'slope_per_second', 'lower_ci_per_second', 'upper_ci_per_second', 'warnings'
        ])

    return pd.DataFrame(results)


def _generate_windows(t_series, window_size, step_size, is_datetime, max_windows=10000):
    """Generate sliding window boundaries."""
    windows = []

    if len(t_series) == 0:
        return windows

    t_min = t_series.min()
    t_max = t_series.max()

    if not is_datetime:
         if step_size <= 0:
             raise ValueError("`step` must be positive.")
         # Reduce floating point accumulation error for numeric data
         # Calculate number of steps
         try:
            n_steps = int((t_max - t_min) / step_size) + 2 # Add buffer
         except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            n_steps = max_windows + 2

         for i in range(n_steps):
            current = t_min + i * step_size
            if current > t_max:
                break
            win_end = current + window_size
            windows.append((current, win_end))

            if len(windows) > max_windows:
                raise ValueError(
                    f"Generated more than {max_windows} windows. "
                    f"Increase `step` or `max_windows`."
                )
    else:
        # Datetime uses Timedelta/Offset which is robust to accumulation usually
        # because it operates on calendar logic or fixed integers (nanoseconds)
        current = t_min
        while current <= t_max:
            win_end = current + window_size
            windows.append((current, win_end))
            next_val = current + step_size
            if next_val <= current:
                raise ValueError("`step` must advance time forward.")
            current = next_val

            if len(windows) > max_windows:
                raise ValueError(
                    f"Generated more than {max_windows} windows. "
                    f"Increase `step` or `max_windows`."
                )

    return windows


def compare_periods(
    x, t, breakpoint, alpha=0.05,
    seasonal: bool = False,
    period: int = 12,
    season_type: str = 'month',
    **kwargs
):
    """
    Compare trends before and after a breakpoint.

    Args:
        x: Data vector
        t: Time vector
        breakpoint: Time value to split data (value in t where split occurs)
        alpha: Significance level
        seasonal: If True, uses `seasonal_trend_test` instead of `trend_test`.
        period: The seasonal period (e.g., 12 for monthly data). Used if seasonal=True.
        season_type: The type of seasonality (e.g., 'month'). Used if seasonal=True.
        **kwargs: Additional arguments for trend_test or seasonal_trend_test

    Returns:
        dict: Dictionary containing:
            - 'before': Result object for pre-breakpoint data
            - 'after': Result object for post-breakpoint data
            - 'slope_difference': slope_after - slope_before
            - 'ci_overlap': Boolean, True if the two (1 - alpha) intervals overlap.
              Non-overlap is a very conservative criterion (for two 95%
              intervals it corresponds to a significance level of about 0.006).
            - 'z_difference', 'p_value_difference': Approximate two-sided test of
              equal slopes, z = (b2 - b1) / sqrt(se1^2 + se2^2), with each
              standard error derived from its interval width
              (se = (upper - lower) / (2 z_{1-alpha/2})).  This assumes the two
              periods are independent (they share no data, but serial
              correlation across the breakpoint is ignored).
            - 'significant_change': Boolean, ``p_value_difference < alpha``.
            - 'breakpoint': The breakpoint used
    """
    t_arr = np.asarray(t)
    x_arr = np.asarray(x) if not isinstance(x, pd.DataFrame) else x

    if isinstance(x, pd.DataFrame):
        mask_before = t_arr < breakpoint
        x_before = x[mask_before].reset_index(drop=True)
        x_after = x[~mask_before].reset_index(drop=True)
    else:
        mask_before = t_arr < breakpoint
        x_before = x_arr[mask_before]
        x_after = x_arr[~mask_before]

    t_before = t_arr[mask_before]
    t_after = t_arr[~mask_before]

    if len(t_before) < 3 or len(t_after) < 3:
         warnings.warn("Insufficient data in one or both periods for comparison.", UserWarning)

    common_kwargs = {'alpha': alpha, **kwargs}

    if seasonal:
        if 'min_size' in common_kwargs:
             del common_kwargs['min_size']

        result_before = seasonal_trend_test(
            x=x_before, t=t_before, period=period, season_type=season_type, **common_kwargs
        )
        result_after = seasonal_trend_test(
            x=x_after, t=t_after, period=period, season_type=season_type, **common_kwargs
        )
    else:
        result_before = trend_test(x=x_before, t=t_before, **common_kwargs)
        result_after = trend_test(x=x_after, t=t_after, **common_kwargs)

    slope_diff = result_after.slope - result_before.slope

    from scipy.stats import norm
    cis = [result_before.lower_ci, result_before.upper_ci, result_after.lower_ci, result_after.upper_ci]
    if any(pd.isna(v) for v in cis):
        ci_overlap = np.nan
        z_diff, p_diff = np.nan, np.nan
        significant_change = False
    else:
        no_overlap = (result_before.lower_ci > result_after.upper_ci) or \
                     (result_after.lower_ci > result_before.upper_ci)
        ci_overlap = not no_overlap
        zc = norm.ppf(1 - alpha / 2)
        se_b = (result_before.upper_ci - result_before.lower_ci) / (2 * zc)
        se_a = (result_after.upper_ci - result_after.lower_ci) / (2 * zc)
        se = np.sqrt(se_b ** 2 + se_a ** 2)
        if se > 0:
            z_diff = slope_diff / se
            p_diff = float(2 * norm.sf(abs(z_diff)))
        else:
            z_diff, p_diff = np.nan, np.nan
        significant_change = bool(p_diff < alpha) if not np.isnan(p_diff) else bool(no_overlap)

    return {
        'before': result_before,
        'after': result_after,
        'slope_difference': slope_diff,
        'ci_overlap': ci_overlap,
        'z_difference': z_diff,
        'p_value_difference': p_diff,
        'significant_change': significant_change,
        'breakpoint': breakpoint
    }
