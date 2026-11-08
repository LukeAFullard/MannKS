import warnings
import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame
from ._datetime import _is_datetime_like, _datetime_to_seconds, _raise_if_string_times


def _preprocessing(x):
    x = np.asarray(x)

    # Convert datetime objects to numeric timestamps if necessary
    if _is_datetime_like(x):
        x = _datetime_to_seconds(x)
    else:
        _raise_if_string_times(x)

    x = x.astype(float)

    if x.ndim == 2:
        (n, c) = x.shape
        if c == 1:
            x = x.flatten()
    return x, (1 if x.ndim == 1 else x.shape[1])

def _missing_values_analysis(x, method='skip'):
    if method.lower() == 'skip':
        if x.ndim == 1:
            x = x[~np.isnan(x)]
        else:
            x = x[~np.isnan(x).any(axis=1)]
    return x, len(x)


def _aggregate_censored_median(group, is_datetime):
    """
    Computes a robust median for a group of observations which may contain
    censored data, following the LWP-TRENDS R script logic.

    .. note::
        The logic for determining if the aggregated median is censored
        (i.e., `median_val <= max_censored_value`) is a heuristic method.
        It is designed to replicate the behavior of the LWP-TRENDS R script
        and may not be statistically robust in all scenarios.

        The censoring *type* of a censored median is the most common type
        among the censored observations.  LWP-TRENDS instead takes the most
        common type among all observations in the increment, which can label
        a censored median as 'not' censored; values and censored flags agree.
    """
    n = len(group)
    if n == 0:
        return pd.DataFrame({
            'value': [], 'censored': [], 'cen_type': [],
            't_original': [], 't': []
        })

    # Compute median value
    median_val = group['value'].median()

    # Determine if median is censored (R logic)
    if not group['censored'].any():
        is_censored = False
        cen_type = 'not'
    else:
        # Get maximum censored value
        max_censored = group.loc[group['censored'], 'value'].max()
        is_censored = median_val <= max_censored

        if is_censored:
            # Safely get the most common censor type
            cen_type_mode = group.loc[group['censored'], 'cen_type'].mode()
            if len(cen_type_mode) == 0:
                # All censored values are NaN, default to 'not'
                cen_type = 'not'
                is_censored = False
            else:
                cen_type = cen_type_mode.iloc[0]
        else:
            cen_type = 'not'

    row_data = {
        'value': median_val,
        'censored': is_censored,
        'cen_type': cen_type,
    }

    # Always aggregate time using the median of the original timestamps
    row_data['t_original'] = group['t_original'].median() if is_datetime else np.median(group['t_original'])
    row_data['t'] = np.median(group['t'])

    return pd.DataFrame([row_data])


def _prepare_data(x, t, hicensor):
    """
    Internal helper to prepare and validate data for trend tests.
    """
    if isinstance(x, pd.DataFrame) and all(col in x.columns for col in ['value', 'censored', 'cen_type']):
        data = x.copy()
    elif hasattr(x, '__iter__') and any(isinstance(i, str) for i in x):
        raise TypeError("Input data `x` contains strings. Please pre-process it with `prepare_censored_data` first.")
    else:
        x_proc, _ = _preprocessing(x)
        data = pd.DataFrame({
            'value': x_proc,
            'censored': np.zeros(len(x_proc), dtype=bool),
            'cen_type': np.full(len(x_proc), 'not', dtype=object)
        })

    t_raw = np.asarray(t)
    is_datetime = _is_datetime_like(t_raw)
    t_numeric, _ = _preprocessing(t_raw)
    data['t_original'] = t_raw
    data['t'] = t_numeric

    # Handle missing values.  NaN values are dropped silently (missing data);
    # infinite values and missing times are dropped with a warning.
    vals = data['value'].to_numpy(dtype=float)
    times = data['t'].to_numpy(dtype=float)
    bad = np.isinf(vals) | (np.isnan(times) & ~np.isnan(vals))
    if bad.any():
        warnings.warn(f"Removed {int(bad.sum())} observation(s) with an infinite value or a "
                      f"missing time.", UserWarning)
    mask = ~np.isnan(vals) & ~bad
    data_filtered = data[mask].copy()

    # Apply HiCensor rule if requested
    if hicensor:
        if isinstance(hicensor, bool):
            if 'lt' in data_filtered['cen_type'].values:
                max_lt_censor = data_filtered.loc[
                    data_filtered['cen_type'] == 'lt', 'value'].max()
            else:
                max_lt_censor = None # No left-censored data, so do nothing
        elif isinstance(hicensor, (int, float)):
            natural_max = data_filtered.loc[
                data_filtered['cen_type'] == 'lt', 'value'].max()
            max_lt_censor = min(natural_max, hicensor) if pd.notna(natural_max) else hicensor
        else:
            raise ValueError("hicensor must be bool or numeric")

        if max_lt_censor is not None:
            hi_censor_mask = data_filtered['value'] < max_lt_censor
            data_filtered.loc[hi_censor_mask, 'censored'] = True
            data_filtered.loc[hi_censor_mask, 'cen_type'] = 'lt'
            data_filtered.loc[hi_censor_mask, 'value'] = max_lt_censor

    return data_filtered, is_datetime


_INCREMENTS_PER_YEAR = {'Y': 1, 'A': 1, 'Q': 4, 'M': 12}


def _lwp_period_midpoint(ts, period):
    """
    Mid-point of the time increment containing timestamp ``ts``.

    For annual, quarterly and monthly increments this reproduces LWP-TRENDS
    (``MidTimeIncrList``): with k increments per year the i-th increment's
    mid-point is ``1 Jan + (365/k/2 + i*365/k - 1)`` days.  For other periods
    the mid-point of the calendar period is used; with ``period=None`` the
    caller should use the group's time range instead.
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    key = str(period).upper()[0] if period else None
    if key in _INCREMENTS_PER_YEAR:
        k = _INCREMENTS_PER_YEAR[key]
        i = 0 if k == 1 else (ts.quarter - 1 if k == 4 else ts.month - 1)
        offset = 365.0 / k / 2.0 + i * 365.0 / k - 1.0
        return pd.Timestamp(year=ts.year, month=1, day=1) + pd.Timedelta(days=offset)
    p = ts.to_period(period)
    return p.start_time + (p.end_time - p.start_time) / 2


def _closest_to_midpoint(group, period):
    """Row label of the observation closest to the increment mid-point (first on ties)."""
    times = pd.to_datetime(group['t_original'])
    if getattr(times.dt, 'tz', None) is not None:
        times = times.dt.tz_convert(None)
    if period is None:
        mid = times.min() + (times.max() - times.min()) / 2
    else:
        mid = _lwp_period_midpoint(times.iloc[0], period)
    return (times - mid).abs().idxmin()


def _value_for_time_increment(df: DataFrame, group_key: pd.Series, period: str) -> DataFrame:
    """
    Aggregates data to one observation per time increment using the LWP method.

    Replicates ``ValueForTimeIncr(UseMidObs=TRUE)`` from LWP-TRENDS: for each
    time increment the observation closest in time to the increment's
    mid-point is kept (the earliest one on ties).  ``df`` should be sorted by
    time.

    Args:
        df (pd.DataFrame): The input data frame, must contain 't_original'.
        group_key (pd.Series): Grouping of rows into time increments.
        period (str): Pandas frequency of the increment ('Y', 'Q', 'M', 'W',
            'D', ...), or None to use each group's time range.

    Returns:
        pd.DataFrame: One row per time increment.
    """
    keep = [g.index[0] if len(g) == 1 else _closest_to_midpoint(g, period)
            for _, g in df.groupby(group_key, sort=False)]
    return df.loc[keep].reset_index(drop=True)


def _aggregate_by_group(group, agg_method, is_datetime, period=None):
    """
    Aggregates a group of data points using the specified method.
    """
    if len(group) <= 1:
        return group

    if agg_method == 'median':
        if group['censored'].any():
            import warnings
            warnings.warn(
                "The 'median' aggregation method uses a simple heuristic for censored data, "
                "which may not be statistically robust. Consider using 'robust_median' for "
                "more accurate censored data aggregation.", UserWarning)
        median_val = group['value'].median()
        is_censored = median_val <= group[group['censored']]['value'].max() if group['censored'].any() else False

        new_row = {
            'value': median_val,
            't_original': group['t_original'].median() if is_datetime else np.median(group['t_original']),
            't': np.median(group['t']),
            'censored': is_censored,
            'cen_type': group.loc[group['censored'], 'cen_type'].mode()[0] if is_censored else 'not'
        }
        return pd.DataFrame([new_row])
    elif agg_method == 'robust_median':
        return _aggregate_censored_median(group, is_datetime)
    elif agg_method == 'middle':
        t_numeric_group = group['t'].to_numpy()
        closest_idx = np.argmin(np.abs(t_numeric_group - np.mean(t_numeric_group)))
        return group.iloc[[closest_idx]]
    elif agg_method == 'middle_lwp':
        if not is_datetime:
            # For numeric time, 'middle_lwp' is equivalent to 'middle'
            t_numeric_group = group['t'].to_numpy()
            closest_idx = np.argmin(np.abs(t_numeric_group - np.mean(t_numeric_group)))
            return group.iloc[[closest_idx]]
        else:
            # Observation closest to the mid-point of the time increment
            # (LWP-TRENDS rule) when the increment is known; otherwise the
            # mid-point of the group's time range.
            return group.loc[[_closest_to_midpoint(group, period)]]
    return group


def _get_slope_scaling_factor(unit: str) -> float:
    """Returns the scaling factor to convert units/sec to units/[unit]."""
    if not isinstance(unit, str):
        raise TypeError("Time unit for scaling must be a string.")

    unit = unit.lower()
    if unit in ['year', 'years', 'yr', 'y', 'annum']:
        return 365.25 * 24 * 60 * 60
    elif unit in ['month', 'months', 'mon']:
        return 30.44 * 24 * 60 * 60 # Average month length
    elif unit in ['week', 'weeks', 'wk', 'w']:
        return 7 * 24 * 60 * 60
    elif unit in ['day', 'days', 'd']:
        return 24 * 60 * 60
    elif unit in ['hour', 'hours', 'hr', 'h']:
        return 60 * 60
    elif unit in ['minute', 'minutes', 'min', 'm']:
        return 60
    elif unit in ['second', 'seconds', 'sec', 's']:
        return 1.0
    else:
        raise ValueError(
            f"Invalid time unit for slope scaling: '{unit}'. "
            "Must be one of: year, month, week, day, hour, minute, second."
        )
