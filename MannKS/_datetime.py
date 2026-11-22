import datetime
import numpy as np
import pandas as pd


def _is_datetime_like(x):
    """Checks if an array or scalar is datetime-like."""
    if isinstance(x, (pd.Timestamp, datetime.date, np.datetime64)):
        return True
    if pd.api.types.is_datetime64_any_dtype(x):
        return True
    if hasattr(x, 'dtype') and isinstance(x.dtype, pd.api.extensions.ExtensionDtype):
        return False
    if not hasattr(x, 'dtype'):
        x = np.asarray(x)

    if np.issubdtype(x.dtype, np.datetime64):
        return True
    if x.dtype == 'O':
        if x.ndim == 0:
            return hasattr(x.item(), 'year')
        for item in x.ravel():
            if (
                item is not None
                and item is not pd.NA
                and not (isinstance(item, float) and np.isnan(item))
            ):
                return hasattr(item, 'year')
    return False


def _extract_timezone(t):
    """
    Extract a common timezone from a datetime-like input, or None if tz-naive.

    If mixed timezones are present across elements, returns
    ``datetime.timezone.utc`` to match the UTC-instant conversion used
    internally by ``_datetime_to_seconds``.
    """
    if isinstance(t, pd.DatetimeIndex):
        return t.tz
    if isinstance(t, pd.Series) and isinstance(t.dtype, pd.DatetimeTZDtype):
        return t.dtype.tz
    if isinstance(t, (pd.Timestamp, datetime.datetime)):
        return t.tzinfo
    try:
        return pd.DatetimeIndex(t).tz
    except (TypeError, ValueError):
        arr = np.asarray(t, dtype=object)
        for item in arr.ravel():
            if getattr(item, 'tzinfo', None) is not None:
                return datetime.timezone.utc
        return None


def _datetime_to_seconds(t):
    """
    Seconds since 1970-01-01 (float) for any datetime-like array.

    numpy ``datetime64`` and naive datetimes are taken as UTC wall time;
    timezone-aware values are converted to their UTC instant.  Sub-second
    precision (ms/us/ns) is preserved without overflowing dates outside
    1677-2262.  Missing values (NaT/None) become NaN.
    """
    if isinstance(t, pd.DatetimeIndex) and t.tz is not None:
        arr = t.tz_convert('UTC').tz_localize(None).to_numpy()
    elif isinstance(t, pd.Series) and isinstance(t.dtype, pd.DatetimeTZDtype):
        arr = t.dt.tz_convert('UTC').dt.tz_localize(None).to_numpy()
    else:
        arr = np.asarray(t)
        if not np.issubdtype(arr.dtype, np.datetime64):
            idx = pd.to_datetime(pd.Index(arr.ravel(), dtype=object), utc=True)
            arr = idx.tz_convert(None).to_numpy().reshape(arr.shape)

    unit, step = np.datetime_data(arr.dtype)
    if unit in ('ns', 'us', 'ms', 's') and step == 1:
        target, target_unit = arr, unit
    elif unit in ('Y', 'M', 'W', 'D', 'h', 'm'):
        target, target_unit = arr.astype('datetime64[s]'), 's'
    else:
        target = None
        target_unit = 's'
        for u in ('ns', 'us', 's'):
            try:
                target = arr.astype(f'datetime64[{u}]')
                target_unit = u
                break
            except OverflowError:
                continue
        if target is None:
            target = arr.astype('datetime64[s]')
            target_unit = 's'

    missing = np.isnat(target)
    epoch = np.datetime64('1970-01-01T00:00:00', target_unit)
    out = np.asarray((target - epoch) / np.timedelta64(1, 's'), dtype=float)
    out[missing] = np.nan
    return out


def _seconds_to_datetime(values, tz=None):
    """
    Convert float seconds since 1970-01-01 to pandas DatetimeIndex (or scalar
    Timestamp) without overflowing pandas 2.x's nanosecond bounds (1677-2262).
    """
    is_scalar = np.ndim(values) == 0
    if is_scalar:
        if pd.isna(values) or not np.isfinite(float(values)):
            return pd.NaT
        try:
            ts = pd.to_datetime(float(values), unit='s')
        except pd.errors.OutOfBoundsDatetime:
            us = np.datetime64('1970-01-01T00:00:00', 'us') + np.timedelta64(
                int(round(float(values) * 1e6)), 'us'
            )
            ts = pd.Timestamp(us)
        if tz is not None:
            ts = ts.tz_localize('UTC').tz_convert(tz)
        return ts

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        idx = pd.DatetimeIndex([])
    else:
        try:
            idx = pd.to_datetime(arr, unit='s')
        except pd.errors.OutOfBoundsDatetime:
            epoch = np.datetime64('1970-01-01T00:00:00', 'us')
            dt_arr = np.full(arr.shape, np.datetime64('NaT', 'us'), dtype='datetime64[us]')
            finite = np.isfinite(arr)
            if np.any(finite):
                dt_arr[finite] = epoch + np.round(arr[finite] * 1e6).astype('int64').astype('timedelta64[us]')
            idx = pd.DatetimeIndex(dt_arr)
    if tz is not None:
        idx = idx.tz_localize('UTC').tz_convert(tz)
    return idx


def _raise_if_string_times(arr):
    if arr.dtype.kind in 'US' or (arr.dtype == 'O' and arr.size and isinstance(arr.ravel()[0], str)):
        raise TypeError("The time vector `t` contains strings. Convert it first, "
                        "e.g. with pd.to_datetime(t), or pass numbers.")


def _to_numeric_time(t):
    """
    Converts a time vector to numeric (seconds since 1970-01-01 for
    datetimes, see ``_datetime_to_seconds``).  Returns a float array.
    """
    if _is_datetime_like(t):
        out = _datetime_to_seconds(t)
        return out.reshape(1) if out.ndim == 0 else out
    if isinstance(t, (pd.Series, pd.Index)) and pd.api.types.is_numeric_dtype(t):
        out = t.to_numpy(dtype=float, na_value=np.nan)
        return out.reshape(1) if out.ndim == 0 else out

    t_arr = np.asarray(t)
    if t_arr.ndim == 0:
        t_arr = t_arr.reshape(1)
    try:
        return t_arr.astype(float)
    except (ValueError, TypeError):
        # Strings (e.g. breakpoint dates given as text)
        try:
            return _datetime_to_seconds(pd.to_datetime(t_arr.ravel()).to_numpy()).reshape(t_arr.shape)
        except Exception:
            raise ValueError("Could not convert time vector `t` to numeric values.")


def _get_dt_prop(dt, prop):
    return getattr(dt.dt, prop) if isinstance(dt, pd.Series) else getattr(dt, prop)

# Define season specifications at module level
# Structure: season_type -> (expected_period, extraction_function)
_SEASON_SPECS = {
    'year': (1, lambda dt: _get_dt_prop(dt, 'year')),
    'month': (12, lambda dt: _get_dt_prop(dt, 'month')),
    'day_of_week': (7, lambda dt: _get_dt_prop(dt, 'day_of_week')),
    'quarter': (4, lambda dt: _get_dt_prop(dt, 'quarter')),
    'hour': (24, lambda dt: _get_dt_prop(dt, 'hour')),
    'week_of_year': ([52, 53], lambda dt: _get_dt_prop(dt, 'isocalendar')().week),
    'biweekly': ([26, 27], lambda dt: (_get_dt_prop(dt, 'isocalendar')().week - 1) // 2),
    'day_of_year': (None, lambda dt: _get_dt_prop(dt, 'day_of_year')),
    'minute': (60, lambda dt: _get_dt_prop(dt, 'minute')),
    'second': (60, lambda dt: _get_dt_prop(dt, 'second')),
}

def _infer_period(season_type):
    """
    Attempts to infer the period from the season_type.
    Returns the period if it is a fixed integer, otherwise None.
    """
    if season_type not in _SEASON_SPECS:
        return None

    expected_period, _ = _SEASON_SPECS[season_type]

    # If it's a list (e.g. week_of_year), we can't infer a single value unambiguously
    if isinstance(expected_period, list):
        return None

    return expected_period

def _get_season_func(season_type, period=None):
    """
    Returns a function to extract seasonal data based on the season_type,
    and validates the period if provided.
    """
    if season_type not in _SEASON_SPECS:
        raise ValueError(f"Unknown season_type: '{season_type}'. Must be one of {list(_SEASON_SPECS.keys())}")

    expected_period, season_func = _SEASON_SPECS[season_type]

    if period is not None and expected_period is not None:
        if isinstance(expected_period, list):
            if period not in expected_period:
                raise ValueError(f"For season_type='{season_type}', period must be one of {expected_period}.")
        elif period != expected_period:
            raise ValueError(f"For season_type='{season_type}', period must be {expected_period}.")

    return season_func

def _get_cycle_identifier(dt_series, season_type):
    """
    Returns a numeric series that uniquely identifies the larger time cycle
    for each timestamp, used for aggregation.
    """
    dt_accessor = dt_series.dt if isinstance(dt_series, pd.Series) else dt_series

    if season_type in ['month', 'quarter', 'year', 'day_of_year', 'week_of_year', 'biweekly']:
        # The cycle is the year
        return dt_accessor.year.to_numpy()

    elif season_type == 'day_of_week':
        # The cycle is the week, identified by year and week number
        iso_cal = dt_accessor.isocalendar()
        return (iso_cal.year * 100 + iso_cal.week).to_numpy()

    elif season_type in ['hour', 'minute', 'second']:
        # The cycle is the day, identified by the Unix timestamp of the day's
        # start (independent of the datetime resolution, ns or us).
        return _datetime_to_seconds(np.asarray(dt_accessor.normalize()))

    else:
        # Default to year if the concept of a cycle is not obvious
        return dt_accessor.year.to_numpy()


def _get_time_ranks(t_values, cycles):
    """Convert timestamps to cycle-based ranks matching R implementation."""
    # Create unique cycle identifiers and sort them to ensure rank order
    unique_cycles = np.unique(cycles)
    ranks = np.zeros_like(t_values, dtype=float)

    # Assign sequential ranks to each cycle
    for i, cycle in enumerate(unique_cycles, start=1):
        mask = cycles == cycle
        ranks[mask] = i

    return ranks


def _get_theoretical_midpoint(datetime_series, period=None):
    """
    Mid-point of the time increment for a group of datetimes.

    With ``period`` ('Y', 'Q', 'M', ...) the LWP-TRENDS increment mid-point is
    returned (see ``_helpers._lwp_period_midpoint``); otherwise the mid-point
    of the group's time range.
    """
    from ._helpers import _lwp_period_midpoint
    if not isinstance(datetime_series, pd.Series):
        datetime_series = pd.Series(datetime_series)
    times = pd.to_datetime(datetime_series)
    if period is not None:
        return _lwp_period_midpoint(times.iloc[0], period)
    return times.min() + (times.max() - times.min()) / 2


def _get_agg_func(agg_period: str):
    """
    Returns a function to extract aggregation period identifiers from a
    datetime series.
    """
    def get_dt_prop(dt, prop):
        return getattr(dt.dt, prop) if isinstance(dt, pd.Series) else getattr(dt, prop)

    agg_map = {
        'year': lambda dt: get_dt_prop(dt, 'year'),
        'month': lambda dt: get_dt_prop(dt, 'year') * 100 + get_dt_prop(dt, 'month'),
        'day': lambda dt: get_dt_prop(dt, 'date'),
        'week': lambda dt: get_dt_prop(dt, 'isocalendar')().year * 100 + get_dt_prop(dt, 'isocalendar')().week,
        'quarter': lambda dt: get_dt_prop(dt, 'year') * 10 + get_dt_prop(dt, 'quarter'),
    }
    agg_period_lower = agg_period.lower()

    if agg_period_lower not in agg_map:
        raise ValueError(f"Unknown agg_period: '{agg_period}'. Must be one of {list(agg_map.keys())}")

    return agg_map[agg_period_lower]
