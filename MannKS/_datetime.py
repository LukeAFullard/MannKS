import numpy as np
import pandas as pd


def _is_datetime_like(x):
    """Checks if an array is datetime-like."""
    if not hasattr(x, 'dtype'):
        x = np.asarray(x)

    if np.issubdtype(x.dtype, np.datetime64):
        return True
    if x.dtype == 'O':
        if x.ndim == 0:
            return hasattr(x.item(), 'year')
        elif len(x) > 0:
            return hasattr(x[0], 'year')
    return False


def _datetime_to_seconds(t):
    """
    Seconds since 1970-01-01 (float) for any datetime-like array.

    numpy ``datetime64`` and naive datetimes are taken as UTC wall time;
    timezone-aware values are converted to their UTC instant.  Missing
    values (NaT/None) become NaN.
    """
    arr = np.asarray(t)
    if not np.issubdtype(arr.dtype, np.datetime64):
        idx = pd.to_datetime(pd.Index(arr.ravel(), dtype=object), utc=True)
        arr = idx.tz_convert(None).to_numpy().reshape(arr.shape)
    missing = np.isnat(arr)
    out = arr.astype('datetime64[s]').astype(float)
    out[missing] = np.nan
    return out


def _raise_if_string_times(arr):
    if arr.dtype.kind in 'US' or (arr.dtype == 'O' and arr.size and isinstance(arr.ravel()[0], str)):
        raise TypeError("The time vector `t` contains strings. Convert it first, "
                        "e.g. with pd.to_datetime(t), or pass numbers.")


def _to_numeric_time(t):
    """
    Converts a time vector to numeric (seconds since 1970-01-01 for
    datetimes, see ``_datetime_to_seconds``).  Returns a float array.
    """
    t_arr = np.asarray(t)
    if t_arr.ndim == 0:
        t_arr = t_arr.reshape(1)
    if _is_datetime_like(t_arr):
        return _datetime_to_seconds(t_arr)
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
