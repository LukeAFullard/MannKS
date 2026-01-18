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


def _to_numeric_time(t):
    """
    Converts a time vector to numeric (Unix timestamp) if it is datetime-like.
    Returns float array.
    """
    t_arr = np.asarray(t)
    if t_arr.ndim == 0:
        t_arr = t_arr.reshape(1)

    if _is_datetime_like(t_arr):
        # Handle numpy datetime64
        if np.issubdtype(t_arr.dtype, np.datetime64):
            return t_arr.astype('datetime64[s]').astype(float)
        # Handle pandas Series or Index
        if isinstance(t, (pd.Series, pd.Index)):
             return t.astype('int64').to_numpy() / 1e9 # ns to s
        # Handle object array of datetimes
        try:
             return np.array([x.timestamp() for x in t_arr], dtype=float)
        except AttributeError:
             pass # Fall through to default

    # Try forcing float conversion (handles int/float inputs)
    try:
        return t_arr.astype(float)
    except (ValueError, TypeError):
        # Fallback: try pandas conversion then numeric
        try:
            converted = pd.to_datetime(t_arr)
            if isinstance(converted, pd.Timestamp):
                return converted.timestamp()
            return converted.astype('int64').to_numpy() / 1e9
        except Exception:
             raise ValueError("Could not convert time vector `t` to numeric values.")


def _get_dt_prop(dt, prop):
    return getattr(dt.dt, prop) if isinstance(dt, pd.Series) else getattr(dt, prop)

# Define season specifications at module level
# Structure: season_type -> (expected_period, extraction_function)
_SEASON_SPECS = {
    'year': (1, lambda dt: _get_dt_prop(dt, 'year')),
    'month': (12, lambda dt: _get_dt_prop(dt, 'month')),
    'day_of_week': (7, lambda dt: _get_dt_prop(dt, 'dayofweek')),
    'quarter': (4, lambda dt: _get_dt_prop(dt, 'quarter')),
    'hour': (24, lambda dt: _get_dt_prop(dt, 'hour')),
    'week_of_year': ([52, 53], lambda dt: _get_dt_prop(dt, 'isocalendar')().week),
    'biweekly': ([26, 27], lambda dt: (_get_dt_prop(dt, 'isocalendar')().week - 1) // 2),
    'day_of_year': (None, lambda dt: _get_dt_prop(dt, 'dayofyear')),
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
        # The cycle is the day, identified by the Unix timestamp of the day's start
        # Convert to int64 (nanoseconds) and then to float seconds
        return (dt_accessor.normalize().astype(np.int64) / 10**9)

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


def _get_theoretical_midpoint(datetime_series):
    """
    Calculates the theoretical midpoint of a time period for a series of datetimes.
    This is used for the 'middle_lwp' aggregation method to replicate R's logic.
    """
    if not isinstance(datetime_series, pd.Series):
        datetime_series = pd.Series(datetime_series)

    # All dates in the group should be in the same period (e.g., month, week)
    first_date = datetime_series.iloc[0]

    # Determine the period (month, week, etc.)
    # Infer the period from the first date to determine start/end bounds.
    # Note: Explicit period handling would be required for non-standard periods.
    if first_date.day == 1 and first_date.month != 2:
        # Likely start of a month, find end of month
        start_of_period = first_date.replace(day=1)
        end_of_period = (start_of_period + pd.DateOffset(months=1)) - pd.Timedelta(nanoseconds=1)
    else:
        # Assume weekly or other, and just use the range of the data in the group
        start_of_period = datetime_series.min()
        end_of_period = datetime_series.max()

    midpoint = start_of_period + (end_of_period - start_of_period) / 2
    return midpoint

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
