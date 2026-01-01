import numpy as np
import pandas as pd


def _is_datetime_like(x):
    """Checks if an array is datetime-like."""
    return np.issubdtype(x.dtype, np.datetime64) or \
           (x.dtype == 'O' and len(x) > 0 and hasattr(x[0], 'year'))

def _get_season_func(season_type, period):
    """
    Returns a function to extract seasonal data based on the season_type,
    and validates the period.
    """
    def get_dt_prop(dt, prop):
        return getattr(dt.dt, prop) if isinstance(dt, pd.Series) else getattr(dt, prop)

    season_map = {
        'year': (1, lambda dt: get_dt_prop(dt, 'year')),
        'month': (12, lambda dt: get_dt_prop(dt, 'month')),
        'day_of_week': (7, lambda dt: get_dt_prop(dt, 'dayofweek')),
        'quarter': (4, lambda dt: get_dt_prop(dt, 'quarter')),
        'hour': (24, lambda dt: get_dt_prop(dt, 'hour')),
        'week_of_year': ([52, 53], lambda dt: get_dt_prop(dt, 'isocalendar')().week),
        'biweekly': ([26, 27], lambda dt: (get_dt_prop(dt, 'isocalendar')().week - 1) // 2),
        'day_of_year': (None, lambda dt: get_dt_prop(dt, 'dayofyear')),
        'minute': (60, lambda dt: get_dt_prop(dt, 'minute')),
        'second': (60, lambda dt: get_dt_prop(dt, 'second')),
    }
    if season_type not in season_map:
        raise ValueError(f"Unknown season_type: '{season_type}'. Must be one of {list(season_map.keys())}")

    expected_period, season_func = season_map[season_type]

    if expected_period is not None:
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
    # This is a simplification; a more robust version would take period as an arg
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
