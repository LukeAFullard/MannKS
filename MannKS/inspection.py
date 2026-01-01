
"""
This module provides the inspect_trend_data function, inspired by the
LWP-TRENDS R script, to assess data availability over a specified trend period
and determine the most suitable time increment for analysis.
"""
import pandas as pd
import numpy as np
from collections import namedtuple
from .plotting import plot_inspection_data

from typing import Optional, Dict, Union

def inspect_trend_data(
    data: pd.DataFrame,
    trend_period: Optional[int] = None,
    end_year: Optional[int] = None,
    prop_year_tol: float = 0.9,
    prop_incr_tol: float = 0.9,
    return_summary: bool = False,
    value_col: str = 'value',
    time_col: str = 't',
    custom_increments: Optional[Dict[str, int]] = None,
    plot: bool = False,
    plot_path: Optional[str] = None
) -> Union[pd.DataFrame, namedtuple]:
    """
    Inspects data availability over a trend period and determines the best time
    increment for trend analysis.

    Args:
        data (pd.DataFrame): Input DataFrame containing at least a time column
                             and a value column.
        trend_period (int, optional): The number of years in the trend period.
                                      Defaults to the full range of data.
        end_year (int, optional): The last year of the trend period.
                                  Defaults to the last year in the data.
        prop_year_tol (float): The minimum proportion of years in the trend
                               period that must have at least one observation.
        prop_incr_tol (float): The minimum proportion of time increments within
                               the trend period that must have at least one
                               observation.
        return_summary (bool): If True, returns a namedtuple containing the modified
                               DataFrame and a summary of data availability.
        value_col (str): The name of the column containing data values.
        time_col (str): The name of the column containing datetime objects.
        custom_increments (dict, optional): A dictionary of custom time increments.
                                            Keys are the increment names (e.g.,
                                            'weekly'), and values are the number
                                            of increments in a year. If not
                                            provided, a default set of increments
                                            is used. Supported increments are:
                                            'annually', 'bi-annually', 'quarterly',
                                            'bi-monthly', 'monthly', 'weekly', 'daily'.
        plot (bool): If True, generates and saves a set of inspection plots.
                     Requires `plot_path` to be set.
        plot_path (str, optional): The file path to save the inspection plots.

    Returns:
        pd.DataFrame or namedtuple:
        - The filtered DataFrame with a new 'time_increment' column. If no
          suitable increment is found, this column will be filled with 'none'.
        - If `return_summary` is True, a namedtuple `InspectionResult` with
          `data` and `summary` attributes is returned.
    """
    InspectionResult = namedtuple('InspectionResult', ['data', 'summary'])

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame.")

    if time_col not in data.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame.")

    if value_col not in data.columns:
        raise ValueError(f"Value column '{value_col}' not found in DataFrame.")

    df = data.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # --- 1. Filter by Trend Period ---
    if end_year is None:
        end_year = df[time_col].dt.year.max()

    if trend_period is None:
        trend_period = end_year - df[time_col].dt.year.min() + 1

    start_year = end_year - trend_period + 1
    start_date = pd.to_datetime(f'{start_year}-01-01')
    end_date = pd.to_datetime(f'{end_year}-12-31')

    df_filtered = df[(df[time_col] >= start_date) & (df[time_col] <= end_date)].copy()
    df_filtered.dropna(subset=[value_col], inplace=True)

    if df_filtered.empty:
        df_filtered['time_increment'] = 'none'
        if return_summary:
            return InspectionResult(data=df_filtered, summary=pd.DataFrame())
        return df_filtered

    # --- 2. Define Time Increments and Helper Columns ---
    if custom_increments:
        time_increments = custom_increments
    else:
        time_increments = {
            'daily': 365,
            'weekly': 52,
            'monthly': 12,
            'bi-monthly': 6,
            'quarterly': 4,
            'bi-annually': 2,
            'annually': 1
        }

    df_filtered['year'] = df_filtered[time_col].dt.year
    df_filtered['month'] = df_filtered[time_col].dt.month
    df_filtered['quarter'] = df_filtered[time_col].dt.quarter
    df_filtered['bi-month'] = (df_filtered['month'] - 1) // 2 + 1
    df_filtered['bi-annual'] = (df_filtered['month'] - 1) // 6 + 1
    df_filtered['week'] = df_filtered[time_col].dt.isocalendar().week
    df_filtered['day'] = df_filtered[time_col].dt.dayofyear


    increment_map = {
        'monthly': 'month',
        'bi-monthly': 'bi-month',
        'quarterly': 'quarter',
        'bi-annually': 'bi-annual',
        'annually': 'year',
        'weekly': 'week',
        'daily': 'day'
    }

    # --- 3. Data Availability Summary ---
    availability_summary = []
    best_time_incr = 'none'

    # Sort increments by frequency (descending) to find the best one first
    sorted_increments = sorted(time_increments.items(), key=lambda item: item[1], reverse=True)

    for name, num_increments in sorted_increments:
        if name not in increment_map:
            raise ValueError(f"Custom increment '{name}' is not supported. Supported increments are: {list(increment_map.keys())}")
        col = increment_map[name]

        n_years_with_data = df_filtered['year'].nunique()
        prop_year = n_years_with_data / trend_period

        n_incr_year = df_filtered.groupby(['year', col]).ngroups

        if name == 'annually':
            prop_incr_year = prop_year
        else:
            total_possible_increments = trend_period * num_increments
            prop_incr_year = n_incr_year / total_possible_increments if total_possible_increments > 0 else 0

        is_ok = (prop_year >= prop_year_tol and prop_incr_year >= prop_incr_tol)

        summary = {
            'increment': name,
            'n_obs': len(df_filtered),
            'n_year': n_years_with_data,
            'prop_year': prop_year,
            'n_incr_year': n_incr_year,
            'prop_incr_year': prop_incr_year,
            'data_ok': is_ok
        }
        availability_summary.append(summary)

        # The first valid increment found will be the best one because we sorted by frequency
        if is_ok and best_time_incr == 'none':
            best_time_incr = name

    # --- 4. Add 'time_increment' Column ---
    if best_time_incr != 'none':
        increment_col_name = increment_map[best_time_incr]
        df_filtered['time_increment'] = df_filtered[increment_col_name]
    else:
        df_filtered['time_increment'] = 'none'

    # --- 5. Generate Plots if Requested ---
    if plot:
        if not plot_path:
            raise ValueError("A 'plot_path' must be provided to save the plots.")

        # Re-join original data to get all necessary columns for plotting,
        # especially 'censored' and 'cen_type' if they exist.
        plot_df = df_filtered.copy()
        if 'censored' in data.columns and 'censored' not in plot_df.columns:
             # Use a left merge to preserve the filtered rows
             plot_df = pd.merge(plot_df, data[['t', 'censored', 'cen_type']], on='t', how='left')

        plot_inspection_data(
            data=plot_df,
            plot_path=plot_path,
            value_col=value_col,
            time_col=time_col,
            time_increment=best_time_incr,
            increment_map=increment_map
        )


    # Clean up helper columns
    df_filtered.drop(columns=['year', 'month', 'quarter', 'bi-month', 'bi-annual', 'week', 'day'], inplace=True)

    summary_df = pd.DataFrame(availability_summary)

    if return_summary:
        return InspectionResult(data=df_filtered, summary=summary_df)

    return df_filtered
