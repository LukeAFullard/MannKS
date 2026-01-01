"""
This module provides utility functions for pre-processing data for trend analysis.
"""
import numpy as np
import pandas as pd
import warnings

from typing import Union, List

def prepare_censored_data(x: Union[List, np.ndarray, pd.Series]) -> pd.DataFrame:
    """
    Pre-processes a 1D array-like object containing censored data.

    This function takes an array that may contain a mix of numeric and
    string values (e.g., 10, '<5', '>20') and converts it into a structured
    pandas DataFrame with separate columns for the numeric value, a boolean
    censored flag, and the type of censoring.

    Note:
        `np.nan` values in the input array will be converted to `np.nan` in
        the output 'value' column and will be treated as non-censored
        (`censored=False`).

    Args:
        x (array-like): A 1D array or list containing the data.

    Returns:
        pandas.DataFrame: A DataFrame with three columns:
            - 'value': The numeric value of the data point.
            - 'censored': A boolean flag, True if the data point was censored.
            - 'cen_type': A string indicating the type of censoring
                          ('lt', 'gt', or 'not').
    """
    values = []
    censored_flags = []
    cen_types = []

    if not hasattr(x, '__iter__') or isinstance(x, str):
        raise TypeError("Input data must be an iterable (e.g., list, numpy array).")

    for item in x:
        if isinstance(item, str):
            item_stripped = item.strip()
            if item_stripped.startswith('<'):
                try:
                    values.append(float(item_stripped[1:]))
                    censored_flags.append(True)
                    cen_types.append('lt')
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid left-censored value format: '{item}'. Expected a number after the '<' symbol.")
            elif item_stripped.startswith('>'):
                try:
                    values.append(float(item_stripped[1:]))
                    censored_flags.append(True)
                    cen_types.append('gt')
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid right-censored value format: '{item}'. Expected a number after the '>' symbol.")
            else:
                try:
                    values.append(float(item_stripped))
                    censored_flags.append(False)
                    cen_types.append('not')
                except ValueError:
                    raise ValueError(f"Could not convert string '{item}' to a float.")
        else:
            try:
                values.append(float(item))
                censored_flags.append(False)
                cen_types.append('not')
            except (ValueError, TypeError):
                 raise ValueError(f"Could not convert non-string value '{item}' to a float.")

    # --- Mixed Censoring Validation ---
    value_censor_map = {}
    for val, cen_type in zip(values, cen_types):
        if val in value_censor_map and value_censor_map[val] != cen_type:
            warnings.warn(
                f"Value {val} has conflicting censoring types. "
                f"This may indicate data quality issues.", UserWarning
            )
        value_censor_map[val] = cen_type


    return pd.DataFrame({
        'value': np.array(values, dtype=float),
        'censored': np.array(censored_flags, dtype=bool),
        'cen_type': np.array(cen_types, dtype=object)
    })
