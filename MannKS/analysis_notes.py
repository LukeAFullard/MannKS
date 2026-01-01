"""
This module provides functions for data quality checks inspired by the
LWP-TRENDS R script. These checks generate "Analysis Notes" to warn
the user about potential issues with their data that could affect
trend analysis results.
"""
import numpy as np
from ._stats import _rle_lengths

# --- Module-level Constants for Data Quality Checks ---
MIN_UNIQUE_VALUES = 3
MIN_NON_CENSORED = 5
MIN_UNIQUE_PER_SEASON = 2
MAX_RUN_LENGTH_RATIO = 0.75  # Maximum ratio of identical values

def get_analysis_note(data, values_col='value', censored_col='censored',
                        is_seasonal=False, post_aggregation=False, season_col='season'):
    """
    Performs data quality checks and returns an analysis note.

    Args:
        data (pd.DataFrame): The input data.
        values_col (str): The name of the column with data values.
        censored_col (str): The name of the boolean column indicating censored data.
        is_seasonal (bool): Whether to perform seasonal checks.
        post_aggregation (bool): Whether the checks are for after data aggregation.
        season_col (str): The name of the column with season identifiers.

    Returns:
        str: An analysis note string. "ok" if no issues are found.
    """

    # --- First round of filtering (pre-aggregation) ---
    if not post_aggregation:
        if data[values_col].isnull().all():
            return "Data all NA values"

        non_censored_values = data.loc[~data[censored_col] & ~data[values_col].isnull(), values_col]

        if len(non_censored_values.unique()) < MIN_UNIQUE_VALUES:
            return f"< {MIN_UNIQUE_VALUES} unique values"

        if len(non_censored_values) < MIN_NON_CENSORED:
            return f"< {MIN_NON_CENSORED} Non-censored values"

        return "ok"

    # --- Second round of filtering (post-aggregation) ---
    if post_aggregation:
        if is_seasonal:
            if season_col not in data.columns or data.empty:
                return "ok"

            # Order of checks matters here, mirroring the R script's logic.
            # 1. Check for sufficient non-NA values in each season
            season_counts = data.groupby(season_col)[values_col].count()
            if not season_counts.empty and season_counts.min() < MIN_UNIQUE_VALUES:
                 return f"< {MIN_UNIQUE_VALUES} non-NA values in Season"

            # 2. Check for sufficient unique values in each season
            season_unique_counts = data.groupby(season_col)[values_col].nunique()
            if not season_unique_counts.empty and season_unique_counts.min() < MIN_UNIQUE_PER_SEASON:
                return f"< {MIN_UNIQUE_PER_SEASON} unique values in Season"

            # 3. Check for long runs of identical values within each season
            def check_run_length_seasonal(series):
                series = series.dropna()
                if len(series) <= 1:
                    return False
                rle_res = _rle_lengths(series.to_numpy())
                if not rle_res.any(): return False
                return rle_res.max() / len(series) > MAX_RUN_LENGTH_RATIO

            long_run_in_season = data.groupby(season_col)[values_col].apply(check_run_length_seasonal).any()
            if long_run_in_season:
                return "Long run of single value in a Season"

        else:  # Not seasonal
            values = data[values_col].dropna().to_numpy()
            if len(values) > 1:
                rle_res = _rle_lengths(values)
                if len(rle_res) > 0 and rle_res.max() / len(values) > 0.5:
                    return "Long run of single value"

    return "ok"


def get_sens_slope_analysis_note(slopes, t, cen_type):
    """
    Provides warnings about censored values used in the derivation of the Sen's Slope.

    Args:
        slopes (np.array): Array of calculated slopes.
        t (np.array): Array of timestamps.
        cen_type (np.array): Array of censor types ('lt', 'gt', 'not').

    Returns:
        str: An analysis note string. "ok" if no issues are found.
    """
    if slopes is None or len(slopes) == 0:
        return "ok"

    median_slope = np.nanmedian(slopes)
    if np.isnan(median_slope):
        return "ok"

    n = len(t)
    if n < 2:
        return "ok"
    i, j = np.triu_indices(n, k=1)
    t_diff = t[j] - t[i]
    valid_mask = t_diff != 0
    # Ensure slopes and cen_type_pairs align
    if len(slopes) != np.sum(valid_mask):
        # This case indicates a mismatch that shouldn't happen in normal operation
        return "ok"

    cen_type_pairs = (cen_type[i] + " " + cen_type[j])[valid_mask]

    # Find the minimum absolute difference from the median
    abs_diffs = np.abs(slopes - median_slope)
    min_abs_diff = np.nanmin(abs_diffs)

    # Find all slopes that are close to this minimum difference
    indices_of_median = np.where(np.isclose(abs_diffs, min_abs_diff))[0]

    if len(indices_of_median) == 0:
        return "ok"

    median_cen_labels = cen_type_pairs[indices_of_median]
    unique_median_cen_labels = np.unique(median_cen_labels)

    # Check for influence from censored data
    is_fully_censored = all(label in ['lt lt', 'gt gt', 'lt gt', 'gt lt'] for label in unique_median_cen_labels)
    has_lt = any('lt' in label for label in unique_median_cen_labels)
    has_gt = any('gt' in label for label in unique_median_cen_labels)

    if is_fully_censored:
        return "CRITICAL: Sen slope is based on a pair of two censored values."

    if has_lt and has_gt:
        return "WARNING: Sen slope influenced by left- and right-censored values."
    elif has_lt:
        return "WARNING: Sen slope influenced by left-censored values."
    elif has_gt:
        return "WARNING: Sen slope influenced by right-censored values."

    # Check for ties in non-censored data
    elif np.isclose(median_slope, 0):
        return "WARNING: Sen slope based on tied non-censored values"

    return "ok"
