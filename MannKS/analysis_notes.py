"""
Data-quality "analysis notes", following the checks in the LWP-TRENDS R
script (``GetAnalysisNote`` and the Sen-slope warnings in ``SenSlope``).

Differences from LWP-TRENDS
---------------------------
* LWP-TRENDS does not compute a trend when a note other than "ok" is raised;
  MannKS always computes and reports the note alongside the result.
* The run-length checks are applied to the data in time order (LWP-TRENDS
  applies the non-seasonal check to data ordered by its "Month-Year" label).
* The "unique values per season" message states the threshold actually used
  (2); LWP-TRENDS prints "< 3 unique values in Season" for the same check.
"""
import numpy as np
from ._stats import _rle_lengths

# --- Module-level Constants for Data Quality Checks (as in LWP-TRENDS) ---
MIN_UNIQUE_VALUES = 3
MIN_NON_CENSORED = 5
MIN_UNIQUE_PER_SEASON = 2
MAX_RUN_LENGTH_RATIO = 0.75          # seasonal run-length threshold
MAX_RUN_LENGTH_RATIO_NONSEASONAL = 0.5


def _long_run(values, threshold):
    """
    LWP-TRENDS run-length rule: the longest run of identical consecutive
    *differences* (``rle(diff(x))``) relative to the number of values.
    """
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return False
    lengths = _rle_lengths(np.diff(values))
    return len(lengths) > 0 and lengths.max() / len(values) > threshold


def _first_tier(data, values_col, censored_col):
    if data[values_col].isnull().all():
        return "Data all NA values"
    censored = data[censored_col].astype(bool)
    non_censored_values = data.loc[~censored & ~data[values_col].isnull(), values_col]
    if len(non_censored_values.unique()) < MIN_UNIQUE_VALUES:
        return f"< {MIN_UNIQUE_VALUES} unique values"
    if len(non_censored_values) < MIN_NON_CENSORED:
        return f"< {MIN_NON_CENSORED} Non-censored values"
    return "ok"


def get_analysis_note(data, values_col='value', censored_col='censored',
                      is_seasonal=False, post_aggregation=False, season_col='season',
                      expected_seasons=None, time_col='t'):
    """
    Performs data quality checks and returns an analysis note.

    Args:
        data (pd.DataFrame): The input data.
        values_col (str): Column with data values.
        censored_col (str): Boolean column indicating censored data.
        is_seasonal (bool): Whether to perform seasonal checks.
        post_aggregation (bool): Whether the checks are for data after
            aggregation to one value per time increment.  As in LWP-TRENDS,
            the basic checks are repeated, followed by run-length (and
            seasonal) checks.
        season_col (str): Column with season identifiers.
        expected_seasons (iterable, optional): All seasons that should be
            present (e.g. 1..12 for months).  A season with no data then fails
            the "non-NA values in Season" check, as in LWP-TRENDS.
        time_col (str): Column used to order the data for run-length checks.

    Returns:
        str: An analysis note string. "ok" if no issues are found.
    """
    note = _first_tier(data, values_col, censored_col)
    if note != "ok" or not post_aggregation:
        return note

    if time_col in data.columns:
        data = data.sort_values(time_col, kind='stable')

    if is_seasonal:
        if season_col not in data.columns or data.empty:
            return "ok"
        valid = data[~data[values_col].isnull()]
        counts = valid.groupby(season_col)[values_col].count()
        if expected_seasons is not None:
            counts = counts.reindex(list(expected_seasons), fill_value=0)
        if not counts.empty and counts.min() < MIN_UNIQUE_VALUES:
            return f"< {MIN_UNIQUE_VALUES} non-NA values in Season"

        unique_counts = valid.groupby(season_col)[values_col].nunique()
        if not unique_counts.empty and unique_counts.min() < MIN_UNIQUE_PER_SEASON:
            return f"< {MIN_UNIQUE_PER_SEASON} unique values in Season"

        long_run = valid.groupby(season_col)[values_col].apply(
            lambda s: _long_run(s.to_numpy(), MAX_RUN_LENGTH_RATIO)).any()
        if long_run:
            return "Long run of single value in a Season"
    else:
        values = data[values_col].dropna().to_numpy()
        if _long_run(values, MAX_RUN_LENGTH_RATIO_NONSEASONAL):
            return "Long run of single value"

    return "ok"


def get_sens_slope_analysis_note(slopes, t, cen_type):
    """
    Warnings about censored values used in deriving the Sen's slope
    (LWP-TRENDS ``SenSlope``).  The pairs whose slope is closest to the median
    are inspected:

    * all of them pairs of two censored values ->
      "WARNING: Sen slope based on two censored values"
    * any of them involving a censored value ->
      "WARNING: Sen slope influenced by censored values"
    * none censored and the slope is exactly zero ->
      "WARNING: Sen slope based on tied non-censored values"

    ``slopes`` must contain every pair (in ``np.triu_indices`` order, pairs
    with equal times removed), as returned by the exact estimators.  For
    sampled slopes the pairs cannot be identified and "ok" is returned.
    """
    if slopes is None or len(slopes) == 0:
        return "ok"
    median_slope = np.nanmedian(slopes)
    if np.isnan(median_slope):
        return "ok"

    t = np.asarray(t, dtype=float)
    n = len(t)
    if n < 2:
        return "ok"
    # Only exact (all-pair) slope sets can be matched to their pairs; check the
    # count before materialising the pair indices.
    _, counts = np.unique(t, return_counts=True)
    expected = n * (n - 1) // 2 - int(np.sum(counts * (counts - 1) // 2))
    if len(slopes) != expected:
        return "ok"
    i, j = np.triu_indices(n, k=1)
    valid = np.abs(t[j] - t[i]) > 1e-10
    if int(valid.sum()) != expected:
        return "ok"
    i, j = i[valid], j[valid]

    abs_diffs = np.abs(slopes - median_slope)
    idx = np.where(abs_diffs == np.nanmin(abs_diffs))[0]
    cen = np.isin(np.asarray(cen_type).astype(str), ['lt', 'gt'])
    ci, cj = cen[i[idx]], cen[j[idx]]

    if np.any(ci | cj):
        if np.all(ci & cj):
            return "WARNING: Sen slope based on two censored values"
        return "WARNING: Sen slope influenced by censored values"
    if median_slope == 0:
        return "WARNING: Sen slope based on tied non-censored values"
    return "ok"
