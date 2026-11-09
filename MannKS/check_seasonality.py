"""
This script provides a function to test for seasonality in a time series
using the Kruskal-Wallis H-test.
"""
from collections import namedtuple
import numpy as np
import pandas as pd
from scipy.stats import kruskal
import warnings
from ._datetime import _get_season_func, _is_datetime_like, _get_agg_func
from ._helpers import _prepare_data, _aggregate_by_group

from typing import Union, Optional

def check_seasonality(
    x: np.ndarray,
    t: np.ndarray,
    period: Optional[int] = None,
    alpha: float = 0.05,
    season_type: str = 'month',
    agg_method: str = 'none',
    agg_period: str = None,
    hicensor: Union[bool, int, float] = False,

) -> namedtuple:
    """
    Performs a Kruskal-Wallis H-test to determine if there is a statistically
    significant difference between the distributions of seasons in a time series.

    This function is a valuable precursor to a seasonal trend test, as it helps
    justify the seasonal approach. It now includes temporal aggregation options
    to ensure that the seasonality check is consistent with any aggregation
    performed in the main trend analysis (e.g., `seasonal_trend_test`). Using
    the same aggregation for both is highly recommended to prevent misleading
    conclusions.

    Args:
        x (np.ndarray): A vector of data, which can be numeric or a pandas
            DataFrame from `prepare_censored_data`.
        t (np.ndarray): A vector of corresponding timestamps, which can be
            numeric or datetime-like.
        period (int, optional): The number of seasons in a full cycle.  For
            datetime ``t`` it is inferred from ``season_type`` (e.g. 12 for
            'month', 4 for 'quarter').  For numeric ``t`` it defaults to 12,
            and ``t`` must be expressed in season units (season =
            floor(t - min(t)) % period).
        alpha (float, optional): The significance level for the test.
            Defaults to 0.05.
        season_type (str, optional): The type of seasonality for datetime inputs
            (e.g., 'month', 'day_of_week'). Defaults to 'month'.
        agg_method (str, optional): The method for temporal aggregation. It is
            strongly recommended to use the same method as in the subsequent
            trend test. Defaults to 'none'.
                - 'none': No aggregation is performed.
                - 'median': Aggregates to the median value within each period.
                - 'robust_median': A more robust median for censored data.
                - 'middle': Selects the observation closest to the time mean.
                - 'lwp' / 'middle_lwp': Selects the observation closest to the
                  mid-point of the `agg_period` increment (LWP-TRENDS rule).
        agg_period (str, optional): The time period for aggregation (e.g.,
            'year', 'month', 'week'). Required if `agg_method` is not 'none'.
        hicensor (Union[bool, int, float], optional): The high-censor threshold.
            If True, uses the highest left-censored ('<') value. If a number is
            provided, that value is used as the threshold. Defaults to False.

    Returns:
        namedtuple: A named tuple with the following fields:
            - h_statistic (float): The Kruskal-Wallis H-statistic.
            - p_value (float): The p-value of the test.
            - is_seasonal (bool): True if seasonality was detected at the
              specified alpha level.
            - seasons_tested (list): The seasons included in the test.
            - seasons_skipped (list): Seasons skipped due to insufficient data.

    Statistical Assumptions:
    ----------------------
    The Kruskal-Wallis H-test is a non-parametric test used to determine if
    there are statistically significant differences between two or more groups
    of an independent variable on a continuous or ordinal dependent variable.
    It is the non-parametric equivalent of the one-way ANOVA.

    1.  **Independence of Observations**: The observations in each group (season)
        must be independent of each other.
    2.  **Ordinal or Continuous Data**: The dependent variable (the data `x`)
        should be measured on an ordinal or continuous scale.
    3.  **Similar Distribution Shape**: The test assumes that the distributions
        of the data in each group have a similar shape. If the shapes of the
        distributions are different, the test may lead to incorrect conclusions
        about the medians.
    4.  **Minimum Sample Size**: While there is no strict rule, it is generally
        recommended to have at least 5 observations per group for the test to
        be reliable. This implementation checks for a minimum of 3, following
        the LWP-TRENDS script.
    """
    res = namedtuple('Seasonality_Test', ['h_statistic', 'p_value', 'is_seasonal', 'seasons_tested', 'seasons_skipped'])

    x_arr = np.asarray(x) if not isinstance(x, pd.DataFrame) else x
    t_arr = np.asarray(t)
    if len(x_arr) != len(t_arr):
        raise ValueError(f"Input vectors `x` and `t` must have the same length. Got {len(x_arr)} and {len(t_arr)}.")

    data, is_datetime = _prepare_data(x, t, hicensor)

    valid_agg = ['none', 'median', 'robust_median', 'middle', 'middle_lwp', 'lwp']
    if agg_method not in valid_agg:
        raise ValueError(f"Invalid `agg_method`. Must be one of {valid_agg}.")

    if period is None:
        if is_datetime:
            from ._datetime import _infer_period
            period = _infer_period(season_type)
        else:
            period = 12

    if agg_method != 'none':
        method = 'middle_lwp' if agg_method == 'lwp' else agg_method
        freq = None
        if is_datetime:
            if agg_period is None:
                raise ValueError("`agg_period` must be specified for datetime aggregation.")
            agg_func = _get_agg_func(agg_period)
            agg_col = agg_func(pd.to_datetime(data['t_original']))
            freq = {'year': 'Y', 'quarter': 'Q', 'month': 'M', 'week': 'W', 'day': 'D'}.get(agg_period.lower())
        else:
            if agg_period is None:
                 raise ValueError("`agg_period` must be specified for numeric aggregation.")
            agg_col = np.floor(data['t'] / agg_period)

        data = data.assign(_agg_key=np.asarray(agg_col)).sort_values('t', kind='stable')
        data = pd.DataFrame(
            [_aggregate_by_group(grp.drop(columns='_agg_key'), method, is_datetime, period=freq, return_dict=True)
             for _, grp in data.groupby('_agg_key', sort=True)]
        )

    if len(data) < 2:
        return res(np.nan, np.nan, False, [], [])

    if is_datetime:
        season_func = _get_season_func(season_type, period)
        seasons = season_func(pd.to_datetime(data['t_original']))
    else:
        t_numeric = np.asarray(data['t'], dtype=np.float64)
        t_normalized = t_numeric - np.min(t_numeric)
        seasons = (np.floor(t_normalized) % period).astype(np.intp)
        uniq = np.unique(t_numeric)
        spacing = np.median(np.diff(uniq)) if len(uniq) > 1 else 1.0
        if np.ptp(t_numeric) < 2 * period or (period > 1 and spacing < 0.5):
            warnings.warn(
                "Numeric `t` should be expressed in season units (season = floor(t - min(t)) % period). "
                "Check that `t` and `period` are consistent (e.g. months with period=12, not decimal years).",
                UserWarning)

    unique_seasons = np.unique(seasons)
    if len(unique_seasons) < 2:
        return res(np.nan, np.nan, False, [], list(unique_seasons))

    seasonal_data = []
    skipped_seasons = []
    tested_seasons = []
    for s in unique_seasons:
        group = data['value'][seasons == s]
        if len(group) < 3 or len(np.unique(group)) < 2:
            if len(group) < 3:
                warnings.warn(f"Season '{s}' has less than 3 samples and will be skipped.", UserWarning)
            else:
                warnings.warn(f"Season '{s}' has less than 2 unique values and will be skipped.", UserWarning)
            skipped_seasons.append(s)
            continue
        seasonal_data.append(group)
        tested_seasons.append(s)

    if len(seasonal_data) < 2:
        return res(np.nan, np.nan, False, tested_seasons, skipped_seasons)

    h_statistic, p_value = kruskal(*seasonal_data)
    is_seasonal = p_value < alpha

    return res(h_statistic, p_value, is_seasonal, tested_seasons, skipped_seasons)
