import numpy as np
import pandas as pd
import pytest
import warnings

from MannKS import trend_test, seasonal_trend_test, check_seasonality
from MannKS.analysis_notes import get_sens_slope_analysis_note
from MannKS._stats import _mk_score_and_var_censored

# --- seasonal_trend_test coverage ---

def test_seasonal_trend_test_invalid_season_type():
    """Test error handling for invalid season_type."""
    with pytest.raises(ValueError, match="Unknown season_type: 'invalid'"):
        seasonal_trend_test(x=[1,2,3], t=pd.to_datetime(['2020-01-01', '2021-01-01', '2022-01-01']), season_type='invalid')

# --- check_seasonality coverage ---

def test_check_seasonality_no_unique_seasons():
    """Test case where all data falls into a single season."""
    t = pd.to_datetime(['2020-01-01', '2021-01-15', '2022-01-30'])
    result = check_seasonality(x=[1, 2, 3], t=t)
    assert result.seasons_skipped == [1]
    assert result.seasons_tested == []

def test_check_seasonality_agg_errors():
    """Test ValueError for agg_period in check_seasonality."""
    with pytest.raises(ValueError, match="`agg_period` must be specified for datetime aggregation."):
        check_seasonality(x=[1,2,3], t=pd.to_datetime(['2020-01-01', '2021-01-01', '2022-01-01']), agg_method='median')

    with pytest.raises(ValueError, match="`agg_period` must be specified for numeric aggregation."):
        check_seasonality(x=[1,2,3], t=[1,2,3], agg_method='median')


# --- analysis_notes coverage ---

def test_get_sens_slope_analysis_note_mismatch():
    """Test case where slopes and cen_type_pairs have different lengths."""
    slopes = np.array([1.0, 2.0])
    t = np.array([1, 2, 3, 4])
    cen_type = np.array(['not'] * 4)
    note = get_sens_slope_analysis_note(slopes, t, cen_type)
    assert note == "ok"

def test_get_sens_slope_analysis_note_no_indices():
    """Test when indices_of_median is empty."""
    slopes = np.array([np.nan, np.nan])
    t = np.array([1, 2, 3])
    cen_type = np.array(['not'] * 3)
    note = get_sens_slope_analysis_note(slopes, t, cen_type)
    assert note == "ok"


# --- _stats coverage ---

def test_mk_score_and_var_censored_large_n_warning():
    """Test memory warning for large n."""
    n = 6000
    x = np.arange(n)
    t = np.arange(n)
    censored = np.zeros(n, dtype=bool)
    cen_type = np.full(n, 'not')

    with pytest.warns(UserWarning, match="Large sample size"):
        _mk_score_and_var_censored(x, t, censored, cen_type)

def test_mk_score_and_var_censored_tie_break_methods():
    """Cover the 'lwp' branches for tie_break_method."""
    x = [1, 1, 2, 3]
    t = [1, 2, 2, 3]
    censored = [False] * 4
    cen_type = ['not'] * 4
    _mk_score_and_var_censored(x, t, censored, cen_type, tie_break_method='lwp')

def test_mk_score_and_var_censored_default_delx():
    """Cover the default delx=1.0 case when all values are identical."""
    x = [5, 5, 5, 5]
    t = [1, 2, 3, 4]
    censored = [False] * 4
    cen_type = ['not'] * 4
    s, var_s, _, _ = _mk_score_and_var_censored(x, t, censored, cen_type)
    assert s == 0
