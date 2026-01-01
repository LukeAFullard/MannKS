
import numpy as np
import pandas as pd
import pytest
import warnings
from MannKS.analysis_notes import get_analysis_note, get_sens_slope_analysis_note
from MannKS._stats import _sens_estimator_censored
from MannKS.preprocessing import prepare_censored_data
from MannKS.trend_test import trend_test
from MannKS.seasonal_trend_test import seasonal_trend_test

def test_get_analysis_note_all_na():
    data = pd.DataFrame({'value': [np.nan, np.nan, np.nan], 'censored': [False, False, False]})
    note = get_analysis_note(data)
    assert note == "Data all NA values"

def test_get_analysis_note_less_than_3_unique_values():
    data = pd.DataFrame({'value': [1, 1, 2, 2, 1], 'censored': [False, False, False, False, False]})
    note = get_analysis_note(data)
    assert note == "< 3 unique values"

def test_get_analysis_note_less_than_5_non_censored_values():
    data = pd.DataFrame({'value': [1, 2, 3, 4, 5], 'censored': [True, True, False, False, False]})
    note = get_analysis_note(data)
    assert note == "< 5 Non-censored values"

def test_get_analysis_note_long_run_of_single_value():
    data = pd.DataFrame({'value': [1]*11 + [2, 3, 4, 5, 6, 7, 8, 9, 10], 'censored': [False]*20})
    note = get_analysis_note(data, post_aggregation=True)
    assert note == "Long run of single value"

def test_get_analysis_note_seasonal_less_than_3_non_na():
    data = pd.DataFrame({
        'value': [1, 2, np.nan, 4, 5, 6],
        'censored': [False, False, False, False, False, False],
        'season': [1, 1, 1, 2, 2, 2]
    })
    note = get_analysis_note(data, is_seasonal=True, post_aggregation=True)
    assert note == "< 3 non-NA values in Season"


def test_get_analysis_note_seasonal_less_than_2_unique_values():
    data = pd.DataFrame({
        'value': [1, 1, 1, 4, 5, 6],
        'censored': [False, False, False, False, False, False],
        'season': [1, 1, 1, 2, 2, 2]
    })
    note = get_analysis_note(data, is_seasonal=True, post_aggregation=True)
    assert note == "< 2 unique values in Season"

def test_get_analysis_note_seasonal_long_run():
    data = pd.DataFrame({
        'value': [1, 1, 1, 1, 2, 5, 6, 7, 8],
        'censored': [False, False, False, False, False, False, False, False, False],
        'season': [1, 1, 1, 1, 1, 2, 2, 2, 2]
    })
    note = get_analysis_note(data, is_seasonal=True, post_aggregation=True)
    assert note == "Long run of single value in a Season"

def test_get_sens_slope_analysis_note_ok():
    slopes = np.array([1, 2, 3, 4, 5])
    t = np.array([1, 2, 3, 4, 5])
    cen_type = np.array(['not', 'not', 'not', 'not', 'not'])
    note = get_sens_slope_analysis_note(slopes, t, cen_type)
    assert note == "ok"

def test_get_sens_slope_analysis_note_influenced_by_left_censored():
    x_raw = ['<1', 5, 10, 12]
    t = np.arange(len(x_raw))
    data = prepare_censored_data(x_raw)
    slopes = _sens_estimator_censored(data['value'].values, t, data['cen_type'].values, method='lwp')
    note = get_sens_slope_analysis_note(slopes, t, data['cen_type'].values)
    assert note == "WARNING: Sen slope influenced by left-censored values."

def test_get_sens_slope_analysis_note_influenced_by_right_censored():
    x_raw = [1, 5, 2, '>10']
    t = np.arange(len(x_raw))
    data = prepare_censored_data(x_raw)
    slopes = _sens_estimator_censored(data['value'].values, t, data['cen_type'].values, method='lwp')
    note = get_sens_slope_analysis_note(slopes, t, data['cen_type'].values)
    assert note == "WARNING: Sen slope influenced by right-censored values."

def test_get_sens_slope_analysis_note_influenced_by_both_censored():
    x_raw = ['<1', 5, 2, '>10']
    t = np.arange(len(x_raw))
    data = prepare_censored_data(x_raw)
    slopes = _sens_estimator_censored(data['value'].values, t, data['cen_type'].values, method='lwp')
    note = get_sens_slope_analysis_note(slopes, t, data['cen_type'].values)
    assert note == "WARNING: Sen slope influenced by left- and right-censored values."

def test_get_sens_slope_analysis_note_based_on_two_censored():
    x_raw = ['<1', '<2', '>3', '>4']
    t = np.arange(len(x_raw))
    data = prepare_censored_data(x_raw)
    slopes = _sens_estimator_censored(data['value'].values, t, data['cen_type'].values, method='lwp')
    note = get_sens_slope_analysis_note(slopes, t, data['cen_type'].values)
    assert note == "CRITICAL: Sen slope is based on a pair of two censored values."

def test_get_sens_slope_analysis_note_tied_non_censored():
    x_raw = [1, 2, 1, 2, 1]
    t = np.arange(len(x_raw))
    data = prepare_censored_data(x_raw)
    slopes = _sens_estimator_censored(data['value'].values, t, data['cen_type'].values, method='lwp')
    note = get_sens_slope_analysis_note(slopes, t, data['cen_type'].values)
    assert note == "WARNING: Sen slope based on tied non-censored values"

def test_trend_test_min_size_warning():
    """Test that trend_test returns a note for small sample sizes."""
    x = [1, 2, 3, 4, 5]
    t = [1, 2, 3, 4, 5]
    result = trend_test(x, t, min_size=6)
    assert 'sample size (5) below minimum (6)' in result.analysis_notes

    # Test that no note is issued when min_size is None
    result_none = trend_test(x, t, min_size=None)
    assert 'sample size' not in ' '.join(result_none.analysis_notes)


def test_seasonal_trend_test_min_size_warning():
    """Test that seasonal_trend_test returns a note for small seasonal sample sizes."""
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    t = [1, 13, 25, 2, 14, 26, 3, 15, 27]  # 3 seasons, each with 3 values
    result = seasonal_trend_test(x, t, period=12, min_size_per_season=4)
    assert 'minimum season size (3) below minimum (4)' in result.analysis_notes

    # Test that no note is issued when min_size_per_season is None
    result_none = seasonal_trend_test(x, t, period=12, min_size_per_season=None)
    assert 'minimum season size' not in ' '.join(result_none.analysis_notes)
