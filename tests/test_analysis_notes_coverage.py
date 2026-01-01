import numpy as np
import pandas as pd
import pytest

from MannKS.analysis_notes import (
    get_analysis_note,
    get_sens_slope_analysis_note
)

def test_get_analysis_note_all_na():
    """Test get_analysis_note returns correct message for all NA values."""
    data = pd.DataFrame({'value': [np.nan, np.nan], 'censored': [False, False]})
    note = get_analysis_note(data, post_aggregation=False)
    assert note == "Data all NA values"

def test_get_analysis_note_seasonal_no_season_col():
    """Test get_analysis_note returns 'ok' when season_col is missing."""
    data = pd.DataFrame({'value': [1, 2, 3], 'censored': [False, False, False]})
    note = get_analysis_note(data, is_seasonal=True, post_aggregation=True)
    assert note == "ok"

def test_get_analysis_note_post_agg_seasonal_insufficient_non_na():
    """Test for '< 3 non-NA values in Season' post-aggregation."""
    data = pd.DataFrame({
        'value': [1, 2, np.nan, 4, 5, 6],
        'censored': False,
        'season': [1, 1, 1, 2, 2, 2]
    })
    note = get_analysis_note(data, is_seasonal=True, post_aggregation=True)
    assert note == "< 3 non-NA values in Season"

def test_get_analysis_note_post_agg_seasonal_insufficient_unique():
    """Test for '< 2 unique values in Season' post-aggregation."""
    data = pd.DataFrame({
        'value': [1, 1, 1, 2, 3, 4],
        'censored': False,
        'season': [1, 1, 1, 2, 2, 2]
    })
    note = get_analysis_note(data, is_seasonal=True, post_aggregation=True)
    assert note == "< 2 unique values in Season"

def test_get_analysis_note_post_agg_seasonal_long_run():
    """Test for 'Long run of single value in a Season' post-aggregation."""
    data = pd.DataFrame({
        'value': [1, 1, 1, 1, 2, 3, 4, 4, 4, 4],
        'censored': False,
        'season': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    })
    note = get_analysis_note(data, is_seasonal=True, post_aggregation=True)
    assert note == "Long run of single value in a Season"

def test_get_analysis_note_post_agg_non_seasonal_long_run():
    """Test for 'Long run of single value' post-aggregation."""
    data = pd.DataFrame({
        'value': [1, 1, 1, 1, 1, 1, 3, 4, 5, 6],
        'censored': False
    })
    note = get_analysis_note(data, is_seasonal=False, post_aggregation=True)
    assert note == "Long run of single value"

def test_get_sens_slope_analysis_note_two_censored():
    """Test Sen's slope note for slope based on two censored values."""
    slopes = np.array([3.5, 3.5, 3.5, 3.5, 3.5, 3.5])
    t = np.array([1, 2, 3, 4])
    cen_type = np.array(['lt', 'gt', 'lt', 'gt'])
    note = get_sens_slope_analysis_note(slopes, t, cen_type)
    assert note == "CRITICAL: Sen slope is based on a pair of two censored values."

def test_get_sens_slope_analysis_note_left_right_censored():
    """Test Sen's slope note for influence from left- and right-censored values."""
    slopes = np.array([3.5, 3.5, 3.5, 3.5, 3.5, 3.5])
    t = np.array([1, 2, 3, 4])
    cen_type = np.array(['lt', 'not', 'not', 'gt'])
    note = get_sens_slope_analysis_note(slopes, t, cen_type)
    assert note == "WARNING: Sen slope influenced by left- and right-censored values."

def test_get_sens_slope_analysis_note_right_censored():
    """Test Sen's slope note for influence from right-censored values."""
    slopes = np.array([2.0, 2.0, 2.0])
    t = np.array([1, 2, 3])
    cen_type = np.array(['not', 'not', 'gt'])
    note = get_sens_slope_analysis_note(slopes, t, cen_type)
    assert note == "WARNING: Sen slope influenced by right-censored values."

def test_get_sens_slope_analysis_note_nan_median():
    """Test that get_sens_slope_analysis_note returns 'ok' if median slope is NaN."""
    slopes = np.array([np.nan, np.nan])
    t = np.array([1, 2, 3])
    cen_type = np.array(['not', 'not', 'not'])
    note = get_sens_slope_analysis_note(slopes, t, cen_type)
    assert note == "ok"
