"""Input validation for MannKS.decompose (brief v2 section 5, test 7)."""
import warnings

import numpy as np
import pandas as pd
import pytest

import MannKS


def _series(n=120, years=10, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(2000 + rng.uniform(0, years, n))
    y = np.exp(0.5 * np.cos(2 * np.pi * t) + rng.normal(0, 0.2, n))
    return y, t


def test_too_short_record():
    y, t = _series(years=2.5)
    with pytest.raises(ValueError, match='3 years'):
        MannKS.decompose(y, t)


def test_too_few_detects():
    y, t = _series(n=20)
    with pytest.raises(ValueError, match='24 detected'):
        MannKS.decompose(y, t)


@pytest.mark.parametrize("kw, match", [
    (dict(trend_span_years=1.5), 'trend_span_years'),
    (dict(seasonal_window_years=2.5), 'seasonal_window_years'),
    (dict(n_harmonics=5), 'n_harmonics'),
    (dict(n_harmonics=0), 'n_harmonics'),
    (dict(seasonal='weekly'), 'seasonal'),
    (dict(transform='sqrt'), 'transform'),
    (dict(censored_method='substitution'), 'censored_method'),
    (dict(block_years=1.5, n_boot=5), 'block_years'),
    (dict(n_imputations=1), 'n_imputations'),
    (dict(n_outer=0), 'n_outer'),
])
def test_invalid_options(kw, match):
    y, t = _series()
    with pytest.raises(ValueError, match=match):
        MannKS.decompose(y, t, **kw)


def test_non_positive_values_with_log():
    y, t = _series()
    y[5] = 0.0
    with pytest.raises(ValueError, match='> 0'):
        MannKS.decompose(y, t)
    # fine without the log transform
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        MannKS.decompose(y, t, transform='none')


def test_raw_strings_need_prepare_censored_data():
    y, t = _series()
    x = list(y)
    x[0] = '<0.1'
    with pytest.raises(TypeError, match='prepare_censored_data'):
        MannKS.decompose(x, t)


def test_length_mismatch():
    y, t = _series()
    with pytest.raises(ValueError, match='same length'):
        MannKS.decompose(y[:-1], t)


def test_unsorted_and_duplicate_times_are_accepted():
    y, t = _series()
    t = t.copy()
    t[10] = t[11]
    perm = np.random.default_rng(1).permutation(t.size)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = MannKS.decompose(y[perm], t[perm])
    assert np.all(np.diff(res.t_years) >= 0)
    np.testing.assert_array_equal(res.order, np.argsort(t[perm], kind='stable'))
    assert 'tied timestamps present without aggregation' in res.analysis_notes


def test_numeric_time_not_in_years_warns():
    y, t = _series()
    with pytest.warns(UserWarning, match='decimal years'):
        MannKS.decompose(y, (t - 2000) * 365.25)  # days, not years
