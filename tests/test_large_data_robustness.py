"""
Large-data robustness: exact O(n log n) censored Mann-Kendall S, memory-bounded
exact slopes, order-statistic shortcuts, bootstrap pair caps, time handling.
"""
import time
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import MannKS as mk
import MannKS._stats as st
from MannKS._ats import ats_slope
from MannKS._large_dataset import (detect_size_tier, MAX_EXACT_N, _sample_pairs,
                                   sampled_median_error)
from MannKS.analysis_notes import get_sens_slope_analysis_note


def _random_censored(rng, n):
    kind = rng.integers(3)
    if kind == 0:
        x = rng.normal(size=n)
    elif kind == 1:
        x = rng.integers(0, 6, n).astype(float)
    else:
        x = np.arange(n) * 0.01 + rng.integers(0, 3, n)
    t = rng.integers(0, max(2, n // 2), n).astype(float)
    lods = rng.choice([0.5, 1.0, 2.0, 3.0], size=n)
    cens = rng.random(n) < rng.choice([0, 0.1, 0.5, 0.9, 1.0])
    ct = np.where(cens, 'lt', 'not').astype(object)
    gt = (~cens) & (rng.random(n) < 0.1)
    ct[gt] = 'gt'
    cens = cens | gt
    x = np.where(ct == 'lt', lods, x)
    return x, t, cens, ct


@pytest.mark.parametrize("tau,mk_method,tie", [('b', 'robust', 'robust'), ('a', 'lwp', 'lwp'),
                                               ('b', 'lwp', 'robust')])
def test_fast_censored_s_matches_pairwise(monkeypatch, tau, mk_method, tie):
    rng = np.random.default_rng(1)
    for _ in range(150):
        x, t, c, ct = _random_censored(rng, int(rng.integers(2, 90)))
        monkeypatch.setattr(st, '_FAST_S_MIN_N', 10**9)
        ref = st._mk_score_and_var_censored(x, t, c, ct, tau, mk_method, tie)
        monkeypatch.setattr(st, '_FAST_S_MIN_N', 1)
        new = st._mk_score_and_var_censored(x, t, c, ct, tau, mk_method, tie)
        np.testing.assert_allclose(new, ref, rtol=1e-12, atol=1e-9)


def test_censored_trend_test_scales():
    """Censored data in fast mode used to need O(n^2) memory for S."""
    rng = np.random.default_rng(2)
    n = 30000
    x = 1e-4 * np.arange(n) + rng.normal(size=n)
    d = mk.prepare_censored_data([f'<0' if v < 0 else v for v in x])
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = mk.trend_test(d, np.arange(n))
    assert time.time() - t0 < 20
    assert r.computation_mode == 'fast'
    assert r.trend == 'increasing' and abs(r.slope / 1e-4 - 1) < 0.05
    assert r.approximation_error is not None and r.approximation_error > 0


def _ci_reference(slopes, var_s, alpha, method, total_pairs):
    v = np.sort(slopes[~np.isnan(slopes)])
    n_s = len(v)
    if n_s == 0 or var_s < 1e-10:
        return np.nan, np.nan
    n_pop = st._valid_pair_count(slopes, total_pairs)
    c = norm.ppf(1 - alpha / 2) * np.sqrt(var_s)
    r1, r2 = (n_pop - c) / 2 / n_pop * n_s, (n_pop + c) / 2 / n_pop * n_s
    if method == 'lwp':
        return tuple(np.interp([r1, r2], np.arange(1, n_s + 1), v))
    return (v[int(np.clip(np.round(r1 - 1), 0, n_s - 1))],
            v[int(np.clip(np.round(r2), 0, n_s - 1))])


def test_ci_and_sen_probability_match_sorted_reference():
    rng = np.random.default_rng(3)
    for _ in range(2000):
        n = int(rng.integers(1, 50))
        v = rng.integers(-3, 4, n).astype(float) if rng.random() < .5 else rng.normal(size=n)
        if rng.random() < .3:
            v[rng.random(n) < .3] = np.nan
        var = float(rng.choice([0.0, 1.0, 50.0])) * rng.random()
        tp = None if rng.random() < .5 else int(n * 3)
        for m in ('direct', 'lwp'):
            assert np.allclose(st._confidence_intervals(v.copy(), var, 0.1, m, tp),
                               _ci_reference(v, var, 0.1, m, tp), equal_nan=True, rtol=0, atol=0)
        valid = np.sort(v[~np.isnan(v)])
        if len(valid) and var >= 1e-10:
            r = [st._rank_of_zero(valid, k) for k in ('median', 'max', 'min')]
            assert np.allclose(st._zero_ranks(valid), r, equal_nan=True)


def _ats_reference_solve(score, level):
    level = level * score.scale
    P = np.unique(np.concatenate([score.A, score.B]))
    P = P[np.isfinite(P)]
    if len(P) == 0:
        return np.nan
    span = max(1.0, np.abs(P).max())
    locs = np.empty(2 * len(P) + 1)
    locs[0], locs[-1] = P[0] - span, P[-1] + span
    locs[1::2] = P
    locs[2:-1:2] = (P[:-1] + P[1:]) / 2.0
    Sv = score.S(locs)
    le, ge = np.where(Sv <= level)[0], np.where(Sv >= level)[0]
    if len(le) == 0 or len(ge) == 0:
        return np.nan
    k_lo, k_hi = le[0], ge[-1]
    if k_lo == 0 or k_hi == len(locs) - 1:
        return np.nan
    lo = P[(k_lo - 1) // 2] if k_lo % 2 == 1 else P[k_lo // 2 - 1]
    hi = P[(k_hi - 1) // 2] if k_hi % 2 == 1 else P[k_hi // 2]
    return 0.5 * (lo + hi)


def test_ats_bisection_matches_grid_search():
    from MannKS._ats import _ATSScore, make_intervals
    rng = np.random.default_rng(4)
    for _ in range(200):
        n = int(rng.integers(3, 80))
        x, t, c, ct = _random_censored(rng, n)
        lower, upper = make_intervals(x, c, cen_type=np.asarray(ct).astype(str), lod=x)
        score = _ATSScore(t, lower, upper)
        for level in (0.0, 5.0, -5.0, 40.0):
            assert np.allclose(score.solve(level), _ats_reference_solve(score, level), equal_nan=True,
                               rtol=0, atol=0)


def test_sens_note_blockwise_matches_reference():
    rng = np.random.default_rng(5)
    for _ in range(200):
        n = int(rng.integers(3, 60))
        x, t, c, ct = _random_censored(rng, n)
        for m in ('lwp', 'drop_ambiguous'):
            sl = st._sens_estimator_censored(x, t, ct, method=m)
            med = np.nanmedian(sl) if np.any(~np.isnan(sl)) else np.nan
            got = get_sens_slope_analysis_note(sl, t, ct)
            if np.isnan(med):
                assert got == 'ok'
                continue
            i, j = np.triu_indices(n, k=1)
            ok = (t[j] - t[i]) != 0.0
            i, j = i[ok], j[ok]
            d = np.abs(sl - med)
            idx = np.where(d == np.nanmin(d))[0]
            cen = np.isin(np.asarray(ct).astype(str), ['lt', 'gt'])
            ci, cj = cen[i[idx]], cen[j[idx]]
            if np.any(ci | cj):
                exp = ("WARNING: Sen slope based on two censored values" if np.all(ci & cj)
                       else "WARNING: Sen slope influenced by censored values")
            elif med == 0:
                exp = "WARNING: Sen slope based on tied non-censored values"
            else:
                exp = 'ok'
            assert got == exp


def test_exact_slopes_blockwise_match_triu():
    rng = np.random.default_rng(6)
    for n in (2, 3, 50, 2500):
        x = rng.normal(size=n)
        t = np.sort(rng.integers(0, n, n)).astype(float)
        i, j = np.triu_indices(n, k=1)
        ok = (t[j] - t[i]) != 0.0
        np.testing.assert_array_equal(st._sens_estimator_unequal_spacing(x, t),
                                      (x[j] - x[i])[ok] / (t[j] - t[i])[ok])


def test_sample_pairs_distinct_and_seeded():
    i, j = _sample_pairs(50000, 100000, None)
    assert np.all(i < j) and len(np.unique(i * 50000 + j)) == len(i)
    i2, j2 = _sample_pairs(50000, 100000, None)
    np.testing.assert_array_equal(i, i2)


def test_full_mode_beyond_limit_switches_to_fast():
    info = detect_size_tier(MAX_EXACT_N + 1, user_mode='full')
    assert info['strategy'] == 'fast'
    assert any("not possible" in w for w in info['warnings'])
    assert detect_size_tier(MAX_EXACT_N, user_mode='full')['strategy'] == 'full'


def test_bootstrap_slope_uses_pair_cap():
    """Bootstrap slopes used to be exact (O(n^2)) and NaN above the exact limit."""
    from MannKS._bootstrap import _slope
    n = MAX_EXACT_N + 1000
    rng = np.random.default_rng(7)
    t = np.arange(n, dtype=float)
    x = 1e-4 * t + rng.normal(size=n)
    c = np.zeros(n, bool)
    ct = np.full(n, 'not', dtype=object)
    t0 = time.time()
    b = _slope(x, t, c, ct, 'drop_ambiguous', max_pairs=100000, random_state=0)
    assert time.time() - t0 < 5
    assert abs(b / 1e-4 - 1) < 0.05


def test_bootstrap_ci_finite_in_fast_mode():
    rng = np.random.default_rng(8)
    n = 6000
    t = np.arange(n, dtype=float)
    x = 1e-3 * t + rng.normal(size=n)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = mk.trend_test(x, t, autocorr_method='block_bootstrap', n_bootstrap=30, random_state=1)
    assert r.computation_mode == 'fast'
    assert np.isfinite(r.lower_ci) and np.isfinite(r.upper_ci) and r.lower_ci < r.slope < r.upper_ci


def test_sampled_median_error_is_order_statistic_interval():
    v = np.arange(10000, dtype=float)
    half = 1.959964 * 100 / 2
    exp = (np.ceil(5000 + half) - np.floor(5000 - half)) / 2
    assert sampled_median_error(v) == exp
    assert sampled_median_error(np.array([1.0])) is None


def test_mode_and_warnings_follow_aggregated_size():
    n = 8000
    dates = pd.date_range('2000-01-01', periods=n, freq='h')
    x = np.arange(n) * 1e-3 + np.random.default_rng(9).normal(size=n)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = mk.trend_test(x, dates, agg_method='median', agg_period='day')
    assert r.computation_mode == 'full'
    assert not any('Large dataset' in str(m.message) for m in w)


def test_missing_and_infinite_times_values_dropped():
    t = pd.Series(pd.date_range('2000-01-01', periods=40, freq='MS'))
    t[3] = pd.NaT
    x = np.arange(40.0)
    x[5] = np.inf
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = mk.trend_test(x, t)
    assert any('Removed 2 observation' in str(m.message) for m in w)
    years = (t.dropna().max() - t.dropna().min()).days / 365.25
    assert 0 < r.slope * 3.15576e7 < 2 * 39 / years
    out = mk.inspect_trend_data(pd.DataFrame({'t': t, 'value': x}), time_col='t')
    assert len(out) == 39


def test_time_conversions_consistent():
    from MannKS._helpers import _preprocessing
    naive = pd.date_range('2000-01-01', periods=10, freq='MS')
    ref = naive.to_numpy().astype('datetime64[s]').astype(float)
    pydt = np.array([d.to_pydatetime() for d in naive], dtype=object)
    np.testing.assert_array_equal(_preprocessing(pydt)[0], ref)
    aware = naive.tz_localize('Pacific/Auckland')
    utc = aware.tz_convert('UTC').tz_localize(None).to_numpy().astype('datetime64[s]').astype(float)
    np.testing.assert_array_equal(_preprocessing(np.asarray(pd.Series(aware)))[0], utc)
    with pytest.raises(TypeError, match='contains strings'):
        mk.trend_test(np.arange(10.0), [str(d.date()) for d in naive])


def test_notes_not_duplicated():
    r = mk.trend_test(np.r_[np.zeros(10), np.ones(10)], np.arange(20))
    assert len(r.analysis_notes) == len(set(r.analysis_notes))


def test_approximation_error_in_slope_units():
    n = 8000
    dates = pd.date_range('2000-01-01', periods=n, freq='h')
    x = 0.01 * np.arange(n) + np.random.default_rng(10).normal(size=n)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        per_s = mk.trend_test(x, dates)
        per_h = mk.trend_test(x, dates, slope_scaling='hour')
        seas_s = mk.seasonal_trend_test(x, dates, season_type='hour')
        seas_h = mk.seasonal_trend_test(x, dates, season_type='hour', slope_scaling='hour')
    assert per_h.approximation_error == pytest.approx(per_s.approximation_error * 3600)
    assert seas_h.approximation_error == pytest.approx(seas_s.approximation_error * 3600)
