"""Regression tests for the defects fixed in v0.6.0."""
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import MannKS as mk
from MannKS._stats import (_mk_score_and_var_censored, _p_value, _sen_probability,
                           _sens_estimator_unequal_spacing)
from MannKS._ats import ats_slope, S_of_beta, make_intervals, _ATSScore
from MannKS._helpers import _lwp_period_midpoint


def _censored_series(n=40, slope=0.05, dl=2.0, seed=0):
    g = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    v = 2 + slope * t + g.normal(0, 1, n)
    return mk.prepare_censored_data([f"<{dl}" if a < dl else a for a in v]), t, v


# --- Var(S) ------------------------------------------------------------------

def test_variance_includes_tie_correction():
    x = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7, 7, 7, 8], float)
    n = len(x)
    _, counts = np.unique(x, return_counts=True)
    expected = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18
    var_s = _mk_score_and_var_censored(x, np.arange(n), np.zeros(n, bool), np.full(n, 'not'))[1]
    assert var_s == pytest.approx(expected)


def test_tie_break_methods_agree():
    g = np.random.default_rng(3)
    for _ in range(30):
        n = int(g.integers(5, 40))
        x = np.round(g.normal(10, 2, n))
        cen = np.where(x < 9, 'lt', 'not').astype(object)
        x[cen == 'lt'] = 9
        a = _mk_score_and_var_censored(x, np.arange(n), cen != 'not', cen, tie_break_method='lwp')[1]
        b = _mk_score_and_var_censored(x, np.arange(n), cen != 'not', cen, tie_break_method='robust')[1]
        assert a == pytest.approx(b, rel=1e-12)


def test_tied_times_score_zero_and_are_order_invariant():
    t = np.array([1, 1, 2, 2, 3, 3, 4, 4], float)
    a = mk.trend_test(np.arange(8.0), t)
    b = mk.trend_test(np.array([2, 1, 4, 3, 6, 5, 8, 7], float), t)
    assert a.s == b.s == 24


def test_fast_path_exact_with_tied_times():
    g = np.random.default_rng(1)
    n = 6000
    t = np.repeat(np.arange(n // 2), 2).astype(float)
    x = np.round(0.001 * t + g.normal(0, 1, n), 1)
    fast = mk.trend_test(x, t)
    exact = mk.trend_test(x, t, mk_test_method='lwp')  # pairwise path
    assert fast.s == exact.s
    assert fast.var_s == pytest.approx(exact.var_s)


def test_p_value_does_not_underflow():
    p, _, _ = _p_value(9.0, 0.05)
    assert p > 0


# --- Censored slope ----------------------------------------------------------

def test_default_censored_slope_is_ats_and_nearly_unbiased():
    assert inspect.signature(mk.trend_test).parameters['sens_slope_method'].default == 'ats'
    assert inspect.signature(mk.seasonal_trend_test).parameters['sens_slope_method'].default == 'ats'
    est = []
    for seed in range(60):
        X, t, _ = _censored_series(seed=seed)  # ~20% censored
        est.append(mk.trend_test(X, t).slope)
    assert abs(np.mean(est) - 0.05) / 0.05 < 0.06


def test_ats_equals_sen_without_censoring():
    g = np.random.default_rng(2)
    for n in (7, 8, 31):
        t = np.sort(g.uniform(0, 10, n))
        y = 0.3 * t + g.normal(0, 1, n)
        a = ats_slope(t, y, np.zeros(n, bool))
        assert a['beta'] == pytest.approx(np.median(_sens_estimator_unequal_spacing(y, t)), rel=1e-12)


def test_ats_score_matches_reference_and_respects_time():
    X, t, _ = _censored_series(n=25, seed=4)
    lo, hi = make_intervals(X.value.to_numpy(), X.censored.to_numpy(), X.cen_type.to_numpy())
    score = _ATSScore(t, lo, hi)
    for b in (-1, 0, 0.05, 0.1, 1):
        assert S_of_beta(b, t, lo, hi) == score.S(b)
    # S depends on the time ordering: negating time (and the slope) flips its sign
    assert S_of_beta(-0.05, -t, lo, hi) == -S_of_beta(0.05, t, lo, hi)


@pytest.mark.parametrize('method', ['ats', 'lwp', 'unbiased'])
def test_trend_test_invariant_to_row_order(method):
    X, t, _ = _censored_series(n=30, seed=3)
    perm = np.random.default_rng(0).permutation(len(t))
    a = mk.trend_test(X, t, sens_slope_method=method)
    b = mk.trend_test(X.iloc[perm].reset_index(drop=True), t[perm], sens_slope_method=method)
    assert a.slope == pytest.approx(b.slope)
    assert a.lower_ci == pytest.approx(b.lower_ci)
    assert a.upper_ci == pytest.approx(b.upper_ci)


def test_ats_on_uncensored_data_uses_correct_ci_scaling():
    g = np.random.default_rng(0)
    t = np.arange(1000.0)
    x = 0.01 * t + g.normal(0, 1, 1000)
    a = mk.trend_test(x, t)
    b = mk.trend_test(x, t, sens_slope_method='ats')
    assert a.lower_ci == pytest.approx(b.lower_ci)
    assert a.upper_ci == pytest.approx(b.upper_ci)


def test_sen_probability_matches_lwp_ties_rule():
    # Three slopes tied at 0 (ranks 2-4): R approx(ties=median) uses rank 3.
    slopes = np.array([-1, 0, 0, 0, 1, 2.0])
    p_med, p_max, p_min = _sen_probability(slopes, 4.0)
    assert p_med == pytest.approx(norm.cdf((2 * 3 - 6) / 2))
    assert p_max == pytest.approx(norm.cdf((2 * 4 - 6) / 2))
    assert p_min == pytest.approx(norm.cdf((2 * 2 - 6) / 2))
    # it is the probability that the slope is below zero
    assert _sen_probability(np.array([0.1, 0.2, 0.3, 0.4, -0.1]), 4.0)[0] < 0.5


# --- Large data sets ---------------------------------------------------------

def test_full_mode_is_exact():
    g = np.random.default_rng(0)
    n = 1000
    t = np.arange(n, dtype=float)
    x = 0.01 * t + g.normal(0, 1, n)
    i, j = np.triu_indices(n, 1)
    exact = np.median((x[j] - x[i]) / (t[j] - t[i]))
    r1 = mk.trend_test(x, t, large_dataset_mode='full')
    r2 = mk.trend_test(x, t, large_dataset_mode='full')
    assert r1.slope == exact == r2.slope


def test_fast_mode_is_reproducible_by_default():
    g = np.random.default_rng(0)
    t = np.arange(800, dtype=float)
    x = 0.01 * t + g.normal(0, 1, 800)
    a = mk.trend_test(x, t, large_dataset_mode='fast', max_pairs=20000)
    b = mk.trend_test(x, t, large_dataset_mode='fast', max_pairs=20000)
    assert a.slope == b.slope
    assert a.pairs_used == 20000


def test_seasonal_sampling_weights_seasons_by_pairs():
    g = np.random.default_rng(1)
    k1 = np.sort(g.choice(np.arange(0, 20000, 2), 500, replace=False)).astype(float)
    k2 = np.sort(g.choice(np.arange(1, 20000, 2), 3000, replace=False)).astype(float)
    x1 = 0.0005 * k1 + g.normal(0, 1, 500)
    x2 = 0.0025 * k2 + g.normal(0, 1, 3000)
    exact = np.median(np.r_[_sens_estimator_unequal_spacing(x1, k1), _sens_estimator_unequal_spacing(x2, k2)])
    t, x = np.r_[k1, k2], np.r_[x1, x2]
    o = np.argsort(t)
    full = mk.seasonal_trend_test(x[o], t[o], period=2, large_dataset_mode='full')
    fast = mk.seasonal_trend_test(x[o], t[o], period=2, large_dataset_mode='fast')
    assert full.slope == pytest.approx(exact)
    assert fast.slope == pytest.approx(exact, rel=0.02)


# --- Aggregation -------------------------------------------------------------

def test_lwp_midpoints_match_r():
    assert _lwp_period_midpoint('2001-03-10', 'M') == pd.Timestamp('2001-03-17 01:00')
    assert _lwp_period_midpoint('2001-08-01', 'Q') == pd.Timestamp('2001-08-16 03:00')
    assert _lwp_period_midpoint('2001-12-31', 'Y') == pd.Timestamp('2001-07-01 12:00')


def test_lwp_aggregation_picks_observation_nearest_r_midpoint():
    # R annual mid-point is 1 July 12:00; 30 June is 1.5 days away, 3 July 1.5 days.
    # 1 July 00:00 vs 2 July 23:00: R's mid-point favours 1 July.
    dates = pd.to_datetime(['2001-07-01', '2001-07-02 23:00', '2002-07-01', '2002-07-02 23:00'] +
                           [f'{y}-07-01' for y in range(2003, 2010)], format='ISO8601')
    x = np.r_[1.0, 50.0, 2.0, 60.0, np.arange(3.0, 10.0)]
    for agg in ('lwp', 'middle_lwp'):
        r = mk.trend_test(x, dates, agg_method=agg, agg_period='year', min_size=None)
        assert r.s == 9 * 8 / 2  # monotone after choosing 1 July in 2001 and 2002


def test_prop_censored_refers_to_raw_data():
    d = pd.to_datetime(["2000-01-05", "2000-01-20", "2000-02-05", "2000-02-20", "2000-03-05", "2000-03-20"])
    X = mk.prepare_censored_data(["<1", "<1", "<1", "<1", 2, 3])
    r = mk.trend_test(X, d, agg_method="lwp", agg_period="month", min_size=None)
    assert r.prop_censored == pytest.approx(4 / 6)


# --- Notes -------------------------------------------------------------------

def test_no_spurious_tied_note_for_datetime_slopes():
    d = pd.date_range("2000-01-01", periods=30, freq="365D")
    x = 10 + 0.2 * np.arange(30) + np.random.default_rng(1).normal(0, 0.3, 30)
    r = mk.trend_test(x, d, slope_scaling="year")
    assert not any('tied' in n for n in r.analysis_notes)


def test_seasonal_notes_not_duplicated():
    d = pd.date_range("2000-01-01", periods=120, freq="MS") + pd.Timedelta(days=14)
    X, _, _ = _censored_series(n=120, slope=0.01, dl=2.3, seed=2)
    r = mk.seasonal_trend_test(X, d, sens_slope_method='lwp')
    assert len(r.analysis_notes) == len(set(r.analysis_notes))


def test_seasonal_numeric_time_in_wrong_units_warns():
    tdec = 2000 + np.arange(120) / 12
    x = np.sin(2 * np.pi * np.arange(120) / 12)
    with pytest.warns(UserWarning, match="season units"):
        mk.seasonal_trend_test(x, tdec, period=12)


# --- Autocorrelation and bootstrap -------------------------------------------

def test_yue_wang_does_not_penalise_trend_with_iid_noise():
    g = np.random.default_rng(11)
    t = np.arange(50, dtype=float)
    x = 0.05 * t + g.normal(0, 0.5, 50)
    r = mk.trend_test(x, t, autocorr_method='yue_wang')
    assert r.n_effective == pytest.approx(50)
    assert r.p == pytest.approx(mk.trend_test(x, t).p)
    auto = mk.trend_test(x, t, autocorr_method='auto')
    assert not any('Autocorrelation detected' in n for n in auto.analysis_notes)


def test_yue_wang_type_one_error():
    rej = 0
    for seed in range(300):
        g = np.random.default_rng(seed)
        x = g.normal(size=60)
        rej += mk.trend_test(x, np.arange(60.0), autocorr_method='yue_wang').p < 0.05
    assert rej / 300 < 0.08


def test_lwp_ci_keeps_yue_wang_inflation():
    g = np.random.default_rng(5)
    t = np.arange(80, dtype=float)
    e = np.zeros(80)
    for k in range(1, 80):
        e[k] = 0.8 * e[k - 1] + g.normal()
    x = 0.02 * t + e
    corrected = mk.trend_test(x, t, autocorr_method='yue_wang', ci_method='lwp')
    plain = mk.trend_test(x, t, ci_method='lwp')
    assert corrected.n_effective < 80
    assert corrected.upper_ci - corrected.lower_ci > plain.upper_ci - plain.lower_ci


def test_block_bootstrap_reproducible_and_p_never_zero():
    g = np.random.default_rng(5)
    t = np.arange(60, dtype=float)
    x = 0.5 * t + g.normal(0, 1, 60)
    a = mk.trend_test(x, t, autocorr_method='block_bootstrap', n_bootstrap=99, random_state=1)
    b = mk.trend_test(x, t, autocorr_method='block_bootstrap', n_bootstrap=99, random_state=1)
    assert a.p == b.p == pytest.approx(1 / 100)
    assert (a.lower_ci, a.upper_ci) == (b.lower_ci, b.upper_ci)


def test_seasonal_block_bootstrap_with_ats_runs():
    d = pd.date_range('2000-01-01', periods=72, freq='MS') + pd.Timedelta(days=14)
    X, _, _ = _censored_series(n=72, slope=0.01, dl=1.8, seed=1)
    r = mk.seasonal_trend_test(X, d, autocorr_method='block_bootstrap', sens_slope_method='ats',
                               n_bootstrap=30, random_state=0)
    assert np.isfinite(r.slope) and np.isfinite(r.lower_ci) and r.lower_ci <= r.upper_ci


# --- Regional ----------------------------------------------------------------

def _regional_inputs(day_offsets):
    g = np.random.default_rng(8)
    months = pd.date_range('2005-01-01', periods=60, freq='MS')
    common = g.normal(0, 1, 60)
    rows = []
    for k, site in enumerate(['A', 'B', 'C', 'D']):
        v = 0.01 * (k + 1) * np.arange(60) + common + g.normal(0, 1, 60)
        rows += [dict(site=site, time=m + pd.Timedelta(days=day_offsets[k]), value=val)
                 for m, val in zip(months, v)]
    trends = pd.DataFrame({'site': list('ABCD'), 's': [50, 80, 30, -10], 'C': [0.9, 0.97, 0.8, 0.6]})
    return trends, pd.DataFrame(rows)


def test_regional_aligns_sites_by_time_increment():
    tr, same = _regional_inputs([14, 14, 14, 14])
    _, diff = _regional_inputs([2, 9, 17, 25])
    a = mk.regional_test(tr, same)
    b = mk.regional_test(tr, diff)
    assert np.isfinite(b.CorrectedVarTAU)
    assert b.CorrectedVarTAU == pytest.approx(a.CorrectedVarTAU)
    assert b.CT == pytest.approx(a.CT)


def test_regional_nan_correlation_does_not_force_full_confidence():
    tr = pd.DataFrame({"site": ["A", "B", "C"], "s": [10, 12, -3], "C": [0.80, 0.85, 0.6]})
    ts = pd.DataFrame({"site": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
                       "time": list(range(5)) + list(range(10, 15)) + list(range(20, 25)),
                       "value": np.r_[np.arange(5), np.arange(5), np.arange(5)].astype(float)})
    with pytest.warns(UserWarning, match="correlations could not be computed"):
        r = mk.regional_test(tr, ts)
    assert r.CorrectedVarTAU == pytest.approx(r.VarTAU)
    assert r.CT == pytest.approx(norm.cdf((r.TAU - 0.5) / np.sqrt(r.VarTAU)))
    assert r.CT < 1.0


# --- Segmented, rolling, compare_periods --------------------------------------

def test_segmented_uses_trend_test_censoring_rules():
    from MannKS.segmented_trend_test import _prepare_data
    X = mk.prepare_censored_data(["<1", 2, 3, 4, 5, 6, 7, 8, 9, 10])
    df, _ = _prepare_data(X, np.arange(10.0), hicensor=5)
    # min(max lt limit, hicensor) = 1 -> nothing changes, labels are 'not'
    assert df.value.tolist() == list(np.arange(1.0, 11.0))
    assert set(df.cen_type) == {'lt', 'not'}


def test_segmented_bagging_ci_uses_alpha_and_plot_accepts_censored(tmp_path):
    g = np.random.default_rng(1)
    t = np.arange(80.0)
    x = np.where(t < 40, 0.5 * t, 20 - 0.3 * (t - 40)) + g.normal(0, 1, 80)
    X = mk.prepare_censored_data(['<3' if a < 3 else a for a in x])
    r90 = mk.segmented_trend_test(X, t, n_breakpoints=1, use_bagging=True, n_bootstrap=40,
                                  random_state=0, alpha=0.10)
    r50 = mk.segmented_trend_test(X, t, n_breakpoints=1, use_bagging=True, n_bootstrap=40,
                                  random_state=0, alpha=0.50)
    w90 = np.diff(r90.breakpoint_cis[0])[0]
    w50 = np.diff(r50.breakpoint_cis[0])[0]
    assert w50 < w90
    mk.plot_segmented_trend(r90, X, t, save_path=str(tmp_path / 'seg.png'))


def test_rolling_numeric_time_with_custom_index():
    X = mk.prepare_censored_data(list(np.arange(40.0)))
    X.index = np.arange(100, 140)
    out = mk.rolling_trend_test(X, np.arange(40.0), window=20, step=10)
    assert len(out) > 0


def test_compare_periods_reports_slope_difference_test():
    g = np.random.default_rng(0)
    t = np.arange(100, dtype=float)
    x = np.where(t < 50, 0.0, 0.3 * (t - 50)) + g.normal(0, 1, 100)
    res = mk.compare_periods(x, t, breakpoint=50)
    assert res['p_value_difference'] < 0.05
    assert res['significant_change']
