"""Core tests for MannKS.decompose (brief v2 section 5: tests 1, 2, 4, 6, 9)."""
import time
import warnings

import numpy as np
import pandas as pd
import pytest

import MannKS
from MannKS import _decompose as d


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _seasonal2(t):
    return d.harmonics(t, 2) @ np.array([0.5, 0.3, 0.2, -0.1])


# ---------------------------------------------------------------- test 1
@pytest.mark.parametrize("robust", [False, True])
def test_exact_recovery_noiseless(robust):
    rng = np.random.default_rng(1)
    t = np.sort(2000 + rng.uniform(0, 15, 180))
    trend = 0.05 * (t - 2000)
    seas = _seasonal2(t)
    res = _quiet(MannKS.decompose, trend + seas, t, transform='none', n_harmonics=2,
                 robust=robust, random_state=0)
    assert np.max(np.abs(res.trend - trend)) < 1e-6
    assert np.max(np.abs(res.seasonal - seas)) < 1e-6
    assert np.all(res.weights == 1.0)
    if robust:
        assert 'robustness skipped: residual scale is approximately zero' in res.analysis_notes


# ---------------------------------------------------------------- test 2
def test_robustness_outliers():
    ratios, zero_share = [], []
    rob_err, non_err = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        t = np.sort(2000 + rng.uniform(0, 15, 180))
        tt = 0.05 * (t - 2000) + 0.2 * np.sin(2 * np.pi * (t - 2000) / 7)
        sig = 0.3
        y = tt + _seasonal2(t) + rng.normal(0, sig, t.size)
        out = rng.random(t.size) < 0.10
        y[out] -= 8 * sig
        r1 = _quiet(MannKS.decompose, y, t, transform='none', n_harmonics=2, robust=True)
        r0 = _quiet(MannKS.decompose, y, t, transform='none', n_harmonics=2, robust=False)
        rob_err.append(np.sqrt(np.mean((r1.trend - tt) ** 2)))
        non_err.append(np.sqrt(np.mean((r0.trend - tt) ** 2)))
        zero_share.append((np.sum(r1.weights[out] == 0), out.sum()))
    assert np.mean(non_err) >= 3 * np.mean(rob_err)
    z = np.array(zero_share)
    assert z[:, 0].sum() / z[:, 1].sum() >= 0.95


# ---------------------------------------------------------------- test 4
def _storm_design(rng, years=15):
    t = []
    for y in range(years):
        for m in range(12):
            t.append(2000 + y + (m + 0.5) / 12 + rng.uniform(-5, 5) / 365.2425)
        for _ in range(6):
            start = 2000 + y + rng.uniform(0, 1)
            t.extend(start + np.sort(rng.uniform(0, 3, rng.integers(3, 7))) / 365.2425)
    return np.sort(np.array(t))


def test_irregular_storm_bursts():
    rng = np.random.default_rng(3)
    t = _storm_design(rng)
    tt = 0.03 * (t - 2000)
    sig = 0.3
    y = tt + _seasonal2(t) + rng.normal(0, sig, t.size)
    res = _quiet(MannKS.decompose, y, t, transform='none')
    for arr in (res.trend, res.seasonal, res.remainder, res.weights):
        assert np.all(np.isfinite(arr))
    assert np.max(np.abs(res.trend - tt)) < 4 * sig


def test_irregular_winter_sparse_reduces_K():
    rng = np.random.default_rng(4)
    t = []
    for y in range(15):
        for m in range(2, 11):  # March .. November only
            t.append(2000 + y + (m + 0.5) / 12 + rng.uniform(-5, 5) / 365.2425)
    t = np.sort(np.array(t))
    sig = 0.3
    y = 0.03 * (t - 2000) + _seasonal2(t) + rng.normal(0, sig, t.size)
    res = _quiet(MannKS.decompose, y, t, transform='none')
    assert res.K_max < 4
    assert res.n_harmonics_used <= res.K_max
    assert np.all(np.isfinite(res.trend))
    assert np.max(np.abs(res.trend - 0.03 * (t - 2000))) < 4 * sig
    assert any('Phase coverage limits' in m for m in res.warnings)
    # a single isolated winter sample must not lift the cap
    t2 = np.sort(np.append(t, 2007.03))
    y2 = 0.03 * (t2 - 2000) + _seasonal2(t2) + np.random.default_rng(5).normal(0, sig, t2.size)
    res2 = _quiet(MannKS.decompose, y2, t2, transform='none')
    assert res2.K_max == res.K_max


def test_singular_window_identical_timestamps():
    rng = np.random.default_rng(6)
    t = np.concatenate([2000 + np.arange(0, 6, 1 / 12), np.full(12, 2008.0),
                        2010 + np.arange(0, 6, 1 / 12)])
    y = 0.02 * (t - 2000) + _seasonal2(t) + rng.normal(0, 0.2, t.size)
    res = _quiet(MannKS.decompose, y, t, transform='none', n_harmonics=2)
    assert np.all(np.isfinite(res.trend))
    assert res.diagnostics['n_trend_widenings'] > 0
    assert 'tied timestamps present without aggregation' in res.analysis_notes


# ---------------------------------------------------------------- test 6
def _censored_frame(seed=0, n=200):
    rng = np.random.default_rng(seed)
    t = np.sort(2000 + rng.uniform(0, 12, n))
    y = np.exp(0.02 * (t - 2000) + _seasonal2(t) + rng.normal(0, 0.3, n))
    dl = np.quantile(y, 0.25)
    x = MannKS.prepare_censored_data([f"<{dl:.5f}" if v < dl else v for v in y])
    return x, t


def test_reproducibility_and_global_rng_untouched():
    pytest.importorskip("ndimpute")
    x, t = _censored_frame()
    state = np.random.get_state()
    a = _quiet(MannKS.decompose, x, t, n_boot=4, random_state=11)
    b = _quiet(MannKS.decompose, x, t, n_boot=4, random_state=11)
    for name in ('trend_low', 'trend_high', 'seasonal_low', 'seasonal_high',
                 'trend_simul_low', 'trend_simul_high'):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    c = _quiet(MannKS.decompose, x, t, n_imputations=3, random_state=5)
    e = _quiet(MannKS.decompose, x, t, n_imputations=3, random_state=5)
    np.testing.assert_array_equal(c.trend_imputation_sd, e.trend_imputation_sd)
    np.testing.assert_array_equal(c.seasonal_imputation_sd, e.seasonal_imputation_sd)
    f = _quiet(MannKS.decompose, x, t, seasonal='auto', n_boot_auto=5, random_state=2)
    g = _quiet(MannKS.decompose, x, t, seasonal='auto', n_boot_auto=5, random_state=2)
    assert f.seasonal_constancy_pvalue == g.seasonal_constancy_pvalue
    after = np.random.get_state()
    assert after[0] == state[0]
    np.testing.assert_array_equal(after[1], state[1])
    assert after[2:] == state[2:]


# ---------------------------------------------------------------- predict / output
def test_predict_and_dataframe_datetime():
    rng = np.random.default_rng(8)
    t = np.sort(2000 + rng.uniform(0, 10, 150))
    dt = pd.to_datetime('2000-01-01') + pd.to_timedelta((t - 2000) * 365.2425, unit='D')
    y = np.exp(0.02 * (t - 2000) + _seasonal2(t) + rng.normal(0, 0.2, t.size))
    res = _quiet(MannKS.decompose, y, dt)
    np.testing.assert_allclose(res.predict(res.times), res.trend + res.seasonal, atol=1e-10)
    with pytest.warns(UserWarning, match='outside the data span'):
        out = res.predict(pd.to_datetime(['1990-01-01']))
    assert np.isnan(out[0])
    with pytest.raises(TypeError):
        res.predict([2005.0])
    df = res.to_dataframe()
    assert len(df) == t.size and {'trend', 'seasonal', 'remainder', 'weight'} <= set(df.columns)
    assert 1 <= res.peak_day_of_year <= 366
    assert res.seasonal_amplitude_ratio == pytest.approx(np.exp(res.seasonal_amplitude))
    x_ds, t_ds = res.deseasonalized_series()
    assert list(x_ds.columns) == ['value', 'censored', 'cen_type']
    assert isinstance(t_ds, pd.DatetimeIndex)
    r = _quiet(MannKS.trend_test, x_ds, t_ds, slope_scaling='year')
    assert np.isfinite(r.slope)


def test_order_maps_back_to_input():
    rng = np.random.default_rng(9)
    t = 2000 + rng.uniform(0, 10, 150)   # unsorted input
    y = np.exp(_seasonal2(t) + rng.normal(0, 0.2, t.size))
    y[3] = np.nan                         # dropped silently
    res = _quiet(MannKS.decompose, y, t)
    assert 3 not in res.order
    np.testing.assert_array_equal(res.times, t[res.order])
    np.testing.assert_allclose(res.values, np.log(y[res.order]))
    assert np.all(np.diff(res.t_years) >= 0)


def test_plot_returns_and_saves(tmp_path):
    rng = np.random.default_rng(10)
    t = np.sort(2000 + rng.uniform(0, 8, 120))
    y = np.exp(_seasonal2(t) + rng.normal(0, 0.2, t.size))
    res = _quiet(MannKS.decompose, y, t)
    fig, axes = res.plot()
    assert len(axes) == 5
    import matplotlib.pyplot as plt
    plt.close(fig)
    p = tmp_path / 'decomp.png'
    assert res.plot(save_path=p) is None
    assert p.exists()


# ---------------------------------------------------------------- test 9
def test_performance_global_n600():
    rng = np.random.default_rng(12)
    t = np.sort(2000 + rng.uniform(0, 20, 600))
    y = np.exp(0.03 * (t - 2000) + _seasonal2(t) + rng.normal(0, 0.3, t.size))
    _quiet(MannKS.decompose, y, t)  # warm-up
    t0 = time.perf_counter()
    _quiet(MannKS.decompose, y, t)
    elapsed = time.perf_counter() - t0
    print(f"decompose global n=600: {elapsed:.3f} s (target 0.3 s)")
    assert elapsed < 1.0
