"""Censoring tests for MannKS.decompose (brief v2 section 5, test 3)."""
import sys
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


def _frame(values, limits, cens, kind='lt'):
    """MannKS censored DataFrame from values, limits and a mask."""
    return pd.DataFrame({
        'value': np.where(cens, limits, values),
        'censored': cens,
        'cen_type': np.where(cens, kind, 'not').astype(object),
    })


def _amp(res):
    return res.seasonal_amplitude


def _design_two_dl(seed):
    """Section 5, test 3(a): ~30% left-censored, two detection limits."""
    rng = np.random.default_rng(seed)
    t = np.sort(2000 + rng.uniform(0, 15, 180))
    x = -1.0 + 0.03 * (t - 2000) + 0.8 * np.cos(2 * np.pi * t) + rng.normal(0, 0.25, t.size)
    early = t < 2007.5
    dl = np.where(early, np.quantile(x[early], 0.35), np.quantile(x[~early], 0.25))
    cens = x < dl
    return t, x, dl, cens


def _design_trough(seed):
    """Section 5, test 3(b): ~50% censored, concentrated in the seasonal trough."""
    rng = np.random.default_rng(100 + seed)
    t = np.sort(2000 + rng.uniform(0, 15, 180))
    x = -1.0 + 0.03 * (t - 2000) + 0.8 * np.cos(2 * np.pi * t) + rng.normal(0, 0.25, t.size)
    dl = np.where(t < 2007.5, -0.6, -0.9)
    cens = x < dl
    return t, x, dl, cens


# ---------------------------------------------------------------- 3(a), 3(c)
@pytest.mark.parametrize("method", ["parametric", pytest.param("ros", marks=pytest.mark.slow)])
def test_two_detection_limits_amplitude_and_limits(method):
    pytest.importorskip("ndimpute")
    ratios, shares = [], []
    for seed in range(10):
        t, x, dl, cens = _design_two_dl(seed)
        shares.append(cens.mean())
        truth = _quiet(MannKS.decompose, np.exp(x), t, n_harmonics=1)
        res = _quiet(MannKS.decompose, _frame(np.exp(x), np.exp(dl), cens), t,
                     n_harmonics=1, censored_method=method)
        c = res.censored
        assert np.all(res.fitted_values[c] <= res.values[c] + 1e-12)
        ratios.append(_amp(res) / _amp(truth))
    assert 0.25 < np.mean(shares) < 0.35
    assert abs(np.mean(ratios) - 1) < 0.10


# ---------------------------------------------------------------- 3(b)
@pytest.mark.slow
def test_censored_trough_discriminates_conditional_from_marginal():
    pytest.importorskip("ndimpute")
    r_rob, r_non, r_ctrl, shares = [], [], [], []
    for seed in range(10):
        t, x, dl, cens = _design_trough(seed)
        shares.append(cens.mean())
        xf = _frame(np.exp(x), np.exp(dl), cens)
        truth = _quiet(MannKS.decompose, np.exp(x), t, n_harmonics=1)
        a0 = _amp(truth)
        r_rob.append(_amp(_quiet(MannKS.decompose, xf, t, n_harmonics=1, robust=True)) / a0)
        r_non.append(_amp(_quiet(MannKS.decompose, xf, t, n_harmonics=1, robust=False)) / a0)
        r_ctrl.append(_amp(_quiet(MannKS.decompose, xf, t, n_harmonics=1, robust=False,
                                  impute_max_iter=0)) / a0)
    print(f"censored {np.mean(shares):.2f}; ratios robust {np.mean(r_rob):.3f}, "
          f"non-robust {np.mean(r_non):.3f}, marginal control {np.mean(r_ctrl):.3f}")
    assert 0.40 < np.mean(shares) < 0.60
    assert abs(np.mean(r_rob) - 1) < 0.07
    assert abs(np.mean(r_non) - 1) < 0.07
    assert np.mean(r_ctrl) < 0.85


# ---------------------------------------------------------------- 3(d)
def test_right_censored_case():
    pytest.importorskip("ndimpute")
    rng = np.random.default_rng(21)
    t = np.sort(2000 + rng.uniform(0, 12, 200))
    x = 0.02 * (t - 2000) + 0.6 * np.cos(2 * np.pi * t) + rng.normal(0, 0.25, t.size)
    rl = np.quantile(x, 0.9)
    cens = x > rl
    res = _quiet(MannKS.decompose, _frame(np.exp(x), np.exp(rl), cens, kind='gt'), t)
    assert np.all(res.cen_type[res.censored] == 'gt')
    assert np.all(res.fitted_values[res.censored] >= res.values[res.censored] - 1e-12)
    xd, _ = res.deseasonalized_series()
    np.testing.assert_allclose(xd['value'][res.censored], (rl - res.seasonal)[res.censored],
                               atol=1e-12)
    assert np.all(xd['cen_type'][res.censored] == 'gt')


# ---------------------------------------------------------------- 3(e)
def test_output_censoring_and_trend_test_integration():
    pytest.importorskip("ndimpute")
    t, x, dl, cens = _design_two_dl(3)
    res = _quiet(MannKS.decompose, _frame(np.exp(x), np.exp(dl), cens), t, n_harmonics=1)
    xd, td = res.deseasonalized_series()
    xo, to = res.deseasonalized_series('original')
    c = res.censored
    np.testing.assert_array_equal(xd['censored'].to_numpy(), c)
    np.testing.assert_array_equal(xd['cen_type'].to_numpy()[c], 'lt')
    np.testing.assert_allclose(xd['value'][c], (np.log(np.exp(dl[res.order])) - res.seasonal)[c],
                               atol=1e-12)
    np.testing.assert_allclose(xo['value'][c], (np.exp(dl[res.order]) * np.exp(-res.seasonal))[c],
                               rtol=1e-12)
    r1 = _quiet(MannKS.deseasonalized_trend_test, res)
    r2 = _quiet(MannKS.deseasonalized_trend_test, res, scale='original')
    assert np.isfinite(r1.p) and r1.p == pytest.approx(r2.p, abs=1e-12)
    assert r1.s == r2.s
    with pytest.raises(ValueError, match="sens_slope_method='ats'"):
        MannKS.deseasonalized_trend_test(res, sens_slope_method='lwp')


# ---------------------------------------------------------------- 3(f)
def test_only_model_based_imputation_is_used(monkeypatch):
    nd = pytest.importorskip("ndimpute")
    calls = []
    real = nd.impute

    def spy(values, status=None, method='ros', **kw):
        calls.append(dict(kw, method=method))
        return real(values, status=status, method=method, **kw)

    monkeypatch.setattr(nd, "impute", spy)
    t, x, dl, cens = _design_two_dl(4)
    _quiet(MannKS.decompose, _frame(np.exp(x), np.exp(dl), cens), t, n_boot=2, random_state=1)
    _quiet(MannKS.decompose, _frame(np.exp(x), np.exp(dl), cens), t, censored_method='ros',
           n_imputations=2, random_state=1)
    assert calls
    for kw in calls:
        assert kw['method'] in ('parametric', 'ros')
        assert 'lwp_compat' not in kw and 'strategy' not in kw and 'multiplier' not in kw
        assert kw['dist'] == 'normal'


# ---------------------------------------------------------------- 3(g)
def test_missing_ndimpute_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "ndimpute", None)
    t, x, dl, cens = _design_two_dl(5)
    with pytest.raises(ImportError, match=r"mannks\[censored\]"):
        _quiet(MannKS.decompose, _frame(np.exp(x), np.exp(dl), cens), t)
    res = _quiet(MannKS.decompose, np.exp(x), t)  # uncensored still works
    assert np.all(np.isfinite(res.trend))


def test_heavy_censoring_validation():
    rng = np.random.default_rng(30)
    t = np.sort(2000 + rng.uniform(0, 10, 200))
    x = rng.normal(0, 1, t.size)
    cens = x < np.quantile(x, 0.85)
    with pytest.raises(ValueError, match='censored'):
        MannKS.decompose(_frame(np.exp(x), np.exp(np.quantile(x, 0.85)), cens), t)


def test_mixed_censoring_ros_converges():
    """ROS can enter a period-2 imputation cycle (rank ordering); it must settle."""
    pytest.importorskip("ndimpute")
    rng = np.random.default_rng(3)
    t = np.sort(2000 + rng.uniform(0, 12, 200))
    x = 0.02 * (t - 2000) + 0.6 * np.cos(2 * np.pi * t) + rng.normal(0, 0.25, t.size)
    lo, hi = np.quantile(x, 0.15), np.quantile(x, 0.92)
    cens = (x < lo) | (x > hi)
    df = pd.DataFrame({'value': np.exp(np.clip(x, lo, hi)), 'censored': cens,
                       'cen_type': np.where(x < lo, 'lt', np.where(x > hi, 'gt', 'not')).astype(object)})
    res = _quiet(MannKS.decompose, df, t, censored_method='ros', random_state=1)
    assert all(res.diagnostics['convergence']['impute_converged'])
    assert not any('did not converge' in m for m in res.warnings)
    lt = res.cen_type == 'lt'
    gt = res.cen_type == 'gt'
    assert np.all(res.fitted_values[lt] <= res.values[lt] + 1e-12)
    assert np.all(res.fitted_values[gt] >= res.values[gt] - 1e-12)
