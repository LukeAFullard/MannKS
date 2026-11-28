"""Time-varying and automatic seasonal modes (brief v2 section 5, test 5)."""
import warnings

import numpy as np
import pytest

import MannKS
from MannKS import _decompose as d


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _monthly(years, rng, start=2000):
    return np.array([start + y + (m + 0.5) / 12 + rng.uniform(-5, 5) / 365.2425
                     for y in range(years) for m in range(12)])


def test_time_varying_recovers_growing_amplitude():
    rng = np.random.default_rng(1)
    t = _monthly(20, rng)
    amp = 0.3 + 0.6 * (t - t[0]) / (t[-1] - t[0])
    y = 0.02 * (t - 2000) + amp * np.cos(2 * np.pi * t) + rng.normal(0, 0.1, t.size)
    res = _quiet(MannKS.decompose, y, t, transform='none', n_harmonics=1,
                 seasonal='time_varying', seasonal_window_years=7.0)
    assert res.seasonal_mode_used == 'time_varying'
    grid, coef = res.coefficients['grid'], res.coefficients['coef']
    true_amp = 0.3 + 0.6 * (grid - t[0]) / (t[-1] - t[0])
    est_amp = np.hypot(coef[:, 0], coef[:, 1])
    interior = (grid > t[0] + 3.5) & (grid < t[-1] - 3.5)
    rel = np.abs(est_amp[interior] / true_amp[interior] - 1)
    assert rel.max() < 0.15
    assert res.seasonal_by_year is not None and len(res.seasonal_by_year) >= 20


def test_time_varying_falls_back_to_global_when_window_cannot_support_K():
    rng = np.random.default_rng(2)
    # 8 years, 3 samples a year at fixed phases: coverage supports only K = 1,
    # so with K = 2 every window widens to the record span (= global mode).
    t = np.sort(np.array([2000 + y + p + rng.uniform(-10, 10) / 365.2425
                          for y in range(8) for p in (0.1, 0.43, 0.77)]))
    y = 0.02 * (t - 2000) + 0.5 * np.cos(2 * np.pi * t) + rng.normal(0, 0.1, t.size)
    g = _quiet(MannKS.decompose, y, t, transform='none', n_harmonics=2, seasonal='global')
    tv = _quiet(MannKS.decompose, y, t, transform='none', n_harmonics=2,
                seasonal='time_varying', seasonal_window_years=7.0)
    assert tv.settings['seasonal_model'].capped.all()
    np.testing.assert_allclose(tv.seasonal, g.seasonal, atol=1e-8)
    np.testing.assert_allclose(tv.trend, g.trend, atol=1e-8)
    assert any('shorter than 2 x' in m for m in tv.warnings)


@pytest.mark.slow
def test_auto_prefers_global_for_constant_seasonality():
    chosen = []
    for seed in range(10):
        rng = np.random.default_rng(50 + seed)
        t = _monthly(16, rng)
        y = 0.02 * (t - 2000) + d.harmonics(t, 2) @ np.array([0.5, 0.3, 0.2, -0.1]) \
            + rng.normal(0, 0.3, t.size)
        res = _quiet(MannKS.decompose, y, t, transform='none', seasonal='auto',
                     n_boot_auto=49, random_state=seed)
        assert 0 < res.seasonal_constancy_pvalue <= 1
        assert res.diagnostics['auto']['n_failed'] == 0
        chosen.append(res.seasonal_mode_used)
    assert chosen.count('global') >= 8


def test_auto_detects_strongly_changing_seasonality():
    rng = np.random.default_rng(7)
    t = _monthly(16, rng)
    amp = np.where(t < 2008, 0.2, 1.2)
    y = amp * np.cos(2 * np.pi * t) + rng.normal(0, 0.15, t.size)
    res = _quiet(MannKS.decompose, y, t, transform='none', n_harmonics=1, seasonal='auto',
                 n_boot_auto=39, random_state=1)
    assert res.seasonal_constancy_pvalue == pytest.approx(1 / 40)
    assert res.seasonal_mode_used == 'time_varying'


def test_auto_warns_when_bootstrap_too_small_to_select():
    rng = np.random.default_rng(8)
    t = _monthly(8, rng)
    y = np.cos(2 * np.pi * t) + rng.normal(0, 0.2, t.size)
    with pytest.warns(UserWarning, match='n_boot_auto'):
        MannKS.decompose(y, t, transform='none', n_harmonics=1, seasonal='auto',
                         n_boot_auto=9, random_state=0)
