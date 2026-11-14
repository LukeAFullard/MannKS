"""
Zhang trend-free pre-whitening (``prewhiten='zhang'``): MannKS reproduces
``zyp::zyp.zhang`` and therefore EGRET's ``pValueAdj``.

* Fixture tests compare with EGRET output stored in ``tests/data/egret_qk``
  (no R needed).
* ``TestZhangLive`` compares with ``zyp`` directly on random series when R,
  zyp and rpy2 are available.
"""
import importlib.util
import io
import os
import warnings

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MannKS import plot_quantile_trend, quantile_trend_test
from MannKS._zhang import _acf1, _kendall_single_precision, zhang_prewhitened_test

_PARITY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                            'validation', '43_Quantile_Kendall', 'egret_parity.py')
_spec = importlib.util.spec_from_file_location('egret_parity_pw', _PARITY_PATH)
egret_parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(egret_parity)

# Kendall::Kendall (the last step of zyp.zhang) is single precision.
_SINGLE = 1e-6


def _ar1(rng, n, rho):
    e = np.empty(n)
    e[0] = rng.normal()
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal()
    return e


class TestZhangAgainstEgretFixtures:

    @pytest.mark.parametrize('scenario', sorted(egret_parity.SCENARIOS))
    @pytest.mark.parametrize('engine', ['trend_test', 'fast'])
    def test_p_adj_reproduces_egret_pvalueadj(self, scenario, engine):
        daily, egret, sc = egret_parity.load_fixture(scenario)
        kw = egret_parity.mannks_settings(sc['paStart'], sc['paLong'])
        assert kw['prewhiten'] == 'zhang'
        res = quantile_trend_test(daily['q'].to_numpy(), pd.DatetimeIndex(daily['date']), engine=engine, **kw)
        ref = egret['results']
        np.testing.assert_allclose(res.results['p_adj'].to_numpy(), ref['pValueAdj'].to_numpy(), atol=_SINGLE)
        # the adjustment is not a no-op in these scenarios ...
        assert np.abs(ref['pValueAdj'].to_numpy() - ref['pValue'].to_numpy()).max() > 1e-3
        # ... and it leaves the unadjusted test untouched
        np.testing.assert_allclose(res.results['p'].to_numpy(), ref['pValue'].to_numpy(), atol=1e-12)
        assert res.results['tau_adj'].notna().all() and res.results['acf1_adj'].notna().all()

    def test_default_is_no_prewhitening(self):
        daily, _, sc = egret_parity.load_fixture('winter_djf')
        kw = egret_parity.mannks_settings(sc['paStart'], sc['paLong'])
        kw.pop('prewhiten')
        res = quantile_trend_test(daily['q'].to_numpy(), pd.DatetimeIndex(daily['date']), **kw)
        assert res.results[['p_adj', 'tau_adj', 'acf1_adj']].isna().all().all()
        res_none = quantile_trend_test(daily['q'].to_numpy(), pd.DatetimeIndex(daily['date']), prewhiten=None, **kw)
        assert res_none.results['p_adj'].isna().all()


class TestZhangUnit:

    def test_acf1_matches_definition(self):
        x = np.random.default_rng(0).normal(size=50)
        d = x - x.mean()
        assert _acf1(x) == pytest.approx(np.sum(d[:-1] * d[1:]) / np.sum(d * d), rel=1e-12)
        assert np.isnan(_acf1(np.full(10, 2.0)))

    def test_no_prewhitening_below_threshold_is_plain_mann_kendall(self):
        # lag-1 autocorrelation < 0.05: zyp tests the original series
        rng = np.random.default_rng(3)
        for _ in range(50):
            y = rng.normal(size=25)
            t = 2000.0 + np.arange(25)
            if _acf1(y) < 0.05:
                break
        out = zhang_prewhitened_test(y, t)
        tau, p = _kendall_single_precision(t, y)
        assert out['n_iter'] == 1 and out['p'] == p and out['tau'] == tau
        assert out['acf1'] == pytest.approx(_acf1(y))

    def test_kendall_normal_approximation_with_ties(self):
        # hand computation: n=6, y has one tie pair
        t = np.arange(6.0)
        y = np.array([1.0, 2.0, 2.0, 3.0, 5.0, 4.0])
        s = 12.0                      # 13 concordant, 1 discordant, 1 tie
        var = (6 * 5 * 17 - 2 * 1 * 9) / 18.0
        from scipy.stats import norm
        p = 2 * (1 - norm.cdf((s - 1) / np.sqrt(var)))
        tau, got = _kendall_single_precision(t, y)
        assert got == pytest.approx(p, rel=1e-12)
        assert tau == pytest.approx(s / np.sqrt(15 * 14))

    def test_single_precision_ties(self):
        # values that differ only beyond single precision are tied, as in Kendall::Kendall
        t = np.arange(6.0)
        y = np.array([1.0, 2.0, 2.0 + 1e-12, 3.0, 5.0, 4.0])
        assert _kendall_single_precision(t, y) == _kendall_single_precision(t, np.array([1.0, 2.0, 2.0, 3.0, 5.0, 4.0]))

    def test_returns_nan_like_zyp(self):
        t = 2000.0 + np.arange(20)
        assert np.isnan(zhang_prewhitened_test(np.full(20, 3.0), t)['p'])        # constant: acf undefined
        y = np.zeros(20)
        y[:2] = 1.0
        assert np.isnan(zhang_prewhitened_test(y, t)['p'])                       # fewer than 3 non-zero values
        assert np.isnan(zhang_prewhitened_test([1.0, 2.0], [0.0, 1.0])['p'])
        y_nan = np.arange(20.0)
        y_nan[3] = np.nan
        assert np.isnan(zhang_prewhitened_test(y_nan, t)['p'])

    def test_zero_slope_iteration_does_not_raise(self):
        # Sen slope of exactly 0 during the iteration: R gives Inf/NaN, not an error
        rng = np.random.default_rng(5)
        y = np.round(_ar1(rng, 40, 0.7))
        out = zhang_prewhitened_test(y, 1990.0 + np.arange(40))
        assert 0.0 <= out['p'] <= 1.0

    def test_time_scale_invariance(self):
        rng = np.random.default_rng(6)
        y = _ar1(rng, 30, 0.5) + 0.1 * np.arange(30)
        a = zhang_prewhitened_test(y, 2000.0 + np.arange(30))
        b = zhang_prewhitened_test(y, np.arange(30.0))
        assert a['p'] == pytest.approx(b['p'], abs=1e-6) and a['tau'] == pytest.approx(b['tau'], abs=1e-6)

    def test_prewhitening_reduces_false_positives_under_ar1(self):
        rng = np.random.default_rng(7)
        n_reps, raw, adj = 300, 0, 0
        t = 1990.0 + np.arange(30)
        for _ in range(n_reps):
            y = _ar1(rng, 30, 0.6)
            raw += _kendall_single_precision(t, y)[1] < 0.05
            adj += zhang_prewhitened_test(y, t)['p'] < 0.05
        assert raw / n_reps > 0.15          # plain Mann-Kendall is badly anti-conservative here
        assert adj < 0.6 * raw              # pre-whitening removes a large part of it


class TestPrewhitenIntegration:

    @pytest.fixture
    def series(self):
        t = pd.date_range('1995-01-01', '2019-12-31', freq='7D')
        return np.random.default_rng(21).lognormal(size=len(t)), t

    def test_validation(self, series):
        x, t = series
        with pytest.raises(ValueError, match="`prewhiten` must be 'none' or 'zhang'"):
            quantile_trend_test(x, t, prewhiten='yue')

    def test_columns_and_agreement_with_direct_call(self, series):
        x, t = series
        res = quantile_trend_test(x, t, quantiles=[0.1, 0.5, 0.9], prewhiten='zhang')
        assert {'p_adj', 'tau_adj', 'acf1_adj'} <= set(res.results.columns)
        years = (res.matrix.index - pd.Timestamp('1970-01-01')).total_seconds().to_numpy() / (365.25 * 86400.0)
        for k in range(3):
            direct = zhang_prewhitened_test(res.matrix.iloc[:, k].to_numpy(), years)
            assert res.results['p_adj'].iloc[k] == direct['p']
            assert res.results['acf1_adj'].iloc[k] == direct['acf1']
        # the unadjusted results are those of a run without pre-whitening
        base = quantile_trend_test(x, t, quantiles=[0.1, 0.5, 0.9])
        pd.testing.assert_frame_equal(res.results.drop(columns=['p_adj', 'tau_adj', 'acf1_adj', 'analysis_notes']),
                                      base.results.drop(columns=['p_adj', 'tau_adj', 'acf1_adj', 'analysis_notes']))

    def test_log_transform_and_numeric_time(self, series):
        x, t = series
        res = quantile_trend_test(x, t, quantiles=[0.5], transform='log', prewhiten='zhang')
        assert 0.0 <= res.results['p_adj'].iloc[0] <= 1.0
        tn = np.arange(len(x), dtype=float)
        res_n = quantile_trend_test(x, tn, block_frequency=52.0, quantiles=[0.5], prewhiten='zhang')
        assert 0.0 <= res_n.results['p_adj'].iloc[0] <= 1.0

    def test_censored_quantiles_get_nan_and_a_note(self, series):
        x, t = series
        cens = x < 0.4
        df = pd.DataFrame({'value': np.where(cens, 0.4, x), 'censored': cens,
                           'cen_type': np.where(cens, 'lt', 'not')})
        res = quantile_trend_test(df, t, quantiles=[0.05, 0.9], prewhiten='zhang')
        assert np.isnan(res.results['p_adj'].iloc[0])
        assert 0.0 <= res.results['p_adj'].iloc[1] <= 1.0
        assert any("prewhiten='zhang'" in n and 'censored' in n for n in res.analysis_notes)

    def test_plot_color_by_p_adj(self, series):
        x, t = series
        res = quantile_trend_test(x, t, quantiles=7, prewhiten='zhang')
        fig, ax = plt.subplots()
        try:
            plot_quantile_trend(res, ax=ax, color_by='p_adj')
            labels = ax.get_legend_handles_labels()[1]
            assert labels and all('adjusted p' in lab for lab in labels)
        finally:
            plt.close(fig)
        with pytest.raises(ValueError, match="requires results computed with `prewhiten='zhang'`"):
            plot_quantile_trend(quantile_trend_test(x, t, quantiles=7), save_path=io.BytesIO(), color_by='p_adj')


def _zyp_available():
    try:
        import rpy2.robjects as ro
        ro.r('suppressPackageStartupMessages(library(zyp))')
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _zyp_available(), reason="R with zyp and rpy2 not available")
class TestZhangLive:
    """Direct comparison with zyp::zyp.zhang on random series."""

    def test_matches_zyp_on_random_series(self):
        import rpy2.robjects as ro
        zyp_zhang = ro.r('function(y, t) zyp.zhang(y, t)')
        rng = np.random.default_rng(0)
        n_compared = n_na = n_iterated = 0
        for trial in range(600):
            n = int(rng.integers(5, 60))
            t = 2000.0 + np.arange(n) + rng.uniform(-0.3, 0.3, n) * (trial % 3 == 0)
            y = _ar1(rng, n, rng.choice([0.0, 0.3, 0.6, 0.9, -0.4])) + rng.choice([0.0, 0.05, 0.3]) * np.arange(n)
            if trial % 4 == 0:
                y = np.round(y, 1)                 # ties
            if trial % 50 == 0:
                y = np.zeros(n)
                y[:2] = 1.0                        # zyp returns NA
            if trial % 77 == 0:
                y = np.full(n, 3.0)                # constant: NA
            r = np.array(zyp_zhang(ro.FloatVector(y), ro.FloatVector(t)), dtype=float)
            # zyp order: lbound, trend, trendp, ubound, tau, sig, nruns, autocor, ...
            r_na = np.isnan(r[5]) or r[5] < -1e9
            got = zhang_prewhitened_test(y, t)
            assert np.isnan(got['p']) == r_na
            if r_na:
                n_na += 1
                continue
            n_compared += 1
            n_iterated += r[6] > 1
            assert got['p'] == pytest.approx(r[5], abs=_SINGLE)
            assert got['tau'] == pytest.approx(r[4], abs=_SINGLE)
            assert got['trend'] == pytest.approx(r[1], rel=1e-9, abs=1e-12)
            assert got['acf1'] == pytest.approx(r[7], abs=1e-9)
            assert got['n_iter'] == r[6]
        assert n_compared > 500 and n_na > 10 and n_iterated > 100
