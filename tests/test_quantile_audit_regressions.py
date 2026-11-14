"""
Regression tests for defects found in the v0.7.0 Quantile-Kendall audit:

* C1 - ``equalize='aggregate'`` inverted the sampling-drift artifact (medians of
  different numbers of observations are not comparable between blocks).
* M1 - an unaligned ``block_origin`` silently dropped the observations that
  preceded the first block edge.
* M3 - timezone-aware timestamps were blocked and month-filtered in UTC.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from MannKS import quantile_trend_test
from MannKS._quantile import _assign_blocks, _block_edges, _equalize_blocks


def _block_df(dates, values):
    dates = pd.DatetimeIndex(dates)
    return pd.DataFrame({
        't': (dates - pd.Timestamp('1970-01-01')).total_seconds().to_numpy(),
        't_original': dates,
        'value': np.asarray(values, dtype=float),
        'censored': False,
        'cen_type': 'not',
    })


def _drift_series(rng, seasonal=False):
    """Stationary process sampled monthly for 10 years, then weekly for 10 years."""
    t_list = []
    for yr in range(2000, 2010):
        t_list.extend(pd.date_range(f"{yr}-01-15", f"{yr}-12-15", periods=12))
    for yr in range(2010, 2020):
        t_list.extend(pd.date_range(f"{yr}-01-05", f"{yr}-12-25", periods=52))
    t = pd.DatetimeIndex(t_list)
    if seasonal:
        x = 1.5 * np.sin(2.0 * np.pi * t.dayofyear / 365.25) + rng.normal(size=len(t))
    else:
        x = rng.lognormal(2.0, 0.4, size=len(t))
    return x, t


# ==============================================================================
# C1: equalize='aggregate'
# ==============================================================================

class TestAggregateCommonSupport:

    def test_mixed_frequency_blocks_use_support_one(self):
        rng = np.random.default_rng(0)
        monthly = _block_df(pd.date_range('2001-01-15', '2001-12-15', periods=12), rng.normal(size=12))
        weekly = _block_df(pd.date_range('2011-01-05', '2011-12-25', periods=52), rng.normal(size=52))
        info = {}
        out = _equalize_blocks([monthly, weekly], 'aggregate', is_datetime=True, info=info)
        assert info['aggregate_support'] == 1
        assert [len(o) for o in out] == [12, 12]
        # With support 1 nothing is averaged: every value is an actual observation,
        # so both blocks keep the single-observation sampling distribution.
        assert set(out[0]['value']) <= set(monthly['value'])
        assert set(out[1]['value']) <= set(weekly['value'])
        # one observation per calendar month
        assert sorted(pd.DatetimeIndex(out[1]['t_original']).month) == list(range(1, 13))

    def test_uniform_weekly_blocks_keep_true_medians(self):
        rng = np.random.default_rng(1)
        blocks = [_block_df(pd.date_range(f'{y}-01-03', periods=52, freq='7D'), rng.normal(size=52))
                  for y in (2001, 2002, 2003)]
        info = {}
        out = _equalize_blocks(blocks, 'aggregate', is_datetime=True, info=info)
        assert info['aggregate_support'] == 4
        assert all(len(o) == 12 for o in out)
        # every value is the median of exactly 4 observations of its month
        b = blocks[0]
        months = pd.DatetimeIndex(b['t_original']).month
        for m, val in zip(range(1, 13), out[0]['value']):
            vals = b['value'][months == m].to_numpy()
            n = len(vals)
            idx = np.floor((np.arange(4) + 0.5) * n / 4).astype(int)
            assert val == pytest.approx(np.median(vals[idx]))

    def test_every_subinterval_uses_the_same_number_of_observations(self):
        # 5 observations in one month must not give that month a tighter median.
        rng = np.random.default_rng(2)
        dates = pd.date_range('2001-01-01', '2001-12-31', freq='D')
        blk = _block_df(dates, rng.normal(size=len(dates)))
        sparse = _block_df(pd.date_range('2002-01-02', periods=52, freq='7D'), rng.normal(size=52))
        info = {}
        _equalize_blocks([blk, sparse], 'aggregate', is_datetime=True, info=info)
        assert info['aggregate_support'] == 4  # smallest weekly month, not 28-31 daily values

    def test_numeric_aggregate_has_exactly_twelve_bins(self):
        df = pd.DataFrame({'t': np.arange(100.0), 't_original': np.arange(100.0),
                           'value': np.arange(100.0), 'censored': False, 'cen_type': 'not'})
        out = _equalize_blocks([df], 'aggregate', is_datetime=False)
        assert len(out[0]) == 12

    def test_result_reports_support(self):
        x, t = _drift_series(np.random.default_rng(3))
        res = quantile_trend_test(x, t, equalize='aggregate', engine='fast')
        assert res.sampling_diagnostic['aggregate_support'] == 1
        assert res.sampling_diagnostic['flag'] is False
        assert any("equalize='aggregate'" in n and 'stratified thinning' in n for n in res.analysis_notes)
        assert res.n_ranks == 12

    @pytest.mark.parametrize('seasonal', [False, True])
    def test_aggregate_is_calibrated_under_sampling_drift(self, seasonal):
        """
        Stationary process, monthly -> weekly sampling.  Before the fix the lowest
        rank was 'increasing' in ~80% and the highest 'decreasing' in ~73% of
        replicates at alpha = 0.05.
        """
        rng = np.random.default_rng(20260 + int(seasonal))
        n_reps = 80
        rejections = 0
        signed = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for _ in range(n_reps):
                x, t = _drift_series(rng, seasonal=seasonal)
                df = quantile_trend_test(x, t, equalize='aggregate', engine='fast').results
                rejections += int(df['h'].iloc[0]) + int(df['h'].iloc[-1])
                signed += float(df['Tau'].iloc[0]) - float(df['Tau'].iloc[-1])
        # 160 tests at alpha=0.05 -> 8 expected; P(X > 20 | p=0.05) < 1e-4.  Old code: ~120.
        assert rejections <= 20
        # no systematic "spread is shrinking" signature (old code: mean ~ +0.45 per tail)
        assert abs(signed / n_reps) < 0.15

    def test_thin_and_aggregate_agree_on_no_trend(self):
        x, t = _drift_series(np.random.default_rng(11))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = quantile_trend_test(x, t, engine='fast')
            agg = quantile_trend_test(x, t, equalize='aggregate', engine='fast')
        assert raw.sampling_diagnostic['flag'] is True
        assert raw.sampling_diagnostic['ratio_max_min'] > 4
        assert agg.sampling_diagnostic['ratio_max_min'] == 1.0


# ==============================================================================
# M1: block_origin must never drop observations
# ==============================================================================

class TestBlockOriginKeepsAllObservations:

    @pytest.fixture
    def weekly(self):
        t = pd.date_range('2000-03-01', '2019-12-31', freq='7D')
        x = np.random.default_rng(5).lognormal(1.0, 0.5, len(t))
        return x, t

    @pytest.mark.parametrize('origin', ['2000-02-01', '1999-06-15', '2000-03-15', '2000-04-01',
                                        '2030-01-01', '1950-11-30'])
    def test_all_observations_are_assigned(self, weekly, origin):
        x, t = weekly
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = quantile_trend_test(x, t, block_frequency='YS-APR', block_origin=origin,
                                      min_size_per_block=1, min_coverage=0.0, quantiles=[0.5])
        assert res.blocks['n_obs'].sum() == len(x)  # was 1030 of 1035 for origin='2000-02-01'
        assert pd.Timestamp(res.blocks['block_start'].iloc[0]) == pd.Timestamp('1999-04-01')

    @pytest.mark.parametrize('freq', ['YE', 'YS-APR', 'YE-SEP', '5YE', 'QE', 'ME', '2QS-FEB', 'W', '10D',
                                      pd.Timedelta('36h')])
    @pytest.mark.parametrize('origin', [None, '1999-06-15', '2000-03-02', '2000-03-01', '2004-07-04 05:00',
                                        '2031-01-01'])
    def test_edges_span_the_data(self, freq, origin):
        t_min, t_max = pd.Timestamp('2000-03-01'), pd.Timestamp('2019-12-31 18:00')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            edges = pd.DatetimeIndex(_block_edges(t_min, t_max, freq, origin, is_datetime=True))
        assert edges[0] <= t_min
        assert edges[-1] > t_max
        assert edges.is_monotonic_increasing and edges.is_unique

    def test_unaligned_origin_warns_and_aligned_does_not(self, weekly):
        x, t = weekly
        kw = dict(block_frequency='YS-APR', min_size_per_block=1, min_coverage=0.0, quantiles=[0.5])
        with pytest.warns(UserWarning, match="`block_origin` .* is not aligned with block_frequency 'YS-APR'"):
            quantile_trend_test(x, t, block_origin='2000-02-01', **kw)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            quantile_trend_test(x, t, block_origin='1998-04-01', **kw)
        assert not [m for m in w if 'block_origin' in str(m.message)]

    def test_origin_selects_phase_of_multi_year_blocks(self, weekly):
        x, t = weekly
        kw = dict(block_frequency='5YE', min_size_per_block=1, min_coverage=0.0, min_blocks=2, quantiles=[0.5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            res = quantile_trend_test(x, t, block_origin='1999-01-01', **kw)
        assert not [m for m in w if 'block_origin' in str(m.message)]
        assert [str(s)[:10] for s in res.blocks['block_start']] == [
            '1999-01-01', '2004-01-01', '2009-01-01', '2014-01-01', '2019-01-01']
        assert res.blocks['n_obs'].sum() == len(x)

    def test_fixed_length_origin_is_honoured_exactly(self, weekly):
        x, t = weekly
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            res = quantile_trend_test(x, t, block_frequency=pd.Timedelta('365D'), block_origin='1998-07-01',
                                      min_size_per_block=1, min_coverage=0.0, quantiles=[0.5])
        assert not [m for m in w if 'block_origin' in str(m.message)]
        assert pd.Timestamp(res.blocks['block_start'].iloc[0]) == pd.Timestamp('1999-07-01')
        assert res.blocks['n_obs'].sum() == len(x)

    def test_assign_blocks_refuses_to_drop_data(self, monkeypatch):
        # If the edges ever fail to span the data, fail loudly instead of dropping rows.
        import MannKS._quantile as qmod
        t = pd.date_range('2000-01-01', periods=40, freq='30D')
        data = _block_df(t, np.arange(40.0))
        bad_edges = pd.date_range('2001-01-01', '2004-01-01', freq='YS').to_numpy()
        monkeypatch.setattr(qmod, '_block_edges', lambda *a, **k: bad_edges)
        with pytest.raises(RuntimeError, match="fall outside the block edges"):
            _assign_blocks(data, 'YE', None, None, 1, 0.0, False, True)

    @pytest.mark.parametrize('origin', [None, 0.0, 75.0, 49.0, 1e6, -3.3])
    def test_numeric_origin_keeps_all_observations(self, origin):
        t = np.arange(1000.0) + 50.0
        x = np.random.default_rng(6).normal(size=1000)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = quantile_trend_test(x, t, block_frequency=100, block_origin=origin,
                                      min_size_per_block=1, min_coverage=0.0, quantiles=[0.5])
        assert res.blocks['n_obs'].sum() == 1000


# ==============================================================================
# M3: timezone-aware timestamps are analysed in local wall-clock time
# ==============================================================================

class TestTimezoneAwareLocalBlocks:

    @pytest.fixture
    def nz(self):
        t = pd.date_range('2000-01-01', '2019-12-31 23:00', freq='12h', tz='Pacific/Auckland')
        x = np.random.default_rng(8).normal(size=len(t))
        return x, t

    def test_blocks_start_at_local_midnight(self, nz):
        x, t = nz
        res = quantile_trend_test(x, t, quantiles=[0.1, 0.5, 0.9])
        # was: first block 1999-01-01 (UTC) holding 2 observations
        assert pd.Timestamp(res.blocks['block_start'].iloc[0]) == pd.Timestamp('2000-01-01')
        assert res.blocks['n_obs'].tolist()[:3] == [732, 730, 730]
        assert len(res.blocks) == 20 and res.blocks['valid'].all()
        assert res.matrix.index.tz is None
        assert any('local wall-clock time' in n and 'Pacific/Auckland' in n for n in res.analysis_notes)

    def test_equals_analysis_of_naive_local_times(self, nz):
        x, t = nz
        aware = quantile_trend_test(x, t, quantiles=[0.1, 0.5, 0.9]).results.drop(columns='analysis_notes')
        naive = quantile_trend_test(x, t.tz_localize(None), quantiles=[0.1, 0.5, 0.9]).results.drop(columns='analysis_notes')
        pd.testing.assert_frame_equal(aware, naive)

    @pytest.mark.parametrize('container', ['index', 'series', 'list', 'pydatetime', 'object_array'])
    def test_all_input_containers(self, nz, container):
        x, t = nz
        t_in = {'index': t, 'series': pd.Series(t), 'list': list(t),
                'pydatetime': t.to_pydatetime(), 'object_array': np.asarray(t, dtype=object)}[container]
        res = quantile_trend_test(x, t_in, quantiles=[0.5])
        assert pd.Timestamp(res.blocks['block_start'].iloc[0]) == pd.Timestamp('2000-01-01')
        assert res.blocks['n_obs'].iloc[0] == 732

    def test_months_filter_uses_local_month(self):
        # 06:00 on the 1st of each month in Auckland is still the previous month in UTC.
        days = pd.date_range('2000-01-01', '2011-12-31', freq='D')
        t = (days + pd.Timedelta(hours=6)).tz_localize('Pacific/Auckland')
        x = np.random.default_rng(9).normal(size=len(t))
        res = quantile_trend_test(x, t, months=[2], quantiles=[0.5], min_size_per_block=5)
        feb_days = [29 if pd.Timestamp(f'{y}-01-01').is_leap_year else 28 for y in range(2000, 2012)]
        assert res.blocks['n_obs'].tolist() == feb_days  # UTC months would give 28/29 +- 1 shifted days

    def test_timezone_aware_origin(self, nz):
        x, t = nz
        for origin in (pd.Timestamp('2000-01-01', tz='Pacific/Auckland'),
                       pd.Timestamp('1999-12-31 11:00', tz='UTC')):  # the same instant
            res = quantile_trend_test(x, t, quantiles=[0.5], block_frequency=pd.Timedelta('365D'),
                                      block_origin=origin, min_coverage=0.5)
            assert pd.Timestamp(res.blocks['block_start'].iloc[0]) == pd.Timestamp('2000-01-01')

    def test_utc_data_unchanged(self):
        t = pd.date_range('2000-01-01', '2012-12-31', freq='D', tz='UTC')
        x = np.random.default_rng(10).normal(size=len(t))
        a = quantile_trend_test(x, t, quantiles=[0.5]).results.drop(columns='analysis_notes')
        b = quantile_trend_test(x, t.tz_localize(None), quantiles=[0.5]).results.drop(columns='analysis_notes')
        pd.testing.assert_frame_equal(a, b)

    def test_equalize_and_plot_with_timezone(self, nz, tmp_path):
        x, t = nz
        res = quantile_trend_test(x, t, quantiles=5, equalize='aggregate', field_significance=True,
                                  n_permutations=20, random_state=1,
                                  plot_path=str(tmp_path / 'tz.png'))
        assert (tmp_path / 'tz.png').exists()
        assert res.sampling_diagnostic['aggregate_support'] >= 1


# ==============================================================================
# Minor / Nit findings
# ==============================================================================

import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MannKS import plot_quantile_trend
from MannKS._quantile import _auto_u_grid, _standardize_block

_ALL_METHODS = ['hazen', 'linear', 'weibull', 'median_unbiased', 'normal_unbiased',
                'interpolated_inverted_cdf', 'inverted_cdf', 'averaged_inverted_cdf',
                'closest_observation', 'lower', 'higher', 'nearest', 'midpoint']
_CONTINUOUS = _ALL_METHODS[:6]


class TestAutoGridMatchesQuantileMethod:

    @pytest.mark.parametrize('method', _CONTINUOUS)
    @pytest.mark.parametrize('n', [2, 5, 12, 365])
    def test_block_of_target_size_is_returned_unchanged(self, method, n):
        # The default grid must be the one that belongs to `quantile_method`
        # (it was linspace(0, 1, M), which is only right for 'linear').
        vals = np.sort(np.random.default_rng(n).lognormal(size=n))
        df = pd.DataFrame({'value': vals[::-1].copy()})
        got, cens, _ = _standardize_block(df, n, 'interpolate', method, 'propagate', 0.5, 1.1)
        np.testing.assert_allclose(got, vals, rtol=1e-12, atol=1e-12)
        assert not cens.any()

    def test_grids(self):
        np.testing.assert_allclose(_auto_u_grid(4, 'hazen'), [0.125, 0.375, 0.625, 0.875])
        np.testing.assert_allclose(_auto_u_grid(4, 'linear'), [0.0, 1 / 3, 2 / 3, 1.0])
        np.testing.assert_allclose(_auto_u_grid(4, 'weibull'), [0.2, 0.4, 0.6, 0.8])
        np.testing.assert_allclose(_auto_u_grid(1, 'hazen'), [0.5])

    def test_end_to_end_matrix_holds_observations_of_smallest_blocks(self):
        t = pd.date_range('2001-01-01', '2012-12-31', freq='D')
        x = np.random.default_rng(1).lognormal(size=len(t))
        res = quantile_trend_test(x, t)  # defaults: 'auto', 'min', 'hazen'
        assert res.n_ranks == 365
        np.testing.assert_allclose(res.results['quantile'], (np.arange(1, 366) - 0.5) / 365)
        first = np.sort(x[t.year == 2001])
        np.testing.assert_allclose(res.matrix.iloc[0].to_numpy(), first, rtol=1e-12)

    def test_one_censored_observation_flags_one_rank(self):
        vals = np.arange(1.0, 13.0)
        cens = np.zeros(12, dtype=bool)
        cens[0] = True
        df = pd.DataFrame({'value': vals, 'censored': cens, 'cen_type': np.where(cens, 'lt', 'not')})
        _, sc, st = _standardize_block(df, 12, 'interpolate', 'hazen', 'propagate', 0.5, 1.1)
        assert sc.tolist() == [True] + [False] * 11  # was two flagged ranks with the linspace grid


class TestCensorPropagationAllMethods:

    @pytest.mark.parametrize('method', _ALL_METHODS)
    def test_flag_iff_a_censored_order_statistic_contributes(self, method):
        """A quantile is censored exactly when perturbing a censored observation changes it."""
        rng = np.random.default_rng(abs(hash(method)) % 2**32)
        for _ in range(150):
            n = int(rng.integers(2, 25))
            M = int(rng.integers(2, 30))
            vals = np.cumsum(rng.uniform(1.0, 2.0, n))
            k = int(rng.integers(0, n + 1))
            if rng.random() < 0.5:   # left-censored prefix
                cens = np.arange(n) < k
                ctype = np.where(cens, 'lt', 'not')
            else:                    # right-censored suffix
                cens = np.arange(n) >= n - k
                ctype = np.where(cens, 'gt', 'not')
            u = np.linspace(0.0, 1.0, M)
            df = pd.DataFrame({'value': vals, 'censored': cens, 'cen_type': ctype.astype(object)})
            _, sc, _ = _standardize_block(df, M, 'interpolate', method, 'propagate', 0.5, 1.1, u_grid=u)
            base = np.quantile(vals, u, method=method)
            depends = np.zeros(M, dtype=bool)
            for i in np.flatnonzero(cens):
                v2 = vals.copy()
                v2[i] += 0.25
                depends |= ~np.isclose(np.quantile(v2, u, method=method), base, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(sc, depends)

    def test_mixed_lt_gt_bracket_is_reported(self):
        df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0], 'censored': [False, True, True, False],
                           'cen_type': ['not', 'lt', 'gt', 'not']})
        info = {}
        _, sc, st = _standardize_block(df, 3, 'interpolate', 'hazen', 'propagate', 0.5, 1.1,
                                       u_grid=np.array([0.25, 0.5, 0.75]), info=info)
        assert st.tolist() == ['lt', 'lt', 'gt']
        assert info['n_mixed_brackets'] == 1
        # the EGRET merge uses the same precedence ('lt' first) and reports it too
        df2 = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0], 'censored': [False, True, True, False],
                            'cen_type': ['not', 'gt', 'lt', 'not']})
        info2 = {}
        _, _, st2 = _standardize_block(df2, 3, 'egret', 'hazen', 'propagate', 0.5, 1.1, info=info2)
        assert st2.tolist() == ['not', 'lt', 'not']
        assert info2['n_mixed_brackets'] == 1

    def test_mixed_bracket_warns_end_to_end(self):
        t = pd.date_range('2000-01-01', '2011-12-31', freq='7D')
        x = np.random.default_rng(4).lognormal(size=len(t))
        lo, hi = x < 0.9, x > 1.1   # the median sits between a '<' and a '>' value
        df = pd.DataFrame({'value': np.where(lo, 0.9, np.where(hi, 1.1, x)), 'censored': lo | hi,
                           'cen_type': np.where(lo, 'lt', np.where(hi, 'gt', 'not'))})
        df = df[lo | hi].reset_index(drop=True)
        with pytest.warns(UserWarning, match="left-censored .* and a right-censored"):
            res = quantile_trend_test(df, t[lo | hi], quantiles=[0.5], min_coverage=0.5)
        assert any('not a valid bound' in n for n in res.analysis_notes)


class TestValidationAndWarnings:

    @pytest.fixture
    def series(self):
        t = pd.date_range('2000-01-01', '2011-12-31', freq='7D')
        return np.random.default_rng(12).lognormal(size=len(t)), t

    def test_input_validation(self, series):
        x, t = series
        with pytest.raises(ValueError, match="must not contain duplicate"):
            quantile_trend_test(x, t, quantiles=[0.5, 0.1, 0.5])
        with pytest.raises(ValueError, match="`min_coverage` must be a number between"):
            quantile_trend_test(x, t, min_coverage='a')
        with pytest.raises(ValueError, match="array-like sequences, not scalars"):
            quantile_trend_test(1.0, 2.0)
        for std in ('interpolate', 'egret'):   # also validated when 'egret' ignores it
            with pytest.raises(ValueError, match="Invalid `quantile_method` 'bogus'"):
                quantile_trend_test(x, t, quantile_method='bogus', standardization=std)

    def test_warning_categories_are_preserved(self, series):
        x, t = series
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            quantile_trend_test(x, t, quantiles=[0.5], sens_slope_method='nan')
        cats = {m.category for m in w if 'deprecated' in str(m.message)}
        assert cats and UserWarning not in cats   # was re-emitted as UserWarning

    def test_warnings_survive_an_exception(self):
        t = pd.date_range('2000-01-01', '2011-12-31', freq='7D')
        x = np.random.default_rng(13).lognormal(size=len(t))
        x[::40] = np.inf   # "Removed ... infinite" warning, then a failure
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            with pytest.raises(ValueError, match="requires all standardized block values to be > 0"):
                quantile_trend_test(-x, t, transform='log')
        assert any('infinite value' in str(m.message) for m in w)

    def test_fast_engine_fallback_is_noted(self, series):
        x, t = series
        res = quantile_trend_test(x - 3.0, t, quantiles=[0.1, 0.9], transform='pct_of_median', engine='fast')
        assert res.computation_mode == 'exact'
        assert any("engine='fast' not applicable" in n and 'NaN' in n for n in res.analysis_notes)
        ok = quantile_trend_test(x, t, quantiles=[0.1, 0.9], engine='fast')
        assert ok.computation_mode == 'fast'
        assert not any("engine='fast'" in n for n in ok.analysis_notes)

    def test_log_slope_units_describe_slope_and_plot_labels(self, series):
        x, t = series
        res = quantile_trend_test(x, t, quantiles=5, transform='log', x_unit='mg/L')
        assert (res.results['slope_units'] == 'ln(mg/L) per year').all()
        fig, ax = plt.subplots()
        try:
            plot_quantile_trend(res, ax=ax, show_fdc=True)
            assert ax.get_ylabel() == 'Trend (% per year)'
            assert fig.axes[-1].get_ylabel() == 'Median Concentration / Flow (mg/L)'
        finally:
            plt.close(fig)

    def test_caller_figure_not_closed_when_saving(self, series):
        x, t = series
        res = quantile_trend_test(x, t, quantiles=5)
        fig, ax = plt.subplots()
        try:
            plot_quantile_trend(res, ax=ax, save_path=io.BytesIO())
            assert plt.fignum_exists(fig.number)
        finally:
            plt.close(fig)

    def test_minimum_attainable_p_values_quoted_in_docs(self):
        # Docs: strictly monotone series give p = 0.089 (B=4), 0.027 (B=5), 0.0085 (B=6).
        expected = {4: 0.0894, 5: 0.0275, 6: 0.0085}
        for B, p_min in expected.items():
            t = pd.date_range('2000-01-01', periods=B * 365, freq='D')
            x = np.repeat(np.arange(B), 365) + np.random.default_rng(0).random(B * 365) * 0.1
            res = quantile_trend_test(x, t, quantiles=[0.5], min_blocks=2, block_frequency=pd.Timedelta('365D'))
            assert len(res.matrix) == B
            assert res.results['p'].iloc[0] == pytest.approx(p_min, abs=2e-4)
