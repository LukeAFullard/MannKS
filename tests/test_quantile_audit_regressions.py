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
