"""
Unit tests for core blocking, normalization, coverage, and assignment routines.

Target module: MannKS._quantile
Functions:
    - _parse_block_frequency
    - _normalize_start_freq
    - _block_edges
    - _block_coverage
    - _assign_blocks
"""
import numpy as np
import pandas as pd
import pytest

from MannKS._quantile import (
    _parse_block_frequency,
    _normalize_start_freq,
    _block_edges,
    _block_coverage,
    _assign_blocks,
)


class TestParseBlockFrequency:
    """Tests for _parse_block_frequency for datetime and numeric data."""

    def test_datetime_standard_frequencies(self):
        # Annual frequencies
        off_ye = _parse_block_frequency('YE', is_datetime=True)
        code = getattr(off_ye, 'rule_code', 'YE')
        assert code in ('YE', 'Y', 'A', 'YE-DEC')
        
        off_ys_oct = _parse_block_frequency('YS-OCT', is_datetime=True)
        assert getattr(off_ys_oct, 'month', None) == 10 or 'OCT' in str(off_ys_oct)

        # Multi-period frequency
        off_5ye = _parse_block_frequency('5YE', is_datetime=True)
        assert off_5ye.n == 5

        # Quarterly and Monthly
        off_qe = _parse_block_frequency('QE', is_datetime=True)
        assert off_qe.n == 1
        off_ms = _parse_block_frequency('MS', is_datetime=True)
        assert off_ms.n == 1

    def test_datetime_invalid_frequencies_raise(self):
        with pytest.raises(ValueError, match="Invalid block_frequency 'not_a_frequency'"):
            _parse_block_frequency('not_a_frequency', is_datetime=True)

        with pytest.raises(ValueError, match="Invalid block_frequency"):
            _parse_block_frequency(12345, is_datetime=True)

    def test_numeric_valid_widths(self):
        w1 = _parse_block_frequency(1.0, is_datetime=False)
        assert w1 == 1.0
        w2 = _parse_block_frequency('2.5', is_datetime=False)
        assert w2 == 2.5
        w3 = _parse_block_frequency(10, is_datetime=False)
        assert w3 == 10.0

    def test_numeric_invalid_widths_raise(self):
        with pytest.raises(ValueError, match="Must be a positive number"):
            _parse_block_frequency(0, is_datetime=False)

        with pytest.raises(ValueError, match="Must be a positive number"):
            _parse_block_frequency(-2.5, is_datetime=False)

        with pytest.raises(ValueError, match="Invalid block_frequency 'bad_str'"):
            _parse_block_frequency('bad_str', is_datetime=False)


class TestNormalizeStartFreq:
    """Tests for _normalize_start_freq start-period alignment."""

    @pytest.mark.parametrize("input_str,expected", [
        ('YE', 'YS'),
        ('Y', 'YS'),
        ('A', 'YS'),
        ('5YE', '5YS'),
        ('YE-SEP', 'YS-OCT'),
        ('YE-DEC', 'YS-JAN'),
        ('YE-JAN', 'YS-FEB'),
        ('YE-MAR', 'YS-APR'),
        ('5YE-SEP', '5YS-OCT'),
        ('QE', 'QS'),
        ('Q', 'QS'),
        ('2QE', '2QS'),
        ('QE-NOV', 'QS-DEC'),
        ('ME', 'MS'),
        ('M', 'MS'),
        ('3ME', '3MS'),
        ('YS', 'YS'),
        ('YS-OCT', 'YS-OCT'),
        ('MS', 'MS'),
    ])
    def test_normalize_start_freq_mappings(self, input_str, expected):
        norm = _normalize_start_freq(input_str)
        assert norm == expected

    def test_normalize_start_freq_with_pandas_offset_object(self):
        try:
            from pandas.tseries.offsets import to_offset
        except ImportError:
            from pandas.tseries.frequencies import to_offset
        off = to_offset('YE-SEP')
        norm = _normalize_start_freq(off)
        assert norm == 'YS-OCT'

    def test_normalize_start_freq_unrecognized_returns_str(self):
        res = _normalize_start_freq('CUSTOM-OFFSET')
        assert res == 'CUSTOM-OFFSET'


class TestBlockEdges:
    """Tests for _block_edges calculation and boundary stability."""

    def test_datetime_annual_edges(self):
        t_min = pd.Timestamp('2010-03-15')
        t_max = pd.Timestamp('2014-07-20')
        edges = _block_edges(t_min, t_max, block_frequency='YE', block_origin=None, is_datetime=True)
        # start_freq for 'YE' normalizes to 'YS', origin rolled back to '2010-01-01'
        assert pd.Timestamp(edges[0]) == pd.Timestamp('2010-01-01')
        assert pd.Timestamp(edges[-1]) >= t_max
        assert len(edges) >= 5

    def test_datetime_water_year_edges(self):
        # Water year: Oct 1 to Sep 30
        t_min = pd.Timestamp('2010-05-01')
        t_max = pd.Timestamp('2012-11-01')
        edges = _block_edges(t_min, t_max, block_frequency='YE-SEP', block_origin=None, is_datetime=True)
        # YE-SEP normalizes to YS-OCT; origin rollback from 2010-05-01 should be 2009-10-01
        assert pd.Timestamp(edges[0]) == pd.Timestamp('2009-10-01')
        assert pd.Timestamp(edges[1]) == pd.Timestamp('2010-10-01')
        assert pd.Timestamp(edges[-1]) >= t_max

    def test_datetime_custom_origin(self):
        t_min = pd.Timestamp('2010-05-01')
        t_max = pd.Timestamp('2012-11-01')
        custom_origin = pd.Timestamp('2010-01-01')
        edges = _block_edges(t_min, t_max, block_frequency='YE', block_origin=custom_origin, is_datetime=True)
        assert pd.Timestamp(edges[0]) == custom_origin

    def test_datetime_multi_period_rollback_stability(self):
        # Trimming 10 days of data does not shift 5-year block edges
        t_min_full = pd.Timestamp('2000-01-01')
        t_min_trimmed = pd.Timestamp('2000-01-11')
        t_max = pd.Timestamp('2019-12-31')

        edges_full = _block_edges(t_min_full, t_max, block_frequency='5YE', block_origin=None, is_datetime=True)
        edges_trimmed = _block_edges(t_min_trimmed, t_max, block_frequency='5YE', block_origin=None, is_datetime=True)

        assert pd.Timestamp(edges_full[0]) == pd.Timestamp(edges_trimmed[0])
        assert len(edges_full) == len(edges_trimmed)
        assert np.array_equal(edges_full, edges_trimmed)

    def test_datetime_tz_handling(self):
        # Tz-aware input should be converted to naive without error
        t_min = pd.Timestamp('2010-01-01 00:00:00+02:00')
        t_max = pd.Timestamp('2012-01-01 00:00:00+02:00')
        edges = _block_edges(t_min, t_max, block_frequency='YE', block_origin=None, is_datetime=True)
        assert pd.Timestamp(edges[0]).tzinfo is None

    def test_numeric_edges(self):
        t_min = 1.2
        t_max = 8.7
        width = 2.0
        edges = _block_edges(t_min, t_max, block_frequency=width, block_origin=None, is_datetime=False)
        # origin = floor(1.2 / 2.0) * 2.0 = 0.0
        assert edges[0] == 0.0
        assert edges[-1] >= t_max
        assert np.allclose(np.diff(edges), 2.0)

    def test_numeric_custom_origin(self):
        t_min = 5.2
        t_max = 14.8
        width = 3.0
        edges = _block_edges(t_min, t_max, block_frequency=width, block_origin=5.0, is_datetime=False)
        assert edges[0] == 5.0
        assert np.allclose(np.diff(edges), 3.0)


class TestBlockCoverage:
    """Tests for _block_coverage temporal data span calculation."""

    def test_empty_observations_return_zero(self):
        cov = _block_coverage('2010-01-01', '2011-01-01', [], is_datetime=True)
        assert cov == 0.0

        cov_num = _block_coverage(0.0, 10.0, [], is_datetime=False)
        assert cov_num == 0.0

    def test_datetime_full_year_coverage(self):
        dates = pd.date_range('2010-01-01', '2010-12-31', freq='D')
        cov = _block_coverage('2010-01-01', '2011-01-01', dates, is_datetime=True)
        assert cov >= 0.99 and cov <= 1.0

    def test_datetime_partial_season_coverage(self):
        # Only Summer months (June, July, August = ~92 days in 365 days -> ~0.25)
        dates = pd.date_range('2010-06-01', '2010-08-31', freq='D')
        cov = _block_coverage('2010-01-01', '2011-01-01', dates, is_datetime=True)
        assert cov < 0.30

    def test_datetime_with_months_subset(self):
        # Subset to summer months [6, 7, 8]. Full coverage within summer months.
        dates = pd.date_range('2010-06-01', '2010-08-31', freq='D')
        cov = _block_coverage('2010-01-01', '2011-01-01', dates, is_datetime=True, months=[6, 7, 8])
        assert cov >= 0.95

    def test_numeric_coverage(self):
        # Spanning [1.0, 9.0] with step 1.0 in block [0.0, 10.0]:
        # span = (9.0 - 1.0) + 1.0 = 9.0, duration = 10.0 -> coverage = 0.90
        t_obs = np.arange(1.0, 10.0, 1.0)
        cov = _block_coverage(0.0, 10.0, t_obs, is_datetime=False)
        assert np.isclose(cov, 0.9)

    def test_datetime_sub_daily_hourly_coverage(self):
        """Verify that sub-daily (hourly) data calculates correct median spacing and fractional coverage."""
        t_hourly = pd.date_range('2020-01-01 00:00:00', '2020-01-01 11:00:00', freq='h')
        cov = _block_coverage('2020-01-01 00:00:00', '2020-01-02 00:00:00', t_hourly, is_datetime=True)
        assert np.isclose(cov, 0.50, atol=0.01)



class TestAssignBlocks:
    """Tests for _assign_blocks grouping, validation, and filtering."""

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=['t', 'value', 't_original'])
        groups, summary, centers = _assign_blocks(
            data=df, block_frequency='YE', block_origin=None, months=None,
            min_size_per_block=10, min_coverage=0.9, require_complete=False,
            is_datetime=True
        )
        assert len(groups) == 0
        assert summary.empty
        assert len(centers) == 0

    def test_assign_blocks_half_open_interval_boundary(self):
        # Half-open interval: [start, end).
        # An observation exactly at the edge should go to the later block.
        dates = pd.to_datetime(['2010-01-01', '2011-01-01', '2012-01-01'])
        vals = [10.0, 20.0, 30.0]
        data = pd.DataFrame({'t': dates, 't_original': dates, 'value': vals})

        groups, summary, centers = _assign_blocks(
            data=data, block_frequency='YE', block_origin=pd.Timestamp('2010-01-01'),
            months=None, min_size_per_block=1, min_coverage=0.0, require_complete=False,
            is_datetime=True
        )
        # Block 0: [2010-01-01, 2011-01-01) -> contains '2010-01-01'
        # Block 1: [2011-01-01, 2012-01-01) -> contains '2011-01-01'
        # Block 2: [2012-01-01, 2013-01-01) -> contains '2012-01-01'
        assert len(groups) == 3
        assert pd.Timestamp(groups[0]['t_original'].iloc[0]) == pd.Timestamp('2010-01-01')
        assert pd.Timestamp(groups[1]['t_original'].iloc[0]) == pd.Timestamp('2011-01-01')
        assert pd.Timestamp(groups[2]['t_original'].iloc[0]) == pd.Timestamp('2012-01-01')

    def test_filter_by_min_size_per_block(self):
        dates_b1 = pd.date_range('2010-01-01', periods=15, freq='D')
        dates_b2 = pd.date_range('2011-01-01', periods=5, freq='D')
        dates = dates_b1.union(dates_b2)
        data = pd.DataFrame({'t': dates, 't_original': dates, 'value': np.arange(len(dates))})

        groups, summary, centers = _assign_blocks(
            data=data, block_frequency='YE', block_origin=pd.Timestamp('2010-01-01'),
            months=None, min_size_per_block=10, min_coverage=0.0, require_complete=False,
            is_datetime=True
        )
        assert len(groups) == 1
        assert len(groups[0]) == 15
        assert summary.loc[summary['block_start'] == pd.Timestamp('2011-01-01'), 'valid'].values[0] == False
        assert "below min_size_per_block" in summary.loc[summary['block_start'] == pd.Timestamp('2011-01-01'), 'reason'].values[0]

    def test_filter_by_min_coverage(self):
        # 15 samples clustered in 2 weeks (Jan 1 to Jan 15) of an annual block -> coverage ~ 15/365 < 0.1
        dates = pd.date_range('2010-01-01', periods=15, freq='D')
        data = pd.DataFrame({'t': dates, 't_original': dates, 'value': np.arange(len(dates))})

        groups, summary, centers = _assign_blocks(
            data=data, block_frequency='YE', block_origin=pd.Timestamp('2010-01-01'),
            months=None, min_size_per_block=10, min_coverage=0.8, require_complete=False,
            is_datetime=True
        )
        assert len(groups) == 0
        assert not summary.empty
        assert "coverage" in summary['reason'].iloc[0]

    def test_require_complete_rejection(self):
        # Leap year 2012 has 366 days. Provide 365 daily observations (1 missing).
        dates_2012 = pd.date_range('2012-01-01', '2012-12-30', freq='D')  # missing Dec 31
        data = pd.DataFrame({'t': dates_2012, 't_original': dates_2012, 'value': np.arange(len(dates_2012))})

        groups, summary, centers = _assign_blocks(
            data=data, block_frequency='YE', block_origin=pd.Timestamp('2012-01-01'),
            months=None, min_size_per_block=10, min_coverage=0.9, require_complete=True,
            is_datetime=True
        )
        assert len(groups) == 0
        assert "incomplete block" in summary['reason'].iloc[0]

    def test_assign_blocks_numeric(self):
        t_vals = np.array([1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0])
        data = pd.DataFrame({'t': t_vals, 'value': t_vals * 2.0})

        groups, summary, centers = _assign_blocks(
            data=data, block_frequency=5.0, block_origin=0.0,
            months=None, min_size_per_block=3, min_coverage=0.5, require_complete=False,
            is_datetime=False
        )
        assert len(groups) == 2
        assert len(groups[0]) == 3  # [1, 2, 3] in [0, 5)
        assert len(groups[1]) == 4  # [6, 7, 8, 9] in [5, 10)
        assert np.isclose(centers[0], 2.5)
        assert np.isclose(centers[1], 7.5)
