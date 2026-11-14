"""
Unit tests for quantile_trend_test orchestrator, analytical trend scenarios,
non-positive median handling, warning aggregation, and QuantileTrendResult methods.

Target module: MannKS.quantile_trend_test
Functions:
    - quantile_trend_test
    - QuantileTrendResult
    - summarize_quantile_regimes
"""
import warnings
import numpy as np
import pandas as pd
import pytest

from MannKS.quantile_trend_test import (
    quantile_trend_test,
    QuantileTrendResult,
    summarize_quantile_regimes,
)


class TestInputValidationAndKwargs:
    """Tests for argument validation, rejection of conflicting kwargs, and input types."""

    def test_rejected_kwargs_raise(self):
        dates = pd.date_range('2010-01-01', periods=100, freq='D')
        vals = np.arange(100, dtype=float)
        with pytest.raises(ValueError, match="Arguments.*not supported in `quantile_trend_test`"):
            quantile_trend_test(vals, dates, agg_method='median')

        with pytest.raises(ValueError, match="Arguments.*not supported in `quantile_trend_test`"):
            quantile_trend_test(vals, dates, min_size=5)

    def test_unknown_kwargs_raise(self):
        dates = pd.date_range('2010-01-01', periods=100, freq='D')
        vals = np.arange(100, dtype=float)
        with pytest.raises(ValueError, match="Unknown keyword argument"):
            quantile_trend_test(vals, dates, non_existent_kwarg=42)

    def test_invalid_alpha_raises(self):
        dates = pd.date_range('2010-01-01', periods=100, freq='D')
        vals = np.arange(100, dtype=float)
        with pytest.raises(ValueError, match="Significance level `alpha` must be between 0 and 1"):
            quantile_trend_test(vals, dates, alpha=1.5)

    def test_undocumented_aliases_are_rejected(self):
        # `min_samples` / `classification` were never part of a released API.
        dates = pd.date_range('2010-01-01', periods=5 * 365, freq='D')
        vals = np.arange(len(dates), dtype=float)
        with pytest.raises(ValueError, match="Unknown keyword argument"):
            quantile_trend_test(vals, dates, min_samples=20, min_blocks=3)
        with pytest.raises(ValueError, match="Unknown keyword argument"):
            quantile_trend_test(vals, dates, classification=False)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Input vectors `x` and `t` must have the same length"):
            quantile_trend_test([1, 2, 3], pd.date_range('2010-01-01', periods=4))

    def test_invalid_options_raise(self):
        dates = pd.date_range('2010-01-01', periods=6 * 365, freq='D')
        vals = np.arange(len(dates), dtype=float)

        # quantiles as invalid string
        with pytest.raises(ValueError, match="String value for `quantiles` must be 'auto'"):
            quantile_trend_test(vals, dates, quantiles='invalid_str')

        # quantiles as invalid integer (0 or negative)
        with pytest.raises(ValueError, match="Integer `quantiles` must be between 1 and"):
            quantile_trend_test(vals, dates, quantiles=0)

        # quantiles as invalid sequence (out of [0, 1])
        with pytest.raises(ValueError, match="`quantiles` sequence must be a non-empty 1D sequence in"):
            quantile_trend_test(vals, dates, quantiles=[-0.1, 0.5])

        # target_length invalid
        with pytest.raises(ValueError, match="`target_length` must be 'min', 'median', 'max', or an integer >= 2"):
            quantile_trend_test(vals, dates, target_length='invalid')

        # standardization invalid
        with pytest.raises(ValueError, match="`standardization` must be 'interpolate' or 'egret'"):
            quantile_trend_test(vals, dates, standardization='invalid')

        # equalize invalid
        with pytest.raises(ValueError, match="`equalize` must be 'none', 'thin', or 'aggregate'"):
            quantile_trend_test(vals, dates, equalize='invalid')

        # censor_quantile_method invalid
        with pytest.raises(ValueError, match="`censor_quantile_method` must be 'propagate', 'substitute', or 'error'"):
            quantile_trend_test(vals, dates, censor_quantile_method='invalid')

        # transform invalid
        with pytest.raises(ValueError, match="`transform` must be 'none', 'log', or 'pct_of_median'"):
            quantile_trend_test(vals, dates, transform='invalid')

        # engine invalid
        with pytest.raises(ValueError, match="`engine` must be 'trend_test' or 'fast'"):
            quantile_trend_test(vals, dates, engine='invalid')

        # standardization='egret' with non-auto quantiles
        with pytest.raises(ValueError, match="`standardization='egret'` requires `quantiles='auto'`"):
            quantile_trend_test(vals, dates, standardization='egret', quantiles=4)
        with pytest.raises(ValueError, match="`standardization='egret'` requires `quantiles='auto'`"):
            quantile_trend_test(vals, dates, standardization='egret', quantiles=[0.25, 0.75])


class TestWarningEmissionsAndAggregation:
    """Tests for warning emission to caller and deduplication/aggregation."""

    def test_warning_on_insufficient_data(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = quantile_trend_test([1.0], pd.to_datetime(['2020-01-01']))
        assert len(w) > 0
        assert any("Insufficient data" in str(msg.message) for msg in w)
        assert len(res.warnings) > 0
        assert any("Insufficient data" in msg for msg in res.warnings)

    def test_warning_on_insufficient_valid_blocks(self):
        # 10 days of data with annual frequency yields 0 or 1 valid block < min_blocks=5
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        vals = np.arange(10, dtype=float)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = quantile_trend_test(vals, dates, block_frequency='YE', min_blocks=5)
        assert len(w) > 0
        assert any("Insufficient valid blocks" in str(msg.message) for msg in w)
        assert any("Insufficient valid blocks" in msg for msg in res.warnings)
        assert res.computation_mode == 'insufficient'
        assert res.to_frame().empty

    def test_cross_year_warning_behavior(self):
        dates = pd.date_range('2010-01-01', periods=10 * 365, freq='D')
        rng = np.random.default_rng(42)
        vals = rng.normal(10, 1, len(dates))

        # Aligned: winter season [11, 12, 1, 2] anchored to November YS-NOV -> NO warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            quantile_trend_test(
                vals, dates, months=[11, 12, 1, 2], block_frequency='YS-NOV',
                min_size_per_block=20, min_blocks=5
            )
        assert not any("Cross-year" in str(msg.message) for msg in w)

        # Unaligned: winter season [11, 12, 1, 2] with calendar year YE -> emits warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            quantile_trend_test(
                vals, dates, months=[11, 12, 1, 2], block_frequency='YE',
                min_size_per_block=20, min_blocks=5
            )
        assert any("Cross-year" in str(msg.message) for msg in w)

    def test_multi_quantile_warning_aggregation(self):
        dates = pd.to_datetime(['2010-01-01', '2010-06-01', '2011-01-01', '2011-06-01', '2012-01-01', '2012-06-01'])
        vals = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = quantile_trend_test(
                vals, dates, block_frequency='YE', quantiles=3,
                min_blocks=3, min_size_per_block=2, min_coverage=0.0
            )
        assert any("In 3 of 3 quantiles:" in str(item.message) for item in w)
        assert any("In 3 of 3 quantiles:" in msg for msg in res.warnings)


class TestNonPositiveMedians:
    """Tests for handling non-positive medians in slope percentage and CIs."""

    def test_strictly_negative_data_sets_slope_pct_to_nan(self):
        # 6 years of negative data with a positive upward slope
        years = 6
        dates = pd.date_range('2010-01-01', periods=years * 365, freq='D')
        # Range [-30, -5], upward slope
        vals = -30.0 + (np.arange(len(dates)) / len(dates)) * 25.0

        res = quantile_trend_test(
            vals, dates, block_frequency='YE', min_size_per_block=50, min_blocks=5
        )
        df = res.to_frame()
        assert not df.empty
        # Raw slope must be strictly positive
        assert (df['slope'] > 0.0).all()
        # Because medians are <= 0, slope_pct and its CIs must be NaN (preventing inversion)
        assert np.isnan(df['slope_pct'].to_numpy()).all()
        assert np.isnan(df['lower_ci_pct'].to_numpy()).all()
        assert np.isnan(df['upper_ci_pct'].to_numpy()).all()

    def test_pct_of_median_with_negative_medians(self):
        years = 6
        dates = pd.date_range('2010-01-01', periods=years * 365, freq='D')
        vals = -20.0 + (np.arange(len(dates)) / len(dates)) * 5.0

        res = quantile_trend_test(
            vals, dates, block_frequency='YE', transform='pct_of_median',
            min_size_per_block=50, min_blocks=5
        )
        df = res.to_frame()
        assert np.isnan(df['slope_pct'].to_numpy()).all()
        assert any("encountered non-positive quantile median" in note for note in res.analysis_notes)


class TestAnalyticalScenarios:
    """Tests for known analytical scenarios: location shift, scale growth, multiplicative trend."""

    def test_uniform_location_shift(self):
        # In a uniform location shift x_t = x_0 + beta * t, all quantiles have slope == beta.
        years = 12
        dates = pd.date_range('2000-01-01', periods=years * 365, freq='D')
        beta_per_year = 2.5
        t_years = (dates - dates[0]).total_seconds() / (365.25 * 86400.0)
        
        rng = np.random.default_rng(1234)
        noise = rng.normal(0, 1.0, len(dates))
        vals = 100.0 + beta_per_year * t_years + noise

        res = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=5,
            slope_scaling='year', min_size_per_block=50, min_blocks=10
        )
        df = res.to_frame()
        assert len(df) == 5
        # All quantile slopes should be approximately equal to 2.5
        for s in df['slope']:
            assert np.isclose(s, beta_per_year, atol=0.5)

    def test_scale_expansion(self):
        # Scale growth: mean is constant at 0, but variance grows with time: sigma(t) = 1 + alpha * t
        # Lower quantiles decrease (negative slope), median is ~0, upper quantiles increase (positive slope).
        years = 15
        dates = pd.date_range('2000-01-01', periods=years * 365, freq='D')
        t_years = (dates - dates[0]).total_seconds() / (365.25 * 86400.0)
        sigma = 1.0 + 0.3 * t_years

        rng = np.random.default_rng(5678)
        vals = rng.normal(0.0, 1.0, len(dates)) * sigma

        res = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=[0.1, 0.5, 0.9],
            slope_scaling='year', min_size_per_block=50, min_blocks=10
        )
        df = res.to_frame()
        slopes = df['slope'].to_numpy()
        # q=0.1: negative slope; q=0.5: near zero slope; q=0.9: positive slope
        assert slopes[0] < -0.1
        assert np.isclose(slopes[1], 0.0, atol=0.3)
        assert slopes[2] > 0.1

    def test_multiplicative_log_trend(self):
        # Multiplicative trend: y(t) = y_0 * exp(gamma * t)
        # With transform='log', slope on log(y) is gamma per year.
        # slope_pct is 100 * (exp(gamma) - 1).
        years = 12
        dates = pd.date_range('2000-01-01', periods=years * 365, freq='D')
        t_years = (dates - dates[0]).total_seconds() / (365.25 * 86400.0)
        gamma = 0.05  # 5% growth rate
        expected_slope_pct = 100.0 * (np.exp(gamma) - 1.0)

        rng = np.random.default_rng(9999)
        noise = rng.lognormal(0.0, 0.05, len(dates))
        vals = 10.0 * np.exp(gamma * t_years) * noise

        res = quantile_trend_test(
            vals, dates, block_frequency='YE', transform='log',
            quantiles=5, slope_scaling='year', min_size_per_block=50, min_blocks=10
        )
        df = res.to_frame()
        assert len(df) == 5
        for sp in df['slope_pct']:
            assert np.isclose(sp, expected_slope_pct, atol=1.0)
        # `slope` is in natural-log units; `slope_pct` is the % per year figure
        assert df['slope_units'].iloc[0] == 'ln(units) per year'

    def test_stationary_series(self):
        # Stationary series: rejection rate roughly near alpha = 0.05, mostly No Trend
        years = 15
        dates = pd.date_range('2000-01-01', periods=years * 365, freq='D')
        rng = np.random.default_rng(2024)
        vals = rng.normal(50.0, 5.0, len(dates))

        res = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=9,
            min_size_per_block=50, min_blocks=10
        )
        df = res.to_frame()
        # Proportion of rejected nulls should be low (e.g. <= 30%)
        rejections = df['h'].sum()
        assert rejections <= 3


class TestQuantileTrendResultMethods:
    """Tests for QuantileTrendResult helper methods and post-processing."""

    @pytest.fixture
    def sample_result(self):
        years = 10
        dates = pd.date_range('2010-01-01', periods=years * 365, freq='D')
        vals = 10.0 + (np.arange(len(dates)) / len(dates)) * 5.0
        return quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=5,
            min_size_per_block=50, min_blocks=5
        )

    def test_to_frame_returns_dataframe_copy(self, sample_result):
        df = sample_result.to_frame()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'slope' in df.columns
        assert 'p' in df.columns

    def test_significant_filtering(self, sample_result):
        sig_df = sample_result.significant(alpha=0.05)
        assert isinstance(sig_df, pd.DataFrame)
        assert (sig_df['p'] < 0.05).all()

    def test_get_quantile_series_by_rank_and_prob(self, sample_result):
        # By rank (integer 1..5)
        s_rank = sample_result.get_quantile_series(1)
        assert isinstance(s_rank, pd.DataFrame)
        assert len(s_rank) == len(sample_result.matrix)
        assert 'value' in s_rank.columns
        assert 't' in s_rank.columns
        assert s_rank['rank'].iloc[0] == 1

        # By quantile probability (float ~0.5)
        s_prob = sample_result.get_quantile_series(0.5)
        assert isinstance(s_prob, pd.DataFrame)
        assert len(s_prob) == len(sample_result.matrix)

    def test_get_quantile_series_invalid_rank_raises(self, sample_result):
        with pytest.raises(ValueError, match="Rank 99 not found"):
            sample_result.get_quantile_series(99)

    def test_quantiles_property(self, sample_result):
        assert sample_result.quantiles is sample_result.results

    def test_get_quantile_series_on_censored_result(self):
        years = 6
        dates = pd.date_range('2010-01-01', periods=years * 365, freq='D')
        vals = np.linspace(5, 15, len(dates))
        cen = vals < 7.0
        cen_type = np.where(cen, 'lt', 'not')
        df_cen = pd.DataFrame({'value': vals, 'censored': cen, 'cen_type': cen_type})

        res = quantile_trend_test(
            df_cen, dates, block_frequency='YE', quantiles=5,
            min_blocks=5, min_size_per_block=50
        )
        s1 = res.get_quantile_series(1)
        assert isinstance(s1, pd.DataFrame)
        assert 'censored' in s1.columns
        assert 'cen_type' in s1.columns
        assert s1['censored'].any()
        assert (s1.loc[s1['censored'], 'cen_type'] == 'lt').all()

    def test_get_quantile_series_on_empty_result(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res = quantile_trend_test([1.0], pd.to_datetime(['2020-01-01']))
        s = res.get_quantile_series(1)
        assert isinstance(s, pd.DataFrame)
        assert list(s.columns) == ['t', 'value', 'censored', 'cen_type']
        assert len(s) == 0

    def test_summarize_quantile_regimes(self, sample_result):
        regimes_df = summarize_quantile_regimes(sample_result)
        assert isinstance(regimes_df, pd.DataFrame)
        assert 'regime' in regimes_df.columns
        assert 'median_slope' in regimes_df.columns
        assert 'modal_classification' in regimes_df.columns
        assert set(regimes_df['regime'].values).issubset({'low', 'mid', 'high'})

    def test_summarize_quantile_regimes_on_empty_result(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res = quantile_trend_test([1.0], pd.to_datetime(['2020-01-01']))
        reg = summarize_quantile_regimes(res)
        assert isinstance(reg, pd.DataFrame)
        assert len(reg) == 0
        expected_cols = [
            'regime', 'p_min', 'p_max', 'n_quantiles', 'median_slope',
            'median_slope_pct', 'prop_significant', 'modal_classification'
        ]
        assert list(reg.columns) == expected_cols

    def test_summarize_quantile_regimes_empty_regimes(self, sample_result):
        # Regime outside the range [0, 1]
        reg = summarize_quantile_regimes(sample_result, regimes={'out_of_bounds': (2.0, 3.0)})
        assert isinstance(reg, pd.DataFrame)
        assert len(reg) == 0


class TestEqualizeIntegration:
    """Integration tests for quantile_trend_test with equalize='aggregate'."""

    def test_equalize_aggregate_datetime(self):
        years = 5
        dates = pd.date_range('2015-01-01', periods=years * 365, freq='D')
        rng = np.random.default_rng(42)
        vals = 10.0 + np.linspace(0, 5, len(dates)) + rng.normal(0, 0.5, len(dates))

        res = quantile_trend_test(
            vals, dates, block_frequency='YE', equalize='aggregate', quantiles=4
        )
        assert res.computation_mode == 'exact'
        df = res.to_frame()
        assert len(df) == 4
        assert (df['slope'] > 0).all()
        assert np.all(np.isfinite(df['slope']))
        assert np.all(np.isfinite(df['p']))

    def test_equalize_aggregate_numeric(self):
        years = 5
        n_points = years * 365
        t_num = np.arange(n_points, dtype=float)
        rng = np.random.default_rng(42)
        vals = 10.0 + np.linspace(0, 5, n_points) + rng.normal(0, 0.5, n_points)

        res = quantile_trend_test(
            vals, t_num, block_frequency=365.0, equalize='aggregate', quantiles=4
        )
        assert res.computation_mode == 'exact'
        df = res.to_frame()
        assert len(df) == 4
        assert (df['slope'] > 0).all()
        assert np.all(np.isfinite(df['slope']))
        assert np.all(np.isfinite(df['p']))


class TestMissingDataPatterns:
    """Tests covering various missing data patterns (NaNs)."""

    def test_interleaved_nans(self):
        # 6 years of daily data with 50% interleaved NaNs (x[::2] = np.nan)
        years = 6
        dates = pd.date_range('2010-01-01', periods=years * 365, freq='D')
        vals = 10.0 + (np.arange(len(dates)) / len(dates)) * 5.0
        vals[::2] = np.nan

        res = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=5,
            min_size_per_block=50, min_blocks=5
        )
        assert res.computation_mode == 'exact'
        df = res.to_frame()
        assert len(df) == 5
        assert (df['slope'] > 0).all()
        assert np.all(np.isfinite(df['slope']))
        assert np.all(np.isfinite(df['p']))

    def test_leading_trailing_nans(self):
        # 6-year daily record with 50 NaNs at start and end
        years = 6
        dates = pd.date_range('2010-01-01', periods=years * 365, freq='D')
        vals = 10.0 + (np.arange(len(dates)) / len(dates)) * 5.0
        vals[:50] = np.nan
        vals[-50:] = np.nan

        # With min_coverage=0.90: first and last block coverage (~0.86) < 0.90 -> 4 blocks < min_blocks=5
        with pytest.warns(UserWarning, match="Insufficient valid blocks"):
            res_strict = quantile_trend_test(
                vals, dates, block_frequency='YE', quantiles=5,
                min_coverage=0.90, min_size_per_block=50, min_blocks=5
            )
        assert res_strict.computation_mode == 'insufficient'
        assert res_strict.to_frame().empty

        # With min_coverage=0.70: all 6 blocks pass coverage (315/365 = 0.86 >= 0.70)
        res_lenient = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=5,
            min_coverage=0.70, min_size_per_block=50, min_blocks=5
        )
        assert res_lenient.computation_mode == 'exact'
        df_lenient = res_lenient.to_frame()
        assert len(df_lenient) == 5
        assert (df_lenient['slope'] > 0).all()

    def test_all_nans_emits_warning_and_returns_empty(self):
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        vals = np.full(100, np.nan)
        with pytest.warns(UserWarning, match="Insufficient data for Quantile-Kendall analysis"):
            res = quantile_trend_test(vals, dates)
        assert res.computation_mode == 'insufficient'
        assert res.to_frame().empty


class TestBoundaryBlockSizes:
    """Tests for minimal datasets and minimal valid blocks (B=2)."""

    def test_minimal_blocks_b2_success(self):
        # B=2 blocks with n=2 obs/block (N=4, B=2, min_blocks=2, min_size_per_block=2)
        dates = pd.to_datetime(['2020-01-01', '2020-06-01', '2021-01-01', '2021-06-01'])
        vals = [10.0, 12.0, 20.0, 25.0]
        res = quantile_trend_test(
            vals, dates, block_frequency='YE', min_blocks=2,
            min_size_per_block=2, min_coverage=0.0
        )
        assert res.computation_mode == 'exact'
        df = res.to_frame()
        assert len(df) == 2
        assert np.all(np.isfinite(df['slope']))
        assert (df['p'] == 1.0).all()

    def test_minimal_dataset_b2_n2_quantiles_1(self):
        # B=2 blocks with 1 obs/block (N=2, B=2) and explicit quantiles=1
        dates = pd.to_datetime(['2020-01-01', '2021-01-01'])
        vals = [10.0, 20.0]
        res = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=1, min_blocks=2,
            min_size_per_block=1, min_coverage=0.0
        )
        assert res.computation_mode == 'exact'
        df = res.to_frame()
        assert len(df) == 1
        assert np.all(np.isfinite(df['slope']))
        assert (df['p'] == 1.0).all()

    def test_minimal_dataset_n2_default_raises_informative_error(self):
        # B=2 with 1 obs/block and default target_length='min' raises ValueError
        dates = pd.to_datetime(['2020-01-01', '2021-01-01'])
        vals = [10.0, 20.0]
        with pytest.raises(ValueError, match="Target quantile length must be at least 2"):
            quantile_trend_test(
                vals, dates, block_frequency='YE', min_blocks=2,
                min_size_per_block=1, min_coverage=0.0
            )


class TestExtremeLengthsAndScalability:
    """Stress tests verifying execution scalability and dense quantiles."""

    def test_large_time_series_scalability(self):
        # N >= 20,000 observations across 25 years asserting execution in < 1.0s
        import time
        n_points = 25000
        dates = pd.date_range('2000-01-01', periods=n_points, freq='9h')
        rng = np.random.default_rng(42)
        vals = 100.0 + np.linspace(0, 10, n_points) + rng.normal(0, 1, n_points)

        start = time.perf_counter()
        res = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=10,
            engine='fast', min_size_per_block=50, min_blocks=10
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"Execution took {elapsed:.2f}s, expected < 1.0s"
        assert res.computation_mode == 'fast'
        assert len(res.quantiles) == 10
        assert np.all(np.isfinite(res.to_frame()['slope']))

    def test_dense_quantiles(self):
        # quantiles=100 asserting valid 100-rank output
        years = 10
        dates = pd.date_range('2010-01-01', periods=years * 365, freq='D')
        rng = np.random.default_rng(123)
        vals = 50.0 + np.linspace(0, 5, len(dates)) + rng.normal(0, 2, len(dates))

        res = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=100,
            engine='fast', min_size_per_block=50, min_blocks=5
        )
        assert res.computation_mode == 'fast'
        df = res.to_frame()
        assert len(df) == 100
        assert list(df['rank']) == list(range(1, 101))
        assert np.all(np.isfinite(df['slope']))
        assert np.all(np.diff(df['plotting_position']) > 0)


class TestPublicOptionsAndTransforms:
    """Tests for target_length options, transforms (pct_of_median with positive data, log_offset)."""

    def test_target_length_median_and_int(self):
        # 3 blocks with sizes 30, 50, 70
        t1 = pd.date_range('2010-01-01', periods=30, freq='D')
        t2 = pd.date_range('2011-01-01', periods=50, freq='D')
        t3 = pd.date_range('2012-01-01', periods=70, freq='D')
        dates = pd.DatetimeIndex(np.concatenate([t1, t2, t3]))
        vals = np.arange(len(dates), dtype=float)

        # target_length='median': median size is 50 -> M=50 quantiles
        res_med = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles='auto',
            target_length='median', min_blocks=3, min_size_per_block=10, min_coverage=0.0
        )
        assert len(res_med.quantiles) == 50

        # target_length=4: explicit integer M=4
        res_int = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles='auto',
            target_length=4, min_blocks=3, min_size_per_block=10, min_coverage=0.0
        )
        assert len(res_int.quantiles) == 4

    def test_pct_of_median_with_positive_medians(self):
        years = 6
        dates = pd.date_range('2010-01-01', periods=years * 365, freq='D')
        vals = 100.0 + (np.arange(len(dates)) / len(dates)) * 50.0

        res = quantile_trend_test(
            vals, dates, block_frequency='YE', transform='pct_of_median',
            min_size_per_block=50, min_blocks=5
        )
        df = res.to_frame()
        assert not df.empty
        assert (df['slope'] > 0.0).all()
        # No non-positive median notes emitted
        assert not any("encountered non-positive quantile median" in note for note in res.analysis_notes)

    def test_log_transform_with_log_offset(self):
        years = 6
        dates = pd.date_range('2010-01-01', periods=years * 365, freq='D')
        # Data with values >= 0
        vals = (np.arange(len(dates)) / len(dates)) * 10.0
        vals[:5] = 0.0  # several exact zeros, so the lowest standardized rank is 0 for any grid
        # Without log_offset the zeros would raise ValueError
        with pytest.raises(ValueError, match="requires all standardized block values to be > 0"):
            quantile_trend_test(
                vals, dates, block_frequency='YE', transform='log',
                min_size_per_block=50, min_blocks=5
            )

        # With log_offset=1.0, values become > 0 and succeed
        res = quantile_trend_test(
            vals, dates, block_frequency='YE', transform='log', log_offset=1.0,
            min_size_per_block=50, min_blocks=5
        )
        df = res.to_frame()
        assert not df.empty
        assert (df['slope_pct'] > 0.0).all()

    def test_slope_pct_default_transform_none(self):
        """Verify that default transform='none' accurately computes slope_pct = 100.0 * slope / median."""
        dates = pd.date_range('2010-01-01', periods=10 * 365, freq='D')
        t_years = (dates - dates[0]).total_seconds() / (365.25 * 86400.0)
        vals = 100.0 + 2.0 * t_years
        res = quantile_trend_test(vals, dates, block_frequency='YE', quantiles=5, transform='none')
        df = res.results
        for _, row in df.iterrows():
            col = f"r{int(row['rank'])}"
            med_val = np.median(res.matrix[col])
            expected_pct = 100.0 * row['slope'] / med_val
            assert np.isclose(row['slope_pct'], expected_pct, rtol=1e-5)

    def test_equalize_and_transform_none_equivalent_to_string_none(self):
        """Verify that equalize=None and transform=None work identically to 'none'."""
        dates = pd.date_range('2010-01-01', periods=6 * 365, freq='D')
        vals = 10.0 + np.random.default_rng(42).normal(0.0, 1.0, len(dates))

        res_str = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=5,
            equalize='none', transform='none'
        )
        res_none = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=5,
            equalize=None, transform=None
        )

        pd.testing.assert_frame_equal(res_str.results, res_none.results)
        pd.testing.assert_frame_equal(res_str.matrix, res_none.matrix)
        assert res_str.transform == res_none.transform == 'none'

    def test_sampling_diagnostic_flag_false_after_thinning(self):
        """Verify that when equalize='thin', sampling_diagnostic['flag'] is False and ratio is 1.0."""
        dates = pd.date_range('2005-01-01', periods=120, freq='MS').append(
            pd.date_range('2015-01-01', periods=520, freq='W')
        )
        vals = np.random.default_rng(42).lognormal(2.0, 0.6, len(dates))

        res_raw = quantile_trend_test(vals, dates, block_frequency='YE', equalize='none')
        assert res_raw.sampling_diagnostic['flag'] is True
        assert res_raw.sampling_diagnostic['ratio_max_min'] > 1.5

        res_thinned = quantile_trend_test(vals, dates, block_frequency='YE', equalize='thin', random_state=42)
        assert res_thinned.sampling_diagnostic['flag'] is False
        assert res_thinned.sampling_diagnostic['ratio_max_min'] == 1.0
        assert res_thinned.sampling_diagnostic['n_min'] == res_thinned.sampling_diagnostic['n_max']

    def test_target_length_max(self):
        """Verify target_length='max' uses maximum valid block size."""
        dates = pd.date_range('2010-01-01', periods=10 * 365 + 3, freq='D')
        vals = np.random.default_rng(1).normal(10.0, 1.0, len(dates))
        res = quantile_trend_test(vals, dates, block_frequency='YE', quantiles='auto', target_length='max')
        assert res.n_ranks == 366
        assert res.target_length == 366

    def test_fast_engine_respects_ci_and_tau_method(self):
        """Verify engine='fast' matches engine='trend_test' when ci_method or tau_method is customized."""
        dates = pd.date_range('2010-01-01', periods=10 * 365, freq='D')
        t_yrs = (dates - dates[0]).total_seconds() / (365.25 * 86400.0)
        vals = np.asarray(10.0 + 0.5 * t_yrs + np.random.default_rng(7).normal(0.0, 1.0, len(dates)), dtype=float)
        # Introduce ties
        vals[:30] = 10.0
        res_exact = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=5,
            engine='trend_test', ci_method='lwp', tau_method='a'
        )
        res_fast = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=5,
            engine='fast', ci_method='lwp', tau_method='a'
        )
        assert res_fast.computation_mode == 'fast'
        np.testing.assert_allclose(res_fast.results['Tau'], res_exact.results['Tau'], rtol=1e-12)
        np.testing.assert_allclose(res_fast.results['lower_ci'], res_exact.results['lower_ci'], rtol=1e-12)
        np.testing.assert_allclose(res_fast.results['upper_ci'], res_exact.results['upper_ci'], rtol=1e-12)

    def test_normalize_start_freq_qe_and_hourly(self):
        """Verify QE- maps to the following start month and sub-daily 'h' stays lowercase."""
        from MannKS._quantile import _normalize_start_freq
        assert _normalize_start_freq('QE-MAR') == 'QS-APR'
        assert _normalize_start_freq('QE-DEC') == 'QS-JAN'
        assert _normalize_start_freq('h') == 'h'
        assert _normalize_start_freq('2h') == '2h'

    def test_field_significance_handles_nan_columns(self):
        """Verify field_significance=True handles NaN columns (e.g. zero median in pct_of_median) safely."""
        dates = pd.date_range('2010-01-01', periods=8 * 20, freq='18D')
        rng = np.random.default_rng(12)
        vals = rng.normal(5.0, 1.0, len(dates))
        # Force lowest values in every block to 0 so lowest quantile median is 0
        for yr in range(2010, 2018):
            idx = np.flatnonzero(dates.year == yr)[:3]
            vals[idx] = 0.0
        res = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=[0.05, 0.50, 0.95],
            min_size_per_block=5, min_coverage=0.5, transform='pct_of_median',
            field_significance=True, n_permutations=49, random_state=42
        )
        assert res.field_significance is not None
        assert np.isfinite(res.field_significance['p_count'])
        assert np.isfinite(res.field_significance['p_max_z'])


class TestEdgeCasesAndValidationRegressions:
    """Comprehensive regression tests for validation guards, scaling/unit parity, and edge inputs."""

    def test_x_unit_none_and_slope_scaling_none_both_engines(self):
        dates = pd.date_range('2010-01-01', periods=6 * 365, freq='D')
        vals = 10.0 + (np.arange(len(dates)) / len(dates)) * 5.0

        for eng in ('trend_test', 'fast'):
            res_raw = quantile_trend_test(
                vals, dates, block_frequency='YE', quantiles=3,
                x_unit=None, slope_scaling=None, engine=eng,
            )
            assert res_raw.results['slope_units'].iloc[0] == 'units per second'

            res_log = quantile_trend_test(
                vals, dates, block_frequency='YE', quantiles=3,
                transform='log', x_unit=None, slope_scaling=None, engine=eng,
            )
            assert res_log.results['slope_units'].iloc[0] == 'ln(units) per second'

    def test_invalid_slope_scaling_decade_parity_across_engines(self):
        dates = pd.date_range('2010-01-01', periods=6 * 365, freq='D')
        vals = 10.0 + (np.arange(len(dates)) / len(dates)) * 5.0

        with pytest.warns(UserWarning, match="Slope scaling failed"):
            res_exact = quantile_trend_test(
                vals, dates, block_frequency='YE', quantiles=3,
                slope_scaling='decade', engine='trend_test',
            )
        with pytest.warns(UserWarning, match="Slope scaling failed"):
            res_fast = quantile_trend_test(
                vals, dates, block_frequency='YE', quantiles=3,
                slope_scaling='decade', engine='fast',
            )
        assert res_exact.results['slope_units'].iloc[0] == 'units per second'
        assert res_fast.results['slope_units'].iloc[0] == 'units per second'
        np.testing.assert_allclose(res_fast.results['slope'], res_exact.results['slope'], rtol=1e-12)

    def test_upfront_validation_rejects_bad_inputs_even_on_fast_engine(self):
        dates = pd.date_range('2010-01-01', periods=6 * 365, freq='D')
        vals = np.arange(len(dates), dtype=float) + 1.0

        with pytest.raises(ValueError, match="min_size_per_block"):
            quantile_trend_test(vals, dates, engine='fast', min_size_per_block=True)
        with pytest.raises(ValueError, match="min_size_per_block"):
            quantile_trend_test(vals, dates, engine='fast', min_size_per_block=2.5)

        for bad_months in ([], [0], [13], [True], '6'):
            with pytest.raises(ValueError, match="`months`"):
                quantile_trend_test(vals, dates, engine='fast', months=bad_months)

        for bad_offset in (True, np.nan, np.inf, 'invalid'):
            with pytest.raises(ValueError, match="`log_offset`"):
                quantile_trend_test(vals, dates, engine='fast', log_offset=bad_offset)

        with pytest.raises(ValueError, match="`quantiles` cannot be a boolean"):
            quantile_trend_test(vals, dates, engine='fast', quantiles=True)

        with pytest.raises(ValueError, match="`target_length`"):
            quantile_trend_test(vals, dates, engine='fast', target_length=True)

        with pytest.raises(ValueError, match="`standardization='egret'` requires `target_length='min'`"):
            quantile_trend_test(vals, dates, standardization='egret', target_length='median', min_size_per_block=100)

    def test_months_warnings_for_numeric_and_unsorted_months(self):
        # Numeric t emits ignored-months warning
        t_num = np.arange(6 * 12, dtype=float)
        x_num = np.linspace(1.0, 10.0, len(t_num))
        with pytest.warns(UserWarning, match="Parameter `months` is ignored"):
            quantile_trend_test(
                x_num, t_num, block_frequency=12.0, months=[6, 7],
                min_size_per_block=5, min_blocks=5, quantiles=3,
            )

        # Unsorted summer months [8, 6, 7] should NOT trigger Cross-year warning
        dates = pd.date_range('2010-01-01', periods=8 * 365, freq='D')
        rng = np.random.default_rng(11)
        vals = rng.normal(10.0, 1.0, len(dates))
        with warnings.catch_warnings(record=True) as w_summer:
            warnings.simplefilter("always")
            quantile_trend_test(
                vals, dates, block_frequency='YE', months=[8, 6, 7],
                min_size_per_block=20, min_blocks=5,
            )
        assert not any("Cross-year" in str(item.message) for item in w_summer)

        # [1, 12] spans calendar boundary under YE -> Cross-year warning
        with pytest.warns(UserWarning, match="Cross-year"):
            quantile_trend_test(
                vals, dates, block_frequency='YE', months=[1, 12],
                min_size_per_block=20, min_blocks=5,
            )

    def test_fast_engine_uncensored_sens_slope_ats_matches_trend_test(self):
        dates = pd.date_range('2010-01-01', periods=8 * 365, freq='D')
        rng = np.random.default_rng(7)
        vals = 10.0 + 0.01 * np.arange(len(dates)) + rng.normal(0.0, 1.0, len(dates))
        res_fast = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=5,
            engine='fast', sens_slope_method='ats',
        )
        res_exact = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=5,
            engine='trend_test', sens_slope_method='ats',
        )
        assert res_fast.computation_mode == 'fast'
        np.testing.assert_allclose(res_fast.results['slope'], res_exact.results['slope'], rtol=1e-12)

    def test_monthly_blocks_equalize_aggregate_end_to_end(self):
        dates = pd.date_range('2020-01-01', '2020-08-31 18:00:00', freq='6h')
        vals = 5.0 + np.linspace(0.0, 4.0, len(dates))
        res = quantile_trend_test(
            vals, dates, block_frequency='ME', equalize='aggregate',
            quantiles=[0.25, 0.50, 0.75], min_blocks=6,
        )
        assert len(res.results) == 3
        assert (res.results['slope'] > 0.0).all()

    def test_equalize_aggregate_residual_imbalance_adds_analysis_note(self):
        # First 5 years have 6 distinct months/year (bi-monthly); next 5 have 12 months/year -> ratio 2.0 > 1.5 even after monthly aggregate
        t_parts = []
        for yr in range(2010, 2015):
            for m in (1, 3, 5, 7, 9, 11):
                t_parts.extend(pd.date_range(f"{yr}-{m:02d}-01", periods=3, freq='3D'))
        for yr in range(2015, 2020):
            for m in range(1, 13):
                t_parts.extend(pd.date_range(f"{yr}-{m:02d}-01", periods=4, freq='3D'))
        dates = pd.DatetimeIndex(t_parts)
        vals = np.linspace(1.0, 10.0, len(dates))
        res = quantile_trend_test(
            vals, dates, block_frequency='YE', equalize='aggregate',
            quantiles=3, min_size_per_block=6, min_coverage=0.70,
        )
        assert res.sampling_diagnostic['flag'] is True
        assert any("Residual sampling-frequency imbalance remains" in note for note in res.analysis_notes)

    def test_fast_uncensored_matrix_trends_empty_dt_valid_no_warning(self):
        from MannKS.quantile_trend_test import _fast_uncensored_matrix_trends
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        t_same = np.array([5.0, 5.0])
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always", category=RuntimeWarning)
            rows = _fast_uncensored_matrix_trends(
                M, t_same, is_datetime=False, alpha=0.05, slope_scaling=None,
                x_unit='mg/L', continuous_confidence=True, category_map=None,
            )
        assert not [w for w in rec if issubclass(w.category, RuntimeWarning)]
        assert np.isnan(rows[0]['slope'])
        assert np.isnan(rows[0]['lower_ci'])

    def test_result_helpers_guards_and_dataframe_schema(self):
        dates = pd.date_range('2010-01-01', periods=6 * 365, freq='D')
        vals = 10.0 + np.linspace(0.0, 5.0, len(dates))
        res = quantile_trend_test(vals, dates, block_frequency='YE', quantiles=3)

        with pytest.raises(ValueError, match="cannot be a boolean"):
            res.get_quantile_series(True)
        for bad_q in (np.nan, -0.1, 1.5, -2):
            with pytest.raises(ValueError, match="Probability `quantile_or_rank` must be in"):
                res.get_quantile_series(bad_q)

        # summarize_quantile_regimes preserves expected columns on empty match and accepts DataFrame input
        empty_reg = summarize_quantile_regimes(res, regimes={'none': (0.99, 1.0)})
        assert list(empty_reg.columns) == [
            'regime', 'p_min', 'p_max', 'n_quantiles', 'median_slope',
            'median_slope_pct', 'prop_significant', 'modal_classification',
        ]
        assert empty_reg.empty

        df_reg = summarize_quantile_regimes(res.to_frame())
        assert not df_reg.empty

    def test_edge_cases_constant_all_censored_negative_tz_and_timedelta(self):
        dates = pd.date_range('2010-01-01', periods=6 * 365, freq='D')

        # 1. 100% left-censored with changing DL does not claim false trend
        vals_dl = np.where(dates.year < 2013, 5.0, 2.0)
        cens_df = pd.DataFrame({
            'value': vals_dl,
            'censored': np.ones(len(dates), dtype=bool),
            'cen_type': np.full(len(dates), 'lt', dtype=object),
        })
        res_cens = quantile_trend_test(cens_df, dates, quantiles=3)
        assert np.allclose(res_cens.results['p'], 1.0)
        assert res_cens.results['slope'].isna().all()

        # 2. All-negative values: transform='none' yields valid slope and NaN slope_pct
        neg_vals = -20.0 + np.linspace(0.0, 5.0, len(dates))
        res_neg = quantile_trend_test(neg_vals, dates, quantiles=3, transform='none')
        assert (res_neg.results['slope'] > 0.0).all()
        assert res_neg.results['slope_pct'].isna().all()

        # 3. Timezone-aware timestamps + months + equalize='aggregate' + pd.Timedelta block_frequency
        dates_tz = pd.date_range('2010-01-01', periods=6 * 365, freq='D', tz='Pacific/Auckland')
        res_tz = quantile_trend_test(
            np.arange(len(dates_tz), dtype=float), dates_tz,
            block_frequency=pd.Timedelta('365D'), months=[1, 2, 3, 4],
            equalize='aggregate', quantiles=3, min_coverage=0.5,
        )
        assert len(res_tz.results) == 3
        assert (res_tz.results['slope'] > 0.0).all()

    def test_default_quantile_method_is_hazen(self):
        # Verify quantile_trend_test defaults to quantile_method='hazen'
        rng = np.random.default_rng(77)
        dates = pd.date_range('2010-01-15', periods=10 * 12, freq='MS')
        vals = rng.lognormal(mean=0.0, sigma=0.8, size=len(dates))

        res_default = quantile_trend_test(vals, dates, block_frequency='YE', quantiles=[0.05, 0.50, 0.95])
        res_hazen = quantile_trend_test(vals, dates, block_frequency='YE', quantiles=[0.05, 0.50, 0.95], quantile_method='hazen')
        res_linear = quantile_trend_test(vals, dates, block_frequency='YE', quantiles=[0.05, 0.50, 0.95], quantile_method='linear')

        pd.testing.assert_frame_equal(res_default.matrix, res_hazen.matrix)
        pd.testing.assert_frame_equal(res_default.results, res_hazen.results)
        # Tail quantile 0.95 on N=12 monthly blocks differs between Hazen (v=10.90) and Type-7 linear (v=10.45)
        assert not np.allclose(res_default.matrix.iloc[:, -1], res_linear.matrix.iloc[:, -1])
