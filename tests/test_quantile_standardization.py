"""
Unit tests for length standardization, plotting positions, sampling diagnostics, and block equalization.

Target module: MannKS._quantile
Functions:
    - _standardize_block
    - _plotting_positions
    - _sampling_diagnostic
    - _equalize_blocks
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from MannKS._quantile import (
    _standardize_block,
    _plotting_positions,
    _sampling_diagnostic,
    _equalize_blocks,
    _block_coverage,
    _assign_blocks,
    _mk_s_matrix,
)


class TestPlottingPositions:
    """Tests for _plotting_positions Weibull formula and normal quantiles."""

    @pytest.mark.parametrize("M", [3, 10, 100])
    def test_weibull_and_normal_quantiles(self, M):
        ranks, p_grid, z_grid = _plotting_positions(M)
        
        # Ranks should be 1 to M
        assert np.array_equal(ranks, np.arange(1, M + 1, dtype=float))
        
        # Weibull: j / (M + 1)
        expected_p = np.arange(1, M + 1, dtype=float) / (M + 1.0)
        assert np.allclose(p_grid, expected_p)
        assert np.all(p_grid > 0.0) and np.all(p_grid < 1.0)
        
        # Standard normal inverse: Phi^-1(p)
        expected_z = norm.ppf(expected_p)
        assert np.allclose(z_grid, expected_z)
        
        # Symmetry about 0: z_j == -z_{M+1-j}
        assert np.allclose(z_grid + z_grid[::-1], 0.0, atol=1e-12)


class TestStandardizeBlockInterpolate:
    """Tests for _standardize_block under 'interpolate' standardization."""

    def test_identity_when_n_equals_m(self):
        # When n == M, interpolated quantiles on linspace(0, 1, M) equal sorted values
        vals = np.array([4.2, 1.1, 9.8, 3.5, 7.0])
        df = pd.DataFrame({'value': vals})
        std_vals, std_cens, std_type = _standardize_block(
            group_df=df, M=len(vals), standardization='interpolate',
            quantile_method='linear', censor_quantile_method='propagate',
            lt_mult=0.5, gt_mult=1.1,
        )
        assert np.allclose(std_vals, np.sort(vals))
        assert not std_cens.any()
        assert (std_type == 'not').all()

    def test_exact_preservation_of_endpoints(self):
        # Min and max must be exactly preserved when u in [0, 1]
        vals = np.array([12.5, 3.2, 88.9, 45.0, 1.0, 99.4, 60.1])
        df = pd.DataFrame({'value': vals})
        std_vals, _, _ = _standardize_block(
            group_df=df, M=4, standardization='interpolate',
            quantile_method='linear', censor_quantile_method='propagate',
            lt_mult=0.5, gt_mult=1.1,
        )
        assert np.isclose(std_vals[0], vals.min())
        assert np.isclose(std_vals[-1], vals.max())

    def test_monotonicity(self):
        # Output must be non-decreasing for any arbitrary input
        rng = np.random.default_rng(42)
        vals = rng.normal(loc=50.0, scale=15.0, size=75)
        df = pd.DataFrame({'value': vals})
        std_vals, _, _ = _standardize_block(
            group_df=df, M=20, standardization='interpolate',
            quantile_method='linear', censor_quantile_method='propagate',
            lt_mult=0.5, gt_mult=1.1,
        )
        assert np.all(np.diff(std_vals) >= 0.0)

    def test_permutation_invariance(self):
        # Permuting input order must yield identical standardized values
        vals1 = np.array([10.0, 50.0, 30.0, 20.0, 40.0])
        vals2 = np.array([40.0, 20.0, 10.0, 50.0, 30.0])
        df1 = pd.DataFrame({'value': vals1})
        df2 = pd.DataFrame({'value': vals2})

        std1, _, _ = _standardize_block(
            group_df=df1, M=3, standardization='interpolate',
            quantile_method='linear', censor_quantile_method='propagate',
            lt_mult=0.5, gt_mult=1.1,
        )
        std2, _, _ = _standardize_block(
            group_df=df2, M=3, standardization='interpolate',
            quantile_method='linear', censor_quantile_method='propagate',
            lt_mult=0.5, gt_mult=1.1,
        )
        assert np.allclose(std1, std2)

    def test_constant_block(self):
        # All values identical
        vals = np.full(10, 7.5)
        df = pd.DataFrame({'value': vals})
        std_vals, _, _ = _standardize_block(
            group_df=df, M=5, standardization='interpolate',
            quantile_method='linear', censor_quantile_method='propagate',
            lt_mult=0.5, gt_mult=1.1,
        )
        assert np.allclose(std_vals, 7.5)

    def test_single_target_quantile_m1(self):
        # M = 1 defaults to u = 0.5 (median)
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df = pd.DataFrame({'value': vals})
        std_vals, _, _ = _standardize_block(
            group_df=df, M=1, standardization='interpolate',
            quantile_method='linear', censor_quantile_method='propagate',
            lt_mult=0.5, gt_mult=1.1,
        )
        assert len(std_vals) == 1
        assert np.isclose(std_vals[0], 3.0)

    def test_explicit_u_grid(self):
        vals = np.linspace(0.0, 100.0, 101)
        df = pd.DataFrame({'value': vals})
        u_grid = np.array([0.10, 0.50, 0.90])
        std_vals, _, _ = _standardize_block(
            group_df=df, M=3, standardization='interpolate',
            quantile_method='linear', censor_quantile_method='propagate',
            lt_mult=0.5, gt_mult=1.1, u_grid=u_grid,
        )
        assert np.allclose(std_vals, [10.0, 50.0, 90.0])


class TestStandardizeBlockEGRET:
    """Tests for _standardize_block under 'egret' mode (leapOdd / leapEven)."""

    def test_egret_identity_when_n_equals_m(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df = pd.DataFrame({'value': vals})
        std_vals, std_cens, std_type = _standardize_block(
            group_df=df, M=5, standardization='egret',
            quantile_method='linear', censor_quantile_method='propagate',
            lt_mult=0.5, gt_mult=1.1,
        )
        assert np.allclose(std_vals, vals)

    def test_egret_odd_reduction_leap_odd(self):
        # n_obs = 5 (odd), M = 4 (M+1 = 5). mid = 5 // 2 = 2.
        # Index 2 (the middle observation) should be dropped!
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        df = pd.DataFrame({'value': vals})
        std_vals, _, _ = _standardize_block(
            group_df=df, M=4, standardization='egret',
            quantile_method='linear', censor_quantile_method='propagate',
            lt_mult=0.5, gt_mult=1.1,
        )
        # Dropping index 2 (value 30.0) leaves [10.0, 20.0, 40.0, 50.0]
        assert len(std_vals) == 4
        assert np.allclose(std_vals, [10.0, 20.0, 40.0, 50.0])

    def test_egret_even_reduction_leap_even(self):
        # n_obs = 6 (even), M = 5 (M+1 = 6). mid = 6 // 2 = 3.
        # leapEven averages the two middle elements (mid-1=2 and mid=3), and drops mid.
        # vals: [10, 20, 30, 40, 50, 60] -> middle two are 30 and 40 -> avg = 35.
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        df = pd.DataFrame({'value': vals})
        std_vals, _, _ = _standardize_block(
            group_df=df, M=5, standardization='egret',
            quantile_method='linear', censor_quantile_method='propagate',
            lt_mult=0.5, gt_mult=1.1,
        )
        assert len(std_vals) == 5
        assert np.allclose(std_vals, [10.0, 20.0, 35.0, 50.0, 60.0])

    def test_egret_invalid_length_difference_raises(self):
        # n_obs - M must be in {0, 1}
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        df = pd.DataFrame({'value': vals})
        with pytest.raises(ValueError, match="standardization='egret' requires block sample size"):
            _standardize_block(
                group_df=df, M=3, standardization='egret',
                quantile_method='linear', censor_quantile_method='propagate',
                lt_mult=0.5, gt_mult=1.1,
            )

        with pytest.raises(ValueError, match="standardization='egret' requires block sample size"):
            _standardize_block(
                group_df=df, M=7, standardization='egret',
                quantile_method='linear', censor_quantile_method='propagate',
                lt_mult=0.5, gt_mult=1.1,
            )


class TestSamplingDiagnostic:
    """Tests for _sampling_diagnostic drift detection."""

    def test_too_few_blocks(self):
        diag = _sampling_diagnostic(np.array([1.0, 2.0]), np.array([10, 10]))
        assert diag['flag'] == False
        assert diag['ratio_max_min'] == 1.0

    def test_stable_sampling(self):
        centers = pd.date_range('2010-01-01', periods=10, freq='YS')
        sizes = np.array([52, 52, 51, 52, 53, 52, 52, 51, 52, 52])
        diag = _sampling_diagnostic(centers, sizes)
        assert diag['flag'] == False
        assert diag['ratio_max_min'] < 1.5

    def test_drift_trap_detected(self):
        # Shifts from 12 samples/yr to 52 samples/yr: ratio = 52/12 = 4.33 > 1.5
        centers = pd.date_range('2000-01-01', periods=10, freq='YS')
        sizes = np.array([12, 12, 12, 12, 12, 52, 52, 52, 52, 52])
        diag = _sampling_diagnostic(centers, sizes)
        assert diag['flag'] == True
        assert diag['ratio_max_min'] > 1.5
        assert diag['p_n_vs_time'] < 0.10


class TestEqualizeBlocks:
    """Tests for _equalize_blocks thinning and aggregation."""

    def test_equalize_none(self):
        dfs = [pd.DataFrame({'value': [1, 2]}), pd.DataFrame({'value': [3, 4, 5]})]
        res = _equalize_blocks(dfs, mode='none')
        assert len(res) == 2
        assert len(res[0]) == 2
        assert len(res[1]) == 3

    def test_equalize_thin(self):
        df1 = pd.DataFrame({'t': [1, 2, 3], 'value': [10, 20, 30]})
        df2 = pd.DataFrame({'t': [1, 2, 3, 4, 5], 'value': [10, 20, 30, 40, 50]})
        res = _equalize_blocks([df1, df2], mode='thin', is_datetime=False)
        assert len(res[0]) == 3
        assert len(res[1]) == 3
        # Linspace indices for 5 points to 3 points: 0, 2, 4 -> values [10, 30, 50]
        assert np.allclose(res[1]['value'].values, [10, 30, 50])

    def test_equalize_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown equalize mode: 'bad_mode'"):
            _equalize_blocks([pd.DataFrame({'value': [1]})], mode='bad_mode')

    def test_equalize_aggregate(self):
        # 365 daily observations across 1 calendar year
        dates = pd.date_range('2021-01-01', periods=365, freq='D')
        df = pd.DataFrame({
            't': dates,
            't_original': dates,
            'value': np.arange(365, dtype=float),
            'censored': False,
            'cen_type': 'not',
        })
        res = _equalize_blocks([df], mode='aggregate', is_datetime=True)
        assert len(res) == 1
        assert len(res[0]) == 12  # Downsampled to 12 monthly robust medians
        expected_cols = {'value', 'censored', 'cen_type', 't_original', 't'}
        assert set(res[0].columns) >= expected_cols
        # Medians must be monotonically increasing matching values
        assert (np.diff(res[0]['value'].values) > 0).all()

        # Numeric aggregation
        df_num = pd.DataFrame({
            't': np.arange(365, dtype=float),
            't_original': np.arange(365, dtype=float),
            'value': np.arange(365, dtype=float),
            'censored': False,
            'cen_type': 'not',
        })
        res_num = _equalize_blocks([df_num], mode='aggregate', is_datetime=False)
        assert len(res_num) == 1
        assert len(res_num[0]) == 13
        assert set(res_num[0].columns) >= expected_cols


class TestQuantileCoverageAndMatrixEdgeCases:
    """Tests for edge cases in _block_coverage, _assign_blocks, and _mk_s_matrix."""

    def test_block_coverage_zero_or_negative_duration(self):
        # Datetime duration <= 0
        t_dt = pd.to_datetime(['2020-01-01', '2020-01-01'])
        cov_dt = _block_coverage(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-01'), t_dt, is_datetime=True)
        assert cov_dt == 1.0

        # Numeric duration <= 0
        t_num = np.array([10.0, 10.0])
        cov_num = _block_coverage(10.0, 10.0, t_num, is_datetime=False)
        assert cov_num == 1.0

    def test_assign_blocks_require_complete_numeric_and_months(self):
        # Datetime with months filter and require_complete=True
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data_dt = pd.DataFrame({
            't': (dates - dates[0]).total_seconds(),
            't_original': dates,
            'value': np.arange(10, dtype=float),
            'censored': False,
            'cen_type': 'not',
        })
        valid, blocks_df, centers = _assign_blocks(
            data_dt, block_frequency='YE', block_origin=None, months=[1],
            min_size_per_block=1, min_coverage=0.0, require_complete=True, is_datetime=True
        )
        assert len(valid) == 0  # Incomplete block dropped

        # Numeric with require_complete=True
        t_num = np.array([0.0, 1.0, 2.0])
        data_num = pd.DataFrame({
            't': t_num,
            't_original': t_num,
            'value': t_num,
            'censored': False,
            'cen_type': 'not',
        })
        valid_n, blocks_df_n, centers_n = _assign_blocks(
            data_num, block_frequency=10.0, block_origin=None, months=None,
            min_size_per_block=1, min_coverage=0.0, require_complete=True, is_datetime=False
        )
        assert len(valid_n) == 0

    def test_mk_s_matrix_edge_cases(self):
        # B < 2: 1 block, 3 quantiles
        M = np.array([[10.0, 20.0, 30.0]])
        s, var_s, tau = _mk_s_matrix(M, np.array([0.0]))
        assert np.array_equal(s, [0.0, 0.0, 0.0])
        assert np.array_equal(var_s, [1.0, 1.0, 1.0])
        assert np.array_equal(tau, [0.0, 0.0, 0.0])

        # All dt == 0 (identical timestamps for all blocks)
        M2 = np.array([[10.0, 20.0], [15.0, 25.0]])
        s2, var_s2, tau2 = _mk_s_matrix(M2, np.array([5.0, 5.0]))
        assert np.array_equal(s2, [0.0, 0.0])
        assert np.array_equal(var_s2, [1.0, 1.0])
        assert np.array_equal(tau2, [0.0, 0.0])

    def test_equalize_aggregate_subannual_and_subdaily(self):
        # 1-month block of 6-hourly obs -> span ~30d (>2d, <=45d) -> daily aggregation
        dates_me = pd.date_range('2021-03-01', '2021-03-31 18:00:00', freq='6h')
        df_me = pd.DataFrame({
            't': (dates_me - pd.Timestamp('1970-01-01')).total_seconds(),
            't_original': dates_me,
            'value': np.arange(len(dates_me), dtype=float),
            'censored': False,
            'cen_type': 'not',
        })
        res_me = _equalize_blocks([df_me], mode='aggregate', is_datetime=True)
        assert len(res_me) == 1
        assert len(res_me[0]) == 31  # 31 days in March

        # 15-minute block of 1-minute obs -> span ~14min (<=2h) -> minute aggregation
        dates_15m = pd.date_range('2021-03-01 12:00:00', '2021-03-01 12:14:00', freq='1min')
        df_15m = pd.DataFrame({
            't': (dates_15m - pd.Timestamp('1970-01-01')).total_seconds(),
            't_original': dates_15m,
            'value': np.arange(len(dates_15m), dtype=float),
            'censored': False,
            'cen_type': 'not',
        })
        res_15m = _equalize_blocks([df_15m], mode='aggregate', is_datetime=True)
        assert len(res_15m) == 1
        assert len(res_15m[0]) == 15


