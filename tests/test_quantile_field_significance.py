"""
Unit tests for permutation field significance, rank correlation preservation,
vectorized matrix Mann-Kendall calculations, and engine parity under ties.

Target modules: MannKS._quantile, MannKS.quantile_trend_test
Functions:
    - _compute_field_significance
    - _mk_s_matrix
    - quantile_trend_test (field_significance=True, engine='fast' vs 'trend_test')
"""
import numpy as np
import pandas as pd
import pytest

from MannKS._quantile import _mk_s_matrix
from MannKS.quantile_trend_test import (
    quantile_trend_test,
    _compute_field_significance,
)


class TestMkSMatrixUnit:
    """Unit tests for _mk_s_matrix tie-corrected Kendall S and variance."""

    def test_mk_s_matrix_untied(self):
        # 5 points strictly increasing
        t = np.arange(5, dtype=float)
        M = np.tile(t[:, np.newaxis], (1, 3))  # 3 identical columns
        s_scores, var_s, tau = _mk_s_matrix(M, t)
        # S should be 5*4/2 = 10
        assert np.allclose(s_scores, 10.0)
        # Var(S) without ties: 5 * 4 * (2*5 + 5) / 18 = 20 * 15 / 18 = 300 / 18 = 16.6666...
        expected_var = 5.0 * 4.0 * 15.0 / 18.0
        assert np.allclose(var_s, expected_var)
        assert np.allclose(tau, 1.0)

    def test_mk_s_matrix_exact_tie_correction(self):
        # 5 points with ties: [1, 1, 2, 3, 3]
        # Ties of size 2 at 1, and size 2 at 3.
        # base_var = 16.66666...
        # tie_adj = (2 * 1 * 9 / 18) + (2 * 1 * 9 / 18) = 1.0 + 1.0 = 2.0 (or 2 * 1 * 9 / 18 = 1.0 each)
        # expected var_s = 16.66667 - (1.0 + 1.0) = 14.66667... Wait, let's verify exact formula:
        # tie count t_i = 2: 2 * (2-1) * (2*2+5) / 18 = 2 * 1 * 9 / 18 = 1.0
        # Two ties of size 2 -> tie_adj = 2.0, so var_s = 16.66667 - 2.0 = 14.66667
        t = np.arange(5, dtype=float)
        vals = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
        M = vals[:, np.newaxis]
        s_scores, var_s, tau = _mk_s_matrix(M, t)
        
        base_var = 5.0 * 4.0 * 15.0 / 18.0
        tie_adj = 2.0 * (2.0 * 1.0 * 9.0 / 18.0)
        expected_var = base_var - tie_adj
        assert np.isclose(var_s[0], expected_var)


class TestFastEngineParityWithTies:
    """Verifies engine='fast' matches engine='trend_test' on series with ties."""

    def test_fast_engine_matches_trend_test(self):
        # Generate 6 blocks with 10 quantiles containing repeated (tied) values
        years = 6
        dates = pd.date_range('2010-01-01', f"{2010+years-1}-12-31", freq='10D')
        t_arr = (dates - dates[0]).total_seconds() / (365.25 * 86400.0)
        # Discretized values to ensure ties
        vals = np.round(10.0 + 1.5 * t_arr + np.sin(np.arange(len(dates))))

        res_std = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=10,
            engine='trend_test', min_size_per_block=20, min_blocks=5
        ).to_frame()

        res_fast = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=10,
            engine='fast', min_size_per_block=20, min_blocks=5
        ).to_frame()

        assert np.allclose(res_std['s'], res_fast['s'])
        assert np.allclose(res_std['var_s'], res_fast['var_s'])
        assert np.allclose(res_std['Tau'], res_fast['Tau'])
        assert np.allclose(res_std['z'], res_fast['z'])
        assert np.allclose(res_std['p'], res_fast['p'])
        assert np.allclose(res_std['slope'], res_fast['slope'])
        assert np.allclose(res_std['intercept'], res_fast['intercept'], rtol=1e-5)


class TestFieldSignificancePermutations:
    """Tests for _compute_field_significance and whole-block row permutations."""

    def test_field_significance_output_fields(self):
        # Create a block matrix (10 blocks x 5 quantiles)
        B, K = 10, 5
        M = np.tile(np.linspace(1, 10, B)[:, np.newaxis], (1, K))
        t = np.arange(B, dtype=float)
        var_s = np.full(K, 50.0)
        z_obs = np.full(K, 3.5)

        sig_dict = _compute_field_significance(
            M_matrix=M, t_numeric=t, var_s_vec=var_s, z_obs_vec=z_obs,
            alpha=0.05, n_permutations=99, random_state=42
        )

        assert isinstance(sig_dict, dict)
        assert 'n_significant' in sig_dict
        assert 'p_count' in sig_dict
        assert 'max_abs_z' in sig_dict
        assert 'p_max_z' in sig_dict
        assert 'n_permutations' in sig_dict
        assert sig_dict['n_permutations'] == 99
        assert sig_dict['n_significant'] == 5
        assert np.isclose(sig_dict['max_abs_z'], 3.5)

    def test_field_significance_reproducibility(self):
        B, K = 8, 4
        rng = np.random.default_rng(123)
        M = rng.normal(size=(B, K))
        t = np.arange(B, dtype=float)
        var_s = np.full(K, 20.0)
        z_obs = np.array([2.1, 0.5, 1.8, -2.4])

        run1 = _compute_field_significance(
            M_matrix=M, t_numeric=t, var_s_vec=var_s, z_obs_vec=z_obs,
            alpha=0.05, n_permutations=200, random_state=999
        )
        run2 = _compute_field_significance(
            M_matrix=M, t_numeric=t, var_s_vec=var_s, z_obs_vec=z_obs,
            alpha=0.05, n_permutations=200, random_state=999
        )

        assert run1['p_count'] == run2['p_count']
        assert run1['p_max_z'] == run2['p_max_z']

    def test_strong_trend_field_significance(self):
        # With a strong trend across all quantiles, p_count and p_max_z should be very small
        years = 10
        dates = pd.date_range('2010-01-01', f"{2010+years-1}-12-31", freq='10D')
        t_arr = (dates - dates[0]).total_seconds() / (365.25 * 86400.0)
        rng = np.random.default_rng(42)
        vals = 10.0 + 5.0 * t_arr + rng.normal(0, 0.5, len(dates))

        res = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=5,
            field_significance=True, n_permutations=199, random_state=42,
            min_size_per_block=20, min_blocks=5
        )
        fs = res.field_significance
        assert fs is not None
        assert fs['n_significant'] == 5
        assert fs['p_max_z'] <= 0.01
        assert fs['p_count'] <= 0.10

    def test_stationary_field_significance(self):
        # Stationary series: field significance p-values should not reject
        years = 12
        dates = pd.date_range('2010-01-01', f"{2010+years-1}-12-31", freq='10D')
        rng = np.random.default_rng(777)
        vals = rng.normal(100.0, 10.0, len(dates))

        res = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=5,
            field_significance=True, n_permutations=199, random_state=777,
            min_size_per_block=20, min_blocks=5
        )
        fs = res.field_significance
        assert fs is not None
        # On stationary data, permutation p-values are not near zero
        assert fs['p_count'] > 0.05

    def test_field_significance_two_sided_z_crit_boundary(self):
        """Verify that z_crit uses two-sided alpha (e.g. z=1.80 is not significant at alpha=0.05)."""
        M = np.ones((10, 3))
        t = np.arange(10, dtype=float)
        var_s = np.full(3, 50.0)
        z_obs = np.full(3, 1.80)
        sig_dict = _compute_field_significance(M, t, var_s, z_obs, alpha=0.05, n_permutations=10, random_state=42)
        assert sig_dict['n_significant'] == 0

