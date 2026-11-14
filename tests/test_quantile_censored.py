"""
Unit tests for censored data handling, censor propagation without plateauing,
substitution, error modes, and censored DataFrame integration in Quantile-Kendall analysis.

Target modules: MannKS._quantile, MannKS.quantile_trend_test
"""
import numpy as np
import pandas as pd
import pytest

from MannKS._quantile import _standardize_block
from MannKS.quantile_trend_test import quantile_trend_test


class TestCensorPropagationUnit:
    """Unit tests for _standardize_block censor propagation."""

    def test_censor_propagation_without_plateau(self):
        # 9 uncensored values [1, 2, 3, 4, 5, 6, 7, 8, 9] and 1 detection limit < 5.0
        # Total 10 observations.
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 5.0])
        cens = np.array([False] * 9 + [True])
        types = np.array(['not'] * 9 + ['lt'], dtype=object)
        df = pd.DataFrame({'value': vals, 'censored': cens, 'cen_type': types})

        std_vals, std_cens, std_type = _standardize_block(
            group_df=df, M=10, standardization='interpolate',
            quantile_method='linear', censor_quantile_method='propagate',
            lt_mult=0.5, gt_mult=1.1,
        )

        # Monotonicity must be preserved
        assert np.all(np.diff(std_vals) >= 0)

        # Lower uncensored values 1.0, 2.0, 3.0, 4.0 must NOT be plateaued/overwritten with 5.0
        assert std_vals[0] == 1.0
        assert not std_cens[0]
        assert std_vals[1] == 2.0
        assert not std_cens[1]
        assert std_vals[2] == 3.0
        assert not std_cens[2]
        assert std_vals[3] == 4.0
        assert not std_cens[3]

        # The rank corresponding to the detection limit 5.0 must be flagged censored lt
        assert std_vals[4] == 5.0
        assert std_cens[4]
        assert std_type[4] == 'lt'

        # Values above 5.0 must be uncensored
        assert std_vals[-1] == 9.0
        assert not std_cens[-1]

    def test_right_censor_propagation(self):
        # Right-censored observation > 50.0
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
        cens = np.array([False, False, False, False, True, False, False])
        types = np.array(['not', 'not', 'not', 'not', 'gt', 'not', 'not'], dtype=object)
        df = pd.DataFrame({'value': vals, 'censored': cens, 'cen_type': types})

        std_vals, std_cens, std_type = _standardize_block(
            group_df=df, M=7, standardization='interpolate',
            quantile_method='linear', censor_quantile_method='propagate',
            lt_mult=0.5, gt_mult=1.1,
        )
        assert np.all(np.diff(std_vals) >= 0)
        # Rank 4 (50.0) should be flagged gt
        assert std_cens[4]
        assert std_type[4] == 'gt'
        # Rank 0 (10.0) uncensored
        assert not std_cens[0]

    def test_censor_substitute_mode(self):
        vals = np.array([10.0, 20.0, 30.0])
        cens = np.array([True, False, True])
        types = np.array(['lt', 'not', 'gt'], dtype=object)
        df = pd.DataFrame({'value': vals, 'censored': cens, 'cen_type': types})

        std_vals, std_cens, std_type = _standardize_block(
            group_df=df, M=3, standardization='interpolate',
            quantile_method='linear', censor_quantile_method='substitute',
            lt_mult=0.5, gt_mult=1.5,
        )
        # In substitute mode, censored flags are reset to False
        assert not std_cens.any()
        assert (std_type == 'not').all()
        # 10.0 * 0.5 = 5.0, 20.0, 30.0 * 1.5 = 45.0
        assert np.allclose(std_vals, [5.0, 20.0, 45.0])

    def test_censor_error_mode_raises(self):
        vals = np.array([10.0, 20.0])
        cens = np.array([True, False])
        types = np.array(['lt', 'not'], dtype=object)
        df = pd.DataFrame({'value': vals, 'censored': cens, 'cen_type': types})

        with pytest.raises(ValueError, match="Censored observations encountered with censor_quantile_method='error'"):
            _standardize_block(
                group_df=df, M=2, standardization='interpolate',
                quantile_method='linear', censor_quantile_method='error',
                lt_mult=0.5, gt_mult=1.1,
            )

    def test_hazen_exact_censor_bracketing_and_node_snapping(self):
        # 12 sorted values: index 0 is uncensored 1.0, index 1 is <2.0,
        # index 6 is uncensored 7.0 (while index 5 is <6.0 and index 7 is >8.0),
        # index 10 is >11.0, index 11 is uncensored 12.0.
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        cens = np.array([False, True, False, False, False, True, False, True, False, False, True, False])
        types = np.array(['not', 'lt', 'not', 'not', 'not', 'lt', 'not', 'gt', 'not', 'not', 'gt', 'not'], dtype=object)
        df = pd.DataFrame({'value': vals, 'censored': cens, 'cen_type': types})

        # Under Hazen (n=12):
        # - u=0.04 -> v = 12*0.04 - 0.5 = -0.02 -> clipped to 0.0 (only index 0 uncensored 1.0; no bleed from idx 1 <2.0)
        # - u=0.96 -> v = 12*0.96 - 0.5 = 11.02 -> clipped to 11.0 (only index 11 uncensored 12.0; no bleed from idx 10 >11.0)
        u_grid = np.array([0.04, 0.96])
        std_vals, std_cens, std_type = _standardize_block(
            group_df=df, M=2, standardization='interpolate',
            quantile_method='hazen', censor_quantile_method='propagate',
            lt_mult=0.5, gt_mult=1.1, u_grid=u_grid,
        )
        assert np.allclose(std_vals, [1.0, 12.0])
        assert not std_cens.any()
        assert list(std_type) == ['not', 'not']

        # Exact integer node snapping under Hazen (n=10, u=0.55):
        # 10*0.55 - 0.5 = 5.000000000000001 in IEEE-754 float; must snap to index 5 (not ceil to 6)
        vals10 = np.arange(1.0, 11.0)
        cens10 = np.zeros(10, dtype=bool)
        types10 = np.full(10, 'not', dtype=object)
        cens10[6] = True
        types10[6] = 'gt'
        df10 = pd.DataFrame({'value': vals10, 'censored': cens10, 'cen_type': types10})
        sv10, sc10, st10 = _standardize_block(
            group_df=df10, M=1, standardization='interpolate',
            quantile_method='hazen', censor_quantile_method='propagate',
            lt_mult=0.5, gt_mult=1.1, u_grid=np.array([0.55]),
        )
        assert np.isclose(sv10[0], 6.0)
        assert not sc10[0]
        assert st10[0] == 'not'



class TestCensoredQuantileTrendIntegration:
    """Integration tests for quantile_trend_test with censored inputs."""

    def test_censored_dataframe_input(self):
        # Construct 6 annual blocks with lower detection limits (< 2.0)
        years = 6
        n_per_year = 30
        dates_list = []
        vals_list = []
        cens_list = []
        types_list = []

        base_year = 2010
        for y_idx in range(years):
            yr = base_year + y_idx
            d = pd.date_range(f"{yr}-01-01", f"{yr}-12-31", freq='10D')
            n_obs_year = len(d)
            dates_list.extend(d)
            # Simulated concentration increasing from year to year
            raw_v = 1.0 + 0.5 * y_idx + np.linspace(0.1, 5.0, n_obs_year)
            for v in raw_v:
                if v < 2.0:
                    vals_list.append(2.0)
                    cens_list.append(True)
                    types_list.append('lt')
                else:
                    vals_list.append(v)
                    cens_list.append(False)
                    types_list.append('not')

        cens_df = pd.DataFrame({
            'value': vals_list,
            'censored': cens_list,
            'cen_type': types_list,
        })
        dates_all = pd.DatetimeIndex(dates_list)

        res = quantile_trend_test(
            cens_df, dates_all, block_frequency='YE', quantiles=5,
            min_size_per_block=20, min_blocks=5,
        )
        df = res.to_frame()
        assert not df.empty
        assert len(df) == 5

        # Lower ranks should have non-zero prop_censored
        assert df['prop_censored'].iloc[0] > 0.0
        # Upper ranks should be completely uncensored (prop_censored == 0)
        assert df['prop_censored'].iloc[-1] == 0.0

        # Slopes for upper ranks should be positive reflecting the trend
        assert df['slope'].iloc[-1] > 0.0

    def test_uncensored_quantiles_above_limit_match(self):
        # Create identical dataset with uncensored values and left-censored low values.
        # Upper quantiles (e.g. 90th percentile) well above detection limit should yield
        # identical results between uncensored and censored datasets.
        years = 8
        n_per_year = 40
        dates_list = []
        uncens_vals = []
        cens_vals = []
        cens_flags = []
        cen_types = []

        for y_idx in range(years):
            yr = 2010 + y_idx
            d = pd.date_range(f"{yr}-01-01", f"{yr}-12-31", freq='7D')
            n_obs_year = len(d)
            dates_list.extend(d)
            v = 10.0 + 1.2 * y_idx + np.linspace(1.0, 20.0, n_obs_year)
            uncens_vals.extend(v)
            for val in v:
                if val < 13.0:
                    cens_vals.append(13.0)
                    cens_flags.append(True)
                    cen_types.append('lt')
                else:
                    cens_vals.append(val)
                    cens_flags.append(False)
                    cen_types.append('not')

        dates = pd.DatetimeIndex(dates_list)
        df_cens = pd.DataFrame({'value': cens_vals, 'censored': cens_flags, 'cen_type': cen_types})

        res_uncens = quantile_trend_test(
            uncens_vals, dates, block_frequency='YE', quantiles=[0.9],
            min_size_per_block=30, min_blocks=5
        ).to_frame()

        res_cens = quantile_trend_test(
            df_cens, dates, block_frequency='YE', quantiles=[0.9],
            min_size_per_block=30, min_blocks=5
        ).to_frame()

        assert np.isclose(res_uncens['slope'].iloc[0], res_cens['slope'].iloc[0], atol=1e-5)
        assert np.isclose(res_uncens['p'].iloc[0], res_cens['p'].iloc[0], atol=1e-5)
        assert res_cens['prop_censored'].iloc[0] == 0.0

    def test_log_transform_censored_sens_slope_method_guard(self):
        dates = []
        vals = []
        cens = []
        types = []
        for yr in range(2010, 2016):
            d = pd.date_range(f"{yr}-01-01", f"{yr}-12-31", freq='15D')
            dates.extend(d)
            v = np.linspace(1.0, 10.0, len(d)) + 0.5 * (yr - 2010)
            vals.extend(np.maximum(v, 2.0))
            c = v < 2.0
            cens.extend(c)
            types.extend(np.where(c, 'lt', 'not'))
        df_cens = pd.DataFrame({'value': vals, 'censored': cens, 'cen_type': types})
        t_idx = pd.DatetimeIndex(dates)

        # Propagated censoring + log transform must reject lwp and drop_ambiguous
        for bad_method in ('lwp', 'drop_ambiguous'):
            with pytest.raises(ValueError, match="requires `sens_slope_method='ats'`"):
                quantile_trend_test(
                    df_cens, t_idx, block_frequency='YE', quantiles=[0.05, 0.5, 0.95],
                    min_size_per_block=10, min_blocks=5,
                    transform='log', censor_quantile_method='propagate',
                    sens_slope_method=bad_method,
                )

        # ats with propagate must succeed
        res_ats = quantile_trend_test(
            df_cens, t_idx, block_frequency='YE', quantiles=[0.05, 0.5, 0.95],
            min_size_per_block=10, min_blocks=5,
            transform='log', censor_quantile_method='propagate',
            sens_slope_method='ats',
        )
        assert res_ats.computation_mode == 'exact'
        assert res_ats.results['slope_units'].iloc[0] == 'ln(units) per year'

        # substitute with lwp must succeed
        res_sub = quantile_trend_test(
            df_cens, t_idx, block_frequency='YE', quantiles=[0.05, 0.5, 0.95],
            min_size_per_block=10, min_blocks=5,
            transform='log', censor_quantile_method='substitute',
            sens_slope_method='lwp',
        )
        assert res_sub.computation_mode == 'exact'

