"""
Parity tests between MannKS.quantile_trend_test and USGS EGRET.

Two independent references are used:

1. ``TestRealEgret*``: output of the **genuine EGRET R code** (EGRET's own
   ``makeSortQ`` / ``trendSortQ`` run on an EGRET ``eList`` with the EGRET, rkt
   and zyp packages; see ``validation/43_Quantile_Kendall/egret_reference.R``).
   The R output is stored in ``tests/data/egret_qk`` so these tests run without
   R; when R + rpy2 are installed the stored output is also re-derived live.
2. ``TestEgretParity``: a Python port of the same algorithm (kept as a fast,
   dependency-free cross-check of the port itself).

Authoritative reference:
    USGS EGRET R package (Hirsch & De Cicco, 2015; doi:10.3133/tm4A10)
    Vignette: https://doi-usgs.github.io/EGRET/articles/streamflow_trend.html
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from MannKS.quantile_trend_test import quantile_trend_test


# ==============================================================================
# USGS EGRET Reference Implementation (Python Port)
# ==============================================================================

def egret_leap_even(s_vals):
    """
    EGRET leapEven: when n_obs is even and exceeds target by 1,
    averages the two middle values and drops one element.
    """
    n = len(s_vals)
    mid = n // 2
    merged = 0.5 * (s_vals[mid - 1] + s_vals[mid])
    keep = np.ones(n, dtype=bool)
    keep[mid] = False
    out = s_vals[keep].copy()
    out[mid - 1] = merged
    return out


def egret_leap_odd(s_vals):
    """
    EGRET leapOdd: when n_obs is odd and exceeds target by 1,
    drops the middle value.
    """
    n = len(s_vals)
    mid = n // 2
    keep = np.ones(n, dtype=bool)
    keep[mid] = False
    return s_vals[keep].copy()


def egret_make_sort_q(df_daily, pa_start=4):
    """
    EGRET makeSortQ:
    Partitions daily discharge into climate/water years starting in month pa_start,
    verifies complete years (365 or 366 days), downsamples leap years (366 days)
    to minimum year length (365 days) via leapEven, and returns (Y, 365) matrix
    along with decimal year midpoints.
    """
    df = df_daily.sort_values('date').reset_index(drop=True)
    dates = pd.to_datetime(df['date'])
    vals = df['q'].to_numpy(dtype=float)

    # Determine climate year for each day:
    # If pa_start=4 (April), months 4..12 belong to calendar year, months 1..3 belong to calendar year - 1.
    c_years = np.where(dates.dt.month >= pa_start, dates.dt.year, dates.dt.year - 1)
    unique_years = np.unique(c_years)

    valid_year_matrices = []
    midpoint_decimal_years = []

    for yr in unique_years:
        mask = (c_years == yr)
        y_dates = dates[mask]
        y_vals = vals[mask]
        n_days = len(y_vals)

        # Expected full duration for climate year: check if leap year contains Feb 29
        # In climate year starting April yr, February is in calendar year yr + 1.
        has_feb29 = pd.Timestamp(f"{yr+1}-01-01").is_leap_year
        expected_days = 366 if has_feb29 else 365

        # Strict completeness check
        if n_days != expected_days:
            continue

        s_vals = np.sort(y_vals)
        if n_days == 366:
            # 366 is even, so use leapEven to reduce to 365
            reduced_vals = egret_leap_even(s_vals)
        else:
            reduced_vals = s_vals

        valid_year_matrices.append(reduced_vals)
        # Decimal year midpoint: e.g. yr + 0.5 (or midpoint of start date to end date)
        start_ts = pd.Timestamp(f"{yr}-{pa_start:02d}-01")
        end_ts = pd.Timestamp(f"{yr+1}-{pa_start:02d}-01")
        mid_sec = start_ts.timestamp() + (end_ts.timestamp() - start_ts.timestamp()) / 2.0
        # Decimal year
        mid_dt = pd.to_datetime(mid_sec, unit='s')
        dec_yr = mid_dt.year + (mid_dt.day_of_year - 0.5) / (366.0 if mid_dt.is_leap_year else 365.0)
        midpoint_decimal_years.append(dec_yr)

    sort_q_matrix = np.array(valid_year_matrices)
    decimal_years = np.array(midpoint_decimal_years)
    return sort_q_matrix, decimal_years


def egret_trend_sort_q(sort_q_matrix, decimal_years):
    """
    EGRET trendSortQ:
    Computes Mann-Kendall and Theil-Sen trend on log(Q) for each rank column.
    Reports slope as 100 * (exp(b) - 1) % per year.
    """
    Y, M = sort_q_matrix.shape
    log_q = np.log(sort_q_matrix)

    i_idx, j_idx = np.triu_indices(Y, k=1)
    dt = decimal_years[j_idx] - decimal_years[i_idx]

    results = []
    base_var = Y * (Y - 1) * (2 * Y + 5) / 18.0
    n0 = Y * (Y - 1) / 2.0

    for k in range(M):
        y_col = log_q[:, k]
        dy = y_col[j_idx] - y_col[i_idx]
        s_score = float(np.sum(np.sign(dy) * np.sign(dt)))

        # Tie adjustment
        _, counts = np.unique(y_col, return_counts=True)
        tie_counts = counts[counts > 1]
        if len(tie_counts) > 0:
            tie_adj = np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5)) / 18.0
            var_s = max(0.0, base_var - tie_adj)
            n_ties = np.sum(tie_counts * (tie_counts - 1)) / 2.0
            denom = np.sqrt(max(0.0, n0 - n_ties)) * np.sqrt(n0)
            tau = s_score / denom if denom > 0 else 0.0
        else:
            var_s = base_var
            tau = s_score / n0 if n0 > 0 else 0.0

        # Z score and p-value
        if s_score > 0:
            z = (s_score - 1.0) / np.sqrt(var_s)
        elif s_score < 0:
            z = (s_score + 1.0) / np.sqrt(var_s)
        else:
            z = 0.0
        p_val = 2.0 * (1.0 - norm.cdf(abs(z)))

        # Theil-Sen slope
        pairwise_slopes = dy / dt
        sen_slope = float(np.median(pairwise_slopes))
        slope_pct = 100.0 * (np.exp(sen_slope) - 1.0)

        results.append({
            'rank': k + 1,
            's': s_score,
            'var_s': var_s,
            'tau': tau,
            'z': z,
            'p': p_val,
            'slope_log': sen_slope,
            'slope_pct': slope_pct,
        })

    return pd.DataFrame(results)


# ==============================================================================
# Parity Tests
# ==============================================================================

class TestEgretParity:
    """Verifies mathematical parity between Python EGRET reference port and MannKS."""

    @pytest.fixture
    def synthetic_streamflow_data(self):
        """Generates 10 complete climate years (April 1 to March 31) of daily discharge."""
        rng = np.random.default_rng(2026)
        start_date = pd.Timestamp('2000-04-01')
        end_date = pd.Timestamp('2010-03-31')
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Log-normal streamflow with seasonal cycle and mild upward trend
        t_years = (dates - dates[0]).total_seconds() / (365.25 * 86400.0)
        seasonal = 1.0 + 0.5 * np.sin(2.0 * np.pi * dates.day_of_year / 365.25)
        trend = np.exp(0.02 * t_years)
        noise = rng.lognormal(mean=0.0, sigma=0.4, size=len(dates))
        q_vals = 15.0 * seasonal * trend * noise

        return pd.DataFrame({'date': dates, 'q': q_vals})

    def test_egret_sort_q_parity(self, synthetic_streamflow_data):
        # 1. Run EGRET reference port
        sort_q_ref, dec_years_ref = egret_make_sort_q(synthetic_streamflow_data, pa_start=4)
        assert sort_q_ref.shape == (10, 365)
        egret_res = egret_trend_sort_q(sort_q_ref, dec_years_ref)

        # 2. Run MannKS quantile_trend_test with matching EGRET settings
        dates = synthetic_streamflow_data['date']
        vals = synthetic_streamflow_data['q']

        mannks_res = quantile_trend_test(
            vals, dates,
            block_frequency='YS-APR',
            require_complete=True,
            standardization='egret',
            transform='log',
            slope_scaling='year',
            min_size_per_block=300,
            min_blocks=5,
        )
        mannks_df = mannks_res.to_frame()

        # 3. Assert exact match across all 365 quantiles:
        assert len(mannks_df) == 365
        assert len(egret_res) == 365

        # Kendall S scores match exactly
        assert np.array_equal(mannks_df['s'].to_numpy(), egret_res['s'].to_numpy())

        # Var(S) matches to machine precision
        assert np.allclose(mannks_df['var_s'].to_numpy(), egret_res['var_s'].to_numpy(), atol=1e-12)

        # Tau matches to machine precision
        assert np.allclose(mannks_df['Tau'].to_numpy(), egret_res['tau'].to_numpy(), atol=1e-12)

        # Z score and p-values match to machine precision
        assert np.allclose(mannks_df['z'].to_numpy(), egret_res['z'].to_numpy(), atol=1e-10)
        assert np.allclose(mannks_df['p'].to_numpy(), egret_res['p'].to_numpy(), atol=1e-10)

        # Slopes and percent slopes match within calendar day vs decimal year time discretization (< 0.5% relative)
        assert np.allclose(mannks_df['slope_pct'].to_numpy(), egret_res['slope_pct'].to_numpy(), rtol=0.01)

    def test_leap_odd_and_even_unit_parity(self):
        # Test leapOdd
        v_odd = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        res_odd = egret_leap_odd(v_odd)
        assert np.allclose(res_odd, [1.0, 3.0, 7.0, 9.0])

        # Test leapEven
        v_even = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        res_even = egret_leap_even(v_even)
        assert np.allclose(res_even, [2.0, 4.0, 7.0, 10.0, 12.0])


# ==============================================================================
# Parity with the genuine EGRET R code
# ==============================================================================

import importlib.util
import os

_PARITY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                            'validation', '43_Quantile_Kendall', 'egret_parity.py')


def _load_parity_module():
    spec = importlib.util.spec_from_file_location('egret_parity', _PARITY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


egret_parity = _load_parity_module()
_SCENARIOS = sorted(egret_parity.SCENARIOS)


def _assert_matches_egret(mannks_result, egret):
    """MannKS output equals EGRET output (sortQ, S, Var(S), tau, p, slopes, plotting positions)."""
    df = mannks_result.results
    res = egret['results']
    matrix = mannks_result.matrix.to_numpy().T  # ranks x years, EGRET orientation
    assert matrix.shape == egret['sortQ'].shape
    np.testing.assert_allclose(matrix, egret['sortQ'], rtol=1e-14, atol=0)
    np.testing.assert_array_equal(df['s'].to_numpy(), res['S'].to_numpy())
    np.testing.assert_allclose(df['var_s'].to_numpy(), res['varS'].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(df['Tau'].to_numpy(), res['tau'].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(df['p'].to_numpy(), res['pValue'].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(df['slope'].to_numpy(), res['slopeLog'].to_numpy(), rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(df['slope_pct'].to_numpy(), res['slopePct'].to_numpy(), rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(df['plotting_position'].to_numpy(), res['freq'].to_numpy(), atol=1e-14)
    np.testing.assert_allclose(df['z_position'].to_numpy(), res['z'].to_numpy(), atol=1e-12)


class TestRealEgretFixtures:
    """MannKS reproduces stored output of the real EGRET R package (no R needed)."""

    @pytest.mark.parametrize('scenario', _SCENARIOS)
    def test_reproduces_egret(self, scenario):
        daily, egret, sc = egret_parity.load_fixture(scenario)
        res = egret_parity.run_mannks(daily, sc['paStart'], sc['paLong'])
        assert res.computation_mode == 'exact'
        _assert_matches_egret(res, egret)
        assert (res.results['slope_units'] == 'ln(units) per year').all()

    @pytest.mark.parametrize('scenario', _SCENARIOS)
    def test_fast_engine_reproduces_egret(self, scenario):
        daily, egret, sc = egret_parity.load_fixture(scenario)
        kw = egret_parity.mannks_settings(sc['paStart'], sc['paLong'])
        res = quantile_trend_test(daily['q'].to_numpy(), pd.DatetimeIndex(daily['date']), engine='fast', **kw)
        assert res.computation_mode == 'fast'
        _assert_matches_egret(res, egret)

    def test_fixtures_cover_ties_and_leap_odd(self):
        # The scenarios must actually exercise tie corrections and both leap reductions.
        _, eg_ties, _ = egret_parity.load_fixture('water_year_ties')
        var_s = eg_ties['results']['varS'].to_numpy()
        assert (var_s < var_s.max() - 1e-9).sum() > 100
        daily, eg_djf, sc = egret_parity.load_fixture('winter_djf')
        assert eg_djf['sortQ'].shape[0] == 90  # 91-day leap winters reduced by leapOdd
        res = egret_parity.run_mannks(daily, sc['paStart'], sc['paLong'])
        assert sorted(set(res.blocks.loc[res.blocks['valid'], 'n_obs'])) == [90, 91]
        _, eg_cy, _ = egret_parity.load_fixture('climate_year')
        assert eg_cy['sortQ'].shape[0] == 365  # 366-day years reduced by leapEven

    def test_block_time_matches_egret_mid_dec_year(self):
        from MannKS._quantile import _egret_decimal_year
        # EGRET DecYear values printed by EGRET 3.0.12
        got = _egret_decimal_year(pd.to_datetime(['2000-04-01', '2000-12-31', '2001-12-31', '2004-02-29']))
        np.testing.assert_allclose(got, [2000.25, 2000.99863388, 2001.99863014, 2004.16256831], atol=5e-9)
        daily, egret, sc = egret_parity.load_fixture('climate_year')
        d = pd.DatetimeIndex(daily['date'])
        first_year = d[(d >= '2000-04-01') & (d <= '2001-03-31')]
        mid = 0.5 * (_egret_decimal_year(first_year[:1]) + _egret_decimal_year(first_year[-1:]))[0]
        assert mid == pytest.approx(egret['time'][0], abs=1e-10)

    def test_default_block_center_differs_only_in_slope(self):
        # With the default calendar block centre the rank statistics are still
        # EGRET's; only the slope moves (slightly), because the time axis differs.
        daily, egret, sc = egret_parity.load_fixture('water_year_ties')
        kw = egret_parity.mannks_settings(sc['paStart'], sc['paLong'])
        kw['block_time'] = 'center'
        res = quantile_trend_test(daily['q'].to_numpy(), pd.DatetimeIndex(daily['date']), **kw)
        df, ref = res.results, egret['results']
        np.testing.assert_array_equal(df['s'].to_numpy(), ref['S'].to_numpy())
        np.testing.assert_allclose(df['p'].to_numpy(), ref['pValue'].to_numpy(), atol=1e-12)
        ok = np.abs(ref['slopeLog'].to_numpy()) > 1e-6
        rel = np.abs(df['slope'].to_numpy()[ok] / ref['slopeLog'].to_numpy()[ok] - 1.0)
        assert rel.max() < 0.01
        assert rel.max() > 1e-8  # ... but it is not the EGRET slope: block_time='egret' is needed for that

    def test_block_time_validation(self):
        daily, _, _ = egret_parity.load_fixture('winter_djf')
        x, t = daily['q'].to_numpy(), pd.DatetimeIndex(daily['date'])
        with pytest.raises(ValueError, match="`block_time` must be 'center' or 'egret'"):
            quantile_trend_test(x, t, block_time='bogus')
        with pytest.raises(ValueError, match="requires `slope_scaling='year'`"):
            quantile_trend_test(x, t, block_time='egret', slope_scaling='day')
        with pytest.raises(ValueError, match="requires datetime-like `t`"):
            quantile_trend_test(x, np.arange(len(x), dtype=float), block_frequency=365.0, block_time='egret')

    def test_block_time_egret_units_without_transform(self):
        daily, _, _ = egret_parity.load_fixture('climate_year')
        res = quantile_trend_test(daily['q'].to_numpy(), pd.DatetimeIndex(daily['date']),
                                  block_frequency='YS-APR', require_complete=True, quantiles=[0.5],
                                  block_time='egret', x_unit='m3/s')
        row = res.results.iloc[0]
        assert row['slope_units'] == 'm3/s per year'
        assert row['slope_per_second'] == pytest.approx(row['slope'] / (365.25 * 86400.0))


@pytest.mark.skipif(not egret_parity.r_available(),
                    reason="R with EGRET, rkt, zyp and rpy2 not available")
class TestRealEgretLive:
    """Runs the EGRET R code now (via rpy2) and compares it with MannKS and with the stored fixtures."""

    @pytest.mark.parametrize('scenario', _SCENARIOS)
    def test_live_egret_matches_mannks_and_fixture(self, scenario):
        dataset, pa_start, pa_long, _ = egret_parity.SCENARIOS[scenario]
        daily, stored, _ = egret_parity.load_fixture(scenario)
        live = egret_parity.run_egret(daily, pa_start, pa_long)
        _assert_matches_egret(egret_parity.run_mannks(daily, pa_start, pa_long), live)
        # The committed fixture is what EGRET produces
        np.testing.assert_allclose(live['sortQ'], stored['sortQ'], rtol=1e-14)
        np.testing.assert_allclose(live['time'], stored['time'], rtol=1e-14)
        for col in ['slopeLog', 'pValue', 'pValueAdj', 'tau', 'S', 'varS']:
            np.testing.assert_allclose(live['results'][col].to_numpy(), stored['results'][col].to_numpy(),
                                       rtol=1e-9, atol=1e-12)

    def test_stored_dataset_is_the_generated_dataset(self):
        for dataset in ('smooth', 'ties'):
            stored = pd.read_csv(os.path.join(egret_parity.FIXTURE_DIR, f'daily_{dataset}.csv'), parse_dates=['date'])
            gen = egret_parity.make_dataset(dataset)
            assert (stored['date'].to_numpy() == gen['date'].to_numpy()).all()
            np.testing.assert_allclose(stored['q'].to_numpy(), gen['q'].to_numpy(), rtol=1e-15)
