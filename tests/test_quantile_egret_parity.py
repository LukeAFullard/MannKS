"""
Python reference port of USGS EGRET makeSortQ + leapOdd/leapEven + trendSortQ,
and parity tests against MannKS.quantile_trend_test(standardization='egret').

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
