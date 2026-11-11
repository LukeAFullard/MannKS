"""
Unit tests for Generalized Quantile-Kendall trend analysis on monthly sampled datasets
with missing values, incomplete years, and censored detection limits.

Target module: MannKS.quantile_trend_test
"""
import numpy as np
import pandas as pd
import pytest

from MannKS import (
    quantile_trend_test,
    prepare_censored_data,
    summarize_quantile_regimes,
)


class TestMonthlyQuantileAnalysis:
    """Tests evaluating monthly sampling characteristics (10-25 years, missing months, censoring)."""

    def test_monthly_with_missing_months_and_incomplete_years(self):
        # 15 years of monthly data (2010 to 2024 = 180 planned observations)
        rng = np.random.default_rng(101)
        dates = pd.date_range("2010-01-01", "2024-12-31", freq="MS") + pd.Timedelta(days=14)
        n = len(dates)
        t_years = (dates - dates[0]).total_seconds() / (365.25 * 86400.0)

        # Baseline downward trend
        base = 10.0 - 0.2 * t_years + rng.normal(0, 0.5, size=n)

        # Ingest missing months (~15% random missing)
        missing_mask = rng.random(n) < 0.15

        # Year 2018 has only 3 samples collected (below min_size_per_block=8)
        year_2018_mask = (dates.year == 2018) & (dates.month > 3)
        missing_mask = missing_mask | year_2018_mask

        dates_valid = dates[~missing_mask]
        vals_valid = base[~missing_mask]

        res = quantile_trend_test(
            x=vals_valid,
            t=dates_valid,
            block_frequency="YE",
            quantiles=[0.25, 0.50, 0.75],
            min_size_per_block=8,
            min_blocks=10,
            slope_scaling="year",
        )

        df = res.to_frame()
        assert len(df) == 3
        # Year 2018 must be dropped, leaving at most 14 valid annual blocks
        assert (df["n_blocks"] <= 14).all()
        assert (df["n_blocks"] >= 10).all()
        # All quantiles should detect the downward trend
        assert (df["slope"] < 0.0).all()
        assert (df["p"] < 0.05).all()

    def test_monthly_censored_with_changing_detection_limits(self):
        # 16 years of monthly data (2005 to 2020)
        rng = np.random.default_rng(202)
        dates = pd.date_range("2005-01-01", "2020-12-31", freq="MS") + pd.Timedelta(days=14)
        t_years = (dates - dates[0]).total_seconds() / (365.25 * 86400.0)

        # Divergent process:
        # Base concentration decreases (improves)
        clean_base = 0.06 * np.exp(-0.03 * t_years)
        # Periodic storm flushes in winter increase (worsens)
        flush = np.where(dates.month.isin([12, 1, 2]), 0.20 * (1.0 + 0.05 * t_years), 0.0)
        true_conc = np.maximum(0.005, clean_base + flush + rng.normal(0, 0.005, size=len(dates)))

        # Historical detection limits: <0.04 mg/L before 2013, <0.02 mg/L in 2013 and later
        raw_strings = []
        for d, val in zip(dates, true_conc):
            dl = 0.04 if d.year < 2013 else 0.02
            if val < dl:
                raw_strings.append(f"<{dl}")
            else:
                raw_strings.append(f"{val:.4f}")

        cens_df = prepare_censored_data(raw_strings)

        res = quantile_trend_test(
            x=cens_df,
            t=dates,
            block_frequency="YE",
            quantiles=[0.10, 0.25, 0.50, 0.75, 0.90],
            min_size_per_block=10,
            censor_quantile_method="propagate",
            sens_slope_method="ats",
            slope_scaling="year",
        )

        df = res.to_frame()
        assert len(df) == 5

        # Lowest quantile should have non-zero prop_censored
        assert df.loc[df["quantile"] == 0.10, "prop_censored"].iloc[0] > 0.0

        # Upper quantiles (flushes) should be completely uncensored
        assert df.loc[df["quantile"] == 0.90, "prop_censored"].iloc[0] == 0.0

        # Divergent trends: low quantiles decreasing, high quantiles increasing
        assert df.loc[df["quantile"] == 0.10, "slope"].iloc[0] < 0.0
        assert df.loc[df["quantile"] == 0.90, "slope"].iloc[0] > 0.0

        # Regimes summary
        regimes = summarize_quantile_regimes(res)
        assert len(regimes) == 3
        assert "Decreasing" in regimes.loc[regimes["regime"] == "low", "modal_classification"].iloc[0]
        assert "Increasing" in regimes.loc[regimes["regime"] == "high", "modal_classification"].iloc[0]
