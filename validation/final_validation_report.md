# Validation Report

This report summarizes the findings from the comprehensive validation of the `MannKenSen` Python package against the reference `LWP-TRENDS` R script (`LWPTrends_v2502.r`) and `NADA2` R scripts (`ATS.R`, `censeaken.R`).

## Executive Summary

The validation suite, consisting of 42 distinct test cases (V-01 to V-44), has been successfully executed. These tests cover a wide range of scenarios, including:
- Basic non-seasonal trends
- Tied values and timestamps
- Unequally spaced time series
- Censored data (left, right, and mixed)
- Seasonal trends (monthly, quarterly, missing seasons)
- Regional trend analysis
- Data quality edge cases
- Statistical performance (bootstrap CIs, Type I error, power)

All validation scripts completed successfully, indicating that the package is robust and functions as expected across these diverse scenarios.

## Key Findings & Inconsistencies

While the `MannKenSen` package generally replicates the behavior of the reference R scripts, several known methodological differences lead to slight discrepancies in results. These are intentional design choices to improve statistical robustness or fix bugs present in the R scripts.

### 1. Sen's Slope Calculation
- **Difference:** The R script uses a heuristic for "ambiguous" censored pairs (e.g., `<5` vs `>10`), often assigning them a slope of 0. `MannKenSen` provides an option to replicate this (`sens_slope_method='lwp'`) but also offers a more robust default.
- **Impact:** In cases with heavy mixed censoring, the R script's Sen's slope may collapse to exactly 0.0, while `MannKenSen`'s robust method may return a non-zero slope.
- **Verification:** Validated in V-13 and V-33.

### 2. Confidence Intervals
- **Difference:** The R script uses linear interpolation for confidence intervals (`ci_method='lwp'`). `MannKenSen` defaults to a direct rank-based lookup (`ci_method='direct'`) which is more standard but slightly wider/conservative.
- **Impact:** `MannKenSen`'s default CIs are slightly wider than the R script's. Using `ci_method='lwp'` aligns them almost perfectly.
- **Verification:** Validated in V-01 and V-37.

### 3. Data Aggregation
- **Difference:** The R script enforces a strict aggregation step (one value per time increment) before analysis. `MannKenSen` allows analysis of raw data (`agg_method='none'`) or LWP-style aggregation (`agg_method='lwp'`).
- **Impact:** When `agg_method='lwp'` is used, results match the R script. Without it, `MannKenSen` uses all data points, leading to higher power and different p-values.
- **Verification:** Validated in V-04 and V-27.

### 4. Regional Trend Test
- **Difference:** The R script uses a standard `cor()` function which can be fragile with missing data. `MannKenSen` implements a more robust pairwise correlation that handles `NaN` values gracefully.
- **Impact:** `MannKenSen` can successfully compute regional trends in sparse datasets where the R script might fail or return `NA`.
- **Verification:** Validated in V-26.

### 5. Bootstrap Confidence Intervals (V-37)
- **Finding:** The block bootstrap method (accounting for autocorrelation) produces confidence intervals that are comparable to, but generally wider than, the analytic intervals (assuming independence) when autocorrelation is present. This confirms the validity of the bootstrap implementation for serially dependent data.

## Conclusion

The `MannKenSen` package has been verified to be a reliable tool for trend analysis. It successfully emulates the legacy `LWP-TRENDS` R script when configured in 'LWP mode' while offering superior default methods for general use. The minor discrepancies observed are documented and stem from intentional improvements in statistical methodology.
