# MannKS Verification Plan

This document outlines a comprehensive plan for verifying the `MannKS` Python package against the original LWP-TRENDS R script. The goal is to ensure that `MannKS`'s LWP-emulation mode produces results that are as close as possible to the R script, and to clearly document any intentional deviations.

## Verification Methodology

For each verification case, five analyses will be performed and their results compared:

1.  **`MannKS` (Standard):** A standard run of the relevant `MannKS` function (`trend_test` or `seasonal_trend_test`) using its default, robust statistical methods.
2.  **`MannKS` (LWP Mode):** A run of the `MannKS` function with all LWP-compatibility parameters enabled. This typically includes:
    *   `mk_test_method='lwp'`
    *   `ci_method='lwp'`
    *   `agg_method='lwp'` (or other LWP-style aggregation)
    *   `sens_slope_method='lwp'`
    *   `tie_break_method='lwp'` (for non-seasonal tests)
3.  **LWP-TRENDS R Script:** A run of the original `LWPTrends_v2502.r` script using `rpy2` to execute the analysis on the identical synthetic dataset.
4.  **`MannKS` (ATS Mode):** A run of the `MannKS` function using the Akritas-Theil-Sen estimator (`sens_slope_method='ats'`).
5.  **R NADA2 Script:** A run of the relevant NADA2 function (`cenken` for non-seasonal, `censeaken` for seasonal) using `rpy2` to execute the analysis on the identical synthetic dataset.

All data used for these tests will be synthetically generated to provide full control over the inputs and expected outcomes. The majority of tests will use monthly data frequency, with a smaller subset using annual data to ensure both are handled correctly.

### Scenario Structure

For each numbered verification case below (e.g., V-01), **three distinct scenarios** will be tested to ensure robustness across different trend characteristics:
*   **Strong Increasing Trend:** A clear, statistically significant positive trend.
*   **Weak Decreasing Trend:** A subtle, statistically significant (or borderline) negative trend.
*   **Stable (No Trend):** Data with no underlying trend, expected to be non-significant.

The results for all three scenarios will be presented in a single summary table for each verification case. No new plots are required for this expanded scope; the focus is on the tabular comparison of statistical results.

## Master Results Tracking

A master CSV file, `validation/master_results.csv`, will be created and updated as part of the verification process. This file will consolidate the key outputs from every single test run. For each test (e.g., "V-01_strong_increasing"), the CSV will contain a row with the following columns:
*   `test_id`: A unique identifier (e.g., "V-01_strong_increasing").
*   `mk_py_slope`, `mk_py_p_value`, `mk_py_lower_ci`, `mk_py_upper_ci`: Results from `MannKS` (Standard).
*   `lwp_py_slope`, `lwp_py_p_value`, `lwp_py_lower_ci`, `lwp_py_upper_ci`: Results from `MannKS` (LWP Mode).
*   `r_slope`, `r_p_value`, `r_lower_ci`, `r_upper_ci`: Results from the LWP-TRENDS R Script.
*   `ats_py_slope`, `ats_py_p_value`, `ats_py_lower_ci`, `ats_py_upper_ci`: Results from `MannKS` (ATS Mode).
*   `nada_r_slope`, `nada_r_p_value`, `nada_r_lower_ci`, `nada_r_upper_ci`: Results from the R NADA2 Script.
*   `slope_error`: The absolute error (`lwp_py_slope` - `r_slope`).
*   `slope_pct_error`: The percentage error, calculated as `(slope_error / r_slope) * 100`.

This master file will provide a quantitative, high-level overview of the accuracy and consistency across all methods.

---

## Verification Scenarios

### Category 1: Basic Non-Seasonal Trends

**V-01: Simple Trend**
*   **Objective:** Verify that all methods correctly identify trends in a simple, non-seasonal, uncensored dataset.
*   **Data Description:** Annual linear data with random noise. This test uses annual data to ensure compatibility with the `NonSeasonalTrendAnalysis` function in the R script, which is brittle with other data frequencies.

**V-02: Data with Tied Values**
*   **Objective:** Test the handling of tied values in the data vector.
*   **Data Description:** Monthly data with a clear trend but multiple repeated values.

**V-03: Kendall's Tau Method (`tau_method`)**
*   **Objective:** Verify the `tau_method` parameter, comparing Tau-a and Tau-b.
*   **Data Description:** Monthly data with a high number of tied values to highlight the difference between Tau-a (no tie correction) and Tau-b (tie correction).

**V-04: Aggregation Methods for Tied Timestamps (`agg_method`)**
*   **Objective:** Verify the various `agg_method` options for handling tied timestamps.
*   **Data Description:** Data with multiple observations at the same timestamp, testing 'median', 'robust_median', 'middle', and 'middle_lwp' aggregation methods.

**V-05: Unequally Spaced Time Series**
*   **Objective:** Verify a core feature of `MannKS` on a non-seasonal, unequally spaced time series.
*   **Data Description:** Data with a clear trend but with random, non-uniform time gaps between samples. This test highlights a key methodological difference where the R script is expected to differ.

**V-06: Numeric Time Vector**
*   **Objective:** Verify that the functions work correctly with a simple numeric time vector instead of datetimes.
*   **Data Description:** Annual data with a trend, where the time vector is a simple array of numbers (e.g., years as `[2000, 2001.5, 2002.7]`).

**V-07: Slope Scaling (`slope_scaling`)**
*   **Objective:** Verify the `slope_scaling` feature for converting the slope and CIs into user-friendly units.
*   **Data Description:** Monthly data with a known trend, to verify that the slope is correctly scaled to 'per year'.

### Category 2: Censored Data

**V-08: Left-Censored Trend**
*   **Objective:** Verify the handling of left-censored (`<`) data.
*   **Data Description:** Monthly data with a trend where some of the lower values are left-censored.

**V-09: Right-Censored Trend**
*   **Objective:** Verify the handling of right-censored (`>`) data.
*   **Data Description:** Monthly data with a trend where some of the higher values are right-censored.

**V-10: Mixed Censoring**
*   **Objective:** Test a combination of left- and right-censored data.
*   **Data Description:** A monthly dataset containing both `<` and `>` values.

**V-11: High Censor Rule (`hicensor`)**
*   **Objective:** Verify the implementation of the "high censor" rule.
*   **Data Description:** Monthly data with multiple censoring levels, where some uncensored values fall below the highest censoring limit.

**V-12: Sen's Slope Censored Multipliers (`lt_mult`, `gt_mult`)**
*   **Objective:** Isolate and verify the effect of the `lt_mult` and `gt_mult` parameters.
*   **Data Description:** A carefully crafted censored dataset where the median Sen's slope is sensitive to the multipliers, demonstrating that they only affect the slope and not the MK test itself.

**V-13: Censored LWP Compatibility Modes**
*   **Objective:** Explicitly compare the different methods for handling censored data (`mk_test_method`, `sens_slope_method`, `ci_method`).
*   **Data Description:** A right-censored dataset used to verify that the 'lwp' compatibility modes correctly replicate the R script's behavior.

### Category 3: Seasonal Data

**V-14: Monthly Seasonal Trend**
*   **Objective:** Verify the seasonal test on a simple monthly dataset.
*   **Data Description:** Monthly data with an overall trend and a clear seasonal cycle.

**V-15: Quarterly Seasonal Data**
*   **Objective:** Test seasonality with a different period (quarterly).
*   **Data Description:** Quarterly data with an underlying trend.

**V-16: Monthly Seasonal with Left-Censoring**
*   **Objective:** Test the combination of monthly seasonality and left-censored data.
*   **Data Description:** Monthly data with a trend, a seasonal cycle, and some left-censored values.

**V-17: Monthly Seasonal with Right-Censoring**
*   **Objective:** Test the combination of monthly seasonality and right-censored data.
*   **Data Description:** Monthly data with a trend, a seasonal cycle, and some right-censored values.

**V-18: Seasonal Data with Missing Seasons**
*   **Objective:** Verify behavior when entire seasons are missing from the data.
*   **Data Description:** A seasonal dataset with one or more seasons (e.g., all "July" data) completely removed.

**V-19: Alternative Seasonal Patterns (`season_type`)**
*   **Objective:** Test less common but important seasonal patterns using the `season_type` parameter.
*   **Data Description:** Data designed to test 'week_of_year' and 'day_of_year' seasonality to ensure the datetime logic is robust.

### Category 4: Helper Function Validation

**V-20: Seasonality Check (`check_seasonality`)**
*   **Objective:** Verify the `check_seasonality` helper function.
*   **Data Description:** One dataset with a strong seasonal pattern and another with no seasonal pattern to confirm the function correctly identifies them.

**V-21: Trend Classification (`classify_trend`)**
*   **Objective:** Verify the `classify_trend` function, including the use of a custom `category_map`.
*   **Data Description:** A series of trend results with varying confidence levels will be used to test both the default and a custom classification map.

**V-22: Data Inspection (`inspect_trend_data`)**
*   **Objective:** Validate the `inspect_trend_data` utility.
*   **Data Description:** A dataset with a mix of censored and non-censored data will be used to ensure the function correctly processes and visualizes the data without error.

### Category 5: Regional Trend Test

**V-23: Basic Regional Trend**
*   **Objective:** Verify the `regional_test` function with a simple case.
*   **Data Description:** Data for 3-5 sites, all exhibiting a similar underlying trend.

**V-24: Regional Trend with Mixed-Direction Sites**
*   **Objective:** Verify the regional test's behavior when sites have conflicting trends.
*   **Data Description:** Data for multiple sites, where some are increasing, some are decreasing, and some are stable.

**V-25: Regional Trend with High Inter-site Correlation**
*   **Objective:** Test the variance correction with highly correlated sites.
*   **Data Description:** Data for multiple sites that have a high positive or negative correlation to confirm the variance correction is working.

**V-26: Regional Trend with Insufficient Site Data**
*   **Objective:** Test how the regional test behaves when some sites have insufficient data.
*   **Data Description:** A set of trend results where one or more sites have `nan` results and must be excluded from the aggregate analysis.

### Category 6: Data Quality and Edge Cases

**V-27: LWP Aggregation (`agg_method='lwp'`)**
*   **Objective:** Verify the LWP-style temporal aggregation.
*   **Data Description:** A dataset with multiple observations per time period (e.g., several samples per month).

**V-28: Insufficient Data (`min_size`)**
*   **Objective:** Verify how systems handle datasets that are too small for a valid trend test.
*   **Data Description:** A dataset with very few data points (e.g., n < 5) to test the `min_size` and `min_size_per_season` parameters.

**V-29: All Values Censored**
*   **Objective:** Verify behavior when all data points are censored.
*   **Data Description:** A dataset where every value is a left-censored value (e.g., `<5`, `<5`, `<5`).

**V-30: Long Run of Identical Values**
*   **Objective:** Verify the data quality check for long runs of a single value.
*   **Data Description:** A dataset where a large proportion of the values are identical and consecutive.

**V-31: Data Quality Analysis Notes**
*   **Objective:** Explicitly trigger and verify each of the data quality warnings from `analysis_notes.py`.
*   **Data Description:** Several small, targeted datasets will be created to trigger specific warnings (e.g., 'Sen slope based on censored data', 'denominator is zero') to ensure they are produced under the correct conditions.

**V-32: Classification Sensitivity Sweep**
*   **Objective:** Verify that `MannKS` and the LWP R script produce identical trend classifications across a wide range of slope and noise combinations, specifically targeting transition zones between confidence categories.
*   **Data Description:** A large set (e.g., 50-100) of synthetic datasets with systematically varied signal-to-noise ratios.

**V-33: Censored Trend Sensitivity Sweep (Non-Seasonal)**
*   **Objective:** Verify Sen's slope, p-value, confidence intervals, and classification for non-seasonal data with varying mixed censoring levels (mixed detection limits).
*   **Data Description:** A parameter sweep of synthetic datasets featuring different slopes, noise levels, and censoring intensities (e.g., 20%, 50%) with multiple detection limits.

**V-34: Seasonal Trend Sensitivity Sweep**
*   **Objective:** Verify Sen's slope, p-value, confidence intervals, and classification for seasonal data, including scenarios with mixed censoring.
*   **Data Description:** A parameter sweep similar to V-33 but applied to seasonal (e.g., monthly) data patterns.

**V-37: Bootstrap CI Comparison**
*   **Objective:** Compare confidence intervals generated by the analytic method vs. the block bootstrap method.
*   **Data Description:** Synthetic datasets with varying linear slopes and noise levels, for both non-seasonal and seasonal data. This validates that the bootstrap method produces reasonable confidence intervals compared to the standard analytic approach (when assumptions hold) and highlights differences where they diverge.
