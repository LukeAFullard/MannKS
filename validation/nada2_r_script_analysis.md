# NADA2 R Script: Seasonal and Non-Seasonal ATS Analysis Guide

This document provides a detailed breakdown of the key functions and their parameters within the NADA2 R scripts (`ATS.R` and `censeaken.R`) located in `Example_Files/R/NADA2`. It serves as a reference for understanding the methodology of the Akritas-Theil-Sen (ATS) trend analysis used in the NADA2 package.

## Core Trend Analysis Functions

These are the primary functions for performing trend analysis on censored data using the Akritas-Theil-Sen (ATS) method.

### 1. `ATS` (Non-Seasonal Analysis)

*   **Source File:** `ATS.R`
*   **Purpose:** Performs a non-seasonal Mann-Kendall test and calculates the Akritas-Theil-Sen line (slope and intercept) for censored data. It handles censoring in both the response variable (`y`) and the explanatory variable (`x`), though typically only `y` is censored in trend analysis.
*   **Signature:** `ATS(y.var, y.cen, x.var, x.cen = rep(0, length(x.var)), LOG = TRUE, retrans = FALSE, ...)`
*   **Parameters:**
    *   `y.var`: Numeric vector of the response variable (e.g., concentration). Includes both detected values and detection limits.
    *   `y.cen`: Logical or numeric vector indicating censoring for `y.var`. `TRUE` (or `1`) indicates a censored value (detection limit), and `FALSE` (or `0`) indicates a detected value.
    *   `x.var`: Numeric vector of the explanatory variable (typically time).
    *   `x.cen`: Logical/numeric vector indicating censoring for `x.var`. Defaults to all zeros (uncensored), which is standard for time.
    *   `LOG` (boolean, default: `TRUE`): If `TRUE`, the analysis is performed on the natural logarithm of `y.var`.
        *   **Important:** This defaults to `TRUE`. If your data is already log-transformed or if you want a linear trend in original units, you must explicitly set `LOG = FALSE`.
    *   `retrans` (boolean, default: `FALSE`): If `TRUE` (and `LOG=TRUE`), the plot will show the curve retransformed back to original units.
*   **Output:**
    *   Prints statistics: Kendall's tau, p-value, slope, and intercept.
    *   Generates a plot with the trend line.
    *   Returns a `data.frame` (invisible) with columns: `intercept`, `slope`, `S`, `tau`, `pval`.

### 2. `censeaken` (Seasonal Analysis)

*   **Source File:** `censeaken.R`
*   **Purpose:** Performs a Seasonal Kendall test on censored data. It computes the Kendall-S statistic for each season and sums them for the overall test. Significance is determined via a permutation test.
*   **Signature:** `censeaken(time, y, y.cen, group, data = NULL, LOG = FALSE, R = 4999, nmin = 4, ...)`
*   **Parameters:**
    *   `time`: Numeric vector of the time variable.
    *   `y`: Numeric vector of the response variable.
    *   `y.cen`: Logical/numeric vector indicating censoring (TRUE=censored).
    *   `group`: Vector (factor) defining the seasons (e.g., "Spring", "Summer" or numeric months).
    *   `LOG` (boolean, default: `FALSE`): **Note the difference from `ATS`.** Defaults to `FALSE`. Set to `TRUE` to analyze log-transformed data.
    *   `R` (integer, default: `4999`): The number of permutations to perform for calculating the overall p-value.
    *   `nmin` (integer, default: `4`): The minimum number of observations required per season. Seasons with fewer observations are dropped from the analysis.
    *   `seaplots` (boolean, default: `FALSE`): If `TRUE`, plots a trend line for each individual season in addition to the overall plot.
*   **Methodology Notes:**
    *   **Slope Calculation:** The overall slope is calculated by applying the non-seasonal ATS estimator to the **entire dataset** (ignoring seasons). It does *not* calculate the median of seasonal slopes (like the Seasonal Sen's Slope).
        *   Code reference: `ats_all <- ATSmini(dat$y, dat$y.cen, dat$time)`
    *   **P-Value:** Calculated using a permutation test where data is shuffled within seasons to preserve seasonal variance but break the time-trend relationship.
*   **Output:**
    *   Prints per-season statistics (N, S, Tau, P-value, Intercept, MedianSlope).
    *   Prints and returns an overall `RESULTS` dataframe containing: `S_SK` (Sum of S), `tau_SK` (overall Tau), `pval` (permutation-based), `intercept`, and `slope`.

### 3. `ATSmini` (Internal Helper)

*   **Source File:** `ATSmini.R`
*   **Purpose:** A streamlined version of `ATS` designed for speed. It is used internally by `censeaken` to calculate stats for each season and for the overall dataset during permutations.
*   **Differences from `ATS`:**
    *   Does not support censored `x` variables.
    *   Does not produce plots or print output.
    *   Returns a simple dataframe of results.

## Key Differences from LWP-TRENDS

1.  **Slope Calculation (Seasonal):**
    *   **NADA2 (`censeaken`):** Calculates the overall slope by running a non-seasonal ATS on all data combined.
    *   **LWP-TRENDS:** Calculates the "Seasonal Sen's Slope," which is typically the median of all pairwise slopes computed *within* each season.

2.  **P-Value Calculation (Seasonal):**
    *   **NADA2:** Uses a permutation test (default 4999 reps).
    *   **LWP-TRENDS:** Uses the normal approximation (Z-score) derived from the exact variance of S (with tie corrections).

3.  **Data Aggregation:**
    *   **NADA2:** Does not inherently aggregate data. It uses all available data points.
    *   **LWP-TRENDS:** Strongly enforces aggregation (e.g., one value per season/year) before analysis.

4.  **Log Transformation:**
    *   **NADA2 (`ATS`):** Defaults to `LOG=TRUE`.
    *   **LWP-TRENDS:** Typically works on raw values unless explicitly transformed by the user or configuration.
