# LWP-TRENDS R Script: Detailed Function and Parameter Analysis

This document provides a detailed breakdown of the key functions and their parameters within the `LWPTrends_v2502.r` script. It is intended to serve as a reference for the verification process and for understanding the methodology of the original LWP-TRENDS tool.

## High-Level Wrapper Functions

These are the main functions a user would typically call to perform a trend analysis. They orchestrate the pre-processing, analysis, and plotting steps.

### 1. `NonSeasonalTrendAnalysis`

*   **Purpose:** Acts as a high-level wrapper to perform a standard (non-seasonal) Mann-Kendall test and Sen's slope analysis. It calls the lower-level `MannKendall` and `SenSlope` functions internally.
*   **Signature:** `NonSeasonalTrendAnalysis(x, do.plot=F, ...)`
*   **Parameters:**
    *   `x`: An R `data.frame`. This is the most critical input. The function expects this dataframe to have been pre-processed and contain specific columns, including `RawValue`, `Censored`, `CenType`, `myDate`, `Year`, and `TimeIncr`. Without this pre-processing, the function will fail.
    *   `do.plot` (boolean, default: `FALSE`): If `TRUE`, the function will generate and return a ggplot object visualizing the trend, including the Sen's slope and confidence intervals.
    *   `...`: Ellipsis used to pass additional arguments down to the internal `MannKendall` and `SenSlope` functions. This is the primary mechanism for controlling the analysis. See below for a detailed list of these pass-through parameters.

### 2. `SeasonalTrendAnalysis`

*   **Purpose:** Acts as a high-level wrapper for seasonal trend analysis. It determines seasonality, then calls `SeasonalKendall` and `SeasonalSenSlope` internally.
*   **Signature:** `SeasonalTrendAnalysis(x, do.plot=F, ...)`
*   **Parameters:**
    *   `x`: A pre-processed R `data.frame`. Similar to the non-seasonal version, this dataframe must be correctly prepared with columns like `RawValue`, `Censored`, `myDate`, `Year`, and a `Season` column.
    *   `do.plot` (boolean, default: `FALSE`): If `TRUE`, generates and returns a trend plot.
    *   `...`: Passes arguments down to the lower-level seasonal functions.

---

## Key Pass-Through Parameters (for `...`)

These parameters are not explicitly in the wrapper signatures but are passed down to the core functions to control the analysis.

*   `ValuesToUse` (string, default: `"RawValue"`): Specifies the name of the column in the dataframe `x` that contains the numeric data for the trend test. This is essential if you are using flow-adjusted or imputed data stored in a different column.

*   `HiCensor` (boolean or numeric, default: `FALSE`): Implements the "high censor" rule, a key feature for LWP emulation.
    *   If `TRUE`, it finds the highest left-censor limit in the dataset (e.g., the `5` in `<5`). All data points below this limit (whether originally censored or not) are then treated as being left-censored at this highest limit.
    *   If a numeric value is provided (e.g., `HiCensor = 10`), that value is used as the censoring threshold instead of the auto-detected maximum.

*   `TimeIncrMed` (boolean, default: `TRUE`): This is one of the most critical parameters for controlling the LWP-TRENDS methodology. It governs whether temporal aggregation is performed.
    *   **When `TRUE` (Default):** The script aggregates the data so there is only **one observation per time increment** (e.g., one value per year for an annual analysis). This is the standard LWP-TRENDS workflow and is essential for replicating its results. The specific method of aggregation is determined by `UseMidObs`.
    *   **When `FALSE`:** No aggregation is performed. The analysis runs on all raw data points. **Note:** The R script has known bugs in this workflow path, and it may crash.

*   `UseMidObs` (boolean, default: `TRUE`): Determines the aggregation method *only when* `TimeIncrMed` is `TRUE`.
    *   **When `TRUE` (Default):** For each time period, the script selects the single observation that is chronologically closest to the theoretical midpoint of that period.
    *   **When `FALSE`:** It calculates the median of all observation values and the median of all timestamps within that period. If the data contains censored values, this can produce statistically biased results.

*   `RawValues` (boolean, default: `TRUE`): This parameter is primarily for internal consistency, especially when dealing with covariate-adjusted data. It ensures that the special numeric substitutions for censored values (e.g., `<5` becomes `2.5`) are applied correctly during the Sen's slope calculation. For standard verification against raw data, this should be left as `TRUE`.

*   `Year` (string, default: `"Year"`): Specifies the name of the column that contains the year for the analysis.

---

## Pre-Processing and Data Preparation Functions

A user of the R script **must** run these functions in the correct order to prepare the data before calling the analysis wrappers. The workflow is rigid and requires specific column names.

### 1. `RemoveAlphaDetect`

*   **Purpose:** The first and most critical pre-processing step. It parses a column containing censored string data and creates the three standardized columns (`RawValue`, `Censored`, `CenType`) that all other LWP-TRENDS functions depend on.
*   **Signature:** `RemoveAlphaDetect(Data, ColToUse="Value")`
*   **Parameters:**
    *   `Data`: The input `data.frame`.
    *   `ColToUse` (string, default: `"Value"`): The name of the column containing the mixed string/numeric data (e.g., `"5"`, `"<2"`, `">10"`).
*   **Output:** Returns the dataframe with three new, essential columns: `RawValue` (numeric), `Censored` (boolean), and `CenType` (factor: `"lt"`, `"gt"`, `"not"`).

### 2. `GetMoreDateInfo`

*   **Purpose:** Adds numerous date-related columns required for grouping and seasonal analysis. It is mandatory for any seasonal or sub-annual analysis.
*   **Signature:** `GetMoreDateInfo(Data, firstMonth=1, FindDateShifts=TRUE)`
*   **Parameters:**
    *   `Data`: The input `data.frame`. **Crucially, this dataframe must already contain a date column named `myDate`.**
    *   `firstMonth` (integer, default: `1`): Defines the starting month of a "custom year". For standard analysis, this is left at 1 (January).
    *   `FindDateShifts` (boolean, default: `TRUE`): An optional, complex feature that attempts to shift dates from the end of one month to the start of the next to handle irregular sampling schedules. For reproducible verification, it is highly recommended to set this to `FALSE`.
*   **Output:** Returns the dataframe with new columns like `Year`, `Month`, `Qtr` (Quarter), `BiMonth`, etc.

### 3. `InspectTrendData`

*   **Purpose:** Filters the dataset to a specific time window and, most importantly, determines the appropriate time frequency for the analysis based on data availability.
*   **Signature:** `InspectTrendData(x, Year = "Year", TrendPeriod = NA, EndYear = NA, propIncrTol=0.9, propYearTol=0.9, ...)`
*   **Parameters:**
    *   `x`: The input `data.frame`.
    *   `Year` (string, default: `"Year"`): The name of the column containing the year information.
    *   `TrendPeriod` (integer, default: `NA`): The length of the desired analysis period in years.
    *   `EndYear` (integer, default: `NA`): The final year of the analysis period.
    *   `propIncrTol` (float, default: `0.9`): The minimum proportion of time increments (e.g., months) that must contain data.
    *   `propYearTol` (float, default: `0.9`): The minimum proportion of years in the period that must contain data.
*   **Output:** Returns the filtered dataframe with a new column named `TimeIncr`. This column is populated with the highest frequency that met the data sufficiency criteria (e.g., the values from the `Month` column) and is used by the main analysis functions to define the time increments.
