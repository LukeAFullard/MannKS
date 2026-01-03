# MannKS: A User Guide Through Examples

Welcome to the user guide for the `MannKS` Python package. This collection of examples is designed to be a practical, hands-on guide to all the features of the library, from basic trend tests to advanced seasonal analysis.

Each example is a self-contained "chapter" that includes the full Python code, the exact output, and a detailed interpretation of the results.

## Table of Contents

### Part 1: Core Concepts & Basic Workflow

*   [**Example 01: Getting Started - Inspecting Your Data**](./01_Getting_Started_Inspecting_Data/README.md)
    *   **Goal:** Showcase the first step in any analysis: visualizing and understanding the data's structure using `inspect_trend_data`.
*   [**Example 02: Basic Non-Seasonal Trend Test (Numeric Time)**](./02_Basic_Non_Seasonal_Numeric/README.md)
    *   **Goal:** Demonstrate the simplest use case with an integer time vector (e.g., Years).
*   [**Example 03: Non-Seasonal Trend Test with Timestamps & Slope Scaling**](./03_Non_Seasonal_Datetime_And_Scaling/README.md)
    *   **Goal:** Introduce the handling of real-world time series data using `pandas` `datetime` objects and demonstrate **Slope Scaling** to get results in user-friendly units.
*   [**Example 04: Handling Basic Censored Data**](./04_Basic_Censored_Data/README.md)
    *   **Goal:** Explain the essential workflow for data containing censored values using `prepare_censored_data`.
*   [**Example 05: Basic Seasonal Trend Test & Seasonality Check**](./05_Basic_Seasonal_Trend_Test/README.md)
    *   **Goal:** Introduce seasonal trend analysis for monthly data and demonstrate the best practice of **Consistent Aggregation**.

### Part 2: Handling Complex & Messy Data

*   [**Example 06: Deep Dive into Censored Data Options**](./06_Censored_Data_Options/README.md)
    *   **Goal:** Compare the `'robust'` vs. `'lwp'` methods for handling censored data to show their impact.
*   [**Example 07: The High Censor Rule (`hicensor`)**](./07_The_High_Censor_Rule/README.md)
    *   **Goal:** Explain and demonstrate the `hicensor` rule for mitigating bias from changing detection limits.
*   [**Example 08: Aggregation for Tied and Clustered Data**](./08_Aggregation_and_Clustered_Data/README.md)
    *   **Goal:** Show how to resolve data with multiple measurements at the same timestamp (tied) and demonstrate temporal aggregation for irregularly sampled data.
*   [**Example 09: Comparing Right-Censored Data Methods (`mk_test_method`)**](./09_Comparing_Right_Censored_Methods/README.md)
    *   **Goal:** Demonstrate and explain the difference between the `'robust'` and `'lwp'` methods for handling right-censored data.
*   [**Example 10: Handling Data with Multiple Censoring Levels**](./10_Multiple_Censoring_Levels/README.md)
    *   **Goal:** Showcase the robustness of the package with complex, real-world censored data containing multiple detection limits.

### Part 3: Advanced Features & Nuances

*   [**Example 11: Advanced Seasonality (Non-Monthly Data)**](./11_Advanced_Seasonality_Non_Monthly/README.md)
    *   **Goal:** Demonstrate seasonal analysis on non-monthly data (e.g., weekly, quarterly).
*   [**Example 12: The Impact of Censored Data Multipliers**](./12_Censored_Data_Multipliers/README.md)
    *   **Goal:** Show the sensitivity of the Sen's slope to the `lt_mult` and `gt_mult` parameters.
*   [**Example 13: Comparing Confidence Interval Methods**](./13_Comparing_Confidence_Interval_Methods/README.md)
    *   **Goal:** Visually and numerically compare the `'direct'` vs. `'lwp'` confidence interval methods.
*   [**Example 14: Time Vector Nuances (Numeric Data)**](./14_Time_Vector_Nuances_Numeric/README.md)
    *   **Goal:** Explain the difference between using simple integer ranks vs. fractional years as the time vector for numeric analysis.
*   [**Example 15: Regional Trend Analysis**](./15_Regional_Trend_Analysis/README.md)
    *   **Goal:** Demonstrate how to aggregate trend results from multiple sites using `regional_test`.

### Part 4: Interpreting Results & Validation

*   [**Example 16: Interpreting the Full Output**](./16_Interpreting_Full_Output/README.md)
    *   **Goal:** Provide a detailed explanation of every field in the `namedtuple` returned by the test functions.
*   [**Example 17: Interpreting Analysis Notes**](./17_Interpreting_Analysis_Notes/README.md)
    *   **Goal:** Explain the meaning of the common "Analysis Notes" returned by the package, such as "Long run of single value".
*   [**Example 18: Interpreting Confidence and Customizing Classification**](./18_Interpreting_Confidence_and_Classification/README.md)
    *   **Goal:** Provide a deeper dive into the trend classification system and confidence scores.
*   [**Example 19: Visual Diagnostics of Trend Plots**](./19_Visual_Diagnostics_of_Trend_Plots/README.md)
    *   **Goal:** Show users how to "read" the output of `plot_trend` to diagnose issues like wide confidence intervals.
*   [**Example 20: Advanced Sen's Slope Methods (ATS vs. LWP)**](./20_Advanced_Sens_Slope_Methods/README.md)
    *   **Goal:** Demonstrate the difference between standard methods and the robust **Akritas-Theil-Sen (ATS)** estimator for censored data.

### Part 5: Advanced Seasonal & High-Frequency Analysis

*   [**Example 21: Seasonal Trend with Weekly Data (Decreasing Trend)**](./21_Seasonal_Trend_Weekly_Decreasing/README.md)
    *   **Goal:** Demonstrate seasonal analysis on weekly data (`season_type='day_of_week'`) with a clear decreasing trend.
*   [**Example 22: Seasonal Trend with Daily Data (No Trend)**](./22_Seasonal_Trend_Daily_No_Trend/README.md)
    *   **Goal:** Show seasonal analysis on daily data where a strong seasonal pattern exists but no significant long-term trend.
*   [**Example 23: High Frequency Data (Hours, Minutes, Seconds)**](./23_High_Frequency_Data/README.md)
    *   **Goal:** Demonstrate analysis on high-frequency data to verify the package handles various time units correctly.
*   [**Example 24: Advanced Seasonality with `day_of_year`**](./24_Advanced_Seasonality_DayOfYear/README.md)
    *   **Goal:** Showcase a more granular seasonal analysis using the day of the year.

### Part 6: Advanced Parameter Nuances

*   [**Example 25: Advanced Parameter Nuances**](./25_Advanced_Parameter_Nuances/README.md)
    *   **Goal:** Demonstrate and explain the use of less common but important function parameters like `tau_method` and `min_size`.
*   [**Example 26: Alpha and Confidence Intervals**](./26_Alpha_and_Confidence_Intervals/README.md)
    *   **Goal:** Demonstrate how the `alpha` parameter influences the confidence intervals in the trend plot.
*   [**Example 27: Continuous Confidence vs Classical Testing**](./27_Continuous_Confidence_Comparison/README.md)
    *   **Goal:** Compare the default "Continuous Confidence" approach against traditional binary hypothesis testing to show how it provides more nuanced results for weak trends.
*   [**Example 28: Residual Diagnostics**](./28_Residual_Diagnostics/README.md)
    *   **Goal:** Learn how to interpret residual diagnostic plots to check assumptions of linearity and homoscedasticity.
*   [**Example 29: Block Bootstrap for Autocorrelation**](./29_Block_Bootstrap_Autocorrelation/README.md)
    *   **Goal:** Learn how to use Block Bootstrap to avoid false positives when analyzing data with serial correlation (autocorrelation).
*   [**Example 30: Rolling Trend Analysis**](./30_Rolling_Trend_Analysis/README.md)
    *   **Goal:** Learn how to perform a rolling window analysis to detect when trends start, stop, or change direction over time.

## Detailed Guides

In addition to the practical examples, the following deep-dive guides are available:

*   [**Check Seasonality Guide**](./Detailed_Guides/check_seasonality_guide/README.md): Detailed usage of the `check_seasonality` function.
*   [**Inspect Trend Data Guide**](./Detailed_Guides/inspect_trend_data_guide/README.md): How to use visual inspection tools.
*   [**Interpreting Test Outputs Guide**](./Detailed_Guides/interpreting_test_outputs_guide/README.md): A comprehensive reference for all output fields.
*   [**Regional Test Guide**](./Detailed_Guides/regional_test_guide/README.md): In-depth guide on performing regional trend analysis.
*   [**Analysis Notes Guide**](./Detailed_Guides/analysis_notes_guide.md): Reference for all possible analysis notes and warnings.
*   [**Regional Test Parameters Guide**](./Detailed_Guides/regional_test_parameters_guide.md): Parameter reference for `regional_test`.
*   [**Seasonal Trend Test Parameters Guide**](./Detailed_Guides/seasonal_trend_test_parameters_guide.md): Parameter reference for `seasonal_trend_test`.
*   [**Trend Classification Guide**](./Detailed_Guides/trend_classification_guide.md): How trends are classified based on confidence levels.
*   [**Trend Test Parameters Guide**](./Detailed_Guides/trend_test_parameters_guide.md): Parameter reference for `trend_test`.
