# Plan for `MannKS` Package Examples (Comprehensive)

This document outlines a plan for creating a comprehensive suite of examples to guide users through the functionality of the `MannKS` package. The examples are designed to build upon each other, progressing from basic usage to advanced, nuanced scenarios.

Each example will be self-contained in its own directory within `Examples/` and will include a Python script and a markdown (`.md`) file explaining the process, code, and results with embedded plots.

**IMPORTANT:** **Example 2 (`Examples/02_Basic_Non_Seasonal_Numeric/`) serves as the definitive template for all subsequent examples.**
*   All examples must be "lecture-style": deep, educational, and explanatory, not just code dumps.
*   They must be self-contained, generating their own `README.md` and plots.
*   The generated `README.md` must include the Python code used, the textual output, and embedded plots.

---

### **Part 1: Core Concepts & Basic Workflow**

#### Example 1: Getting Started - Inspecting Your Data
- **Goal:** Showcase the first step in any analysis: visualizing and understanding the data's structure.
- **Functions:** `MannKS.inspect_trend_data(plot=True)`.

#### Example 2: Basic Non-Seasonal Trend Test (Numeric Time)
- **Goal:** Demonstrate the simplest use case with an integer time vector (e.g., Years).
- **Functions:** `MannKS.trend_test()`.

#### Example 3: Non-Seasonal Trend Test with Timestamps & Slope Scaling
- **Goal:** Introduce the handling of real-world time series data using `pandas` `datetime` objects and demonstrate **Slope Scaling** to get results in user-friendly units (e.g., "mg/L per year").
- **Functions:** `MannKS.trend_test(slope_scaling='year')`, `plot_path`.

#### Example 4: Handling Basic Censored Data
- **Goal:** Explain the essential workflow for data containing censored values.
- **Functions:** `MannKS.prepare_censored_data()`, `MannKS.trend_test()`.

#### Example 5: Basic Seasonal Trend Test & Seasonality Check
- **Goal:** Introduce seasonal trend analysis for monthly data. Explicitly demonstrate the best practice of **Consistent Aggregation**: using the same aggregation method for both the `check_seasonality` step and the final `seasonal_trend_test`.
- **Functions:** `MannKS.check_seasonality()`, `MannKS.seasonal_trend_test()`.

---

### **Part 2: Handling Complex & Messy Data**

#### Example 6: Deep Dive into Censored Data Options
- **Goal:** Compare the `'robust'` vs. `'lwp'` methods for handling censored data to show their impact.
- **Comparison:** `mk_test_method='robust'` vs. `'lwp'`, `sens_slope_method='unbiased'` vs. `'lwp'`.

#### Example 7: The High Censor Rule (`hicensor`)
- **Goal:** Explain and demonstrate the `hicensor` rule for mitigating bias from changing detection limits.
- **Dataset:** A time series where the censoring level (`<X`) improves (decreases) over time.

#### Example 8: Aggregation for Tied and Clustered Data
- **Goal:** Show how to resolve data with multiple measurements at the same timestamp (tied) and demonstrate temporal aggregation for irregularly sampled data (clustered).
- **Comparison:** `agg_method` with `'median'`, `'robust_median'`, `'middle'`, `'none'`, and `'lwp'`.

#### Example 9: Comparing Right-Censored Data Methods (`mk_test_method`)
- **Goal:** Demonstrate and explain the difference between the `'robust'` and `'lwp'` methods for handling right-censored data.
- **Comparison:** `mk_test_method='robust'` vs. `mk_test_method='lwp'`.

#### Example 10: Handling Data with Multiple Censoring Levels
- **Goal:** Showcase the robustness of the package with complex, real-world censored data.
- **Dataset:** Data with numerous different censoring levels (e.g., `<1`, `<2`, `<5`, `>50`, `>100`).

---

### **Part 3: Advanced Features & Nuances**

#### Example 11: Advanced Seasonality (Non-Monthly Data)
- **Goal:** Demonstrate seasonal analysis on non-monthly data (e.g., weekly, quarterly).
- **Functions:** `seasonal_trend_test(season_type='day_of_week')`, `plot_seasonal_distribution()`.

#### Example 12: The Impact of Censored Data Multipliers
- **Goal:** Show the sensitivity of the Sen's slope to the `lt_mult` and `gt_mult` parameters.
- **Comparison:** Run the same analysis with different multiplier values.

#### Example 13: Comparing Confidence Interval Methods
- **Goal:** Visually and numerically compare the `'direct'` vs. `'lwp'` confidence interval methods.
- **Comparison:** `ci_method='direct'` vs. `ci_method='lwp'`.

#### Example 14: Time Vector Nuances (Numeric Data)
- **Goal:** Explain the difference between using simple integer ranks vs. fractional years as the time vector for numeric analysis.
- **Comparison:** `t=np.arange(n)` vs. `t=df['Year'] + df['Month']/12`.

#### Example 15: Regional Trend Analysis
- **Goal:** Demonstrate how to aggregate trend results from multiple sites.
- **Functions:** `MannKS.regional_test()`.

---

### **Part 4: Interpreting Results & Validation**

#### Example 16: Interpreting the Full Output
- **Goal:** Provide a detailed explanation of every field in the `namedtuple` returned by the test functions.
- **Focus:** Explain less obvious fields like `Tau`, `s`, `var_s`, `sen_probability`, and `n_censor_levels`.

#### Example 17: Interpreting Analysis Notes
- **Goal:** Explain the meaning of the common "Analysis Notes" returned by the package.
- **Scenarios:** Generate results that produce notes like `"Long run of single value"`, `"Sen slope influenced by censored values"`, and `"Insufficient data"`.

#### Example 18: Interpreting Confidence and Customizing Classification
- **Goal:** Provide a deeper dive into the trend classification system.
- **Functions:** `alpha`, `category_map`, and explaining the `C` and `Cd` scores.

#### Example 19: Visual Diagnostics of Trend Plots
- **Goal:** Show users how to "read" the output of `plot_trend` to diagnose issues.
- **Scenarios:** Generate plots showing wide confidence intervals (high uncertainty), clear trends, and no trend.

#### Example 20: Advanced Sen's Slope Methods (ATS vs. LWP)
- **Goal:** Demonstrate the difference between the standard methods and the robust **Akritas-Theil-Sen (ATS)** estimator for censored data.
- **Comparison:** `sens_slope_method='unbiased'` vs. `sens_slope_method='lwp'` vs. `sens_slope_method='ats'`.

---

### **Part 5: Advanced Seasonal & High-Frequency Analysis**

#### Example 21: Seasonal Trend with Weekly Data (Decreasing Trend)
- **Goal:** Demonstrate seasonal analysis on weekly data (`season_type='day_of_week'`).
- **Dataset:** A multi-year dataset with a clear decreasing trend and a weekly pattern (e.g., lower values on weekends).
- **Functions:** `seasonal_trend_test()`.

#### Example 22: Seasonal Trend with Daily Data (No Trend)
- **Goal:** Show seasonal analysis on daily data over multiple years, where a strong seasonal pattern exists but no significant long-term trend.
- **Dataset:** Daily data with a "summer vs. winter" pattern but no overall upward or downward slope.
- **Functions:** `seasonal_trend_test(season_type='month')`.

#### Example 23: High Frequency Data (Hours, Minutes, Seconds)
- **Goal:** Demonstrate analysis on high-frequency data to verify the package handles various time units correctly.
- **Dataset:**
    -   **Hourly:** Data showing a diurnal (daily) pattern.
    -   **Minutes/Seconds:** A short-duration, high-resolution dataset (e.g., sensor data) to demonstrate `slope_scaling` with small units.
- **Functions:** `seasonal_trend_test(season_type='hour')`, `trend_test(slope_scaling='minute')`.

#### Example 24: Advanced Seasonality with `day_of_year`
- **Goal:** Showcase a more granular seasonal analysis using the day of the year, which is useful for environmental data.
- **Dataset:** Multi-year daily data with a distinct pattern related to a specific time of year (e.g., spring runoff).
- **Functions:** `seasonal_trend_test(season_type='day_of_year')`.

---

### **Part 6: Advanced Parameter Nuances**

#### Example 25: Advanced Parameter Nuances
- **Goal:** Demonstrate and explain the use of less common but important function parameters.
- **Topics Covered:**
    - `tau_method` ('a' vs. 'b')
    - Standalone `classify_trend` with a custom map
    - `min_size`
    - `seasonal_trend_test` with a numeric time vector

#### Example 26: Alpha and Confidence Intervals
- **Goal:** Demonstrate how the `alpha` parameter influences the confidence intervals (CI) in the trend plot.
- **Functions:** `trend_test(alpha=...)`.
- **Approach:** Run the same analysis with `alpha=0.05`, `0.20`, and `0.40`, plotting the results to visualize the changing width of the confidence interval.
