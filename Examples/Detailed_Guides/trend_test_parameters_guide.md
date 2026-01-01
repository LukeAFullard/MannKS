# A Comprehensive Guide to `trend_test` Parameters

The `trend_test` function is the workhorse for non-seasonal trend analysis in the `MannKS` package. It has a rich set of parameters that allow you to fine-tune the analysis to your specific needs. This guide provides a detailed explanation of each parameter, including its purpose, usefulness, and limitations.

---

### Core Parameters

These are the essential parameters you will always need to provide.

#### `x`
-   **Type:** `numpy.ndarray` or `pandas.DataFrame`
-   **Description:** This is your primary data vector containing the observations.
-   **Usefulness and Nuances:** If your data is purely numeric, a simple NumPy array is sufficient. If you have censored data (e.g., `"<5"`), you **must** pre-process it first using the `prepare_censored_data` function and provide the resulting DataFrame here.

#### `t`
-   **Type:** `numpy.ndarray`
-   **Description:** This is the time vector corresponding to your `x` values. The values in `t` must correspond one-to-one with the values in `x`.
-   **Usefulness and Nuances:** The key feature of `MannKS` is its ability to handle unequally spaced timestamps. `t` can contain numeric values (e.g., `2001.5`, `2002.75`) or, more commonly, datetime objects. The function automatically handles the conversion of datetimes for calculations.

#### `alpha`
-   **Type:** `float`, **Default:** `0.05`
-   **Description:** The statistical significance level. It's the threshold for deciding if a trend is statistically significant.
-   **Usefulness and Limitations:** An `alpha` of `0.05` corresponds to a 95% confidence level, a common standard in many scientific fields. Setting a lower `alpha` (e.g., `0.01`) makes the test stricter, reducing the risk of a "false positive" (detecting a trend that isn't real). However, this increases the risk of a "false negative" (missing a real but weak trend). The choice of `alpha` depends on the standards of your field and how you want to balance these risks.

#### `plot_path`
-   **Type:** `str` (optional), **Default:** `None`
-   **Description:** If you provide a file path (e.g., `"my_trend_plot.png"`), the function will save a plot visualizing the trend analysis.
-   **Usefulness:** This is extremely useful for diagnostics. A plot can help you visually assess the trend, identify potential outliers, and understand the relationship between the data, the Sen's slope, and the confidence intervals.

#### `seasonal_coloring`
-   **Type:** `bool`, **Default:** `False`
-   **Description:** When plotting data, this parameter allows for distinct coloring of data points based on their season.
-   **Usefulness:** If your dataset contains a `'season'` column (e.g., from `seasonal_trend_test`), setting this to `True` will color-code the points in the output plot by season. This helps in visually inspecting seasonal patterns alongside the overall trend. Note that `trend_test` does not automatically calculate seasons, so this is typically relevant if you are plotting data that has been pre-processed or if you are using `seasonal_trend_test` (which shares this plotting logic).

---

### Censored Data Parameters

These parameters control how the test handles censored data.

#### `hicensor`
-   **Type:** `bool` or `float`, **Default:** `False`
-   **Description:** Activates the "high censor" rule, a conservative method to handle improving detection limits over time.
-   **Usefulness:** Imagine your detection limit improved from `<10` in the 1990s to `<1` in the 2010s. An actual concentration of `3` would be recorded as `<10` in 1995 but `3` in 2015. This can create a misleading "increasing" trend. `hicensor=True` mitigates this by treating all values (censored or not) below the highest detection limit (`10` in this case) as being censored at that high limit. This provides a more conservative and potentially more accurate assessment of the true underlying trend.
-   **Limitations:** By being conservative, this rule can sometimes mask a real but weak trend. It is most appropriate when you have a clear history of changing detection limits. See **[Example 8](../08_High_Censor_Rule/README.md)**.

#### `lt_mult` and `gt_mult`
-   **Type:** `float`, **Defaults:** `lt_mult=0.5`, `gt_mult=1.1`
-   **Description:** Multipliers used to substitute censored values **only for the Sen's slope calculation**. They do **not** affect the p-value or trend significance.
-   **Usefulness and Nuances:** To calculate a slope, you need two numbers. A value like `"<10"` is temporarily replaced by `10 * 0.5 = 5`. The default `lt_mult=0.5` is chosen because it's the midpoint of the possible range `[0, 10]`, representing a neutral estimate. `gt_mult=1.1` is used to place the substituted value slightly outside the uncensored range. These parameters allow you to perform a sensitivity analysis by seeing how the slope magnitude changes with different multipliers.
-   **Limitations:** This is a simple substitution, not a statistical model like Regression on Order Statistics (ROS). The resulting slope is an estimate whose accuracy depends on how well the multiplier reflects the true underlying values. See **[Example 13](../13_Censored_Data_Multipliers/README.md)**.

#### `sens_slope_method`
-   **Type:** `str`, **Default:** `'nan'`
-   **Description:** Controls the method used for calculating the Sen's slope, particularly in the presence of censored data.
-   **Usefulness:**
    *   `'nan'` (Default): A slope between two incompatible censored values (e.g., `<5` and `<10`) is treated as ambiguous (`NaN`) and excluded from the median calculation. This is statistically neutral.
    *   `'ats'`: Uses the **Akritas-Theil-Sen (ATS)** estimator. This is a statistically robust method specifically designed for censored data. It calculates the slope by finding the value that zeroes the generalized Kendall's S statistic for the residuals. It handles mixed censoring types (left and right) and is the recommended method for rigorous analysis of censored datasets.
    *   `'lwp'`: Assigns a slope of `0` to ambiguous censored pairs. This mimics the behavior of the legacy LWP-TRENDS R script.
-   **Limitations:** `'ats'` is computationally more intensive than the other methods and uses bootstrapping for confidence intervals. `'lwp'` can artificially bias the slope towards zero and is only recommended for backward compatibility. See **[Example 21](../21_Sens_Slope_Methods/README.md)**.

#### `mk_test_method`
-   **Type:** `str`, **Default:** `'robust'`
-   **Description:** Controls how the Mann-Kendall test ranks right-censored data (`>`).
-   **Usefulness:** The default `'robust'` method uses a statistically sound approach where a value like `>100` is treated as having a rank higher than any value below 100, but its exact rank relative to other right-censored values is handled by specific tie-correction rules. The `'lwp'` method uses a simpler heuristic that replaces all `>` values with a single number slightly larger than the max uncensored value. This is less robust but is provided for compatibility with the LWP-TRENDS script.
-   **Limitations:** The `'lwp'` heuristic can be sensitive to outliers and is generally not recommended unless direct comparison to that specific script is the primary goal. See **[Example 10](../10_Right_Censored_Methods/README.md)**.

---

### Aggregation Parameters

#### `agg_method`
-   **Type:** `str`, **Default:** `'none'`
-   **Description:** Determines how to handle multiple observations that occur at the same time or within the same time period.
-   **Usefulness:** This is critical for two scenarios:
    1.  **Tied Timestamps:** If you have multiple measurements for the exact same timestamp, you must aggregate them to avoid calculation errors.
        *   `'median'`: Uses the median of values at the tied timestamp.
        *   `'robust_median'`: Uses a censored-data-aware median (recommended for censored data).
        *   `'middle'`: Selects the observation whose timestamp is closest to the mean of the actual timestamps in the group.
        *   `'middle_lwp'`: Selects the observation closest to the theoretical midpoint of the time period.
    2.  **Clustered Data (Thinning):** If you have high-frequency data (e.g., daily samples for one month, then monthly for years), it can bias the trend result. You can "thin" the data to a consistent frequency (e.g., one value per year) using these methods:
        *   `'lwp'`: Selects the single observation closest to the theoretical midpoint of the period (mimics LWP R script `UseMidObs=TRUE`).
        *   `'lwp_median'`: Calculates the median of all observations in the period (mimics LWP R script `UseMidObs=FALSE`).
        *   `'lwp_robust_median'`: As above, but uses a robust median logic suitable for censored data.
-   **Limitations:** Aggregation, by definition, reduces the amount of data used in the test, which can lower its statistical power. For censored data, `'robust_median'` or `'lwp_robust_median'` are the recommended methods, but be aware that aggregating censored values is a difficult statistical problem and the result should be interpreted with care. See **[Example 9](../09_Aggregation_Tied_Clustered_Data/README.md)**.

#### `agg_period`
-   **Type:** `str`, **Default:** `None`
-   **Description:** The time window for aggregation when using any `lwp*` aggregation method (e.g., `'year'`, `'month'`).
-   **Usefulness:** This allows you to define the granularity of your analysis when dealing with clustered data. For example, you can analyze trends on an aggregated `'year'` basis.
-   **Defaults:** Although the default is `None`, if `agg_method` is set to any of the LWP methods (`'lwp'`, `'lwp_median'`, `'lwp_robust_median'`) and `agg_period` is not specified, it will effectively default to `'year'`.

---

### Advanced Statistical Parameters

Defaults are recommended for most users.

#### `tau_method`
-   **Type:** `str`, **Default:** `'b'`
-   **Description:** Controls which version of Kendall's Tau is calculated.
-   **Usefulness:** Real-world data almost always contains "ties" (identical values). Tau-b (the default) is the standard modification that correctly handles ties. Tau-a does not and is generally only appropriate for theoretical, continuous data with no ties.

#### `ci_method`
-   **Type:** `str`, **Default:** `'direct'`
-   **Description:** Controls the method for calculating the Sen's slope confidence intervals.
-   **Usefulness:** The confidence interval gives you a range of plausible values for the true slope. The `'direct'` method is a standard, statistically robust approach. The `'lwp'` method uses a specific interpolation technique designed to replicate the LWP-TRENDS R script. Unless you need that specific replication, `'direct'` is recommended. See **[Example 14](../14_Confidence_Interval_Methods/README.md)**.

#### `tie_break_method`
-   **Type:** `str`, **Default:** `'robust'`
-   **Description:** Controls the internal epsilon used for detecting tied values during the Mann-Kendall test calculation.
-   **Usefulness:** This parameter is primarily for cross-software validation.
    *   `'robust'` (Default): Uses a very small epsilon derived from the data's precision (half the minimum difference between unique values). This safely detects ties without accidentally matching very close but distinct floating-point numbers.
    *   `'lwp'`: Uses a larger epsilon (minimum difference / 1000) to replicate the specific behavior of the LWP-TRENDS R script.
-   **Limitations:** Generally, you should stick to `'robust'`. The `'lwp'` method is only needed if you are trying to reproduce exact numbers from the legacy R script and are facing floating-point discrepancies.

#### `continuous_confidence`
-   **Type:** `bool`, **Default:** `True`
-   **Description:** Controls how trend direction is reported.
-   **Usefulness:**
    -   `True` (default): The trend classification (e.g., "Probably Increasing") is based on a continuous probability score (`C`), even if the p-value is greater than `alpha`. This provides a more nuanced interpretation (e.g., "weak evidence of an increase") rather than a binary "No Trend".
    -   `False`: The function behaves like a classical hypothesis test. If `p > alpha`, the result is simply classified as "No Trend", regardless of the Z-score direction. This is a stricter, more traditional approach.

---

### Other Parameters

#### `min_size`
-   **Type:** `int` (optional), **Default:** `10`
-   **Description:** The minimum sample size required to avoid an "analysis note" warning.
-   **Usefulness:** This serves as a guideline to prevent over-interpretation of results from very small datasets, which have low statistical power and may not be reliable.
-   **Limitations:** This is a heuristic, not a hard statistical rule. You can set it to a different value or `None` to disable it if the standards for your field of work differ.

#### `category_map`
-   **Type:** `dict` (optional), **Default:** `None`
-   **Description:** Allows you to provide your own rules for trend classification.
-   **Usefulness:** If your organization or project has a specific set of reporting standards (e.g., "Improving", "Degrading", "No Change"), you can use this to map the statistical results directly to your required terminology. See our **[Trend Classification Guide](./trend_classification_guide.md)** for a full explanation.

---

### Slope Scaling Parameters

These parameters make the Sen's slope output more intuitive and readable.

#### `x_unit`
-   **Type:** `str`, **Default:** `"units"`
-   **Description:** A string to describe the units of your data `x`.
-   **Usefulness:** This string is used to construct the final `slope_units` field in the output. For example, if you provide `x_unit="mg/L"`, the output units will be formatted like `"mg/L per year"`, making the result and any plots much clearer.

#### `slope_scaling`
-   **Type:** `str` (optional), **Default:** `None`
-   **Description:** Specifies the desired time unit for the Sen's slope.
-   **Usefulness:** This is a powerful convenience feature. When your time vector `t` is made of datetimes, the raw slope is calculated in units per second. By setting `slope_scaling="year"`, the function will automatically multiply the raw slope by the number of seconds in a year and return a value in "units per year".
-   **Valid Units:** `year`, `month`, `week`, `day`, `hour`, `minute`, `second`.
-   **Limitations:** This parameter **only works** when `t` is a datetime-like array. It has no effect and will raise a `UserWarning` if `t` is numeric, because the function cannot know the underlying unit of a numeric time vector. See **[Example 3](../03_Non_Seasonal_Timestamps/README.md)**.
