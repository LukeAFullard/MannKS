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
-   **Usefulness and Limitations:** `alpha` also sets the level of the slope's confidence interval (1 - `alpha`). An `alpha` of `0.05` corresponds to a 95% confidence level, a common standard in many scientific fields. (LWP-TRENDS reports 90% intervals; use `alpha=0.1` to compare with it.) Setting a lower `alpha` (e.g., `0.01`) makes the test stricter, reducing the risk of a "false positive" (detecting a trend that isn't real). However, this increases the risk of a "false negative" (missing a real but weak trend). The choice of `alpha` depends on the standards of your field and how you want to balance these risks.

#### `plot_path`
-   **Type:** `str` (optional), **Default:** `None`
-   **Description:** If you provide a file path (e.g., `"my_trend_plot.png"`), the function will save a plot visualizing the trend analysis.
-   **Usefulness:** This is extremely useful for diagnostics. A plot can help you visually assess the trend, identify potential outliers, and understand the relationship between the data, the Sen's slope, and the confidence intervals.

#### `residual_plot_path`
-   **Type:** `str` (optional), **Default:** `None`
-   **Description:** If provided, saves residual diagnostic plots (residuals against time and their histogram) for the fitted Sen's slope line. See **[Example 28](../28_Residual_Diagnostics/README.md)**.

#### `seasonal_coloring`
-   **Type:** `bool`, **Default:** `False`
-   **Description:** When plotting data, this parameter allows for distinct coloring of data points based on their season.
-   **Usefulness:** If your dataset contains a `'season'` column (e.g., from `seasonal_trend_test`), setting this to `True` will color-code the points in the output plot by season. This helps in visually inspecting seasonal patterns alongside the overall trend. Note that `trend_test` does not automatically calculate seasons, so this is typically relevant if you are plotting data that has been pre-processed or if you are using `seasonal_trend_test` (which shares this plotting logic).

---

### Autocorrelation & Bootstrap Parameters

These parameters enable the block bootstrap method for handling autocorrelated data.

#### `autocorr_method`
-   **Type:** `str`, **Default:** `'none'`
-   **Description:** Method to handle autocorrelation.
-   **Usefulness:** Standard Mann-Kendall tests assume that observations are independent. Autocorrelation (serial correlation) can violate this assumption.
    *   `'none'` (Default): No correction.
    *   `'auto'`: Removes the Sen's slope trend and checks the lag-1 autocorrelation of the residuals. If it is significant, it switches to `'block_bootstrap'`.
    *   `'block_bootstrap'`: Uses a **detrended residual circular block bootstrap** for the p-value (simulating H0: no trend while keeping short-range dependence) and a **pairs block bootstrap** for the slope's confidence interval. See [bootstrap.md](../../bootstrap.md).
    *   `'yue_wang'`: Inflates the variance of S by the Yue & Wang (2004) factor, estimated from the significant autocorrelations of the detrended series. Fast, and it keeps the nominal size for independent data, but like all variance corrections it under-corrects strong autocorrelation in short series.
-   **Limitations:** No method fully corrects strong autocorrelation in short series. In simulations with AR(1) noise (rho = 0.6, n = 60) the false-positive rate at `alpha=0.05` was about 0.30 without correction, 0.18 with `'yue_wang'` and 0.15 with `'block_bootstrap'`.

#### `block_size`
-   **Type:** `int` or `'auto'`, **Default:** `'auto'`
-   **Description:** The size of the blocks used for resampling in the block bootstrap.
-   **Usefulness:** Block bootstrapping preserves the correlation structure within blocks while reshuffling the blocks themselves.
    *   `'auto'` (Default): Twice the first lag at which the autocorrelation of the detrended data drops below 0.1, bounded to [3, sqrt(n)].
    *   `int`: Manually specifying the block size allows for sensitivity analysis.

#### `n_bootstrap`
-   **Type:** `int`, **Default:** `1000`
-   **Description:** The number of bootstrap resamples to generate.
-   **Usefulness:** A higher number of bootstraps provides a more precise estimate of the p-value and confidence intervals but increases computation time. The bootstrap p-value is `(1 + #{|S*| >= |S|}) / (n_bootstrap + 1)`, so it is never exactly zero. Set `random_state` for reproducible results.

---

### Censored Data Parameters

These parameters control how the test handles censored data.

#### `hicensor`
-   **Type:** `bool` or `float`, **Default:** `False`
-   **Description:** Activates the "high censor" rule, a conservative method to handle improving detection limits over time.
-   **Usefulness:** Imagine your detection limit improved from `<10` in the 1990s to `<1` in the 2010s. An actual concentration of `3` would be recorded as `<10` in 1995 but `3` in 2015. This can create a misleading "increasing" trend. `hicensor=True` mitigates this by treating all values (censored or not) below the highest detection limit (`10` in this case) as being censored at that high limit. This provides a more conservative and potentially more accurate assessment of the true underlying trend.
-   **Limitations:** By being conservative, this rule can sometimes mask a real but weak trend. It is most appropriate when you have a clear history of changing detection limits. See **[Example 7](../07_The_High_Censor_Rule/README.md)**.

#### `lt_mult` and `gt_mult`
-   **Type:** `float`, **Defaults:** `lt_mult=0.5`, `gt_mult=1.1`
-   **Description:** Multipliers used to substitute censored values in the **pairwise** slope estimators (`sens_slope_method='lwp'` or `'drop_ambiguous'`). They do **not** affect the p-value, the trend significance, or the default `'ats'` slope (which needs no substitution).
-   **Usefulness and Nuances:** To calculate a pairwise slope, you need two numbers. A value like `"<10"` is temporarily replaced by `10 * 0.5 = 5`. The default `lt_mult=0.5` is chosen because it's the midpoint of the possible range `[0, 10]`, representing a neutral estimate. `gt_mult=1.1` is used to place the substituted value slightly outside the uncensored range. These parameters allow you to perform a sensitivity analysis by seeing how the slope magnitude changes with different multipliers.
-   **Limitations:** This is a simple substitution, not a statistical model like Regression on Order Statistics (ROS). The resulting slope is an estimate whose accuracy depends on how well the multiplier reflects the true underlying values. See **[Example 12](../12_Censored_Data_Multipliers/README.md)**.

#### `sens_slope_method`
-   **Type:** `str`, **Default:** `'ats'`
-   **Description:** Slope estimator used when the data contain censored values (uncensored data always use the exact Sen's slope).
-   **Usefulness:**
    *   `'ats'` (Default): The **Akritas-Theil-Sen** estimator (as in `NADA::cenken`): the slope at which Kendall's S between time and the censored residuals is zero. Censored values are treated as intervals, so no substitution is needed and both left and right censoring are handled. Its confidence interval inverts the censored Kendall test (the same construction as Sen's interval). In simulations it was the least biased option (within a few percent up to 50% censoring).
    *   `'lwp'`: Pairwise slopes with censored values substituted (`lt_mult`, `gt_mult`) and ambiguous pairs set to `0`, exactly as the LWP-TRENDS R script. Use it to reproduce LWP-TRENDS. It is biased towards zero as censoring increases (about -55% at 50% censoring in simulations).
    *   `'drop_ambiguous'` (formerly `'unbiased'`/`'nan'`, which still work but are deprecated): Pairwise slopes with ambiguous pairs dropped. It is biased *away* from zero (about +17% at 20% censoring and +70% at 50%), because the dropped pairs are mostly those opposing the trend.
-   **Limitations:** The ATS estimate is computed exactly from all pairs (sampled pairs in fast mode); for several thousand censored observations this takes a few seconds. See **[Example 20](../20_Advanced_Sens_Slope_Methods/README.md)**.

#### `mk_test_method`
-   **Type:** `str`, **Default:** `'robust'`
-   **Description:** Controls how the Mann-Kendall test ranks right-censored data (`>`).
-   **Usefulness:** The default `'robust'` method treats a value like `>100` as an interval: it is ranked above any value at or below 100, and comparisons that cannot be resolved (e.g. `>100` vs `150`) count as ties (Helsel 2012). The `'lwp'` method follows LWP-TRENDS: every right-censored value is replaced by a single value 0.1 above the largest right-censoring limit and treated as uncensored, so all `>` values tie with each other and rank above everything else.
-   **Limitations:** The `'lwp'` heuristic discards the information in differing right-censoring limits and is only recommended for reproducing LWP-TRENDS. See **[Example 9](../09_Comparing_Right_Censored_Methods/README.md)**.

---

### Aggregation Parameters

#### `agg_method`
-   **Type:** `str`, **Default:** `'none'`
-   **Description:** Determines how to handle multiple observations that occur at the same time or within the same time period.
-   **Usefulness:** This is useful in two scenarios:
    1.  **Tied Timestamps:** Observations with identical timestamps are handled without aggregation (they score 0 in S and have no pairwise slope), but you may prefer to combine them first:
        *   `'median'`: Uses the median of values at the tied timestamp.
        *   `'robust_median'`: Uses a censored-data-aware median (recommended for censored data).
        *   `'middle'`: Selects the observation whose timestamp is closest to the mean of the actual timestamps in the group.
        *   `'middle_lwp'`: With `agg_period`, selects the observation closest to the LWP-TRENDS mid-point of the increment (same rule as `'lwp'`); for identical timestamps it keeps the first.
    2.  **Clustered Data (Thinning):** If you have high-frequency data (e.g., daily samples for one month, then monthly for years), it can bias the trend result. You can "thin" the data to a consistent frequency (e.g., one value per year) using these methods:
        *   `'lwp'`: Selects the single observation closest to the mid-point of each period, using the LWP-TRENDS mid-points (LWP R script `UseMidObs=TRUE`).
        *   `'lwp_median'`: Calculates the median of all observations in the period (mimics LWP R script `UseMidObs=FALSE`).
        *   `'lwp_robust_median'`: As above, but uses a robust median logic suitable for censored data.
-   **Limitations:** Aggregation, by definition, reduces the amount of data used in the test, which can lower its statistical power. For censored data, `'robust_median'` or `'lwp_robust_median'` are the recommended methods, but be aware that aggregating censored values is a difficult statistical problem and the result should be interpreted with care. See **[Example 8](../08_Aggregation_and_Clustered_Data/README.md)**.

#### `agg_period`
-   **Type:** `str`, **Default:** `None`
-   **Description:** The time window for aggregation (`'year'`, `'quarter'`, `'month'`, `'week'`, `'day'`, `'hour'`, `'minute'`, `'second'`). Requires datetime `t`.
-   **Usefulness:** This allows you to define the granularity of your analysis when dealing with clustered data. For example, you can analyze trends on an aggregated `'year'` basis.
-   **Defaults:** Although the default is `None`, if `agg_method` is set to any of the LWP methods (`'lwp'`, `'lwp_median'`, `'lwp_robust_median'`) and `agg_period` is not specified, it will effectively default to `'year'`.

---

### Advanced Statistical Parameters

Defaults are recommended for most users.

#### `tau_method`
-   **Type:** `str`, **Default:** `'b'`
-   **Description:** Controls which version of Kendall's Tau is calculated.
-   **Usefulness:** Tau-b (the default) adjusts the denominator for ties in the data. Tau-a divides S by n(n-1)/2 and is what LWP-TRENDS reports. Only the reported `Tau` changes; S, its variance and the p-value are the same.

#### `ci_method`
-   **Type:** `str`, **Default:** `'direct'`
-   **Description:** Controls the method for calculating the Sen's slope confidence intervals.
-   **Usefulness:** The confidence interval gives you a range of plausible values for the true slope. The `'direct'` method is a standard, statistically robust approach. The `'lwp'` method interpolates between ranks and computes the variance ignoring censoring, as the LWP-TRENDS R script does (combine it with `alpha=0.1` to reproduce LWP-TRENDS' 90% intervals). Unless you need that replication, `'direct'` is recommended. With the default `'ats'` slope for censored data, the interval is obtained by inverting the censored Kendall test instead. See **[Example 13](../13_Comparing_Confidence_Interval_Methods/README.md)**.

#### `tie_break_method`
-   **Type:** `str`, **Default:** `'robust'`
-   **Description:** Size of the tiny internal shift used to separate censored from uncensored values in the variance of S (the `delx` constant of NADA/LWP-TRENDS).
-   **Usefulness:** Both options give identical results in practice (verified against LWP-TRENDS); `'lwp'` uses exactly the R constant (smallest gap / 1000), `'robust'` scales the shift with the sample size so that it can never cross a neighbouring value. Leave it at the default unless you are reproducing LWP-TRENDS.

#### `continuous_confidence`
-   **Type:** `bool`, **Default:** `True`
-   **Description:** Controls how trend direction is reported.
-   **Usefulness:**
    -   `True` (default): The trend classification (e.g., "Likely Increasing") is based on the continuous confidence `C` in the trend direction, even if the p-value is greater than `alpha`. This provides a more nuanced interpretation (e.g., "weak evidence of an increase") rather than a binary "No Trend".
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

### Large Dataset Parameters

#### `large_dataset_mode`
-   **Type:** `str`, **Default:** `'auto'`
-   **Description:** `'full'` computes every pair (exact); `'fast'` estimates the slope from a random sample of `max_pairs` pairs; `'auto'` uses `'full'` up to n = 5,000 and `'fast'` above (n counted after any aggregation). `'full'` is refused above n = 46,340 (memory) and fast mode is used instead, with a warning. The Mann-Kendall test itself (S, variance, p-value) is always exact. See the **[Large Dataset Guide](./large_dataset_guide.md)**.

#### `max_pairs`
-   **Type:** `int` (optional), **Default:** `None` (100,000 in fast mode)
-   **Description:** Number of pairs sampled for the slope in fast mode. With 100,000 pairs the slope's relative standard deviation around the exact value is about 0.3%.

#### `random_state`
-   **Type:** `int` (optional), **Default:** `None`
-   **Description:** Seed for pair sampling and the bootstrap. Pair sampling uses a fixed seed when `None`, so results are reproducible by default.

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
-   **Limitations:** This parameter **only works** when `t` is a datetime-like array. It has no effect and will raise a `UserWarning` if `t` is numeric, because the function cannot know the underlying unit of a numeric time vector. See **[Example 3](../03_Non_Seasonal_Datetime_And_Scaling/README.md)**.
