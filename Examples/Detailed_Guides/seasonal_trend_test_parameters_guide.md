# A Comprehensive Guide to `seasonal_trend_test` Parameters

The `seasonal_trend_test` function is designed to perform a Mann-Kendall trend test on data with seasonal cycles. It computes the Mann-Kendall statistic within each season (e.g. all Januaries) and sums S and its variance over seasons (Hirsch, Slack & Smith 1982), so that seasonal differences in level do not mask or create a trend. The slope is the median of the within-season pairwise slopes, or the seasonal Akritas-Theil-Sen estimate for censored data. This guide provides a detailed explanation of each parameter, with a focus on those specific to seasonal analysis.

---

### Core Parameters

These are the essential parameters you will always need to provide.

#### `x`
-   **Type:** `numpy.ndarray` or `pandas.DataFrame`
-   **Description:** This is your primary data vector containing the observations.
-   **Usefulness:** Just like in `trend_test`, this can be numeric or a pre-processed censored DataFrame.

#### `t`
-   **Type:** `numpy.ndarray`
-   **Description:** This is the time vector corresponding to your `x` values.
-   **Usefulness:** For seasonal tests, `t` is critical for defining both the trend period and the season. Datetime objects are strongly recommended for automatic seasonal extraction.

#### `alpha`
-   **Type:** `float`, **Default:** `0.05`
-   **Description:** The statistical significance level.
-   **Usefulness:** Sets the threshold for determining if the overall seasonal trend is significant, and the level (1 - `alpha`) of the slope's confidence interval. Use `alpha=0.1` to compare with LWP-TRENDS' 90% intervals.

#### `plot_path`
-   **Type:** `str` (optional), **Default:** `None`
-   **Description:** If provided, saves a visualization of the trend analysis.

#### `seasonal_coloring`
-   **Type:** `bool`, **Default:** `False`
-   **Description:** When plotting data, this parameter allows for distinct coloring of data points based on their season.
-   **Usefulness:** If set to `True`, the points in the output plot will be color-coded by season (e.g., all January points in blue, July in red). This is extremely helpful for visually inspecting seasonal patterns and spotting if a trend is being driven by a specific season.

---

### Seasonality Parameters

These parameters are crucial for defining the seasonal structure of your data. The goal is to correctly group your data into distinct seasons (like "January", "February", etc.) so the test can analyze them.

#### `period`
-   **Type:** `int`, **Default:** `None`
-   **Description:** This defines the length of a full seasonal cycle.
-   **Usefulness and Nuances:**
    -   **Datetime Inputs:** If `t` contains datetime objects, this parameter is **automatically inferred** from the `season_type`. For example, `season_type='day_of_week'` automatically sets `period=7`. You only need to provide `period` if you want to override this default or use a custom cycle.
    -   **Numeric Inputs:** If your time vector `t` is **numeric**, this parameter is **required**, and `t` must be expressed in **season units**: the season of an observation is `floor(t - min(t)) % period`. For quarterly data use a quarter index (`0, 1, 2, 3, 4, ...`) with `period=4`; for monthly data a month index with `period=12`. Decimal years (e.g. `2001.0, 2001.25, ...`) with `period=4` would put each *year* into a different "season"; MannKS warns when `t` does not look like season units. Datetime `t` with `season_type` avoids this pitfall.

#### `season_type`
-   **Type:** `str`, **Default:** `'month'`
-   **Description:** When `t` contains datetime objects, this parameter specifies how to extract the seasonal component from each timestamp. It also automatically sets the `period`.
-   **Usefulness and Nuances:** This is the most flexible and common way to define seasons.
    -   `'month'` (default): For standard monthly seasonality (Infers `period=12`).
    -   `'quarter'`: For quarterly data (4 seasons).
    -   `'day_of_week'`: For cycles within a week, like Monday vs. Tuesday (7 seasons).
    -   `'week_of_year'`: For weekly patterns across a year (52/53 seasons).
    -   `'day_of_year'`: For daily patterns across a year (365/366 seasons).
    -   `'biweekly'`: Two-week periods across a year (26/27 seasons).
    -   `'hour'`, `'minute'`, `'second'`: Cycles within a day, hour or minute (24/60/60 seasons).
    -   `'year'`: A single season (equivalent to a non-seasonal test on the same data).
    The choice is critical: using `'day_of_year'` for data collected over five years means you are testing for a trend in Jan 1st across those five years, Jan 2nd across those five years, and so on. See **[Example 11: Advanced Seasonality](../11_Advanced_Seasonality_Non_Monthly/README.md)**.

---

### Autocorrelation & Bootstrap Parameters

These parameters enable the block bootstrap for data that are correlated from one cycle to the next (e.g. a wet year affecting all months).

#### `autocorr_method`
-   **Type:** `str`, **Default:** `'none'`
-   **Description:** Method to handle autocorrelation.
-   **Usefulness:**
    *   `'none'` (Default): No correction (the seasonal test assumes independence between years).
    *   `'block_bootstrap'`: Resamples whole cycles (e.g. years) in circular blocks. The p-value comes from cycles detrended with the seasonal slope (simulating no trend), and the slope's confidence interval from resampling the observed cycles (percentile interval). See [bootstrap.md](../../bootstrap.md).
    *   `'auto'` and `'yue_wang'` are only available in `trend_test`.

#### `block_size`
-   **Type:** `int` or `'auto'`, **Default:** `'auto'`
-   **Description:** The size of the blocks used for resampling.
-   **Usefulness:** If your seasonal data is autocorrelated over years (e.g., January 2000 is correlated with January 2001), standard tests may be invalid. Block bootstrapping corrects for this.
    *   `'auto'` (Default): Chosen from the autocorrelation of the detrended cycle means (in numbers of cycles).
    *   `int`: Manually specifies the block size, in cycles.

#### `n_bootstrap`
-   **Type:** `int`, **Default:** `1000`
-   **Description:** The number of bootstrap resamples to generate.
-   **Usefulness:** Controls the precision of the bootstrap p-value, `(1 + #{|S*| >= |S|}) / (n_bootstrap + 1)`. Set `random_state` for reproducible results.

---

### Aggregation Parameters

Aggregation in a seasonal context is about ensuring that you have **one representative value per season per cycle**. For example, one value for "January 2010", one for "January 2011", etc.

#### `agg_method`
-   **Type:** `str`, **Default:** `'none'`
-   **Description:** This determines the method for reducing multiple data points that fall within the same season-cycle block (e.g., all samples in "January 2010") to a single representative point.
-   **Usefulness and Implications:**
    -   **When is it needed?** If you have several samples for a season-year (e.g., two samples in Jan 2010), the test still runs (pairs within a season are compared in time order), but clustered sampling then gives those periods more weight. Aggregating to one value per season per cycle is the standard approach, and it is what LWP-TRENDS does.
    -   `'none'`: No aggregation.
    -   `'lwp'`: Selects the single observation closest to the mid-point of the season's time increment, using the LWP-TRENDS mid-points (LWP R script `UseMidObs=TRUE`).
    -   `'median'` (alias `'lwp_median'`): Calculates the median of all values within the season-cycle block (LWP R script `UseMidObs=FALSE`).
    -   `'robust_median'` (alias `'lwp_robust_median'`): As above, but uses a robust median logic suitable for censored data.
    -   `'middle'`: Selects the observation whose timestamp is closest to the mean of the actual timestamps in that season-year group.
    -   `'middle_lwp'`: Selects the observation closest to the mid-point of the season's time increment (LWP-TRENDS definition; same rule as `'lwp'`).
-   **Note:** `agg_period` is not used by `seasonal_trend_test`; the aggregation groups always follow `season_type`/`period`.
-   **Limitations:** **Aggregating censored data is statistically complex and can introduce bias.** For example, the median of `['<2', '5', '10']` is `5`, but the median of `['<2', '<5', '10']` is ambiguous. The `'robust_median'` method uses a reasonable heuristic, but you should be aware of this underlying uncertainty. See **[Example 8](../08_Aggregation_and_Clustered_Data/README.md)**.

---

### Censored Data Parameters

#### `sens_slope_method`
-   **Type:** `str`, **Default:** `'ats'`
-   **Description:** Slope estimator used when the data contain censored values (uncensored data always use the exact seasonal Sen's slope: the median of all within-season pairwise slopes).
-   **Usefulness:**
    *   `'ats'` (Default): The **seasonal Akritas-Theil-Sen** estimator: the slope at which the *sum* of the within-season Kendall scores of the residuals is zero, treating censored values as intervals (no substitution). Its confidence interval inverts the seasonal Kendall test, and `sen_probability` is reported. Recommended for censored data.
    *   `'lwp'`: Pairwise slopes with censored values substituted (`lt_mult`, `gt_mult`) and ambiguous pairs set to 0, exactly as LWP-TRENDS. Biased towards zero as censoring increases.
    *   `'drop_ambiguous'` (formerly `'unbiased'`/`'nan'`, which still work but are deprecated): Pairwise slopes with ambiguous pairs dropped. Biased *away* from zero as censoring increases.
-   **Limitations:** ATS is computed from all within-season pairs (sampled pairs in fast mode), which takes a few seconds for several thousand censored observations.

#### `hicensor`
-   **Type:** `bool` or `float`, **Default:** `False`
-   **Description:** Activates the "high censor" rule (see `trend_test`). It is applied to the whole series before seasons are formed: all values below the highest left-censoring limit (or below `min(that limit, hicensor)` when numeric) are set as censored at that limit.

#### `lt_mult` and `gt_mult`
-   **Type:** `float`, **Defaults:** `lt_mult=0.5`, `gt_mult=1.1`
-   **Description:** Multipliers used to substitute censored values in the **pairwise** slope estimators (`'lwp'`, `'drop_ambiguous'`). They do not affect the test or the default `'ats'` slope.
-   **Usefulness:** Identical to `trend_test`, but applied to the seasonal data.

#### `mk_test_method`
-   **Type:** `str`, **Default:** `'robust'`
-   **Description:** Controls how the Mann-Kendall test ranks right-censored data (`>`).
-   **Usefulness:** Identical to `trend_test`.

---

### Advanced Statistical Parameters

#### `tau_method`
-   **Type:** `str`, **Default:** `'b'`
-   **Description:** Controls which version of Kendall's Tau is calculated.
-   **Usefulness:** 'b' (default) adjusts the denominator for ties; 'a' divides by the number of pairs and is what LWP-TRENDS reports. Only the reported `Tau` changes.

#### `tie_break_method`
-   **Type:** `str`, **Default:** `'robust'`
-   **Description:** Internal constant used in the variance of S; `'lwp'` uses exactly the LWP-TRENDS value. Both give the same results in practice.

#### `ci_method`
-   **Type:** `str`, **Default:** `'direct'`
-   **Description:** Controls the method for calculating the Sen's slope confidence intervals.
-   **Usefulness:** Identical to `trend_test`.

---

### Other Parameters

#### `min_size_per_season`
-   **Type:** `int` (optional), **Default:** `5`
-   **Description:** The minimum number of observations per season below which an analysis note is added.
-   **Usefulness and Implications:** A trend for "January" based on only two data points is not reliable. Seasons below this size are **not** removed (all seasons with at least two observations contribute to S); the note `minimum season size (X) below minimum (Y)` flags the result for careful interpretation. Separately, as in LWP-TRENDS, a season with fewer than 3 values (or none at all, for month/quarter/day-of-week/hour seasons) triggers the note `< 3 non-NA values in Season`. Use `None` to disable the size note.

#### `category_map`
-   **Type:** `dict` (optional), **Default:** `None`
-   **Description:** Allows you to provide your own custom rules for trend classification, applied to the final overall trend result.
-   **Usefulness:** Same as in `trend_test`, this allows you to align the output with specific reporting requirements. See our **[Trend Classification Guide](./trend_classification_guide.md)**.

#### `continuous_confidence`
-   **Type:** `bool`, **Default:** `True`
-   **Description:** Controls how trend direction is reported.
-   **Usefulness:**
    -   `True` (default): The trend classification is based on the continuous confidence `C` in the trend direction.
    -   `False`: The function behaves like a classical hypothesis test (significant or "No Trend").

---

### Slope Scaling Parameters

#### `x_unit` and `slope_scaling`
These parameters function identically to their counterparts in `trend_test`. They allow you to provide a unit for your data (`x_unit`) and specify a desired time unit for the final Sen's slope (`slope_scaling`), such as `'year'`. This is a powerful convenience feature that removes the need for manual slope conversion when working with datetime objects.

-   **Usefulness:** Automatically get an interpretable slope like "mg/L per year" directly in the output and on the plot.
-   **Limitations:** `slope_scaling` is only effective when your time vector `t` is datetime-like.

For a full explanation, please see the **[Slope Scaling Parameters section in the `trend_test` guide](./trend_test_parameters_guide.md#slope-scaling-parameters)**.

---

### Large Dataset Parameters

`large_dataset_mode`, `max_pairs` and `random_state` work as in `trend_test`. In fast mode, sampled pairs are allocated to seasons in proportion to their number of within-season pairs, so the pooled slope weights seasons exactly as the full calculation does.

#### `max_per_season`
-   **Type:** `int` (optional), **Default:** `None` (1,000 in fast mode)
-   **Description:** For very large data sets (more than 10,000 observations in fast mode), each season is randomly subsampled to at most this many observations before the test. An analysis note records when this happens. See the **[Large Dataset Guide](./large_dataset_guide.md)**.
