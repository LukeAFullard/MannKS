# A Comprehensive Guide to Analysis Notes

The `MannKS` package includes a system of "Analysis Notes" to provide data quality warnings. These notes do not stop the analysis, but they alert you to potential issues in your dataset that could affect the reliability and interpretation of the trend results. This guide provides a deep dive into what each note means, its statistical and practical implications, and what actions you can take in response.

---

## Data Sufficiency & Quality

### `Data all NA values`
-   **What it means:** The input data vector contains only `NaN` (Not a Number) or missing values.
-   **Implications:** No analysis can be performed. The result will be empty/null.
-   **Recommended Actions:** Check your data loading or filtering process.

### `< 3 unique values`
-   **What it means:** There are fewer than three distinct, non-censored values in your dataset. For example, the data might be `[1, 1, 2, 2, 1, 1, 2, 2]`.
-   **Usefulness and Implications:** This note tells you that your data has extremely low variability. A trend analysis seeks to find a systematic change over time, but with only one or two unique values, it's impossible to distinguish a trend from random fluctuation. The mathematical foundation of the test breaks down, and any result would be statistically meaningless.
-   **Limitations and Nuances:** This check is applied *after* filtering out any censored or NaN values. A dataset might look varied at first glance but be reduced to this state after cleaning.
-   **Recommended Actions:**
    1.  **Verify Data Source:** Check if your data query or measurement process is correct. This note often points to an error in data collection or processing.
    2.  **Re-evaluate Goal:** If the data is correct, trend analysis is not the appropriate tool for this dataset. Consider using descriptive statistics instead.

### `< 5 Non-censored values`
-   **What it means:** There are fewer than five non-censored data points available for the analysis.
-   **Usefulness and Implications:** This is a critical warning about **low statistical power**. The Mann-Kendall test, like any statistical test, needs a minimum amount of data to reliably detect a trend. With too few points, the test is highly unlikely to find a statistically significant result (a low p-value), even if a real trend exists. You have a high risk of a "false negative" conclusion.
-   **Limitations and Nuances:** The threshold of 5 is a heuristic; it is not a magic number. However, it represents a common minimum standard for this type of non-parametric test.
-   **Recommended Actions:**
    1.  **Be Cautious:** Treat any results with extreme skepticism. A "No Trend" conclusion is not very meaningful because the test lacked the power to find one.
    2.  **Collect More Data:** If possible, the best solution is to expand your dataset with more observations over a longer period.

### `sample size (X) below minimum (Y)`
-   **What it means:** The total number of data points in your dataset is below a recommended minimum size (default is 10).
-   **Usefulness and Implications:** Similar to the `< 5 Non-censored values` note, this warns of low statistical power. It serves as a general guideline for result reliability. A trend found in a very small dataset might be a result of random chance, even if statistically significant.
-   **Limitations and Nuances:** The default `min_size=10` is a configurable parameter. In some disciplines, a different minimum may be standard. This check is about the total sample size, not just the uncensored portion.
-   **Recommended Actions:**
    1.  **Justify a Smaller Sample:** If you proceed, you should have a strong reason for trusting the results from such a small dataset.
    2.  **Adjust the Threshold:** You can change the `min_size` parameter in `trend_test` if your analysis has different requirements. See [Example 26](../26_advanced_parameter_nuances/README.md).

### `insufficient data`
-   **What it means:** There were fewer than 2 data points available for analysis (either initially or after filtering).
-   **Implications:** The trend test was aborted and returns default/NaN values.

### `insufficient data post-aggregation`
-   **What it means:** After aggregating the data (e.g., thinning daily data to yearly), there were fewer than 2 data points remaining.
-   **Implications:** The aggregation was too aggressive for the amount of data available.

---

## Data Structure & Aggregation

### `Long run of single value`
-   **What it means:** A significant portion of your data (by default, >50%) consists of a single, repeated value (e.g., `[1, 2, 3, 3, 3, 3, 3, 3, 4, 5]`).
-   **Usefulness and Implications:** This highlights a high number of **ties** in your data. While the Mann-Kendall test (specifically, Kendall's Tau-b) has corrections for ties, an excessive number can still reduce the test's statistical power, making it harder to detect a real trend. It can also artificially pull the Sen's slope magnitude towards zero if the tied value is prevalent throughout the time series.
-   **Limitations and Nuances:** This can be a natural feature of discrete data (e.g., counts of items) or data with a low measurement resolution.
-   **Recommended Actions:**
    1.  **Investigate the Value:** Is the repeated value a true measurement, or does it represent a reporting limit or a default placeholder (e.g., `0`)?
    2.  **Consider the Context:** If the ties are legitimate, proceed with the analysis but be aware that the statistical power might be reduced.

### `tied timestamps present without aggregation`
-   **What it means:** Your time vector `t` contains duplicate values (e.g., multiple measurements were recorded at the exact same time), and you have not chosen an aggregation method.
-   **Usefulness and Implications:** This is a critical warning. The Sen's slope is calculated from the slopes of all pairs of points. If two points have the same timestamp, the slope denominator (`t_j - t_i`) is zero, leading to an infinite slope (`inf`), which breaks the calculation. The `MannKS` package removes these pairs, but this can **bias the result**. It effectively ignores some of your data in an unmanaged way.
-   **Limitations and Nuances:** This is especially problematic if the tied values are not identical. If you have `(t1, x1)` and `(t1, x2)`, the package cannot know which value truly represents that time point.
-   **Recommended Actions:**
    1.  **Aggregate (Tie-Breaking):** Use `agg_method='median'` or `'robust_median'` to combine the duplicate-time observations into a single point. This keeps the timestamps as they are.
    2.  **Aggregate (Thinning):** If the ties are part of a larger clustering issue (e.g., daily samples for monthly analysis), consider using `agg_method='lwp_median'` to thin the data to a consistent frequency. See [Example 9](../09_Aggregation_Tied_Clustered_Data/README.md) for a detailed guide.

### `'[method]' aggregation used with censored data`
-   **What it means:** You are using an aggregation method (like `'median'`) that may not be statistically rigorous for censored data.
-   **Implications:** Standard median calculations on data like `['<5', '10']` can be ambiguous or biased.
-   **Recommended Action:** Use `agg_method='robust_median'` or `agg_method='lwp_robust_median'` which are designed to handle censored values more safely.

---

## Seasonal Analysis Notes

These notes are specific to the `seasonal_trend_test`.

### `< 3 non-NA values in Season` / `< 2 unique values in Season`
-   **What it means:** A specific season (e.g., "January") has very little usable data across the entire record.
-   **Implications:** That specific season cannot be reliably analyzed for a trend. It will likely be dropped or contribute very little to the overall seasonal test.

### `Long run of single value in a Season`
-   **What it means:** Within a specific season (e.g., "March"), more than 75% of the values are identical.
-   **Implications:** Similar to the general "Long run" note, but specific to one season. It reduces the power of the test for that season.

### `minimum season size (X) below minimum (Y)`
-   **What it means:** One or more seasons have fewer observations than the `min_size_per_season` threshold.
-   **Implications:** The trend analysis for those seasons is considered unreliable due to small sample size.

---

## Sen's Slope & Censored Data

### `WARNING: Sen slope influenced by [left/right]-censored values.`
-   **What it means:** The final reported Sen's slope (which is the median of all pairwise slopes) was a slope calculated from a pair of data points where at least one of the points was censored.
-   **Usefulness and Implications:** This is a crucial advisory note. It tells you that the **magnitude** of your trend is an **estimate** that is directly influenced by the substitution multipliers (`lt_mult`, `gt_mult`). A different choice of multiplier could lead to a different slope. This note increases transparency, reminding you that while the *direction* of the trend might be statistically robust, the *magnitude* has a degree of uncertainty tied to how censored data was handled.
-   **Limitations and Nuances:** This is very common in censored datasets and is not necessarily an error, but it requires careful interpretation.
-   **Recommended Actions:**
    1.  **Acknowledge Uncertainty:** When reporting the Sen's slope, mention that its magnitude is an estimate influenced by censored values.
    2.  **Sensitivity Analysis:** As a best practice, re-run the test with different multipliers (e.g., `lt_mult=0.55`, `lt_mult=0.6`) to see how sensitive your slope estimate is. If it changes dramatically, be more cautious. See [Example 13](../13_Censored_Data_Multipliers/README.md).

### `CRITICAL: Sen slope is based on a pair of two censored values.`
-   **What it means:** The median Sen's slope was calculated from a pair where *both* points were censored (e.g., the slope between `<5` and `<10`).
-   **Usefulness and Implications:** This is a more severe version of the warning above. The slope's magnitude is now **entirely dependent** on the substitution multipliers. The calculation `(10*lt_mult - 5*lt_mult) / (t2 - t1)` has very little connection to the true, unobserved values. The resulting slope should be considered a rough estimate at best.
-   **Limitations and Nuances:** This is rare but can occur in heavily censored datasets with few uncensored points to anchor the calculations.
-   **Recommended Actions:**
    1.  **Extreme Caution:** Treat the Sen's slope result with extreme skepticism. It is not a reliable indicator of the true trend magnitude.
    2.  **Focus on Significance:** In this situation, the p-value (i.e., whether a trend exists at all) is a more reliable result than the Sen's slope.

### `WARNING: Sen slope based on tied non-censored values`
-   **What it means:** The calculated Sen's slope is approximately zero, often because the median slope is derived from a pair of tied values (slope = 0).
-   **Implications:** This typically happens when the trend is flat or "No Trend". It serves as a confirmation that the lack of trend is driven by tied data points.
