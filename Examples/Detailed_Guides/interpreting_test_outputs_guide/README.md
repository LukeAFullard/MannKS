
# Guide to Interpreting Test Outputs

The `trend_test` and `seasonal_trend_test` functions return a `namedtuple` object that contains a comprehensive set of results from the analysis. This guide explains what each field in the output means and how to interpret it.

## Output Fields

Here is a breakdown of each field in the output object:

-   **`trend`** (`str`): A string that classifies the trend. Possible values are:
    -   `'increasing'`: A statistically significant increasing trend was detected.
    -   `'decreasing'`: A statistically significant decreasing trend was detected.
    -   `'no trend'`: No statistically significant trend was detected.

-   **`h`** (`bool`): The hypothesis test result. `True` if a significant trend was detected (rejecting the null hypothesis of no trend), otherwise `False`.

-   **`p`** (`float`): The p-value of the Mann-Kendall test. This value indicates the probability of observing the data if there were no actual trend. A small p-value (typically ≤ 0.05) suggests that the observed trend is statistically significant.

-   **`z`** (`float`): The Z-score of the Mann-Kendall test. This is a normalized statistic where a positive value indicates an increasing trend and a negative value indicates a decreasing trend. The magnitude of `z` indicates the strength of the evidence against the null hypothesis.

-   **`Tau`** (`float`): Kendall's Tau, a non-parametric measure of the correlation between the data and time. It ranges from -1 to 1, where:
    -   `1` indicates a perfect increasing monotonic trend.
    -   `-1` indicates a perfect decreasing monotonic trend.
    -   `0` indicates no monotonic trend.

-   **`s`** (`float`): The Mann-Kendall S-statistic. This is the raw score of the test, representing the difference between the number of increasing and decreasing data pairs. It is rarely interpreted directly but is used to calculate `z` and `Tau`.

-   **`var_s`** (`float`): The variance of the S-statistic. This value accounts for ties in the data and is used to calculate the `z` score.

-   **`slope`** (`float`): The Sen's slope, representing the magnitude of the trend. By default, this is the same as `scaled_slope`.

-   **`intercept`** (`float`): The intercept of the Sen's slope line.

-   **`slope_per_second`** (`float`): The raw Sen's slope, always expressed in **units per second**. This is useful for analyses that require a consistent, unscaled time unit.

-   **`scaled_slope`** (`float`): The Sen's slope, scaled to a more intuitive time unit as specified by the `slope_scaling` parameter. For example, if you set `slope_scaling='year'`, this field will contain the slope in units per year.

-   **`slope_units`** (`str`): A string describing the units of the `scaled_slope`, combining the `x_unit` and `slope_scaling` parameters (e.g., "mg/L per year").

-   **`lower_ci`** (`float`): The lower confidence interval of the Sen's slope. This provides a lower bound on the estimated slope at the specified confidence level (e.g., 95%).

-   **`upper_ci`** (`float`): The upper confidence interval of the Sen's slope. This provides an upper bound on the estimated slope.
