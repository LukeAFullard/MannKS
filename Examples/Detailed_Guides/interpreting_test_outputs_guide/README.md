# Guide to Interpreting Test Outputs

The `trend_test` and `seasonal_trend_test` functions return a `namedtuple` containing the results of the analysis. Access fields by name, e.g. `result.slope`. This guide explains each field.

## Test statistics

-   **`trend`** (`str`): Direction of the trend.
    -   With `continuous_confidence=True` (default): `'increasing'` or `'decreasing'` according to the sign of the test statistic, whether or not it is significant, or `'indeterminate'` when Z = 0.
    -   With `continuous_confidence=False`: `'increasing'`/`'decreasing'` only if significant at `alpha`, otherwise `'no trend'`.
-   **`h`** (`bool`): `True` if the trend is significant at `alpha` (two-sided).
-   **`p`** (`float`): Two-sided p-value of the Mann-Kendall test (or of the block bootstrap, if used).
-   **`z`** (`float`): Continuity-corrected Z-score, `(S - sign(S)) / sqrt(Var(S))`.
-   **`s`** (`float`): Mann-Kendall S statistic: the number of increasing pairs minus decreasing pairs (summed over seasons in the seasonal test). Pairs that cannot be ordered because of censoring, and pairs with equal times, score 0.
-   **`var_s`** (`float`): Variance of S, corrected for ties and censoring (inflated by the Yue-Wang factor, or the bootstrap variance, when those methods are used).
-   **`Tau`** (`float`): Kendall's tau (tau-b by default, tau-a with `tau_method='a'`), from -1 to 1.

## Confidence and classification

-   **`C`** (`float`): Confidence in the reported trend direction, `1 - p/2` (0.5 to 1).
-   **`Cd`** (`float`): Confidence that the trend is decreasing (`C` for a decreasing trend, `1 - C` for an increasing one).
-   **`classification`** (`str`): Label such as "Very Likely Increasing"; see the [Trend Classification Guide](../trend_classification_guide.md).
-   **`analysis_notes`** (`list` of `str`): Data-quality notes; see the [Analysis Notes Guide](../analysis_notes_guide.md). An empty list means no issues were found.
-   **`warnings`** (`list` of `str`): Warnings raised during the calculation.

## Slope

-   **`slope`** (`float`): The trend slope: Sen's slope for uncensored data, the Akritas-Theil-Sen slope for censored data by default (see `sens_slope_method`). In units of `slope_units`.
-   **`lower_ci`**, **`upper_ci`** (`float`): Two-sided (1 - `alpha`) confidence interval of the slope, in the same units.
-   **`scaled_slope`** (`float`): Same as `slope`.
-   **`slope_units`** (`str`): e.g. "mg/L per year" (with `slope_scaling`), "units per second" (datetime `t` without scaling) or "units per unit of t" (numeric `t`).
-   **`slope_per_second`**, **`lower_ci_per_second`**, **`upper_ci_per_second`** (`float`): Unscaled slope and interval, per second for datetime `t` and per unit of `t` for numeric `t`.
-   **`intercept`** (`float`): Intercept of the trend line on the unscaled time axis, at t = 0 (the Unix epoch, 1970-01-01, for datetime `t`): `value ≈ intercept + slope_per_second * t_seconds`.
-   **`sen_probability`** (`float`): Probability that the true slope is **below zero**, from the rank of zero among the pairwise slopes (LWP-TRENDS `Sen_Probability`). `sen_probability_max` and `sen_probability_min` use the largest and smallest rank among slopes tied at zero.

## Data summaries

These refer to the data **before** any aggregation, as in LWP-TRENDS.

-   **`prop_censored`** (`float`): Proportion of censored values.
-   **`prop_unique`** (`float`): Number of distinct values divided by the number of values.
-   **`n_censor_levels`** (`int`): Number of distinct censoring limits.

## Autocorrelation

-   **`acf1`** (`float`): Lag-1 autocorrelation of the detrended series (`trend_test` with an autocorrelation method).
-   **`n_effective`** (`float`): Effective sample size used by the Yue-Wang correction.
-   **`block_size_used`** (`int` or `None`): Block length of the block bootstrap.

## Large data sets

-   **`computation_mode`** (`str`): `'full'`, `'fast'`, `'aggregate'` or `'insufficient'`.
-   **`pairs_used`** (`int` or `None`): Number of pairs used for the slope in fast/aggregate mode.
-   **`approximation_error`** (`float` or `None`): Approximate 95% half-width of the sampling error of the slope when pairs were sampled.
