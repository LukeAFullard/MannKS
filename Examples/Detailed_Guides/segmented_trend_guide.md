# Segmented Trend Analysis Guide

## Overview

The `segmented_trend_test` provides a **Hybrid Segmented Trend Analysis**. It is designed to detect structural changes (breakpoints) in a time series and analyze the trends within each distinct segment.

Unlike a standard trend test which assumes a monotonic trend across the entire dataset, segmented analysis can tell you:
*   "The trend was increasing until 2010, then it stabilized."
*   "There was a sudden shift in the mean value in 2015."

### The Hybrid Approach
The method combines two powerful statistical techniques:
1.  **Piecewise Regression (OLS):** Used to identify the *location* of the breakpoints. It can automatically select the optimal number of breakpoints using Information Criteria (BIC or AIC).
2.  **Mann-Kendall / Sen's Slope:** Once the segments are defined, these robust non-parametric methods are applied to each segment individually to estimate the slope and significance. This ensures the trend estimates are robust to outliers and non-normality, which pure OLS is not.

## Basic Usage

```python
from MannKS import segmented_trend_test, plot_segmented_trend

# 1. Run the test
# Automatically find up to 3 breakpoints
result = segmented_trend_test(
    x=data,
    t=dates,
    max_breakpoints=3,
    criterion='bic',      # Bayesian Information Criterion for selection
    slope_scaling='year'  # Scale slopes to "units per year"
)

# 2. Inspect Results
print(f"Detected {result.n_breakpoints} breakpoints.")
for i, segment in enumerate(result.segments.itertuples()):
    print(f"Segment {i+1}: Slope = {segment.slope:.2f}, CI = ({segment.lower_ci:.2f}, {segment.upper_ci:.2f})")

# 3. Visualize
plot_segmented_trend(result, x_data=data, t_data=dates, save_path='segmented_plot.png')
```

## Key Parameters

### Breakpoint Selection
*   **`n_breakpoints`**: (int, optional) If you know exactly how many breakpoints to look for (e.g., you are testing for a specific known intervention), set this value.
*   **`max_breakpoints`**: (int, default=5) If `n_breakpoints` is not set, the function will test all models from 0 up to `max_breakpoints` and select the best one.
*   **`criterion`**: (str, default='bic') The metric used to select the best number of breakpoints.
    *   `'bic'`: Bayesian Information Criterion. Penalizes complexity more heavily. Better for finding strong, clear signals and avoiding false positives.
    *   `'aic'`: Akaike Information Criterion. Less penalizing. Better for prediction but may find "spurious" breakpoints in noisy data.
*   **`slope_scaling`**: (str, optional) Scale the slope to a specific time unit (e.g., `'year'`). Only applicable if `t` contains datetime objects.

### Robustness via Bagging
*   **`use_bagging`**: (bool, default=False)
    *   **The Problem:** Breakpoint detection in noisy data can be unstable. Removing a few points might shift the detected breakpoint by months or years.
    *   **The Solution:** Bagging (Bootstrap Aggregating) resamples the data 100+ times, finds the breakpoints in each sample, and uses Density Estimation (KDE) to find the most robust breakpoint locations.
    *   **Recommendation:** Set to `True` for noisy environmental data or when you need confidence intervals for the breakpoint locations themselves.
*   **`n_bootstrap`**: (int, default=100) Number of iterations for bagging. Higher = more stable but slower.

### Censored Data
*   **`hicensor`**: (bool, default=False)
    *   **The Problem:** Standard OLS (used for breakpoint detection) cannot handle `<5` strings.
    *   **The Solution:** If `True`, the function applies a "High Censor" rule (treating values below the max detection limit as censored) and substitutes them with a fixed value (e.g., 0.5 * limit) specifically for the *breakpoint detection* phase.
    *   **Note:** The subsequent slope estimation (Mann-Kendall) still uses the fully robust censored methods (ATS, etc.) on the identified segments.

## Interpreting the Results

The function returns a `SegmentedTrendResult` object containing:

### `breakpoints`
A list of timestamps (or numeric values) where the structural changes occur.

### `segments`
A pandas DataFrame, with one row for each period between breakpoints. Columns include:
*   `slope`: Sen's slope for that segment.
*   `intercept`: The intercept for the line equation $y = mx + c$.
*   `lower_ci`, `upper_ci`: Confidence bounds for the slope.
*   `n`: Number of samples in the segment.

### `predict(t)`
A method to generate the fitted trend line for any time vector `t`. Useful for plotting.

```python
# Generate trend line for plotting
y_fitted = result.predict(dates)
```

## Advanced: Breakpoint Uncertainty

The result always includes **Confidence Intervals for the Breakpoints** (`result.breakpoint_cis`), but their source depends on the method used:

1.  **Standard OLS (`use_bagging=False`)**: The CIs are derived from the standard error of the Piecewise Regression model. These assume normality and are typically symmetric.
2.  **Bootstrap Bagging (`use_bagging=True`)**: The CIs are derived non-parametrically from the bootstrap distribution of the breakpoints. These are more robust to noise and can handle asymmetric uncertainty (e.g., "the change happened *at least* by 2010, but possibly much earlier").

*   A wide CI (e.g., "2010 to 2014") indicates the exact timing of the change is uncertain, often due to a gradual transition or high noise.
*   A narrow CI (e.g., "May 2010 to June 2010") indicates a sharp, sudden change.

### Calculating Probability of a Change
If you used **bagging** (`use_bagging=True`), you can calculate the probability that a structural break occurred within a specific time window. This is not possible with standard OLS.

```python
from MannKS import calculate_breakpoint_probability

# Calculate probability a change occurred in 2010
prob = calculate_breakpoint_probability(
    result,
    start_date='2010-01-01',
    end_date='2011-01-01'
)
print(f"Probability: {prob:.1%}")
```

See **[Example 32](../32_Segmented_Regression/README.md)** for a detailed walkthrough of bagging and uncertainty.
