# Rolling Trend Analysis

## Overview

Rolling trend analysis calculates Sen's slope and Mann-Kendall statistics over moving time windows. This technique reveals how trends evolve over time, identifying periods of acceleration, deceleration, or reversal that a single global trend test might miss.

## Key Concepts

*   **Window Size:** The duration of time (e.g., '10 years') included in each trend test.
*   **Step Size:** How far the window advances for the next test (e.g., '1 year').
*   **Continuous Confidence:** Instead of binary "significant/not significant" results, rolling analysis typically tracks the *confidence* in the trend direction (e.g., "Likely Increasing" -> "Very Likely Increasing").

## Basic Usage

```python
import pandas as pd
import numpy as np
from MannKS import rolling_trend_test, plot_rolling_trend

# 1. Prepare Data
# Simulate 20 years of monthly data with an accelerating trend
dates = pd.date_range('2000-01-01', periods=240, freq='ME')
values = np.linspace(0, 10, 240) + np.random.normal(0, 2, 240)
# Make the trend steeper in the second half
values[120:] += np.linspace(0, 10, 120)

# 2. Run Rolling Trend Test
results = rolling_trend_test(
    x=values,
    t=dates,
    window='10Y',        # 10-year moving window
    step='1Y',           # Advance 1 year at a time
    min_size=20,         # Require at least 20 points
    slope_scaling='year' # Scale slope to units/year
)

print(results[['window_center', 'slope', 'classification']].head())

# 3. Visualize
plot_rolling_trend(
    results,
    data=pd.DataFrame({'date': dates, 'value': values}),
    time_col='date',
    value_col='value',
    highlight_significant=True,
    save_path='rolling_trend_plot.png'
)
```

## Interpreting the Output

The `rolling_trend_test` function returns a DataFrame where each row represents a time window:

| Column | Description |
| :--- | :--- |
| `window_start` | Start time of the window (inclusive). |
| `window_end` | End time of the window (exclusive). |
| `window_center` | Midpoint of the window (useful for x-axis plotting). |
| `slope` | Sen's slope for that specific window. |
| `C` | Confidence in trend direction (0.5 = random, 1.0 = certain). |
| `h` | Boolean: is the trend statistically significant at `alpha`? |
| `classification` | Descriptive trend category (e.g., "Likely Increasing"). |

## Before/After Comparison

If you suspect a specific event caused a change (e.g., a policy change in 2010), you can compare the periods before and after that date directly using `compare_periods`.

```python
from MannKS import compare_periods

comparison = compare_periods(
    x=values,
    t=dates,
    breakpoint=pd.Timestamp('2010-01-01'),
    slope_scaling='year'
)

print(f"Slope Before: {comparison['before'].slope:.3f}")
print(f"Slope After:  {comparison['after'].slope:.3f}")
print(f"Significant Change? {comparison['significant_change']}")
```

## Best Practices

1.  **Window Size:** Choose a window large enough to be statistically meaningful (usually $\ge$ 10 points) but small enough to capture the changes you are interested in. For annual environmental data, 10-15 year windows are common.
2.  **Continuous Confidence:** For rolling plots, tracking the continuous confidence (`C`) or the classification is often more informative than looking for binary p-value significance, which can jump erratically near the threshold.
3.  **Data Gaps:** The function handles missing data by checking `min_size`. Windows with insufficient data are skipped.

## Edge Handling

Users often notice that rolling trend plots start and end "inside" the full data range. This is expected behavior for moving window analyses. The implementation uses an **Asymmetric** approach to balance data coverage with statistical rigor:

*   **Leading Edge (Start - Truncated):** The first window begins strictly at the start of your data. We do *not* create partial windows before this point. For a 10-year window, the first plotted point (window center) appears at Year 5. This ensures the initial trends are based on a full window of data.
*   **Trailing Edge (End - Adaptive):** As the window slides past the end of your data, it is allowed to "shrink" (become partial). The calculation continues using whatever data remains in the tail of the window, until the sample size drops below `min_size`. This allows you to see the most recent trend possible, even if the full forward window isn't complete.
*   **No Imputation:** The method strictly uses existing data. It does not attempt to extrapolate trends into the past or future to fill visual gaps.
