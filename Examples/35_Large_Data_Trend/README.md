# Example 35: Large Dataset Trend Analysis

## The "Why": Handling Big Environmental Data
Hourly sensor data over a decade yields nearly 90,000 observations. The Mann-Kendall test and Sen's slope compare every pair of observations, and the number of pairs grows with the square of n (72 million pairs for n = 12,000).

MannKS handles this automatically:
1.  **The Mann-Kendall test is always exact.** S, Var(S) and the p-value are computed with an O(n log n) algorithm, for censored and seasonal data too.
2.  **The slope is estimated from a random sample of pairs** (fast mode, automatic above n = 5,000). The sampling is reproducible, and `approximation_error` reports how large the sampling error can be (95% half-width).
3.  **Seasonal tests use every observation.** When pairs are sampled for the slope, each season gets a share of the budget in proportion to its number of pairs.

## The "How": Code Walkthrough
Five scenarios with n = 12,000 and known true slopes:

```python
import numpy as np
import pandas as pd
from MannKS import trend_test, seasonal_trend_test, segmented_trend_test, prepare_censored_data

rng = np.random.default_rng(42)
n = 12000
t = np.arange(n)

# 1-2. Linear trends: fast mode is chosen automatically for n > 5,000
res1 = trend_test(0.5 * t + rng.normal(0, 10, n), t)
res2 = trend_test(0.1 * t + rng.normal(0, 50, n), t)

# 3. Censored data (values below 1000 reported as '<1000')
x3 = 0.5 * t + rng.normal(0, 200, n)
res3 = trend_test(prepare_censored_data([f'<1000' if v < 1000 else v for v in x3]), t)

# 4. Seasonal: hourly data with a daily cycle, 24 hourly seasons
dates = pd.date_range(start='2000-01-01', periods=n, freq='h')
x4 = 0.2 * t + 20 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 5, n)
res4 = seasonal_trend_test(x4, dates, season_type='hour', slope_scaling='hour')

# 5. Segmented: slope 1.0 then -0.5
half = n // 2
x5 = np.r_[1.0 * t[:half], 1.0 * half - 0.5 * (t[half:] - half)] + rng.normal(0, 5, n)
res5 = segmented_trend_test(x5, t, n_breakpoints=1, random_state=42)

for r in (res1, res2, res3, res4):
    print(r.computation_mode, r.slope, r.approximation_error, (r.lower_ci, r.upper_ci))
```

## Output
Timings depend on the machine.

```text
--- 1. Linear trend (medium noise) ---
Time: 0.05s
Mode: fast
Pairs used for the slope: 99,919
Estimated slope: 0.499995 units per unit of t
True slope:      0.500000
Error:           0.00%
Sampling error (95% half-width, approximation_error): 2.75e-05
Conf. interval:  (0.499944, 0.500047)
Trend: increasing, p = 0

--- 2. Linear trend (high noise) ---
Time: 0.05s
Mode: fast
Pairs used for the slope: 99,919
Estimated slope: 0.100192 units per unit of t
True slope:      0.100000
Error:           0.19%
Sampling error (95% half-width, approximation_error): 0.000142
Conf. interval:  (0.099917, 0.100441)
Trend: increasing, p = 0

(Proportion censored: 17%)
--- 3. Censored data (Akritas-Theil-Sen slope) ---
Time: 0.05s
Mode: fast
Pairs used for the slope: 99,919
Estimated slope: 0.499298 units per unit of t
True slope:      0.500000
Error:           0.14%
Sampling error (95% half-width, approximation_error): 0.000765
Conf. interval:  (0.497731, 0.501064)
Trend: increasing, p = 0

--- 4. Seasonal trend (24 hourly seasons) ---
Time: 0.46s
Mode: fast
Pairs used for the slope: 98,424
Estimated slope: 0.199999 units per hour
True slope:      0.200000
Error:           0.00%
Sampling error (95% half-width, approximation_error): 1.48e-05
Conf. interval:  (0.199972, 0.200027)
Trend: increasing, p = 0

--- 5. Segmented trend ---
Time: 5.54s
Breakpoint: 5999.9 (true: 6000)
  Segment 1: slope = 1.0000, CI = (1.0000, 1.0001)
  Segment 2: slope = -0.5000, CI = (-0.5000, -0.4999)
True slopes: segment 1 = 1.0, segment 2 = -0.5
```

## Interpretation & Insights

1.  **Speed**: each fast-mode test on 12,000 points takes a fraction of a second. Exact mode would need all 72 million pairwise slopes (several seconds and gigabytes of memory).
2.  **Accuracy**: the slope errors are well below 1%, and each confidence interval contains the true slope. The difference between the sampled and the exact slope is of the order of `approximation_error`, which is much smaller than the confidence interval. Fast mode adds little uncertainty beyond that of the data.
3.  **Censored data**: censored data run as fast as uncensored data. The default Akritas-Theil-Sen estimator uses the censored values as intervals (no substitution) and recovers the true slope.
4.  **Seasonal data**: the seasonal test uses all 12,000 observations (500 per season). Before version 0.6.1, seasons were cut to 1,000 observations by default; this is now only done if you set `max_per_season`.
5.  **Segmented data**: the breakpoint and both segment slopes are recovered. The breakpoint search (`piecewise-regression`) is the slowest part and scales with n. Above about 20,000 observations, aggregate first.

See the [Large Dataset Guide](../Detailed_Guides/large_dataset_guide.md) for accuracy measurements and performance figures.
