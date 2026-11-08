"""
Example 35: Large Dataset Trend Analysis

Generates synthetic data sets with known trends (n = 12,000), analyses them
with trend_test, seasonal_trend_test and segmented_trend_test, and writes
README.md with the code, the output and the interpretation.
"""
import contextlib
import io
import os
import time

import numpy as np
import pandas as pd
from MannKS import (trend_test, seasonal_trend_test, segmented_trend_test,
                    prepare_censored_data)


def print_result(title, result, true_slope, elapsed):
    print(f"\n--- {title} ---")
    print(f"Time: {elapsed:.2f}s")
    print(f"Mode: {result.computation_mode}")
    if result.pairs_used is not None:
        print(f"Pairs used for the slope: {result.pairs_used:,}")
    print(f"Estimated slope: {result.slope:.6f} {result.slope_units}")
    print(f"True slope:      {true_slope:.6f}")
    print(f"Error:           {abs(result.slope - true_slope) / abs(true_slope):.2%}")
    if result.approximation_error is not None:
        print(f"Sampling error (95% half-width, approximation_error): {result.approximation_error:.3g}")
    print(f"Conf. interval:  ({result.lower_ci:.6f}, {result.upper_ci:.6f})")
    print(f"Trend: {result.trend}, p = {result.p:.3g}")
    if result.analysis_notes:
        print(f"Notes: {result.analysis_notes}")


def main():
    rng = np.random.default_rng(42)
    n = 12000
    t = np.arange(n)

    # 1. Linear trend, medium noise
    x1 = 0.5 * t + rng.normal(0, 10, n)
    start = time.time()
    res1 = trend_test(x1, t)
    print_result("1. Linear trend (medium noise)", res1, 0.5, time.time() - start)

    # 2. Linear trend, high noise
    x2 = 0.1 * t + rng.normal(0, 50, n)
    start = time.time()
    res2 = trend_test(x2, t)
    print_result("2. Linear trend (high noise)", res2, 0.1, time.time() - start)

    # 3. Censored data: values below a detection limit of 1000 reported as '<1000'
    x3 = 0.5 * t + rng.normal(0, 200, n)
    values = [f'<1000' if v < 1000 else v for v in x3]
    data3 = prepare_censored_data(values)
    start = time.time()
    res3 = trend_test(data3, t)
    print(f"\n(Proportion censored: {res3.prop_censored:.0%})", end='')
    print_result("3. Censored data (Akritas-Theil-Sen slope)", res3, 0.5, time.time() - start)

    # 4. Seasonal trend: hourly data with a daily cycle
    dates = pd.date_range(start='2000-01-01', periods=n, freq='h')
    x4 = 0.2 * t + 20 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 5, n)
    start = time.time()
    res4 = seasonal_trend_test(x4, dates, season_type='hour', slope_scaling='hour')
    print_result("4. Seasonal trend (24 hourly seasons)", res4, 0.2, time.time() - start)

    # 5. Segmented trend: slope 1.0, then -0.5
    half = n // 2
    x5 = np.r_[1.0 * t[:half], 1.0 * half - 0.5 * (t[half:] - half)] + rng.normal(0, 5, n)
    start = time.time()
    res5 = segmented_trend_test(x5, t, n_breakpoints=1, random_state=42)
    print(f"\n--- 5. Segmented trend ---")
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Breakpoint: {res5.breakpoints[0]:.1f} (true: {half})")
    for i, row in res5.segments.iterrows():
        print(f"  Segment {i + 1}: slope = {row['slope']:.4f}, CI = ({row['lower_ci']:.4f}, {row['upper_ci']:.4f})")
    print("True slopes: segment 1 = 1.0, segment 2 = -0.5")


CODE = '''import numpy as np
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
'''

INTRO = """# Example 35: Large Dataset Trend Analysis

## The "Why": Handling Big Environmental Data
Hourly sensor data over a decade yields nearly 90,000 observations. The Mann-Kendall test and Sen's slope compare every pair of observations, and the number of pairs grows with the square of n (72 million pairs for n = 12,000).

MannKS handles this automatically:
1.  **The Mann-Kendall test is always exact.** S, Var(S) and the p-value are computed with an O(n log n) algorithm, for censored and seasonal data too.
2.  **The slope is estimated from a random sample of pairs** (fast mode, automatic above n = 5,000). The sampling is reproducible, and `approximation_error` reports how large the sampling error can be (95% half-width).
3.  **Seasonal tests use every observation.** When pairs are sampled for the slope, each season gets a share of the budget in proportion to its number of pairs.

## The "How": Code Walkthrough
Five scenarios with n = 12,000 and known true slopes:

```python
{code}```

## Output
Timings depend on the machine.

```text
{output}```

## Interpretation & Insights

1.  **Speed**: each fast-mode test on 12,000 points takes a fraction of a second. Exact mode would need all 72 million pairwise slopes (several seconds and gigabytes of memory).
2.  **Accuracy**: the slope errors are well below 1%, and each confidence interval contains the true slope. The difference between the sampled and the exact slope is of the order of `approximation_error`, which is much smaller than the confidence interval. Fast mode adds little uncertainty beyond that of the data.
3.  **Censored data**: censored data run as fast as uncensored data. The default Akritas-Theil-Sen estimator uses the censored values as intervals (no substitution) and recovers the true slope.
4.  **Seasonal data**: the seasonal test uses all 12,000 observations (500 per season). Before version 0.6.1, seasons were cut to 1,000 observations by default; this is now only done if you set `max_per_season`.
5.  **Segmented data**: the breakpoint and both segment slopes are recovered. The breakpoint search (`piecewise-regression`) is the slowest part and scales with n. Above about 20,000 observations, aggregate first.

See the [Large Dataset Guide](../Detailed_Guides/large_dataset_guide.md) for accuracy measurements and performance figures.
"""


if __name__ == "__main__":
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main()
    output = buf.getvalue().lstrip('\n')
    print(output)
    readme = INTRO.replace('{code}', CODE).replace('{output}', output)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md'), 'w') as f:
        f.write(readme)
