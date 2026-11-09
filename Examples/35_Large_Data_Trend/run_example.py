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

    # Generate 4-panel diagnostic figure
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    try:
        sub = slice(0, n, 10)  # Subsample scatter for clean display

        # 1. Linear trends
        ax1 = axes[0, 0]
        ax1.scatter(t[sub], x1[sub], alpha=0.2, s=8, color='steelblue', label='Data (m=0.5)')
        ax1.scatter(t[sub], x2[sub], alpha=0.15, s=8, color='darkorange', label='Data (m=0.1)')
        ax1.plot(t, res1.slope * t + res1.intercept, color='navy', linewidth=2,
                 label=f'Fit (slope={res1.slope:.4f})')
        ax1.plot(t, res2.slope * t + res2.intercept, color='darkred', linewidth=2,
                 label=f'Fit (slope={res2.slope:.4f})')
        ax1.set_title('1-2. Linear Trends in Fast Mode (n=12,000)')
        ax1.set_xlabel('Index (t)')
        ax1.set_ylabel('Value')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. Censored data
        ax2 = axes[0, 1]
        uncensored_mask = (x3 >= 1000)
        ax2.scatter(t[uncensored_mask & (np.arange(n) % 10 == 0)],
                    x3[uncensored_mask & (np.arange(n) % 10 == 0)],
                    alpha=0.3, s=8, color='seagreen', label='Observed (>=1000)')
        ax2.scatter(t[~uncensored_mask & (np.arange(n) % 10 == 0)],
                    np.full((~uncensored_mask & (np.arange(n) % 10 == 0)).sum(), 1000),
                    alpha=0.4, s=12, color='crimson', marker='x', label='Censored (<1000)')
        ax2.axhline(1000, color='red', linestyle='--', linewidth=1, label='Detection Limit (1000)')
        ax2.plot(t, res3.slope * t + res3.intercept, color='darkgreen', linewidth=2,
                 label=f'ATS Fit (slope={res3.slope:.4f})')
        ax2.set_title(f'3. Censored Data (Prop Censored: {res3.prop_censored:.0%})')
        ax2.set_xlabel('Index (t)')
        ax2.set_ylabel('Value')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # 3. Seasonal trend
        ax3 = axes[1, 0]
        ax3.scatter(dates[sub], x4[sub], alpha=0.2, s=8, color='purple', label='Hourly Data (Sampled)')
        ts_sec = dates.astype('datetime64[s]').astype('int64').to_numpy()
        y_seas = res4.slope_per_second * ts_sec + res4.intercept
        ax3.plot(dates, y_seas, color='darkmagenta', linewidth=2,
                 label=f'Seasonal Sen (slope={res4.slope:.4f} /hr)')
        ax3.set_title('4. Seasonal Trend (24 Hourly Seasons, n=12,000)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Value')
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3)

        # 4. Segmented trend
        ax4 = axes[1, 1]
        ax4.scatter(t[sub], x5[sub], alpha=0.2, s=8, color='dimgray', label='Data')
        bp = res5.breakpoints[0]
        ax4.axvline(bp, color='red', linestyle='--', linewidth=1.5,
                    label=f'Detected Breakpoint ({bp:.0f})')
        for i, row in res5.segments.iterrows():
            t_seg = np.array([0, bp]) if i == 0 else np.array([bp, n - 1])
            y_seg = row['slope'] * t_seg + row['intercept']
            ax4.plot(t_seg, y_seg, linewidth=2.5,
                     label=f"Seg {i+1} (slope={row['slope']:.2f})")
        ax4.set_title('5. Segmented Trend & Breakpoint Detection')
        ax4.set_xlabel('Index (t)')
        ax4.set_ylabel('Value')
        ax4.legend(loc='lower left', fontsize=9)
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'large_data_trends.png')
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"Plot saved to {fig_path}")
    finally:
        plt.close(fig)


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

## Visual Diagnostics

![Large Dataset Trends](large_data_trends.png)

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

