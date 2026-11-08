# Large Dataset Analysis Guide

MannKS handles large data sets (tested up to 1,000,000 observations) without running out of memory. This guide explains what changes as the data grow, what stays exact, and how to control the trade-off between speed and accuracy.

## What is exact and what is approximated

| Quantity | Small data (full mode) | Large data (fast mode) |
| :--- | :--- | :--- |
| Mann-Kendall test: `s`, `var_s`, `z`, `p`, `C`, `Tau` | Exact | **Exact**, also for censored data and in seasonal tests |
| Slope: `slope`, `lower_ci`, `upper_ci`, `sen_probability` | Exact (all pairs) | Estimated from a random sample of pairs |
| `approximation_error` | `None` | Approximate 95% half-width of the slope's sampling error |

The test uses an exact O(n log n) algorithm, so its result never depends on the mode. Only the slope and its interval use pair sampling.

## The three modes (`large_dataset_mode`)

| Mode | Chosen automatically when | What it does |
| :--- | :--- | :--- |
| `'full'` | n ≤ 5,000 | Uses every pair: exact slope and interval. |
| `'fast'` | 5,000 < n ≤ 50,000 | Samples `max_pairs` pairs (default 100,000) for the slope. |
| `'aggregate'` | n > 50,000 | Same calculation as `'fast'`, plus a warning recommending temporal aggregation. It does **not** aggregate by itself. |

The mode is chosen from the number of observations **after** any aggregation (`agg_method`/`agg_period`). Hourly data aggregated to daily medians, for example, is analysed in full mode if few enough days remain.

You can force a mode. `'full'` needs memory for all n(n−1)/2 slopes, so it is refused above n = 46,340 (more than 8 GB). MannKS then uses fast mode and says so in a warning.

## Performance

Measured on a single core (hourly datetime data, default settings):

| n | Mode | Uncensored | 30% censored (ATS) | Peak memory |
| ---: | :--- | ---: | ---: | ---: |
| 5,000 | full (exact) | 1.2 s | 1.5 s | 0.6–0.9 GB |
| 10,000 | `'full'` forced | 4 s | 6 s | 1.7–2.5 GB |
| 10,000 | fast (default) | 0.1 s | 0.1 s | 0.2 GB |
| 50,000 | fast | 0.13 s | 0.2 s | 0.2 GB |
| 1,000,000 | aggregate | 1.0 s | 1.2 s | 0.5 GB |

`seasonal_trend_test` has similar timings (1.3 s for 1,000,000 hourly values with monthly seasons). Some options are slower at large n:

* **Block bootstrap** (`autocorr_method='block_bootstrap'`, or `'auto'` when autocorrelation is detected) recomputes the test and the slope for every resample: about 0.05 s per resample at n = 50,000, i.e. about a minute for the default 1,000 resamples. Reduce `n_bootstrap` for exploration.
* **`segmented_trend_test`**: the breakpoint search (`piecewise-regression`) scales with n and takes about 20 s at n = 10,000 and over a minute at n = 30,000. A warning is issued above 20,000 observations; aggregate first.

## Accuracy of fast mode

In simulations with n = 6,000–8,000:

* **Slope.** The sampled slope differed from the exact one by less than about 0.2 of the exact confidence interval's half-width in most scenarios (normal, skewed, heavily tied, tied times, censored with ATS). The largest error was about 0.5 half-widths, for `sens_slope_method='lwp'` at 50% censoring: that method sets ambiguous slopes to exactly 0, and the median can sit next to that block of zeros.
* **Confidence interval.** The fast-mode intervals contained the true slope 94.7–96.3% of the time (nominal 95%), for uncensored and censored (30%, 60%) data. The bias was below 0.3% of the slope.
* **Test.** With no trend, the rejection rate at p < 0.05 was 5.0%.

`result.approximation_error` is a distribution-free 95% half-width for the slope's sampling error: half the distance between the sampled slopes ranked K/2 ∓ 0.98√K, where K is the number of sampled pairs. With ATS for censored data it is the range of slopes over which the sampled score stays within its 95% sampling band. In the simulations it covered the actual error 90–97% of the time.

Pair sampling uses a fixed seed by default, so results are reproducible. Pass `random_state` to vary it.

### Choosing `max_pairs`

The sampling error shrinks like 1/√`max_pairs`:

| `max_pairs` | Relative speed | Typical relative s.d. of the slope | Use case |
| :--- | :--- | :--- | :--- |
| 50,000 | fastest | ~0.4% | Exploratory analysis |
| 100,000 | fast | ~0.3% | **Default** |
| 500,000 | moderate | ~0.1% | High accuracy |
| all pairs (`'full'`) | slowest | exact | When memory allows (n up to ~20,000) |

## Seasonal data

The seasonal test always uses every observation: S and Var(S) are summed over the seasons and computed exactly. When the slope is estimated from sampled pairs, each season's share of the `max_pairs` budget is proportional to its number of within-season pairs, so seasons are weighted as in the full calculation.

`max_per_season` (optional) randomly subsamples seasons with more observations than this. It reduces the power of the test and is rarely needed; it is off by default.

## Aggregation strategies

For high-frequency data, aggregating to the time scale of interest is usually better than analysing every value: it reduces serial correlation and noise, and makes the analysis exact.

```python
# Hourly -> daily medians
result = trend_test(hourly_values, timestamps, agg_method='median', agg_period='day')

# Daily -> monthly, censored data
result = trend_test(daily_values, dates, agg_method='robust_median', agg_period='month')
```

## Examples

### Hourly sensor data
```python
import pandas as pd
from MannKS import trend_test

dates = pd.date_range('2010-01-01', '2020-01-01', freq='h')   # ~88,000 points
result = trend_test(measurements, dates, agg_method='median', agg_period='day',
                    slope_scaling='year', x_unit='µg/m³')
print(result.slope, result.slope_units, result.computation_mode)   # ~3,650 daily values: 'full'
```

### Fast mode with more pairs
```python
result = trend_test(values, dates, large_dataset_mode='fast', max_pairs=500_000,
                    slope_scaling='year')
print(f"{result.slope:.4f} ± {result.approximation_error:.4f} (sampling) "
      f"from {result.pairs_used:,} pairs")
```

## Validation

* The O(n log n) censored Mann-Kendall algorithm is tested against the pairwise LWP-TRENDS rules on thousands of random data sets (ties, several detection limits, right and mixed censoring, tied times): S, Var(S) and Tau agree exactly (`tests/test_large_data_robustness.py`).
* The memory-saving order-statistic, ATS and analysis-note code is tested against straightforward reference implementations, with bit-for-bit identical results.
* Fast-mode accuracy, interval coverage and the error estimate were measured by simulation, as summarised above.
