# Large Dataset Analysis Guide

## Understanding the Three Modes

### 1. Full Mode (Exact)
**When:** n ≤ 5,000 (automatic) or `large_dataset_mode='full'`

**What it does:**
- Calculates ALL n×(n-1)/2 pairwise slopes
- Exact median, exact confidence intervals
- No approximation error

**Use when:**
- Dataset is small enough
- You need exact results for publication
- Computational time is not a concern

### 2. Fast Mode (Hybrid Optimized)
**When:** 5,000 < n ≤ 50,000 (automatic)

**What it does:**
- **MK Score:** Uses an optimized $O(N \log N)$ algorithm for exact calculation (uncensored data only).
- **Sen's Slope:** Samples random pairs (default: 100,000) to estimate the median slope.
- Maintains statistical validity with massive performance gains.

**Accuracy:**
- **MK test (S, Var(S), p):** Exact (no approximation), for censored data too.
- **Slope:** With 100,000 sampled pairs the relative standard deviation of the
  estimate around the exact value was about 0.3% in simulations (n = 6000), with
  no detectable bias.
- Pair sampling uses a fixed seed by default, so results are reproducible; pass
  `random_state` to vary it.

**Use when:**
- Dataset is medium-large
- You need fast results
- Small approximation error is acceptable

### 3. Aggregate Mode (Recommended for Very Large Data)
**When:** n > 50,000 (automatic) or `large_dataset_mode='aggregate'`

**What it does:**
- The calculations are the same as in fast mode; MannKS issues a warning
  recommending temporal aggregation.
- Aggregation is **not** applied automatically: aggregate yourself with
  `agg_method` and `agg_period` (see below), which reduces the data before
  the full or fast calculations are applied.

**Use when:**
- Dataset is very large
- High-frequency data with long-term trend
- Reducing noise is beneficial

## Statistical Theory

### Why Random Pair Sampling Works

Sen's slope is the **median** of all pairwise slopes. By the Central Limit Theorem:

- The sample median of K uniformly sampled pairwise slopes estimates the median of
  all pairwise slopes, with standard error of order IQR / √K
- K = number of sampled pairs (100,000 by default)
- In simulation (n = 6,000) the relative standard deviation was about 0.3%, with
  no detectable bias

### Seasonal Data Special Handling

Seasonal tests require **stratified sampling**:

1. Group observations by season
2. For very large data (n > 10,000 in fast mode), sample up to `max_per_season` observations from each season
3. When the slope is estimated from sampled pairs, each season's share of the `max_pairs` budget is proportional to its number of within-season pairs, so the pooled slopes weight seasons exactly as the full calculation does (S = Σ Sᵢ is always computed exactly on the data used)

## Practical Guidelines

### Choosing max_pairs

The sampling error of the slope shrinks like 1/√max_pairs. Indicative relative
standard deviations (measured at 100,000 pairs for n = 6,000 and scaled; the exact
value depends on the spread of the pairwise slopes):

| max_pairs | Speed | Typical relative s.d. | Use Case |
|-----------|-------|----------|----------|
| 50,000 | Very Fast | ~0.4% | Exploratory analysis |
| 100,000 | Fast | ~0.3% | **Default, balanced** |
| 500,000 | Medium | ~0.1% | High accuracy needs |
| all pairs (`large_dataset_mode='full'`) | Slowest | exact | Publication-quality, if memory allows |

`result.approximation_error` reports an approximate 95% half-width for the sampled
slope (1.96 × IQR(slopes) / √pairs).

### Aggregation Strategies

For n > 50,000:

**High-frequency data (e.g., hourly):**
```python
# Aggregate to daily
result = trend_test(
    hourly_data, timestamps,
    agg_method='median',
    agg_period='day'
)
```

**Daily data:**
```python
# Aggregate to monthly
result = trend_test(
    daily_data, dates,
    agg_method='robust_median',  # Better for censored data
    agg_period='month'
)
```

## Examples

### Example 1: Hourly Sensor Data (87,600 points/year)
```python
import pandas as pd
from MannKS import trend_test

# 10 years of hourly data = 876,000 points
dates = pd.date_range('2010-01-01', '2020-01-01', freq='h')
measurements = generate_sensor_data()  # Your data here

# Aggregate to daily first (reduces to ~3650 points)
result = trend_test(
    measurements, dates,
    agg_method='median',
    agg_period='day',
    slope_scaling='year',
    x_unit='μg/m³'
)

print(f"Trend: {result.slope:.2f} {result.slope_units}")
print(f"Mode: {result.computation_mode}")
```

### Example 2: Medium Dataset with Fast Mode
```python
# 20,000 daily measurements
result = trend_test(
    data_20k, dates_20k,
    large_dataset_mode='fast',
    max_pairs=200000,  # Higher accuracy
    random_state=42,    # Reproducible
    slope_scaling='year'
)

print(f"Used {result.pairs_used:,} pairs")
print(f"Approx error: ±{result.approximation_error:.4f}")
```

### Example 3: Comparing Modes
```python
# Same dataset, different modes
result_full = trend_test(data, dates, large_dataset_mode='full')
result_fast = trend_test(data, dates, large_dataset_mode='fast', random_state=42)

print(f"Full slope: {result_full.slope:.6f}")
print(f"Fast slope: {result_fast.slope:.6f}")
print(f"Difference: {abs(result_full.slope - result_fast.slope):.6f}")
```

## Validation

- The fast O(N log N) Mann-Kendall path is tested to give exactly the same S and
  Var(S) as the pairwise calculation, including heavy ties and tied times
  (`tests/test_audit_v050.py`, `tests/test_regressions_v060.py`).
- `large_dataset_mode='full'` is tested to return the exact Sen's slope.
- Seasonal pair sampling is tested to weight seasons like the full calculation.
- The bias and spread of the sampled slope were measured by simulation
  (`validation/r_parity/monte_carlo.py`): no detectable bias, relative s.d. ≈ 0.26%
  with 100,000 pairs (n = 6,000).
