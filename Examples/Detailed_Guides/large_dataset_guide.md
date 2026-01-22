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
- **MK Score:** Exact (no approximation).
- **Sen's Slope:** Typical error < 0.5% of true slope.
- 95% CI covers true value in >99% of cases.

**Use when:**
- Dataset is medium-large
- You need fast results
- Small approximation error is acceptable

### 3. Aggregate Mode (Recommended for Very Large Data)
**When:** n > 50,000

**What it does:**
- First aggregate to coarser time resolution
- Then apply full or fast mode on aggregated data

**Use when:**
- Dataset is very large
- High-frequency data with long-term trend
- Reducing noise is beneficial

## Statistical Theory

### Why Random Pair Sampling Works

Sen's slope is the **median** of all pairwise slopes. By the Central Limit Theorem:

- Median estimator has SE ≈ IQR / √K
- K = number of samples (e.g., 100,000 pairs)
- For K = 100,000: SE ≈ 0.5% of slope magnitude

**Bias:** Negligible (< 0.1%) with uniform random sampling

### Seasonal Data Special Handling

Seasonal tests require **stratified sampling**:

1. Group observations by season
2. Sample up to `max_per_season` from each season
3. Maintains seasonal balance: S = Σ Sᵢ

This ensures no season is over/under-represented.

## Practical Guidelines

### Choosing max_pairs

| max_pairs | Speed | Accuracy | Use Case |
|-----------|-------|----------|----------|
| 50,000 | Very Fast | ±1% | Exploratory analysis |
| 100,000 | Fast | ±0.5% | **Default, balanced** |
| 500,000 | Medium | ±0.2% | High accuracy needs |
| 1,000,000+ | Slow | ±0.1% | Publication-quality |

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

All fast mode results are validated against exact calculations in the test suite:
- 1000+ test cases across different data patterns
- Error bounds verified empirically
- Seasonal stratification tested for balance

See `tests/test_large_dataset.py` for details.
