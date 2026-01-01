# A Guide to `check_seasonality` and the Importance of Aggregation

The `check_seasonality` function is a crucial first step in a robust trend analysis workflow. Its primary purpose is to determine if your data exhibits a statistically significant seasonal pattern. This is a key justification for using a seasonal trend test (like `seasonal_trend_test`) over a simpler, non-seasonal one.

A critical aspect of this check is ensuring **analytical consistency**. If you plan to aggregate your data for the final trend test, you **must** perform the seasonality check on the same aggregated data. Checking for seasonality on raw data and then testing for trends on aggregated data can lead to contradictory or misleading conclusions.

This guide explains the function's parameters, with a special focus on how to use its aggregation features to maintain a consistent and reliable analysis.

### Example: Consistent Aggregation

The following example shows how to perform the seasonality check on both raw and aggregated data.

```python
import numpy as np
import pandas as pd
from MannKS import check_seasonality

# --- 1. Synthetic Data Generation ---
# Create a dataset with high-frequency (daily) data over several years.
# This simulates a scenario where multiple data points exist for each month.
dates = pd.to_datetime(pd.date_range(start='2020-01-01', end='2022-12-31', freq='D'))

# Introduce a strong seasonal pattern (high in summer, low in winter)
seasonal_signal = 15 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
noise = np.random.normal(0, 2, len(dates))
values = 50 + seasonal_signal + noise

# --- 2. Seasonality Check on Raw Data ---
# Here, we check the raw daily data for monthly seasonality.
# We expect a strong seasonal signal.
print("--- Seasonality Check on Raw Daily Data ---")
result_raw = check_seasonality(
    x=values,
    t=dates,
    period=12,
    season_type='month'
)
print(f"Is seasonal? {result_raw.is_seasonal}")
print(f"P-value: {result_raw.p_value:.4f}\n")

# --- 3. Seasonality Check on Aggregated Data ---
# Now, we aggregate the daily data into monthly medians.
# This is the correct approach if you plan to run seasonal_trend_test
# on monthly aggregated data.
print("--- Seasonality Check on Monthly Aggregated Data ---")
result_agg = check_seasonality(
    x=values,
    t=dates,
    period=12,
    season_type='month',
    agg_method='median',  # Aggregate to the monthly median
    agg_period='month'    # Define the aggregation window as 'month'
)
print(f"Is seasonal? {result_agg.is_seasonal}")
print(f"P-value: {result_agg.p_value:.4f}")
```

### Output

```
--- Seasonality Check on Raw Daily Data ---
Is seasonal? True
P-value: 0.0000

--- Seasonality Check on Monthly Aggregated Data ---
Is seasonal? True
P-value: 0.0003
```

### Interpretation of the Results

Both checks correctly identify the strong seasonal pattern in the data, as indicated by the `True` result and the very low p-values.

The key takeaway is the **process**. The first check confirms seasonality in the high-frequency raw data. The second check demonstrates the correct workflow: applying the same monthly aggregation to the seasonality check that you would use for the final trend test. This ensures your analytical steps are consistent and your conclusions are sound. Failure to aggregate consistently could, in more marginal cases, lead you to incorrectly assume seasonality does not exist in your aggregated dataset, or vice-versa.

### Parameter Reference

#### `x` and `t`
-   **Description:** The data (`x`) and time (`t`) vectors.
-   **Usefulness:** Just like in `trend_test`, `x` can be a numpy array or a DataFrame from `prepare_censored_data`. `t` can be numeric or datetime.

#### `alpha`
-   **Type:** `float`, **Default:** `0.05`
-   **Description:** The significance level for the Kruskal-Wallis test.
-   **Usefulness:** If the p-value is below this threshold, `is_seasonal` returns `True`. Adjust this if you need stricter (0.01) or looser (0.10) evidence of seasonality.

#### `period`
-   **Type:** `int`, **Default:** `12`
-   **Description:** The number of seasons in a full cycle.
-   **Usefulness:** Essential for numeric time data. For example, if `t` is in years, `period=4` implies quarterly data. Ignored if `season_type` is used with datetime data (except for 'month' where it defaults to 12).

#### `season_type`
-   **Type:** `str`, **Default:** `'month'`
-   **Description:** Defines how to extract seasons from datetime objects. Options: `'month'`, `'quarter'`, `'day_of_week'`, etc.
-   **Usefulness:** Allows you to test for different types of cycles (e.g., weekly cycles vs. annual cycles) without manual data manipulation.

#### `agg_method`
-   **Type:** `str`, **Default:** `'none'`
-   **Description:** The method for temporal aggregation.
-   **Benefits:** Ensures consistency with your trend test. If you plan to run `seasonal_trend_test(..., agg_method='median')`, you **must** set this to `'median'` here as well.
-   **Options:** `'none'`, `'median'`, `'robust_median'`, `'lwp'`, `'middle'`, `'middle_lwp'`.

#### `agg_period`
-   **Type:** `str`, **Default:** `None`
-   **Description:** The time window for aggregation (e.g., `'year'`, `'month'`).
-   **Limitations:** This is **required** if `agg_method` is not `'none'`. It tells the function how to group the data (e.g., "group by month") before calculating the aggregate value (e.g., "median").

#### `hicensor`
-   **Type:** `bool` or `float`, **Default:** `False`
-   **Description:** Activates the high-censor rule.
-   **Usefulness:** If your data has varying detection limits, this ensures the seasonality check isn't biased by artifacts of the detection method. Consistent use across all your tests is key.
