# A Comprehensive Guide to `inspect_trend_data`

The `inspect_trend_data` function is a diagnostic tool designed to help you visualize your data *before* running a statistical test. It provides a set of plots that reveal the structure, distribution, and potential issues in your dataset, such as outliers, heavy censoring, or sampling gaps.

## Why use this tool?

Blindly running a statistical test can be dangerous. A "significant trend" might be driven by a few outliers or a change in detection limits rather than a true environmental change. This function helps you:
1.  **Spot Outliers:** Quickly see extreme values in the time series plot.
2.  **Assess Censoring:** Understand what proportion of your data is censored and how detection limits have changed over time.
3.  **Check Sampling Frequency:** See if your data collection has been consistent or if there are large gaps.

---

## Usage Example

```python
import pandas as pd
import numpy as np
from MannKS import inspect_trend_data, prepare_censored_data

# 1. Create a synthetic dataset
# We simulate data with a changing detection limit to show the tool's value
dates = pd.date_range(start='2010-01-01', end='2015-12-31', freq='ME')
values = np.random.lognormal(mean=1, sigma=0.5, size=len(dates))

# Create a list with some censoring
data_list = []
for i, (d, v) in enumerate(zip(dates, values)):
    # Early years have a higher detection limit (<5)
    if d.year < 2012 and v < 5:
        data_list.append("<5")
    # Later years have a better detection limit (<1)
    elif d.year >= 2012 and v < 1:
        data_list.append("<1")
    else:
        data_list.append(v)

# 2. Prepare the data (REQUIRED)
# The inspection tool works on the *processed* DataFrame, not raw lists.
df_processed = prepare_censored_data(data_list)
df_processed['t'] = dates # Add the time column

# 3. Run the inspection
# We save the plot to a file to view it.
print("--- Running Inspection ---")
inspect_trend_data(
    data=df_processed,
    time_col='t',
    plot=True,
    plot_path='inspection_plot.png'
)
print("Inspection complete. Plot saved to 'inspection_plot.png'.")
```

### Output
```
--- Running Inspection ---
Inspection complete. Plot saved to 'inspection_plot.png'.
```

## Parameter Reference

### `data`
-   **Type:** `pandas.DataFrame`
-   **Description:** The processed data frame. This **must** be the output from `prepare_censored_data`. It requires the columns `'value'`, `'censored'`, and `'cen_type'`.
-   **Important:** You must manually add your time column (e.g., `'t'`) to this DataFrame before passing it to the function, as `prepare_censored_data` only handles the value vector.

### `time_col`
-   **Type:** `str`
-   **Description:** The name of the column in `data` that contains the timestamps or numeric time values.

### `plot`
-   **Type:** `bool`, **Default:** `False`
-   **Description:** If `True`, the function generates a 2x2 grid of diagnostic plots.

### `plot_path`
-   **Type:** `str`, **Default:** `None`
-   **Description:** The file path where the plot should be saved (e.g., `'my_plot.png'`). Required if `plot=True` and you want to save the file.

### `trend_period` (optional)
-   **Type:** `int`
-   **Description:** The number of years to include in the analysis (counting back from `end_year`). Useful for subsetting the data.

### `end_year` (optional)
-   **Type:** `int`
-   **Description:** The last year of the analysis period. Defaults to the maximum year in the data.

## Understanding the Output Plots

The function generates a figure with four panels:

1.  **Time Series Plot (Top Left):** Shows values over time. Censored data is typically distinguished (e.g., open circles vs. filled circles). Look for trends, seasonality, and outliers.
2.  **Value Matrix (Top Right):** A heat map or scatter plot of values vs. season (e.g., Month). This helps identify seasonal patterns or specific months with high/low values.
3.  **Censoring Matrix (Bottom Left):** Shows the detection limits over time. This is critical for spotting "step changes" in your data quality (e.g., a lab upgrade that lowered the detection limit from 10 to 1).
4.  **Sample Count Matrix (Bottom Right):** Shows the number of observations per season per year. This highlights sampling gaps (e.g., "We stopped sampling in Winter 2014").
