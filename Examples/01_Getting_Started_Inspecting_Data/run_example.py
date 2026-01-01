import os
import io
import contextlib
import numpy as np
import pandas as pd
import MannKS as mk
import matplotlib.pyplot as plt

# --- 1. Define the Example Code as a String ---
example_code = """
import numpy as np
import pandas as pd
import MannKS as mk

# 1. Generate Synthetic Data
# We create a 5-year monthly dataset (60 points) with some "messy" real-world features:
# - An underlying upward trend.
# - Some random noise.
# - Missing data (NaNs).
# - Censored data (values below detection limit).
np.random.seed(42)
n_years = 5
dates = pd.date_range(start='2020-01-01', periods=n_years*12, freq='ME')
t = np.arange(len(dates))
values = 0.1 * t + np.random.normal(0, 1, len(t)) + 10
censored_mask = values < 10.5
values_str = values.astype(str)
values_str[censored_mask] = '<' + np.round(values[censored_mask] + 0.5, 1).astype(str)
values_str[10:13] = np.nan
values_str[45] = np.nan

# 2. Pre-process the Data
# Raw environmental data often comes as strings (e.g., '< 0.5', '12.4').
# Standard statistical functions fail on these strings.
# The `prepare_censored_data` function is critical because it:
#   1. Parses the strings to identify censored values (detects '<').
#   2. Separates the data into a numeric 'value' column and a boolean 'censored' column.
#   3. Handles multiple detection limits automatically.
df = mk.prepare_censored_data(values_str)
df['date'] = dates

# 3. Inspect the Data
# Before running a trend test, we must verify the data is suitable.
# The `inspect_trend_data` function acts as a diagnostic tool.
# It checks for:
#   - Data Availability: Do we have enough data points?
#   - Time Structure: Is the data monthly? Quarterly? Irregular?
#   - Gaps: Are there long periods with no data?
#   - Censoring: What percentage of data is non-detect?
# We request `return_summary=True` to get the statistical table back.
print("Running Data Inspection...")
result = mk.inspect_trend_data(
    data=df,
    time_col='date',
    return_summary=True,
    plot=True,
    plot_path='inspection_plots.png'
)

# Print the availability summary
print("\\nData Availability Summary:")
print(result.summary.to_markdown(index=False))
"""

# --- 2. Execute the Code and Capture Output ---
output_buffer = io.StringIO()

with contextlib.redirect_stdout(output_buffer):
    local_scope = {}
    exec(example_code, globals(), local_scope)

captured_output = output_buffer.getvalue()

# --- 3. Generate the README.md ---
readme_content = f"""
# Example 1: Getting Started - Inspecting Your Data

## The "Why": Verify Before You Analyze
In environmental data analysis, datasets are rarely perfect. They often contain:
*   **Missing values (Gaps):** Sensors fail, samples get lost.
*   **Censored data:** Concentrations fall below laboratory detection limits (e.g., `< 0.5 mg/L`).
*   **Irregular sampling:** Samples might be taken daily in summer but monthly in winter.

Running a trend test blindly on such data can lead to misleading results. The `MannKS.inspect_trend_data` function is your "sanity check."

## The "How": Code Walkthrough

In this example, we generate a synthetic "messy" dataset and inspect it. We use `return_summary=True` to get a programmatic report on data availability across different potential time increments (monthly, quarterly, etc.).

### Step 1: Python Code
```python
{example_code.strip()}
```

### Step 2: Text Output
The function returns a summary DataFrame, which we printed:

```text
{captured_output}
```

## Interpreting the Results

### 1. Statistical Summary (Text Output)
The table above evaluates different time increments (monthly, quarterly, etc.) to see if they are suitable for analysis:
*   **`increment`**: The time unit being tested.
*   **`prop_year`**: Fraction of years that have at least one sample. High values (near 1.0) are good.
*   **`prop_incr_year`**: Fraction of expected periods (e.g., 12 months/year) that have data.
*   **`data_ok`**: A boolean flag suggesting if this increment is viable for seasonal analysis.
    *   For **monthly** analysis, we see high coverage, confirming our data is suitable despite the gaps.

### 2. Visual Diagnostics (Plots)
The function generated `inspection_plots.png`:

![Inspection Plots](inspection_plots.png)

*   **Top-Left (Time Series):** Visualizes the trend, gaps, and censored values (red dots).
*   **Top-Right (Value Matrix):** Heatmap of values (Row=Year, Col=Month). Useful for spotting seasonal blocks.
*   **Bottom-Left (Censoring Matrix):** Heatmap of censored data locations.
*   **Bottom-Right (Sample Count Matrix):** Heatmap of sampling frequency.

## Conclusion
We have confirmed our data is messy but sufficient for a **monthly** trend analysis. We are ready to proceed!
"""

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'w') as f:
    f.write(readme_content)

print("Example 1 generated successfully.")
