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

# 1. Generate Synthetic Data with Multiple Censoring Levels
# Imagine a dataset of chemical concentrations over 10 years.
# Over time, laboratory equipment improves, lowering the detection limit.
# However, sometimes high concentrations exceed the upper limit of the old equipment.

# 20 years of semi-annual data (40 points)
# Note: Using '6ME' (MonthEnd) requires pandas >= 2.1.0.
# If you are using an older version, use '6M'.
try:
    dates = pd.date_range(start='2000-01-01', periods=40, freq='6ME')
except ValueError:
    dates = pd.date_range(start='2000-01-01', periods=40, freq='6M')

# We manually construct a list of values to represent this scenario
# Note the mix of numeric values, "less than" censoring, and "greater than" censoring.
raw_data = [
    # Early years (2000-2005): High detection limits, limited range
    '> 100', '95', '80', '< 50', '60', '< 50', '45', '< 20', '35', '< 20',
    '30', '40', '< 20', '25', '< 20',
    # Middle years (2006-2012): Better equipment, lower detection limits
    '30', '< 10', '25', '< 10', '15', '12', '< 5', '18', '< 5', '10',
    '8', '12', '< 5',
    # Recent years (2013-2019): Modern equipment, very low detection limits
    '8', '4.5', '< 1', '2.1', '< 0.5', '3.0', '< 0.5', '1.2', '< 0.1', '0.5',
    '< 0.1', '0.2'
]

# Ensure lengths match
if len(raw_data) < len(dates):
    # Pad with some low values if needed
    raw_data.extend(['< 0.1'] * (len(dates) - len(raw_data)))
elif len(raw_data) > len(dates):
    raw_data = raw_data[:len(dates)]

df = pd.DataFrame({'Date': dates, 'Concentration': raw_data})

print("--- Raw Data Sample ---")
print(df.head())
print("...")
print(df.tail())

# 2. Preprocess the Data
# The package cannot read strings like '< 50' directly.
# We must use `prepare_censored_data` to parse them into numeric values and flags.
print("\\nPreprocessing data...")
processed_data = mk.prepare_censored_data(df['Concentration'])

# Inspect the processed DataFrame.
# Notice 'value' contains the number, 'censored' is boolean, and 'cen_type' is '<' or '>'.
print("\\n--- Processed Data ---")
print(processed_data.head())

# 3. Run the Trend Test
# We pass the PROCESSED DataFrame as `x`.
# The function detects it's a DataFrame and uses the 'value', 'censored', and 'cen_type' columns.
# We also use `slope_scaling='year'` to get the result in "units per year".
print("\\nRunning Mann-Kendall Trend Test...")
result = mk.trend_test(
    x=processed_data,
    t=df['Date'],
    slope_scaling='year',
    plot_path='multi_censor_trend.png'
)

# 4. Inspect the Results
print("\\n--- Trend Test Results ---")
print(f"Basic Trend: {result.trend}")
print(f"Classification: {result.classification} (Confidence: {result.C:.1%})")
print(f"Kendall's S: {result.s}")
print(f"p-value: {result.p:.4f}")
print(f"Sen's Slope: {result.slope:.4f} {result.slope_units}")
print(f"Confidence Interval: [{result.lower_ci:.4f}, {result.upper_ci:.4f}]")
print(f"Proportion Censored: {result.prop_censored:.1%}")
print(f"Number of Censor Levels: {result.n_censor_levels}")
"""

# --- 2. Execute the Code and Capture Output ---
output_buffer = io.StringIO()

with contextlib.redirect_stdout(output_buffer):
    local_scope = {}

    # We need to handle plot generation carefully.
    # The trend_test inside the exec block will generate 'multi_censor_trend.png' in the CURRENT directory.
    # We want it to be in the same directory as this script.

    # We change the working directory to the script's directory for execution
    original_cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    try:
        exec(example_code, globals(), local_scope)
    finally:
        os.chdir(original_cwd)

captured_output = output_buffer.getvalue()

# Extract result from local_scope to use in the dynamic explanation
result = local_scope.get('result')

# Define dynamic text for the README
classification_text = f"**Classification ({result.classification})**: Despite the messy data, the test identified a {result.classification.lower()} trend."
kendall_s_text = f"**Kendall's S ({result.s:.1f})**: A strongly negative score indicates that as time goes on, values tend to get lower." if result.s < 0 else f"**Kendall's S ({result.s:.1f})**: A positive score indicates that as time goes on, values tend to get higher."
prop_censored_text = f"**Proportion Censored ({result.prop_censored:.1%})**: A significant portion of the data is censored. Standard regression would fail here."
censor_levels_text = f"**Number of Censor Levels ({result.n_censor_levels})**: We successfully handled {result.n_censor_levels} different thresholds."

# --- 3. Generate the README.md ---
readme_content = f"""
# Example 10: Handling Data with Multiple Censoring Levels

## The "Why": Real-World Data is Messy
In long-term environmental monitoring, measurement technology often evolves.
*   **1990s:** You might report values as `< 50`.
*   **2000s:** Better equipment allows reporting `< 10`.
*   **Today:** Precision instruments report `< 0.1`.

This creates a dataset with **multiple censoring levels**. A naive analysis might treat `< 50` as `25` (half) and `< 0.1` as `0.05`. This can artificially create a "decreasing trend" just because the detection limit got lower, not because the pollutant concentration actually dropped.

The **Mann-Kendall test** handles this robustly. It doesn't use the values directly; it compares their *ranks*.
*   Is `30` less than `< 50`? **Uncertain** (could be 49, could be 0).
*   Is `< 0.1` less than `< 50`? **Yes**, definitely.
*   Is `> 100` greater than `95`? **Yes**.

This example shows how the package handles a dataset containing a mix of high detection limits, low detection limits, and even right-censored ("greater than") data.

## The "How": Code Walkthrough

We simulate a 20-year dataset where the pollution levels are decreasing, but the detection limits are also changing.

### Step 1: Python Code
```python
{example_code.strip()}
```

### Step 2: Text Output
```text
{captured_output}
```

## Interpreting the Results

### 1. Statistical Results
*   {classification_text}
*   {kendall_s_text}
*   {prop_censored_text}
*   {censor_levels_text}

### 2. Visual Results (`multi_censor_trend.png`)
The plot visualizes this complexity:

![Trend Plot](multi_censor_trend.png)

*   **Red Triangles (Left-Censored)**: These point *down*, indicating the value is somewhere below that mark (e.g., `< 50`). Notice how the "ceiling" of these triangles drops over time as technology improves.
*   **Blue Triangles (Right-Censored)**: These point *up*, indicating values above a limit (e.g., `> 100`).
*   **Blue Dots**: The actual measured values.
*   **Trend Line**: The solid line shows the estimated rate of decrease, pivoting through the "middle" of this complex cloud of data.

## Key Takeaway
The `MannKS` package is designed for this reality. You don't need to throw away censored data or use arbitrary substitution rules (like "replace with 1/2 DL").
1.  Use `mk.prepare_censored_data()` to parse your strings.
2.  Pass the result to `mk.trend_test()`.
3.  Trust the non-parametric statistics to handle the ambiguity correctly.
"""

with open(os.path.join(script_dir, 'README.md'), 'w') as f:
    f.write(readme_content)

print("Example 10 generated successfully.")
