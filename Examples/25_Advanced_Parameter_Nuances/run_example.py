import os
import io
import contextlib
import numpy as np
import pandas as pd
import MannKS as mk
import warnings
import matplotlib.pyplot as plt

# --- 1. Define the Example Code as a String ---
example_code = """
import numpy as np
import pandas as pd
import MannKS as mk
import warnings

# === Topic 1: Kendall's Tau-a vs. Tau-b ===
# Kendall's Tau is a correlation coefficient.
# Tau-a: Basic calculation, does not account for ties.
# Tau-b: Adjusts for ties in values and time. It is generally preferred.
print("=== Topic 1: Kendall's Tau-a vs. Tau-b ===")
t = np.arange(10)
x = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4]) # Highly tied data
print(f"Data with ties: {x}")

res_a = mk.trend_test(x, t, tau_method='a')
print(f"Tau-a (No tie correction): {res_a.Tau:.4f}")

res_b = mk.trend_test(x, t, tau_method='b')
print(f"Tau-b (Tie corrected):     {res_b.Tau:.4f}")


# === Topic 2: Custom Trend Classification ===
# The package has a default classification system (Increasing, Likely Increasing, etc.).
# You can override this with your own probability thresholds using `classify_trend`.
print("\\n=== Topic 2: Custom Trend Classification ===")

# Create a custom mapping: {Confidence_Threshold: 'Label'}
# Note: Confidence (C) = 1 - p/2
my_map = {
    0.99: "Virtual Certainty",
    0.95: "High Confidence",
    0.90: "Likely",
    0.50: "Possible"
}

# Apply to previous result
# Note: classify_trend returns a STRING, not a namedtuple
custom_classification = mk.classify_trend(res_b, category_map=my_map)
print(f"Default Classification: {res_b.classification}")
print(f"Custom Classification:  {custom_classification}")


# === Topic 3: Minimum Sample Size Warning ===
# Running statistical tests on tiny datasets can be misleading.
# The `min_size` parameter sets the threshold for adding a caution note to the results.
print("\\n=== Topic 3: Minimum Sample Size Warning ===")
x_small = [10, 11, 12]
t_small = [1, 2, 3] # n=3

# Case A: Strict requirement (min_size=5)
# Since n=3 < 5, we expect a note.
res_strict = mk.trend_test(x_small, t_small, min_size=5)
print(f"n=3, min_size=5 -> Notes: {res_strict.analysis_notes}")

# Case B: Lax requirement (min_size=2)
# Since n=3 >= 2, we expect NO note about sample size.
res_lax = mk.trend_test(x_small, t_small, min_size=2)
# Filter out other potential notes (like CI warnings) to focus on sample size
size_notes = [n for n in res_lax.analysis_notes if 'sample size' in n]
print(f"n=3, min_size=2 -> Notes: {size_notes}")


# === Topic 4: Seasonal Trend Test with Numeric Time ===
# `seasonal_trend_test` is usually used with Datetime objects and 'month'.
# However, you can use numeric time if you manually specify the `period` (season length).
print("\\n=== Topic 4: Seasonal Trend Test with Numeric Time ===")

# Create data: 3 seasons per cycle (e.g., Morning, Noon, Evening), 8 cycles.
# Pattern: 10, 20, 30 repeats, plus a trend.
# Increasing cycles from 4 to 8 to avoid small-sample warnings for CI calculation.
n_cycles = 8
period = 3
t_num = np.arange(1, (n_cycles * period) + 1)
pattern = np.array([10, 20, 30])
trend_signal = np.repeat(np.arange(n_cycles), period) # Adds 0,0,0, 1,1,1, etc.
x_seas = np.tile(pattern, n_cycles) + trend_signal

print(f"Numeric Time (Head): {t_num[:6]}...")
print(f"Seasonal Data (Head): {x_seas[:6]}...")

# We MUST specify `period=3` because `t` is numeric.
# We also generate a plot here.
res_seas = mk.seasonal_trend_test(
    x_seas, t_num,
    period=period,
    plot_path='numeric_seasonal_plot.png'
)

print(f"Seasonal Trend: {res_seas.trend}")
print(f"Sen's Slope: {res_seas.slope:.4f}")
"""

# --- 2. Execute the Code and Capture Output ---
output_buffer = io.StringIO()

# Change to the script directory to ensure plots are saved there
script_dir = os.path.dirname(__file__)
original_dir = os.getcwd()
if script_dir:
    os.chdir(script_dir)

try:
    with contextlib.redirect_stdout(output_buffer):
        local_scope = {}
        exec(example_code, globals(), local_scope)
finally:
    # Always restore original directory
    os.chdir(original_dir)

captured_output = output_buffer.getvalue()

# --- 3. Generate the README.md ---
readme_content = f"""
# Example 25: Advanced Parameter Nuances

## The "Why": Fine-Tuning Your Analysis
While the default settings of `MannKS` cover 95% of use cases, advanced users sometimes need more control. This example covers four specific "power user" features:

1.  **Kendall's Tau Method**: Choosing between the standard and tie-corrected correlation coefficient.
2.  **Custom Classification**: Renaming trend confidence levels to match your organization's terminology (e.g., IPCC standards).
3.  **Minimum Sample Size**: Enforcing quality checks for data that is too sparse.
4.  **Numeric Seasonality**: Performing seasonal analysis on abstract time steps (e.g., "Cycles") rather than calendar dates.

## The "How": Code Walkthrough

### Step 1: Python Code
```python
{example_code.strip()}
```

### Step 2: Text Output
```text
{captured_output}
```

### Step 3: Visual Output
The numeric seasonal analysis generated this plot. Note that the x-axis is numeric integers, not dates.

![Numeric Seasonal Plot](numeric_seasonal_plot.png)

## Detailed Explanation

### 1. Tau-a vs. Tau-b
*   **Tau-a** is the raw difference between concordant and discordant pairs divided by the total number of pairs. It does not penalize for ties.
*   **Tau-b** (the default) accounts for ties in the denominator. When data has many tied values (like our example `[1, 1, 1...]`), the effective number of comparable pairs decreases. This makes the remaining comparisons "count more," typically resulting in a higher Tau value.

### 2. Custom Classification
The `classify_trend` function is flexible. If your project requires specific terms (e.g., "Virtual Certainty" instead of "Increasing"), you can pass a `category_map`.
*   The keys are confidence thresholds (0.0 to 1.0).
*   The values are the labels to use if the confidence exceeds that threshold.
*   The system checks from highest threshold to lowest.

### 3. Minimum Sample Size (`min_size`)
Statistical tests on $n=3$ points are mathematically possible but practically dangerous.
*   By setting `min_size=5`, you force the system to flag results from deceptively small datasets.
*   The function adds a note to the `analysis_notes` field (e.g., `'sample size (3) below minimum (5)'`) rather than crashing or issuing a Python warning. This allows you to filter these results programmatically later.

### 4. Numeric Seasonal Analysis
You don't always have dates. Sometimes your "season" is a rotation number, a machine cycle, or a biological phase.
*   By passing `t` as numbers and explicitly setting `period=3`, we told the system: "Every 3 time steps completes one full cycle."
*   The system then treats indices `0, 3, 6, 9` as Season 1, `1, 4, 7, 10` as Season 2, etc.
*   The plot confirms the increasing trend (Slope ~ 0.33, as we added +1 every 3 steps) despite the strong seasonal oscillation.
"""

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'w') as f:
    f.write(readme_content)

print("Example 25 generated successfully.")
