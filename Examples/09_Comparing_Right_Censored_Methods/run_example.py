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
# Scenario: A monitoring station measures a chemical concentration that is steadily increasing.
# However, the measurement instrument has an upper limit of 80.
# Once the concentration exceeds 80, it is recorded as '>80' (Right-Censored).
t = np.arange(2010, 2020)
# Data: Steady increase, then hitting the "ceiling"
raw_values = [10, 20, 30, 40, 50, 60, 80, '>80', '>80', '>80']

print("Raw Data:")
print(raw_values)

# 2. Pre-process the Data
# Since we have strings ('>80'), we must convert them to a DataFrame
# that the package can understand (numeric values + censored flags).
data = mk.prepare_censored_data(raw_values)

# 3. Method A: The Default "Robust" Method
# This method uses a standard rank-based approach (adapted from NADA for left-censoring).
# It is "conservative" regarding right-censored data.
print("\\n--- Running Test with Method: 'robust' (Default) ---")
result_robust = mk.trend_test(
    data,
    t,
    mk_test_method='robust',
    plot_path='robust_plot.png'
)

print(f"Trend: {result_robust.trend} ({result_robust.classification})")
print(f"S-statistic: {result_robust.s}")
print(f"p-value: {result_robust.p:.4f}")

# 4. Method B: The "LWP" Method
# This method uses a heuristic from the LWP-TRENDS R script.
# It replaces right-censored values with a number slightly larger than the maximum
# observed right-censored value (e.g., >80 becomes 80.1) and treats them as uncensored.
# This forces the test to recognize that these values are indeed "higher".
print("\\n--- Running Test with Method: 'lwp' ---")
result_lwp = mk.trend_test(
    data,
    t,
    mk_test_method='lwp',
    plot_path='lwp_plot.png'
)

print(f"Trend: {result_lwp.trend} ({result_lwp.classification})")
print(f"S-statistic: {result_lwp.s}")
print(f"p-value: {result_lwp.p:.4f}")
"""

# --- 2. Execute the Code and Capture Output ---
output_buffer = io.StringIO()

# Change CWD to the script's directory so the plot is saved there
script_dir = os.path.dirname(os.path.abspath(__file__))
original_cwd = os.getcwd()
os.chdir(script_dir)

try:
    with contextlib.redirect_stdout(output_buffer):
        local_scope = {}
        exec(example_code, globals(), local_scope)
finally:
    os.chdir(original_cwd)

captured_output = output_buffer.getvalue()

# --- 3. Generate the README.md ---
readme_content = f"""
# Example 9: Comparing Right-Censored Data Methods

## The Problem with "Upper Limits" (Right-Censoring)

In environmental monitoring, we usually deal with **Left-Censored** data (values below a detection limit, e.g., `<0.1`). However, sometimes we encounter **Right-Censored** data (values above an upper limit, e.g., `>80`).

This happens when:
*   An instrument becomes saturated.
*   A value exceeds the calibrated range.
*   "Too numerous to count" (e.g., bacteria).

Determining a trend when the highest values are unknown (just "greater than X") is tricky. This package offers two ways to handle this, controlled by the `mk_test_method` parameter.

## The Methods

### 1. `mk_test_method='robust'` (Default)
*   **How it works:** Uses non-parametric interval comparison logic where right-censored values `>X` are represented as intervals $(X, \infty)$.
*   **Behavior:** An interval $(X, \infty)$ is strictly greater than any observed value $\le X$ (contributing $+1$ to $S$). Overlapping right-censored intervals (e.g., two `>80` measurements) cannot be ordered and are treated as tied (contributing $0$ to $S$).
*   **Consequence:** Robust interval scoring accurately recognizes upward trends that hit a ceiling ($S = 42.0, p = 0.0002$) without requiring arbitrary numerical substitution.

### 2. `mk_test_method='lwp'`
*   **How it works:** Implements the heuristic from the LWP-TRENDS R script.
*   **Behavior:** It finds the maximum right-censored value in the dataset (e.g., 80) and replaces all instances of `>80` with a slightly higher numeric value (e.g., `80.1`), treating them as uncensored real numbers.
*   **Consequence:** All `>80` measurements are ranked above $\le 80$ values, and ties among the substituted `80.1` values score 0, producing $S = 42.0, p = 0.0002$.

## The "How": Code Walkthrough

We simulate a dataset that hits a ceiling of 80.

### Step 1: Python Code
```python
{example_code.strip()}
```

### Step 2: Text Output
```text
{captured_output}
```

## Interpreting the Results

### S-Statistic and Significance

*   **Robust Method (S = {local_scope['result_robust'].s}, p = {local_scope['result_robust'].p:.4f})**:
    *   The test correctly identifies that all measurements of `>80` are strictly greater than earlier measurements from 10 to 80.
    *   The three `>80` values are mutually tied (contributing 0 to the score), giving $S = 42.0$ (out of a theoretical maximum of 45 for 10 unique values).
    *   **Result**: Highly significant increasing trend ($p = {local_scope['result_robust'].p:.4f}$), confirming the physical reality that concentration was rising and then hit the sensor ceiling.

*   **LWP Method (S = {local_scope['result_lwp'].s}, p = {local_scope['result_lwp'].p:.4f})**:
    *   The method substitutes `>80` with `80.1`.
    *   The comparisons of $80.1$ against $10 \dots 80$ add positive score, while ties among $80.1$ contribute 0.
    *   **Result**: Identical score ($S = 42.0, p = {local_scope['result_lwp'].p:.4f}$) and identical trend classification.

### Visual Comparison

| Robust Method | LWP Method |
| :---: | :---: |
| ![Robust Plot](robust_plot.png) | ![LWP Plot](lwp_plot.png) |

*Note: The trend lines (Sen's Slope) look identical because `mk_test_method` only affects the significance test (p-value and S), not the slope calculation. The slope is determined by `sens_slope_method`.*

## Recommendation
*   Both methods correctly detect increasing trends that hit an instrument ceiling ($p = 0.0002$).
*   The default `'robust'` method is recommended for principled non-parametric interval scoring without arbitrary numerical constants.
*   Use `mk_test_method='lwp'` when exact parity with legacy LWP-TRENDS R workflows is required.
"""

with open(os.path.join(script_dir, 'README.md'), 'w') as f:
    f.write(readme_content)

print("Example 9 generated successfully.")
