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
import os

# 1. Generate Synthetic Data with Right-Censored Values
# We create a dataset with an underlying increasing trend but containing right-censored data ('>').
# Right-censored data often represents values exceeding a measurement range (e.g., "Over Range").
t = np.arange(2010, 2020)
x_raw = [
    1.0, 1.2, '>1.5', 1.8, 2.0,
    '>2.2', 2.5, 2.8, 3.0, '>3.5'
]

# Pre-process the data
# We must convert the mixed list of numbers and strings into a format the package can use.
# `prepare_censored_data` returns a DataFrame with 'value', 'censored', and 'cen_type' columns.
df = mk.prepare_censored_data(x_raw)
x = df['value'].values
censored = df['censored'].values
cen_type = df['cen_type'].values

# Combine x, censored, and cen_type into the format expected by trend_test.
# Currently, trend_test expects 'x' to be numeric, and separate 'censored' and 'cen_type' arrays.
# OR, it can accept the dataframe directly if we pass it correctly.
# But for clarity here, we'll construct the input DataFrame explicitly as trend_test accepts it.

print("--- Original Data ---")
print(pd.DataFrame({'Year': t, 'Raw': x_raw}))

# 2. Method A: The 'Robust' Approach (Default)
# - Mann-Kendall: Uses statistical ranks. A value of '>1.5' is treated as strictly greater than 1.5.
#   It is considered "tied" or ambiguous when compared to 1.8, because >1.5 could be 1.6 or 100.
#   This is statistically conservative.
# - Sen's Slope: Uses 'nan' method. Slopes involving ambiguous pairs (e.g., >1.5 vs 1.8)
#   are set to NaN and excluded from the median calculation.
print("\\n\\n--- Method A: Robust (Default) ---")

# Determine absolute path for plot
plot_path_robust = os.path.join(os.path.dirname(__file__), 'plot_robust.png')

result_robust = mk.trend_test(
    df, t,
    mk_test_method='robust',       # Conservative ranking
    sens_slope_method='unbiased',  # Exclude ambiguous slopes
    plot_path=plot_path_robust
)
print(f"Trend: {result_robust.trend}")
print(f"S-statistic: {result_robust.s}")
print(f"Sen's Slope: {result_robust.slope:.4f}")


# 3. Method B: The 'LWP' Approach (Legacy/Emulation)
# - Mann-Kendall: Replicates the 'LWP-TRENDS' R script heuristic.
#   It replaces ALL right-censored values with a single value slightly larger than the
#   maximum uncensored value in the dataset.
#   In our data, max uncensored is 3.0. So '>1.5', '>2.2', and '>3.5' ALL become ~3.1.
#   This artificially inflates the early censored values (like >1.5 becoming 3.1).
# - Sen's Slope: Uses 'lwp' method. Ambiguous slopes are forced to 0.0 instead of being excluded.
print("\\n\\n--- Method B: LWP (Legacy) ---")

plot_path_lwp = os.path.join(os.path.dirname(__file__), 'plot_lwp.png')

result_lwp = mk.trend_test(
    df, t,
    mk_test_method='lwp',          # Aggressive substitution
    sens_slope_method='lwp',       # Force ambiguous slopes to 0
    lt_mult=0.5,                   # Standard multiplier for left-censored (not used here but good practice)
    plot_path=plot_path_lwp
)
print(f"Trend: {result_lwp.trend}")
print(f"S-statistic: {result_lwp.s}")
print(f"Sen's Slope: {result_lwp.slope:.4f}")
"""

# --- 2. Execute the Code and Capture Output ---
output_buffer = io.StringIO()

with contextlib.redirect_stdout(output_buffer):
    local_scope = {}
    # Ensure correct path resolution when executing string code.
    # We need to inject __file__ into the execution scope.
    exec_globals = globals().copy()
    exec_globals['__file__'] = __file__

    exec(example_code, exec_globals, local_scope)

captured_output = output_buffer.getvalue()

# --- 3. Generate the README.md ---
readme_content = f"""
# Example 6: Deep Dive into Censored Data Options

## The "Why": Robustness vs. Legacy Compatibility
When handling censored data (values like `<5` or `>100`), there isn't one single "correct" way to perform statistics. Different software packages make different assumptions.

This package offers two main philosophies:
1.  **`robust` (Default)**: Uses modern survival analysis techniques (like those in the NADA package). It treats censored values as ranges and acknowledges ambiguity. It is generally preferred for scientific rigor.
2.  **`lwp`**: Emulates the specific heuristics used in the "LWP-TRENDS" R script. This is useful if you are required to replicate legacy analyses or regulatory reports based on that specific tool.

This example demonstrates how these choices can significantly alter your results, especially with **right-censored data** (e.g., "Over Range").

## The "How": Code Walkthrough

We analyze a synthetic dataset with an increasing trend that includes "Over Range" (`>`) values.

### Step 1: Python Code
```python
{example_code.strip()}
```

### Step 2: Text Output
```text
{captured_output}
```

## Interpreting the Results

### Visual Comparison

| Method A: Robust (Default) | Method B: LWP (Legacy) |
| :---: | :---: |
| ![Robust Plot](plot_robust.png) | ![LWP Plot](plot_lwp.png) |

### Key Differences Observed

1.  **The S-Statistic (Trend Strength)**
    *   **Robust (S={local_scope['result_robust'].s})**: The test calculates the score by comparing ranks. Comparisons involving `>1.5` and `1.8` are treated as ambiguous (tied), so they don't contribute to the score. This effectively bases the trend on the clear, unambiguous signal in the data.
    *   **LWP (S={local_scope['result_lwp'].s})**: The algorithm forces *all* right-censored values (`>1.5`, `>2.2`, `>3.5`) to the same high number (approx `3.1`).
        *   Because the early `>1.5` is replaced by `3.1`, it becomes larger than many subsequent values (like `1.8`, `2.0`, `2.5`).
        *   This creates artificial "decreasing" pairs (e.g., `3.1` vs `1.8`) that penalize the score.
        *   As a result, the S-statistic is **lower** ({local_scope['result_lwp'].s} vs {local_scope['result_robust'].s}) than the robust method, underestimating the trend strength.

2.  **Sen's Slope (Magnitude)**
    *   **Robust**: Calculates slopes only for valid pairs. If a pair is ambiguous (e.g., `>1.5` vs `1.8`), it is excluded. This tends to preserve the true slope of the underlying data.
    *   **LWP**: Forces ambiguous slopes to `0.0`. If you have many censored values, this "dilutes" the median, pulling the slope estimate towards zero.

## Recommendation
*   **`mk_test_method='robust'`** (Default): Recommended. It avoids creating artificial trends or suppressing real ones based on substitution artifacts.
*   **`mk_test_method='lwp'`**: Use only for legacy replication.
"""

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'w') as f:
    f.write(readme_content)

print("Example 6 generated successfully.")
