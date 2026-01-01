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
import matplotlib.pyplot as plt
import os

def run_and_show_notes(scenario_name, x, t, **kwargs):
    print(f"\\n--- Scenario: {scenario_name} ---")
    print(f"Time: {t}")
    # Handle mixed types for display
    print(f"Values: {x if isinstance(x, list) else x.tolist()}")

    # Pre-process if needed (for censored data)
    censored = None
    cen_type = None
    x_val = x

    # Check if input looks like it has strings (censored data)
    if isinstance(x, list) and any(isinstance(v, str) for v in x):
        df = mk.prepare_censored_data(x)
        x_val = df['value'].values
        censored = df['censored'].values
        cen_type = df['cen_type'].values
        print("(Data processed as censored)")

    # Run the test
    if censored is not None:
        # If we have prepared censored data, we pass the DataFrame so trend_test handles it
        result = mk.trend_test(df, t, **kwargs)
    else:
        result = mk.trend_test(x_val, t, **kwargs)

    # Display Results
    print(f"Trend: {result.trend}")
    print(f"Sen's Slope: {result.slope:.4f}")
    if result.analysis_notes:
        print("Analysis Notes:")
        for note in result.analysis_notes:
            print(f"  - {note}")
    else:
        print("Analysis Notes: None (Data looks good!)")

    return x_val, t

# Setup for plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Visualizing Data Scenarios that Trigger Analysis Notes", fontsize=14)

# 1. Scenario: Insufficient Data
t_small = [2000, 2001, 2002, 2003]
x_small = [1.0, 2.0, 1.5, 2.5]
x_val, t_val = run_and_show_notes("Insufficient Data", x_small, t_small)

axes[0].plot(t_val, x_val, 'o-', color='tab:blue')
axes[0].set_title("Insufficient Data\\n(n < 5)", fontsize=12)
axes[0].set_xticks(t_val)
axes[0].grid(True, linestyle='--', alpha=0.6)

# 2. Scenario: Long Run of Single Value
t_ties = np.arange(2000, 2010)
x_ties = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0])
x_val, t_val = run_and_show_notes("Long Run of Single Value", x_ties, t_ties)

axes[1].plot(t_val, x_val, 'o-', color='tab:orange')
axes[1].set_title("Long Run of Single Value\\n(Flat line dominates)", fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.6)

# 3. Scenario: Sen Slope Influenced by Censored Values
t_cen = [1, 2, 3]
x_cen = ['<1', 2, 3]
x_val, t_val = run_and_show_notes("Sen Slope Influenced by Censored Values", x_cen, t_cen)

# For plotting, convert censored string '<1' to numeric for display
# Note: In real analysis, be careful how you visualize censored data.
# Here we use the pre-processed values (where <1 became 1.0, but marked as censored)
# We will color the censored point differently.
censored_indices = [0] # We know the first point is censored
non_censored_indices = [1, 2]

axes[2].plot(t_val, x_val, '-', color='gray', alpha=0.5)
axes[2].scatter([t_val[i] for i in non_censored_indices], [x_val[i] for i in non_censored_indices],
                color='tab:green', label='Observed')
axes[2].scatter([t_val[i] for i in censored_indices], [x_val[i] for i in censored_indices],
                color='red', marker='v', label='Censored (<1)')
axes[2].set_title("Censored Influence\\n(Slope depends on substitute)", fontsize=12)
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

# Save plot to current directory (which we will handle via wrapping script context or explicit path)
# Here we use __file__ based path for robustness if run directly
plot_path = os.path.join(os.path.dirname(__file__), 'scenarios_plot.png')
plt.savefig(plot_path)
print(f"\\nPlot saved to 'scenarios_plot.png'")
"""

# --- 2. Execute the Code and Capture Output ---
output_buffer = io.StringIO()

# Determine the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

with contextlib.redirect_stdout(output_buffer):
    local_scope = {}
    exec_globals = globals().copy()
    exec_globals['__file__'] = __file__
    exec(example_code, exec_globals, local_scope)

captured_output = output_buffer.getvalue()

# --- 3. Generate the README.md ---
readme_content = f"""
# Example 17: Interpreting Analysis Notes

## The "Why": Trusting Your Results
Statistical tests like Mann-Kendall are powerful, but they have assumptions. If your data violates these assumptions—like having too few points, too many ties, or heavy censoring—the results might be misleading.

The `MannKS` package includes a built-in "Quality Assurance" system. Every time you run a test, it checks your data against several rules (inspired by the LWP-TRENDS R script) and returns **Analysis Notes** if potential issues are found.

This example teaches you how to interpret these warnings and visualizes the data patterns that trigger them.

## The "How": Scenarios and Visualizations

### Step 1: Python Code
```python
{example_code.strip()}
```

### Step 2: Output and Interpretation
```text
{captured_output}
```

### Step 3: Visualizing the Issues
The code above generates this plot, illustrating what "bad" data often looks like:

![Scenarios Plot](scenarios_plot.png)

1.  **Insufficient Data (Left):** With only 4 points, the confidence interval is massive, and statistical power is non-existent.
2.  **Long Run (Middle):** The flat line for the first 6 years means most pairwise comparisons yield a "tie" (slope = 0), reducing the test's ability to detect the subsequent rise.
3.  **Censored Influence (Right):** The slope between the first point (censored) and the others depends entirely on what value we substitute for `<1`. The warning alerts you to this dependency.

## Detailed Explanations of Notes

### 1. "Long run of single value"
*   **Trigger:** A single value repeats for a large portion of the time series (e.g., > 50% or 75% of the data, depending on configuration).
*   **Implication:** Mann-Kendall relies on *changes* in rank. If nothing changes for most of the record, the test loses power and sensitivity.
*   **Action:** Check your data. Is the instrument stuck? Is this a detection limit artifact (e.g., many "0"s that should be censored)? If it's real data (e.g., a regulated level that rarely changes), be cautious about interpreting "No Trend."

### 2. "Sen slope influenced by left/right-censored values"
*   **Trigger:** The Sen's Slope is calculated as the *median* of all possible slopes between pairs of points. If the median pair involves a censored value (e.g., `<1` vs `5`), the slope calculation requires substituting a number for `<1`.
*   **Implication:** The package substitutes censored values using a multiplier (default `0.5 * limit`). This means your calculated slope magnitude (e.g., "0.3 units/year") is directly dependent on that arbitrary multiplier.
*   **Action:** The *direction* (increasing/decreasing) is usually robust, but treat the exact *magnitude* of the slope with caution. Try changing the `lt_mult` parameter (see Example 12) to see if the slope changes significantly.

### 3. "< 5 Non-censored values" / "< 3 unique values"
*   **Trigger:** The dataset is too small.
*   **Implication:** Statistical tests need a minimum sample size to be reliable. With only 3 or 4 points, it's almost impossible to distinguish a real trend from random chance.
*   **Action:** Collect more data. Results from such small datasets should be considered anecdotal at best.

## Key Takeaway
Always check `result.analysis_notes`! An empty list `[]` or a `None` result is the goal. If notes are present, they don't necessarily mean the result is wrong, but they are "yellow flags" indicating you should review your data and interpret the findings with extra care.
"""

with open(os.path.join(script_dir, 'README.md'), 'w') as f:
    f.write(readme_content)

print("Example 17 generated successfully.")
