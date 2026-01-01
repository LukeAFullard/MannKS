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

# 1. Generate Synthetic Data for Three Scenarios
# We create 20 data points for each scenario.
t = np.arange(20)
dates = pd.date_range(start='2020-01-01', periods=20, freq='ME')

# Scenario A: Weak/Likely Increasing Trend
# Slope = 0.04, Noise Sigma = 0.5. Signal-to-noise is low but detectable.
# Use seed 18 to get p ~ 0.055, which is > 0.05 (No Trend for Classical) but High Confidence for Continuous.
np.random.seed(18)
vals_increasing = t * 0.04 + np.random.normal(0, 0.5, 20) + 10

# Scenario B: Very Weak / Ambiguous Trend
# Slope = 0.015, Noise Sigma = 2.0.
# We use a specific seed (11) to produce a dataset with confidence ~0.59.
rng_ambiguous = np.random.RandomState(11)
vals_ambiguous = t * 0.015 + rng_ambiguous.normal(0, 2.0, 20) + 10

# Scenario C: Flat / Uncertain (No Trend)
# Slope = 0, Noise Sigma = 0.5.
rng_flat = np.random.RandomState(32)
vals_flat = rng_flat.normal(0, 0.5, 20) + 10

scenarios = {
    "Weak Increasing": vals_increasing,
    "Ambiguous Trend": vals_ambiguous,
    "Flat No Trend": vals_flat
}

# 2. Run Comparative Analysis
results_summary = []

for name, x in scenarios.items():
    print(f"\\n--- Analyzing: {name} ---")

    # Sanitize filename
    safe_name = name.replace(' ', '_').replace('/', '').lower()

    # Run with Continuous Confidence (Default)
    # This interprets direction based on probability, even if p > 0.05
    res_cont = mk.trend_test(x, t, alpha=0.05, continuous_confidence=True,
                             plot_path=f"{safe_name}_cont.png")

    # Run with Classical Testing
    # This falls back to 'No Trend' if p > 0.05
    res_class = mk.trend_test(x, t, alpha=0.05, continuous_confidence=False,
                              plot_path=f"{safe_name}_class.png")

    print(f"  P-value: {res_cont.p:.4f}")
    print(f"  Confidence (C): {res_cont.C:.4f}")
    print(f"  [Continuous] Classification: {res_cont.classification}")
    print(f"  [Classical ] Classification: {res_class.classification}")

"""

# --- 2. Execute the Code and Capture Output ---
output_buffer = io.StringIO()

with contextlib.redirect_stdout(output_buffer):
    local_scope = {}
    exec(example_code, globals(), local_scope)

captured_output = output_buffer.getvalue()

# --- 3. Generate the README.md ---
readme_content = f"""
# Example 27: Continuous Confidence vs Classical Testing

## The Concept

Traditionally, trend analysis relies on **hypothesis testing** with a strict significance threshold (e.g., $p < 0.05$). If the p-value is 0.06, the result is binary: "No Trend." This can be misleading for environmental monitoring, where a "likely" trend might still warrant attention.

**Continuous Confidence** (enabled by default in `MannKS`) changes the question from *"Is the trend non-zero?"* to *"How confident are we in the trend direction?"*.

*   **Classical Mode (`continuous_confidence=False`)**: Returns "No Trend" if statistical significance is not met.
*   **Continuous Mode (`continuous_confidence=True`)**: Returns the most probable direction (Increasing/Decreasing) along with a descriptive confidence level (e.g., "Likely Increasing", "As Likely as Not").

## The "How": Code Walkthrough

We simulate three scenarios, including one with an extremely weak ("ambiguous") trend where the signal is barely distinguishable from noise. This demonstrates how the method handles low-confidence situations.

### Step 1: Python Code
```python
{example_code.strip()}
```

### Step 2: Text Output
```text
{captured_output}
```

## Interpreting the Results

### Scenario A: Weak Increasing Trend
*   **Classical**: "No Trend" (because $p > 0.05$, typically). Here $p \\approx 0.0556$, which is just above the standard 0.05 threshold. So Classical says **No Trend**.
*   **Continuous**: "Highly Likely Increasing" (Confidence ~97%). This tells us there is a very high probability the trend is real, providing an early warning that the classical test missed.

### Scenario B: Ambiguous Trend
*   **Classical**: "No Trend".
*   **Continuous**: "As Likely as Not Increasing" (Confidence ~0.59). The confidence is very low, correctly indicating that while there is a slight upward tilt, the evidence is weak and the direction is uncertain. This is a much more nuanced result than simply "No Trend".

### Scenario C: Flat Data
*   **Classical**: "No Trend".
*   **Continuous**: "As Likely as Not Increasing" (Confidence ~0.54). The confidence drops to near 0.5 (50%), correctly indicating that the direction is essentially a coin flip.

## Visual Comparison

The plots generated show the same data and trend lines, but the interpretation in the title/header differs based on the mode.

### Weak Increasing Trend (Continuous)
![Increasing Cont](weak_increasing_cont.png)

### Ambiguous Trend (Continuous)
![Ambiguous Cont](ambiguous_trend_cont.png)

### Flat Trend (Continuous)
![Flat Cont](flat_no_trend_cont.png)

"""

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'w') as f:
    f.write(readme_content)

print("Example 27 generated successfully.")
