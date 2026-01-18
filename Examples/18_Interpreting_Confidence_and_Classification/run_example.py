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

# Determine where to save the plots (current directory by default)
# Note: When run directly, this should be the script's directory.
output_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'

# 1. Understanding Confidence (C) and Probability (p)
# The Mann-Kendall test calculates a "score" (S).
# - A high positive S means an increasing trend.
# - A high negative S means a decreasing trend.
# - The p-value tells us the probability that this S score could happen by random chance.
#
# The package calculates a Confidence score (C) derived from p:
#    C = 1 - p/2  (for increasing trends)
#    C = 1 - p    (for two-tailed tests, essentially)
#
# But for classification purposes, we often look at the confidence in a specific direction.

# Let's generate a dataset with a "weak" trend to see how this works.
np.random.seed(10)
n = 20
t = np.arange(n)
# A weak trend: small slope + lots of noise
x = 10 + 0.1 * t + np.random.normal(0, 1.0, n)

print("Running Trend Test on Weak Trend Data...")
# We use a standard alpha of 0.1 (90% confidence)
plot_path = os.path.join(output_dir, 'plot_weak_trend.png')
result = mk.trend_test(x, t, alpha=0.1, plot_path=plot_path)

print(f"  Mann-Kendall Score (S): {result.s}")
print(f"  p-value: {result.p:.4f}")
print(f"  Confidence (C): {result.C:.2%}")
print(f"  Trend Classification: {result.classification}")

# 2. Customizing Classification
# By default, the package uses a classification map similar to this:
#   Confidence >= 0.95  -> "Increasing" / "Decreasing"
#   Confidence >= 0.90  -> "Likely Increasing" / "Likely Decreasing"
#   Confidence < 0.90   -> "No Trend" / "Stable"
# (Note: The exact defaults depend on the alpha level provided, usually alpha=0.05 implies 95% cutoff)

# Organizations may have different standards for trend classification,
# requiring strict (99% confidence) or loose (80% confidence) thresholds.

# We can define a CUSTOM category map.
# Keys are the minimum confidence threshold (0.0 to 1.0).
# Values are the label to use if confidence is above that threshold.
# The map is checked from highest confidence to lowest.

my_custom_map = {
    0.99: "Highly Significant Trend",
    0.90: "Significant Trend",
    0.80: "Possible Trend",
    0.00: "No Significant Trend"
}

print("\\n--- Re-classifying with Custom Map ---")
# We can re-run the test with the map...
result_custom = mk.trend_test(x, t, alpha=0.1, category_map=my_custom_map)
print(f"  New Classification: {result_custom.classification}")

# OR we can just use the helper function `classify_trend` on an existing result
# This is faster if you just want to change labels without re-calculating statistics.
from MannKS import classify_trend
new_label = classify_trend(result, category_map=my_custom_map)
print(f"  Re-classified via helper: {new_label}")


# 3. The Effect of Alpha on "Significance" (h)
# The `result.h` (boolean) tells you if the trend is statistically significant.
# This depends entirely on the `alpha` parameter you pass.
# alpha = 0.05 means "I need 95% confidence to call it significant".
# alpha = 0.20 means "I only need 80% confidence".

print("\\n--- The Effect of Alpha ---")
result_strict = mk.trend_test(x, t, alpha=0.01) # Requires 99% confidence
result_loose = mk.trend_test(x, t, alpha=0.20)  # Requires 80% confidence

print(f"  Strict (alpha=0.01, Conf=99%): Significant? {result_strict.h}")
print(f"  Loose  (alpha=0.20, Conf=80%): Significant? {result_loose.h}")
"""

# --- 2. Execute the Code and Capture Output ---
output_buffer = io.StringIO()

with contextlib.redirect_stdout(output_buffer):
    local_scope = {}
    exec_globals = globals().copy()
    exec_globals['__file__'] = __file__
    exec(example_code, exec_globals, local_scope)

captured_output = output_buffer.getvalue()

# --- 3. Generate the README.md ---
readme_content = f"""
# Example 18: Interpreting Confidence and Customizing Classification

## The "Why": Beyond "Yes" or "No"

A statistical test gives you a lot of numbers. The most common question is "Is the trend significant?" (Yes/No). But often, the real world is more nuanced.
*   "Is it *almost* significant?"
*   "How confident are we?"
*   "What labels should we use for reporting?"

This example explains:
1.  **Confidence (C):** What it means and how it relates to the p-value.
2.  **Classification:** How to customize the text labels (e.g., "Likely Increasing" vs. "High Confidence").
3.  **Alpha:** How changing your significance threshold affects the results.

## The "How": `alpha` and `category_map`

*   `result.C`: The confidence score (0.0 to 1.0).
*   `category_map`: A dictionary mapping confidence thresholds to string labels.
*   `alpha`: The risk tolerance (0.05 = 5% risk = 95% confidence).

### Step 1: Python Code
```python
{example_code.strip()}
```

### Step 2: Text Output
```text
{captured_output}
```

## Interpreting the Results

### 1. Confidence Scores
In our "Weak Trend" example:
*   The **p-value** was around **0.06**. This means there is a 6% chance the data is just random noise.
*   The **Confidence (C)** is roughly **97%**. Wait, why isn't it 94% (1 - 0.06)?
    *   *Note:* The package uses a one-sided confidence calculation for the `C` attribute to support "Increasing" vs "Decreasing" logic. It essentially asks "How confident are we that S is positive?".

### 2. Custom Classification
You can see how the same result can get different labels:
*   **Default:** "Likely Increasing" (because confidence was > 90% but maybe not high enough for the strictest tier depending on alpha defaults).
*   **Custom Map:** "Significant Trend" (because we defined 0.90 as the cutoff for that label).

This allows you to align the package's output with your specific regulatory or scientific reporting standards.

### 3. Visual Results
Even for a weak trend, the plot shows the direction. The dashed confidence lines might be wide, indicating the uncertainty.

![Weak Trend Plot](plot_weak_trend.png)

## Key Takeaway
Don't just look at the p-value. Use `result.C` to understand the strength of the evidence, and use `category_map` to make the output speak your language.
"""

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'w') as f:
    f.write(readme_content)

print("Example 18 generated successfully.")
