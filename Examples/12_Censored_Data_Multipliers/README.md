
# Example 12: The Impact of Censored Data Multipliers

## The "Why": Tweaking the Unknown
When dealing with censored data (e.g., "< 5 mg/L"), we know the value is somewhere between 0 and 5, but we don't know exactly where.

For the **Mann-Kendall test** (significance), this usually doesn't matter much because it is rank-based.

However, for **Sen's Slope** (magnitude), we need actual numbers to calculate the slope between two points. If we have a point at `Time=0, Value=<5` and `Time=1, Value=6`, the calculated slope depends entirely on what value we substitute for `<5`.
*   If `<5` is treated as 0.5 (e.g., `lt_mult=0.1`): Slope = (6 - 0.5) / 1 = 5.5.
*   If `<5` is treated as 4.5 (e.g., `lt_mult=0.9`): Slope = (6 - 4.5) / 1 = 1.5.

The `lt_mult` (less-than multiplier) and `gt_mult` (greater-than multiplier) parameters allow you to control this assumption.
*   `lt_mult=0.5` (Default): Replaces `<X` with `0.5 * X`.
*   `gt_mult=1.1`: Replaces `>X` with `1.1 * X`.

## The Scenario
We use a small synthetic dataset designed to highlight this sensitivity: `['<10', 12.0, 14.0]`.
By changing `lt_mult`, we drastically change the slopes calculated between the first censored point and the subsequent data points, which shifts the median slope.

**Note:** This is an advanced feature. For most users, the default `0.5` is standard practice in environmental statistics (simulating a value halfway between 0 and the detection limit).

## The "How": Code Walkthrough

### Step 1: Python Code
```python
import numpy as np
import pandas as pd
import MannKS as mk
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter("ignore")

# 1. Generate Synthetic Censored Data
# We create a MINIMAL dataset to guarantee the multiplier affects the median slope.
# T=0: <10
# T=1: 12
# T=2: 14
#
# Slopes:
# (0,1): 12 - (10 * mult)
# (1,2): 14 - 12 = 2
# (0,2): (14 - (10 * mult)) / 2
t = np.arange(3)
x_raw = ['<10', 12.0, 14.0]

# Convert to DataFrame
data = mk.prepare_censored_data(x_raw)
print("Data Head:")
print(pd.DataFrame({'Time': t, 'Raw': x_raw, 'Value': data['value'], 'Censored': data['censored']}))

# 2. Run Trend Test with Default Multiplier (lt_mult=0.5)
# <10 becomes 5.
# Slopes: (12-5)=7, 2, (14-5)/2=4.5.
# Sorted: 2, 4.5, 7. Median = 4.5.
print("\n--- Test 1: Default Multiplier (0.5) ---")
res_default = mk.trend_test(
    data, t,
    lt_mult=0.5,
    mk_test_method='lwp',
    sens_slope_method='lwp'
)
print(f"Slope (lt_mult=0.5): {res_default.slope:.4f}")

# 3. Run Trend Test with High Multiplier (lt_mult=0.9)
# <10 becomes 9.
# Slopes: (12-9)=3, 2, (14-9)/2=2.5.
# Sorted: 2, 2.5, 3. Median = 2.5.
print("\n--- Test 2: High Multiplier (0.9) ---")
res_high = mk.trend_test(
    data, t,
    lt_mult=0.9,
    mk_test_method='lwp',
    sens_slope_method='lwp'
)
print(f"Slope (lt_mult=0.9): {res_high.slope:.4f}")

# 4. Run Trend Test with Low Multiplier (lt_mult=0.1)
# <10 becomes 1.
# Slopes: (12-1)=11, 2, (14-1)/2=6.5.
# Sorted: 2, 6.5, 11. Median = 6.5.
print("\n--- Test 3: Low Multiplier (0.1) ---")
res_low = mk.trend_test(
    data, t,
    lt_mult=0.1,
    mk_test_method='lwp',
    sens_slope_method='lwp'
)
print(f"Slope (lt_mult=0.1): {res_low.slope:.4f}")

# 5. Visualize the Difference
slopes = [res_default.slope, res_high.slope, res_low.slope]
labels = ['Default (0.5)', 'High (0.9)', 'Low (0.1)']

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, slopes, color=['blue', 'green', 'red'])
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Impact of Censored Value Multiplier on Sen's Slope")
plt.ylabel("Sen's Slope")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=12)

# Save to the script's directory
import os
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
plt.savefig(os.path.join(script_dir, 'multiplier_comparison.png'))
print("\nSaved 'multiplier_comparison.png'.")
```

### Step 2: Text Output
```text
Data Head:
   Time   Raw  Value  Censored
0     0   <10   10.0      True
1     1  12.0   12.0     False
2     2  14.0   14.0     False

--- Test 1: Default Multiplier (0.5) ---
Slope (lt_mult=0.5): 4.5000

--- Test 2: High Multiplier (0.9) ---
Slope (lt_mult=0.9): 2.5000

--- Test 3: Low Multiplier (0.1) ---
Slope (lt_mult=0.1): 6.5000

Saved 'multiplier_comparison.png'.

```

## Interpreting the Results

### 1. Sensitivity Analysis
*   **Default (0.5)**: The slope is **4.5000**.
*   **High (0.9)**: The censored value is treated as being close to 10 (9.0). This reduces the rise to the next point (12.0), flattening the slope to **2.5000**.
*   **Low (0.1)**: The censored value is treated as near zero (1.0). This creates a steep rise to the next point (12.0), increasing the slope to **6.5000**.

### 2. Visual Comparison (`multiplier_comparison.png`)
![Multiplier Comparison](multiplier_comparison.png)

This chart demonstrates that while the *direction* (increasing) is consistent, the *rate of change* you report can depend heavily on your assumptions about the censored data.

### Key Takeaway
Always report the multiplier used (`lt_mult`) if you deviate from the standard 0.5. If your dataset is heavily censored, run a sensitivity analysis like this to see if your conclusions about the trend's magnitude are robust.
