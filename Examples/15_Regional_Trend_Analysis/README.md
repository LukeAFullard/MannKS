
# Example 15: Regional Trend Analysis

## The "Why": The Big Picture
Sometimes, we don't just care if *one* specific lake is getting warmer; we want to know if the *entire region* is getting warmer.

Simply counting how many sites have a significant trend (e.g., "3 out of 5 sites are increasing") can be misleading if the sites are close to each other.
*   **Correlation Problem**: If it rains today at Site A, it probably rains at Site B nearby. Their data is not independent.
*   **False Confidence**: If you treat them as independent, you might overestimate the significance of a regional trend.

The `regional_test` function implements a robust method (Van Belle and Hughes, 1984) to aggregate results from multiple sites while **mathematically correcting for the correlation between them**.

## The Scenario
We simulate 5 sites in a region.
*   **Sites A, B, C**: Strong increasing trend.
*   **Site D**: Weak increasing trend.
*   **Site E**: Stable (no trend).
*   **Shared Signal**: All sites share a common random noise component (simulating regional weather), making them correlated.

## The "How": Code Walkthrough

### Step 1: Python Code
```python
import numpy as np
import pandas as pd
import MannKS as mk
import matplotlib.pyplot as plt

# 1. Generate Synthetic Data for Multiple Sites
# We simulate 5 sites.
# - Sites A, B, C: Strong increasing trend.
# - Site D: Weak increasing trend.
# - Site E: No trend / stable.
# All sites are somewhat correlated (e.g., they share the same weather patterns).

sites = ['Site_A', 'Site_B', 'Site_C', 'Site_D', 'Site_E']
trends = [0.5, 0.4, 0.6, 0.1, 0.0] # Slope per year

n_years = 10
t_years = np.arange(2010, 2020)
dates = pd.date_range(start='2010-01-01', periods=n_years, freq='YE')

# Shared "Regional" Signal (creates correlation)
np.random.seed(42)
shared_signal = np.random.normal(0, 1.0, n_years)

all_data = []

print("Generating data for 5 sites...")
for site, trend in zip(sites, trends):
    # Individual site noise
    local_noise = np.random.normal(0, 0.5, n_years)

    # Value = Base + Trend + Shared Signal + Local Noise
    values = 10 + (trend * (t_years - 2010)) + shared_signal + local_noise

    site_df = pd.DataFrame({
        'site': site,
        'date': dates,
        'value': values
    })
    all_data.append(site_df)

# Combine into one large DataFrame (Long format)
full_data = pd.concat(all_data, ignore_index=True)
print(full_data.head())


# 2. Run Individual Trend Tests
# We must run a trend test for each site first.
print("\n--- Running Individual Trend Tests ---")
results_list = []

for site in sites:
    site_data = full_data[full_data['site'] == site]

    # We use trend_test for each site
    result = mk.trend_test(
        site_data['value'],
        site_data['date'],
        slope_scaling='year'
    )

    # Collect the specific columns needed for regional_test
    # We need: 'site', 's' (Score), 'C' (Confidence)
    results_list.append({
        'site': site,
        's': result.s,
        'C': result.C,
        'slope': result.slope,
        'trend': result.trend
    })

results_df = pd.DataFrame(results_list)
print("Individual Results:")
print(results_df)


# 3. Run Regional Trend Test
# Now we aggregate the results. The function corrects for the fact that
# sites A, B, C, D, E are correlated (they move up and down together due to shared_signal).
print("\n--- Running Regional Trend Test ---")

regional_result = mk.regional_test(
    trend_results=results_df,
    time_series_data=full_data,
    site_col='site',
    value_col='value',
    time_col='date',
    s_col='s',
    c_col='C'
)

print(f"Number of Sites (M): {regional_result.M}")
print(f"Modal Trend Direction (DT): {regional_result.DT}")
print(f"Regional Confidence (CT): {regional_result.CT:.4f}")
print(f"Regional Tau (Trend Consistency): {regional_result.TAU:.4f}")

# 4. Visualize the Regional Data
# Plotting all sites together helps visualize the common pattern.
plt.figure(figsize=(10, 6))

for site in sites:
    site_data = full_data[full_data['site'] == site]
    plt.plot(site_data['date'], site_data['value'], marker='o', label=site)

plt.title(f"Regional Data (Regional Trend: {regional_result.DT}, Conf: {regional_result.CT:.1%})")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Save to script's directory
import os
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
plt.savefig(os.path.join(script_dir, 'regional_plot.png'))
print("\nSaved 'regional_plot.png'.")
```

### Step 2: Text Output
```text
Generating data for 5 sites...
     site       date      value
0  Site_A 2010-12-31  10.265005
1  Site_A 2011-12-31  10.128871
2  Site_A 2012-12-31  11.768670
3  Site_A 2013-12-31  12.066390
4  Site_A 2014-12-31  10.903388

--- Running Individual Trend Tests ---
Individual Results:
     site     s         C     slope       trend
0  Site_A  31.0  0.996355  0.463962  increasing
1  Site_B  31.0  0.996355  0.351809  increasing
2  Site_C  33.0  0.997896  0.503549  increasing
3  Site_D   5.0  0.639743  0.064203  increasing
4  Site_E   1.0  0.500000  0.003851    no trend

--- Running Regional Trend Test ---
Number of Sites (M): 5
Modal Trend Direction (DT): Increasing
Regional Confidence (CT): 0.9946
Regional Tau (Trend Consistency): 1.0000

Saved 'regional_plot.png'.

```

## Interpreting the Results

### 1. Individual Results
The table shows the Mann-Kendall score (`s`) and Confidence (`C`) for each site.
*   Sites A, B, C show high confidence (>95%).
*   Site D shows moderate confidence.
*   Site E shows low confidence (near 50%).

### 2. Regional Results
*   **Modal Trend Direction (Increasing)**: The dominant direction across the region is UP.
*   **Regional Tau (1.0)**: This indicates high consistency. 90% of the "trend evidence" points in the modal direction.
*   **Regional Confidence**: This is the key value. It tells us how confident we are that the *region as a whole* is trending, after discounting the fact that the sites are just copying each other's ups and downs.

### 3. Visualizing the Region (`regional_plot.png`)
![Regional Plot](regional_plot.png)

You can see the bundle of lines moving generally upward. The `regional_test` confirms this visual intuition with a rigorous statistical number.
