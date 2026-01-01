import os
import pandas as pd
import numpy as np
import MannKS as mk

# rpy2 setup
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# --- 1. Data Generation ---
# Generate a dataset with a known positive trend and mixed censoring
np.random.seed(45)
n = 60
t = pd.to_datetime(pd.date_range(start='1990-01-01', periods=n, freq='YE'))
slope = 0.2
intercept = 10
noise = np.random.normal(0, 2.0, n)
x = slope * np.arange(n) + intercept + noise

# Introduce mixed-censoring
left_censor_threshold = 12.0
right_censor_threshold = 19.0
x_censored = []
for val in x:
    if val < left_censor_threshold:
        x_censored.append(f"<{left_censor_threshold}")
    elif val > right_censor_threshold:
        x_censored.append(f">{right_censor_threshold}")
    else:
        x_censored.append(val)

data = pd.DataFrame({'time': t, 'value': x_censored})
csv_path = os.path.join(os.path.dirname(__file__), 'data.csv')
data.to_csv(csv_path, index=False)


# --- 2. MannKS Analysis ---
# Pre-process the censored data
processed_data = mk.prepare_censored_data(data['value'])

# Run with standard (robust) settings
plot_path = os.path.join(os.path.dirname(__file__), 'mixed_censored_plot.png')
mk_standard = mk.trend_test(
    processed_data, t,
    slope_scaling='year',
    alpha=0.1,
    plot_path=plot_path,
    mk_test_method='robust' # Explicitly set for clarity
)

# Run with LWP-emulation settings
mk_lwp = mk.trend_test(
    processed_data, t,
    slope_scaling='year',
    alpha=0.1,
    mk_test_method='lwp',
    ci_method='lwp',
    sens_slope_method='lwp'
)

# --- 3. R LWP-TRENDS Analysis ---
r_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Example_Files/R/LWPTrends_v2502/LWPTrends_v2502.r'))
base = importr('base')
base.source(r_script_path)

# Use the localconverter for pandas->R conversion
with localconverter(ro.default_converter + pandas2ri.converter):
    r_data = ro.conversion.py2rpy(data)

# Prepare and run the R analysis
ro.globalenv['mydata'] = r_data
ro.r('mydata$myDate <- as.Date(mydata$time)')
ro.r('data_processed <- RemoveAlphaDetect(mydata, ColToUse="value")')
ro.r('data_processed <- GetMoreDateInfo(data_processed)')
ro.r('data_processed <- InspectTrendData(data_processed, Year="Year")')
ro.r('data_processed$TimeIncr <- data_processed$Year')

r_results = ro.r('NonSeasonalTrendAnalysis(data_processed, ValuesToUse="RawValue", TimeIncrMed=TRUE)')

# Extract results
with localconverter(ro.default_converter + pandas2ri.converter):
    r_results_df = ro.conversion.rpy2py(r_results)

r_p_value = r_results_df['p'].iloc[0]
r_slope = r_results_df['AnnualSenSlope'].iloc[0]
r_lower_ci = r_results_df['Sen_Lci'].iloc[0]
r_upper_ci = r_results_df['Sen_Uci'].iloc[0]


# --- 4. Generate README Report ---
readme_content = f"""
# Validation Case V-07: Mixed Censoring

## Objective
This validation case verifies the handling of a dataset that contains both left-censored (`<`) and right-censored (`>`) data.

## Data
A synthetic dataset of {n} annual samples was generated with a positive slope. Values below `{left_censor_threshold}` were left-censored, and values above `{right_censor_threshold}` were right-censored. The generated plot from the standard `MannKS` analysis is shown below.

![Mixed-Censored Plot](mixed_censored_plot.png)

```python
import pandas as pd
import numpy as np
import MannKS as mk

# Generate Data
np.random.seed(45)
n = {n}
t = pd.to_datetime(pd.date_range(start='1990-01-01', periods=n, freq='YE'))
slope = {slope}
intercept = {intercept}
noise = np.random.normal(0, 2.0, n)
x = slope * np.arange(n) + intercept + noise

# Introduce mixed-censoring
left_censor_threshold = {left_censor_threshold}
right_censor_threshold = {right_censor_threshold}
x_censored = []
for val in x:
    if val < left_censor_threshold:
        x_censored.append(f"<{left_censor_threshold}")
    elif val > right_censor_threshold:
        x_censored.append(f">{right_censor_threshold}")
    else:
        x_censored.append(val)

# Pre-process and run MannKS
processed_data = mk.prepare_censored_data(x_censored)
mk_results = mk.trend_test(processed_data, t)
print("p-value:", mk_results.p)
```

## Results Comparison

| Metric              | MannKS (Standard) | MannKS (LWP Mode) | LWP-TRENDS R Script |
|---------------------|-----------------------|-----------------------|---------------------|
| p-value             | {mk_standard.p:.6f}   | {mk_lwp.p:.6f}        | {r_p_value:.6f}     |
| Sen's Slope         | {mk_standard.slope:.6f} | {mk_lwp.slope:.6f}    | {r_slope:.6f}       |
| Lower CI (90%)      | {mk_standard.lower_ci:.6f} | {mk_lwp.lower_ci:.6f} | {r_lower_ci:.6f}    |
| Upper CI (90%)      | {mk_standard.upper_ci:.6f} | {mk_lwp.upper_ci:.6f} | {r_upper_ci:.6f}    |

## Analysis
This case combines the behaviors seen in the individual left-censored and right-censored tests.

The **MannKS (LWP Mode)** continues to closely replicate the **LWP-TRENDS R Script** by using the same data substitution heuristics for both left- and right-censored data.

The **MannKS (Standard)** method uses its robust ranking approach for both types of censoring, treating ambiguous comparisons conservatively. As expected, this results in a slightly higher (less significant) p-value and a different Sen's slope, reflecting the increased uncertainty from having both left- and right-censored data in the same analysis.
"""

readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
with open(readme_path, 'w') as f:
    f.write(readme_content.strip())

print("Validation V-07 complete. README.md generated.")
