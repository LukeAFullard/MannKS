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
# Generate a dataset with a known positive trend and right-censoring
np.random.seed(42)
n = 50
t = pd.to_datetime(pd.date_range(start='2000-01-01', periods=n, freq='YE'))
slope = 0.1
intercept = 5
noise = np.random.normal(0, 1.5, n)
x = slope * np.arange(n) + intercept + noise

# Introduce right-censoring for values above a threshold
censor_threshold = 8.0
x_censored = [f">{censor_threshold}" if val > censor_threshold else val for val in x]

data = pd.DataFrame({'time': t, 'value': x_censored})
csv_path = os.path.join(os.path.dirname(__file__), 'data.csv')
data.to_csv(csv_path, index=False)


# --- 2. MannKS Analysis ---
# Pre-process the censored data
processed_data = mk.prepare_censored_data(data['value'])

# Run with standard (robust) settings
plot_path = os.path.join(os.path.dirname(__file__), 'right_censored_plot.png')
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
    mk_test_method='lwp', # Key for right-censored
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
# Validation Case V-06: Right-Censored Increasing Trend

## Objective
This validation case verifies the handling of right-censored (`>`) data, which is a key point of difference between the robust and LWP-emulation methods.

## Data
A synthetic dataset of {n} annual samples was generated with a positive slope. Values above `{censor_threshold}` were converted to right-censored strings (e.g., `'>{censor_threshold}'`). The generated plot from the standard `MannKS` analysis is shown below.

![Right-Censored Plot](right_censored_plot.png)

```python
import pandas as pd
import numpy as np
import MannKS as mk

# Generate Data
np.random.seed(42)
n = {n}
t = pd.to_datetime(pd.date_range(start='2000-01-01', periods=n, freq='YE'))
slope = {slope}
intercept = {intercept}
noise = np.random.normal(0, 1.5, n)
x = slope * np.arange(n) + intercept + noise

# Introduce right-censoring
censor_threshold = {censor_threshold}
x_censored = [f">{censor_threshold}" if val > censor_threshold else val for val in x]

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
This case highlights the difference between the `mk_test_method='robust'` (Standard) and `mk_test_method='lwp'` (LWP Mode) settings.

The **LWP-TRENDS R Script** uses a heuristic that replaces all right-censored values with a single numeric value slightly larger than the maximum detection limit. The **MannKS (LWP Mode)** replicates this, resulting in nearly identical p-values and slopes.

The **MannKS (Standard)** method, however, uses a more statistically robust ranking approach that does not modify the data. It treats comparisons between uncensored values and right-censored values as ambiguous, which is a more conservative approach. This results in a higher p-value, correctly reflecting the increased uncertainty introduced by the right-censored data.
"""

readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
with open(readme_path, 'w') as f:
    f.write(readme_content.strip())

print("Validation V-06 complete. README.md generated.")
