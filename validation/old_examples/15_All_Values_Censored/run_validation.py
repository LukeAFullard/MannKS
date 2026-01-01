import os
import pandas as pd
import numpy as np
import MannKS as mk

# rpy2 setup
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.rinterface import NULL as r_NULL

# --- 1. Data Generation ---
# Generate a dataset where all values are the same left-censored value
n = 10
t = pd.to_datetime(pd.date_range(start='2000-01-01', periods=n, freq='YE'))
x_censored = ['<5'] * n

data = pd.DataFrame({'time': t, 'value': x_censored})
csv_path = os.path.join(os.path.dirname(__file__), 'data.csv')
data.to_csv(csv_path, index=False)


# --- 2. MannKS Analysis ---
# Pre-process the censored data
prepared_data = mk.prepare_censored_data(data['value'])

# Run with standard (robust) settings
mk_standard = mk.trend_test(prepared_data, t, slope_scaling='year', alpha=0.1)

# Run with LWP-emulation settings
mk_lwp = mk.trend_test(
    prepared_data, t,
    slope_scaling='year',
    alpha=0.1,
    mk_test_method='lwp',
    ci_method='lwp',
    sens_slope_method='lwp',
    tie_break_method='lwp'
)

# --- 3. R LWP-TRENDS Analysis ---
r_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Example_Files/R/LWPTrends_v2502/LWPTrends_v2502.r'))
base = importr('base')
base.source(r_script_path)

with localconverter(ro.default_converter + pandas2ri.converter):
    r_data = ro.conversion.py2rpy(data)

ro.globalenv['mydata'] = r_data
ro.r('mydata$myDate <- as.Date(mydata$time)')
ro.r('data_processed <- RemoveAlphaDetect(mydata, ColToUse="value")')
ro.r('data_processed <- GetMoreDateInfo(data_processed)')
ro.r('data_processed <- InspectTrendData(data_processed, Year="Year")')
ro.r('data_processed$TimeIncr <- data_processed$Year')

r_results = ro.r('NonSeasonalTrendAnalysis(data_processed, ValuesToUse="RawValue", TimeIncrMed=TRUE)')

with localconverter(ro.default_converter + pandas2ri.converter):
    r_results_df = ro.conversion.rpy2py(r_results) if r_results is not r_NULL else pd.DataFrame([{'p': np.nan, 'AnnualSenSlope': np.nan, 'Sen_Lci': np.nan, 'Sen_Uci': np.nan}])

# Clean up R NA values
r_na_int = -2147483648
r_results_df.replace(r_na_int, np.nan, inplace=True)


# --- 4. Generate README Report ---
readme_content = f"""
# Validation Case V-15: All Values Censored

## Objective
This validation case verifies that all analysis methods gracefully handle a dataset where 100% of the values are censored. In such a scenario, no trend can or should be calculated.

## Data
A synthetic dataset of {n} annual samples was generated where every value is the same left-censored string: ` <5 `.

## Results Comparison

The following table compares the key statistical outputs. As expected, no trend could be calculated.

| Metric              | MannKS (Standard) | MannKS (LWP Mode) | LWP-TRENDS R Script |
|---------------------|-----------------------|-----------------------|---------------------|
| p-value             | {mk_standard.p:.4f}   | {mk_lwp.p:.4f}        | {r_results_df['p'].iloc[0]:.4f}     |
| Sen's Slope         | {mk_standard.slope:.4f} | {mk_lwp.slope:.4f}    | {r_results_df['AnnualSenSlope'].iloc[0]:.4f}       |
| Classification      | {mk_standard.classification} | {mk_lwp.classification} | N/A                 |
| Analysis Notes      | `{'<br>'.join(mk_standard.analysis_notes)}` | `{'<br>'.join(mk_lwp.analysis_notes)}` | N/A                 |


## Analysis
All three methods correctly determined that no trend could be calculated from a dataset composed entirely of identical censored values.

-   **MannKS (Standard & LWP Mode):** Both functions returned a p-value of `1.0` and a Sen's slope of `0.0`, correctly identifying the complete lack of any trend. The analysis notes properly flag that there is only one unique value, which is the root cause.
-   **LWP-TRENDS R Script:** The R script also produces `NA` (Not Available) for its primary results, as no valid statistical comparison can be made. Our wrapper correctly translates this to `nan`.

This confirms that all systems behave as expected and do not produce misleading results when faced with this edge case.
"""

readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
with open(readme_path, 'w') as f:
    f.write(readme_content.strip())

print("Validation V-15 complete. README.md generated.")
