
# Example 30: Rolling Trend Analysis

This example demonstrates the `rolling_trend_test` feature, which applies the Mann-Kendall test and Sen's slope estimator over a moving window. This is crucial for non-stationary data where trends may start, stop, or reverse over time.

## Scenario
We generated 30 years of synthetic monthly data (1990-2020).
- **1990-2005:** No trend (random noise).
- **2005-2020:** Increasing trend.

## Rolling Analysis (10-Year Window)
We applied a **10-year rolling window** sliding by **1 year**.

### Python Code
```python
import pandas as pd
import numpy as np
import MannKS as mk

# [Data Generation Code Omitted - See run_example.py]

# Run Rolling Test (10Y)
rolling_results_10y = mk.rolling_trend_test(
    x=df['Value'],
    t=df['Date'],
    window='10Y',
    step='1Y',
    slope_scaling='year'
)

# Visualize
mk.plot_rolling_trend(
    rolling_results_10y,
    data=df,
    time_col='Date',
    value_col='Value',
    save_path='rolling_trend_analysis_10y.png'
)
```

### Results Table (10Y Snippet)
The rolling analysis detects the transition. Early windows (purely in the 1990-2005 range) should show no trend, while later windows capture the increase.

| window_center       |      slope |        C | classification              |
|:--------------------|-----------:|---------:|:----------------------------|
| 1995-01-15 12:00:00 | 0.037735   | 0.735429 | Likely Increasing           |
| 1995-12-31 12:00:00 | 0.0997992  | 0.934271 | Very Likely Increasing      |
| 1996-12-30 12:00:00 | 0.0391695  | 0.709246 | Likely Increasing           |
| 1997-12-31 00:00:00 | 0.0254953  | 0.660961 | As Likely as Not Increasing |
| 1998-12-31 00:00:00 | 0.00719657 | 0.544249 | As Likely as Not Increasing |
| 1999-12-31 12:00:00 | 0.0331101  | 0.682272 | Likely Increasing           |
| 2000-12-30 12:00:00 | 0.0429994  | 0.742472 | Likely Increasing           |
| 2001-12-31 00:00:00 | 0.11366    | 0.969971 | Highly Likely Increasing    |
| 2002-12-31 00:00:00 | 0.201128   | 0.999555 | Highly Likely Increasing    |
| 2003-12-31 12:00:00 | 0.210847   | 0.999729 | Highly Likely Increasing    |
| 2004-12-30 12:00:00 | 0.249812   | 0.999954 | Highly Likely Increasing    |
| 2005-12-31 00:00:00 | 0.305494   | 0.999999 | Highly Likely Increasing    |
| 2006-12-31 00:00:00 | 0.344532   | 1        | Highly Likely Increasing    |
| 2007-12-31 12:00:00 | 0.361481   | 1        | Highly Likely Increasing    |
| 2008-12-30 12:00:00 | 0.445643   | 1        | Highly Likely Increasing    |
| 2009-12-31 00:00:00 | 0.491485   | 1        | Highly Likely Increasing    |
| 2010-12-31 00:00:00 | 0.491611   | 1        | Highly Likely Increasing    |
| 2011-12-31 12:00:00 | 0.529855   | 1        | Highly Likely Increasing    |
| 2012-12-30 12:00:00 | 0.595248   | 1        | Highly Likely Increasing    |
| 2013-12-31 00:00:00 | 0.531863   | 1        | Highly Likely Increasing    |
| 2014-12-31 00:00:00 | 0.492711   | 1        | Highly Likely Increasing    |
| 2015-12-31 12:00:00 | 0.461411   | 1        | Highly Likely Increasing    |
| 2016-12-30 12:00:00 | 0.464708   | 1        | Highly Likely Increasing    |
| 2017-12-31 00:00:00 | 0.381008   | 0.99998  | Highly Likely Increasing    |
| 2018-12-31 00:00:00 | 0.320397   | 0.999002 | Highly Likely Increasing    |
| 2019-12-31 12:00:00 | 0.225124   | 0.952931 | Highly Likely Increasing    |
| 2020-12-30 12:00:00 | 0.0218824  | 0.565047 | As Likely as Not Increasing |
| 2021-12-31 00:00:00 | 0.157077   | 0.638004 | As Likely as Not Increasing |

### Visualization (10Y Window)
![Rolling Trend Plot 10Y](rolling_trend_analysis_10y.png)

## Rolling Analysis (5-Year Window)
We also applied a **5-year rolling window** to see how window size affects sensitivity. Shorter windows react faster to changes but may be noisier.

```python
# Run Rolling Test (5Y)
rolling_results_5y = mk.rolling_trend_test(
    x=df['Value'],
    t=df['Date'],
    window='5Y',
    step='1Y',
    slope_scaling='year'
)
```

### Visualization (5Y Window)
![Rolling Trend Plot 5Y](rolling_trend_analysis_5y.png)

## Change Point Verification
We manually compared the periods before and after 2005 using `compare_periods`.

- **Slope Before 2005:** 0.0512
- **Slope After 2005:** 0.4967
- **Significant Change:** True

The test correctly identifies a significant shift in the trend trajectory.
