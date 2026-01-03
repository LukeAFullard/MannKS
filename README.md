<div align="center">
  <img src="https://raw.githubusercontent.com/LukeAFullard/MannKS/main/assets/logo.png" alt="MannKS Logo" width="600"/>

  # MannKS
  ### (Mann-Kendall Sen)

  **Robust Trend Analysis in Python**
</div>

---

## 📦 Installation

```bash
pip install mannks
```

**Requirements:** Python 3.7+, NumPy, Pandas, SciPy, Matplotlib

---

## ✨ What is MannKS?

**MannKS** (Mann-Kendall Sen) is a Python package for detecting trends in time series data using non-parametric methods. It's specifically designed for environmental monitoring, water quality analysis, and other fields where data is messy, irregular, or contains detection limits.

### When to Use MannKS

Use this package when your data has:
- **Irregular sampling intervals** (daily → monthly → quarterly)
- **Censored values** (measurements like `<5` or `>100`)
- **Seasonal patterns** you need to account for
- **No normal distribution** (non-parametric methods don't require it)
- **Small to moderate sample sizes** (n < 5,000 recommended)

**Don't use** if you need n > 46,340 observations.

**NEW IN V0.3.0**: **Rolling Trend Analysis**. The `rolling_trend_test` function allows you to perform a rolling window analysis, calculating the Sen's slope, Mann-Kendall score, and confidence intervals over time. This enables the detection of when a trend started, stopped, or changed direction, rather than just providing a single global summary. The feature includes:
*   Flexible window sizes (numeric or time-based, e.g., '10 years')
*   `compare_periods` utility to statistically test for changes in trend between two time periods (e.g., before vs. after an intervention).

**NEW IN V0.2.0**: The `trend_test` and `seasonal_trend_test` functions now support a **Block Bootstrap** method (`autocorr_method='block_bootstrap'`). This feature provides robust trend testing for data with serial correlation (autocorrelation) by resampling blocks of data rather than individual points, preserving the internal dependency structure.

See [Statistical Methodology: Bootstrap](./bootstrap.md) for a detailed explanation of the hybrid methodology:
*   **Hypothesis Testing (P-values)**: Uses *Detrended Residual Block Bootstrap* to generate a null distribution while preserving autocorrelation.
*   **Confidence Intervals (Sen's Slope)**: Uses *Pairs Block Bootstrap* to avoid bias when "reconstructing" censored data values from residuals.

---

## 🚀 Quick Start

```python
import pandas as pd
from MannKS import prepare_censored_data, trend_test

# 1. Prepare data with censored values
# Converts strings like '<5' into a structured format
values = [10, 12, '<5', 14, 15, 18, 20, '<5', 25, 30]
dates = pd.date_range(start='2020-01-01', periods=len(values), freq='ME')
data = prepare_censored_data(values)

# 2. Run trend test
# slope_scaling converts slope from "per second" to "per year"
result = trend_test(
    x=data,
    t=dates,
    slope_scaling='year',
    x_unit='mg/L',
    plot_path='trend.png'
)

# 3. Interpret results
print(f"Trend: {result.classification}")
print(f"Slope: {result.slope:.2f} {result.slope_units}")
print(f"Confidence: {result.C:.2%}")
```

**Output:**
```
Trend: Highly Likely Increasing
Slope: 24.57 mg/L per year
Confidence: 98.47%
```

![Trend Analysis Plot](https://raw.githubusercontent.com/LukeAFullard/MannKS/main/assets/quick_start_trend.png)

---

## 🎯 Key Features

### Core Functionality
- **Mann-Kendall Trend Test**: Detect monotonic trends with statistical significance
- **Sen's Slope Estimator**: Calculate trend magnitude with confidence intervals
- **Seasonal Analysis**: Separate seasonal signals from long-term trends
- **Regional Aggregation**: Combine results across multiple monitoring sites

### Data Handling
- **Censored Data Support**: Native handling of detection limits (`<5`, `>100`)
  - Three methods: Standard, LWP-compatible, Akritas-Theil-Sen (ATS)
  - Handles left-censored, right-censored, and mixed censoring
- **Unequal Spacing**: Uses actual time differences (not just rank order)
- **Missing Data**: Automatically handles NaN values and missing seasons
- **Temporal Aggregation**: Multiple strategies for high-frequency data

### Statistical Features
- **Continuous Confidence**: Reports likelihood ("Highly Likely Increasing") not just p-values
- **Data Quality Checks**: Automatic warnings for tied values, long runs, insufficient data
- **Robust Methods**: ATS estimator for heavily censored data
- **Flexible Testing**: Kendall's Tau-a or Tau-b, custom significance levels
- **Rolling Trends** (New in v0.3.0): Analyze how trends evolve over time with `rolling_trend_test`. See [Example 30](./Examples/30_Rolling_Trend_Analysis).
- **Block Bootstrap** (New in v0.2.0): Robust trend testing for autocorrelated data with automatic ACF-based block size selection. See [bootstrap.md](./bootstrap.md) for details and [Example 29](./Examples/29_Block_Bootstrap_Autocorrelation).


---

## 📊 Example Use Cases

### Seasonal Water Quality Trend
```python
from MannKS import seasonal_trend_test, check_seasonality

# Check if seasonality exists
seasonality = check_seasonality(x=data, t=dates, period=12, season_type='month')
print(f"Seasonal pattern detected: {seasonality.is_seasonal}")

# Run seasonal trend test
result = seasonal_trend_test(
    x=data,
    t=dates,
    period=12,
    season_type='month',
    agg_method='robust_median',  # Aggregates multiple samples per month
    slope_scaling='year'
)
```

### Regional Analysis Across Sites
```python
from MannKS import regional_test

# Run trend tests for each site
site_results = []
for site in ['Site_A', 'Site_B', 'Site_C']:
    result = trend_test(x=site_data[site], t=dates)
    site_results.append({
        'site': site,
        's': result.s,
        'C': result.C
    })

# Aggregate regional trend
regional = regional_test(
    trend_results=pd.DataFrame(site_results),
    time_series_data=all_site_data,
    site_col='site'
)
print(f"Regional trend: {regional.DT}, confidence: {regional.CT:.2%}")
```

---

## ⚠️ Important Limitations

### Sample Size
- **Recommended maximum: n = 5,000** (triggers memory warning)
- **Hard limit: n = 46,340** (prevents integer overflow)
- For larger datasets, use `regional_test()` to aggregate multiple smaller sites

### Statistical Assumptions
- **Independence**: Data points must be serially independent
  - Autocorrelation violates this and causes spurious significance
  - Pre-test with ACF or use block bootstrap methods if autocorrelated
- **Monotonic trend**: Cannot detect U-shaped or cyclical patterns
- **Homogeneous variance**: Most powerful when variance is constant over time

---

## 📚 Documentation

### Detailed Guides
- **[Trend Test Parameters](./Examples/Detailed_Guides/trend_test_parameters_guide.md)** - Full parameter reference
- **[Seasonal Analysis](./Examples/Detailed_Guides/seasonal_trend_test_parameters_guide.md)** - Season types and aggregation
- **[Regional Tests](./Examples/Detailed_Guides/regional_test_guide/README.md)** - Multi-site aggregation
- **[Analysis Notes](./Examples/Detailed_Guides/analysis_notes_guide.md)** - Interpreting data quality warnings
- **[Trend Classification](./Examples/Detailed_Guides/trend_classification_guide.md)** - Understanding confidence levels
- **[Bootstrap Methodology](./bootstrap.md)** - Block bootstrap for autocorrelated data

### Examples
The [Examples](./Examples/README.md) folder contains step-by-step tutorials from basic to advanced usage.

---

## 🔬 Validation

Extensively validated against:
- **LWP-TRENDS R script** (34 test cases, 99%+ agreement)
- **NADA2 R package** (censored data methods)
- Edge cases: missing data, tied values, all-censored data, insufficient samples

See [validation/](./validation/) for detailed comparison reports.

---

## 🙏 Acknowledgments

This package is heavily inspired by the excellent work of **[LandWaterPeople (LWP)](https://landwaterpeople.co.nz/)**. The robust censored data handling and regional aggregation methods are based on their R scripts and methodologies.

---

## 📖 References

1.  **Helsel, D.R. (2012).** *Statistics for Censored Environmental Data Using Minitab and R* (2nd ed.). Wiley.
2.  **Gilbert, R.O. (1987).** *Statistical Methods for Environmental Pollution Monitoring*. Wiley.
3.  **Hirsch, R.M., Slack, J.R., & Smith, R.A. (1982).** Techniques of trend analysis for monthly water quality data. *Water Resources Research*, 18(1), 107-121.
4.  **Mann, H.B. (1945).** Nonparametric tests against trend. *Econometrica*, 13(3), 245-259.
5.  **Sen, P.K. (1968).** Estimates of the regression coefficient based on a particular kind of rank correlation. *Journal of the American Statistical Association*, 63(324), 1379-1389.
6.  **Fraser, C., & Whitehead, A. L. (2022).** Continuous measures of confidence in direction of environmental trends at site and other spatial scales. *Environmental Challenges*, 9, 100601.
7.  **Fraser, C., Snelder, T., & Matthews, A. (2018).** State and trends of river water quality in the Manawatu-Whanganui region. Report for Horizons Regional Council.
