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

**Requirements:** Python 3.9+, NumPy, Pandas (≥ 2.2), SciPy, Matplotlib, Seaborn, Piecewise-Regression

---

## ✨ What is MannKS?

**MannKS** (Mann-Kendall Sen) is a Python package for detecting trends in time series data using non-parametric methods. It's specifically designed for environmental monitoring, water quality analysis, and other fields where data is messy, irregular, or contains detection limits.

### When to Use MannKS

Use this package when your data has:
- **Irregular sampling intervals** (daily → monthly → quarterly)
- **Censored values** (measurements like `<5` or `>100`)
- **Seasonal patterns** you need to account for
- **No normal distribution** (non-parametric methods don't require it)
- **Small to large sample sizes** (Automatic support for n > 10,000)

**NEW IN V0.6.0**: **Correctness release.** A pre-release audit against the LWP-TRENDS R code (run live) and independent references fixed several statistical defects, and the defaults changed. Results can differ from v0.5.x:
*   **Var(S) now includes the tie and censoring corrections** (Kendall 1975; NADA/LWP-TRENDS). Previously ties were never detected, which inflated Var(S) and p-values for any tied or censored data.
*   **The censored slope defaults to the Akritas–Theil–Sen estimator** (`sens_slope_method='ats'`, as in NADA::cenken), with a confidence interval from inverting the censored Kendall test. The former default, which drops ambiguous pairs, is biased away from zero (about +17% at 20% censoring and +70% at 50%). It remains available as `'drop_ambiguous'`; its old names `'unbiased'` and `'nan'` still work but are deprecated.
*   **Results no longer depend on input row order**: data are sorted by time, and observations with equal times score 0 in S.
*   **`large_dataset_mode='full'` is now exact.** Pair sampling, used only in fast mode, is reproducible by default.
*   **Smaller fixes:** LWP-TRENDS mid-observation selection (`agg_method='lwp'`), data-sufficiency notes aligned with LWP-TRENDS, Yue–Wang and `'auto'` autocorrelation estimated on detrended data, a seeded circular block bootstrap with a `(b+1)/(B+1)` p-value, `regional_test` aligned to time increments (and never reporting CT = 1 from a missing correlation), `sen_probability` matching LWP-TRENDS, right-closed classification thresholds, and a direct test of slope change in `compare_periods`.

See [validation/r_parity](./validation/r_parity) for the parity suite and the settings that reproduce LWP-TRENDS exactly. The full list of changes is in the [changelog](https://github.com/LukeAFullard/MannKS/blob/main/CHANGELOG.md).

**NEW IN V0.5.0**: **Large Dataset Support**. MannKS now automatically handles datasets with 10,000+ observations (up to 50,000+) by switching to optimized stochastic algorithms ("Fast Mode") for Sen's slope and using stratified sampling for seasonal tests. This reduces computation time from minutes to seconds while preserving statistical validity. See [Example 35](./Examples/35_Large_Data_Trend) and the [Large Dataset Guide](./Examples/Detailed_Guides/large_dataset_guide.md).

```python
# Automatic mode handles large data
result = trend_test(x_large, t_large) # Uses fast approximation if n > 5000

# Or force a specific mode
result = trend_test(x, t, large_dataset_mode='fast', max_pairs=100000)
```

**NEW IN V0.4.0**: **Segmented Trend Analysis**. The `segmented_trend_test` function performs a hybrid segmented regression analysis. It uses **Piecewise Regression** (OLS) to automatically identify structural breakpoints in the time series, followed by robust **Mann-Kendall / Sen's Slope** estimation on each identified segment. This allows you to detect distinct phases in a trend (e.g., "Stable" -> "Rapid Decrease" -> "Stable").

**NEW IN V0.3.0**: **Rolling Trend Analysis**. The `rolling_trend_test` function allows you to perform a rolling window analysis, calculating the Sen's slope, Mann-Kendall score, and confidence intervals over time. This enables the detection of when a trend started, stopped, or changed direction, rather than just providing a single global summary. The feature includes:
*   Flexible window sizes (numeric or time-based, e.g., '10 years')
*   `compare_periods` utility to statistically test for changes in trend between two time periods (e.g., before vs. after an intervention).

**NEW IN V0.2.0**: The `trend_test` and `seasonal_trend_test` functions now support a **Block Bootstrap** method (`autocorr_method='block_bootstrap'`). This feature provides robust trend testing for data with serial correlation (autocorrelation) by resampling blocks of data rather than individual points, preserving the internal dependency structure.

See [Statistical Methodology: Bootstrap](./bootstrap.md) for a detailed explanation of the hybrid methodology:
*   **Hypothesis Testing (P-values)**: Uses *Detrended Residual Block Bootstrap* to generate a null distribution while preserving autocorrelation.
*   **Confidence Intervals (Sen's Slope)**: Uses *Pairs Block Bootstrap* to avoid bias when "reconstructing" censored data values from residuals.

### Performance

Uncensored data, single core:

| Dataset Size | Mode | Time | Slope accuracy |
|--------------|------|------|----------|
| 1,000 | Full | 0.2s | Exact |
| 5,000 | Full | ~6s | Exact |
| 10,000 | Fast | < 1s | ±0.3% (1 s.d.) |
| 50,000 | Fast | < 1s | ±0.3% (1 s.d.) |
| 100,000 | Aggregate* | – | Depends** |

The Mann-Kendall test (S, Var(S), p) is exact in every mode; only the slope is sampled in fast mode.

\* Aggregate to ~1000 points first
\** Accuracy depends on aggregation method and trend pattern

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
Confidence: 99.23%
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
  - Slope estimators: Akritas-Theil-Sen (default), LWP-TRENDS-compatible, and drop-ambiguous-pairs
  - Handles left-censored, right-censored, and mixed censoring
- **Unequal Spacing**: Uses actual time differences (not just rank order)
- **Missing Data**: Automatically handles NaN values and missing seasons
- **Temporal Aggregation**: Multiple strategies for high-frequency data
- **Large Dataset Support**: Optimized algorithms for N > 10,000 (New in v0.5.0)

### Statistical Features
- **Continuous Confidence**: Reports likelihood ("Highly Likely Increasing") not just p-values
- **Data Quality Checks**: Automatic warnings for tied values, long runs, insufficient data
- **Robust Methods**: Akritas-Theil-Sen estimator for censored data (default)
- **Flexible Testing**: Kendall's Tau-a or Tau-b, custom significance levels
- **Rolling Trends** (New in v0.3.0): Analyze how trends evolve over time with `rolling_trend_test`. See [Example 31](./Examples/31_Rolling_Trend_Analysis).
- **Segmented Trends** (New in v0.4.0): Automatically detect breakpoints and analyze trends in segments with `segmented_trend_test`. See [Example 32](./Examples/32_Segmented_Regression).
- **Block Bootstrap** (New in v0.2.0): Robust trend testing for autocorrelated data with automatic ACF-based block size selection. See [bootstrap.md](./bootstrap.md) for details and [Example 29](./Examples/29_Block_Bootstrap_Autocorrelation).


---

## 📊 Example Use Cases

### Seasonal Water Quality Trend
```python
from MannKS import seasonal_trend_test, check_seasonality

# Check if seasonality exists (period=12 is inferred from season_type='month')
seasonality = check_seasonality(x=data, t=dates, season_type='month')
print(f"Seasonal pattern detected: {seasonality.is_seasonal}")

# Run seasonal trend test
result = seasonal_trend_test(
    x=data,
    t=dates,
    season_type='month',         # Infers period=12 automatically
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
- **Fast mode** (default above n = 5,000) samples pairs for the slope and handles 50,000+ observations in about a second; the test itself stays exact.
- **Exact slopes** (`large_dataset_mode='full'`) need all n(n-1)/2 pairs: practical up to about n = 10,000 on a typical machine (memory), hard limit n = 46,340.
- For very large or high-frequency data, aggregate first with `agg_method`/`agg_period` (e.g. daily or monthly medians); `large_dataset_mode='aggregate'` only warns, it does not aggregate.

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
- **[Rolling Trend Analysis](./Examples/Detailed_Guides/rolling_trend_guide.md)** - Moving window analysis
- **[Segmented Trend Analysis](./Examples/Detailed_Guides/segmented_trend_guide.md)** - Structural breakpoint detection
- **[Large Dataset Analysis](./Examples/Detailed_Guides/large_dataset_guide.md)** - Fast mode and stratification

### Examples
The [Examples](./Examples/README.md) folder contains step-by-step tutorials from basic to advanced usage.

---

## 🔬 Validation

Validated against:
- **LWP-TRENDS R script (v2502)**, run live via `Rscript`. With the LWP settings (`alpha=0.1`, `mk_test_method='lwp'`, `ci_method='lwp'`, `sens_slope_method='lwp'`, `tau_method='a'`, `tie_break_method='lwp'`, `agg_method='lwp'`), S, Var(S), Z, p, C, Cd, tau, the Sen slope and its confidence interval, and `sen_probability` agree to 1e-6 (relative). This holds on synthetic edge cases and on the LWP example data sets. The documented exceptions are cases where LWP-TRENDS declines to compute, and fast-mode sampling. Frozen R reference values are tested in `tests/test_lwp_parity.py`.
- **NADA::cenken** (Akritas-Theil-Sen): agreement within NADA's own bisection tolerance (median 0.3%).
- **Exact permutation distributions and Monte Carlo** (Type-I error, CI coverage, slope bias).

See [validation/r_parity](./validation/r_parity) for the scripts, and [validation/](./validation/) for the scenario studies.

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
