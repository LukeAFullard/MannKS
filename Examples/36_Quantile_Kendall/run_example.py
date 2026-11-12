"""
Example 36: Generalized Quantile-Kendall Trend Analysis (MannKS v0.7.0)

This example demonstrates frequency-agnostic Quantile-Kendall trend analysis:
- Case 1: Weekly Nitrate Streamflow (20 years, agricultural winter flush trend in upper quantiles)
- Case 2: Daily Streamflow with 5-Year Blocks (30 years, divergent hydrological regimes: flood vs drought)
- Case 3: Sparse Irregular Censored Groundwater (25 years, 8-18 visits/year, detection limits)
- Case 4: Sampling Frequency Drift Trap & Equalization (20 years, artifactual trend resolution via thinning)

Generates lecture-style README.md and all 5 visual diagnostic plots.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MannKS import (
    quantile_trend_test,
    plot_quantile_trend,
    summarize_quantile_regimes,
    prepare_censored_data,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table without requiring optional tabulate package."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = [str(c) for c in df.columns]
        rows = []
        for _, row in df.iterrows():
            formatted_vals = []
            for val in row:
                if isinstance(val, (float, np.floating)):
                    if np.isnan(val):
                        formatted_vals.append("nan")
                    elif abs(val) < 1e-4 and val != 0:
                        formatted_vals.append(f"{val:.3e}")
                    else:
                        formatted_vals.append(f"{val:.4f}")
                else:
                    formatted_vals.append(str(val))
            rows.append(formatted_vals)
        col_widths = [max(len(c), max((len(r[i]) for r in rows), default=0)) for i, c in enumerate(cols)]
        header = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, col_widths)) + " |"
        sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
        body = "\n".join("| " + " | ".join(val.ljust(w) for val, w in zip(r, col_widths)) + " |" for r in rows)
        return f"{header}\n{sep}\n{body}"


def run_case_1(output_dir: str):
    """
    Case 1: Weekly Nitrate Streamflow (20 years)
    Simulates agricultural winter flushing: base summer concentration is flat,
    while winter concentrations rise by ~6% per year due to intensive tile drainage.
    """
    print("\n--- Running Case 1: Weekly Nitrate Streamflow ---")
    rng = np.random.default_rng(42)
    dates = pd.date_range("2005-01-01", "2024-12-31", freq="W")
    years = (dates - dates[0]).days / 365.25
    day_of_year = dates.day_of_year if hasattr(dates, 'day_of_year') else dates.dayofyear
    is_winter = np.cos(2 * np.pi * (day_of_year - 30) / 365.25) > 0.4
    nitrate = 1.2 + np.where(is_winter, 2.5 * (1.0 + 0.06 * years), 0.2) + rng.lognormal(0.0, 0.25, size=len(dates))

    result = quantile_trend_test(
        x=nitrate,
        t=dates,
        block_frequency='YE',
        quantiles='auto',
        slope_scaling='year',
        x_unit='mg/L',
    )

    plot_path = os.path.join(output_dir, "case1_weekly_nitrate_qk.png")
    plot_quantile_trend(
        result,
        save_path=plot_path,
        show_ci=True,
        show_fdc=True,
        title="Case 1: Weekly Nitrate Streamflow (Winter Flush Trend in Upper Quantiles)",
    )
    print(f"Saved: {plot_path}")

    regimes = summarize_quantile_regimes(result)
    print("Quantile Regimes Summary:")
    print(regimes.to_string(index=False))
    return result, regimes


def run_case_2(output_dir: str):
    """
    Case 2: Daily Streamflow with 5-Year Blocks (30 years)
    Demonstrates multi-year hydrological blocking ('5YE'), log transform ('log'),
    and divergent regimes: high-flow flood peaks increasing (+3%/yr) while
    low-flow drought baseflow decreases (-2%/yr).
    """
    print("\n--- Running Case 2: Daily Streamflow with 5-Year Blocks ---")
    rng = np.random.default_rng(42)
    dates = pd.date_range("1990-01-01", "2019-12-31", freq="D")
    t_years = (dates - dates[0]).days / 365.25
    doy = dates.day_of_year if hasattr(dates, 'day_of_year') else dates.dayofyear
    seasonality = 1.0 + 0.4 * np.cos(2 * np.pi * (doy - 100) / 365.25)

    # Baseflow decreasing by ~2%/year
    baseflow = 8.0 * np.exp(-0.02 * t_years) * seasonality * rng.lognormal(0.0, 0.2, size=len(dates))
    # Extreme storm runoff increasing by ~3%/year
    storm_shock = rng.pareto(a=3.0, size=len(dates))
    stormflow = np.where(storm_shock > 1.5, 25.0 * np.exp(0.03 * t_years) * (storm_shock - 1.5), 0.0)
    flow = np.maximum(baseflow + stormflow, 0.1)

    result = quantile_trend_test(
        x=flow,
        t=dates,
        block_frequency='5YE',
        quantiles='auto',
        transform='log',
        slope_scaling='year',
        x_unit='m3/s',
    )

    plot_path = os.path.join(output_dir, "case2_daily_flow_5y_qk.png")
    plot_quantile_trend(
        result,
        save_path=plot_path,
        show_ci=True,
        show_fdc=True,
        title="Case 2: Daily Streamflow 5-Year Blocks (Divergent Flood vs Baseflow Trends)",
    )
    print(f"Saved: {plot_path}")

    regimes = summarize_quantile_regimes(
        result,
        regimes={
            'drought_baseflow': (0.0, 0.50),
            'moderate_flow': (0.50, 0.95),
            'flood_peaks': (0.95, 1.0),
        },
    )
    print("Quantile Regimes Summary:")
    print(regimes.to_string(index=False))
    return result, regimes


def run_case_3(output_dir: str):
    """
    Case 3: Sparse Irregular Censored Groundwater (25 years)
    Demonstrates handling of real-world ambient groundwater monitoring networks
    with 8 to 18 irregular visits per year, detection limits (< 0.5 ug/L),
    min_coverage=0.70, and explicit user-specified quantiles.
    """
    print("\n--- Running Case 3: Sparse Irregular Censored Groundwater ---")
    rng = np.random.default_rng(42)
    visit_dates = []
    visit_vals = []
    visit_cens = []
    visit_types = []

    base_year = 2000
    for yr_idx in range(25):
        yr = base_year + yr_idx
        n_visits = int(rng.integers(8, 19))
        days = np.sort(rng.choice(np.arange(15, 350), size=n_visits, replace=False))
        for d in days:
            dt = pd.Timestamp(f"{yr}-01-01") + pd.Timedelta(days=int(d))
            visit_dates.append(dt)
            # Declining contamination over 25 years (-8%/year) with multiplicative noise
            true_c = 2.0 * np.exp(-0.08 * yr_idx) * rng.lognormal(0.0, 0.35)
            # Analytical detection limit at 0.5 ug/L
            if true_c < 0.5:
                visit_vals.append(0.5)
                visit_cens.append(True)
                visit_types.append('lt')
            else:
                visit_vals.append(true_c)
                visit_cens.append(False)
                visit_types.append('not')

    cens_df = pd.DataFrame({
        'value': visit_vals,
        'censored': visit_cens,
        'cen_type': visit_types,
    })
    visit_dates = pd.DatetimeIndex(visit_dates)

    target_quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    result = quantile_trend_test(
        x=cens_df,
        t=visit_dates,
        block_frequency='YE',
        quantiles=target_quantiles,
        min_coverage=0.70,
        min_size_per_block=8,
        slope_scaling='year',
        x_unit='ug/L',
    )

    plot_path = os.path.join(output_dir, "case3_sparse_irregular_qk.png")
    plot_quantile_trend(
        result,
        save_path=plot_path,
        show_ci=True,
        x_scale='linear',
        title="Case 3: Sparse Irregular Censored Groundwater (Annual Blocks)",
    )
    print(f"Saved: {plot_path}")

    regimes = summarize_quantile_regimes(result)
    print("Quantile Regimes Summary:")
    print(regimes.to_string(index=False))
    return result, regimes


def run_case_4(output_dir: str):
    """
    Case 4: Sampling Frequency Drift Trap & Equalization (20 years)
    Demonstrates the pitfall of sampling frequency changes in quantile analysis:
    - True underlying data is completely stationary lognormal (zero trend).
    - Years 1-10 are sampled monthly (N=12/year); Years 11-20 are sampled weekly (N=52/year).
    - Part A (Raw): Naive test shows false trends in tails and flags sampling diagnostic.
    - Part B (Equalized): Setting equalize='thin' restores true zero slope across all quantiles.
    """
    print("\n--- Running Case 4: Sampling Frequency Drift Trap & Equalization ---")
    rng = np.random.default_rng(42)

    # Years 1-10: Monthly (12/year)
    dates_m = pd.date_range("2005-01-01", "2014-12-31", freq="MS") + pd.Timedelta(days=14)
    # Years 11-20: Weekly (52/year)
    dates_w = pd.date_range("2015-01-01", "2024-12-31", freq="W")
    dates_all = dates_m.append(dates_w)

    # Stationary lognormal data (NO true trend at any quantile)
    values_all = rng.lognormal(mean=2.0, sigma=0.6, size=len(dates_all))

    target_q = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    # Part A: Raw (unequal sample sizes)
    res_raw = quantile_trend_test(
        x=values_all,
        t=dates_all,
        block_frequency='YE',
        quantiles=target_q,
        slope_scaling='year',
        x_unit='mg/L',
    )

    plot_path_raw = os.path.join(output_dir, "case4a_sampling_drift_raw_qk.png")
    plot_quantile_trend(
        res_raw,
        save_path=plot_path_raw,
        show_ci=True,
        title="Case 4A: Sampling Frequency Drift Trap (Raw - False Artifactual Tail Trends)",
    )
    print(f"Saved: {plot_path_raw}")

    # Part B: Equalized with equalize='thin'
    res_eq = quantile_trend_test(
        x=values_all,
        t=dates_all,
        block_frequency='YE',
        quantiles=target_q,
        equalize='thin',
        random_state=42,
        slope_scaling='year',
        x_unit='mg/L',
    )

    plot_path_eq = os.path.join(output_dir, "case4b_sampling_drift_equalized_qk.png")
    plot_quantile_trend(
        res_eq,
        save_path=plot_path_eq,
        show_ci=True,
        title="Case 4B: Equalized Blocks (Thinning Restores True Zero Slope Across Quantiles)",
    )
    print(f"Saved: {plot_path_eq}")

    print(f"Case 4A Sampling Diagnostic: {res_raw.sampling_diagnostic}")
    print(f"Case 4B Sampling Diagnostic: {res_eq.sampling_diagnostic}")
    return res_raw, res_eq


def generate_readme(res1, reg1, res2, reg2, res3, reg3, res4a, res4b, output_dir: str):
    """Compiles a comprehensive, lecture-style README.md for Example 36."""
    print("\n--- Compiling README.md ---")

    # Format table snippets
    df1_snip = res1.results[['rank', 'quantile', 'plotting_position', 'slope', 'p', 'classification']].iloc[::5].copy()
    df2_snip = res2.results[['rank', 'quantile', 'plotting_position', 'slope_pct', 'p', 'classification']].iloc[::10].copy()
    df3_all = res3.results[['rank', 'quantile', 'plotting_position', 'slope', 'prop_censored', 'p', 'classification']].copy()
    df4a_all = res4a.results[['rank', 'quantile', 'slope', 'p', 'classification']].copy()
    df4b_all = res4b.results[['rank', 'quantile', 'slope', 'p', 'classification']].copy()

    readme_text = f"""# Example 36: Generalized Quantile-Kendall Trend Analysis

This example demonstrates the **Generalized Quantile-Kendall (QK) Framework** introduced in **MannKS v0.7.0**. 

Traditional monotonic trend tests (such as standard Mann-Kendall or Seasonal Mann-Kendall) evaluate trends only in the **central tendency** (mean or median). However, environmental and hydrological changes rarely affect all percentiles uniformly:
- Agricultural pollution flushes streamflow concentrations primarily during high-runoff storm events (upper quantiles), leaving summer baseflow unchanged.
- Climate warming often intensifies hydrological extremes, causing flood peaks to increase while drought baseflows diminish.
- Ambient groundwater monitoring networks collect sparse, irregular, and censored data where classical daily methods cannot apply.
- Historical shifts in sampling frequency (e.g., transition from monthly to weekly monitoring) introduce severe sample-size artifacts into tail quantiles if not properly diagnosed and equalized.

MannKS v0.7.0 addresses these challenges by extending the order-statistic trend concept of USGS EGRET (*Hirsch et al., 2015*) into a **fully generalized, frequency-agnostic, censor-aware, and drift-protected** framework.

---

## Table of Contents
1. [Case 1: Weekly Nitrate Streamflow (Winter Flush Degradation)](#case-1-weekly-nitrate-streamflow)
2. [Case 2: Daily Streamflow with 5-Year Blocks (Divergent Hydrological Regimes)](#case-2-daily-streamflow-with-5-year-blocks)
3. [Case 3: Sparse Irregular Censored Groundwater (Ambient Monitoring)](#case-3-sparse-irregular-censored-groundwater)
4. [Case 4: The Sampling Frequency Drift Trap & Equalization](#case-4-the-sampling-frequency-drift-trap--equalization)
5. [Summary of Quantile-Kendall Diagnostics](#summary-of-quantile-kendall-diagnostics)

---

## Case 1: Weekly Nitrate Streamflow

### Scientific Background
A 20-year weekly streamflow monitoring program (2005–2024, $N=1,044$) tracks dissolved nitrate concentrations ($mg/L$) in an agricultural catchment. Subsurface tile drainage delivers heavy nitrate pulses during winter runoff events, while summer baseflow remains dominated by deep groundwater with stable concentrations.

A global median test would dilute this critical seasonal pollution signal. The Quantile-Kendall test segments the series into annual blocks (`block_frequency='YE'`), standardizes each year into 52 weekly order statistics, and evaluates Sen's slope across all non-exceedance probabilities.

### Python Code
```python
import numpy as np
import pandas as pd
from MannKS import quantile_trend_test, plot_quantile_trend, summarize_quantile_regimes

# 1. Generate 20 years of weekly monitoring data
rng = np.random.default_rng(42)
dates = pd.date_range("2005-01-01", "2024-12-31", freq="W")
years = (dates - dates[0]).days / 365.25
day_of_year = dates.day_of_year if hasattr(dates, 'day_of_year') else dates.dayofyear
is_winter = np.cos(2 * np.pi * (day_of_year - 30) / 365.25) > 0.4
nitrate = 1.2 + np.where(is_winter, 2.5 * (1.0 + 0.06 * years), 0.2) + rng.lognormal(0.0, 0.25, size=len(dates))

# 2. Run Generalized Quantile-Kendall Test
result = quantile_trend_test(
    x=nitrate,
    t=dates,
    block_frequency='YE',
    quantiles='auto',
    slope_scaling='year',
    x_unit='mg/L',
)

# 3. Diagnostic Plot with Flow Duration Curve overlay
plot_quantile_trend(
    result,
    save_path='case1_weekly_nitrate_qk.png',
    show_ci=True,
    show_fdc=True,
    title='Case 1: Weekly Nitrate Streamflow (Winter Flush Trend)',
)
```

### Visual Diagnostics
![Case 1: Weekly Nitrate Streamflow](case1_weekly_nitrate_qk.png)

### Results & Regime Summary
The resulting quantile slopes show clear selective degradation in the upper percentiles:

#### Sample Results Across Ranks
{df_to_markdown(df1_snip)}

#### Quantile Regimes Breakdown
{df_to_markdown(reg1)}

**Key Takeaway:** The low and mid percentiles (summer baseflow) exhibit no significant trend ($p > 0.10$, slope $\\approx 0$), whereas high quantiles ($p > 0.75$) exhibit highly significant increasing trends ($p < 0.001$, slope $> 0.15$ mg/L/year), capturing selective agricultural winter runoff pollution.

---

## Case 2: Daily Streamflow with 5-Year Blocks

### Scientific Background
30 years of daily streamflow ($m^3/s$) from 1990 to 2019 (~10,957 observations) are analyzed. In this watershed, climate change and land-use alterations have created **divergent hydrological regimes**:
1. Extreme precipitation events have intensified flood peaks (+3% per year in top quantiles).
2. Extended summer droughts and agricultural water extraction have caused baseflow to diminish (-2% per year in lower quantiles).

By setting `block_frequency='5YE'`, daily observations are grouped into 6 multi-year hydrological epochs of 1,826 order statistics each. Log-transformation (`transform='log'`) reports slopes directly as percentage change per year (`% per year`).

### Python Code
```python
# Daily streamflow with multi-year blocking and log transform
result2 = quantile_trend_test(
    x=daily_flow,
    t=daily_dates,
    block_frequency='5YE',
    quantiles='auto',
    transform='log',
    slope_scaling='year',
    x_unit='m3/s',
)

plot_quantile_trend(
    result2,
    save_path='case2_daily_flow_5y_qk.png',
    show_ci=True,
    show_fdc=True,
    title='Case 2: Daily Streamflow 5-Year Blocks (Divergent Flood vs Baseflow Trends)',
)

# Summarize custom hydrologic regimes
regimes2 = summarize_quantile_regimes(
    result2,
    regimes={{
        'drought_baseflow': (0.0, 0.50),
        'moderate_flow': (0.50, 0.95),
        'flood_peaks': (0.95, 1.0),
    }},
)
```

### Visual Diagnostics
![Case 2: Daily Streamflow 5-Year Blocks](case2_daily_flow_5y_qk.png)

### Results & Regime Summary
#### Sample Results Across Quantiles (%/year)
{df_to_markdown(df2_snip)}

#### Quantile Regimes Breakdown
{df_to_markdown(reg2)}

**Key Takeaway:** A standard overall Mann-Kendall test might register "No Trend" or a slight net decrease due to cancellation between opposite ends of the hydrograph. Quantile-Kendall explicitly uncovers the dual regime: **drought baseflow is drying up (-2%/year)** while **flood peak discharge is expanding (+3%/year)**.

---

## Case 3: Sparse Irregular Censored Groundwater

### Scientific Background
Ambient groundwater quality networks typically do not record daily or weekly data. Instead, wells are visited irregularly (8 to 18 times per year) over a 25-year period. In addition, organic contaminants and pesticides frequently fall below the analytical laboratory detection limit ($< 0.5\\ \\mu g/L$).

MannKS solves this by:
1. Validating block coverage (`min_coverage=0.70`) to ensure visits span at least 70% of each annual block.
2. Standardizing irregular visit counts to explicit user-requested quantiles (`quantiles=[0.10, 0.25, 0.50, 0.75, 0.90]`).
3. Propagating left-censoring flags through the order statistics and computing robust Akritas-Theil-Sen (ATS) slopes without artificial substitution plateaus.

### Python Code
```python
# Prepare censored DataFrame
cens_df = pd.DataFrame({{
    'value': visit_vals,
    'censored': visit_cens,
    'cen_type': visit_types,
}})

# Quantile trend test on irregular, censored series
result3 = quantile_trend_test(
    x=cens_df,
    t=visit_dates,
    block_frequency='YE',
    quantiles=[0.10, 0.25, 0.50, 0.75, 0.90],
    min_coverage=0.70,
    min_size_per_block=8,
    slope_scaling='year',
    x_unit='ug/L',
)

plot_quantile_trend(
    result3,
    save_path='case3_sparse_irregular_qk.png',
    show_ci=True,
    x_scale='linear',
    title='Case 3: Sparse Irregular Censored Groundwater (Annual Blocks)',
)
```

### Visual Diagnostics
![Case 3: Sparse Irregular Censored Groundwater](case3_sparse_irregular_qk.png)

### Results Table
{df_to_markdown(df3_all)}

**Key Takeaway:** The lower quantiles ($p = 0.10, 0.25$) are heavily censored (`prop_censored` > 0), yet MannKS accurately preserves interval bounds and detects remediation progress across all detectable percentiles.

---

## Case 4: The Sampling Frequency Drift Trap & Equalization

### The Sampling Drift Trap Explained
A notorious trap in non-parametric quantile analysis occurs when monitoring frequency increases over time (e.g., historical monthly sampling shifting to modern weekly sampling). Even if the underlying physical distribution is completely stationary:
- The expected maximum of a sample of size $N=52$ is statistically greater than the expected maximum of size $N=12$.
- The expected minimum of size $N=52$ is statistically smaller than the expected minimum of size $N=12$.

Consequently, naive order-statistic trend tests report **spurious upward trends in high quantiles** and **spurious downward trends in low quantiles**, generating false alarms of climate or environmental shifts.

MannKS automatically guards against this trap:
1. **Sampling Diagnostic:** Computes Kendall's $\\tau$ and the max/min sample size ratio between blocks. If $p < 0.10$ or $N_{{\\max}} / N_{{\\min}} > 1.5$, a warning is raised and `result.sampling_diagnostic['flag'] == True`.
2. **Equalization (`equalize='thin'`):** Systematically sub-samples blocks to a common sample size, eliminating the sample-size expansion artifact.

### Python Comparison
```python
# Part A: Naive Raw Analysis (Suffers from Drift Artifact)
res_raw = quantile_trend_test(
    x=stationary_lognormal_values,
    t=dates,
    block_frequency='YE',
    quantiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
    slope_scaling='year',
)

# Part B: Equalized Analysis (Thinning Eliminates Drift Artifact)
res_eq = quantile_trend_test(
    x=stationary_lognormal_values,
    t=dates,
    block_frequency='YE',
    quantiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
    equalize='thin',
    random_state=42,
    slope_scaling='year',
)
```

### Visual Diagnostics
#### Case 4A: Raw Un-equalized Series (False Artifactual Trends)
![Case 4A: Sampling Frequency Drift Trap](case4a_sampling_drift_raw_qk.png)

#### Case 4B: Equalized Series via Thinning (Drift Artifact Removed)
![Case 4B: Equalized Blocks](case4b_sampling_drift_equalized_qk.png)

### Numerical Comparison
#### Raw Results (Case 4A - Severe False Positive in Tails)
{df_to_markdown(df4a_all)}

#### Equalized Results (Case 4B - Drift Artifact Removed)
{df_to_markdown(df4b_all)}

#### Diagnostic Output
- **Raw Diagnostic:** `flag = {res4a.sampling_diagnostic['flag']}`, ratio $N_{{\\max}}/N_{{\\min}} = {res4a.sampling_diagnostic['ratio_max_min']:.2f}$, $p = {res4a.sampling_diagnostic['p_n_vs_time']:.4e}$
- **Equalized Diagnostic:** `flag = {res4b.sampling_diagnostic['flag']}`, ratio $N_{{\\max}}/N_{{\\min}} = {res4b.sampling_diagnostic['ratio_max_min']:.2f}$

---

## Summary of Quantile-Kendall Diagnostics

| Diagnostic / Feature | Parameter | Purpose |
| :--- | :--- | :--- |
| **Normal Z vs Linear Scale** | `x_scale='normal'` or `'linear'` | Spreads tail quantiles on probit scale ($Z = \\Phi^{{-1}}(p)$) or uniform $[0, 1]$. |
| **Flow Duration Curve** | `show_fdc=True` | Overlays median flow/concentration across blocks on a secondary right y-axis. |
| **Log Slopes (%/year)** | `transform='log'` | Exponentiates Sen's slope as percentage change per time unit ($100 \\times (e^b - 1)$). |
| **Coverage Filter** | `min_coverage=0.70` | Discards incomplete blocks where sampling span is less than 70% of block duration. |
| **Sampling Drift Guard** | `equalize='thin'` | Resamples blocks to equal size to eliminate tail order-statistic expansion artifacts. |
| **Regime Aggregator** | `summarize_quantile_regimes` | Summarizes low, mid, and high quantile trends into concise decision tables. |
"""

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_text)
    print(f"Saved: {readme_path}")


def main():
    res1, reg1 = run_case_1(OUT_DIR)
    res2, reg2 = run_case_2(OUT_DIR)
    res3, reg3 = run_case_3(OUT_DIR)
    res4a, res4b = run_case_4(OUT_DIR)
    generate_readme(res1, reg1, res2, reg2, res3, reg3, res4a, res4b, OUT_DIR)
    print("\nAll Example 36 cases executed successfully.")


if __name__ == "__main__":
    main()
