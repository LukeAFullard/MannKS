"""
Example 37: Monthly Sampled Data with Missing Values and Censoring
Generalized Quantile-Kendall Trend Analysis (MannKS v0.7.0)

This example demonstrates how to apply Generalized Quantile-Kendall trend
analysis to real-world monthly environmental monitoring datasets:
- 20-year monthly record (2005–2024, 240 planned observations)
- Real-world missing values (~12% random missing months)
- Incomplete year exclusion via `min_size_per_block=8`
- Left-censored values with historical detection limit change (<0.05 to <0.02 mg/L)
- Asymmetric trend: Baseflow concentrations improving (decreasing) while
  storm-runoff flushes (90th percentile) are worsening (increasing)
- Direct comparison against classical median trend analysis
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
    trend_test,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_monthly_dataset(seed: int = 42):
    """
    Simulates a realistic 20-year monthly water quality monitoring dataset.
    """
    rng = np.random.default_rng(seed)
    years = 20
    start_year = 2005
    dates = pd.date_range(start=f"{start_year}-01-01", periods=years * 12, freq="MS") + pd.Timedelta(days=14)
    t_years = (dates - dates[0]).total_seconds() / (365.25 * 86400.0)

    # 1. Base concentration with seasonal hydrology cycle (peak in winter/spring months)
    seasonal_cycle = 0.02 * np.cos(2 * np.pi * (dates.month - 2) / 12.0)
    
    # 2. Asymmetric trend:
    # Baseline (groundwater / point source) drops over time: ~ -4%/year
    clean_base = 0.08 * np.exp(-0.04 * t_years) + seasonal_cycle

    # Storm flush peaks (episodic pulses in months 11, 12, 1, 2) increase over time: ~ +7%/year
    is_flush_season = dates.month.isin([11, 12, 1, 2])
    flush_magnitudes = np.where(
        is_flush_season,
        0.25 * (1.0 + 0.07 * t_years) * (0.8 + 0.4 * rng.random(len(dates))),
        0.0
    )

    # Combined true continuous concentration
    true_conc = np.maximum(0.008, clean_base + flush_magnitudes + rng.normal(0, 0.008, len(dates)))

    # 3. Simulate missing values:
    # - ~12% randomly missed monthly samples (weather, site inaccessible, equipment downtime)
    missing_mask = rng.random(len(dates)) < 0.12

    # - Year 2012 experienced station damage: only 5 samples collected (7 months lost)
    year_2012_station_down = (dates.year == 2012) & (dates.month > 5)
    missing_mask = missing_mask | year_2012_station_down

    dates_sampled = dates[~missing_mask]
    values_sampled = true_conc[~missing_mask]

    # 4. Simulate censored values with historical detection limit change:
    # - 2005–2014: Standard analytical method (detection limit = 0.05 mg/L)
    # - 2015–2024: High-sensitivity ICP-MS upgrade (detection limit = 0.02 mg/L)
    raw_strings = []
    for d, val in zip(dates_sampled, values_sampled):
        dl = 0.05 if d.year < 2015 else 0.02
        if val < dl:
            raw_strings.append(f"<{dl}")
        else:
            raw_strings.append(f"{val:.4f}")

    censored_df = prepare_censored_data(raw_strings)

    return {
        'planned_n': len(dates),
        'actual_n': len(dates_sampled),
        'dates': dates_sampled,
        'censored_df': censored_df,
    }


def run_monthly_example():
    """Executes the monthly censored Quantile-Kendall analysis."""
    print("======================================================================")
    print("MannKS v0.7.0 — Example 37: Monthly Censored Quantile-Kendall Analysis")
    print("======================================================================")

    data = generate_monthly_dataset()
    dates = data['dates']
    cens_df = data['censored_df']

    n_cens = cens_df['censored'].sum()
    pct_cens = 100.0 * n_cens / len(cens_df)
    print(f"Dataset Overview:")
    print(f"  Planned monthly visits: {data['planned_n']} (20 years x 12 months)")
    print(f"  Successfully collected: {data['actual_n']} ({100.0*data['actual_n']/data['planned_n']:.1f}% data completeness)")
    print(f"  Censored values:        {n_cens} ({pct_cens:.1f}%)")
    print(f"  Detection limits:       <0.05 mg/L (2005-2014), <0.02 mg/L (2015-2024)\n")

    # Step 1: Classical Median Trend Analysis (for direct comparison)
    res_median = trend_test(
        x=cens_df,
        t=dates,
        slope_scaling='year',
        sens_slope_method='ats',
    )
    print(f"Classical Median Trend Analysis (ATS):")
    print(f"  Sen's Slope:    {res_median.slope:.5f} mg/L/year")
    print(f"  p-value:        {res_median.p:.4e}")
    print(f"  Classification: {res_median.classification}")
    print(f"  Diagnosis:      Median analysis detects only baseline improvement, completely")
    print(f"                  blind to upper storm flush risks!\n")

    # Step 2: Generalized Quantile-Kendall Analysis
    # Include NPS-FM regulatory percentiles (92nd and 95th percentiles) alongside
    # standard quantiles (10th baseflow, 25th, 50th median, 75th, 90th storm flush):
    target_quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.92, 0.95]

    res_qk = quantile_trend_test(
        x=cens_df,
        t=dates,
        block_frequency='YE',          # Annual blocks (January-December)
        quantiles=target_quantiles,     # Key distribution percentiles including 92nd and 95th
        standardization='interpolate',  # Robust linear quantile interpolation
        censor_quantile_method='propagate', # Flag order statistics without plateau distortion
        sens_slope_method='ats',        # Akritas-Theil-Sen slope for censored quantiles
        min_size_per_block=8,          # Require >= 8 monthly samples per year (drops year 2012)
        min_blocks=10,                 # Require >= 10 valid annual blocks
        min_coverage=0.70,             # Require >= 70% temporal span coverage
        slope_scaling='year',          # Annual slopes (units/year)
        x_unit='mg/L',
    )

    df_qk = res_qk.to_frame()
    print("Quantile-Kendall Analysis Results:")
    cols_to_print = ['quantile', 'slope', 'lower_ci', 'upper_ci', 'p', 'classification', 'prop_censored', 'n_blocks']
    print(df_qk[cols_to_print].to_string(index=False))

    regimes = summarize_quantile_regimes(res_qk)
    print("\nQuantile Regimes Summary:")
    print(regimes.to_string(index=False))

    # Step 3: Diagnostic Visualizations
    # 3A. Time series plot of original data with censored points and divergent trends
    ts_plot_path = os.path.join(OUT_DIR, "monthly_time_series.png")
    plot_original_time_series(dates, cens_df, res_qk, ts_plot_path)

    # 3B. Quantile-Kendall diagnostic plot across quantiles
    plot_path = os.path.join(OUT_DIR, "monthly_censored_qk.png")
    plot_quantile_trend(
        res_qk,
        save_path=plot_path,
        show_ci=True,
        show_fdc=True,
        x_scale='linear',
        color_by='p',
        title="Example 37: Monthly Quantile-Kendall (2005–2024, Censored & Missing Months)",
    )
    print(f"Generated quantile spectrum plot: {plot_path}")

    # Generate Markdown Table for README
    return {
        'planned_n': data['planned_n'],
        'actual_n': data['actual_n'],
        'n_cens': n_cens,
        'pct_cens': pct_cens,
        'median_res': res_median,
        'qk_res': res_qk,
        'qk_df': df_qk,
        'regimes': regimes,
    }


def plot_original_time_series(dates, cens_df, res_qk, save_path: str):
    """
    Plots the raw monthly monitoring time series, highlighting censored values,
    historical detection limit steps, station outages, and divergent quantile slopes.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    uncens_mask = ~cens_df['censored']
    cens_mask = cens_df['censored']

    # Scatter of observed uncensored values
    ax.scatter(dates[uncens_mask], cens_df.loc[uncens_mask, 'value'],
               color='#1f77b4', s=35, alpha=0.85, label='Observed (Uncensored)', zorder=4)

    # Scatter of left-censored values
    ax.scatter(dates[cens_mask], cens_df.loc[cens_mask, 'value'],
               color='#d62728', marker='v', s=45, label='Left-Censored (< Detection Limit)', zorder=5)

    # Detection limit timeline
    dl_dates = [pd.Timestamp('2005-01-01'), pd.Timestamp('2014-12-31'),
                pd.Timestamp('2015-01-01'), pd.Timestamp('2024-12-31')]
    dl_values = [0.05, 0.05, 0.02, 0.02]
    ax.step(dl_dates, dl_values, where='post', color='#d62728', linestyle='--', linewidth=1.5,
            alpha=0.75, label='Laboratory Detection Limit')

    # Shaded band for excluded 2012 station outage (< 8 samples)
    ax.axvspan(pd.Timestamp('2012-01-01'), pd.Timestamp('2012-12-31'),
               color='gray', alpha=0.18, label='2012 Station Outage (< 8 samples, dropped)')

    # Overlay trend lines for Median (50th) and 95th percentile
    df = res_qk.to_frame()
    slope_50 = df.loc[df['quantile'] == 0.50, 'slope'].iloc[0]
    slope_95 = df.loc[df['quantile'] == 0.95, 'slope'].iloc[0]
    p_50 = df.loc[df['quantile'] == 0.50, 'p'].iloc[0]
    p_95 = df.loc[df['quantile'] == 0.95, 'p'].iloc[0]

    b_dates = res_qk.matrix.index
    t_years = (b_dates - b_dates[0]).total_seconds() / (365.25 * 86400.0)
    t_mid = float(np.median(t_years))

    col_50 = [c for c in res_qk.matrix.columns if '0.5' in c or c == 'r3'][0]
    col_95 = [c for c in res_qk.matrix.columns if '0.95' in c or c == 'r7'][0]
    med_50 = float(np.median(res_qk.matrix[col_50]))
    med_95 = float(np.median(res_qk.matrix[col_95]))

    date_eval = [b_dates[0], dates.max()]
    t_eval = np.array([0.0, (dates.max() - b_dates[0]).total_seconds() / (365.25 * 86400.0)])
    y_pred_50 = med_50 + slope_50 * (t_eval - t_mid)
    y_pred_95 = med_95 + slope_95 * (t_eval - t_mid)

    ax.plot(date_eval, y_pred_50, color='#2ca02c', linestyle='-', linewidth=2.2,
            label=f'50th %ile Trend ({slope_50:+.4f} mg/L/yr, p={p_50:.4f})', zorder=6)
    ax.plot(date_eval, y_pred_95, color='#ff7f0e', linestyle='-', linewidth=2.2,
            label=f'95th %ile Trend ({slope_95:+.4f} mg/L/yr, p={p_95:.1e})', zorder=6)

    ax.set_xlabel('Monitoring Year', fontsize=11)
    ax.set_ylabel('Dissolved Phosphorus (mg/L)', fontsize=11)
    ax.set_title('Raw Monthly Water Quality Time Series (2005–2024)\nObserved Points, Censored Limits, and Divergent Quantile Trajectories',
                 fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, fontsize=9.2, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Generated raw time-series plot: {save_path}")


def write_readme(results: dict):
    """Writes the comprehensive lecture-style README.md for Example 37."""
    readme_path = os.path.join(OUT_DIR, "README.md")
    df = results['qk_df']
    regimes = results['regimes']
    m_res = results['median_res']

    table_rows = []
    for _, row in df.iterrows():
        table_rows.append(
            f"| **{row['quantile']:0.2f}** | {row['slope']:+.5f} | "
            f"[{row['lower_ci']:+.5f}, {row['upper_ci']:+.5f}] | {row['p']:.4e} | "
            f"{row['classification']} | {row['prop_censored']*100:.1f}% | {int(row['n_blocks'])} |"
        )
    table_str = "\n".join(table_rows)

    template = """# Example 37: Monthly Sampled Data with Missing Values & Censoring

## Overview & Real-World Practicality

Environmental, hydrological, and water quality monitoring programs are very commonly conducted on a **monthly sampling schedule** (12 planned observations per year). Long-term monitoring records typically span **10 to 25 years** ($N \\approx 120$ to $300$ observations).

In practical applications, these datasets present three major analytical challenges:
1. **Low Within-Block Sample Size ($N \\approx 12$)**: Each year contains at most 12 samples, meaning high-density order statistics (e.g. 50 or 100 quantiles) cannot be estimated without extreme extrapolation.
2. **Missing Observations**: Bad weather, frozen streams, equipment downtime, or funding gaps frequently cause missed monthly sampling visits (e.g., 9–11 samples collected instead of 12), and occasionally an entire year may lack sufficient data.
3. **Censored Data & Changing Detection Limits**: Trace contaminants (e.g., dissolved phosphorus, nitrate, pesticides, heavy metals) frequently fall below analytical laboratory detection limits ($< 0.05$ mg/L or $< 0.02$ mg/L). As analytical instruments improve over time, the detection limit often drops, creating an artificial downward trend if not handled rigorously.

`MannKS.quantile_trend_test` natively resolves all three challenges.

---

## Dataset Characteristics

In this example, we evaluate a 20-year dissolved phosphorus record (2005–2024) with realistic monitoring imperfections:
- **Planned Sampling**: 20 years $\\times$ 12 months = 240 planned visits.
- **Data Completeness**: ~12% randomly missed monthly visits, leaving {ACTUAL_N} collected observations.
- **Incomplete Year Exclusion**: Years with fewer than 8 samples (2012 station outage with 5 samples, and 2024 where missed monthly visits left 6 samples) are safely excluded by `min_size_per_block=8`, leaving **{VALID_BLOCKS} valid annual blocks** ($N_{\\text{blocks}} = {VALID_BLOCKS} \\ge 10$).
- **Censored Detection Limits**:
  - 2005–2014: Standard method ($< 0.05$ mg/L).
  - 2015–2024: Upgraded high-sensitivity method ($< 0.02$ mg/L).
  - Total censored observations: **{N_CENS} ({PCT_CENS}%)**.
- **Underlying Asymmetric Trend**:
  - Baseline concentrations (lower/median quantiles) are **improving (decreasing)** due to wastewater treatment facility upgrades.
  - Peak storm runoff pulses (upper quantiles, 90th–95th percentiles) are **worsening (increasing)** due to expanding impervious surfaces and agricultural runoff flushes.

---

## 1. Visualizing the Original Monitoring Data

Before decomposing the distribution into quantiles, we inspect the full monthly time series. The plot below illustrates the dataset:
- **Observed non-censored concentrations** (solid blue circles)
- **Left-censored values** (red inverted triangles plotted at the detection limit)
- **Laboratory detection limit transitions** (dashed red step line at $0.05$ mg/L from 2005–2014, improving to $0.02$ mg/L from 2015–2024)
- **Station outage** (shaded grey band during 2012 when only 5 samples were collected and subsequently dropped by `min_size_per_block=8`)
- **Divergent trend trajectories**: The **50th percentile (median) trend** is steadily decreasing (green line), while the **95th percentile storm pulse trend** is rapidly increasing (orange line).

![Raw Monthly Water Quality Time Series](monthly_time_series.png)

---

## 2. The Danger of Classical Median Analysis

Running a classical monotonic trend test (e.g., Mann-Kendall or Seasonal Mann-Kendall on the median) yields:
- **Sen's Slope**: `{MEDIAN_SLOPE} mg/L/year`
- **p-value**: `{MEDIAN_P}`
- **Classification**: **{MEDIAN_CLASS}**

> [!WARNING]
> The classical median test reports a statistically significant "Improving / Decreasing" trend. A water resources manager relying solely on the median would conclude the watershed is recovering, completely missing the fact that **extreme peak pollution flushes (90th–95th percentiles) are actively worsening**!

---

## Generalized Quantile-Kendall Workflow

### Step 1: Preprocess Censored Strings
Use `MannKS.prepare_censored_data` to automatically parse numeric and censored strings (e.g., `'<0.05'`, `'0.084'`):
```python
from MannKS import prepare_censored_data, quantile_trend_test, plot_quantile_trend

# Automatically creates DataFrame with 'value', 'censored', and 'cen_type'
censored_df = prepare_censored_data(raw_data_strings)
```

### Step 2: Configure Parameters for Monthly Data
For monthly data ($N \\approx 12$ per block), choose parameters tailored to the monthly sampling frequency, including regulatory NPS-FM percentiles:
- `quantiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.92, 0.95]`: Spans baseflow (10th/25th), median (50th), storm runoff flush (75th/90th), and NPS-FM regulatory thresholds (92nd and 95th).
- `standardization='interpolate'`: Uses robust linear percentile interpolation to estimate quantiles from 8–12 monthly points per block.
- `min_size_per_block=8`: Requires at least 8 samples per year, allowing years with 1–4 missed months to be retained while dropping severely compromised years (e.g., 2012 with only 5 samples).
- `min_blocks=10`: Ensures at least 10 valid annual blocks exist across the 20-year span.
- `censor_quantile_method='propagate'`: Flags order statistics at or below detection limits as censored without distorting uncensored upper quantiles.
- `sens_slope_method='ats'`: Uses Akritas-Theil-Sen slope estimation for quantiles containing censored points.

```python
result = quantile_trend_test(
    x=censored_df,
    t=dates,
    block_frequency='YE',               # Group by calendar year
    quantiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.92, 0.95],
    standardization='interpolate',       # Linear interpolation
    censor_quantile_method='propagate',  # Rigorous censor tracking
    sens_slope_method='ats',             # Akritas-Theil-Sen
    min_size_per_block=8,               # Tolerate up to 4 missing months
    min_blocks=10,                      # Require >= 10 valid years
    slope_scaling='year',               # Slopes in mg/L/year
    x_unit='mg/L',
)
```

---

## Numerical Results

| Quantile | Sen's Slope (mg/L/yr) | 95% Confidence Interval | p-value | Classification | % Censored | Valid Blocks |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{TABLE_STR}

### Key Findings
1. **Low Quantiles ($p = 0.10, 0.25$)**: Exhibit significant downward slopes with non-zero censoring proportions (`prop_censored`). The Akritas-Theil-Sen estimator correctly accounts for changing detection limits without artificial step-trend artifacts.
2. **Median ($p = 0.50$)**: Confirms steady baseline improvement ($p < 0.05$).
3. **Upper Quantiles ($p = 0.75, 0.90, 0.92, 0.95$)**: 100% uncensored (`prop_censored = 0.0%`) and demonstrate strong **positive slopes**. The Quantile-Kendall test successfully reveals the worsening storm pulse flushes that dominate the regulatory extremes.

---

## NPS-FM Context: Calculating 92nd and 95th Percentiles on Monthly Records

In water quality management under the **New Zealand National Policy Statement for Freshwater Management (NPS-FM 2020)**, attribute states and objective framework bands (e.g., *E. coli* primary contact recreation criteria, dissolved reactive phosphorus, toxic ammonia, and cyanobacteria) frequently rely on the **95th percentile** or **92nd percentile**.

When applying Quantile-Kendall to monthly monitoring datasets over **10 to 20 years**:

### 1. Order-Statistic Behavior for $N \\approx 12$ Samples/Year
Within an annual block of 12 monthly samples:
- The 12th (highest) sample has non-exceedance probability $p \\approx 0.92$ (Weibull $12/13$) or $p \\approx 0.96$ (Hazen $(12 - 0.5)/12$).
- The 11th (second-highest) sample has $p \\approx 0.85$ to $0.88$.
- **The 92nd percentile** is interpolated between the 11th and 12th samples ($1 + 0.92 \\times 11 = 11.12$ in 1-based rank: $0.88\\,x_{(11)} + 0.12\\,x_{(12)}$), primarily tracking the **second-highest monthly value** of the year.
- **The 95th percentile** uses Type-7 linear interpolation ($1 + 0.95 \\times 11 = 11.45$ in 1-based rank, or index $10.45$ in 0-based indexing: $0.55\\,x_{(11)} + 0.45\\,x_{(12)}$), blending nearly equally between the second-highest and highest monthly values.

In hydrological terms, evaluating trends in the 92nd and 95th percentiles tests whether the **annual peak runoff events are intensifying** over time.

### 2. Time Horizon Considerations (10 Years vs. 20 Years)
- **20 Years ($N_{\\text{blocks}} = 20$):** Very reliable. The 20 annual estimates provide sufficient degrees of freedom for the Mann-Kendall test ($S_{\\max} = 190$) to filter out natural year-to-year storm variability and identify true monotonic trends in extreme flushes.
- **10 Years ($N_{\\text{blocks}} = 10$):** Feasible, but confidence intervals around the Sen's slope will naturally be wider. An isolated extreme storm year exerts higher leverage when only 10 blocks are present.

### 3. Multi-Year Pooling Option (`block_frequency='2YE'`)
If within-block percentile stabilization is desired, consecutive years can be pooled into 2-year blocks (`block_frequency='2YE'`, $N \\approx 24$ samples per block), giving more empirical points above the 95th percentile.

---

## Visual Diagnostic Plot

```python
plot_quantile_trend(
    result,
    save_path='monthly_censored_qk.png',
    show_ci=True,
    show_fdc=True,
    x_scale='linear',
    title='Monthly Water Quality (2005–2024, Censored & Missing Months)',
)
```

![Monthly Censored Quantile-Kendall Plot](monthly_censored_qk.png)

---

## Best Practices for Monthly Datasets (10–25 Years)

| Scenario / Decision | Recommendation | Rationale |
| :--- | :--- | :--- |
| **Quantile Selection** | Include median, quartiles, and regulatory percentiles: `[0.10, 0.25, 0.50, 0.75, 0.90, 0.92, 0.95]` | Provides full distributional coverage while capturing critical NPS-FM regulatory thresholds. |
| **Interpolation Method** | `standardization='interpolate'` (default) | Handles varying block sizes ($N=8$ to $12$) smoothly via linear percentile interpolation. |
| **Missing Month Tolerance** | `min_size_per_block=8` (or 9) | Prevents losing entire years just because 1 or 2 monthly visits were missed. |
| **Missing Year Spacing** | Supported natively | Mann-Kendall evaluates pairwise time differences $\\Delta t = t_j - t_i$; missing years do not bias test statistics. |
| **Censored Data** | `censor_quantile_method='propagate'` | Ensures lower percentiles retain `< DL` flags while upper percentiles remain completely uncensored. |
| **Quantile vs Seasonal MK** | Choose based on hypothesis | Use **Seasonal MK** to ask *"Is January getting wetter?"*; use **Quantile-Kendall** to ask *"Are annual drought baseflows or peak storm flushes intensifying?"* regardless of timing. |
"""

    content = (
        template
        .replace("{ACTUAL_N}", str(results['actual_n']))
        .replace("{VALID_BLOCKS}", str(int(df['n_blocks'].iloc[0])))
        .replace("{N_CENS}", str(results['n_cens']))
        .replace("{PCT_CENS}", f"{results['pct_cens']:.1f}")
        .replace("{MEDIAN_SLOPE}", f"{m_res.slope:+.5f}")
        .replace("{MEDIAN_P}", f"{m_res.p:.4e}")
        .replace("{MEDIAN_CLASS}", str(m_res.classification))
        .replace("{TABLE_STR}", table_str)
    )

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated documentation: {readme_path}")


if __name__ == "__main__":
    res = run_monthly_example()
    write_readme(res)
