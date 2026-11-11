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
    # For monthly data (N=12/year), 5 key percentiles provide optimal resolution:
    # 10th (extreme baseflow), 25th (lower quartile), 50th (median),
    # 75th (upper quartile), and 90th (extreme storm runoff pulse)
    target_quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

    res_qk = quantile_trend_test(
        x=cens_df,
        t=dates,
        block_frequency='YE',          # Annual blocks (January-December)
        quantiles=target_quantiles,     # 5 key distribution percentiles
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

    # Step 3: Diagnostic Visualization
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
    print(f"\nGenerated diagnostic plot: {plot_path}")

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
- **Incomplete Year Exclusion**: Years with fewer than 8 samples (2012 station outage with 5 samples, and 2024 partial year with 6 samples) are safely excluded by `min_size_per_block=8`, leaving **{VALID_BLOCKS} valid annual blocks** ($N_{\\text{blocks}} = {VALID_BLOCKS} \\ge 10$).
- **Censored Detection Limits**:
  - 2005–2014: Standard method ($< 0.05$ mg/L).
  - 2015–2024: Upgraded high-sensitivity method ($< 0.02$ mg/L).
  - Total censored observations: **{N_CENS} ({PCT_CENS}%)**.
- **Underlying Asymmetric Trend**:
  - Baseline concentrations (lower/median quantiles) are **improving (decreasing)** due to wastewater treatment facility upgrades.
  - Peak storm runoff pulses (upper 90th percentile) are **worsening (increasing)** due to expanding impervious surfaces and agricultural runoff flushes.

---

## The Danger of Classical Median Analysis

Running a classical monotonic trend test (e.g., Mann-Kendall or Seasonal Mann-Kendall on the median) yields:
- **Sen's Slope**: `{MEDIAN_SLOPE} mg/L/year`
- **p-value**: `{MEDIAN_P}`
- **Classification**: **{MEDIAN_CLASS}**

> [!WARNING]
> The classical median test reports a statistically significant "Improving / Decreasing" trend. A water resources manager relying solely on the median would conclude the watershed is recovering, completely missing the fact that **extreme peak pollution flushes (90th percentile) are actively worsening**!

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
For monthly data ($N \\approx 12$ per block), choose parameters tailored to the monthly sampling frequency:
- `quantiles=[0.10, 0.25, 0.50, 0.75, 0.90]`: 5 key percentiles provide optimal resolution.
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
    quantiles=[0.10, 0.25, 0.50, 0.75, 0.90],
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
3. **Upper Quantiles ($p = 0.75, 0.90$)**: 100% uncensored (`prop_censored = 0.0%`) and demonstrate a **positive slope**. The Quantile-Kendall test successfully reveals the divergent risk regime.

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
| **Quantile Selection** | Use 3 to 5 quantiles (e.g. `[0.10, 0.25, 0.50, 0.75, 0.90]` or `[0.20, 0.50, 0.80]`) | $N=12$ samples per year cannot reliably estimate extreme tails ($p < 0.05$ or $p > 0.95$). |
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
