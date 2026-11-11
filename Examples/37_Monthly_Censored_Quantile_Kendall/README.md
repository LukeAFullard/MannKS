# Example 37: Monthly Sampled Data with Missing Values & Censoring

## Overview & Real-World Practicality

Environmental, hydrological, and water quality monitoring programs are very commonly conducted on a **monthly sampling schedule** (12 planned observations per year). Long-term monitoring records typically span **10 to 25 years** ($N \approx 120$ to $300$ observations).

In practical applications, these datasets present three major analytical challenges:
1. **Low Within-Block Sample Size ($N \approx 12$)**: Each year contains at most 12 samples, meaning high-density order statistics (e.g. 50 or 100 quantiles) cannot be estimated without extreme extrapolation.
2. **Missing Observations**: Bad weather, frozen streams, equipment downtime, or funding gaps frequently cause missed monthly sampling visits (e.g., 9–11 samples collected instead of 12), and occasionally an entire year may lack sufficient data.
3. **Censored Data & Changing Detection Limits**: Trace contaminants (e.g., dissolved phosphorus, nitrate, pesticides, heavy metals) frequently fall below analytical laboratory detection limits ($< 0.05$ mg/L or $< 0.02$ mg/L). As analytical instruments improve over time, the detection limit often drops, creating an artificial downward trend if not handled rigorously.

`MannKS.quantile_trend_test` natively resolves all three challenges.

---

## Dataset Characteristics

In this example, we evaluate a 20-year dissolved phosphorus record (2005–2024) with realistic monitoring imperfections:
- **Planned Sampling**: 20 years $\times$ 12 months = 240 planned visits.
- **Data Completeness**: ~12% randomly missed monthly visits, leaving 202 collected observations.
- **Incomplete Year Exclusion**: Years with fewer than 8 samples (2012 station outage with 5 samples, and 2024 partial year with 6 samples) are safely excluded by `min_size_per_block=8`, leaving **18 valid annual blocks** ($N_{\text{blocks}} = 18 \ge 10$).
- **Censored Detection Limits**:
  - 2005–2014: Standard method ($< 0.05$ mg/L).
  - 2015–2024: Upgraded high-sensitivity method ($< 0.02$ mg/L).
  - Total censored observations: **24 (11.9%)**.
- **Underlying Asymmetric Trend**:
  - Baseline concentrations (lower/median quantiles) are **improving (decreasing)** due to wastewater treatment facility upgrades.
  - Peak storm runoff pulses (upper 90th percentile) are **worsening (increasing)** due to expanding impervious surfaces and agricultural runoff flushes.

---

## The Danger of Classical Median Analysis

Running a classical monotonic trend test (e.g., Mann-Kendall or Seasonal Mann-Kendall on the median) yields:
- **Sen's Slope**: `-0.00157 mg/L/year`
- **p-value**: `1.6945e-02`
- **Classification**: **Highly Likely Decreasing**

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
For monthly data ($N \approx 12$ per block), choose parameters tailored to the monthly sampling frequency:
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
| **0.10** | -0.00236 | [-0.00332, -0.00135] | 1.0550e-02 | Highly Likely Decreasing | 44.4% | 18 |
| **0.25** | -0.00211 | [-0.00268, -0.00142] | 3.9684e-05 | Highly Likely Decreasing | 16.7% | 18 |
| **0.50** | -0.00202 | [-0.00310, -0.00084] | 6.5203e-04 | Highly Likely Decreasing | 0.0% | 18 |
| **0.75** | +0.01275 | [+0.00683, +0.01948] | 8.5842e-04 | Highly Likely Increasing | 0.0% | 18 |
| **0.90** | +0.01659 | [+0.01163, +0.02137] | 7.8377e-06 | Highly Likely Increasing | 0.0% | 18 |

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
| **Missing Year Spacing** | Supported natively | Mann-Kendall evaluates pairwise time differences $\Delta t = t_j - t_i$; missing years do not bias test statistics. |
| **Censored Data** | `censor_quantile_method='propagate'` | Ensures lower percentiles retain `< DL` flags while upper percentiles remain completely uncensored. |
| **Quantile vs Seasonal MK** | Choose based on hypothesis | Use **Seasonal MK** to ask *"Is January getting wetter?"*; use **Quantile-Kendall** to ask *"Are annual drought baseflows or peak storm flushes intensifying?"* regardless of timing. |
