# Generalized Quantile-Kendall Analysis Guide (MannKS v0.7.0)

## Overview & Motivation

Traditional trend tests (such as the standard Mann-Kendall test or Seasonal Mann-Kendall test) evaluate changes in the **central tendency** (mean or median) of a time series over time. However, in environmental, hydrological, and climatological systems, changes rarely manifest uniformly across all percentiles of the distribution:

* **Hydrological Regimes:** Climate change and water management often produce divergent trends in river flow: high-flow flood peaks may intensify due to extreme rainfall events, while low-flow summer baseflows decline due to extended drought and increased groundwater extraction. A global median test averages these opposing dynamics together, falsely reporting "No Trend".
* **Contaminant Pulses:** Non-point agricultural pollutants (e.g., nitrate or particulate phosphorus) flush into rivers during large runoff events. Consequently, concentrations in upper quantiles (> 75th percentile) may increase sharply while dry-weather baseflow concentrations remain completely flat.
* **Ambient Monitoring Gaps:** Ambient groundwater and river networks often collect sparse, irregular, and censored data (measurements below analytical detection limits). Classical daily order-statistic methods fail when annual sample sizes vary from 8 to 20 visits.
* **Sampling Frequency Drift:** Over multi-decade monitoring programs, sampling frequency frequently changes (e.g., historical monthly sampling upgraded to modern weekly sampling). In extreme order statistics, larger sample sizes naturally span wider percentiles, generating severe artificial trends if not recognized and equalized.

`quantile_trend_test` addresses these challenges by extending the order-statistic trend concept of USGS EGRET (*Hirsch et al., 2015*) into a **frequency-agnostic, censor-aware, multi-block, and drift-protected** Generalized Quantile-Kendall framework.

---

## Relationship to USGS EGRET

The USGS R package **EGRET** (*Exploration and Graphics for RivEr Trends*, Hirsch and De Cicco, 2015; Hirsch et al., 2015) introduced the Quantile-Kendall method (`runQuantileTest` and `plotQuantileKendall`) for daily streamflow records. 

MannKS v0.7.0 was developed to generalize EGRET's pioneering methodology:

| Feature | USGS EGRET (`runQuantileTest`) | MannKS (`quantile_trend_test`) |
| :--- | :--- | :--- |
| **Observation Frequency** | Daily data only ($N=365$ or $366$ days/year). | Any frequency: daily, weekly, monthly, multi-year, or irregular visits. |
| **Blocking Duration** | Strict annual blocks (Water Year, Climate Year, Calendar Year). | Flexible: annual (`'YE'`), multi-year (`'5YE'`), seasonal/quarterly (`'QE'`), or numeric intervals. |
| **Leap-Year Standardization** | Rank reduction dropping or averaging middle ranks (`leapOdd`/`leapEven`). | `'egret'` exact mode OR `'interpolate'` linear quantile mapping OR explicit percentiles (`quantiles=[0.1, 0.5, 0.9]`). |
| **Censored Data** | Uncensored streamflow only. | Native handling of detection limits (`<DL`, `>DL`) with Akritas-Theil-Sen (ATS) slope estimation. |
| **Coverage Filtering** | Implicitly requires continuous daily records. | Explicit `min_coverage` parameter to reject incomplete blocks with temporal gaps. |
| **Sampling Drift Guard** | Not supported. | Built-in `sampling_diagnostic` and `equalize='thin'` / `'aggregate'`. |
| **Field Significance** | Not built-in. | Matrix permutation test evaluating joint significance across all quantiles. |

---

## Mathematical Foundations

### 1. Order-Statistic Blocking

The time domain $T = [t_{\min}, t_{\max}]$ is partitioned into $B$ consecutive, non-overlapping blocks $b_1, b_2, \dots, b_B$ defined by `block_frequency` (e.g., annual `'YE'`, water year `'YS-OCT'`, or multi-year `'5YE'`).

For each block $k \in \{1, \dots, B\}$:
1. All valid observations $(x_{i, k}, t_{i, k})$ within the block are collected.
2. The temporal coverage $C_k$ is computed:
   $$C_k = \min\!\left(1,\; \frac{\max(t_{i, k}) - \min(t_{i, k}) + \Delta t_{\text{med}, k}}{\Delta t_{\text{block}}}\right)$$
   where $\Delta t_{\text{med}, k}$ is the median spacing between distinct observation times in the block (defaulting to 1 day for single-observation datetime blocks). Blocks with coverage $C_k < C_{\min}$ (specified by `min_coverage`) or sample count $N_k < N_{\min}$ (specified by `min_size_per_block`) are rejected as incomplete.
3. The remaining observations are sorted in ascending order to obtain the empirical order statistics:
   $$x_{(1), k} \le x_{(2), k} \le \dots \le x_{(n_k), k}$$

### 2. Length Standardization

Because blocks may contain different numbers of observations $n_k$ (e.g., 365 vs 366 days, or 10 vs 14 irregular visits), the order statistics must be standardized to a common target dimension $M$:

* **Linear Interpolation (`standardization='interpolate'`):**
  Evaluates quantiles at uniform non-exceedance probabilities:
  $$u_j = \frac{j - 1}{M - 1}, \quad j = 1, \dots, M$$
* **EGRET Parity (`standardization='egret'`):**
  Replicates EGRET's exact leap-year adjustment: for 366-day leap years, the middle rank ($j=183$) is dropped (`leapOdd`) or averaged with its neighbor (`leapEven`), maintaining exactly 365 ranks per block.
* **Explicit Quantiles (`quantiles=[q_1, q_2, \dots]`):**
  Directly evaluates user-specified percentiles (e.g., `[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]`).

The resulting standardized values form a $B \times M$ matrix $\mathbf{X}$, where row $k$ contains the standardized quantile profile for block $k$, and column $j$ represents the historical time series of the $j$-th quantile rank.

### 3. Weibull Plotting Positions & Dual Normal Coordinates

For plotting and reporting, each rank $j \in \{1, \dots, M\}$ is mapped to a non-exceedance probability using Weibull plotting positions:
$$p_j = \frac{j}{M + 1}$$
To visualize extreme tails with high resolution, probabilities are mapped to standard normal quantiles (Z-scale):
$$Z_j = \Phi^{-1}(p_j)$$
where $\Phi^{-1}$ is the standard normal quantile function (`scipy.stats.norm.ppf`). When `x_scale='normal'`, the plot displays data on the linear $Z$-scale while labeling primary tick marks with intuitive non-exceedance probabilities (`.001`, `.01`, `.05`, `.10`, `.25`, `.50`, `.75`, `.90`, `.95`, `.99`, `.999`).

### 4. Permutation Field Significance

Testing $M$ individual quantiles simultaneously introduces a multiple testing challenge. While False Discovery Rate (FDR) corrections adjust marginal p-values, quantiles across the same hydrograph are strongly cross-correlated.

When `field_significance=True`, MannKS computes an empirical permutation test of global field significance:
1. Block indices are randomly permuted $P$ times (default 999), holding cross-quantile covariance fixed.
2. In each permutation, the maximum test statistic across all quantiles is computed.
3. The global field p-value ($p_{\text{field}}$) is calculated as the proportion of permuted maxima that exceed the observed maximum statistic.

---

## Step-by-Step Practical Recipes

### Recipe 1: Daily Streamflow with Annual Blocks (EGRET Parity)

```python
import pandas as pd
from MannKS import quantile_trend_test, plot_quantile_trend

# Daily streamflow record
# dates: DatetimeIndex (daily)
# flow: 1D array of discharge values (m3/s)
result = quantile_trend_test(
    x=flow,
    t=dates,
    block_frequency='YE',
    standardization='egret',
    slope_scaling='year',
    x_unit='m3/s',
)

# Plot on USGS EGRET style normal probability scale
plot_quantile_trend(
    result,
    x_scale='normal',
    show_ci=True,
    show_fdc=True,
    save_path='daily_flow_egret_qk.png',
)
```

### Recipe 2: Multi-Year Blocking with Log Transform (%/year)

When working with long-term climate records (e.g., 50 years), multi-year blocks reduce inter-annual weather noise while highlighting decadal hydrological shifts.

```python
# 5-year blocks with log transform to report slope in %/year
result = quantile_trend_test(
    x=flow,
    t=dates,
    block_frequency='5YE',
    quantiles='auto',
    transform='log',
    slope_scaling='year',
    x_unit='m3/s',
)

# High quantiles show flood peak growth (+%/yr), low quantiles show drought drop (-%/yr)
print(result.to_frame()[['quantile', 'plotting_position', 'slope_pct', 'classification']])
```

### Recipe 3: Irregular Ambient Water Quality with Coverage Validation

```python
# Irregular sampling (e.g., 8-15 visits/year)
result = quantile_trend_test(
    x=nitrate,
    t=sample_dates,
    block_frequency='YE',
    quantiles=[0.10, 0.25, 0.50, 0.75, 0.90],
    min_coverage=0.70,         # Require samples to span at least 70% of the year
    min_size_per_block=8,       # Require at least 8 samples per year
    slope_scaling='year',
    x_unit='mg/L',
)
```

### Recipe 4: Censored Contaminant Analysis with ATS Slope

```python
import pandas as pd
from MannKS import quantile_trend_test, prepare_censored_data

# Convert values like ['<0.5', 1.2, 0.8, '<0.5', 2.4]
cens_data = prepare_censored_data(raw_values)

result = quantile_trend_test(
    x=cens_data,
    t=sample_dates,
    block_frequency='YE',
    quantiles=[0.10, 0.25, 0.50, 0.75, 0.90],
    censor_quantile_method='propagate', # Robust ATS estimation across ranks
    slope_scaling='year',
    x_unit='ug/L',
)
```

---

## Comprehensive Parameter Reference

### `quantile_trend_test` Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `x` | array-like or DataFrame | *Required* | Observed values. For censored data, provide a DataFrame with `value`, `censored`, and `cen_type` (from `prepare_censored_data`). |
| `t` | array-like or DatetimeIndex | *Required* | Timestamps or numeric time coordinates corresponding to `x`. |
| `block_frequency` | str or float | `'YE'` | Duration of time blocks. For datetime `t`, use pandas frequency strings (e.g. `'YE'`, `'5YE'`, `'QE'`). For numeric `t`, specify block width (e.g. `1.0`). |
| `block_origin` | str, Timestamp, or float | `None` | Fixed anchor for block boundaries. If `None`, anchored at data start or year boundary. |
| `quantiles` | int, sequence, or str | `'auto'` | Quantile configuration. `'auto'` uses the standardized block length (`target_length`). An integer specifies $M$ evenly spaced quantiles. A sequence of floats (e.g. `[0.1, 0.5, 0.9]`) specifies explicit non-exceedance probabilities in $[0, 1]$. |
| `target_length` | str or int | `'min'` | Target standardized sample size $M$ when `quantiles='auto'`: `'min'`, `'median'`, `'max'`, or an explicit integer $\ge 2$. |
| `standardization` | str | `'interpolate'` | Method for standardizing block lengths: `'interpolate'` (linear quantile interpolation) or `'egret'` (exact leap-year rank reduction). |
| `min_coverage` | float | `0.9` | Minimum fraction of block duration $[t_{\min}, t_{\max}]$ that must be spanned by observations (0.0 to 1.0). Incomplete blocks below this threshold are excluded. |
| `min_size_per_block` | int | `10` | Minimum number of observations required to retain a block. |
| `min_blocks` | int | `5` | Minimum number of valid blocks required to perform trend analysis. |
| `equalize` | str or None | `'none' or None` | Subsampling strategy to prevent sampling drift artifacts: `'none'` or `None` (raw), `'thin'` (systematic temporal subsampling to uniform size), or `'aggregate'` (robust median within sub-intervals). |
| `transform` | str or None | `'none' or None` | Transformation applied before trend estimation: `'none'` or `None` (raw), `'log'`, or `'pct_of_median'`. If `'log'`, Sen's slope is exponentiated and reported as percentage change per time unit (`slope_pct`). |
| `censor_quantile_method` | str | `'propagate'` | Handling of censored observations during quantile standardization: `'propagate'` (preserves censoring flags for ATS), `'substitute'` (replaces with multiplier), or `'error'`. |
| `hicensor` | bool or float | `False` | Recensors all observations to the highest detection limit to eliminate false trends from improving laboratory precision. |
| `field_significance` | bool | `False` | Computes global permutation field significance across all quantiles. |
| `n_permutations` | int | `999` | Number of block permutations for field significance testing. |
| `slope_scaling` | str or None | `'year'` | Time unit for slope scaling: `'year'`, `'month'`, `'week'`, `'day'`, `'hour'`, `'minute'`, `'second'`. Pass `None` for unscaled raw units. |
| `x_unit` | str or None | `'units'` | Measurement unit for reporting (e.g. `'mg/L'`, `'m3/s'`). |
| `plot_path` | str or None | `None` | File path to automatically save the diagnostic quantile trend plot. |

---

### `plot_quantile_trend` Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `result` | QuantileTrendResult | *Required* | Output object returned by `quantile_trend_test`. |
| `save_path` | str, Path, or None | `None` | Path to save figure. If `None` and `ax` is `None`, no figure is created and `None` is returned. |
| `x_scale` | str | `'normal'` | X-axis coordinate scaling: `'normal'` (probit Z-scale with probability tick labels `.001` to `.999`) or `'linear'` (uniform $[0, 1]$). |
| `color_by` | str | `'p'` | Point coloring mode: `'p'` (significance bands $p < \alpha$, $\alpha \le p < 2\alpha$, $p \ge 2\alpha$) or `'classification'` (blue for increasing, orange/red for decreasing, grey for no trend). |
| `show_ci` | bool | `False` | Displays vertical error whiskers (confidence intervals) for each quantile. |
| `show_fdc` | bool | `False` | Overlays the median Flow Duration Curve (FDC) on a secondary right-hand y-axis. |
| `y_lim` | tuple | `None` | Manual (ymin, ymax) limits for the primary trend slope axis. |
| `legend_loc` | str or None | `'upper left'` | Matplotlib legend location. Pass `None` to omit legend. |
| `title` | str or None | `None` | Custom figure title. |
| `figsize` | tuple | `(10, 6)` | Figure dimensions in inches. |
| `ax` | plt.Axes or None | `None` | Matplotlib Axes object to draw onto. |
| `dpi` | int | `150` | Resolution in dots per inch for saved figure files. |

---

## Output Object Schema: `QuantileTrendResult`

`quantile_trend_test` returns a `QuantileTrendResult` object containing comprehensive analysis outputs:

* **`results` (DataFrame):**
  Main trend results table with one row per quantile rank:
  * `rank`: Integer index from $1$ to $M$.
  * `quantile`: Nominal percentile in $[0, 1]$.
  * `plotting_position`: Weibull non-exceedance probability $p = j / (M + 1)$.
  * `z_position`: Standard normal quantile $Z = \Phi^{-1}(p)$.
  * `slope`: Sen's slope in scaled units (e.g. $\text{mg/L/year}$).
  * `lower_ci`, `upper_ci`: $(1 - \alpha)$ confidence bounds for the slope.
  * `slope_pct`, `lower_ci_pct`, `upper_ci_pct`: Percentage change per unit time (exponentiated when `transform='log'`, or relative to each quantile's median when `transform='none'`/`'pct_of_median'`).
  * `p`: Mann-Kendall two-sided p-value.
  * `h`: Boolean significance flag ($p < \alpha$).
  * `Tau`: Kendall's $\tau$ correlation coefficient.
  * `classification`: Categorical trend description (e.g. `"Highly Likely Increasing"`).
  * `prop_censored`: Fraction of blocks where this rank is censored.

* **`blocks` (DataFrame):**
  Metadata for each evaluated time block: `block_start`, `block_end`, `block_center`, `n_obs`, `n_censored`, `coverage`, `valid`, and `reason`.

* **`matrix` (DataFrame):**
  Standardized values of dimension $B \text{ blocks} \times M \text{ ranks}$.

* **`sampling_diagnostic` (dict):**
  Diagnostic metrics testing for sampling frequency drift:
  * `flag`: Boolean indicator (`True` if $p < 0.10$ or $N_{\max} / N_{\min} > 1.5$).
  * `tau_n_vs_time`: Kendall's $\tau$ between block time and sample size.
  * `p_n_vs_time`: Significance of sample size change over time.
  * `n_min`, `n_max`: Minimum and maximum observation count per block.
  * `ratio_max_min`: Ratio $N_{\max} / N_{\min}$.

* **`field_significance` (dict or None):**
  Results of permutation field significance testing when `field_significance=True`: `n_significant`, `p_count`, `max_abs_z`, `p_max_z`, and `n_permutations`.

### Convenience Methods

```python
# Extract results as standard DataFrame
df = result.to_frame()

# Filter for statistically significant quantiles
sig_df = result.significant(alpha=0.05)

# Extract time series across blocks for a specific quantile (e.g., median rank or 90th percentile)
series_q90 = result.get_quantile_series(0.90)

# Summarize low (0-25%), mid (25-75%), and high (75-100%) regimes
from MannKS import summarize_quantile_regimes
regime_summary = summarize_quantile_regimes(result)
```

---

## Pitfalls & Best Practices

### 1. The Sampling Frequency Drift Trap
When monitoring programs expand from monthly to weekly sampling over time, the extreme tails of the empirical distribution widen purely as a sample-size artifact:
* Expected sample maximum increases with sample size.
* Expected sample minimum decreases with sample size.

**Remedy:** Always inspect `result.sampling_diagnostic['flag']`. If flagged, re-run with `equalize='thin'`, which systematically sub-samples blocks in time to a uniform sample size, restoring true flat trends across stationary distributions.

### 2. Partial Block Bias & Temporal Coverage
Blocks near the start or end of a record often contain only partial years (e.g., monitoring began in September). Evaluating an incomplete year whose samples represent only summer or winter will severely bias the annual order statistics.

**Remedy:** Set `min_coverage=0.70` (or `0.80`) to automatically reject blocks that do not span at least 70–80% of the block window.

### 3. Censor Handling & Artificial Plateaus
Naive substitution of non-detects (e.g., replacing `< 1.0` with `0.5`) before sorting creates flat artificial plateaus in the lower ranks across blocks.

**Remedy:** Use `censor_quantile_method='propagate'` (the default). It tracks the censor flags through the order statistics and passes the resulting series to the Akritas-Theil-Sen (ATS) estimator, preserving valid interval bounds without creating false plateaus.

### 4. Zero-Flow and Dry-Bed Ties
In ephemeral or arid streams, the hydrograph may record zero discharge for 30–60% of the year. All lower quantiles will equal exactly zero in every block.

**Remedy:** When a quantile series is constant ($S = 0$, $\operatorname{Var}(S) = 0$), MannKS safely identifies the invariant rank, reports `slope = 0.0`, `p = 1.0`, and assigns `'As Likely as Not Indeterminate'` (or `'No Trend'` when `continuous_confidence=False`) without numerical exceptions.

### 5. Analyzing Monthly Sampled Datasets (10–25 Years with Censoring and Missing Values)
A very common practical configuration in water quality and environmental monitoring is **monthly sampling** over 10 to 25 years (120 to 300 observations). In these datasets:
- **Sample Size per Block**: With $N \approx 12$ samples per year, high-density order statistics cannot be estimated reliably. Select **3 to 5 quantiles** (e.g., quartiles `[0.25, 0.50, 0.75]` or quintiles `[0.10, 0.25, 0.50, 0.75, 0.90]`).
- **Missing Visits**: Weather, frozen rivers, or equipment downtime often result in 8 to 11 samples per year instead of 12. Set `min_size_per_block=8` (or 9) to preserve years with a few missed visits while dropping severely incomplete years.
- **Missing Annual Blocks**: If an entire year is skipped due to station outage, Mann-Kendall handles unequally spaced time differences ($\Delta t = t_j - t_i$) between annual block centers natively without bias.
- **Censored Detection Limits**: Use `prepare_censored_data` to parse detection limits (e.g. `< 0.05 mg/L`), `censor_quantile_method='propagate'` to tag lower ranks without artificial plateaus, and `sens_slope_method='ats'` for robust Akritas-Theil-Sen slope estimation.
- **Quantile-Kendall vs Seasonal Mann-Kendall**: Use **Seasonal MK** (`seasonal_trend_test`) when testing whether specific calendar months are changing (e.g., *"Is July getting drier?"*). Use **Quantile-Kendall** (`quantile_trend_test`) when testing whether the distribution extremes are shifting (e.g., *"Are annual drought baseflows declining or peak storm runoff pulses intensifying?"*), regardless of which calendar month the extremes fall into.

See [Example 37](../37_Monthly_Censored_Quantile_Kendall/README.md) for a complete, executable demonstration.

---

## Limitations & Assumptions

1. **Minimum Number of Blocks ($B$):** The Mann-Kendall test is conducted *across blocks*, not across raw observations. With fewer than $10$ valid blocks ($B < 10$), statistical power is low and normal-approximation p-values are coarse (e.g., with $B=5$, the minimum possible two-sided p-value is $0.083$, so $p < 0.05$ is unattainable).
2. **Cross-Quantile Dependence & Multiple Testing:** Adjacent quantiles within the same block are strongly correlated. Individual quantile p-values are marginal (unadjusted); use `field_significance=True` when testing a global null hypothesis across the entire quantile spectrum.
3. **Field Significance Assumptions:** `field_significance=True` permutes entire block rows of the standardized quantile matrix, which assumes blocks are exchangeable under $H_0$ (no strong inter-block autocorrelation) and evaluates numeric standardized values without censored-interval scoring.
4. **Log Transformation Requires Positive Values:** `transform='log'` requires all standardized block values to be strictly $> 0$ (after adding `log_offset` if provided). Zero-flow or negative-value series must use `transform='none'` (or a positive `log_offset`).
5. **Linear Interpolation & Tail Extrapolation:** When `quantiles` requests extreme percentiles (e.g., `0.99`) on a small block ($N = 10$–$12$), `standardization='interpolate'` cannot extrapolate beyond the observed sample minimum and maximum—it reflects the block's largest/smallest order statistics.
6. **Temporal Coverage with Cross-Year Seasons:** `min_coverage` measures the span from first to last observation in a block. If `months` spans a calendar-year boundary (e.g., `[12, 1, 2]` with `block_frequency='YE'`), observations in January and December of the same calendar year span the full year and inflate coverage; use a shifted block origin such as `'YE-NOV'` (blocks starting Dec 1) for cross-year seasons.
