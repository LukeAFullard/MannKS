# Statistical Methodology: Generalized Quantile-Kendall Analysis

The `MannKS` package implements **Generalized Quantile-Kendall trend analysis** via [`quantile_trend_test`](MannKS/quantile_trend_test.py), extending the Quantile-Kendall method introduced in the USGS R package **EGRET** (Hirsch & De Cicco, 2015; Hirsch et al., 2015) from uncensored daily streamflow records to **arbitrary sampling frequencies, flexible block durations, censored data, and datasets with changing sampling density**.

This document explains **what** Quantile-Kendall analysis is, **why** it is used, **how** it works step-by-step, and **what every option does**, including assumptions and limitations.

---

## 1. What Is Quantile-Kendall Analysis?

Standard trend tests (`trend_test`, `seasonal_trend_test`) estimate a single overall trend—essentially asking whether the **central tendency** (median) of a series is increasing or decreasing over time.

**Quantile-Kendall analysis** asks a richer question:

> *How is the entire probability distribution of the variable changing over time—from the lowest baseflows or baseline concentrations up to the highest flood peaks or storm flushes?*

Instead of testing one overall trend, Quantile-Kendall:
1. Splits the time series into **consecutive, non-overlapping time blocks** (e.g., calendar years, water years, or multi-year periods).
2. Computes the **empirical quantiles / order statistics** within each block (e.g., from the 5th percentile up to the 95th percentile, or all $M$ ranks).
3. Forms a separate time series across blocks for **each quantile level** (e.g., all annual 10th percentiles over time, all annual 50th percentiles over time, all annual 95th percentiles over time).
4. Runs a **Mann-Kendall test** and **Sen's slope / Akritas-Theil-Sen (ATS) estimator** independently on each quantile series.

---

## 2. Why Use It (and When Not To)?

Environmental, hydrological, and water quality trends are often **non-uniform across the distribution**:

* **Hydrological extremes vs. baseflow:** High-flow storm peaks may intensify while dry-season low flows decline. A global median test averages opposing tail trends and can report "No Trend".
* **Episodic pollution flushes:** Non-point agricultural runoff (e.g., nitrate, suspended sediment, phosphorus, *E. coli*) may worsen sharply during wet-weather events (upper quantiles $> 75\text{th}$ percentile) while dry-weather baseflow (lower/median quantiles) improves or stays flat.
* **Regulatory extremes:** Water quality frameworks (such as New Zealand's NPS-FM 92nd/95th percentile attributes) specifically target upper-tail compliance.

### Quantile-Kendall vs. Seasonal Mann-Kendall

| Question you want to answer | Recommended test |
| :--- | :--- |
| *"Are the highest/lowest values changing over time, regardless of which month they occur in?"* | **`quantile_trend_test`** |
| *"Is a specific season (e.g. January or Winter) getting higher or lower over time?"* | **`seasonal_trend_test`** |
| *"Is there a single monotonic shift in the overall median?"* | **`trend_test`** |

---

## 3. Step-by-Step Method

### Step 1: Data Preparation & High-Censoring (`_prepare_data`)
Input values `x` (numeric array or DataFrame from `prepare_censored_data`) and timestamps `t` (datetime-like or numeric) are validated, non-finite values are dropped, and observations are sorted chronologically.

* **`hicensor`** (default `False`): If `True`, finds the highest left-censoring limit (`<DL`) in the dataset and recensors every observation below that limit to `<max_DL`. If a numeric threshold is given, recensors below `min(natural_max_lt, hicensor)` (or `hicensor` if no left-censored values exist). This prevents artificial downward trends caused purely by improving laboratory detection limits over time.

### Step 2: Non-Overlapping Temporal Blocking (`block_frequency`, `block_origin`, `months`)
The time domain is partitioned into **disjoint, consecutive half-open intervals** $[b_{\text{start}}, b_{\text{end}})$. Each observation belongs to at most one block—**blocks never overlap**.

* **`block_frequency`** (default `'YE'`):
  * For datetime `t`: any pandas offset string, such as `'YE'` (calendar year), `'YS-OCT'` or `'YE-SEP'` (Oct–Sep water year), `'YS-JUL'` / `'YE-JUN'` (Jul–Jun hydrological year), `'5YE'` (5-year blocks), `'QE'` (quarterly), or `'ME'` (monthly).
  * For numeric `t`: a positive float specifying block width (e.g. `1.0`, `5.0`).
* **`block_origin`** (default `None`): Fixed anchor point for block boundaries. If `None`, boundaries align to calendar period starts (e.g., Jan 1 for `'YE'`, Oct 1 for `'YE-SEP'`) or multiples of width for numeric `t`.
* **`months`** (default `None`): Optional sequence of calendar months (`1`–`12`) to retain before blocking (e.g., `months=[12, 1, 2]` or `[6, 7, 8]`).
  * *Cross-year caveat:* If `months` spans a calendar-year boundary (e.g. `[12, 1, 2]`), do **not** use `'YE'` (Jan–Dec), as it will combine Jan–Feb of year $Y$ with Dec of year $Y$ and inflate temporal coverage. Instead, align the block start using `'YE-NOV'` (blocks run Dec 1–Nov 30).

### Step 3: Block Completeness & Coverage Filtering
Empty boundary blocks at the very start/end of the record are ignored. Every remaining block $k$ must pass up to three checks to be retained:

1. **`min_size_per_block`** (default `10`): Minimum number of valid observations in the block ($n_k \ge N_{\min}$).
2. **`min_coverage`** (default `0.9`): Minimum fraction of the nominal block duration spanned by the block's observations:
   $$C_k = \min\!\left(1,\; \frac{(t_{\max, k} - t_{\min, k}) + \Delta t_{\text{med}, k}}{\Delta t_{\text{block}, k}}\right)$$
   where $\Delta t_{\text{med}, k}$ is the median spacing between distinct observation times in block $k$ (defaulting to 1 day for single-observation datetime blocks, or $0$ for numeric singletons). This ensures, for example, that a partial monitoring year beginning in October is not treated as representative of a full annual distribution.
   * For monthly or irregular ambient sampling where 1–4 visits may be missed, consider lowering `min_coverage` (e.g. `0.70`) and `min_size_per_block` (e.g. `8`).
3. **`require_complete`** (default `False`): Stricter check requiring $n_k \ge \mathrm{round}(\Delta t_{\text{block}, k} / \Delta t_{\text{global}})$, where $\Delta t_{\text{global}}$ is the global median time step. Useful for strict EGRET-style complete daily records.

The total number of surviving blocks $B$ must satisfy $B \ge \max(2, \text{min\_blocks})$ (`min_blocks` defaults to `5`). If fewer blocks survive, `computation_mode='insufficient'` is returned with an empty `results` table and a `UserWarning`.

### Step 4: Sampling Frequency Drift Diagnostic & Equalization (`equalize`)
**Why this matters:** In order statistics, sample extremes depend on sample size $n_k$. For a completely stationary distribution, the expected sample maximum of $n=52$ weekly draws is strictly larger than that of $n=12$ monthly draws, and the expected minimum is strictly smaller. If a monitoring program intensifies over time (e.g., monthly in decade 1, weekly in decade 2), a naive quantile analysis will produce **spurious upward trends in upper quantiles and spurious downward trends in lower quantiles**.

Before standardizing blocks, `quantile_trend_test` computes `sampling_diagnostic`:
* `tau_n_vs_time`, `p_n_vs_time`: Kendall's $\tau$ and two-sided $p$-value between `block_center` and block sample size $n_k$.
* `n_min`, `n_max`, `ratio_max_min`: Minimum/maximum block counts and their ratio $n_{\max} / n_{\min}$.
* `flag`: Set to `True` if `p_n_vs_time < 0.10` or `ratio_max_min > 1.5`.

If `flag` is `True` and `equalize='none'`, a `UserWarning` is emitted.

**`equalize` options:**
* **`'none'`** (or `None`, default): Uses all retained observations in each block as-is. Appropriate when sampling density is roughly stable across blocks, or when differences are minor and you are evaluating interior quantiles (`0.25`–`0.75`).
* **`'thin'`**: **Systematic temporal thinning.** Every valid block is downsampled to $n_{\min} = \min_k n_k$ observations by selecting evenly spaced indices in chronological order (`np.round(np.linspace(0, n_k - 1, n_min))`). This preserves seasonal coverage across the block while eliminating sample-size bias in the order statistics—restoring calibrated Type I error rates when sampling frequency shifts.
  * *Trade-off:* All blocks are thinned down to the smallest valid block's count ($n_{\min}$). If one valid block has unusually few observations (e.g., 10 vs. 52 elsewhere), consider increasing `min_size_per_block` so that one sparse year doesn't force heavy data loss in all other years.
* **`'aggregate'`**: **Within-block sub-interval aggregation.** Reduces higher-frequency observations to a uniform sub-period cadence using robust medians (`_aggregate_by_group(..., 'robust_median')`):
  * For datetime `t`, the sub-interval grain adapts to the median observed block span: calendar months when median block span $> 45$ days (yielding $\le 12$ monthly medians per annual block), days when $> 2$ days, hours when $> 2$ hours, and minutes otherwise (falling back to 12 equal bins if a block collapses into a single sub-interval).
  * For numeric `t`: divides the block's observed span into 12 equal-width sub-intervals and computes the robust median per populated bin (`13` bins if the right endpoint falls on `t_max`).
  * *Trade-off:* Smooths intra-interval short-term correlation and equalizes cadence, but dampens extreme single-day peaks within each sub-interval.

### Step 5: Sorting, Censor Propagation, and Block Standardization
Observations within each block are sorted in ascending order. When censored data are present, interval endpoints are constructed with a small tie-breaking nudge (`make_intervals(..., nudge=True)`) so `<DL` sorts just below uncensored `DL`, and `>DL` sorts just above uncensored `DL`.

Blocks are standardized to a common target length $M$ via `quantiles`, `target_length`, `standardization`, and `quantile_method`:

* **`quantiles`** (default `'auto'`):
  * `'auto'`: Determines $M$ from `target_length` (`'min'` default, `'median'`, `'max'`, or explicit int $\ge 2$) and evaluates on `u_grid = np.linspace(0.0, 1.0, M)`. Plotting positions use Weibull ranks $p_j = j / (M + 1)$ for $j = 1, \dots, M$.
  * `int` ($M \ge 1$): Evaluates $M$ interior quantiles at Weibull probabilities $u_j = p_j = j / (M + 1)$ (requires `standardization='interpolate'`).
  * Sequence of floats in $[0, 1]$ (e.g., `[0.10, 0.25, 0.50, 0.75, 0.90, 0.95]`): Evaluates those exact percentiles in sorted order (requires `standardization='interpolate'`). For $Z$-scale plotting, `plotting_position` is clipped to $[1/(n_{\text{med}}+1),\, n_{\text{med}}/(n_{\text{med}}+1)]$ so `0.0` or `1.0` does not produce $\pm\infty$.
* **`standardization`** (default `'interpolate'`):
  * **`'interpolate'`**: Computes `np.quantile(sorted_vals, u_grid, method=quantile_method)`. Works with any block size $n_k \ge 2$ and any `quantiles` specification.
  * **`'egret'`**: Exact USGS EGRET rank-reduction mode (`quantiles='auto'` and `target_length='min'`—or integer matching `min(block_sizes)`—only). Requires every valid block to have size $M$ or $M+1$ (typically 365 and 366 for complete daily years). Blocks of size $M+1$ are reduced by dropping the middle rank if $n_k$ is odd (`leapOdd`) or averaging the two middle ranks if $n_k$ is even (`leapEven`). Raises `ValueError` if any block's size differs from $M$ or $M+1$.
* **`quantile_method`** (default `'linear'`): Passed to `numpy.quantile` (e.g., `'linear'`, `'weibull'`, `'hazen'`, `'median_unbiased'`).
* **`censor_quantile_method`** (default `'propagate'`):
  * **`'propagate'`**: For `'interpolate'`, maps each quantile probability $u$ to its bracketing order-statistic indices $\lfloor u(n_k - 1) \rfloor$ and $\lceil u(n_k - 1) \rceil$. If either bracketing observation is censored, the standardized quantile value is flagged as censored (`'lt'` or `'gt'`, with `'lt'` taking precedence if both occur). For `'egret'`, censoring flags travel directly with the retained/merged order statistics. When combined with `transform='log'` and censored data, `sens_slope_method='ats'` is required.
  * **`'substitute'`**: Replaces `<DL` with `lt_mult * DL` (default `0.5 * DL`) and `>DL` with `gt_mult * DL` (default `1.1 * DL`) before sorting, clearing all censoring flags. *Note:* This can create artificial flat ties in lower quantiles if detection limits are constant.
  * **`'error'`**: Raises `ValueError` if any censored observation is encountered.

### Step 6: Optional Value Transformation (`transform`, `log_offset`)
Applied to the $B \times M$ standardized matrix prior to trend estimation:

* **`'none'`** (or `None`, default): No transformation. `slope` is in `x_unit` per `slope_scaling` unit; `slope_pct` is computed relative to each quantile's across-block median: $100 \times \text{slope} / \text{median}_j$.
* **`'log'`**: Natural logarithm $\ln(x + \text{log\_offset})$. All standardized values (after `log_offset` if supplied) must be strictly $> 0$.
  * `slope`, `lower_ci`, and `upper_ci` remain in natural-log units per time unit ($\ln(x) / \text{time}$), while `slope_pct` (and its CI columns `lower_ci_pct`, `upper_ci_pct`) exponentiate Sen's slope $b$ into compound percentage change per time unit:
    $$\text{slope\_pct} = 100 \times (e^b - 1)$$
  * `slope_units` is labeled `% per <time_unit>` (used as the primary y-axis on `plot_quantile_trend`). If any standardized quantile is censored under `censor_quantile_method='propagate'`, `sens_slope_method='ats'` is required (`'lwp'`/`'drop_ambiguous'` heuristic multipliers are invalid on negative log-values and raise `ValueError`).
  * This reproduces USGS EGRET's `% per year` trend reporting and makes slopes comparable across vastly different flow magnitudes (e.g., baseflow vs. peak flood).
* **`'pct_of_median'`**: Expresses each quantile column as a percentage of its across-block median ($100 \times x_{k,j} / \text{median}_j$). Columns whose median is $\le 0$ are set to `NaN` with an analysis note.

### Step 7: Per-Quantile Trend Estimation (`engine`, `**mk_kwargs`)
For each quantile column $j \in \{1, \dots, M\}$, a Mann-Kendall trend test is run against the block centers `block_centers`:

* **`engine='trend_test'`** (default): Calls [`MannKS.trend_test`](MannKS/trend_test.py) column-by-column. Supports the full MannKS feature set: censored interval scoring, Akritas-Theil-Sen (`sens_slope_method='ats'`), autocorrelation corrections (`autocorr_method='auto'`, `'yue_wang'`, `'block_bootstrap'`), `ci_method='direct'` or `'lwp'`, `tau_method='a'` or `'b'`, and `continuous_confidence` classifications.
* **`engine='fast'`**: Vectorized matrix implementation ([`_fast_uncensored_matrix_trends`](MannKS/quantile_trend_test.py)) that computes $S$, $\mathrm{Var}(S)$ (with tie correction), Kendall's $\tau$ (`tau_method='a'` or `'b'`), $Z$, $p$-value, pairwise Sen's slopes, and confidence intervals (`ci_method='direct'` or `'lwp'`) across all $M$ quantiles simultaneously. Note that for uncensored data, `sens_slope_method='ats'` reduces identically to standard pairwise Sen's slope, so it remains compatible with `engine='fast'`. Ideal for large $M$ (e.g., daily data with $M=365$ or $1826$).
  * *Automatic fallback:* If censored data, `NaN`s, `autocorr_method != 'none'`, `mk_test_method == 'lwp'`, `tie_break_method == 'lwp'`, `large_dataset_mode in ('fast', 'aggregate')` (or `'auto'` with $> 5000$ blocks), or `max_pairs is not None` are requested, `quantile_trend_test` automatically falls back to `engine='trend_test'` so results remain exact.

### Step 8: Simultaneous Field Significance (`field_significance`)
Testing $M$ quantiles at once creates a multiple-testing problem: if $M=20$ and $\alpha=0.05$, at least one false positive is likely by chance even under $H_0$. Standard Bonferroni or FDR corrections ignore the strong cross-correlation between adjacent order statistics within each block.

When **`field_significance=True`**, MannKS runs a **block-row permutation test** (`n_permutations=999` by default, reproducible via `random_state`):
1. Randomly permutes the row order (time assignment) of the $B \times M$ matrix $P$ times. Permuting entire rows preserves the exact cross-quantile correlation structure inside every block while destroying any monotonic time trend.
2. Recomputes the Mann-Kendall $Z$-statistics across all $M$ columns for each permutation.
3. Reports two global permutation $p$-values (using $(1 + \text{exceedances}) / (P + 1)$):
   * **`p_count`**: Probability under $H_0$ of observing $\ge$ `n_significant` quantiles with $|Z| > z_{1-\alpha/2}$.
   * **`p_max_z`**: Probability under $H_0$ of observing $\max_j |Z_j| \ge \text{max\_abs\_z}$ (Westfall-Young max-$T$ global test).

---

## 4. Complete Options Reference

| Parameter | Default | Options / Type | Purpose & Guidance |
| :--- | :--- | :--- | :--- |
| `x`, `t` | *Required* | array-like / DataFrame | Observations (`prepare_censored_data` output for censored data) and corresponding datetime or numeric times. |
| `block_frequency` | `'YE'` | pandas offset str or `float > 0` | Temporal block duration (`'YE'`, `'YS-OCT'`, `'5YE'`, `'QE'`, `'ME'`, or numeric width). |
| `block_origin` | `None` | datetime, float, or `None` | Explicit anchor for block edges. Defaults to calendar boundary or floored numeric origin. |
| `months` | `None` | sequence of `1..12` or `None` | Filter to specific calendar months before blocking (datetime `t` only). |
| `quantiles` | `'auto'` | `'auto'`, `int`, or sequence in `[0, 1]` | `'auto'` builds $M$ ranks from `target_length`; `int` evaluates $M$ Weibull quantiles $j/(M+1)$; sequence evaluates explicit probabilities. |
| `target_length` | `'min'` | `'min'`, `'median'`, `'max'`, or `int >= 2` | Target grid size $M$ when `quantiles='auto'`. `'egret'` requires `'min'` (or matching integer). |
| `standardization` | `'interpolate'` | `'interpolate'`, `'egret'` | `'interpolate'` uses `np.quantile`; `'egret'` matches USGS EGRET `leapOdd`/`leapEven` (requires `quantiles='auto'`, `target_length='min'`, and $n_k \in \{M, M+1\}$). |
| `quantile_method` | `'linear'` | `numpy.quantile` method str | Interpolation rule (`'linear'`, `'weibull'`, `'hazen'`, etc.) when `standardization='interpolate'`. |
| `min_size_per_block` | `10` | `int >= 1` | Minimum observations needed to keep a block (set to `8` or `9` for monthly data with occasional missed visits). |
| `min_blocks` | `5` | `int >= 1` | Minimum valid blocks required to run the test ($< 10$ produces coarse p-values and low power). |
| `min_coverage` | `0.9` | `float` in `[0, 1]` | Minimum fraction of block duration spanned by samples; set to `0.70`–`0.80` for irregular ambient sampling. |
| `require_complete` | `False` | `bool` | Rejects blocks with fewer observations than expected from the global median sampling step. |
| `equalize` | `'none'` | `'none'`, `None`, `'thin'`, `'aggregate'` | Controls sample-size equalization across blocks to eliminate sampling-frequency drift bias. (`result.blocks['n_obs']` logs pre-equalization counts; `result.sampling_diagnostic` is evaluated post-equalization on standardized block counts.) |
| `censor_quantile_method` | `'propagate'` | `'propagate'`, `'substitute'`, `'error'` | How detection limits are carried through order statistics (`'propagate'` recommended for censored data). |
| `hicensor` | `False` | `bool` or `float` | Recensors values below the highest `<DL` threshold before blocking to prevent false trends from improving detection limits. |
| `transform` | `'none'` | `'none'`, `None`, `'log'`, `'pct_of_median'` | Value transform prior to testing (`'log'` keeps `slope` in $\ln$ units and sets `slope_pct` via $100(e^b - 1)$). |
| `log_offset` | `None` | `float` or `None` | Constant added before `np.log` when `transform='log'` and zeros/non-positives occur. |
| `alpha` | `0.05` | `float` in `(0, 1)` | Significance level for `h`, confidence intervals, and field significance. |
| `continuous_confidence` | `True` | `bool` | Computes confidence $C$, $C_d$, and LWP/IPCC qualitative `classification` labels. |
| `category_map` | `None` | `dict` or `None` | Custom threshold mapping passed to `classify_trend`. |
| `slope_scaling` | `'year'` | `'year'`, `'month'`, `'week'`, `'day'`, `'hour'`, `'minute'`, `'second'`, `None` | Time unit for `slope`, `lower_ci`, `upper_ci`, and `slope_pct` (datetime `t` only). |
| `x_unit` | `'units'` | `str` or `None` | Variable unit label for reporting and plot axes (`None` defaults to `'units'`). |
| `engine` | `'trend_test'` | `'trend_test'`, `'fast'` | `'fast'` accelerates large uncensored grids (automatically falls back when needed). |
| `field_significance` | `False` | `bool` | Runs row-permutation global test across all quantiles. |
| `n_permutations` | `999` | `int >= 1` | Number of permutations when `field_significance=True`. |
| `max_quantiles` | `2000` | `int >= 1` | Safety cap on the number of quantiles $M$. |
| `random_state` | `None` | `int`, `Generator`, or `None` | RNG seed for `field_significance` permutations and `block_bootstrap` kwargs. |
| `plot_path` | `None` | `str`, `Path`, or `None` | Optional file path to auto-save `plot_quantile_trend(res)`. |
| `**mk_kwargs` | — | dict | Forwarded to `trend_test`: `sens_slope_method`, `tau_method`, `mk_test_method`, `ci_method`, `tie_break_method`, `lt_mult`, `gt_mult`, `autocorr_method`, `block_size`, `n_bootstrap`, `large_dataset_mode`, `max_pairs`. |

---

## 5. Visualization & Post-Processing Helpers

* **`plot_quantile_trend(result, ...)`** (aliased as `plot_quantile_kendall`):
  * `x_scale='normal'` (default): Plots quantiles on a standard normal probit $Z$-axis ($Z = \Phi^{-1}(p)$) with probability tick labels (`.001` to `.999`) and a secondary top $Z$-axis, expanding visual resolution in the tails (USGS EGRET style). Use `x_scale='linear'` for uniform $[0, 1]$ spacing (often clearer when plotting a small set of explicit quantiles).
  * `color_by='p'` (default): Highlights $p < \alpha$ (orange-red), $\alpha \le p < 2\alpha$ (dark charcoal), and $p \ge 2\alpha$ (gray). Use `color_by='classification'` to color by Increasing/Decreasing/No Trend.
  * `show_ci=True`: Plots $(1-\alpha)$ confidence interval whiskers per quantile.
  * `show_fdc=True`: Overlays the across-block median empirical distribution (Flow / Concentration Duration Curve) on a secondary right y-axis.
* **`summarize_quantile_regimes(result, regimes=None)`**: Summarizes median slopes, proportion significant, and modal classification across quantile bands (defaults to `'low': (0.0, 0.25)`, `'mid': (0.25, 0.75)`, `'high': (0.75, 1.0)`).
* **`result.significant(alpha=None)`**: Filters `result.results` to rows with `p < alpha`.
* **`result.get_quantile_series(quantile_or_rank)`**: Extracts the block-level time series (including censoring flags) for a specific integer rank (`>= 1`) or nearest quantile probability (`< 1`).

---

## 6. Statistical Limitations & Assumptions

1. **Statistical Power Depends on Block Count ($B$), Not Raw Sample Count ($N$):** Because Mann-Kendall is applied across the $B$ blocks, having 365 daily observations per year for 6 years still yields only $B = 6$ points per quantile test (minimum achievable two-sided $p$-value with $B=5$ is $0.083$; with $B=6$ it is $0.024$, and only if strictly monotonic). Aim for **$B \ge 10$** blocks (and ideally $20+$ for noisy tail quantiles).
2. **Tail Volatility in Small Blocks (e.g., Monthly Data):** With $N \approx 12$ samples/year, `quantile=0.95` under linear interpolation is $0.55\,x_{(11)} + 0.45\,x_{(12)}$—tracking the top two monthly values of the year. It remains statistically valid (Type I error is calibrated), but single-year storm outliers increase variance and widen CIs, especially over shorter 10-year spans. Note also that `standardization='interpolate'` cannot extrapolate beyond the block's sample minimum and maximum.
3. **Inter-Block Autocorrelation:** Non-overlapping blocks share no raw observations, but multi-year persistence (e.g., slow groundwater aquifers, multi-year climate cycles) can still correlate consecutive block quantiles. Pass `autocorr_method='yue_wang'` or `'block_bootstrap'` via `**mk_kwargs` if multi-year serial correlation is suspected.
4. **Field Significance Assumptions:** `field_significance=True` permutes entire block rows assuming blocks are exchangeable under $H_0$ (no strong inter-block autocorrelation) and permutes numeric values without censored-interval scoring (a warning is raised if censored data or `autocorr_method != 'none'` are present).
5. **Monotonicity Per Quantile:** Like all Mann-Kendall methods, each quantile test detects monotonic shifts over the full period. If a quantile reverses direction mid-record (e.g. rises then falls), inspect `result.get_quantile_series(q)` or apply `segmented_trend_test` to that extracted quantile series.
6. **Censoring & Tie Behavior:**
   * If a quantile is constant across all blocks (e.g., zero-flow dry beds or all blocks below a single detection limit), $S = 0$ and $\mathrm{Var}(S) = 0$, returning `slope = 0.0` and `p = 1.0`.
   * Under `censor_quantile_method='propagate'`, censoring flags are checked at the two bracketing order statistics $\lfloor u(n_k-1)\rfloor$ and $\lceil u(n_k-1)\rceil$ (exact for `quantile_method='linear'`, approximate for other interpolation rules).

---

## 7. Further Reading & Examples

* **Practical Guide & Recipes:** [`Examples/Detailed_Guides/quantile_trend_guide.md`](Examples/Detailed_Guides/quantile_trend_guide.md)
* **4-Case Walkthrough (Weekly Nitrate, 5-Year Daily Flow, Sparse Censored Groundwater, Sampling Drift Trap):** [`Examples/36_Quantile_Kendall/`](Examples/36_Quantile_Kendall/README.md)
* **Monthly Monitoring with Missing Visits, Changing Detection Limits & NPS-FM Percentiles:** [`Examples/37_Monthly_Censored_Quantile_Kendall/`](Examples/37_Monthly_Censored_Quantile_Kendall/README.md)
* **USGS EGRET Parity & Monte Carlo Calibration Report:** [`validation/43_Quantile_Kendall/`](validation/43_Quantile_Kendall/README.md)

**References:**
* Hirsch, R. M., Archfield, S. A., & De Cicco, L. A. (2015). A bootstrap method for estimating uncertainty of water quality trends. *Environmental Modelling & Software*, 73, 148–166.
* Hirsch, R. M., & De Cicco, L. A. (2015). User guide to Exploration and Graphics for RivEr Trends (EGRET) and dataRetrieval: R packages for surface-water quality and streamflow. *USGS Techniques and Methods*, Book 4, Chapter A10.
* Akritas, M. G., Murphy, S. A., & LaValley, M. P. (1995). The Theil-Sen estimator with doubly censored data and applications to astronomy. *Journal of the American Statistical Association*, 90(429), 170–177.
