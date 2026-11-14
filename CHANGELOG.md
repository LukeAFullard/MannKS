# Changelog

## Unreleased

### Packaging & Testing
* **Minimum dependency versions corrected** to the lowest versions on which the full test suite actually passes (verified on Python 3.9 with numpy 1.23.0 and pandas 2.2.0): `scipy >= 1.10` (was 1.9; `kendalltau(...).statistic` does not exist in 1.9), `matplotlib >= 3.6.0` (was 3.3.0) and `seaborn >= 0.13.0` (was 0.11.0; `boxplot(legend=...)`). No library code or numerical results changed.
* **Test fix**: the `np.intp` / 32-bit compatibility tests monkeypatch `numpy.repeat` globally and so also intercepted SciPy's own `np.repeat` call inside `scipy.stats.rankdata` (SciPy 1.13–1.15), failing three tests on those SciPy versions. The checks now apply to calls made from MannKS modules only, and assert that they were exercised.
* **Continuous integration**: added `.github/workflows/tests.yml`, running the suite on pull requests and pushes to `main` against the minimum supported versions, a SciPy 1.14 pin, and the latest versions on Linux, Windows and macOS.

## 0.7.0 — Generalized Quantile-Kendall release

Major feature release introducing frequency-agnostic, censor-aware, and drift-protected **Generalized Quantile-Kendall Trend Analysis** (`quantile_trend_test`), generalizing the USGS EGRET methodology (*Hirsch et al., 2015*) to arbitrary time series frequencies, multi-year blocks, and ambient monitoring networks.

### New Features & APIs
* **Generalized Quantile-Kendall Test (`quantile_trend_test`)**:
  - Partitions time series into consecutive, non-overlapping temporal blocks (`block_frequency`, e.g. `'YE'`, `'5YE'`, `'QE'`, `'ME'`, sub-daily offsets, or numeric widths), with optional `block_origin` and `months` seasonal filtering.
  - Standardizes order statistics using quantile interpolation (`'interpolate'`, Hazen by default), USGS EGRET leap-year rank reduction (`'egret'`), or explicit non-exceedance percentiles (`quantiles=[0.05, 0.5, 0.95]`), with configurable `target_length` (`'min'`, `'median'`, `'max'`, or explicit integer).
  - Native handling of detection limits (`<DL`, `>DL`) with rank-based censor propagation (`censor_quantile_method='propagate'`) and robust Akritas-Theil-Sen (ATS) slope estimation, plus `hicensor` recensoring.
  - Optional transformations (`transform='log'`, `'pct_of_median'`, `'none'`/`None`), computing exponentiated percentage change per time unit (`slope_pct`) and vectorized acceleration via `engine='fast'`.
  - Coverage filtering (`min_coverage`, `require_complete`) to eliminate bias from partial blocks, supporting sub-daily blocks and exact seasonal month-window coverage.
* **Sampling Frequency Drift Protection**:
  - Built-in diagnostic (`sampling_diagnostic`) evaluates Kendall's $\tau$ on sample size across time blocks and flags sampling frequency drift ($p < 0.10$ or $N_{\max}/N_{\min} > 1.5$).
  - Equalization strategies remove the artificial tail order-statistic expansion caused by historical shifts in monitoring frequency: `equalize='thin'` (systematic thinning to the smallest block) and `equalize='aggregate'` (robust medians within adaptive sub-intervals, taken on a common support of $k$ observations per sub-interval in every block so that medians are comparable between blocks; $k$ is reported as `sampling_diagnostic['aggregate_support']`).
* **Permutation Field Significance**:
  - Global matrix permutation test (`field_significance=True`) evaluates joint non-monotonicity across all quantiles while preserving empirical cross-quantile correlation.
* **Diagnostic Visualization (`plot_quantile_trend` / `plot_quantile_kendall`)**:
  - Dual normal Z / probability coordinates: plots on probit scale ($Z = \Phi^{-1}(p)$) with probability tick labels (`.001` to `.999`).
  - Linear probability mode (`x_scale='linear'`).
  - Flow Duration Curve overlay (`show_fdc=True`) displaying median concentration/flow on a secondary right-hand y-axis (`twinx`).
  - Significance color bands (`color_by='p'`: $p < \alpha$, $\alpha \le p < 2\alpha$, $p \ge 2\alpha$, in the style of EGRET's Quantile-Kendall plot but using the unadjusted Mann-Kendall `p`), or IPCC/LWP category mapping (`color_by='classification'`).
  - Confidence interval error whiskers (`show_ci=True`) and percentage slope labeling.
* **Regime Summary Tool (`summarize_quantile_regimes`)**:
  - Aggregates quantile slopes into intuitive low (0–25%), mid (25–75%), and high (75–100%) hydrological or water-quality regimes.

### Documentation, Validation & Packaging
* Added comprehensive statistical methodology document: [`quantile_kendall.md`](quantile_kendall.md).
* Added comprehensive deep-dive practical guide: [`Examples/Detailed_Guides/quantile_trend_guide.md`](Examples/Detailed_Guides/quantile_trend_guide.md).
* Added four-part practical example: [`Examples/36_Quantile_Kendall/`](Examples/36_Quantile_Kendall/README.md) covering weekly nitrate, daily flow 5-year blocks, sparse irregular groundwater, and sampling frequency drift trap.
* Added real-world monthly monitoring example: [`Examples/37_Monthly_Censored_Quantile_Kendall/`](Examples/37_Monthly_Censored_Quantile_Kendall/README.md) covering missing visits, incomplete-year exclusion, shifting detection limits, and NPS-FM 92nd/95th percentiles.
* Added USGS EGRET parity & Monte Carlo validation suite: [`validation/43_Quantile_Kendall/`](validation/43_Quantile_Kendall/README.md). Monte Carlo studies use 1,000–2,000 replicates and include a serial-correlation study of the unadjusted and pre-whitened p-values. Parity is measured against the **genuine EGRET R code** (EGRET's `makeSortQ`/`trendSortQ` run with the EGRET, rkt and zyp packages through `rpy2`): with `standardization='egret'`, `transform='log'`, `require_complete=True` and the new `block_time='egret'`, MannKS reproduces EGRET's rank matrix, $S$, $\mathrm{Var}(S)$, $\tau$, p-value and slopes to floating-point rounding for climate, water and calendar years, tied flows, and seasonal periods of analysis. EGRET's output is stored in `tests/data/egret_qk/` so the test suite asserts the same parity without R.
* `prewhiten='zhang'` adds the serial-correlation-adjusted p-value `p_adj` (with `tau_adj`, `acf1_adj`): Zhang's iterative trend-free pre-whitening, a step-by-step port of `zyp::zyp.zhang`. It reproduces EGRET's `pValueAdj` to the single-precision accuracy of the R routine in all parity scenarios, and zyp's slope, autocorrelation and iteration count exactly on 6,000 random series. `plot_quantile_trend(..., color_by='p_adj')` colours by it, as EGRET does.
* `block_time='egret'` uses EGRET's `midDecYear` (mean decimal year of the first and last observation of each block) as the block time, giving slopes per decimal year exactly as EGRET does; the default `'center'` uses the calendar midpoint of each block.
* Timezone-aware timestamps are analysed in local wall-clock time: block boundaries, `months` and reported block times are local (tz-naive), so calendar and water years start at local midnight.
* `quantiles='auto'` evaluates the ranks at the probabilities that belong to `quantile_method` ($(j - 0.5)/M$ for the default `'hazen'`), so a block of the target size is returned as its own order statistics and one censored observation flags one rank. A side effect is that the default analysis is far less sensitive to sampling-frequency drift than pure order statistics.
* Interpolated quantiles that are equal up to floating-point rounding are made exactly equal within each quantile column, so ties across blocks are counted as ties ($S$ and the tie-corrected $\mathrm{Var}(S)$ were otherwise slightly off for rounded data, and the two engines could disagree).
* Censor propagation is exact for every `numpy.quantile` rule, including discontinuous ones; a quantile bracketed by both a `<` and a `>` observation is reported with a warning.
* With `transform='log'`, `slope_units` now describes `slope` (`ln(<x_unit>) per <time unit>`); `slope_pct` remains % per time unit.
* `plot_quantile_trend` no longer closes a caller-supplied figure when `save_path` is given.
* Warnings keep their original category and are no longer lost when the call raises; `engine='fast'` fallbacks are recorded in `analysis_notes`; invalid `quantile_method`, duplicate `quantiles`, non-numeric `min_coverage` and scalar inputs raise clear `ValueError`s.
* `block_origin` never drops observations: block edges always span the record, calendar-anchored frequencies snap to their anchors with a `UserWarning`, and an origin far outside the record only sets the phase of the blocks.
* Exported `quantile_trend_test`, `QuantileTrendResult`, `summarize_quantile_regimes`, `plot_quantile_trend`, and `plot_quantile_kendall` in top-level `MannKS` namespace.

## 0.6.3 — code audit & correctness release

Comprehensive audit release addressing statistical correctness, memory management, performance optimizations, and regression testing while preserving 100% compatibility with LWP-TRENDS R outputs.

### Statistical Correctness & Integrity
* **Confidence interval indexing (`method='direct'`)**: Corrected an off-by-one indexing error where `upper_idx` had an unnecessary double decrement (`rank2 - 1` followed by floor/offset), violating Gilbert (1987) Eq. 16.8 ($M_2 = \frac{N+C}{2} + 1$). The upper and lower ranks (`lower_idx + upper_idx == n_sample - 1`) are now strictly symmetric around the median rank. `ci_method='lwp'` remains unchanged.
* **Robust right-censored interval scoring (`mk_test_method='robust'`)**: Implemented full interval ordering for right-censored observations (`>DL`), correctly evaluating pairs against uncensored and left-censored values via $O(n \log n)$ cross-scoring and variance tie corrections. Pairs such as $(60, 60) < (80, \infty)$ are now properly recognized as strictly concordant (+1). `mk_test_method='lwp'` remains unchanged.
* **Autocorrelation pre-whitening & detrending (`_autocorr.py`)**: `detrend`, `effective_sample_size`, and `should_apply_correction` now use `_sens_estimator_adaptive` with `max_pairs` and `random_state` for safe and scalable evaluation on large series.
* **Seasonal trend test result fields alignment**: Aligned the field ordering of `_RESULT_FIELDS` in `seasonal_trend_test.py` with `trend_test.py`, ensuring consistent tuple unpacking and field-by-name access across both APIs.
* **Exact time comparisons**: Replaced arbitrary float epsilon (`1e-10`) with exact equality checks (`dt != 0.0`) in unequal spacing Sen's slope and pair estimators.

### Performance & Memory Robustness
* **Plotting memory leaks**: Refactored plotting routines (`plot_seasonal_distribution`, `plot_inspection_data`, `plot_trend`, `plot_residuals`, `plot_rolling_trend`, `plot_segmented_trend`) to use explicit object-oriented Figure/Axes lifecycles with guaranteed cleanup in `try ... finally: plt.close(fig)`.
* **Rolling trend performance & bounded execution**: Implemented $O(\log N)$ binary search window slicing (`np.searchsorted`) for monotonic series and added `max_windows` parameter (default 10,000) with positive step validation.
* **Aggregation memory efficiency**: Vectorized single-row aggregations in `_aggregate_by_group` and `_aggregate_censored_median` with `return_dict=True` to eliminate $O(N^2)$ DataFrame concatenation overhead.
* **Pair sampling memory scaling**: Replaced dense `np.triu_indices` allocations in `_sample_pairs` with an $O(n)$ row-start search, preventing memory exhaustion on large datasets.
* **Regional reindexing**: Used `pd.DataFrame.reindex(columns=...)` in `regional_test` to guard against missing site columns.
* **Modern random API**: Updated block bootstrap routines to accept `np.random.Generator` in addition to legacy integer seeds and `RandomState`.
* **Immutable category maps**: Encapsulated `DEFAULT_CATEGORY_MAP` in `types.MappingProxyType` to prevent accidental in-place mutation.

### Compatibility & Regression Testing
* **100% LWP-TRENDS Parity Preserved**: All existing parity tests against verified LWP fixtures (`tests/test_lwp_parity.py`) pass without changes.
* **New regression suite**: Added `tests/test_audit_fixes.py` with 13 focused regression tests covering all audit items.

## 0.6.2 — 32-bit / Pyodide compatibility

* **Fixed `TypeError: Cannot cast array data from dtype('int64') to dtype('int32') according to the rule 'safe'` on 32-bit platforms** (such as Pyodide / WebAssembly in Streamlit) and with mixed-width integer inputs. Internal pair-indexing, run-length, and bootstrap index arrays (`_iter_triu_blocks`, `_sample_pairs`, `_moving_block_bootstrap_indices`, `_rle_lengths`, and related helpers) now use pointer-sized integers (`dtype=np.intp`) instead of hardcoded `int64` or unconstrained default integer types.

## 0.6.1 — large data sets

Bug fixes and performance work for large data sets. Results for data with up to 5,000 observations (after aggregation) are unchanged, except where noted.

### Fixes

* **Censored data did not scale.** The censored Mann-Kendall statistic compared every pair, even in fast mode: 3.4 GB of memory at n = 10,000, and out of memory at n = 50,000. It now uses an exact O(n log n) algorithm, giving the same S, Var(S) and Tau, and handles 1,000,000 values in about a second.
* **Block bootstrap on large data.** Each resample re-estimated the slope from every pair: O(n²) per resample, and a `NaN` confidence interval above n = 46,340. The resamples now use the same pair sampling as the main estimate, in both `trend_test` and `seasonal_trend_test`.
* **Missing dates (NaT) became a time 292 billion years before 1970** instead of being dropped. Missing times and infinite values are now removed, with a warning. `inspect_trend_data` failed on data containing a missing date.
* **Python `datetime` objects were converted using the computer's local time zone** (all other date types are treated as UTC), which put hour-long gaps or overlaps in hourly data at daylight-saving changes. All date types are now converted consistently: naive values as UTC, and time-zone-aware values at their UTC instant.
* **Seasonal tests subsampled large data by default.** Above 10,000 observations each season was randomly cut to 1,000 values before the test, which lost power. The test now uses every observation (it is exact and fast). Subsampling happens only if `max_per_season` is given. *Results change for seasonal tests with more than 10,000 observations.*
* **`approximation_error` underestimated the error** when the sampled slopes cluster (e.g. `sens_slope_method='lwp'`, which sets ambiguous slopes to 0). It is now a distribution-free interval for the sampled median, and is also reported for censored data with the default ATS estimator.
* `approximation_error` was in units per second even when `slope_scaling` was used. It is now in the same units as `slope`.
* `large_dataset_mode='full'` above n = 46,340 returned no slope. It now uses fast mode, with a warning.
* The fast-mode warnings now refer to the number of observations after aggregation, which is what decides the mode.
* Dates given as strings now raise a clear error asking for `pd.to_datetime`.
* Duplicated analysis notes are removed.

### Performance

* Exact (`'full'`) mode is 4–9× faster and uses a third to two-thirds of the memory: 1.2 s and 0.6 GB at n = 5,000 (previously 5.6 s and 1.8 GB). Slopes are built in blocks, and the confidence interval and Sen probability use order statistics instead of a full sort.
* The exact Akritas–Theil–Sen estimate finds its solution by bisection instead of evaluating every candidate (n = 5,000 censored: 1.5 s instead of 12.8 s).
* Pair sampling is 3× faster.
* `segmented_trend_test` warns above 20,000 observations: its breakpoint search (`piecewise-regression`) takes minutes at that size.

### Validation

* `tests/test_large_data_robustness.py` checks the new algorithms against the pairwise calculations, and the new interval, probability, ATS and note code against straightforward reference implementations. They agree exactly.
* Simulations of fast mode: confidence intervals contained the true slope 94.7–96.3% of the time (nominal 95%), and the type I error was 5.0%. See the [Large Dataset Guide](./Examples/Detailed_Guides/large_dataset_guide.md).
* The LWP-TRENDS parity results are unchanged.

### Documentation

* README: the version history is now a table, and there is a new large data section with measured performance.
* Examples 32 and 33 showed an error message instead of their output. Example 32's simulated data were wrong with pandas 3, and its text claimed narrower OLS intervals than the results showed. Examples 28 and two guides are now seeded, so they are reproducible.

## 0.6.0 — correctness release

This release fixes statistical defects found in a pre-release audit against the LWP-TRENDS R code (v2502, run live) and other independent references. **Several results and defaults change.** Re-run any analyses made with earlier versions before comparing them with new results.

### What changes for you

| If you… | What to expect |
| :--- | :--- |
| analyse data with **ties or censored values** | Smaller `var_s`, hence smaller p-values and higher `C`, than in 0.5.x. Var(S) previously ignored ties. |
| analyse **censored** data | A different slope and confidence interval: the default estimator is now Akritas–Theil–Sen (`'ats'`). |
| pass data **not sorted by time** | Results now match the sorted data. Previously they could depend on row order. |
| have **several observations at the same time** | Same-time pairs now score 0 in S, as in LWP-TRENDS. |
| use `sens_slope_method='unbiased'` or `'nan'` | The same estimator under its new name `'drop_ambiguous'`, plus a `FutureWarning`. |
| have a confidence **exactly on a class boundary** | The lower class (e.g. `C = 0.95` is now "Very Likely", not "Highly Likely"), as in LWP-TRENDS. |
| use `regional_test` with sites sampled on different days | Sites are now paired by month (`agg_period='month'`), so the inter-site correlation correction is applied. |
| use Python 3.7/3.8 or older numpy/pandas | You need Python ≥ 3.9, numpy ≥ 1.23, pandas ≥ 2.2 and scipy ≥ 1.9. |

### Critical fixes

* **Var(S) never included the tie and censoring corrections.** Tie groups were never detected, so the corrections were always zero, which inflated `var_s` and the p-value for any data with ties or censoring. Var(S) now matches LWP-TRENDS `GetKendal` and NADA exactly (Kendall 1975).
* **Results depended on the order of the input rows.** Data are now sorted by time, and the rules for censored pairs are applied in time order.
* **`large_dataset_mode='full'` silently sampled pairs.** It is now exact. Pair sampling happens only in `'fast'` mode, and is reproducible by default (seed 0, or `random_state`).

### Censored slope

* **New default: `sens_slope_method='ats'`**, the Akritas–Theil–Sen estimator (as in `NADA::cenken`). Its confidence interval comes from inverting the censored Kendall test. It agrees with NADA to within 0.3% (median) and is nearly unbiased in simulation (+0.4%).
* **`'unbiased'` is renamed `'drop_ambiguous'`.** This estimator drops pairs whose order is ambiguous because of censoring. Despite the old name it is biased *away* from zero: about +17% at 20% censoring and +70% at 50%. The old names `'unbiased'` and `'nan'` still work but give a `FutureWarning`, and will be removed in a future release.
* `'lwp'` (LWP-TRENDS substitution) is unchanged and still reproduces LWP-TRENDS.

### Matching LWP-TRENDS

* **`sen_probability`** follows LWP-TRENDS `Sen_Probability`. It is documented as the probability that the true slope is below zero.
* **Mid-observation selection** (`agg_method='lwp'` / `'middle_lwp'`) uses the LWP-TRENDS increment mid-points.
* **Analysis notes** follow the LWP-TRENDS rules:
  * "Long run of single value" uses runs of identical differences (`rle(diff(x))`);
  * the basic checks are repeated after aggregation;
  * seasons with no data are reported;
  * the Sen-slope notes use the R wording.

  Unlike LWP-TRENDS, MannKS still computes the trend when a check fails, and reports the problem in `analysis_notes`.
* **Classification thresholds are right-closed** (a value on a boundary belongs to the lower class), as in LWP-TRENDS `AssignConfCat`.
* **`regional_test`:**
  * sites are aligned by time increment before the correlations are computed (new `agg_period` parameter), as in LWP-TRENDS `getTAU`;
  * a missing correlation is omitted with a warning (it previously produced `CT = 1`);
  * an even split of directions gives "Increasing" with `TAU = CT = 0.5`, as in R.
* **`inspect_trend_data`:**
  * the default candidate increments are those of LWP-TRENDS (monthly, bi-monthly, quarterly, bi-annually, annually); weekly and daily are available through `custom_increments`;
  * the whole of 31 December of `end_year` is now kept.
* **`seasonal_trend_test`** has a new `tie_break_method` parameter (`'robust'` or `'lwp'`), as `trend_test` already did.

### Autocorrelation and bootstrap

* **Yue–Wang and `'auto'`** estimate autocorrelation on the Sen-detrended series, using only significant lags and capping `n_eff` at `n`. With `ci_method='lwp'`, the autocorrelation inflation is now kept in the slope interval.
* **Block bootstrap:**
  * circular blocks, reproducible with `random_state`;
  * the p-value uses `(b + 1) / (B + 1)`, so it is never exactly 0;
  * block length estimated from the detrended data;
  * the chosen slope method is used in the resamples.
* **Seasonal block bootstrap:** the confidence interval resamples whole cycles, and combining it with `'ats'` no longer crashes.

### Other fixes

* `prop_censored`, `prop_unique` and `n_censor_levels` describe the data before aggregation, as in LWP-TRENDS.
* p-values use the normal survival function, so very small p-values are accurate.
* `compare_periods` reports a direct test of the slope difference (`z_difference`, `p_value_difference`). `significant_change` is now `p_value_difference < alpha`; previously it meant non-overlapping intervals, which is far too conservative.
* **`segmented_trend_test`:**
  * uses the same data preparation as the other tests;
  * uses ATS for censored segments;
  * bagging intervals respect `alpha`.
* `plot_segmented_trend` accepts censored data and draws the final segment to its end point.
* **`check_seasonality`:**
  * `period` is inferred from `season_type` for datetime `t`;
  * `agg_method='lwp'` works;
  * a warning is given when a numeric `t` does not look like season units.
* `seasonal_trend_test` with a numeric `t` warns when `t` does not look like season units (for example, decimal years with `period=12`).
* Lower memory use for large datasets. Duplicated or spurious analysis notes are removed.

### Validation

* `tests/test_lwp_parity.py` checks results against frozen LWP-TRENDS (R) values. `tests/test_regressions_v060.py` has one test per fixed defect.
* `validation/r_parity/` reruns the comparison against R live, and also contains a NADA comparison, a permutation check of Var(S) and Monte Carlo calibration. With the LWP-TRENDS settings, the Mann-Kendall statistics match R in 300 of 300 cases.

### Known limitations

* The block bootstrap is somewhat liberal for short series: about 9% false positives at a nominal 5% for independent data.
* The Yue–Wang correction under-corrects for strong autocorrelation.
* The LWP-TRENDS date-shift adjustment (`FindDateShifts`) is not implemented.
* `large_dataset_mode='fast'` results are approximate by design. `approximation_error` reports the sampling error of the slope.

## 0.5.0 and earlier

See the "NEW IN" notes in the [README](./README.md).
