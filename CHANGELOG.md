# Changelog

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
