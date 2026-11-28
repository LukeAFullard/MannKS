# Brief (v2): add a robust seasonal–trend decomposition to MannKS

> **Revision note.** v2 revises v1 after a review against the MannKS 0.7.2 source, the `ndimpute` 0.1.1 source and a throwaway prototype of the §4 global mode. Section numbers match v1. §0 lists every change. Items marked **[AUTHOR DECISION]** are defaults the reviewer chose where v1 was ambiguous. Confirm or replace each one before work starts.

## 0. Changes from v1

| # | Where | v1 problem | v2 change |
|---|---|---|---|
| 1 | §4.5–4.6 | Imputation was updated once per robustness iteration. With `robust=False` the conditional imputation never ran, so non-detects kept the marginal start. With the default 5 updates it was unconverged at heavy censoring. In the prototype at ~50 % censoring in a seasonal trough, the amplitude ratio was 0.76 with 0 updates, 0.90 with 5 and 0.99 with ≥ 20. | Separate imputation loop that runs to convergence inside each robustness iteration (§4.5). |
| 2 | §4.4, test 1 | Three backfitting passes leave ~2e-4 error on noiseless data (10 passes give 6e-12), so the 1e-6 test cannot pass. | Backfit to a tolerance, with a cap and warm starts. |
| 3 | §4.5, test 1 | On noiseless data the bisquare scale `s` → 0 and the weights become erratic. | Scale floor: if `s` ≈ 0, keep all weights at 1. |
| 4 | test 3 | At 30 % censoring the marginal imputation also passes the 10 % amplitude check, so the test does not discriminate. | A second, discriminating case at ~50 % trough censoring that the marginal variant must fail. |
| 5 | test 2 | "≥ 3× smaller" fails on some single seeds (worst 2.2×, mean 3.8×). | Criterion applies to the mean over 20 fixed seeds. |
| 6 | §3 | `decompose(t, y)` reverses MannKS's `(x, t)` order. `trend_test` needs two arguments, so it cannot take one object "unchanged". | `decompose(x, t)`. `deseasonalized_series()` returns `(x_df, t)`. |
| 7 | §3, §4.6 | The output scale ("back-transformed where needed") was undefined. The MK p-value is the same on both scales; the Sen/ATS slope is not. | Explicit `scale=` argument, defaulting to the transformed scale. |
| 8 | §4.6 | Existing `trend_test(..., sens_slope_method='lwp')` multiplies non-detects by `lt_mult=0.5`. On negative log values this moves them *up*. | The helper requires `'ats'` when censored values are present. "No DL/2" is scoped to the new module. |
| 9 | §3 | Censoring input was given as "flags / strings / DL column". MannKS's actual convention is the `prepare_censored_data` DataFrame. | Accept that DataFrame (or plain numbers) only. The `censored=` argument is removed. |
| 10 | §4.1 | Numeric `t` has arbitrary units in MannKS. | Numeric `t` is defined as decimal years, with a units sanity warning. |
| 11 | §4.6 | `ndimpute` details: (a) a numeric −1/0 status is classed as "mixed"; (b) mixed ROS imputes right-censored values stochastically by default; (c) the residual fit ignores robustness weights, so outliers inflate σ. | Pass `cen_type` strings, set `right_impute_type="mean"` and exclude zero-weight detects from the fit. |
| 12 | §4.2 | Choosing K needs a "final trend", which needs K. | Defined sequence: fit at K_max, compare K with weights and imputations fixed, then refit. |
| 13 | §4.2 | One isolated sample can close the phase gap and allow K = 4. | Coverage counts only phases supported by several years. |
| 14 | §4.3 | ≥ 10 points at one timestamp (storm bursts, duplicates) make the local-linear fit singular. | Distinct-time and condition-number checks, with a local-constant fallback. |
| 15 | §4.7 | ESS ≥ 10·2K forces monthly data with K = 4 to ~9.5-yr windows and quarterly data to ~14-yr windows (tricube ESS ≈ 0.71 n). | **[AUTHOR DECISION]** ESS ≥ 5·2K, with the implied windows documented. |
| 16 | §4.9 | Block bootstrap undefined for blocks with unequal sample counts; censoring of bootstrap points undefined; stochastic imputation inside an iterative loop cannot converge. | Fully specified scheme. Stochastic imputation happens once, after convergence. |
| 17 | §4.10 | Lag-1 autocorrelation by order is distorted by burst sampling. | Use MannKS's own `_autocorr.estimate_acf`, with a documented caveat. |
| 18 | §4.1 | No rule for values ≤ 0 under the log transform, or for duplicate times. | Raise an error for values ≤ 0. Duplicate times are allowed (MannKS convention). |
| 19 | §6B | 600 cells × 500 runs, with auto mode needing 2 × 199 refits per run, is tens of thousands of CPU-hours. | Costed design: auto mode on a subset, a runtime pilot, a compute budget. |
| 20 | §6B | Size window [0.03, 0.07] with 500 runs (MC SE ≈ 0.0097): about 1 in 20 correct cells fails by chance, so ~12 spurious failures are expected across the null cells. | Size cells use 1000 runs and a Holm-corrected binomial test combined with the window. |
| 21 | §6B | Record length, σ, amplitudes, form of evolving seasonality, censoring mechanism, AR(1) on irregular times, SK settings, autocorrelation option and slope scale were all unspecified. | All specified (§6B.1). |
| 22 | §6A | The 20 synthetic designs were not defined; "within 2 % on average" was ambiguous; scoring of held-out non-detects and out-of-span points was undefined; results could not be attributed. | Defined metric; a like-for-like K = 4 run alongside defaults; the author supplies the designs. |
| 23 | §6C | "Keep mg/L" matches nothing (WQP units are `mg/l as N` / `mg/l as P`). Averaging a censored duplicate needs substitution. The legacy WQP endpoint has had no USGS updates since early 2024. | Revised cleaning rules, a pinned data snapshot, censored-median duplicates. |
| 24 | §1 | `ndimpute` needs pandas ≥ 2.0, scipy ≥ 1.13 and numpy ≥ 1.24, above MannKS's floors (scipy 1.10, numpy 1.23). | Optional extra. Core floors unchanged. |
| 25 | §3, §5 | Auto mode in unit tests would take minutes. | `n_boot_auto` is exposed; tests use a small B and are marked slow. |

---

## 1. Goal

Add an **optional** module to MannKS that splits an irregularly sampled water-quality series into **trend + seasonal + remainder**. It uses:
- **annual harmonic regression** for the seasonal part;
- a **robust local-linear (LOWESS-type) smoother** with a time-based window for the trend;
- **backfitting** with **one shared set of robustness weights**;
- **model-conditional imputation of non-detects** with `ndimpute`, used only to estimate the components.

Main uses:
1. **Describe seasonality**: amplitude, timing of the peak, and whether the pattern changes over time.
2. **Deseasonalise** a series so the existing MannKS tests (Mann-Kendall, Sen / Akritas-Theil-Sen slopes, segmented and rolling trends) can be run on it. Censored values stay censored throughout.

**Hard constraints**
- Do not change any existing MannKS default or result. The Seasonal Kendall test stays the recommended default for seasonal data. The new module is opt-in. Adding `decompose` to `MannKS/__init__.py` exports is allowed.
- No value substitution for non-detects (no DL/2 or any multiple of the DL) anywhere **in the new module**. Existing MannKS options that use `lt_mult` (e.g. `sens_slope_method='lwp'`) are not touched, but the new helper must not route censored data through them (§4.6.6).
- All imputation of non-detects goes through `ndimpute`, applied to model residuals as specified in §4.6. ndimpute's substitution method and `lwp_compat=True` must never be used.
- Dependencies:
  - The core decomposition uses numpy and scipy, plus pandas and matplotlib, which MannKS already depends on.
  - `ndimpute` is an **optional extra**: `pip install mannks[censored]`, pinned to `ndimpute>=0.1.1,<0.2`. It is imported lazily, only when censored values are present. Without it, censored input raises an `ImportError` that names the extra.
  - The core numpy/scipy floors in `pyproject.toml` do not change. The extra carries ndimpute's own floors.
  - `statsmodels` is a validation-only dependency (§8). It is never imported by the package.

### 1.1 MannKS conventions to follow (verified against 0.7.2)

The plan note (§9 step 1) must confirm each of these against the code:

- **Argument order:** values first, then times. All public functions are `f(x, t, ...)`.
- **Censoring:** `prepare_censored_data(x)` turns strings such as `'<0.5'` and `'>10'` into a DataFrame with columns `value` (float), `censored` (bool) and `cen_type` (`'lt'`, `'gt'` or `'not'`). Multiple detection limits are simply different `value`s on censored rows. Public functions accept that DataFrame or plain numbers, and raise `TypeError` when given raw strings, pointing to `prepare_censored_data`.
- **Time:** `_helpers._prepare_data(x, t, hicensor)` returns numeric `t`:
  - datetimes become seconds since 1970-01-01 (tz-aware values via UTC, naive values taken as given);
  - numeric `t` is kept as is, with "unit inherited";
  - the original `t` is kept as `t_original`.

  NaN values are dropped silently. Infinite values and missing times are dropped with a warning. Data are sorted by time internally. Duplicate timestamps are allowed and noted, not rejected.
- **Results:** a namedtuple, or for results with methods, a class that subclasses a namedtuple. `SegmentedTrendResult` (with `.predict()`) and `QuantileTrendResult` are the precedents.
- **Warnings:**
  - collect them with `warnings.catch_warnings(record=True)`;
  - store them as a `warnings` list on the result, with short notes in `analysis_notes`;
  - re-emit them as `UserWarning` before returning.
- **Plotting:** functions in `MannKS/plotting.py` take `save_path` (or `plot_path`) and/or `ax`, and close the figures they create.
- **Randomness:** `np.random.default_rng(random_state)`. The global NumPy RNG is never touched.

---

## 2. Evidence this design is based on

Unchanged from v1. A pre-registered comparison was run on 20 synthetic designs and 17 real USGS nitrate and total-phosphorus records (9 rivers, 20 held-out splits each). The reference version was global harmonics 1–4, a robust LOWESS trend with a 3-year span, and backfitting. It was significantly beaten on only 1 of 17 real records and had the lowest error most often. It was best under heavy tails, outliers, heteroscedastic noise and autocorrelated noise. It was 1.5–2.5× worse when the seasonal pattern changes over the years, which is why an opt-in time-varying mode is included. The reference implementation is in §8.

**[AUTHOR TO SUPPLY]** the code or a full specification of the 20 synthetic designs and the original real-data cleaning script, if they exist. They are needed for §6A and §6C. If they are not available, §6A runs on the real records only and §6C uses the fallback cleaning rules given there.

---

## 3. Public API

```python
res = MannKS.decompose(
    x, t,                          # values (array or prepare_censored_data DataFrame), times
    censored_method="parametric",  # ndimpute method for residual imputation: "parametric" | "ros"
    transform="log",               # "log" (default for concentrations) | "none"
    n_harmonics="auto",            # 1..4 or "auto"
    trend_span_years=3.0,          # full width of the trend window; minimum 2.0
    seasonal="global",             # "global" | "time_varying" | "auto"
    seasonal_window_years=7.0,     # full width for time-varying coefficients; minimum 3.0
    robust=True,
    n_outer=5,                     # max robustness iterations
    backfit_tol=1e-8, backfit_max_iter=50,
    impute_tol=1e-4, impute_max_iter=50,
    n_boot=0, block_years=1.0,     # optional block-bootstrap bands
    n_boot_auto=199,               # bootstrap size for seasonal="auto"
    n_imputations=0,               # stochastic re-imputations when n_boot == 0
    random_state=None,
)
```

- `x`: a numeric array or Series, or the DataFrame from `prepare_censored_data`. Raw strings raise the standard MannKS `TypeError`.
- `t`: datetime-like or numeric. Numeric `t` is interpreted as **decimal years** (period 1.0). If a numeric span exceeds 1000, warn that `t` looks like seconds or days rather than years.
- `random_state=None` **[AUTHOR DECISION]**: fresh entropy, matching `_bootstrap.py`. The alternative is to treat `None` as seed 0, matching `trend_test`'s "reproducible by default" pair sampling. Either way, all randomness flows through `np.random.SeedSequence(random_state)`.

### Result: `DecompositionResult` (subclass of a namedtuple)

Arrays are per observation, in time-sorted order, on the transformed scale:
- `times` (original `t` values) and `t_years` (decimal years);
- `order`: indices into the input after NaN removal, so users can map back;
- `values`: observed value or limit;
- `censored`, `cen_type`;
- `trend`, `seasonal`, `remainder`, `weights` (the weights used in the final fit);
- `fitted_values`: the converged conditional imputations on censored rows, NaN on detected rows. Exposed for diagnostics only; never used as data in a test.

Scalars and metadata:
- `n_harmonics_used`, `K_max`, `seasonal_mode_used` (`"global"` / `"time_varying"`), `seasonal_constancy_pvalue` (only when `seasonal="auto"`, otherwise NaN), `transform`.

Seasonal descriptors:
- Global mode: `seasonal_amplitude` (max − min of the seasonal curve on a 1000-point phase grid), `seasonal_amplitude_ratio` (`exp(amplitude)` when log-transformed), `peak_day_of_year` and `trough_day_of_year` (phase × 365.2425, rounded and mapped to 1–366).
- Time-varying mode: a table `seasonal_by_year` with the same quantities per calendar year, evaluated at each year's mid-point.

`diagnostics` (dict):
- `phase_coverage` (max gap, supported-phase count, `K_max`, and the reason for capping);
- `share_zero_weights`;
- `acf1_remainder`;
- `n_trend_widenings`, `n_seasonal_widenings`, `n_local_constant_fallbacks`;
- convergence counts (`backfit_iters`, `impute_iters` per outer iteration, `outer_iters`, `converged` flags);
- `ndimpute` (`df.attrs` from the final call plus de-duplicated ndimpute warnings);
- `prop_censored`.

`warnings` (list) and `analysis_notes` (list), as in MannKS.

Optional uncertainty:
- `trend_low/high`, `seasonal_low/high` (pointwise 95 %) and `trend_simul_low/high` (simultaneous 95 %) when `n_boot > 0`;
- `trend_imputation_sd` and `seasonal_imputation_sd` when `n_imputations > 0`.

Methods:
- `predict(t_new)`: trend interpolated linearly between observation-time values, plus the seasonal evaluated at `t_new` (time-varying mode interpolates the coefficients). `t_new` outside `[t_min, t_max]` returns NaN with a warning; there is no extrapolation.
- `to_dataframe()`: one row per observation with all per-observation fields and the `order` index.
- `plot(save_path=None, axes=None)`: four panels (data with fit and censored points marked, trend with band, seasonal, remainder with zero-weight points marked), plus a seasonal-cycle-by-day-of-year panel. Follows the MannKS plotting conventions in §1.1.
- `deseasonalized_series(scale="transformed")`: returns `(x_df, t)`.
  - `x_df` is a DataFrame with columns `value`, `censored`, `cen_type` in exactly MannKS's format.
  - `t` is the original times in sorted order.
  - `scale="original"` back-transforms (log mode only). See §4.6.5.

Helper (new public function):
```python
MannKS.deseasonalized_trend_test(res, scale="transformed", **trend_test_kwargs)
# == x, t = res.deseasonalized_series(scale); MannKS.trend_test(x, t, **kwargs)
```
- Raises `ValueError` if censored values are present and `sens_slope_method != 'ats'`. The other methods use `lt_mult`/`gt_mult` substitution, which is wrong for shifted log-scale limits.
- Warns if `hicensor` is set. The deseasonalised limits vary by observation, so the high-censor rule censors much more than intended.

**Slope units:** on `scale="transformed"` with `transform="log"`, the slope is in ln-units per time unit. With `slope_scaling="year"` and small slopes, 100 × slope ≈ % change per year. Document this.

---

## 4. Algorithm

Constants are defaults and exposed as arguments unless marked internal.

### 4.1 Time, scale and validation
- Use `_prepare_data` (with `hicensor=False`) for NaN handling and conversion to numbers. Sort by time with a stable sort.
- Decimal years:
  - datetimes: `t_years = 2000 + (sec − sec(2000-01-01T00:00:00)) / (365.2425 × 86400)`;
  - numeric `t`: used as given.

  Never use calendar fractions, which compress leap years.
- `transform="log"`: `x = ln(value)` for detected values and `ln(DL)` for censored limits. Raise `ValueError` if any detected value or any limit is ≤ 0. `transform="none"`: `x = value`.
- Raise `ValueError` if:
  - the span is < 3 years;
  - there are < 24 detected values;
  - more than 80 % of values are censored;
  - `trend_span_years < 2`;
  - `seasonal_window_years < 3`;
  - any option is invalid.

  Warn if more than 50 % of values are censored.
- Duplicate timestamps are allowed, with no aggregation inside `decompose`. Note their presence in `analysis_notes`.

### 4.2 Seasonal basis and number of harmonics
- Basis columns `cos(2πkt), sin(2πkt)`, `k = 1..K`, with no intercept. The level belongs to the trend, so the seasonal curve averages to zero over each year by construction.
- **Phase-coverage cap** (revised to resist isolated samples; new internal constants, not tuned on any validation design):
  1. Phases `φ_i = t_years mod 1`.
  2. A phase is *supported* if observations from at least `m = max(2, ceil(0.1 × n_years))` distinct calendar years fall within ±1/48 year (±7.6 days) of it.
  3. `max_gap` = the largest circular gap between consecutive *supported* phases.
  4. `K_max = clip(floor(1 / (2 · max_gap)), 1, 4)`.
  5. Record `max_gap`, `m`, `K_max` and the reason in the diagnostics.
- **Choosing K when `n_harmonics="auto"`:**
  1. Run the full pipeline (§4.5) at `K = K_max`.
  2. Holding its final weights `w` and converged imputations fixed, run backfitting (§4.4) to convergence for each `K = 1..K_max`.
  3. Compute `BIC(K) = n_d · ln(RSS_w / Σ_d w) + p · ln(n_d)`, where:
     - the sums run over **detected** observations with `w > 0`, and `n_d` is their count;
     - `RSS_w = Σ w r²`;
     - `p = 2K + tr(S_trend)`, with `tr(S_trend)` = Σ of the trend smoother's hat-matrix diagonal from the final pass.
  4. Choose the K that minimises BIC.
  5. If `K ≠ K_max`, rerun the full pipeline at the chosen K.
- With an integer `n_harmonics` greater than `K_max`: use it, and warn that phase coverage does not support it.

### 4.3 Trend smoother (time-based window, irregular-safe)
- At each observation time τ, fit a weighted local-linear regression with weights `w_i · T((t_i − τ)/h)`, where T is the tricube kernel and `h = trend_span_years / 2`.
- **Window safeguards.** Widen `h` in 10 % steps (cap at the record span) until all three hold:
  1. ≥ 10 points with non-zero weight;
  2. those points cover ≥ 3 distinct timestamps;
  3. the condition number of the 2×2 weighted design `XᵀWX` (with centred `t`) is < 1e8.

  If they still fail at the cap, fall back to a local-constant (weighted mean) estimate. Count widenings and fallbacks in the diagnostics.
- Also return the hat-matrix diagonal `L_ii`, used for `tr(S_trend)` in §4.2.
- `predict` interpolates linearly between observation-time trend values.
- Do not call a LOWESS routine with its own robustness iterations. Robustness is handled once, jointly, in §4.5.
- **Performance target** (needed for the §6 budget): a vectorised implementation (no per-point Python loop over all n for the kernel weights) should fit global mode on n = 600 uncensored in ≤ 0.3 s on one core. A naive per-point loop takes about 1 s.

### 4.4 Backfitting (inner loop; fixed weights `w` and fixed imputations)
```
if no previous fit: trend = smooth(x, w), seasonal = 0     # cold start
else:               start from the previous trend/seasonal  # warm start
repeat up to backfit_max_iter (default 50):
    seasonal = WLS fit of harmonics to (x − trend) with weights w, evaluated at t
    trend    = smooth(x − seasonal, w)
    stop when max(|Δtrend|∞, |Δseasonal|∞) < backfit_tol × max(1, MAD(x))
```
Record the iteration counts. If the cap is reached, warn that backfitting did not converge.

### 4.5 Fitting schedule (robustness outer loop, imputation middle loop)
```
x = transformed values; censored rows hold their limit
if any censored: x[censored] = marginal start imputation (§4.6.2)
w = 1
for outer in 1 .. (n_outer if robust else 1):
    # middle loop: imputation to convergence at fixed w
    for j in 1 .. (impute_max_iter if any censored else 1):
        backfit (§4.4, warm start)
        if no censored values: break
        new = conditional imputation given mu = trend + seasonal (§4.6.3)
        if max |new − x[censored]| < impute_tol: x[censored] = new; break
        x[censored] = new
    (one final backfit if the last action was an imputation update)
    if not robust or outer == n_outer: break
    r = x − trend − seasonal
    s = 6 · median(|r| over detected values)
    if s ≤ 1e-10 · max(1, MAD(x)):                     # noiseless / degenerate
        w_new = 1 for all; note "robustness skipped: residual scale ≈ 0"
    else:
        w_new = (1 − (r/s)²)² for |r| < s, else 0      # bisquare, detected values
    w_new[censored] = 1                                # never down-weight from an imputed residual
    if max |w_new − w| < 1e-3: break
    w = w_new
```
- The reported `weights` are the `w` used in the final fit.
- With `robust=False`, there is one outer pass with `w = 1`. **The imputation loop still runs to convergence.**
- Warn if the imputation loop hits `impute_max_iter`.

### 4.6 Censored values: `ndimpute`, conditional on the fitted model

#### 4.6.1 Why residuals, not concentrations
`ndimpute.impute` fits a single distribution to whatever it is given. On raw concentrations it ignores trend and season, so imputed non-detects are pulled toward the overall distribution. That flattens seasonality exactly where the data are censored. Applied to residuals around the current fit, the imputation is model-aware.

#### 4.6.2 Input conversion and the start
- Pass MannKS's `cen_type` array (`'lt'`, `'gt'`, `'not'`) **as strings** as `status`; ndimpute accepts these directly. Do not pass numeric −1/0, because a status with no `censoring_type` is then classed as "mixed".
- Multiple detection limits are simply different limit values.
- **Start (loop initialisation only):**
  ```python
  ndimpute.impute(x_with_limits, status=cen_type, method="ros", dist="normal",
                  impute_type="mean", right_impute_type="mean")
  ```
  `right_impute_type="mean"` is required, because mixed ROS otherwise imputes right-censored values stochastically.

#### 4.6.3 Conditional imputation (each middle-loop step)
1. `mu = trend + seasonal` at the observation times (transformed scale).
2. Residual-scale inputs:
   - detected: `x_i − mu_i`;
   - censored: the limit on the transformed scale minus `mu_i`, with the same `cen_type`.
3. **Exclude down-weighted outliers from the fit:** detected rows with `w_i = 0` are passed as `NaN`. ndimpute carries NaN through and leaves it out of fitting. If this leaves fewer than 24 detected values, pass all detected values instead and warn.
4. Call:
   ```python
   kw = dict(dist="normal", impute_type="mean", random_state=seed)
   if censored_method == "ros": kw["right_impute_type"] = "mean"     # ROS defaults clamp=True
   df = ndimpute.impute(resid_values, status=cen_type, method=censored_method, **kw)
   ```
   `dist="normal"` is required because residual limits can be negative. `impute_type="mean"` keeps the loop deterministic.
5. Set `x_i = mu_i + df["imputed_value"][i]` for censored rows only. Detected values are never changed.
- Capture ndimpute warnings on every call, de-duplicate them and store them in `diagnostics["ndimpute"]["warnings"]`. Store the final call's `df.attrs` there too.
- **Assumption to document:** the residual model is a single normal distribution (homoscedastic, independent). Seasonal or level-dependent variance is not modelled.
- **Censored points keep weight 1** (§4.5). Document that a wrong detection-limit entry therefore cannot be down-weighted.

#### 4.6.4 What censored output means
A censored observation stays censored. The imputed values are used **only** to estimate the components, never as data in a trend test.

#### 4.6.5 Deseasonalised series
- **`scale="transformed"`** (default):
  - detected: `value = x_i − seasonal_i`;
  - censored: `value = limit_i − seasonal_i`;
  - `censored` and `cen_type` are copied unchanged.

  With `transform="log"`, the limit is `ln DL_i − seasonal_i`.
- **`scale="original"`** (only valid with `transform="log"`):
  - detected: `value = y_i · exp(−seasonal_i)`;
  - censored: `value = DL_i · exp(−seasonal_i)`.

  The status is unchanged.
- Both scales give the **same** Mann-Kendall S and p-value, because the map is monotone and preserves censoring. Sen/ATS slopes differ. The transformed scale is the right one for log-linear trends and is what §6B validates.
- Right- or mixed-censored values work the same way, with the shifted limit.

#### 4.6.6 Passing to MannKS tests
Use `trend_test` with `sens_slope_method='ats'` (the MannKS default). That gives the censored Kendall test and the Akritas-Theil-Sen slope, which handle the many distinct shifted limits. `deseasonalized_trend_test` enforces this (§3).

#### 4.6.7 Uncertainty from censoring
- **Bootstrap:** each replicate re-imputes stochastically once, after convergence (§4.9).
- **`n_imputations > 0`** (only when `n_boot = 0` and censored values are present):
  - start from the converged fit;
  - for each m, draw stochastic imputations with
    ```python
    ndimpute.impute(..., impute_type="stochastic", random_state=child_seed_m)   # plus right_impute_type="stochastic" for ROS
    ```
  - run one backfit at fixed weights;
  - report the pointwise SD of trend and seasonal across the m draws.

  Document these SDs as a **lower bound**. ndimpute's imputations condition on the fitted parameters, so they are improper in Rubin's sense.

### 4.7 Time-varying seasonal mode (`seasonal="time_varying"`)
- Replace the global harmonic WLS in §4.4 by **local harmonic fits**. At target time τ, fit the harmonics by WLS to `x − trend` with weights `w_i · T((t_i − τ)/H)`, where `H = seasonal_window_years / 2`. There is no intercept, and only the harmonic coefficients vary.
- K is the global choice from §4.2 (run in global mode first). It is not re-selected per window.
- **Evaluation grid (internal, for cost):** estimate coefficients on a grid with a step of 0.1 year covering the span, and interpolate them linearly to observation times and in `predict`.
- **Per-window safeguards.** Widen H in 10 % steps (cap at the record span, which equals global mode) until all three hold:
  1. the §4.2 phase-coverage rule supports K for the window's non-zero-weight points;
  2. effective sample size `(Σw)²/Σw² ≥ c_ess · 2K`, with **[AUTHOR DECISION]** `c_ess = 5` (v1 used 10);
  3. the condition number of the weighted design < 1e6.

  Record every widening.

  Because tricube ESS ≈ 0.71 × the points in the window, the window actually used looks like this:

  | Sampling | K | `c_ess = 10` | `c_ess = 5` |
  |---|---|---|---|
  | Monthly | 4 | ~9.5 yr | 7 yr (unchanged) |
  | Quarterly | 2 | ~14 yr | ~7 yr |

  Document the effective window length in the result (`diagnostics["seasonal_window_used"]`, a per-grid-point array).
- `seasonal_window_years < 3` → `ValueError`.
- **Known bias:** the local-constant coefficients are biased toward the interior within about H of the record ends. Document this. Test 5 checks only the interior.

### 4.8 Automatic choice (`seasonal="auto"`)
1. Fit global mode (full pipeline). Then fit time-varying mode using the global fit's final weights and imputations, with backfitting to convergence.
2. Statistic: `F = (RSS_global − RSS_tv) / RSS_tv`, where both RSS are weighted and computed over detected values with `w > 0`.
3. Null calibration: `n_boot_auto` (default 199) parametric block-bootstrap series generated from the **global** fit with the §4.9 generator. For each one, run the full global pipeline, then time-varying as in step 1, and compute F*.
4. p-value = `(1 + #{F* ≥ F}) / (n_boot_auto + 1)`. Use time-varying if p < 0.05, otherwise global. Report the p-value.
5. The default stays `seasonal="global"` until §6B passes.
- **Cost:** about `2 × (n_boot_auto + 1)` full fits. Document this. Tests use `n_boot_auto=49`.

### 4.9 Bootstrap bands (optional, `n_boot > 0`)

**Scheme** (non-overlapping block bootstrap with time-shifted blocks):
- Partition `[t_min, t_max)` into consecutive blocks of length `L = block_years`. `L` must be a positive integer number of years (raise otherwise); this keeps seasonal phase when blocks are shifted.
- For each target block b, draw a source block b′ uniformly, with replacement, from the full-length blocks.
- **Copy the source block's observations and shift their times by (b − b′)·L.** This keeps within-block sampling patterns such as storm bursts and winter gaps, so the number of observations differs between replicates. If the last target block is partial, truncate the copied block to fit.
- For a copied observation i (source time `s_i`, new time `t*_i`):
  - **Value:** `x*_i = trend(t*_i) + seasonal(t*_i) + r_i`, where `r_i` is i's remainder in the original fit. For censored observations `r_i` uses the converged imputation. `trend(t*)` comes from `predict`.
  - **Censoring:** a copied observation keeps its own status. If censored, its **residual-scale limit is preserved**: `limit*_i = trend(t*_i) + seasonal(t*_i) + (limit_i − mu(s_i))`. Detected observations stay detected; they carry no limit.
- **Refit** each replicate with the **whole pipeline**: robustness, imputation to convergence and K fixed at `n_harmonics_used`.
- If censored values are present, the replicate then does **one** stochastic re-imputation (`impute_type="stochastic"`, child seed) followed by one backfit at fixed weights. The bands therefore include imputation uncertainty.
- Evaluate each replicate's trend and seasonal at the **original** observation times via `predict`.

**Bands:**
- Pointwise 95 %: the 2.5 % and 97.5 % percentiles.
- Simultaneous trend band: with `sd_t` = the bootstrap SD at each time, `c` = the 95th percentile over replicates of `max_t |trend*_t − trend_t| / sd_t`, and band = `trend ± c · sd_t`.

**Seeds:** `SeedSequence(random_state).spawn(n_boot)`, one child per replicate, used for block draws and the stochastic imputation.

**Document:** the bands do not include smoothing bias at the record ends or inside long gaps, and they condition on the fitted K.

### 4.10 Warnings (in `warnings`, `analysis_notes` and re-emitted as `UserWarning`)
- Phase coverage limits K below 4.
- More than 10 % zero robustness weights.
- |lag-1 autocorrelation of the remainder| > 0.3. Compute it with `MannKS._autocorr.estimate_acf` on the time-ordered remainder of detected values, the same estimator `trend_test` uses. Suggest `autocorr_method='yue_wang'` or `'block_bootstrap'` for the follow-up trend test, and note that burst sampling inflates this value.
- Trend-window or seasonal-window widenings, and local-constant fallbacks.
- Backfitting, imputation or robustness loop not converged.
- Record shorter than 2 × `seasonal_window_years` when `seasonal` is `"time_varying"` or `"auto"`.
- More than 50 % censored.
- `n_harmonics` above what coverage supports.
- Numeric `t` that looks like it is not in years.

---

## 5. Tests (pytest)

Fixed seeds throughout. Mark tests slower than 10 s with `@pytest.mark.slow`.

1. **Exact recovery.** Noiseless linear trend plus two harmonics at 180 random times over 15 years; `n_harmonics=2`.
   - With `robust=False`: trend and seasonal errors < 1e-6 (needs backfitting to converge, §4.4).
   - With `robust=True`: the scale guard keeps all weights at 1 and the same tolerance holds.
2. **Robustness.** 20 seeds; 180 random times over 15 years; σ = 0.3; 10 % of points shifted by −8σ.
   - The **mean over seeds** of trend RMSE with `robust=False` is at least 3× that with `robust=True`.
   - At least 95 % of the shifted points get weight 0, pooled over seeds.
   - The reviewer's prototype gave a mean of 3.8× (worst seed 2.2×) and 100 % zero weights.
3. **Censoring.**
   - (a) **30 % left-censored, two detection limits** (higher DL in the first half). Over 10 seeds:
     - every imputed residual is ≤ its residual-scale limit;
     - the mean seasonal amplitude is within 10 % of the fit to the uncensored truth.
   - (b) **Discriminating case: censored seasonal trough.**
     - Design: 180 random times over 15 years; ln-scale trend `−1 + 0.03(t − 2000)`; one harmonic with cos coefficient 0.8; σ = 0.25; DL = −0.6 before mid-record and −0.9 after (about 50 % censored, mostly in the trough).
     - Over 10 seeds, the mean amplitude ratio to the uncensored fit is within 7 % of 1, for both `robust=False` and `robust=True`.
     - **Control:** the same fit with conditional imputation disabled (`impute_max_iter=0`, start imputation only) gives a mean ratio < 0.85.
     - The reviewer's prototype gave ≈ 0.99 for the converged fit and ≈ 0.76 for the control.
   - (c) **ROS variant** (`censored_method="ros"`) passes (a).
   - (d) **Right-censored case** (10 % `>DL`): runs, keeps `cen_type='gt'`, and the shifted limits are correct.
   - (e) **Output checks:**
     - `cen_type` and `censored` are preserved;
     - deseasonalised limits equal `ln DL − seasonal` (transformed scale) and `DL · exp(−seasonal)` (original scale) to 1e-12;
     - `deseasonalized_trend_test` runs on both scales;
     - the MK p-values are identical on the two scales;
     - `sens_slope_method='lwp'` with censored data raises `ValueError`.
   - (f) **No substitution:** monkeypatch `ndimpute.impute` to assert that every call has `method ∈ {"parametric", "ros"}`, never `"substitution"`, and never `lwp_compat=True`.
   - (g) **Without ndimpute installed** (simulate by blocking the import): censored input raises `ImportError` naming `mannks[censored]`; uncensored input works.
4. **Irregular sampling.**
   - Two designs: (i) monthly samples plus storm bursts (3–6 samples within 3 days, 6 times a year); (ii) winter-sparse sampling.
   - Requirements: no NaN; |trend error| < 4σ everywhere; K reduced when coverage is poor.
   - (iii) ≥ 10 samples on one identical timestamp: the singular-window safeguard triggers (widening or fallback is recorded), with no `LinAlgError`.
5. **Time-varying mode.**
   - (a) First-harmonic amplitude grows linearly from 0.3 to 0.9 over 20 years of monthly data. The recovered amplitude is within 15 % of the truth at every grid point more than `seasonal_window_years/2` from either end.
   - (b) An 8-year record with `seasonal_window_years=7` and sparse sampling: windows widen to the span, and the result equals global mode to 1e-8.
   - (c) `seasonal="auto"` with `n_boot_auto=49` on constant seasonality: global is chosen in ≥ 8 of 10 seeds (slow).
6. **Reproducibility.**
   - Same `random_state` gives identical bootstrap bands, `n_imputations` SDs and auto p-values.
   - `np.random.get_state()` is unchanged by any call.
7. **Input validation.**
   - Errors: span < 3 years; < 24 detected values; > 80 % censored; `trend_span_years < 2`; `seasonal_window_years < 3`; values ≤ 0 under `transform="log"`; raw strings (`TypeError` pointing to `prepare_censored_data`); non-integer `block_years`.
   - Allowed: unsorted input is sorted, and `order` maps back correctly. Duplicate timestamps are accepted and noted.
8. **No regression.** The full existing MannKS test suite passes unchanged. The environment must first install MannKS's declared dependencies (the review container lacked scipy).
9. **Performance (smoke).** Global fit on n = 600 uncensored finishes in < 1 s on CI.

---

## 6. Validation before release (pre-register, then run once)

Write the plan file (designs, seeds, metrics, pass criteria, compute budget) and **commit it before running anything except the timing pilot in §6B.4**. Report every result, including failures. Do not tune constants on these designs. If anything changes, validate it on new designs.

### 6A. Decomposition accuracy (no regression vs §8)
- **Data:** the 17 real records (§6C cleaning). Include the 20 synthetic designs only if the author supplies them (§2).
- **Splits:** for each record, 20 random 80/20 splits with seeds 0–19 (`numpy.random.default_rng(seed).permutation`).
- **Fits:** train both methods on the 80 %.
  - Reference: §8 code.
  - New, configuration D: package defaults.
  - New, configuration L ("like-for-like"): `n_harmonics=4`.
- **Scoring:**
  - Predict held-out points with `trend + seasonal`.
  - Score only held-out **detected** values lying inside the training span. Points outside the span are dropped from scoring for every method.
  - Metric: RMSE on ln(concentration).
  - Also report, descriptively, the share of held-out censored values whose prediction lies below ln DL.
- **Per-record ratio:** `R_rec = mean over splits of RMSE_new / mean over splits of RMSE_ref`.
- **Pass (configuration D):** geometric mean of `R_rec` over the 17 records ≤ 1.02, **and** `max R_rec` ≤ 1.05.
- Configuration L has no pass rule; it is reported so that a failure of D can be attributed (K selection vs. the smoother and robustness changes).
- **Expected difference to report:** statsmodels `lowess(frac=…)` uses a nearest-neighbour window, which near the record ends extends inward to about 3 years one-sided. The new time window is 1.5 years one-sided, so higher end variance is expected.

### 6B. Trend inference (Monte Carlo)

#### 6B.1 Data-generating process **[AUTHOR DECISION: all values below]**
- **Record:** 20 years, 2000-01-01 to 2019-12-31. Work on the ln scale with mean level 0.
- **Noise SD:** σ = 0.5, the marginal SD for all noise types.
- **Seasonality:**
  - Fixed: `s(t) = 0.6 cos 2πt + 0.3 sin 2πt + 0.2 cos 4πt`.
  - Evolving: the first-harmonic amplitude scales linearly from 0.5× to 1.5× across the record, and its phase shifts by 30 days over the record. The second harmonic is fixed.
- **Sampling:**
  - Monthly: one sample per month, on day 15 ± U(−5, 5).
  - Random 12 per year: uniform times.
  - Sparse 4 per year: Feb, May, Aug and Nov, on day 15 ± U(−10, 10).
  - Monthly plus storm bursts: monthly as above, plus 6 bursts per year at uniform start times, each with U{3..6} samples within 3 days. There is no storm effect on concentration, so this is a pure sampling design.
  - Winter-sparse: monthly in March–November, and each of December–February sampled with probability 0.25.
- **Noise:**
  - Gaussian.
  - t₃ scaled to SD σ (σ·t₃/√3).
  - 10 % outliers: Gaussian plus ±6σ shifts with a random sign.
  - AR(1) with ρ = 0.3 and 0.6, defined in **continuous time** as an Ornstein-Uhlenbeck process whose correlation at a lag of 1/12 year is ρ, so it is well defined for irregular times.
- **Censoring (left):**
  - 0 %.
  - 25 % and 50 %: the DL is a constant set per run at that quantile of the simulated ln values.
  - DL change: the second-half DL is the first-half DL − ln 2. The first-half DL is solved per run so that 25 % of the whole record is censored.
- **Trend (ln units per year):** β = 0 (size), 0.0125 (0.25σ per 10 years) and 0.025 (0.5σ per 10 years).
- **Truth for coverage:** β on the ln scale, per year.

#### 6B.2 Methods
1. **SK (existing default):** `seasonal_trend_test(x, t, season_type='month')` with all other MannKS defaults (`sens_slope_method='ats'`, `agg_method='none'`, `min_size_per_season=5`); for the sparse 4/yr design, `season_type='quarter'`.
2. **D-global:** `decompose` with defaults, then `deseasonalized_trend_test(res, slope_scaling='year')` on the transformed scale.
3. **D-auto:** the same with `seasonal="auto"`, `n_boot_auto=99`, on the auto subset only (§6B.3).
- **Autocorrelation:** in AR(1) cells, run methods 1 and 2 with `autocorr_method='none'` and `'yue_wang'`. Run `'block_bootstrap'` on 100 runs per AR cell only, because of cost.

#### 6B.3 Cells and run counts
- Full factorial for methods 1 and 2: 5 sampling × 2 seasonality × 5 noise × 4 censoring × 3 trend = 600 cells.
  - Size cells with independent noise (β = 0; Gaussian, t₃ and outliers): 120 cells × **1000 runs**.
  - All other cells: 480 cells × 500 runs.
  - Total: about 360,000 decompositions.
- Auto subset (method 3): {fixed, evolving} seasonality × {monthly, sparse 4/yr, storm bursts} × Gaussian noise × {0 %, 25 %} censoring × β = 0. That is 12 cells × **100 runs**, each with 2 × 100 refits: about 240,000 fits.
- Seeds: run r of cell c uses `SeedSequence([c, r])`. Cell indices are fixed in the plan file.

#### 6B.4 Compute budget and timing pilot
- The reviewer's naive prototype took about 1 s per global fit at n = 600, so the plan above is of the order of **100–300 CPU-hours** before any vectorisation gain.
- That needs a multi-core machine or cluster; the 4-core review container is not enough. **[AUTHOR DECISION]** Provide compute, or approve a reduced design (e.g. drop the random-12/yr design and use 500 runs in size cells, with the pass rule below unchanged).
- Before freezing the plan, run a **timing pilot** of 5 runs per cell type. Record runtimes only, and **discard the outputs without inspecting the results**. Use the timings to confirm the budget.

#### 6B.5 Pass criteria
Applies to methods 1 and 2 (method 2 is the one that must pass); `α = 0.05` throughout.
- **Size, independent noise.** A cell **fails** if both:
  - its rejection rate is outside [0.03, 0.07]; and
  - an exact two-sided binomial test of rate = 0.05 rejects after a **Holm correction** across the 120 size cells at family-wise α = 0.05.

  In addition, the rejection rate pooled over all 120 cells must lie in [0.04, 0.06].
- **Size, AR(1):** report the rate for each autocorrelation option and flag any cell above 0.10. No pass rule.
- **Coverage:** the 95 % ATS/Sen slope CI covers β in ≥ 0.90 of runs in every cell (β = 0 and β > 0, all noise types). With 500 runs, a true coverage of 0.95 essentially never falls below 0.90 by chance.
- **False trend under DL change** (β = 0, independent noise, DL-change censoring): a cell fails if its rate is > 0.07 and a one-sided binomial test of rate ≤ 0.05 rejects after Holm correction within this family.
- **Auto selection:** time-varying is chosen in ≤ 10 % of runs **pooled over the 6 fixed-seasonality auto cells** (600 runs). Per-cell rates are reported.
- **Descriptive (no pass rule):** power of methods 2 and 3 relative to SK per cell, auto selection rates in the evolving cells, and runtimes.

### 6C. Real records (descriptive)

**Panel:** sites 01491000, 01578310, 01594440, 01646580, 01668000, 02035000, 05465500, 05586100 and 07374000, each with parameter codes 00631 (nitrate+nitrite, mg/L as N) and 00665 (total phosphorus, mg/L as P). Drop 07374000/00665, which has too few samples.

**Data snapshot:**
- Download once from the legacy WQP endpoint `https://www.waterqualitydata.us/data/Result/search?siteid=USGS-<site>&pCode=<code>&mimeType=csv&zip=no`.
- Record the download date.
- Commit the raw CSVs and their SHA-256 hashes under the validation folder.
- The legacy endpoint has not received USGS data since early 2024 (e.g. 01491000/00631 ends 2023-09-25). Say so in the report.

**Cleaning** (fallback if the author's original script is not supplied):
- `ActivityMediaName == "Water"`.
- Keep `ActivityTypeCode` values beginning with `"Sample"`; this drops `Quality Control*` and `Not determined`.
- Units, case-insensitive: 00631 requires `mg/l as N`; 00665 requires `mg/l as P` or `mg/l`.
  - Take the unit from `ResultMeasure/MeasureUnitCode`.
  - For non-detects, take it from `DetectionQuantitationLimitMeasure/MeasureUnitCode`.
  - Drop other units.
- **Non-detects:** `ResultDetectionConditionText` in {`Not Detected`, `Below Detection Limit`, `Below Reporting Limit`, `Detected Not Quantified`, `Present Below Quantification Limit`}. These become left-censored at `DetectionQuantitationLimitMeasure/MeasureValue`. Drop rows with no limit.
- **Same-timestamp duplicates:**
  - all detected: arithmetic mean;
  - any censored: MannKS's censored median rule (`_helpers._aggregate_censored_median`), with **no substitution**.
- Missing sample time → 12:00 local time; time zones are ignored.
- Keep records with ≥ 300 samples over ≥ 15 years after cleaning.
- Analyse ln(concentration).
- Report the censored share per record. Nitrate records are almost uncensored (01491000/00631 has 1 non-detect in 925 rows), so censoring is exercised mainly by the total-phosphorus records.

**Outputs:**
- held-out prediction as in §6A;
- trend test results for SK vs D-global on each record (p-value, slope with CI on the ln scale per year, direction);
- `K_max`, `n_harmonics_used`, share of zero weights, acf1, and any widenings.

---

## 7. Documentation

- **API reference** for `decompose`, `DecompositionResult` and `deseasonalized_trend_test`.
- **Worked example** on a real record (e.g. USGS 01491000 nitrate):
  - the plot;
  - seasonal amplitude (ln and ratio) and peak day;
  - the deseasonalised series passed to a trend test, side by side with Seasonal Kendall;
  - the slope interpreted as approximate % per year.
- **"When to use it":**
  - Seasonal Kendall stays the default trend test for seasonal data.
  - Use `decompose` to describe and visualise seasonality, to deseasonalise for segmented or rolling analyses, and to check robustness to outliers.
  - Use time-varying mode only when the seasonal pattern is known or tested to change.
- **Scale and censoring notes:**
  - Use the transformed scale and `sens_slope_method='ats'` for censored data.
  - MK p-values are identical on both scales; slopes are not.
  - Imputed values are never used as data.
- **Limitations:**
  - Sinusoidal harmonics cannot follow abrupt seasonal switches (e.g. freeze/thaw).
  - Bands exclude smoothing bias at the ends and in long gaps, and condition on K.
  - The imputation model assumes normal, homoscedastic residuals.
  - `*_imputation_sd` is a lower bound.
  - Censored points cannot be down-weighted.
  - The time-varying window may be wider than requested (report `seasonal_window_used`) and is biased near the ends.
  - Deseasonalised Mann-Kendall results depend on the §6B validation.

**Claims you may make:** "robust, censoring-aware harmonic + local-regression decomposition for irregularly sampled data, integrated with MannKS trend tests". After §6B passes you may add: "validated Type I error and slope-interval coverage under the designs tested".

**Claims you may not make:** "new method", "best decomposition", or that it outperforms Seasonal Kendall for trend testing, unless §6B shows that.

---

## 8. Reference implementation (validated baseline; regression target for §6A)

The code is unchanged. It is used only in validation scripts, and needs `statsmodels` pinned to an exact version recorded in the plan file.

```python
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess  # used only by this reference

def harm(t):
    return np.column_stack([f(2*np.pi*k*t) for k in range(1, 5) for f in (np.cos, np.sin)])

def reference_fit(t, y, targets):
    """Global harmonics 1-4 + robust LOWESS trend (3-yr span), 3 backfitting passes.
    t: decimal years (sorted), y: ln(values) with non-detects at ln(DL/2) (reference only)."""
    frac = min(1.0, 3.0 / (t[-1] - t[0]))
    H = np.column_stack([np.ones_like(t), harm(t)])
    trend = lowess(y, t, frac=frac, it=3, return_sorted=False)
    for _ in range(3):
        beta = np.linalg.lstsq(H, y - trend, rcond=None)[0]
        seas = H[:, 1:] @ beta[1:]
        trend = lowess(y - seas, t, frac=frac, it=3, return_sorted=False)
    beta = np.linalg.lstsq(H, y - trend, rcond=None)[0]
    return np.interp(targets, t, trend), harm(targets) @ beta[1:]   # trend, seasonal at targets
```

How the reference differs from the MannKS implementation (all expected, which is why §6A checks for regression):
- Robustness is applied only inside the smoother, rather than through shared weights (§4.5).
- Non-detects are set to DL/2, rather than imputed conditionally with `ndimpute` (§4.6).
- `frac` defines a nearest-neighbour window, rather than a time window (§4.3).
- It uses 3 fixed backfitting passes, rather than convergence (§4.4).
- K is fixed at 4, rather than chosen automatically (§4.2).
- `np.interp` clamps outside the span, whereas `predict` returns NaN; §6A scores only in-span points.

---

## 9. Deliverables and order of work

0. **Author decisions:** confirm or replace every **[AUTHOR DECISION]** item, and supply the §2 materials if available.
1. **Plan:** a short note mapping §3–§4 onto the MannKS conventions in §1.1, with file and function references. Commit it before any code.
2. **Core:** §4.1–4.6 (global mode), with tests 1–4 and 6–9.
3. **Bootstrap bands:** §4.9.
4. **Time-varying and auto modes:** §4.7–4.8, with test 5.
5. **Pre-registered validation plan** for §6, including the timing pilot (§6B.4) and the compute decision. Commit it, run it once, and write the report with figures and the full metric tables.
6. **Documentation:** §7, plus a CHANGELOG entry marking the module as opt-in, with no change to existing results, and the new `censored` extra.

Stop and report back after step 5 if any §6A or §6B pass criterion fails. Do not adjust constants to make it pass.
