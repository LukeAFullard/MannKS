# Seasonal–trend decomposition: implementation plan (brief v2, §9 step 1)

This note maps brief v2 §3–§4 onto MannKS 0.7.2 conventions. It is committed before any code.

- Specification: `brief_v2.md`, as amended by `brief_v2_author_decisions.md`. Both are in this folder.
- Precedence: where this note and the brief disagree, the brief wins. Any deviation is recorded in §9 of this note before it is made.

## 1. Files

| File | Contents |
|---|---|
| `MannKS/decomposition.py` | Public API: `decompose`, `DecompositionResult`, `deseasonalized_trend_test`. Input validation, warning capture, result assembly. |
| `MannKS/_decompose.py` | Numerics: decimal years, phase coverage, harmonic basis, local-linear smoother, backfitting, fitting schedule, K selection, time-varying mode, auto test, block bootstrap. Plain numpy/scipy; no pandas inside the loops. |
| `MannKS/_decompose_censored.py` | The only place that touches `ndimpute`: lazy import, status conversion, start and conditional imputation, warning de-duplication. |
| `MannKS/plotting.py` | New `plot_decomposition(result, save_path=None, axes=None)`. `DecompositionResult.plot` delegates to it. |
| `MannKS/__init__.py` | Export `decompose`, `DecompositionResult`, `deseasonalized_trend_test`, `plot_decomposition`. Nothing existing changes. |
| `pyproject.toml` | `ndimpute = {version = ">=0.1.1,<0.2", optional = true}` and `[tool.poetry.extras] censored = ["ndimpute"]`. Core floors unchanged. |
| `.github/workflows/tests.yml` | The "latest" and "scipy-1.14" jobs install `.[censored]`. The "minimum" job stays as is, so censored tests skip there (ndimpute needs scipy ≥ 1.13). |
| `tests/test_decomposition_*.py` | Brief §5 tests 1–9 (see §8). |
| `tests/conftest.py` | Register a `slow` marker (`pytest_configure`). The existing autouse global-seed fixture is left untouched. |
| `validation/53_Decomposition/` | This note, the §6 pre-registered plan (step 5), harness, report. |

## 2. Input handling (brief §4.1; MannKS `_helpers._prepare_data`)

- **Positional order.** Build a working copy of `x` with a positional index, so `order` is positional even when the user's DataFrame has a custom index.
- **Preparation.** Call `_prepare_data(x, t, hicensor=False)`. It returns `(data, is_datetime)`, where `data` has the columns `value`, `censored`, `cen_type`, `t_original` and `t`:
  - `t` is seconds since 1970 for datetimes, or the numeric value as given;
  - it raises `TypeError` for raw strings, pointing to `prepare_censored_data`;
  - it drops NaN values silently, and infinite values or missing times with a warning.
- **Sorting.** Stable sort on `t`. `order` is the sorted positional index.
- **Decimal years.**
  - Datetimes: `2000 + (t_sec − 946_684_800) / (365.2425 × 86_400)`, where 946,684,800 is 2000-01-01T00:00:00 UTC in Unix seconds.
  - Numeric: used as given. Warn if the span is > 1000 (brief §3).
- **Times returned to users.** `res.times` and the `t` returned by `deseasonalized_series()` are the caller's original `t`, taken positionally with `order` in the caller's container type (numpy array, `pd.Series` or `pd.Index`). That keeps tz-awareness and lets `trend_test(..., slope_scaling=...)` behave exactly as it does on raw input.
- **Transform.**
  - `transform="log"`: raise `ValueError` if any value (detected value or limit) is ≤ 0, then `x = ln(value)`.
  - `transform="none"`: identity.
- **Validation order** (each failure raises `ValueError` with the observed number):
  1. option values;
  2. `trend_span_years ≥ 2`, `seasonal_window_years ≥ 3`, and `block_years` a positive integer;
  3. span ≥ 3 years;
  4. ≥ 24 detected values;
  5. censored share ≤ 80 % (warn above 50 %).
- **Duplicate timestamps** are kept, with a note added to `analysis_notes`. This matches `trend_test`'s "tied timestamps present without aggregation".

## 3. Censoring (brief §4.6; `ndimpute` 0.1.1)

- The censoring representation is MannKS's `cen_type` array (`'lt'`, `'gt'`, `'not'`). It is passed to `ndimpute.impute` **as strings**; ndimpute's `_STATUS_STRINGS` maps these to −1/+1/0.
- `_require_ndimpute()` imports ndimpute on first use, only when `censored.any()`. If the import fails it raises `ImportError("Censored input needs the optional dependency ndimpute: pip install 'mannks[censored]'")`.
- **Start imputation:**
  ```python
  impute(x_lim, status=cen_type, method="ros", dist="normal", impute_type="mean", right_impute_type="mean")
  ```
- **Conditional imputation:**
  ```python
  impute(resid, status=cen_type, method=censored_method, dist="normal", impute_type="mean", random_state=seed)
  ```
  plus `right_impute_type="mean"` for ROS.
  - Detected rows with `w = 0` are passed as NaN. If that leaves fewer than 24 detected values, all detected values are passed instead and a warning is raised.
  - Only censored rows are overwritten.
- **Stochastic draws** (bootstrap replicates and `n_imputations`): the same calls with `impute_type="stochastic"`, plus `right_impute_type="stochastic"` for ROS, and a child seed.
- **Whitelist.** One internal function, `_ndimpute_call(**kw)`, makes every ndimpute call. It rejects `method="substitution"` and `lwp_compat`, and test 3(f) monkeypatches it. No other module imports ndimpute.
- **ndimpute warnings.** Each call runs under `warnings.catch_warnings(record=True)`. Messages are de-duplicated with counts into `diagnostics["ndimpute"]["warnings"]`, and the final call's `df.attrs` go into `diagnostics["ndimpute"]["attrs"]`. A single summary `UserWarning` is re-emitted if any ndimpute warning occurred, so a 50-iteration loop does not produce 50 copies.
- **Deseasonalised output** (brief §4.6.5):
  - transformed scale: `value − seasonal`, applied to both detected values and limits;
  - original scale: `value · exp(−seasonal)`;
  - `censored` and `cen_type` are copied unchanged.

## 4. Result object (MannKS pattern: namedtuple subclass, as `SegmentedTrendResult`)

`_Decomposition_Tuple = namedtuple('Decomposition', [...])`, with this field order:

- **Per observation** (time-sorted, transformed scale): `times`, `t_years`, `order`, `values`, `censored`, `cen_type`, `trend`, `seasonal`, `remainder`, `weights`, `fitted_values`.
- **Model:** `transform`, `n_harmonics_used`, `K_max`, `seasonal_mode_used`, `seasonal_constancy_pvalue`, `coefficients` (global: a length-2K vector; time-varying: grid times plus a grid × 2K array).
- **Descriptors:** `seasonal_amplitude`, `seasonal_amplitude_ratio`, `peak_day_of_year`, `trough_day_of_year`, `seasonal_by_year` (a DataFrame, or `None` in global mode).
- **Uncertainty:** `trend_low`, `trend_high`, `trend_simul_low`, `trend_simul_high`, `seasonal_low`, `seasonal_high`, `trend_imputation_sd`, `seasonal_imputation_sd`. Each is `None` when it was not requested.
- **Bookkeeping:** `diagnostics` (dict), `warnings` (list of str), `analysis_notes` (list of str), plus the private `_settings`, holding the arguments `predict` and the bootstrap need.

`class DecompositionResult(_Decomposition_Tuple)` adds four methods:
- `predict(t_new)`: accepts the same time types as `decompose`. Outside `[t_min, t_max]` it returns NaN with a `UserWarning`.
- `to_dataframe()`
- `plot(save_path=None, axes=None)`
- `deseasonalized_series(scale="transformed")`: returns `(x_df, t)`.

`deseasonalized_trend_test(res, scale="transformed", **kw)`:
1. If censored values are present and `kw.get('sens_slope_method', 'ats') != 'ats'`, raise `ValueError`.
2. If `hicensor` is set, warn.
3. Return `trend_test(x_df, t, **kw)`, unchanged.

## 5. Warnings and notes (MannKS pattern from `trend_test` and `segmented_trend_test`)

- The body of `decompose` runs under `warnings.catch_warnings(record=True)` with `simplefilter("always")`.
- Records are kept as `(message, category)` pairs, as `segmented_trend_test` does since 0.7.2, so categories survive re-emission.
- The message strings go into `res.warnings`, and short notes (e.g. `"K limited to 2 by phase coverage (max gap 0.21 yr)"`) into `analysis_notes`.
- All records are re-emitted after the result is built.
- The brief §4.10 conditions are checked in one function, `_collect_warnings(diag, settings)`, so tests can assert on them directly.
- `acf1_remainder` = `MannKS._autocorr.estimate_acf(remainder[detected], max_lag=1)[0][1]`, on time-ordered detected remainders.

## 6. Plotting (MannKS pattern from `plot_segmented_trend`)

`plot_decomposition(result, save_path=None, axes=None)` draws five axes:
1. data and fit, with censored points drawn as open markers at their limits (down-triangles for `lt`, up-triangles for `gt`);
2. trend with its band, if any;
3. seasonal;
4. remainder, with zero-weight points highlighted;
5. the seasonal cycle by day of year.

It returns `(fig, axes)` when `save_path is None`. Otherwise it saves, closes the figures it created and returns `None`. matplotlib is imported at module level in `plotting.py`, as it already is.

## 7. Numerics (brief §4.2–4.9)

- **Smoother** (§4.3), `_local_linear(t, z, w, h, min_pts=10)`:
  - processes targets in chunks of 256, so memory is O(256·n) and n = 50,000 is feasible;
  - tricube kernel × robustness weights; closed-form 2×2 weighted least squares on centred `t`;
  - a mask of targets that fail the safeguards (fewer than 10 non-zero weights, fewer than 3 distinct times, or condition number ≥ 1e8) is re-solved at `h × 1.1^k`, up to the record span, then falls back to local-constant;
  - returns `(fit, hat_diag, n_widen, n_fallback)`.
  - Performance target (brief §4.3): ≤ 0.3 s at n = 600, measured in test 9.
- **Basis and coverage** (§4.2): `_harmonics(t, K)`; `_phase_coverage(t_years, calendar_year)` returns `K_max` and an info dict. "Supported" phases use `m = max(2, ceil(0.1 · n_years))` distinct years within ±1/48 yr.
- **Backfitting** (§4.4): `_backfit(t, x, w, basis_fn, state, tol, max_iter)`.
  - `state` carries the warm start.
  - In global mode the seasonal step is a single weighted lstsq on the 2K-column basis.
  - In time-varying mode it is `_local_harmonics` on the 0.1-yr grid, followed by linear interpolation of the coefficients.
- **Schedule** (§4.5): `_fit(...)` implements the pseudocode literally. Iteration counts and convergence flags go into `diagnostics["convergence"]`.
- **K selection** (§4.2): `_select_K` follows the stated sequence (full fit at `K_max`, then BIC with weights and imputations fixed, then a refit if K changes). `tr(S_trend)` = the sum of `hat_diag` from the final smoother pass.
- **Time-varying mode** (§4.7): `_local_harmonics(t, r, w, grid, H, K, c_ess=5)` with the three safeguards. The per-grid-point H goes into `diagnostics["seasonal_window_used"]`.
- **Auto** (§4.8): `_auto_select(...)` uses the §4.9 generator with `n_boot_auto` replicates. p-value = `(1 + #{F* ≥ F}) / (B + 1)`.
- **Bootstrap** (§4.9): `_block_bootstrap_series(fit, rng, L)` returns `(t*, x*, censored*, cen_type*)`, using time-shifted whole blocks and preserved residual-scale limits. Each replicate's refit uses `_fit` with K fixed, then one stochastic re-imputation and one backfit. Components are evaluated at the original times via the same interpolation as `predict`.
- **Randomness:** `ss = np.random.SeedSequence(random_state)`, where `None` means fresh entropy (author decision B). `ss.spawn(...)` gives independent children for: deterministic imputation calls (seed unused, but passed for completeness), `n_imputations`, bootstrap replicates and auto replicates. The global `np.random` is never used.

## 8. Tests (brief §5) → files

| Brief test | File | Notes |
|---|---|---|
| 1 exact recovery | `test_decomposition_core.py` | Both `robust=False` and `robust=True` (scale guard). |
| 2 robustness | `test_decomposition_core.py` | 20 seeds; mean-ratio criterion. |
| 3 censoring (a)–(g) | `test_decomposition_censored.py` | `pytest.importorskip("ndimpute")`, except (g), which blocks the import via `monkeypatch.setitem(sys.modules, "ndimpute", None)`. |
| 4 irregular sampling | `test_decomposition_core.py` | Includes the singular-window case. |
| 5 time-varying / auto | `test_decomposition_tv.py` | 5(c) uses `n_boot_auto=49` and is marked `slow`. |
| 6 reproducibility | `test_decomposition_core.py` | Compares `np.random.get_state()` before and after each call. The autouse fixture seeds globally beforehand, so this is meaningful. |
| 7 input validation | `test_decomposition_inputs.py` | |
| 8 no regression | existing suite | Run in full before each commit of library code. |
| 9 performance | `test_decomposition_core.py` | Asserts < 1 s at n = 600 (CI-safe). Reports the time against the 0.3 s target. |

## 9. Deviations, risks and open items

1. **`tspec` materials (author decisions A1–A3) are not yet reachable** from this session, because attaching `LukeAFullard/tspec` was refused. They are not needed until step 5. Before step 5, the designs code (`bench_decomp_v3.py`), the panel snapshot (`wq_panel_raw/`, `SHA256SUMS`), `clean_wq_panel.py` and the cleaned `wq_panel/` must be copied into `validation/53_Decomposition/`, so the validation can be reproduced from this repository alone. Hashes are re-verified on copy.
2. **Long validation runs:** a ≤ 48 CPU-hour design is ≥ 12 h wall-clock on a 4-core cloud container, which can be reclaimed when idle. The harness writes one results file per cell and is resumable; results are committed in batches.
3. **Reduced-design sensitivity** (author decision B, §6B.4): with 500 runs per size cell, the Holm-corrected binomial rule has at least a 50 % chance of flagging a cell only if its true size is above about 0.085, against about 0.075 with 1000 runs (threshold 0.05/120; Monte Carlo SE 0.0097 vs 0.0069). The pooled [0.04, 0.06] criterion is the main guard against small, widespread size inflation. The step-5 plan file records which design was used and states this.
4. **The censoring path is barely exercised by the §6A pass rule:** the real records are 0–5 % censored. Its accuracy evidence comes from synthetic D18 (descriptive) and §6B. The report says so.
5. **No other deviations** from brief v2 were planned when this note was first committed.

### Deviations and additions made during implementation (steps 2–4)

These were added after the note's first commit, each before the code that uses it.

- **`settings`, not `_settings`.** namedtuple field names cannot start with an underscore, so the private settings field is called `settings`.
- **Span-capped time-varying windows use the global WLS call.** For a grid point whose window has widened to the record span, the coefficients come from the same weighted least-squares call as global mode (`lstsq` on the weighted design), not from the normal equations. This makes "cap = global mode" (brief §4.7) exact, and it is numerically better for poorly conditioned designs.
- **The time-varying condition number is checked on the Gram matrix.** The check uses `cond(BᵀWB) < 1e12`, which is `cond(design)² < (1e6)²`, so it is equivalent to brief §4.7's design condition number below 1e6.
- **Auto mode needs at least 20 bootstrap replicates to choose time-varying.** With `n_boot_auto < 20` the smallest attainable p-value, `1/(B+1)`, is at least 0.05, so time-varying can never be selected. `decompose` warns in that case. This is a mathematical consequence of brief §4.8, not a constant change.
- **Failed bootstrap replicates are recorded.** Replicates that raise an exception are skipped and counted (`diagnostics['bootstrap'|'auto']['n_failed']`), and the first error message is kept. A warning is raised when any replicate failed.
- **Selecting time-varying in auto mode refits fully.** When auto mode selects time-varying, the result comes from the full time-varying pipeline (robustness and imputation), the same as `seasonal="time_varying"`. The F test itself uses the global fit's weights and imputations, as brief §4.8 specifies.
- **Top-level descriptors in time-varying mode.** `seasonal_amplitude` and `seasonal_amplitude_ratio` are NaN, and the peak/trough days are `None`. The per-year values are in `seasonal_by_year`.
- **Period-2 cycles in the imputation loop.** With `censored_method="ros"`, the imputation map is piecewise constant, because ROS plotting positions depend on the discrete rank order of residuals and limits. The loop can then fall into a period-2 cycle: in one mixed-censoring example, 2 of 46 values flipped by 0.003. When the new imputation equals the one two steps earlier within `impute_tol`, the loop takes the mid-point of the two states, runs a final backfit, counts the event in `diagnostics['convergence']['impute_cycles']` and adds an analysis note. Parametric imputation (the default) is smooth and was not seen to cycle.
- **ndimpute cost.** Censored fits are dominated by ndimpute's censored maximum-likelihood fit (scipy `CensoredData`, about 30 ms per call at n ≈ 200; one call per imputation step). A censored global fit at n = 240 takes about 2.5–5 s, against about 0.06–0.2 s uncensored. This matters for the §6B budget (author decision B) and will be measured in the timing pilot.
