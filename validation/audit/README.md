# Pre-release correctness audit harness (v0.5.0)

Scripts used to produce `AUDIT_REPORT.md` (repository root). Everything here compares
MannKS against **independent** references: the LWP-TRENDS v2502 R code run live via
`Rscript`, `NADA::cenken`, exact permutation distributions, `pymannkendall`, and
Monte Carlo simulation. Nothing is compared with MannKS's own past output.

| Script | What it checks |
|---|---|
| `kernel_parity.py` + `r_kernel.R` | R `GetKendal` / `GetInterObservationSlopes` vs `_mk_score_and_var_censored` / `_sens_estimator_censored` on 300 random censored datasets (S, VarS, censored Sen slope) |
| `variance_truth.py` | Var(S) formulas vs the exact permutation variance of S; tie-corrected Var(S) vs pymannkendall |
| `run_parity.py` + `r_wrapper.R` | Wrapper-level parity: R `MannKendall`/`SenSlope`/`SeasonalKendall`/`SeasonalSenSlope` vs `trend_test`/`seasonal_trend_test` in LWP mode, 33 cases (synthetic edge cases plus the four LWP example datasets) |
| `censored_slope_study.py` + `r_cenken.R` | ATS slope vs `NADA::cenken`; bias of the censored slope estimators vs a known true slope |
| `regional_parity.py` + `r_regional.R` | `regional_test` vs R `getTAU` |
| `senprob_parity.py` | `sen_probability` vs R `Sen_Probability` |
| `monte_carlo.py` | Type-I error, p-value uniformity, CI coverage, slope bias, fast-mode error, autocorrelation methods |
| `bug_repros.py`, `code_checks.py`, `misc_checks.py` | Minimal reproductions of individual findings |
| `make_tables.py` | Renders `out/parity_tables.md` |
| `proposed_variance_fix.diff` | The 6-line fix that brings Var(S) into exact agreement with R (not applied to the package) |

`out/` holds the outputs from the audit run (R 4.3.3, NADA 1.6-1.2, numpy 2.5,
pandas 3.0, scipy installed in the audit container).

Run everything with `./run_all.sh` after setting
`LWP_R=/path/to/MannKS_dev/Example_Files/R/LWPTrends_v2502/LWPTrends_v2502.r`.
To evaluate a patched package, point `MANNKS_SRC` at the directory that contains it.
