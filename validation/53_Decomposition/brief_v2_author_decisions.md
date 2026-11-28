# Author decisions and supplied materials for brief v2

This file answers every **[AUTHOR DECISION]** and **[AUTHOR TO SUPPLY]** item in `MannKS_decomposition_agent_brief_v2.md`. Brief v2 is accepted as written, apart from what is stated below.

All files named here are on branch `claude/hopeful-johnson-hnarah` of `LukeAFullard/tspec`.

## A. Supplied materials (§2)

**A1. The 20 synthetic designs**
- Specification: `validation/decomposition_heldout_plan_3.md` §2.
- Code: `validation/bench_decomp_v3.py`, function `design(k, rng)` for k = 1..20.
- Replicate seed: replicate `r` of design `k` uses `numpy.random.default_rng(990_000 + 1_000·k + r)`, with 100 replicates per design.
- D07 and D08 use real sampling times: the last 25 years of `validation/data/wq_panel/01646580_00631.csv` and `05465500_00631.csv`.
- Metric: RMSE of the estimated signal (trend + seasonal) against the true signal at the observation times.
- D18 (censoring): the reference receives `DL − ln 2` (half the DL on the concentration scale) for censored values. The new implementation receives the censored values as censored.

**A2. Real-record data snapshot**
- Raw WQP downloads: `validation/data/wq_panel_raw/` (18 CSVs, downloaded 2026-11-28 from the legacy endpoint). SHA-256 hashes are in `SHA256SUMS`.
- Cleaning script, exactly as run: `validation/data/clean_wq_panel.py`. Running `python -I validation/data/clean_wq_panel.py validation/data/wq_panel_raw <out>` reproduces `validation/data/wq_panel/` byte for byte (verified).
- Known differences from the v2 fallback rules:
  - `Not determined` activity types are kept.
  - Any unit starting with `mg/l` is kept.
  - Duplicates are averaged.
- The snapshot contains **no** partly censored duplicate groups (verified: 0), so no substitution occurred.

**A3. Reference results**
- File: `validation/reports/decomp_v3_results.json`, with per-split RMSE for every method, including the reference (`HLR`).
- These used a different split scheme: `default_rng(995_000 + 1_000·k + split)` with a Bernoulli(0.2) test mask.
- Do not compare against these stored numbers. Recompute the reference and the new implementation in the same harness, on the same splits (§6A as written).
- Versions used: statsmodels 0.15.0, numpy, scipy and pandas as in that environment. Pin statsmodels 0.15.0 for the §8 reference.

## B. Decisions

| Item | Decision |
|---|---|
| §3 `random_state=None` | **Fresh entropy**, matching MannKS `_bootstrap.py`. Document it; all tests and validation pass explicit seeds. |
| §4.7 `c_ess` | **5.** Accepted, and report `seasonal_window_used`. |
| §6B.1 data-generating process | **Accepted as written.** |
| §6B.4 compute | Vectorise first (§4.3 target), then run the timing pilot. If the full design costs **≤ 48 CPU-hours** on the available machine, run it in full. Otherwise use the **reduced design**: drop the random-12/yr sampling design, use 500 runs in size cells (pass rules unchanged), and run the auto subset with 50 runs and `n_boot_auto=99`. Record which design was used in the plan file **before** running. |
| §6A synthetic designs | Now available (A1), so include them. Use the same ratio statistic `R_design` as for real records. **Descriptive**, no pass rule, but flag any design with `R_design > 1.10` and explain it in the report. The pass rule stays on the 17 real records only. |
| §6A and §6C data | Use the committed snapshot (A2) for both, for reproducibility and comparability with v3. Optionally report §6C a second time with the v2 fallback cleaning as a sensitivity check. State that the legacy endpoint has no USGS data after early 2024. |
| §9 step 0 | Done with this file. Proceed to step 1. |

## C. One clarification to v2

§6A scores held-out **detected** values only. Across the 17 records the censored share is small (0–5.0% per record; the maximum is Rappahannock nitrate, 01668000/00631), so this does not bias the comparison. Report the censored share per record (§6C already asks for it).
