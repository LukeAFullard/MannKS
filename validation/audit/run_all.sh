#!/usr/bin/env bash
# Reproduce every audit result.  Requirements: R >= 4.3 with plyr, tidyr, viridis, NADA,
# lubridate, gam; Python deps of MannKS plus pymannkendall.  Set LWP_R to the path of
# LWPTrends_v2502.r (from the MannKS_dev repository).
# Optional: MANNKS_SRC=/path/to/alternative/package/parent to test a patched copy.
set -euo pipefail
cd "$(dirname "$0")"
python kernel_parity.py          | tee out/kernel_parity.txt
python variance_truth.py         | tee out/variance_truth.txt
python run_parity.py             | tee out/run_parity.txt      # ~8 min (R long series)
python censored_slope_study.py   | tee out/censored_slope_study.txt
python regional_parity.py        | tee out/regional_parity.txt
python misc_checks.py            | tee out/misc_checks.txt
python senprob_parity.py         | tee out/senprob_parity.txt
python bug_repros.py             | tee out/bug_repros.txt
python code_checks.py            | tee out/code_checks.txt
python monte_carlo.py            | tee out/mc_current.log    # ~5 min
python make_tables.py > /dev/null   # needs out/parity_results_varfix.csv (run_parity with MANNKS_SRC=patched, PARITY_TAG=_varfix)
