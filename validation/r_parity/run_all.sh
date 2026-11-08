#!/usr/bin/env bash
# Run every parity / calibration check.  Requires R (see README.md) and
# LWP_R=/path/to/LWPTrends_v2502.r.  Outputs go to out/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
python kernel_parity.py        | tee out/kernel_parity.txt
python variance_truth.py       | tee out/variance_truth.txt     # uses out/kernel_* from kernel_parity.py
python run_parity.py           | tee out/run_parity.txt         # ~10 min (R on the 6000-point series)
python censored_slope_study.py | tee out/censored_slope_study.txt
python regional_parity.py      | tee out/regional_parity.txt
python monte_carlo.py          | tee out/monte_carlo.txt        # ~10 min
