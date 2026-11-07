"""Shared helpers for the pre-release audit harness (validation/audit)."""
import os
import subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LWP_R = os.environ.get(
    "LWP_R",
    "/home/user/mannks_dev/Example_Files/R/LWPTrends_v2502/LWPTrends_v2502.r")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


def run_r(script, inp, out):
    """Run an R harness script; raise with stderr on failure."""
    cmd = ["Rscript", os.path.join(HERE, script), LWP_R, inp, out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-4000:])
    return pd.read_csv(out)


def rel_diff(a, b):
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return 0.0
    if np.isnan(a) or np.isnan(b):
        return np.inf
    return abs(a - b) / max(abs(b), 1e-12)
