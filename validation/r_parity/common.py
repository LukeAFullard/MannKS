"""Shared helpers for the LWP-TRENDS / NADA parity scripts (validation/r_parity)."""
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


def export_lwp_example_data():
    """
    Write the LWP-TRENDS example data sets (WQData_Ex1..4, distributed with the
    R code as LWPTrends_ExampleData_v2502.rdata next to LWP_R) to out/ as CSV
    with columns myDate, Value ('<'/'>' prefixes for censored values).
    Returns a dict name -> csv path, or {} if the .rdata file is not found.
    """
    rdata = os.path.join(os.path.dirname(LWP_R), "LWPTrends_ExampleData_v2502.rdata")
    if not os.path.exists(rdata):
        return {}
    script = f"""
    load("{rdata}")
    f <- function(df, v) data.frame(myDate = format(if (!is.null(df$myDate)) df$myDate else df$sdate), Value = v)
    v2 <- as.character(WQData_Ex2$RawValue)
    v2[WQData_Ex2$CenType == "lt"] <- paste0("<", v2[WQData_Ex2$CenType == "lt"])
    v2[WQData_Ex2$CenType == "gt"] <- paste0(">", v2[WQData_Ex2$CenType == "gt"])
    write.csv(f(WQData_Ex1, WQData_Ex1$Value), "{OUT}/data_ex1.csv", row.names = FALSE)
    write.csv(f(WQData_Ex2, v2), "{OUT}/data_ex2.csv", row.names = FALSE)
    write.csv(f(WQData_Ex3, WQData_Ex3$Value), "{OUT}/data_ex3.csv", row.names = FALSE)
    write.csv(f(WQData_Ex4, WQData_Ex4$Value), "{OUT}/data_ex4.csv", row.names = FALSE)
    """
    r = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return {f"ex{k}": os.path.join(OUT, f"data_ex{k}.csv") for k in (1, 2, 3, 4)}
