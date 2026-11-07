"""regional_test vs R getTAU, feeding R's own per-site S and C into regional_test."""
import os, sys, subprocess, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.environ.get("MANNKS_SRC", os.path.join(os.path.dirname(__file__), "../..")))
import MannKS as mk
from common import OUT, LWP_R, HERE
warnings.filterwarnings("ignore")
rng = np.random.default_rng(8)
months = pd.date_range("2005-01-01", periods=120, freq="MS")
common = rng.normal(0, 1, 120)
for scenario in ("same_dates", "different_days"):
    rows = []
    for k, site in enumerate(["S1", "S2", "S3", "S4", "S5", "S6"]):
        day = 14 if scenario == "same_dates" else 3 + 4 * k
        v = 5 + 0.01 * (k - 1) * np.arange(120) + 0.8 * common + rng.normal(0, 1, 120)
        for m, val in zip(months, v):
            rows.append(dict(siteID=site, myDate=(m + pd.Timedelta(days=day)).strftime("%Y-%m-%d"), Value=round(val, 3)))
    obs = pd.DataFrame(rows); f = f"{OUT}/regional_{scenario}"
    obs.to_csv(f + "_obs.csv", index=False)
    r = subprocess.run(["Rscript", os.path.join(HERE, "r_regional.R"), LWP_R, f + "_obs.csv", f], capture_output=True, text=True)
    if r.returncode: raise SystemExit(r.stderr)
    sites = pd.read_csv(f + "_sites.csv"); tau = pd.read_csv(f + "_tau.csv").iloc[0]
    tr = sites.rename(columns={"siteID": "site", "S": "s"})[["site", "s", "C"]]
    ts = obs.rename(columns={"siteID": "site", "myDate": "time", "Value": "value"})
    ts["time"] = pd.to_datetime(ts.time)
    py = mk.regional_test(tr, ts)
    print(f"[{scenario}]")
    for a, b in [("M", "M"), ("TAU", "TAU"), ("VarTAU", "VarTAU"), ("CorrectedVarTAU", "CorrectedVarTAU"), ("CT", "CT"), ("DT", "DT")]:
        print(f"   {a:16s} R={tau[b]!s:22s} MannKS={getattr(py, a)!s}")
