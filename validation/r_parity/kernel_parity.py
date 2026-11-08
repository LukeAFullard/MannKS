"""Kernel-level parity: R GetKendal / GetInterObservationSlopes vs MannKS internals.

Random censored datasets (one obs per year, so no aggregation is involved) are
fed to both implementations. Isolates S, VarS and the censored Sen slope from
all wrapper logic.
"""
import sys
import numpy as np
import pandas as pd
from common import run_r, OUT, rel_diff
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "../.."))
from MannKS._stats import _mk_score_and_var_censored, _sens_estimator_censored

rng = np.random.default_rng(20261107)
rows = []
cases = {}
for cid in range(300):
    n = int(rng.integers(5, 40))
    kind = cid % 6
    vals = np.round(rng.normal(10, 3, n) + 0.1 * np.arange(n) * rng.choice([-1, 1]), 0 if kind % 2 else 1)
    cen = np.full(n, "not", dtype=object)
    if kind in (1, 2, 4, 5):  # left censoring, multiple DLs
        dls = rng.choice([7.0, 8.0, 9.0], n)
        m = vals < dls
        vals[m] = dls[m]; cen[m] = "lt"
    if kind in (3, 4, 5):  # right censoring
        al = rng.choice([13.0, 14.0], n)
        m = (vals > al) & (cen == "not")
        vals[m] = al[m]; cen[m] = "gt"
    years = 1990 + np.arange(n)
    dates = pd.to_datetime([f"{y}-07-01" for y in years])
    cases[cid] = (vals, cen, dates)
    for v, c, d, y in zip(vals, cen, dates, years):
        rows.append(dict(case_id=cid, V1=v, Censored=c != "not", CenType=c, Season="A",
                         Year=y, NewDate=d.strftime("%Y-%m-%d")))
inp = f"{OUT}/kernel_in.csv"
pd.DataFrame(rows).to_csv(inp, index=False)
R = run_r("r_kernel.R", inp, f"{OUT}/kernel_r.csv").set_index("case_id")

res = []
for cid, (vals, cen, dates) in cases.items():
    t_days = (dates - pd.Timestamp("1970-01-01")).days.to_numpy() / 365.25
    out = dict(case_id=cid, n=len(vals), n_cens=int((cen != "not").sum()),
               has_gt=bool((cen == "gt").any()))
    for tb in ("lwp", "robust"):
        s, v, d, tau = _mk_score_and_var_censored(vals, np.arange(len(vals)), cen != "not", cen,
                                                  mk_test_method="lwp", tie_break_method=tb)
        out[f"S_{tb}"] = s; out[f"VarS_{tb}"] = v
    sl = _sens_estimator_censored(vals, t_days, cen, method="lwp")
    out["sen_py"] = np.median(sl)
    r = R.loc[cid]
    out.update(S_R=r.S, VarS_R=r.VarS, sen_R=r.sen)
    res.append(out)
df = pd.DataFrame(res)
df["dS"] = (df.S_lwp - df.S_R).abs()
df["relVar_lwp"] = [rel_diff(a, b) for a, b in zip(df.VarS_lwp, df.VarS_R)]
df["relVar_robust"] = [rel_diff(a, b) for a, b in zip(df.VarS_robust, df.VarS_R)]
df["relSen"] = [rel_diff(a, b) for a, b in zip(df.sen_py, df.sen_R)]
df.to_csv(f"{OUT}/kernel_parity.csv", index=False)
tol = 1e-9
print("cases:", len(df))
print("S mismatches:", int((df.dS > 0).sum()))
for col in ["relVar_lwp", "relVar_robust", "relSen"]:
    bad = df[df[col] > tol]
    print(f"{col}: {len(bad)} cases > {tol}; max rel diff {df[col].max():.3e}")
print("VarS mismatch among uncensored cases:", int(((df.n_cens == 0) & (df.relVar_lwp > tol)).sum()))
print("VarS mismatch among censored cases  :", int(((df.n_cens > 0) & (df.relVar_lwp > tol)).sum()), "of", int((df.n_cens > 0).sum()))
print(df[df.relVar_lwp > tol].head(8)[["case_id", "n", "n_cens", "S_R", "VarS_R", "VarS_lwp", "VarS_robust"]].to_string())
