"""(a) ATS parity: MannKS ats_slope vs NADA::cenken (left-censored only, as cenken supports).
(b) Bias of the censored slope estimators across censoring levels vs the known true slope."""
import os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.environ.get("MANNKS_SRC", os.path.join(os.path.dirname(__file__), "../..")))
import MannKS as mk
from MannKS._ats import ats_slope
from common import run_r, OUT
warnings.filterwarnings("ignore")

# ---- (a) ATS vs NADA::cenken ----
rng = np.random.default_rng(99)
rows, cases = [], {}
for cid in range(60):
    n = int(rng.integers(12, 40)); x = np.sort(rng.uniform(0, 20, n))
    y = np.exp(0.5 + 0.03 * x * rng.choice([-1, 1]) + rng.normal(0, 0.5, n))
    dl = rng.choice([1.0, 1.5, 2.0], n) if cid % 2 else np.full(n, 1.5)
    cen = y < dl; y = np.where(cen, dl, np.round(y, 3))
    cases[cid] = (x, y, cen)
    rows += [dict(case_id=cid, x=a, y=b, cen=c) for a, b, c in zip(x, y, cen)]
pd.DataFrame(rows).to_csv(f"{OUT}/cenken_in.csv", index=False)
R = run_r("r_cenken.R", f"{OUT}/cenken_in.csv", f"{OUT}/cenken_r.csv").set_index("case_id")
res = []
for cid, (x, y, cen) in cases.items():
    a = ats_slope(x, y, cen, cen_type=np.where(cen, "lt", "not"), lod=y, bootstrap_ci=False)
    res.append(dict(case_id=cid, n=len(x), pcen=cen.mean(), py_slope=a["beta"], R_slope=R.loc[cid].slope,
                    py_int=a["intercept"], R_int=R.loc[cid].intercept))
df = pd.DataFrame(res)
df["abs_diff"] = (df.py_slope - df.R_slope).abs()
df["rel_diff"] = df.abs_diff / df.R_slope.abs().clip(lower=1e-12)
df.to_csv(f"{OUT}/ats_vs_cenken.csv", index=False)
print("ATS slope vs NADA::cenken over", len(df), "datasets")
print("  median rel diff %.2e, 95th pct %.2e, max %.2e" % (df.rel_diff.median(), df.rel_diff.quantile(.95), df.rel_diff.max()))
print("  intercept median |diff| %.3e" % (df.py_int - df.R_int).abs().median())
print(df.sort_values("rel_diff").tail(4).round(5).to_string(index=False))

# ---- (b) bias study ----
print("\nCensored slope bias (true slope 0.05, n=40, normal noise sd=1, single DL), 400 reps")
out = []
for dl in (1.0, 2.0, 3.0):
    est = {m: [] for m in ("unbiased", "lwp", "ats", "oracle_uncensored")}
    pc = []
    for r in range(400):
        g = np.random.default_rng(70_000 + r)
        n = 40; t = np.arange(n, dtype=float); v = 2 + 0.05 * t + g.normal(0, 1, n)
        pc.append((v < dl).mean())
        X = mk.prepare_censored_data(["<%g" % dl if a < dl else a for a in v])
        for m in ("unbiased", "lwp"):
            est[m].append(mk.trend_test(X, t, sens_slope_method=m).slope)
        cen = X.censored.to_numpy()
        est["ats"].append(ats_slope(t, X.value.to_numpy(), cen, cen_type=X.cen_type.to_numpy(),
                                    lod=X.value.to_numpy(), bootstrap_ci=False)["beta"])
        est["oracle_uncensored"].append(mk.trend_test(v, t).slope)
    for m, e in est.items():
        e = np.array(e, float)
        out.append(dict(DL=dl, pct_censored=round(100 * np.mean(pc), 1), method=m,
                        rel_bias_pct=round(100 * (np.nanmean(e) - 0.05) / 0.05, 2),
                        rel_rmse_pct=round(100 * np.sqrt(np.nanmean((e - 0.05) ** 2)) / 0.05, 2)))
b = pd.DataFrame(out); b.to_csv(f"{OUT}/censored_slope_bias.csv", index=False)
print(b.to_string(index=False))
