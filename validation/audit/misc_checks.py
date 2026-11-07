"""Smaller audit checks that were first run interactively (kept for reproducibility).

1. SeasonalityTest (R) vs check_seasonality on parity cases.
2. ATS S(beta)=0 plateau: MannKS root vs NADA midpoint.
3. NADA tie-break shift (delx = min_diff/1000) is safe for large n.
4. alpha=0.05 vs 0.10 against R's hard-coded 90% CI; validation-style LWP kwargs.
5. R vs Python time-increment mid-points used for UseMidObs / agg_method='lwp'.
Requires run_parity.py and censored_slope_study.py to have been run first (uses out/ files).
"""
import os, sys, subprocess, types, warnings
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("MANNKS_SRC", os.path.join(HERE, "../..")))
import MannKS as mk
from MannKS._ats import S_of_beta, make_intervals
from common import LWP_R, OUT
warnings.filterwarnings("ignore")

print("== 1. SeasonalityTest vs check_seasonality")
ids = "monthly_seasonal,seasonal_missing_month,seasonal_sparse_month,seasonal_lt,real_ex2,monthly_nonseasonal"
r = subprocess.run(["Rscript", os.path.join(HERE, "r_seasonality.R"), LWP_R, f"{OUT}/parity_data.csv", f"{OUT}/seas_r.csv", ids],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr
R = pd.read_csv(f"{OUT}/seas_r.csv").set_index("case_id"); d = pd.read_csv(f"{OUT}/parity_data.csv", dtype=str)
for cid in ids.split(","):
    g = d[d.case_id == cid]; x = mk.prepare_censored_data([np.nan if v == "NA" else v for v in g.Value])
    py = mk.check_seasonality(x, pd.to_datetime(g.myDate).to_numpy())
    print(f"  {cid:24s} R: KW={R.loc[cid].KW:.6g} p={R.loc[cid].p:.6g} note='{R.loc[cid].note}' | "
          f"MannKS: H={py.h_statistic:.6g} p={py.p_value:.6g} skipped={[int(s) for s in py.seasons_skipped]}")

print("\n== 2. ATS plateau (largest MannKS-vs-cenken differences)")
cd = pd.read_csv(f"{OUT}/cenken_in.csv"); a = pd.read_csv(f"{OUT}/ats_vs_cenken.csv").set_index("case_id")
for cid in a.sort_values("rel_diff").index[-4:]:
    z = cd[cd.case_id == cid]; lo, hi = make_intervals(z.y.to_numpy(), z.cen.to_numpy())
    p, q = a.loc[cid].py_slope, a.loc[cid].R_slope
    grid = np.linspace(min(p, q) - 0.01, max(p, q) + 0.01, 4001)
    S = np.array([S_of_beta(b, z.x.to_numpy(), lo, hi) for b in grid]); zero = grid[S == 0]
    print(f"  case {cid}: MannKS={p:.5f} NADA={q:.5f} plateau=[{zero.min():.5f},{zero.max():.5f}] mid={(zero.min()+zero.max())/2:.5f}")

print("\n== 3. delx = min_diff/1000 vs /1e9 for large censored n (uses MANNKS_SRC copy if given)")
src = open(os.path.join(os.path.dirname(mk.__file__), "_stats.py")).read()
def variant(div):
    m = types.ModuleType("v"); s = src.replace("(1000.0 if tie_break_method == 'lwp' else 2.0)", f"({div} if tie_break_method == 'lwp' else 2.0)")
    exec(compile(s, "v", "exec"), m.__dict__); return m
v1, v2 = variant(1000.0), variant(1e9); rng = np.random.default_rng(4)
for n in (200, 1500, 3000):
    x = np.round(rng.normal(10, 2, n), 1); c = x < 9; x[c] = 9.0; ct = np.where(c, "lt", "not").astype(object)
    A = v1._mk_score_and_var_censored(x, np.arange(n), c, ct, mk_test_method="lwp", tie_break_method="lwp")[1]
    B = v2._mk_score_and_var_censored(x, np.arange(n), c, ct, mk_test_method="lwp", tie_break_method="lwp")[1]
    print(f"  n={n}: rel diff {abs(A-B)/B:.1e}")

print("\n== 4. alpha check")
Rw = pd.read_csv(f"{OUT}/parity_r.csv").set_index("case_id")
for cid in ["simple_trend", "monthly_seasonal"]:
    g = d[d.case_id == cid]; x = mk.prepare_censored_data(g.Value.tolist()); t = pd.to_datetime(g.myDate).to_numpy()
    for al in (0.05, 0.10):
        kw = dict(alpha=al, mk_test_method="lwp", ci_method="lwp", sens_slope_method="lwp", slope_scaling="year")
        res = (mk.trend_test(x, t, agg_method="lwp", agg_period="year", tie_break_method="lwp", **kw) if cid == "simple_trend"
               else mk.seasonal_trend_test(x, t, agg_method="lwp", **kw))
        print(f"  {cid:17s} alpha={al:.2f}: MannKS CI=({res.lower_ci:.6f},{res.upper_ci:.6f})  R=({Rw.loc[cid].sen_Sen_Lci:.6f},{Rw.loc[cid].sen_Sen_Uci:.6f})")

print("\n== 5. time-increment mid-points (2001): R MidTimeIncrList vs pandas period midpoint")
for k, f in ((1, "Y"), (4, "Q"), (12, "M")):
    rmid = [pd.Timestamp("2001-01-01") + pd.Timedelta(days=v) for v in np.arange(365 / k / 2, 365, 365 / k) - 1]
    pmid = [p.start_time + (p.end_time - p.start_time) / 2 for p in pd.period_range("2001-01", periods=k, freq=f)]
    print(f"  {k:2d} incr/yr: max |R - Python| = {max(abs((a - b).total_seconds()) for a, b in zip(rmid, pmid)) / 86400:.2f} days")
