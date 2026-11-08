"""Wrapper-level parity: LWP-TRENDS v2502 (R) vs MannKS in its 'LWP mode'.

R side  : RemoveAlphaDetect -> GetMoreDateInfo -> MannKendall/SenSlope or
          SeasonalKendall/SeasonalSenSlope (see r_wrapper.R).
Python  : trend_test / seasonal_trend_test with every LWP switch MannKS offers:
          mk_test_method='lwp', ci_method='lwp', sens_slope_method='lwp',
          tie_break_method='lwp' (trend_test only), tau_method='a',
          agg_method='lwp' (mid-obs) or 'lwp_median'/'median', alpha=0.10
          (R hard-codes a 90% CI: Z <- 1-0.05), slope_scaling='year'.

Run:  python run_parity.py            (writes out/parity_*.csv and prints a table)
Env:  LWP_R=/path/to/LWPTrends_v2502.r
      REUSE_R=1          reuse out/parity_r.csv from a previous run
      LARGE_MODE=full    pass large_dataset_mode to MannKS (default 'auto')

Expected differences (see README.md): cases where LWP-TRENDS declines to
compute (all_identical, n2-n4, exact_linear, seasonal_missing_month,
seasonal_sparse_month, seasonal_lt), same-day duplicate observations in
multi_per_month_mid, and the sampled slope of long_6000 in fast mode.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("MANNKS_SRC", os.path.join(HERE, "../..")))
import MannKS as mk  # noqa: E402
from common import run_r, OUT  # noqa: E402

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)
DATA, CFG = [], []


def add(case_id, dates, values, seasonal=False, timeincr="Year", season=None,
        use_mid=True, shift=False):
    for dt, v in zip(dates, values):
        DATA.append(dict(case_id=case_id, myDate=pd.Timestamp(dt).strftime("%Y-%m-%d"),
                         Value="NA" if (isinstance(v, float) and np.isnan(v)) else str(v)))
    CFG.append(dict(case_id=case_id, seasonal=seasonal, timeincr=timeincr,
                    season=season or timeincr, use_mid=use_mid, shift=shift))


def annual(n, start=1990, day="07-01"):
    return pd.to_datetime([f"{start + i}-{day}" for i in range(n)])


def monthly(nyears, start=2000, day=15):
    return pd.to_datetime([f"{start + i // 12}-{i % 12 + 1:02d}-{day:02d}" for i in range(12 * nyears)])


def fmt(a, nd=3):
    return [round(float(v), nd) for v in a]


# ---------------- synthetic cases ----------------
n = 20
add("simple_trend", annual(n), fmt(10 + 0.5 * np.arange(n) + rng.normal(0, 1, n)))
add("no_trend", annual(n), fmt(10 + rng.normal(0, 1, n)))
add("ties_annual", annual(25), fmt(np.round(5 + 0.08 * np.arange(25) + rng.normal(0, 1, 25)), 0))
add("all_identical", annual(15), [5.0] * 15)
for k in (2, 3, 4, 5):
    add(f"n{k}", annual(k), fmt(10 + np.arange(k) + rng.normal(0, 0.3, k)))
# exact linear series (equal increments)
add("exact_linear", annual(12), fmt(1 + 0.5 * np.arange(12)))
# left censored, single DL
v = 1 + 0.05 * np.arange(25) + rng.normal(0, 0.6, 25)
add("lt_single_DL", annual(25), [f"<1" if x < 1 else round(x, 3) for x in v])
# left censored, multiple (decreasing) DLs
v = 2 + rng.normal(0, 1, 30) - 0.03 * np.arange(30)
dl = np.where(np.arange(30) < 10, 2.0, np.where(np.arange(30) < 20, 1.0, 0.5))
add("lt_multi_DL", annual(30), [f"<{d}" if x < d else round(x, 3) for x, d in zip(v, dl)])
# right censored, multiple ALs
v = 10 + 0.1 * np.arange(30) + rng.normal(0, 2, 30)
al = np.where(np.arange(30) < 15, 12.0, 13.0)
add("gt_multi_AL", annual(30), [f">{a}" if x > a else round(x, 3) for x, a in zip(v, al)])
# mixed
v = 5 + 0.05 * np.arange(30) + rng.normal(0, 2, 30)
add("mixed_cens", annual(30), [("<3" if x < 3 else (">7" if x > 7 else round(x, 2))) for x in v])
# monthly, non-seasonal, monthly time increment
v = 10 + 0.02 * np.arange(120) + rng.normal(0, 1, 120)
add("monthly_nonseasonal", monthly(10), fmt(v), timeincr="Month")
# monthly seasonal
i = np.arange(120)
v = 10 + 3 * np.sin(2 * np.pi * i / 12) + 0.02 * i + rng.normal(0, 0.8, 120)
add("monthly_seasonal", monthly(10), fmt(v), seasonal=True, timeincr="Month")
# quarterly seasonal
qd = pd.to_datetime([f"{2000 + k // 4}-{3 * (k % 4) + 2:02d}-15" for k in range(48)])
k = np.arange(48)
add("quarterly_seasonal", qd, fmt(5 + 2 * np.cos(2 * np.pi * k / 4) + 0.03 * k + rng.normal(0, 0.5, 48)),
    seasonal=True, timeincr="Qtr")
# seasonal with an entirely missing month (Feb)
dm, vm = monthly(10), 10 + 3 * np.sin(2 * np.pi * i / 12) + 0.02 * i + rng.normal(0, 0.8, 120)
keep = dm.month != 2
add("seasonal_missing_month", dm[keep], fmt(vm[keep]), seasonal=True, timeincr="Month")
# seasonal with a season having only 2 obs
keep = ~((dm.month == 3) & (dm.year > 2001))
add("seasonal_sparse_month", dm[keep], fmt(vm[keep]), seasonal=True, timeincr="Month")
# unequal spacing: one obs per year on a random day
dd = pd.to_datetime([f"{1990 + j}-01-01" for j in range(25)]) + pd.to_timedelta(rng.integers(0, 364, 25), "D")
add("unequal_annual", dd, fmt(3 + 0.1 * np.arange(25) + rng.normal(0, 0.7, 25)))
# monthly with random gaps (non-seasonal)
keep = rng.random(120) > 0.3
add("monthly_gaps", monthly(10)[keep], fmt(v[keep]), timeincr="Month")
# multiple obs per month (2-4 per month, random days)
rows_d, rows_v = [], []
for m in range(60):
    for _ in range(int(rng.integers(2, 5))):
        rows_d.append(pd.Timestamp(f"{2010 + m // 12}-{m % 12 + 1:02d}-01") + pd.Timedelta(days=int(rng.integers(0, 28))))
        rows_v.append(round(20 - 0.03 * m + rng.normal(0, 1.5), 2))
add("multi_per_month_mid", rows_d, rows_v, timeincr="Month", use_mid=True)
add("multi_per_month_median", rows_d, rows_v, timeincr="Month", use_mid=False)
# multiple obs per year, annual increment
rows_d2, rows_v2 = [], []
for y in range(20):
    for _ in range(int(rng.integers(1, 4))):
        rows_d2.append(pd.Timestamp(f"{1995 + y}-01-01") + pd.Timedelta(days=int(rng.integers(0, 365))))
        rows_v2.append(round(5 + 0.1 * y + rng.normal(0, 0.5), 2))
add("multi_per_year_mid", rows_d2, rows_v2, timeincr="Year", use_mid=True)
add("multi_per_year_median", rows_d2, rows_v2, timeincr="Year", use_mid=False)
# negative values and zeros
add("neg_and_zero", annual(20), fmt(np.round(-2 + 0.2 * np.arange(20) + rng.normal(0, 1, 20), 0), 0))
# NaN / missing values
vv = list(fmt(10 + 0.3 * np.arange(22) + rng.normal(0, 1, 22)))
for j in (3, 9, 15):
    vv[j] = np.nan
add("with_nan", annual(22), vv)
# seasonal left-censored
vs = 1.5 + np.sin(2 * np.pi * i / 12) - 0.005 * i + rng.normal(0, 0.4, 120)
add("seasonal_lt", monthly(10), [f"<1" if x < 1 else round(x, 3) for x in vs], seasonal=True, timeincr="Month")
vs2 = 8 + 2 * np.sin(2 * np.pi * i / 12) + 0.01 * i + rng.normal(0, 1, 120)
add("seasonal_gt", monthly(10), [f">10" if x > 10 else round(x, 3) for x in vs2], seasonal=True, timeincr="Month")
# long series (> 5000 increments; triggers MannKS fast mode in 'auto')
nl = 12 * 500
vl = 10 + 0.0005 * np.arange(nl) + rng.normal(0, 1, nl)
add("long_6000", monthly(500, start=1500), fmt(vl), timeincr="Month")

# ---------------- real LWP example data (exported from the R package data) ----------------
from common import export_lwp_example_data  # noqa: E402
_examples = export_lwp_example_data()
for name, ti, seas in [("ex1", "Month", False), ("ex2", "Month", True), ("ex3", "Month", False),
                       ("ex4", "Year", False)]:
    if name in _examples:
        df = pd.read_csv(_examples[name], dtype=str)
        add(f"real_{name}", pd.to_datetime(df.myDate), df.Value.tolist(), seasonal=seas, timeincr=ti)

data = pd.DataFrame(DATA)
cfg = pd.DataFrame(CFG)
data.to_csv(f"{OUT}/parity_data.csv", index=False)
cfg.to_csv(f"{OUT}/parity_cfg.csv", index=False)

import subprocess  # noqa: E402
TAG = os.environ.get("PARITY_TAG", "")
if os.environ.get("REUSE_R") and os.path.exists(f"{OUT}/parity_r.csv"):
    r = subprocess.CompletedProcess([], 0)
else:
  print("Running R ...", flush=True)
  r = subprocess.run(["Rscript", os.path.join(HERE, "r_wrapper.R"), os.environ.get("LWP_R", __import__("common").LWP_R),
                    f"{OUT}/parity_data.csv", f"{OUT}/parity_r.csv", f"{OUT}/parity_cfg.csv"],
                   capture_output=True, text=True)
if r.returncode:
    raise SystemExit(r.stderr[-3000:])
R = pd.read_csv(f"{OUT}/parity_r.csv").set_index("case_id")

PERIOD = {"Year": "year", "Month": "month", "Qtr": "quarter"}
SEASON = {"Month": "month", "Qtr": "quarter"}


def run_py(cc, mode="auto"):
    g = data[data.case_id == cc.case_id]
    vals = [np.nan if v == "NA" else v for v in g.Value]
    x = mk.prepare_censored_data(vals)
    t = pd.to_datetime(g.myDate).to_numpy()
    common = dict(alpha=0.10, mk_test_method="lwp", ci_method="lwp", sens_slope_method="lwp",
                  tau_method="a", slope_scaling="year", large_dataset_mode=mode)
    if not cc.seasonal:
        res = mk.trend_test(x, t, tie_break_method="lwp",
                            agg_method="lwp" if cc.use_mid else "lwp_median",
                            agg_period=PERIOD[cc.timeincr], random_state=1, **common)
    else:
        res = mk.seasonal_trend_test(x, t, season_type=SEASON[cc.timeincr], tie_break_method="lwp",
                                     agg_method="lwp" if cc.use_mid else "median",
                                     random_state=1, **common)
    return res


METRICS = [("S", "mk_S", "s"), ("VarS", "mk_VarS", "var_s"), ("tau", "mk_tau", "Tau"),
           ("Z", "mk_Z", "z"), ("p", "mk_p", "p"), ("C", "mk_C", "C"), ("Cd", "mk_Cd", "Cd"),
           ("slope", "sen_AnnualSenSlope", "slope"), ("lower_ci", "sen_Sen_Lci", "lower_ci"),
           ("upper_ci", "sen_Sen_Uci", "upper_ci"), ("prop_censored", "mk_prop.censored", "prop_censored"),
           ("sen_probability", "sen_Sen_Probability", "sen_probability")]
TOL = 1e-6  # relative; both sides are deterministic double-precision computations

rows = []
for _, cc in cfg.iterrows():
    rr = R.loc[cc.case_id]
    try:
        py = run_py(cc, mode=os.environ.get("LARGE_MODE", "auto"))
        err = ""
    except Exception as e:  # pragma: no cover - reported in table
        py, err = None, f"{type(e).__name__}: {e}"
    for name, rcol, pcol in METRICS:
        rv = pd.to_numeric(rr[rcol], errors="coerce")
        pv = np.nan if py is None else float(getattr(py, pcol))
        if np.isnan(rv) and np.isnan(pv):
            diff, ok = 0.0, True
        elif np.isnan(rv) or np.isnan(pv):
            diff, ok = np.inf, False
        else:
            diff = abs(pv - rv) / max(abs(rv), 1e-12) if abs(rv) > 1e-12 else abs(pv - rv)
            ok = diff <= TOL
        rows.append(dict(case=cc.case_id, metric=name, R=rv, MannKS=pv, rel_diff=diff,
                         status="pass" if ok else "FAIL", py_error=err))
    rows.append(dict(case=cc.case_id, metric="R_MK_note", R=rr["mk_AnalysisNote"],
                     MannKS="; ".join(py.analysis_notes) if py is not None else err,
                     rel_diff=np.nan, status="info", py_error=err))
    rows.append(dict(case=cc.case_id, metric="R_Sen_note", R=rr["sen_AnalysisNote"], MannKS="",
                     rel_diff=np.nan, status="info", py_error=err))
    rows.append(dict(case=cc.case_id, metric="classification",
                     R=f"{rr['confcat_direction']} {rr['dir']} / {rr['confcat_decrease']}",
                     MannKS=py.classification if py is not None else err,
                     rel_diff=np.nan, status="info", py_error=err))

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/parity_results{TAG}.csv", index=False)
num = res[res.status != "info"]
summary = num.pivot_table(index="case", columns="metric", values="status",
                          aggfunc=lambda s: "ok" if all(s == "pass") else "FAIL")
cols = [m[0] for m in METRICS]
summary = summary[cols].reindex(cfg.case_id)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(summary.to_string())
print(f"\nTolerance: relative {TOL}. Failures: {int((num.status == 'FAIL').sum())}/{len(num)}")


# ---------------- seasonality test (SeasonalityTest vs check_seasonality) ----------------
ids = [c for c in ["monthly_seasonal", "monthly_nonseasonal", "seasonal_missing_month", "real_ex2"]
       if c in set(cfg.case_id)]
r = subprocess.run(["Rscript", os.path.join(HERE, "r_seasonality.R"), os.environ.get("LWP_R", __import__("common").LWP_R),
                    f"{OUT}/parity_data.csv", f"{OUT}/seasonality_r.csv", ",".join(ids)],
                   capture_output=True, text=True)
if r.returncode == 0:
    RS = pd.read_csv(f"{OUT}/seasonality_r.csv").set_index("case_id")
    print("\nSeasonality (Kruskal-Wallis): R SeasonalityTest vs check_seasonality")
    for cid in ids:
        g = data[data.case_id == cid]
        xs = mk.prepare_censored_data([np.nan if v == "NA" else v for v in g.Value])
        py = mk.check_seasonality(xs, pd.to_datetime(g.myDate).to_numpy())
        print(f"  {cid:24s} R KW={RS.loc[cid].KW:.6g} p={RS.loc[cid].p:.6g} ({RS.loc[cid].note}) | "
              f"MannKS H={py.h_statistic:.6g} p={py.p_value:.6g}")
