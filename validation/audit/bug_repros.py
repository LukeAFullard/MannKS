"""Minimal reproductions for audit findings (each prints evidence)."""
import os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.environ.get("MANNKS_SRC", os.path.join(os.path.dirname(__file__), "../..")))
import MannKS as mk
from MannKS._stats import _mk_score_and_var_censored
warnings.filterwarnings("ignore")

def hdr(s): print("\n=== " + s + " ===")

hdr("F-full: large_dataset_mode='full' is not exact for n>447 (stochastic, unseeded)")
rng = np.random.default_rng(0)
n = 1000
t = np.arange(n, dtype=float); x = 0.01 * t + rng.normal(0, 1, n)
r1 = mk.trend_test(x, t, large_dataset_mode="full"); r2 = mk.trend_test(x, t, large_dataset_mode="full")
i, j = np.triu_indices(n, 1); exact = np.median((x[j] - x[i]) / (t[j] - t[i]))
print(f"n={n}: exact Sen={exact:.8f}  run1={r1.slope:.8f}  run2={r2.slope:.8f}  computation_mode={r1.computation_mode}")

hdr("F-order: trend_test censored Sen slope / ATS depend on input row order")
rng = np.random.default_rng(3)
n = 30; t = np.arange(n, dtype=float); v = 2 + 0.05 * t + rng.normal(0, 0.7, n)
vals = ["<2" if a < 2 else round(a, 2) for a in v]
X = mk.prepare_censored_data(vals)
perm = rng.permutation(n)
a = mk.trend_test(X, t); b = mk.trend_test(X.iloc[perm].reset_index(drop=True), t[perm])
print(f"sorted input  : slope={a.slope:.5f} CI=({a.lower_ci:.5f},{a.upper_ci:.5f}) S={a.s}")
print(f"shuffled input: slope={b.slope:.5f} CI=({b.lower_ci:.5f},{b.upper_ci:.5f}) S={b.s}")
a = mk.trend_test(X, t, sens_slope_method="ats"); b = mk.trend_test(X.iloc[perm].reset_index(drop=True), t[perm], sens_slope_method="ats")
print(f"ATS sorted={a.slope:.5f}  ATS shuffled={b.slope:.5f}")
a = mk.trend_test(X, t, sens_slope_method="lwp"); b = mk.trend_test(X.iloc[perm].reset_index(drop=True), t[perm], sens_slope_method="lwp")
print(f"lwp sorted={a.slope:.5f}  lwp shuffled={b.slope:.5f}")

hdr("F-tiedtime: agg_method='none' with tied timestamps -> S depends on row order")
t = np.array([1, 1, 2, 2, 3, 3, 4, 4], float); x = np.array([1, 2, 3, 4, 5, 6, 7, 8], float)
a = mk.trend_test(x, t); b = mk.trend_test(x[::-1][[6,7,4,5,2,3,0,1]][::-1], t)
x2 = np.array([2, 1, 4, 3, 6, 5, 8, 7], float)
c = mk.trend_test(x2, t)
print(f"x=[1..8] S={a.s}; same pairs with tied-time members swapped S={c.s}; textbook S (tied-time pairs score 0)=24")

hdr("F-propcens: prop_censored uses pre-aggregation n with post-aggregation numerator")
d = pd.to_datetime(["2000-01-05", "2000-01-20", "2000-02-05", "2000-02-20", "2000-03-05", "2000-03-20"] * 1)
X = mk.prepare_censored_data(["<1", "<1", "<1", "<1", 2, 3])
r = mk.trend_test(X, d, agg_method="lwp", agg_period="month")
print(f"raw prop censored = 4/6 = {4/6:.3f}; LWP-R uses raw data (prop.censored = sum(x$Censored)/nrow(x)); MannKS prop_censored = {r.prop_censored:.3f}")

hdr("F-note: spurious 'Sen slope based on tied non-censored values' for datetime t")
d = pd.date_range("2000-01-01", periods=30, freq="365D")
x = 10 + 0.2 * np.arange(30) + np.random.default_rng(1).normal(0, 0.3, 30)
r = mk.trend_test(x, d, slope_scaling="year")
print(f"slope={r.slope:.3f}/yr, notes={r.analysis_notes}")

hdr("F-regional: NaN correlation -> CorrectedVarTAU NaN -> CT forced to 1.0")
tr = pd.DataFrame({"site": ["A", "B", "C"], "s": [10, 12, -3], "C": [0.80, 0.85, 0.6]})
ts = pd.DataFrame({"site": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
                   "time": list(range(5)) + list(range(10, 15)) + list(range(20, 25)),  # no overlapping times
                   "value": np.r_[np.arange(5), np.arange(5), np.arange(5)].astype(float)})
rr = mk.regional_test(tr, ts)
print(f"TAU={rr.TAU:.3f} VarTAU={rr.VarTAU:.4f} CorrectedVarTAU={rr.CorrectedVarTAU} CT={rr.CT}")
print("  (R getTAU uses sum(..., na.rm=TRUE) -> CorrectedVarTAU == VarTAU -> CT = %.4f)" %
      __import__("scipy").stats.norm.cdf((rr.TAU - 0.5) / np.sqrt(rr.VarTAU)))

hdr("F-boot-seed: block bootstrap / ATS CIs ignore random_state")
rng = np.random.default_rng(5); n = 60; t = np.arange(n, dtype=float)
x = np.cumsum(rng.normal(0, 1, n)) * 0.3 + 0.02 * t
a = mk.trend_test(x, t, autocorr_method="block_bootstrap", n_bootstrap=200, random_state=1)
b = mk.trend_test(x, t, autocorr_method="block_bootstrap", n_bootstrap=200, random_state=1)
print(f"same random_state: p={a.p:.4f} vs {b.p:.4f}; CI=({a.lower_ci:.5f},{a.upper_ci:.5f}) vs ({b.lower_ci:.5f},{b.upper_ci:.5f})")

hdr("F-boot-p0: bootstrap p-value can be exactly 0 (no (b+1)/(B+1) correction)")
x = 0.5 * t + rng.normal(0, 1, n)
a = mk.trend_test(x, t, autocorr_method="block_bootstrap", n_bootstrap=200)
print(f"p={a.p}  C={a.C}")

hdr("F-seasonal-tiebreak: seasonal_trend_test has no tie_break_method parameter")
import inspect
print("tie_break_method in trend_test:", "tie_break_method" in inspect.signature(mk.trend_test).parameters,
      "| in seasonal_trend_test:", "tie_break_method" in inspect.signature(mk.seasonal_trend_test).parameters)

hdr("F-seasonal-dupnotes: seasonal notes duplicated")
d = pd.date_range("2000-01-01", periods=120, freq="MS") + pd.Timedelta(days=14)
x = 10 + 0.02 * np.arange(120) + np.random.default_rng(2).normal(0, 1, 120)
r = mk.seasonal_trend_test(x, d)
print(r.analysis_notes)

hdr("F-classif-boundary: C exactly on a category boundary")
from collections import namedtuple
T = namedtuple("T", "C trend")
for c in (0.67, 0.90, 0.95):
    print(f"C={c}: MannKS='{mk.classify_trend(T(c, 'decreasing'))}'  (R cut(right=TRUE) puts {c} in the lower class)")

hdr("F-yuewang: Yue-Wang n_eff estimated on non-detrended data")
rng = np.random.default_rng(11); n = 50; t = np.arange(n, dtype=float)
x = 0.05 * t + rng.normal(0, 0.5, n)   # iid noise + trend
a = mk.trend_test(x, t); b = mk.trend_test(x, t, autocorr_method="yue_wang")
print(f"iid noise + trend: plain p={a.p:.2e}; yue_wang p={b.p:.2e}, n_eff={b.n_effective:.1f} (true n_eff = {n})")
c = mk.trend_test(x, t, autocorr_method="auto")
print(f"autocorr_method='auto' notes: {c.analysis_notes}")

hdr("F-hicensor-seg: segmented hicensor numeric ignores max(lt) rule")
X = mk.prepare_censored_data(["<1", 2, 3, 4, 5, 6, 7, 8, 9, 10])
from MannKS.segmented_trend_test import _prepare_data as seg_prep
from MannKS._helpers import _prepare_data as main_prep
a, _ = seg_prep(X, np.arange(10.0), hicensor=5); b, _ = main_prep(X, np.arange(10.0), 5)
print("segmented values:", a.value.tolist()); print("trend_test values:", b.value.tolist())

hdr("F-rolling-index: numeric t + DataFrame x with non-default index")
X = mk.prepare_censored_data(list(np.arange(40.0)))
X.index = np.arange(100, 140)
try:
    out = mk.rolling_trend_test(X, np.arange(40.0), window=20, step=10)
    print("rows:", len(out))
except Exception as e:
    print("ERROR:", type(e).__name__, e)
