"""Monte Carlo calibration checks for MannKS.

Usage:  python monte_carlo.py [quick]
Set MANNKS_SRC to point at an alternative (e.g. patched) copy of the package.
Writes out/monte_carlo.csv and prints a table.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.environ.get("MANNKS_SRC", os.path.join(os.path.dirname(__file__), "../..")))
import MannKS as mk  # noqa: E402
from common import OUT  # noqa: E402

warnings.filterwarnings("ignore")
QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"
R = 300 if QUICK else 2000
ALPHA = 0.05
rows = []


def record(name, rate=None, ks_p=None, extra=None, reps=None):
    row = dict(check=name, reps=reps, value=rate, ks_uniform_p=ks_p, note=extra)
    rows.append(row)
    print(f"{name:70s} reps={reps:<5} value={rate if rate is None else round(rate, 4)} "
          f"KS_p={'' if ks_p is None else round(ks_p, 4)} {extra or ''}", flush=True)


def binom_ci(k, n):
    lo, hi = stats.binomtest(k, n).proportion_ci(0.95)
    return f"95% MC CI [{lo:.3f},{hi:.3f}]"


def h0_rejection(name, gen, reps=R, **kw):
    ps = []
    for r in range(reps):
        x, t = gen(np.random.default_rng(r))
        ps.append(mk.trend_test(x, t, **kw).p)
    ps = np.array(ps)
    k = int((ps < ALPHA).sum())
    record(name, k / reps, stats.kstest(ps, "uniform").pvalue, binom_ci(k, reps), reps)
    return ps


t0 = time.time()
# ---------------- 1. H0, continuous iid ----------------
for n in (10, 30, 100):
    h0_rejection(f"H0 iid normal n={n}: P(p<0.05)",
                 lambda g, n=n: (g.normal(size=n), np.arange(n, dtype=float)))

# ---------------- 2. H0 with heavy ties ----------------
for n in (30,):
    ps = h0_rejection(f"H0 iid, heavy ties (values rounded to 5 levels) n={n}: P(p<0.05)",
                      lambda g, n=n: (np.round(g.normal(size=n)).clip(-2, 2), np.arange(n, dtype=float)))
    # exact permutation reference for one dataset family: average size of the test
    # (a well-calibrated test is close to 0.05 with discreteness; Var(S) too large -> too small)

# ---------------- 3. H0 with left censoring (2 DLs) ----------------
def gen_cens(g, n=40):
    v = np.exp(g.normal(0, 1, n))
    dl = np.where(np.arange(n) < n // 2, 0.8, 0.5)
    vals = [f"<{d}" if a < d else a for a, d in zip(v, dl)]
    return mk.prepare_censored_data(vals), np.arange(n, dtype=float)


h0_rejection("H0 lognormal, ~30-40% left-censored, 2 DLs, n=40: P(p<0.05)", gen_cens, reps=R // 2)
h0_rejection("  same, mk_test_method='lwp'", gen_cens, reps=R // 2, mk_test_method="lwp")

# ---------------- 4. Seasonal H0 ----------------
def gen_seas(g, ny=10):
    d = pd.date_range("2000-01-15", periods=12 * ny, freq="MS") + pd.Timedelta(days=14)
    i = np.arange(12 * ny)
    return 3 * np.sin(2 * np.pi * i / 12) + g.normal(0, 1, 12 * ny), d


ps = []
for r in range(R // 2):
    x, d = gen_seas(np.random.default_rng(10_000 + r))
    ps.append(mk.seasonal_trend_test(x, d).p)
ps = np.array(ps); k = int((ps < ALPHA).sum())
record("Seasonal H0 (monthly, 10y, seasonal signal): P(p<0.05)", k / len(ps),
       stats.kstest(ps, "uniform").pvalue, binom_ci(k, len(ps)), len(ps))

# ---------------- 5. Sen slope bias & CI coverage ----------------
beta = 0.1
for n, ci_method in [(30, "direct"), (30, "lwp"), (100, "direct")]:
    est, cover, width = [], 0, []
    for r in range(R):
        g = np.random.default_rng(20_000 + r)
        t = np.sort(g.uniform(0, 50, n))
        x = beta * t + g.normal(0, 1, n)
        res = mk.trend_test(x, t, ci_method=ci_method)
        est.append(res.slope); cover += res.lower_ci <= beta <= res.upper_ci
        width.append(res.upper_ci - res.lower_ci)
    est = np.array(est)
    record(f"Sen slope, n={n}, unequal spacing, true={beta}: mean bias (rel.)", (est.mean() - beta) / beta,
           None, f"sd={est.std():.4f}", R)
    record(f"  95% CI coverage (ci_method={ci_method})", cover / R, None, binom_ci(int(cover), R), R)

# heavy ties in x -> CI coverage depends on Var(S) tie correction
cover = 0
for r in range(R):
    g = np.random.default_rng(30_000 + r)
    t = np.arange(40, dtype=float)
    x = np.round(0.05 * t + g.normal(0, 1, 40))
    res = mk.trend_test(x, t)
    cover += res.lower_ci <= 0.05 <= res.upper_ci
record("CI coverage with heavy ties (x rounded to integers), n=40, true=0.05", cover / R, None,
       binom_ci(int(cover), R), R)

# ---------------- 6. Censored slope estimators: bias ----------------
for method in ("ats", "drop_ambiguous", "lwp"):
    reps = R // 2
    est = []
    for r in range(reps):
        g = np.random.default_rng(40_000 + r)
        n = 40; t = np.arange(n, dtype=float)
        v = 2 + 0.05 * t + g.normal(0, 1, n)
        vals = ["<2" if a < 2 else a for a in v]
        est.append(mk.trend_test(mk.prepare_censored_data(vals), t, sens_slope_method=method).slope)
    est = np.array(est, float)
    record(f"Censored (~20% <2) slope, true=0.05, method={method}: mean rel. bias",
           (np.nanmean(est) - 0.05) / 0.05, None, f"sd={np.nanstd(est):.4f}", reps)

# ---------------- 7. Fast mode (stochastic Sen) vs exact ----------------
n = 6000
g = np.random.default_rng(7)
t = np.arange(n, dtype=float); x = 0.001 * t + g.normal(0, 1, n)
i, j = np.triu_indices(n, 1)
exact = float(np.median((x[j] - x[i]) / (t[j] - t[i])))
del i, j
seeds = 20 if QUICK else 50
fast = np.array([mk.trend_test(x, t, random_state=s).slope for s in range(seeds)])
record("Fast mode n=6000 (100k pairs): rel. bias of stochastic Sen vs exact", (fast.mean() - exact) / exact,
       None, f"rel sd={fast.std() / abs(exact):.4f}; exact={exact:.6g}", seeds)

# ---------------- 8. Autocorrelation methods ----------------
def ar1(g, n, phi):
    e = g.normal(size=n); y = np.empty(n); y[0] = e[0] / np.sqrt(1 - phi ** 2)
    for k in range(1, n):
        y[k] = phi * y[k - 1] + e[k]
    return y


nb = 100 if QUICK else 200
reps_b = 100 if QUICK else 300
for phi in (0.0, 0.6):
    for meth in ("none", "yue_wang", "block_bootstrap"):
        reps = reps_b if meth == "block_bootstrap" else R // 2
        rej = 0
        for r in range(reps):
            g = np.random.default_rng(50_000 + r)
            n = 60; t = np.arange(n, dtype=float)
            x = ar1(g, n, phi)
            res = mk.trend_test(x, t, autocorr_method=meth, n_bootstrap=nb, random_state=r)
            rej += res.p < ALPHA
        record(f"H0 AR(1) phi={phi}, n=60, autocorr_method={meth}: P(p<0.05)", rej / reps, None,
               binom_ci(int(rej), reps), reps)

# power with a trend and iid noise (shows yue_wang power loss)
for meth in ("none", "yue_wang"):
    rej = 0
    for r in range(R // 2):
        g = np.random.default_rng(60_000 + r)
        n = 60; t = np.arange(n, dtype=float)
        x = 0.02 * t + g.normal(size=n)
        rej += mk.trend_test(x, t, autocorr_method=meth).p < ALPHA
    record(f"Power: iid noise + slope 0.02/step, n=60, autocorr_method={meth}", rej / (R // 2), None,
           binom_ci(int(rej), R // 2), R // 2)

pd.DataFrame(rows).to_csv(f"{OUT}/monte_carlo{os.environ.get('MC_TAG', '')}.csv", index=False)
print(f"done in {time.time() - t0:.0f}s")
