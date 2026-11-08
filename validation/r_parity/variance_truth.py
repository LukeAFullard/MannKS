"""Is Var(S) correct?  Compare formula variances with the *exact* permutation
variance of S under H0 (all orderings of the observed values equally likely).

S is computed with MannKS's own censored scoring (which matches R exactly), so
the permutation variance is the ground truth for whatever the variance formula
is meant to estimate.  For uncensored data we also compare with pymannkendall.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from MannKS._stats import _mk_score_and_var_censored
from common import OUT

def S_only(x, cen):
    # direct censored-pair scoring (Helsel): uncertain comparisons score 0
    n = len(x); s = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = x[i], x[j]; ca, cb = cen[i], cen[j]
            if ca and cb: continue
            if ca and not cb:   # a is "<a"
                s += 1 if b >= a else 0;  continue
            if cb and not ca:   # b is "<b"
                s += -1 if a >= b else 0; continue
            s += np.sign(b - a)
    return s

def perm_var(x, cen, n_perm, rng):
    vals = []
    idx = np.arange(len(x))
    for _ in range(n_perm):
        p = rng.permutation(idx)
        vals.append(S_only(x[p], cen[p]))
    return np.var(vals), np.mean(vals)

rng = np.random.default_rng(1)
rows = []
d = pd.read_csv(f"{OUT}/kernel_in.csv"); R = pd.read_csv(f"{OUT}/kernel_r.csv").set_index("case_id")
sel = [c for c in d.case_id.unique() if not (d[d.case_id == c].CenType == "gt").any()][:40]
for cid in sel:
    g = d[d.case_id == cid]
    if len(g) > 25: continue
    x = g.V1.to_numpy(); c = (g.CenType == "lt").to_numpy()
    # sanity: our scorer equals MannKS S
    s_mk = _mk_score_and_var_censored(x, np.arange(len(x)), c, np.where(c, "lt", "not"))[0]
    assert S_only(x, c) == s_mk, (cid, S_only(x, c), s_mk)
    pv, pm = perm_var(x, c, 4000, rng)
    v_mk = _mk_score_and_var_censored(x, np.arange(len(x)), c, np.where(c, "lt", "not"),
                                      mk_test_method="lwp", tie_break_method="lwp")[1]
    rows.append(dict(case_id=cid, n=len(x), n_cens=int(c.sum()), perm_var=pv,
                     R_var=R.loc[cid].VarS, MannKS_var=v_mk))
df = pd.DataFrame(rows)
df["R_err_%"] = 100 * (df.R_var / df.perm_var - 1)
df["MannKS_err_%"] = 100 * (df.MannKS_var / df.perm_var - 1)
df.to_csv(f"{OUT}/variance_truth.csv", index=False)
pd.set_option("display.width", 200)
print(df.round(2).to_string(index=False))
print("\nMean |error| vs exact permutation variance (Monte Carlo SE of perm_var ~2.2%):")
print("  uncensored  R: %.2f%%  MannKS: %.2f%%" % (df[df.n_cens == 0]["R_err_%"].abs().mean(), df[df.n_cens == 0]["MannKS_err_%"].abs().mean()))
print("  censored    R: %.2f%%  MannKS: %.2f%%" % (df[df.n_cens > 0]["R_err_%"].abs().mean(), df[df.n_cens > 0]["MannKS_err_%"].abs().mean()))

# pymannkendall for uncensored tied data
import pymannkendall as pmk
x = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7, 7, 7, 8], float)
v_mk = _mk_score_and_var_censored(x, np.arange(len(x)), np.zeros(len(x), bool), np.full(len(x), "not"))[1]
r = pmk.original_test(x)
n = len(x); _, t = np.unique(x, return_counts=True)
v_formula = (n*(n-1)*(2*n+5) - np.sum(t*(t-1)*(2*t+5))) / 18
print(f"\nTied uncensored example: textbook tie-corrected Var(S)={v_formula:.4f}, pymannkendall={r.var_s:.4f}, MannKS={v_mk:.4f}")
