"""Render parity results to markdown (out/parity_tables.md)."""
import numpy as np, pandas as pd
from common import OUT
cur = pd.read_csv(f"{OUT}/parity_results.csv"); fix = pd.read_csv(f"{OUT}/parity_results_varfix.csv")
num = cur[cur.status != "info"].copy(); numf = fix[fix.status != "info"].copy()
lines = []
cases = list(dict.fromkeys(num.case))
metrics = ["S", "VarS", "tau", "Z", "p", "C", "Cd", "slope", "lower_ci", "upper_ci", "prop_censored"]
def fmtv(v):
    try:
        v = float(v)
    except Exception:
        return str(v)
    if np.isnan(v): return "NA"
    return f"{v:.6g}"
lines.append("| case | " + " | ".join(metrics) + " | fails now | fails after var-fix |")
lines.append("|" + "---|" * (len(metrics) + 3))
for c in cases:
    a = num[num.case == c].set_index("metric"); b = numf[numf.case == c].set_index("metric")
    cells = []
    for m in metrics:
        sa, sb = a.loc[m, "status"], b.loc[m, "status"]
        cells.append("✅" if sa == "pass" else ("❌→✅" if sb == "pass" else "❌"))
    lines.append(f"| {c} | " + " | ".join(cells) + f" | {(a.status=='FAIL').sum()} | {(b.status=='FAIL').sum()} |")
lines.append("")
lines.append("### Values (current code) for S, VarS, p, slope, CI")
lines.append("| case | metric | R | MannKS (current) | rel. diff | status | MannKS (var-fix) | status |")
lines.append("|---|---|---|---|---|---|---|---|")
for c in cases:
    for m in ["S", "VarS", "p", "C", "slope", "lower_ci", "upper_ci"]:
        a = num[(num.case == c) & (num.metric == m)].iloc[0]; b = numf[(numf.case == c) & (numf.metric == m)].iloc[0]
        rd = "inf" if np.isinf(a.rel_diff) else f"{a.rel_diff:.1e}"
        lines.append(f"| {c} | {m} | {fmtv(a.R)} | {fmtv(a.MannKS)} | {rd} | {a.status} | {fmtv(b.MannKS)} | {b.status} |")
open(f"{OUT}/parity_tables.md", "w").write("\n".join(lines))
print("\n".join(lines[:40]))
