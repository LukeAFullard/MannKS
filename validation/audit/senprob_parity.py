"""sen_probability vs R Sen_Probability (approx(..., ties=median)) on the parity cases."""
import os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.environ.get("MANNKS_SRC", os.path.join(os.path.dirname(__file__), "../..")))
import MannKS as mk
from common import OUT
warnings.filterwarnings("ignore")
data = pd.read_csv(f"{OUT}/parity_data.csv", dtype=str); cfg = pd.read_csv(f"{OUT}/parity_cfg.csv")
R = pd.read_csv(f"{OUT}/parity_r.csv").set_index("case_id")
P = {"Year": "year", "Month": "month", "Qtr": "quarter"}; SE = {"Month": "month", "Qtr": "quarter"}
for _, cc in cfg.iterrows():
    if cc.case_id.startswith("long") or cc.case_id.startswith("real_ex1"): continue
    g = data[data.case_id == cc.case_id]
    x = mk.prepare_censored_data([np.nan if v == "NA" else v for v in g.Value]); t = pd.to_datetime(g.myDate).to_numpy()
    kw = dict(alpha=0.1, mk_test_method="lwp", ci_method="lwp", sens_slope_method="lwp", tau_method="a")
    r = (mk.trend_test(x, t, tie_break_method="lwp", agg_method="lwp" if cc.use_mid else "lwp_median", agg_period=P[cc.timeincr], **kw)
         if not cc.seasonal else mk.seasonal_trend_test(x, t, season_type=SE[cc.timeincr], agg_method="lwp" if cc.use_mid else "median", **kw))
    rv = pd.to_numeric(R.loc[cc.case_id, "sen_Sen_Probability"], errors="coerce")
    flag = "" if (np.isnan(rv) and np.isnan(r.sen_probability)) or abs(rv - r.sen_probability) < 1e-6 else "  <-- differs"
    print(f"{cc.case_id:24s} R={rv:.6f}  MannKS={r.sen_probability:.6f}{flag}")
