"""Code-level checks: input mutation, tz-aware datetimes, middle_lwp, aliases, sen_probability vs R."""
import os, sys, warnings, copy
import numpy as np, pandas as pd
sys.path.insert(0, os.environ.get("MANNKS_SRC", os.path.join(os.path.dirname(__file__), "../..")))
import MannKS as mk
warnings.filterwarnings("ignore")
def hdr(s): print("\n=== " + s + " ===")

hdr("Input mutation (DataFrame and Series passed in must be unchanged)")
rng = np.random.default_rng(0)
d = pd.Series(pd.date_range("2000-01-01", periods=48, freq="MS") + pd.Timedelta(days=10))
X = mk.prepare_censored_data(["<1" if v < 1 else round(v, 2) for v in 1.5 + rng.normal(0, 0.6, 48)])
X.index = np.arange(500, 548)
funcs = {
    "trend_test": lambda: mk.trend_test(X, d, agg_method="lwp", agg_period="quarter", hicensor=True),
    "seasonal_trend_test": lambda: mk.seasonal_trend_test(X, d, agg_method="median", hicensor=True),
    "check_seasonality": lambda: mk.check_seasonality(X, d),
    "rolling_trend_test": lambda: mk.rolling_trend_test(X, d, window="730D"),
    "segmented_trend_test": lambda: mk.segmented_trend_test(X, d, n_breakpoints=1),
}
for name, f in funcs.items():
    X0, d0 = X.copy(deep=True), d.copy(deep=True)
    try:
        f(); ok = X.equals(X0) and d.equals(d0)
        print(f"{name:22s} input unchanged: {ok}")
    except Exception as e:
        print(f"{name:22s} raised {type(e).__name__}: {e}")
df = pd.DataFrame({"t": d, "value": X.value.values, "censored": X.censored.values, "cen_type": X.cen_type.values})
df0 = df.copy(); mk.inspect_trend_data(df); print(f"{'inspect_trend_data':22s} input unchanged: {df.equals(df0)}")

hdr("Timezone-aware datetimes give the same slope as naive UTC")
dn = pd.date_range("2000-01-01", periods=40, freq="90D")
x = 0.01 * np.arange(40) + rng.normal(0, 0.1, 40)
a = mk.trend_test(x, dn, slope_scaling="year")
for tz_input in (pd.Series(dn.tz_localize("UTC")), dn.tz_localize("Pacific/Auckland"), pd.Series(dn.tz_localize("Pacific/Auckland"))):
    try:
        b = mk.trend_test(x, tz_input, slope_scaling="year")
        print(f"{type(tz_input).__name__:14s} tz={tz_input.dt.tz if hasattr(tz_input,'dt') else tz_input.tz}: slope {b.slope:.6f} vs naive {a.slope:.6f}")
    except Exception as e:
        print(f"{type(tz_input).__name__} tz-aware: raised {type(e).__name__}: {e}")
try:
    s = mk.seasonal_trend_test(x, pd.Series(dn.tz_localize("Pacific/Auckland")))
    print("seasonal tz-aware ok, slope", s.slope)
except Exception as e:
    print("seasonal tz-aware raised", type(e).__name__, e)

hdr("agg_method='middle_lwp' with agg_period='year' (used by validation as the LWP default)")
dd = pd.to_datetime(["2001-01-01", "2001-06-20", "2001-12-20", "2002-02-03", "2002-07-04", "2002-12-01"])
xx = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
for m in ("middle_lwp", "lwp"):
    r = mk.trend_test(xx, dd, agg_method=m, agg_period="year", min_size=None)
    print(f"{m:10s}: n after agg = {r.s and None}", end=" ")
    from MannKS._helpers import _prepare_data
from MannKS._helpers import _prepare_data, _aggregate_by_group, _value_for_time_increment
data, _ = _prepare_data(xx, dd, False)
key = pd.to_datetime(data.t_original).dt.to_period("Y")
print()
for y, g in data.groupby(key):
    mid = _aggregate_by_group(g, "middle_lwp", True)
    lwp = _value_for_time_increment(g, pd.Series([y] * len(g), index=g.index), "Y")
    print(f"  {y}: middle_lwp picks {pd.Timestamp(mid.t_original.iloc[0]).date()}, 'lwp' picks {pd.Timestamp(lwp.t_original.iloc[0]).date()} (R: obs closest to 1 Jul)")

hdr("seasonal_trend_test agg_method='lwp' for sub-daily season types (pandas offset aliases)")
dh = pd.date_range("2020-01-01", periods=24 * 20, freq="h")
xh = rng.normal(size=len(dh))
for st in ("hour", "day_of_week"):
    try:
        r = mk.seasonal_trend_test(xh, dh, season_type=st, agg_method="lwp")
        print(f"season_type={st}: ok (p={r.p:.3f})")
    except Exception as e:
        print(f"season_type={st}: raised {type(e).__name__}: {e}")

hdr("Numeric t in decimal years with seasonal_trend_test")
tdec = 2000 + np.arange(120) / 12
xs = np.sin(2 * np.pi * np.arange(120) / 12) + 0.01 * np.arange(120)
try:
    r = mk.seasonal_trend_test(xs, tdec, period=12)
    print(f"period=12 with decimal-year t: s={r.s}, slope={r.slope:.4f}, notes={r.analysis_notes[:2]}")
    seas = (np.floor(tdec - tdec.min()) % 12).astype(int)
    print("  seasons assigned:", np.unique(seas))
except Exception as e:
    print("raised", e)
