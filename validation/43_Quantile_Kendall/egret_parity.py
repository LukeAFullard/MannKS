"""
Parity of MannKS.quantile_trend_test with the genuine USGS EGRET R code.

The reference numbers come from R, not from a Python re-implementation:
``egret_reference.R`` holds EGRET's own ``makeSortQ`` / ``trendSortQ`` /
``leapOdd`` / ``leapEven`` (verbatim from the EGRET vignette) and runs them on
an EGRET ``eList`` using the EGRET, rkt and zyp packages.  This module

* defines the parity scenarios (same daily series, same period of analysis),
* runs EGRET on them through ``rpy2`` (``run_egret``),
* runs MannKS with the equivalent settings (``run_mannks``),
* compares the two (``compare``), and
* stores EGRET's output as CSV fixtures under ``tests/data/egret_qk`` so the
  comparison is also asserted in the normal test suite on machines without R.

Regenerate the fixtures (requires R with EGRET, rkt, zyp, and rpy2)::

    python validation/43_Quantile_Kendall/egret_parity.py --write-fixtures

Requirements for the live comparison: R (>= 4.0) with packages EGRET, rkt and
zyp, plus ``rpy2`` (use ``rpy2==3.5.17`` with R < 4.5).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

R_SCRIPT = os.path.join(HERE, 'egret_reference.R')
FIXTURE_DIR = os.path.join(REPO_ROOT, 'tests', 'data', 'egret_qk')

_MONTH_ABBR = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# name -> (dataset, EGRET paStart, EGRET paLong, description)
SCENARIOS = {
    'climate_year': (
        'smooth', 4, 12,
        "Climate year (Apr-Mar), continuous flows, record starts/ends mid-year (partial years dropped)"),
    'water_year_ties': (
        'ties', 10, 12,
        "Water year (Oct-Sep), flows rounded to 0.1 (heavy ties), partial first/last years"),
    'calendar_year_ties': (
        'ties', 1, 12,
        "Calendar year (Jan-Dec), flows rounded to 0.1 (heavy ties)"),
    'winter_djf': (
        'ties', 12, 3,
        "Seasonal period of analysis Dec-Feb (90/91 days: exercises leapOdd)"),
}


def make_dataset(name: str) -> pd.DataFrame:
    """Deterministic synthetic daily discharge series used for the parity scenarios."""
    if name == 'smooth':
        rng = np.random.default_rng(42)
        dates = pd.date_range('2000-01-15', '2012-06-20', freq='D')
        t_years = np.arange(len(dates)) / 365.25
        seasonal = 1.0 + 0.6 * np.sin(2.0 * np.pi * dates.dayofyear / 365.25)
        q = 12.0 * seasonal * np.exp(0.015 * t_years) * rng.lognormal(0.0, 0.35, len(dates))
    elif name == 'ties':
        rng = np.random.default_rng(7)
        dates = pd.date_range('1990-12-13', '2003-11-02', freq='D')
        q = np.round(5.0 * rng.lognormal(0.0, 0.6, len(dates)), 1) + 0.1
    else:
        raise ValueError(f"Unknown dataset '{name}'.")
    return pd.DataFrame({'date': dates, 'q': q})


def mannks_settings(pa_start: int, pa_long: int) -> dict:
    """
    ``quantile_trend_test`` keyword arguments equivalent to EGRET's
    ``setPA(paStart, paLong)`` + ``makeSortQ`` + ``trendSortQ``.
    """
    months = None
    if pa_long != 12:
        months = [((pa_start - 1 + i) % 12) + 1 for i in range(pa_long)]
    return dict(
        block_frequency=f"YS-{_MONTH_ABBR[pa_start - 1]}",  # blocks start on the 1st of paStart
        months=months,                 # paLong < 12: keep only the period-of-analysis months
        require_complete=True,         # EGRET keeps a year only if every day is present
        standardization='egret',       # leapOdd / leapEven rank reduction
        transform='log',               # trendSortQ works on log(Q)
        block_time='egret',            # time = (DecYear[first] + DecYear[last]) / 2
        slope_scaling='year',
        tau_method='b',                # rkt reports tau-b (the MannKS default)
        min_size_per_block=1,
        min_blocks=2,
    )


def run_mannks(daily: pd.DataFrame, pa_start: int, pa_long: int):
    from MannKS import quantile_trend_test
    return quantile_trend_test(daily['q'].to_numpy(), pd.DatetimeIndex(daily['date']),
                               **mannks_settings(pa_start, pa_long))


def r_available() -> bool:
    """True if rpy2, R and the EGRET / rkt / zyp packages can be loaded."""
    try:
        import rpy2.robjects as ro
        ro.r('suppressPackageStartupMessages({library(EGRET); library(rkt); library(zyp)})')
        return True
    except Exception:
        return False


def run_egret(daily: pd.DataFrame, pa_start: int, pa_long: int) -> dict:
    """Run the genuine EGRET code (via rpy2) and return its output as numpy / pandas objects."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    ro.r['source'](R_SCRIPT)
    out = ro.r['run_egret_qk'](
        ro.StrVector([d.strftime('%Y-%m-%d') for d in pd.DatetimeIndex(daily['date'])]),
        ro.FloatVector(daily['q'].to_numpy(dtype=float)),
        int(pa_start), int(pa_long),
    )
    with localconverter(ro.default_converter + pandas2ri.converter):
        results = ro.conversion.rpy2py(out.rx2('results'))
    versions = out.rx2('versions')
    return {
        'sortQ': np.array(out.rx2('sortQ'), dtype=float),          # ranks x years
        'time': np.array(out.rx2('time'), dtype=float),            # midDecYear per year
        'results': results.reset_index(drop=True),
        'versions': {k: str(v) for k, v in zip(versions.names, versions)},
    }


def compare(mannks_result, egret: dict) -> dict:
    """Maximum absolute differences between MannKS and EGRET output."""
    df = mannks_result.results
    res = egret['results']
    matrix = mannks_result.matrix.to_numpy().T                      # ranks x years, like sortQ
    if matrix.shape != egret['sortQ'].shape:
        raise AssertionError(f"Matrix shape {matrix.shape} != EGRET sortQ shape {egret['sortQ'].shape}")

    def md(a, b):
        return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))

    return {
        'n_ranks': int(matrix.shape[0]),
        'n_years': int(matrix.shape[1]),
        'n_tied_ranks': int(np.sum(res['varS'].to_numpy() < res['varS'].max() - 1e-9)),
        'sortQ': md(matrix, egret['sortQ']),
        'S': md(df['s'], res['S']),
        'varS': md(df['var_s'], res['varS']),
        'tau': md(df['Tau'], res['tau']),
        'p': md(df['p'], res['pValue']),
        'slopeLog': md(df['slope'], res['slopeLog']),
        'slopePct': md(df['slope_pct'], res['slopePct']),
        'freq': md(df['plotting_position'], res['freq']),
        'z': md(df['z_position'], res['z']),
    }


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

def write_fixtures(out_dir: str = FIXTURE_DIR) -> dict:
    """Run EGRET on every scenario and store inputs + EGRET output as CSV."""
    os.makedirs(out_dir, exist_ok=True)
    meta = {'scenarios': {}, 'versions': None}
    written = set()
    for name, (dataset, pa_start, pa_long, desc) in SCENARIOS.items():
        daily = make_dataset(dataset)
        if dataset not in written:
            daily.assign(date=daily['date'].dt.strftime('%Y-%m-%d')).to_csv(
                os.path.join(out_dir, f'daily_{dataset}.csv'), index=False, float_format='%.17g')
            written.add(dataset)
        eg = run_egret(daily, pa_start, pa_long)
        pd.DataFrame(eg['sortQ']).to_csv(os.path.join(out_dir, f'{name}_sortQ.csv'),
                                        index=False, header=False, float_format='%.17g')
        pd.DataFrame({'midDecYear': eg['time']}).to_csv(os.path.join(out_dir, f'{name}_time.csv'),
                                                       index=False, float_format='%.17g')
        cols = ['rank', 'slopeLog', 'slopePct', 'pValue', 'pValueAdj', 'tau', 'S', 'varS', 'freq', 'z']
        eg['results'][cols].to_csv(os.path.join(out_dir, f'{name}_results.csv'),
                                   index=False, float_format='%.17g')
        meta['scenarios'][name] = {'dataset': dataset, 'paStart': pa_start, 'paLong': pa_long,
                                   'description': desc}
        meta['versions'] = eg['versions']
    with open(os.path.join(out_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write('\n')
    return meta


def load_fixture(name: str, fixture_dir: str = FIXTURE_DIR):
    """Return (daily DataFrame, egret dict, scenario metadata) from the stored EGRET output."""
    with open(os.path.join(fixture_dir, 'metadata.json'), encoding='utf-8') as f:
        meta = json.load(f)
    sc = meta['scenarios'][name]
    daily = pd.read_csv(os.path.join(fixture_dir, f"daily_{sc['dataset']}.csv"), parse_dates=['date'])
    egret = {
        'sortQ': pd.read_csv(os.path.join(fixture_dir, f'{name}_sortQ.csv'), header=None).to_numpy(dtype=float),
        'time': pd.read_csv(os.path.join(fixture_dir, f'{name}_time.csv'))['midDecYear'].to_numpy(dtype=float),
        'results': pd.read_csv(os.path.join(fixture_dir, f'{name}_results.csv')),
        'versions': meta['versions'],
    }
    return daily, egret, sc


if __name__ == '__main__':
    if '--write-fixtures' in sys.argv:
        if not r_available():
            sys.exit("R with EGRET, rkt and zyp (and rpy2) is required to write fixtures.")
        m = write_fixtures()
        print(f"Wrote EGRET fixtures to {FIXTURE_DIR} ({m['versions']})")
    live = r_available()
    print(f"Reference: {'live EGRET via rpy2' if live else 'stored EGRET fixtures'}")
    for scen, (dataset, pa_start, pa_long, desc) in SCENARIOS.items():
        if live:
            daily = make_dataset(dataset)
            eg = run_egret(daily, pa_start, pa_long)
        else:
            daily, eg, _ = load_fixture(scen)
        print(scen, compare(run_mannks(daily, pa_start, pa_long), eg))
