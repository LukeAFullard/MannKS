"""
Parity with the LWP-TRENDS R code (v2502).

Reference values in ``tests/data/lwp_parity_fixtures.json`` were produced by
running the R functions (``GetKendal``, ``GetInterObservationSlopes``,
``MannKendall``/``SenSlope``, ``SeasonalKendall``/``SeasonalSenSlope``) on the
same inputs; see ``validation/r_parity``.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

import MannKS as mk
from MannKS._stats import _mk_score_and_var_censored, _sens_estimator_censored

with open(os.path.join(os.path.dirname(__file__), 'data', 'lwp_parity_fixtures.json')) as fh:
    FIXTURES = json.load(fh)

REL_TOL = 1e-6
PERIOD = {'Year': 'year', 'Month': 'month', 'Qtr': 'quarter'}
SEASON = {'Month': 'month', 'Qtr': 'quarter'}


@pytest.mark.parametrize('case', FIXTURES['kernel'], ids=lambda c: f"kernel{c['case']}")
def test_kernel_matches_getkendal(case):
    x = np.array(case['values'], dtype=float)
    cen = np.array(case['cen_type'], dtype=object)
    s, var_s, _, _ = _mk_score_and_var_censored(x, np.arange(len(x)), cen != 'not', cen,
                                                mk_test_method='lwp', tie_break_method='lwp')
    assert s == case['R']['S']
    assert var_s == pytest.approx(case['R']['VarS'], rel=1e-12)
    # the default tie-break gives the same variance
    var_robust = _mk_score_and_var_censored(x, np.arange(len(x)), cen != 'not', cen,
                                            mk_test_method='lwp')[1]
    assert var_robust == pytest.approx(case['R']['VarS'], rel=1e-12)

    t_years = (pd.to_datetime(case['dates']) - pd.Timestamp('1970-01-01')).days.to_numpy() / 365.25
    sen = np.median(_sens_estimator_censored(x, t_years, cen, method='lwp'))
    assert sen == pytest.approx(case['R']['sen'], rel=1e-10)


def _run_lwp_mode(case):
    values = [np.nan if v == 'NA' else v for v in case['values']]
    x = mk.prepare_censored_data(values)
    t = pd.to_datetime(case['dates']).to_numpy()
    common = dict(alpha=0.10, mk_test_method='lwp', ci_method='lwp', sens_slope_method='lwp',
                  tau_method='a', tie_break_method='lwp', slope_scaling='year')
    if case['seasonal']:
        return mk.seasonal_trend_test(x, t, season_type=SEASON[case['timeincr']],
                                      agg_method='lwp' if case['use_mid'] else 'median', **common)
    return mk.trend_test(x, t, agg_method='lwp' if case['use_mid'] else 'lwp_median',
                         agg_period=PERIOD[case['timeincr']], **common)


@pytest.mark.parametrize('case', FIXTURES['wrapper'], ids=lambda c: c['case'])
def test_wrapper_matches_lwp_trends(case):
    r = _run_lwp_mode(case)
    fields = {'S': 's', 'VarS': 'var_s', 'tau': 'Tau', 'Z': 'z', 'p': 'p', 'C': 'C', 'Cd': 'Cd',
              'slope': 'slope', 'lower_ci': 'lower_ci', 'upper_ci': 'upper_ci',
              'prop_censored': 'prop_censored', 'sen_probability': 'sen_probability'}
    for rkey, pkey in fields.items():
        expected = case['R'][rkey]
        got = float(getattr(r, pkey))
        if abs(expected) < 1e-12:
            assert abs(got) < 1e-9, (rkey, got, expected)
        else:
            assert got == pytest.approx(expected, rel=REL_TOL), (rkey, got, expected)
