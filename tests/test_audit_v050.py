
import pytest
import numpy as np
import pandas as pd
from MannKS import trend_test, seasonal_trend_test

def test_fast_mk_edge_case_n5001():
    """
    Audit Recommendation: n=5001 edge case (just over threshold).
    Verifies that the fast path is triggered and produces correct results.
    """
    np.random.seed(42)
    n = 5001
    t = np.arange(n)
    # Clear trend: x = 0.5*t + noise
    x = 0.5 * t + np.random.normal(0, 10, n)

    result = trend_test(x, t)

    assert result.computation_mode == 'fast'
    assert result.trend == 'increasing'
    assert result.h == True
    # Fast mode slope pairs should be capped at default max_pairs (100,000)
    # n=5001 => ~12.5M pairs.
    assert result.pairs_used is not None
    assert result.pairs_used <= 100000
    assert result.slope > 0.4 and result.slope < 0.6


def test_seasonal_stratification_n15000():
    """
    Seasonal with n=15,000: by default the test uses every observation (it is
    exact in O(n log n)); stratified subsampling only happens on request.
    """
    np.random.seed(42)
    n = 15000
    period = 12
    t = np.arange(n)
    seasonality = np.tile(np.arange(period), n // period + 1)[:n]
    x = 0.01 * t + seasonality * 10 + np.random.normal(0, 5, n)

    result = seasonal_trend_test(x, t, period=period)
    assert result.computation_mode == 'fast'
    assert result.trend == 'increasing'
    assert not any("stratified sampling" in note.lower() for note in result.analysis_notes)

    # S is the sum of the within-season S over all the data.
    from MannKS._stats import _mk_score_and_var_censored
    season = t % period
    s_full = sum(_mk_score_and_var_censored(x[season == k], t[season == k],
                                            np.zeros((season == k).sum(), bool),
                                            np.full((season == k).sum(), 'not'))[0]
                 for k in range(period))
    assert result.s == s_full

    sub = seasonal_trend_test(x, t, period=period, max_per_season=500)
    assert any("stratified sampling" in note.lower() for note in sub.analysis_notes)


def test_heavy_ties():
    """
    Audit Recommendation: Heavy ties (>50% tied values).
    """
    np.random.seed(42)
    n = 6000 # > 5000 to trigger fast path
    t = np.arange(n)

    # 80% of data is 0 to ensure > 50% tied pairs
    # (0.8*n)^2 / n^2 approx 0.64 ratio of tied pairs
    x = np.random.normal(0, 1, n)
    x[:int(0.8*n)] = 0

    # The O(N log N) path recovers S exactly from Tau-b even with heavy ties.
    result = trend_test(x, t)
    exact = trend_test(x, t, mk_test_method='lwp')  # forces the pairwise path
    assert result.s == exact.s
    assert result.var_s == pytest.approx(exact.var_s)

    assert result.computation_mode == 'fast'
    # Result should be valid
    assert isinstance(result.slope, float)
