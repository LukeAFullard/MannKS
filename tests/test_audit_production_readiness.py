
import numpy as np
import pandas as pd
import pytest
import warnings
from MannKS import trend_test

def test_audit_boundary_exactness():
    """
    Verify that N=5000 (Full) and N=5000 (Forced Fast MK Score) produce mathematically
    consistent MK Scores (S) and Variances.
    """
    np.random.seed(42)

    # Case 1: N = 5000 (Uses Full Mode by default)
    n1 = 5000
    x1 = np.random.normal(0, 1, n1)
    t1 = np.arange(n1)

    res1 = trend_test(x1, t1)

    # Case 2: N = 5000 (Forced Fast Mode)
    # This forces the O(N log N) MK calculation even for N=5000
    # The S and VarS should be IDENTICAL to Full Mode.
    res2 = trend_test(x1, t1, large_dataset_mode='fast')

    # Use approx for floating point safety, though they should be mathematically identical
    assert res1.s == pytest.approx(res2.s), "MK Score mismatch between Full and Forced Fast mode"
    assert res1.var_s == pytest.approx(res2.var_s), "Variance mismatch between Full and Forced Fast mode"

def test_mk_score_exactness_large_n():
    """
    Verify O(N log N) MK Score matches O(N^2) exactly for uncensored data at N=5005.
    """
    n = 5005
    np.random.seed(42)
    x = np.random.randn(n)
    t = np.arange(n)

    # Full mode (O(N^2)) - force strict calculation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_full = trend_test(x, t, large_dataset_mode='full')

    # Fast mode (O(N log N) for MK Score)
    res_fast = trend_test(x, t, large_dataset_mode='fast', random_state=42)

    assert res_full.s == res_fast.s, f"S mismatch: Full={res_full.s}, Fast={res_fast.s}"
    assert np.isclose(res_full.var_s, res_fast.var_s), "Var(S) mismatch"
    assert np.isclose(res_full.z, res_fast.z), "Z mismatch"

def test_audit_determinism():
    """
    Verify that setting random_state ensures bit-exact reproducibility
    for stochastic components (Fast Sen's Slope).
    """
    n = 6000 # Triggers fast mode
    x = np.random.normal(0, 1, n)
    t = np.arange(n)

    seed = 12345

    res1 = trend_test(x, t, random_state=seed)
    res2 = trend_test(x, t, random_state=seed)

    assert res1.slope == res2.slope, "Slopes are not identical with same seed!"
    assert res1.lower_ci == res2.lower_ci, "CIs are not identical with same seed!"

    # Verify different seed gives different result
    res3 = trend_test(x, t, random_state=seed+1)
    assert res1.slope != res3.slope, "Slopes should differ with different seeds"

def test_audit_heavy_ties():
    """
    Verify behavior with heavy ties.
    The O(N log N) implementation warns if tied PAIRS > 50%.
    Note: 50% tied values != 50% tied pairs.
    Need ~71% tied values to get >50% tied pairs (0.71^2 approx 0.5).
    Using 80% to be safe.
    """
    n = 5500
    np.random.seed(42)
    # 80% ties (Single value dominates) -> 0.8^2 = 0.64 > 0.5 tied pairs
    x = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    t = np.arange(n)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Run in fast mode
        res = trend_test(x, t, large_dataset_mode='fast')

        # Check for the specific warning in captured warnings
        tie_warnings = [str(warn.message) for warn in w if "Heavy ties detected" in str(warn.message)]

        # Also check result object warnings
        res_warnings = [str(rw) for rw in res.warnings if "Heavy ties detected" in str(rw)]

        assert tie_warnings or res_warnings, "Did not detect heavy ties warning."

def test_audit_censored_fallback():
    """
    Verify that censored data falls back to appropriate algorithms.
    For N > 5000 with censored data, MK Score calculation must fall back
    to exact O(N^2) or Chunked O(N^2) because O(N log N) doesn't support censored.

    We verify this by ensuring S matches the 'full' mode result exactly.
    """
    n = 5005
    np.random.seed(42)
    x = np.random.normal(0, 1, n)
    t = np.arange(n)
    censored = np.zeros(n, dtype=bool)
    censored[::20] = True # 5% censored
    cen_type = np.full(n, 'not', dtype=object)
    cen_type[censored] = 'lt'

    df = pd.DataFrame({'value': x, 't': t, 'censored': censored, 'cen_type': cen_type})

    # Reference run (Full)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_full = trend_test(df, t=df['t'], large_dataset_mode='full')

    # Test run (Fast - requesting fast slope, but MK score should fallback)
    res_fast = trend_test(df, t=df['t'], large_dataset_mode='fast', random_state=42)

    assert res_fast.computation_mode == 'fast', "Overall mode should be 'fast' (slope)"

    # S should match exactly
    assert res_full.s == res_fast.s, "S mismatch: Censored fallback failed to produce exact score"
    assert np.isclose(res_full.var_s, res_fast.var_s), "Variance mismatch"

def test_audit_seasonal_stratification_detail():
    """
    Verify that seasonal stratification produces the exact expected sample sizes.
    """
    # 2 seasons, unbalanced.
    n_per_season = 10000
    s1 = pd.DataFrame({'value': np.random.randn(n_per_season), 'season': 1, 't': np.arange(n_per_season)})
    s2 = pd.DataFrame({'value': np.random.randn(n_per_season), 'season': 2, 't': np.arange(n_per_season, 2*n_per_season)})

    df = pd.concat([s1, s2])

    # We call the internal function directly to verify exact logic
    from MannKS._large_dataset import stratified_seasonal_sampling

    sampled = stratified_seasonal_sampling(df, season_col='season', max_per_season=500, random_state=99)

    counts = sampled['season'].value_counts()

    assert counts[1] == 500
    assert counts[2] == 500
    assert len(sampled) == 1000
