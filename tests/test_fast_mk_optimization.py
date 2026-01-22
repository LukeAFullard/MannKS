
import pytest
import numpy as np
import time
from MannKS._stats import _mk_score_and_var_censored

def test_fast_mk_optimization_correctness():
    """
    Verify that the fast O(N log N) path (activated for N>5000 uncensored)
    produces identical results to the chunked O(N^2) path.
    """
    np.random.seed(42)
    n = 6000 # Triggers fast path
    x = np.random.normal(0, 1, n)
    t = np.arange(n)

    # 1. Run via standard API (triggers Fast Path)
    start = time.time()
    s_fast, _, _, _ = _mk_score_and_var_censored(
        x, t, np.zeros(n, bool), np.full(n, 'not')
    )
    time_fast = time.time() - start

    # 2. Force Slow Path
    # We can force slow path by setting mk_test_method='lwp'
    # For uncensored data with no 'gt' censoring, 'lwp' method behaves identically
    # to 'robust' EXCEPT for the check that disables fast path.
    # We must ensure tie_break_method is 'robust' to match.
    start = time.time()
    s_slow, _, _, _ = _mk_score_and_var_censored(
        x, t, np.zeros(n, bool), np.full(n, 'not'),
        mk_test_method='lwp', tie_break_method='robust'
    )
    time_slow = time.time() - start

    print(f"\nFast Path Time: {time_fast:.4f}s")
    print(f"Slow Path Time: {time_slow:.4f}s")

    # Verify S score
    assert s_fast == s_slow, f"Fast S ({s_fast}) != Slow S ({s_slow})"

    # Verify Variance (var_s) is also identical
    # Since fast path reuses the existing robust variance calculation code,
    # it should match exactly if inputs (dupx, dupy) are set up correctly.
    # Note: _mk_score_and_var_censored returns (kenS, varS, D, Tau)
    var_fast = _mk_score_and_var_censored(x, t, np.zeros(n, bool), np.full(n, 'not'))[1]
    var_slow = _mk_score_and_var_censored(x, t, np.zeros(n, bool), np.full(n, 'not'),
                                          mk_test_method='lwp', tie_break_method='robust')[1]
    assert var_fast == var_slow, f"Fast Var ({var_fast}) != Slow Var ({var_slow})"

    # Expect significant speedup (e.g. > 10x)
    # N=6000: Fast ~0.01s, Slow ~20s.
    if time_slow > 1.0: # Only assert if slow path was actually slow enough to measure
        assert time_fast < time_slow / 5, "Fast path should be significantly faster"

def test_fast_mk_tied_timestamps():
    """
    Verify fast path preserves the 'Ordinal' tie-breaking behavior for timestamps.
    """
    np.random.seed(42)
    n = 6000
    x = np.random.normal(0, 1, n)
    # Create tied timestamps (0,0, 1,1, ...)
    t = np.repeat(np.arange(n//2), 2)

    # Fast path
    s_fast, _, _, _ = _mk_score_and_var_censored(
        x, t, np.zeros(n, bool), np.full(n, 'not')
    )

    # Slow path (via lwp override)
    s_slow, _, _, _ = _mk_score_and_var_censored(
        x, t, np.zeros(n, bool), np.full(n, 'not'),
        mk_test_method='lwp', tie_break_method='robust'
    )

    assert s_fast == s_slow, "Fast path failed to match slow path Ordinal tie handling"

def test_fast_mk_ties_in_x():
    """
    Verify fast path handles ties in X correctly.
    """
    np.random.seed(42)
    n = 6000
    x = np.random.randint(0, 5, n) # Lots of ties
    t = np.arange(n)

    s_fast, _, _, _ = _mk_score_and_var_censored(
        x, t, np.zeros(n, bool), np.full(n, 'not')
    )

    s_slow, _, _, _ = _mk_score_and_var_censored(
        x, t, np.zeros(n, bool), np.full(n, 'not'),
        mk_test_method='lwp', tie_break_method='robust'
    )

    assert s_fast == s_slow
