import numpy as np
import pytest

# Internal functions to be tested
from MannKS._stats import _mk_score_and_var_censored

def test_mk_score_and_var_censored_float_precision():
    """
    Test Case 1.1:
    Tests that the tie-breaking logic in `_mk_score_and_var_censored`
    is robust to floating-point precision issues.
    This test uses data with very small differences that could cause
    `np.diff` to return 0 if not handled carefully.
    """
    # Data with a very small difference between two points
    x = np.array([1.0, 1.0 + 1e-12, 1.5, 2.0])
    t = np.arange(len(x))
    censored = np.zeros_like(x, dtype=bool)
    cen_type = np.full_like(x, 'not', dtype=object)

    # The old method `np.min(np.diff(np.unique(x)))` might evaluate to 0
    # due to precision loss, which would lead to incorrect tie handling
    # and potentially a variance of 0.
    kenS, varS, D, Tau = _mk_score_and_var_censored(x, t, censored, cen_type)

    # Assert that the results are sensible and not zero
    # A score of 6 means all pairs are increasing, which is correct
    assert kenS == 6
    # Tau should be 1.0, as the series is perfectly monotonic
    assert np.isclose(Tau, 1.0)
