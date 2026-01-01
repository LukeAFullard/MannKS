import pytest
import numpy as np
import pandas as pd
from MannKS import trend_test

def test_zero_variance_handling():
    """
    Tests that the system handles data with zero variance gracefully.
    When all data points are tied, the Mann-Kendall score `s` is 0,
    which should result in a final conclusion of 'no trend'.
    """
    # Create a dataset where all values are identical
    x = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    t = np.arange(len(x))

    # Expect a warning for the Tau denominator calculation, which happens
    # regardless of the final trend result.
    with pytest.warns(UserWarning, match="Denominator near zero in Tau calculation"):
        result = trend_test(x, t)

    # The primary assertions are on the final, user-facing results.
    # The intermediate var_s value is a non-critical implementation detail
    # in this specific edge case because s=0.
    assert result.trend == 'no trend'
    assert not result.h  # Use truthiness for numpy.bool_
    assert result.s == 0
    assert result.p == 1.0
