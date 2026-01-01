import pytest
import numpy as np
from MannKS._stats import _confidence_intervals, _sens_estimator_censored

def test_confidence_intervals_lwp_method():
    """
    Test the LWP confidence interval method.
    """
    slopes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    var_s = 12
    alpha = 0.05

    # With the direct method, the indices will be rounded
    lower_ci_direct, upper_ci_direct = _confidence_intervals(slopes, var_s, alpha, method='direct')

    # With the LWP method, the indices will be interpolated
    lower_ci_lwp, upper_ci_lwp = _confidence_intervals(slopes, var_s, alpha, method='lwp')

    # The results should be different
    assert lower_ci_direct != lower_ci_lwp
    assert upper_ci_direct != upper_ci_lwp

    # And the LWP results should be the result of interpolation
    assert lower_ci_lwp == pytest.approx(1.61, abs=0.01)
    assert upper_ci_lwp == pytest.approx(8.39, abs=0.01)

def test_sens_estimator_censored_ambiguity_logic():
    """
    Test the corrected ambiguity logic in `_sens_estimator_censored`.

    This test case is designed to exercise all four ambiguity rules that were
    corrected in the bug fix, ensuring that the index ordering and direction
    checks are now correct.
    """
    # Setup a dataset that triggers all ambiguity rules
    t = np.array([1, 2, 3, 4, 5], dtype=float)
    x = np.array([4, 2, 6, 5, 7], dtype=float)
    cen_type = np.array(['lt', 'not', 'not', 'gt', 'lt'])

    # Manually calculated expected slopes based on corrected logic:
    # Pair (i,j) | cen_pair | raw_slope | Ambiguous? | Final Slope
    # -----------------------------------------------------------------
    # (0,1) <4, 2 | not lt   | < 0       | YES        | 0
    # (0,2) <4, 6 | not lt   | > 0       | NO         | (6 - 4*0.5) / 2 = 2.0
    # (0,3) <4, >5| gt lt    | > 0       | NO         | (5*1.1 - 4*0.5) / 3 = 1.166
    # (0,4) <4, <7| lt lt    | > 0       | YES (type) | 0
    # (1,2) 2, 6  | not not  | > 0       | NO         | (6 - 2) / 1 = 4.0
    # (1,3) 2, >5 | gt not   | > 0       | NO         | (5*1.1 - 2) / 2 = 1.75
    # (1,4) 2, <7 | lt not   | > 0       | YES        | 0
    # (2,3) 6, >5 | gt not   | < 0       | YES        | 0
    # (2,4) 6, <7 | lt not   | > 0       | YES        | 0
    # (3,4) >5, <7| lt gt    | < 0       | NO         | (7*0.5 - 5*1.1) / 1 = -2.0
    #
    # Final slopes: [0, 2.0, 1.166, 0, 4.0, 1.75, 0, 0, 0, -2.0]
    # Sorted: [-2.0, 0, 0, 0, 0, 0, 1.166, 1.75, 2.0, 4.0]
    # Median: (0 + 0) / 2 = 0

    # Run the function with the 'lwp' method to get 0 for ambiguous slopes
    slopes = _sens_estimator_censored(x, t, cen_type, method='lwp')

    # The median of all calculated pairwise slopes is the Sen's slope
    sens_slope = np.median(slopes)

    assert sens_slope == 0.0
