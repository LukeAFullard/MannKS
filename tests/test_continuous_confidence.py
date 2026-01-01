import numpy as np
import pytest
from MannKS.trend_test import trend_test
from MannKS.seasonal_trend_test import seasonal_trend_test
from MannKS.classification import classify_trend

def test_continuous_confidence_true():
    """
    Test that when continuous_confidence=True (default),
    weak trends are reported with a direction (e.g. increasing)
    and classified with a likelihood string (e.g. "Likely Increasing").
    """
    # Create a very weak increasing trend (noisy)
    np.random.seed(42)
    t = np.arange(20)
    # Slope of 0.05 is small relative to noise of sigma=1
    x = t * 0.05 + np.random.normal(0, 1, 20)

    # Run with default (continuous_confidence=True)
    res = trend_test(x, t, alpha=0.05)

    # Even if p > 0.05, we expect a direction if S != 0
    if res.s != 0:
        assert res.trend in ['increasing', 'decreasing']
        # Classification should be descriptive (e.g., Likely, Possible, As Likely as Not)
        # It should NOT be just "Increasing" or "No Trend" unless it hits edge cases
        assert any(x in res.classification for x in ["Likely", "Possible", "As Likely as Not"])

    # Check that h matches significance logic (False if p > 0.05)
    if res.p > 0.05:
        assert not res.h  # Use truthiness check, not 'is False'
    else:
        assert res.h      # Use truthiness check, not 'is True'

def test_continuous_confidence_false():
    """
    Test that when continuous_confidence=False,
    weak trends are reported as 'no trend' and classified as 'No Trend'.
    """
    # Same very weak trend
    np.random.seed(42)
    t = np.arange(20)
    x = t * 0.05 + np.random.normal(0, 1, 20)

    # Run with classical mode and strict alpha to ensure non-significance
    res = trend_test(x, t, alpha=0.01, continuous_confidence=False)

    # Assert it is not significant
    if res.p > 0.01:
        assert not res.h
        assert res.trend == 'no trend'
        assert res.classification == 'No Trend'

def test_strong_trend_continuous_vs_classical():
    """
    Test that for strong trends, both modes agree on direction.
    """
    x = np.arange(20) # Perfect increasing trend
    t = np.arange(20)

    # Continuous
    res_cont = trend_test(x, t, continuous_confidence=True)
    assert res_cont.trend == 'increasing'
    assert "Highly Likely" in res_cont.classification

    # Classical
    res_class = trend_test(x, t, continuous_confidence=False)
    assert res_class.trend == 'increasing'
    assert res_class.classification == 'Increasing'

def test_seasonal_continuous_confidence():
    """
    Verify the parameter works for seasonal_trend_test as well.
    """
    # Create data for 2 years, monthly (24 points)
    # Weak trend + seasonality
    np.random.seed(10)
    t = np.arange(24)
    x = t * 0.05 + np.random.normal(0, 1, 24)
    seasonality = np.tile(np.sin(np.linspace(0, 2*np.pi, 12)), 2)
    x = x + seasonality * 5

    # Continuous
    res = seasonal_trend_test(x, t, period=12, continuous_confidence=True)
    if res.s != 0:
        # Should have a direction
        assert res.trend in ['increasing', 'decreasing']

    # Classical with strict alpha
    res_class = seasonal_trend_test(x, t, period=12, alpha=0.00001, continuous_confidence=False)

    # Ensure it fails significance
    assert not res_class.h  # Truthiness check
    assert res_class.trend == 'no trend'
    assert res_class.classification == 'No Trend'

def test_exact_zero_trend():
    """
    Test behavior when S is exactly zero.
    """
    # Perfectly symmetrical data -> S should be 0
    x = [1, 2, 3, 2, 1]
    t = np.arange(5)

    res = trend_test(x, t, continuous_confidence=True)
    assert res.s == 0
    assert res.z == 0
    assert res.trend == 'indeterminate'
    # Default map for 0.5 confidence (which C is when p=1.0) is "As Likely as Not"
    assert "As Likely as Not" in res.classification
