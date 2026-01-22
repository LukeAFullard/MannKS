
import pytest
import numpy as np
import pandas as pd
from MannKS.segmented_trend_test import segmented_trend_test, SegmentedTrendResult, calculate_breakpoint_probability

def test_result_type():
    t = np.arange(10)
    x = t
    res = segmented_trend_test(x, t, n_breakpoints=0)
    assert isinstance(res, SegmentedTrendResult)

def test_segmented_trend_basic():
    # Linear data, no breakpoints
    t = np.arange(20)
    x = t + np.random.normal(0, 0.1, 20)

    result = segmented_trend_test(x, t, n_breakpoints=0)
    assert result.n_breakpoints == 0
    assert len(result.segments) == 1
    assert result.segments.iloc[0]['slope'] > 0.8
    assert result.segments.iloc[0]['slope'] < 1.2

def test_segmented_trend_breakpoint():
    # One breakpoint at t=10
    t = np.arange(20)
    x = np.concatenate([t[:10], 10 - 0.5 * (t[10:] - 10)]) + np.random.normal(0, 0.1, 20)

    # Force 1 breakpoint
    result = segmented_trend_test(x, t, n_breakpoints=1)

    assert result.n_breakpoints == 1
    assert len(result.breakpoints) == 1
    assert np.abs(result.breakpoints[0] - 10) < 2

    # Check slopes
    s1 = result.segments.iloc[0]['slope']
    s2 = result.segments.iloc[1]['slope']
    assert s1 > 0.8
    assert s2 < -0.3

def test_bagging_integration():
    # Test that bagging runs without error
    t = np.arange(30)
    # Clear breakpoint
    x = np.concatenate([t[:15], 15 - (t[15:] - 15)]) + np.random.normal(0, 0.1, 30)

    result = segmented_trend_test(x, t, n_breakpoints=1, use_bagging=True, n_bootstrap=20)

    assert result.n_breakpoints == 1
    assert len(result.breakpoints) == 1
    # Bagging should be somewhat accurate
    assert np.abs(result.breakpoints[0] - 15) < 3

def test_bagging_with_dataframe_input():
    # Test bagging with DataFrame input (implicit n_breakpoints search)
    t = np.arange(30)
    x_val = np.concatenate([t[:15], 15 - (t[15:] - 15)]) + np.random.normal(0, 0.1, 30)
    df = pd.DataFrame({'value': x_val})

    # We force n_breakpoints=1 to ensure bagging runs
    result = segmented_trend_test(df, t, n_breakpoints=1, use_bagging=True, n_bootstrap=20)

    assert result.n_breakpoints == 1
    assert len(result.breakpoints) == 1
    assert np.abs(result.breakpoints[0] - 15) < 3

def test_insufficient_data():
    t = np.arange(1)
    x = np.array([1])
    # with pytest.warns(UserWarning, match="Insufficient data for segmented analysis"):
    res = segmented_trend_test(x, t)
    assert res.n_breakpoints == 0
    assert res.segments.empty
    assert "Insufficient data for segmented analysis." in res.warnings[0]

def test_dataframe_input():
    t = np.arange(20)
    x = t + np.random.normal(0, 0.1, 20)
    df = pd.DataFrame({'value': x, 'censored': False, 'cen_type': 'none'})

    result = segmented_trend_test(df, t, n_breakpoints=0)
    assert result.n_breakpoints == 0

def test_no_bagging_if_n_zero():
    # Bagging shouldn't run if n_breakpoints=0
    t = np.arange(20)
    x = t + np.random.normal(0, 0.1, 20)
    result = segmented_trend_test(x, t, n_breakpoints=0, use_bagging=True)
    assert result.n_breakpoints == 0

def test_datetime_support():
    t = pd.date_range("2020-01-01", periods=20, freq='D')
    x = np.arange(20) + np.random.normal(0, 0.1, 20)

    result = segmented_trend_test(x, t, n_breakpoints=0)
    assert result.is_datetime
    assert isinstance(result.segments, pd.DataFrame)

def test_slope_scaling():
    # Linear trend: 1 unit per day
    t = pd.date_range('2020-01-01', periods=10, freq='D')
    x = np.arange(10, dtype=float)

    result = segmented_trend_test(x, t, n_breakpoints=0, slope_scaling='year')

    # Slope should be approx 365.25
    slope = result.segments.iloc[0]['slope']
    assert 360 < slope < 370
    assert result.segments.iloc[0]['slope_units'] == "units per year"

def test_hicensor_logic():
    # Test that hicensor correctly modifies data
    t = np.arange(10)
    x = np.arange(10, dtype=float)

    # Introduce one censored value at 8
    df = pd.DataFrame({'value': x})
    df['censored'] = False
    df['cen_type'] = 'none'

    df.loc[8, 'value'] = 8.0
    df.loc[8, 'censored'] = True
    df.loc[8, 'cen_type'] = 'lt'

    # Without hicensor: Only index 8 is censored
    # With hicensor=True: indices 0-7 (<8) become censored <8

    # We can't easily inspect the internal DataFrame, but we can verify execution
    # and maybe check if the slope result differs significantly if we use a different estimator
    # For now, ensure it runs without error

    result = segmented_trend_test(df, t, n_breakpoints=0, hicensor=True)
    assert result.n_breakpoints == 0

def test_criterion_parameter():
    # Verify that criterion can be passed
    t = np.arange(20)
    x = t + np.random.normal(0, 0.1, 20)

    # Check AIC
    result_aic = segmented_trend_test(x, t, n_breakpoints=0, criterion='aic')
    assert result_aic.bic != np.inf
    assert result_aic.aic != np.inf

    # Check BIC
    result_bic = segmented_trend_test(x, t, n_breakpoints=0, criterion='bic')
    assert result_bic.bic != np.inf
    assert result_bic.aic != np.inf

def test_dataframe_missing_value_column():
    t = np.arange(10)
    df = pd.DataFrame({'a': np.arange(10), 'b': np.arange(10)})

    with pytest.raises(ValueError, match="must contain a 'value' column"):
        segmented_trend_test(df, t)

def test_predict_method():
    t = np.arange(20, dtype=float)
    x = 2.0 * t + 5.0

    result = segmented_trend_test(x, t, n_breakpoints=0)

    # Predict at existing points
    y_pred = result.predict(t)
    assert np.allclose(y_pred, x, atol=1e-5)

    # Predict at new points
    t_new = np.array([20.0, 21.0])
    x_expected = 2.0 * t_new + 5.0
    y_pred_new = result.predict(t_new)
    assert np.allclose(y_pred_new, x_expected, atol=1e-5)

def test_predict_with_scaling():
    # Date time t
    t_start = pd.Timestamp('2020-01-01')
    t = pd.date_range(start=t_start, periods=10, freq='D')
    t_numeric = t.astype('int64') / 1e9

    # Slope of 1 unit per day
    slope_per_day = 1.0
    slope_per_sec = slope_per_day / (24*3600)

    x = slope_per_sec * (t_numeric - t_numeric[0]) + 10.0

    # Fit with slope scaling
    result = segmented_trend_test(x, t, n_breakpoints=0, slope_scaling='day')

    # Predict
    t_check = t[5]
    y_pred = result.predict([t_check])[0]
    y_true = x[5]

    assert np.isclose(y_pred, y_true, atol=1e-5)

def test_predict_with_breakpoints():
    t = np.arange(20, dtype=float)
    x = np.concatenate([t[:10], 10 - 0.5 * (t[10:] - 10)])

    # Force 1 breakpoint (should be around 10)
    result = segmented_trend_test(x, t, n_breakpoints=1)

    assert result.n_breakpoints == 1

    # Predict and compare with original data
    # (Allow small error due to potential fitting differences, but should be close)
    y_pred = result.predict(t)

    # Use loose tolerance because OLS breakpoint finding might vary slightly
    # and robust regression on segments might differ slightly from perfect OLS lines
    # But for this clean data it should be quite close.
    # The '10' point is the pivot.

    assert np.allclose(y_pred, x, atol=1.0)

def test_breakpoint_probability():
    t = np.arange(50)
    # Breakpoint at 25
    x = np.concatenate([t[:25], 25 - (t[25:] - 25)]) + np.random.normal(0, 0.1, 50)

    # Run with bagging
    result = segmented_trend_test(x, t, n_breakpoints=1, use_bagging=True, n_bootstrap=20)

    # Probability in window [20, 30] should be high
    prob = calculate_breakpoint_probability(result, 20, 30)
    assert prob > 0.8

    # Probability in window [0, 10] should be low
    prob_low = calculate_breakpoint_probability(result, 0, 10)
    assert prob_low < 0.2

def test_breakpoint_cis_format():
    t = np.arange(30)
    x = np.concatenate([t[:15], 15 - (t[15:] - 15)]) + np.random.normal(0, 0.1, 30)

    # 1. Without bagging (OLS CIs)
    result = segmented_trend_test(x, t, n_breakpoints=1, use_bagging=False)
    assert len(result.breakpoint_cis) == 1
    assert len(result.breakpoint_cis[0]) == 2
    # OLS CIs might be nan if piecewise_regression fails to compute them, but usually they are floats

    # 2. With bagging (Bootstrap CIs)
    result_bag = segmented_trend_test(x, t, n_breakpoints=1, use_bagging=True, n_bootstrap=20)
    assert len(result_bag.breakpoint_cis) == 1
    assert len(result_bag.breakpoint_cis[0]) == 2
    ci = result_bag.breakpoint_cis[0]
    assert ci[0] <= ci[1]

def test_bagging_zero_breakpoints():
    t = np.arange(20)
    x = t + np.random.normal(0, 0.1, 20)

    # Force n=0
    result = segmented_trend_test(x, t, n_breakpoints=0, use_bagging=True)
    assert result.n_breakpoints == 0
    assert len(result.breakpoints) == 0
    assert len(result.breakpoint_cis) == 0
    # Should not crash
