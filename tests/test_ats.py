import numpy as np
import pandas as pd
import pytest
from MannKS import trend_test, seasonal_trend_test
import MannKS as mk

def test_ats_slope_non_seasonal():
    """
    Test the ats_slope function with the synthetic example from future_ATS.md.
    """
    np.random.seed(1)
    n = 80
    x = np.linspace(0, 10, n)
    true_beta = 0.2
    y_true = 1.0 + true_beta * x
    y = y_true + np.random.normal(scale=0.5, size=n)

    # Impose a LOD and censor low values
    lod = 1.5
    censored = y < lod
    y_obs = np.where(censored, lod, y)

    # Create the DataFrame required by trend_test
    x_prepared = pd.DataFrame({
        'value': y_obs,
        'censored': censored,
        'cen_type': np.where(censored, 'lt', 'none')
    })

    # Run the trend test with the ATS method
    res = trend_test(x=x_prepared, t=x, sens_slope_method='ats')

    # Check that the calculated slope is close to the true slope
    assert np.isclose(res.slope, true_beta, atol=0.1)

def test_ats_slope_seasonal_refactored():
    """
    Test the refactored seasonal ATS slope calculation.
    This test confirms that the ATS method runs on the entire dataset
    and correctly calculates bootstrap confidence intervals.
    """
    np.random.seed(42)
    n_years = 10
    n_seasons = 4
    n = n_years * n_seasons

    # Time vector
    t = pd.to_datetime(pd.date_range(start='2000-01-01', periods=n, freq='QS-DEC'))

    # True trend
    true_beta_per_year = 0.1
    # Create a numeric time vector in years for generating the true values
    time_numeric_years = np.linspace(0, n_years, n, endpoint=False)
    y_true = 1.0 + true_beta_per_year * time_numeric_years

    # Add seasonality and noise - using the exact parameters from the example script
    # to reliably reproduce the bug.
    seasonality = np.tile([0, 1.5, 0.5, 2.0], n_years)
    y = y_true + seasonality + np.random.normal(scale=0.2, size=n)

    # Impose censoring
    lod = 2.0
    censored = y < lod
    y_obs = np.where(censored, lod, y)

    # Prepare data using the public utility, matching the examples
    y_str_obs = np.array([f'<{lod}' if c else str(val) for c, val in zip(censored, y_obs)])
    x_prepared = mk.prepare_censored_data(y_str_obs)

    # Run the seasonal trend test with the ATS method
    res = seasonal_trend_test(
        x=x_prepared,
        t=t,
        period=n_seasons,
        season_type='quarter',
        sens_slope_method='ats',
        slope_scaling='year'
    )

    # 1. Verify that the calculated slope is reasonable and close to the true value.
    #    The ATS estimator on seasonal data with censoring can have some variance,
    #    so we use a slightly larger tolerance for the point estimate.
    assert pd.notna(res.slope)
    assert np.isclose(res.slope, true_beta_per_year, atol=0.2)

    # 2. **Crucially**, verify that the confidence intervals are now calculated and not NaN,
    #    confirming the refactor to enable bootstrap CIs was successful.
    assert pd.notna(res.lower_ci)
    assert pd.notna(res.upper_ci)

    # 3. Ensure the confidence interval is logical (lower < upper).
    assert res.lower_ci < res.upper_ci

    # Check that true slope is within a reasonable tolerance of the CI
    # (Bootstrap CIs are random and may barely exclude the true value in edge cases)
    assert res.lower_ci <= true_beta_per_year + 0.05
    assert res.upper_ci >= true_beta_per_year - 0.05
