
import pandas as pd
import pytest
import os
from MannKS.inspection import inspect_trend_data
from MannKS.plotting import plot_inspection_data
from MannKS.preprocessing import prepare_censored_data


@pytest.fixture
def censored_data():
    """Create a sample DataFrame with censored data for plotting tests."""
    dates = pd.to_datetime(pd.date_range(start='2015-01-01', end='2019-12-31', freq='QE'))
    values = [f'<{i}' if i % 3 == 0 else i for i in range(1, len(dates) + 1)]
    df = pd.DataFrame({'t': dates, 'value_str': values})
    prepared_df = prepare_censored_data(df['value_str'])
    return df[['t']].join(prepared_df)


def test_inspection_plot_creation_via_inspect_trend_data(censored_data, tmp_path):
    """Test that the inspection plot is created successfully via the main function."""
    plot_path = tmp_path / "inspection_plot.png"

    # Call the main function which should, in turn, call the plotting function
    inspect_trend_data(
        censored_data,
        plot=True,
        plot_path=str(plot_path),
        trend_period=5,
        end_year=2019
    )

    assert os.path.exists(plot_path)
    assert os.path.getsize(plot_path) > 0


def test_inspection_plot_generalization(tmp_path):
    """Test the plotting function's ability to handle different time increments."""
    monthly_dates = pd.to_datetime(pd.date_range(start='2018-01-01', end='2019-12-31', freq='ME'))
    monthly_values = [f'<{i}' if i % 4 == 0 else i for i in range(len(monthly_dates))]
    monthly_df_raw = pd.DataFrame({'t': monthly_dates, 'value_str': monthly_values})
    prepared_df = prepare_censored_data(monthly_df_raw['value_str'])
    monthly_df = monthly_df_raw[['t']].join(prepared_df)

    plot_path_m = tmp_path / "inspection_plot_monthly.png"
    inspect_trend_data(
        monthly_df,
        plot=True,
        plot_path=str(plot_path_m)
    )
    assert os.path.exists(plot_path_m)


def test_inspection_plot_error_handling(censored_data):
    """Test error conditions for the plotting feature."""
    # Error when plot=True but plot_path is not provided in the main function
    with pytest.raises(ValueError, match="A 'plot_path' must be provided"):
        inspect_trend_data(censored_data, plot=True, plot_path=None)

    # Error when data is not prepared (missing 'censored' column)
    raw_data = pd.DataFrame({
        't': pd.to_datetime(['2020-01-01']),
        'value': [10]
    })
    # This error should now come from the plotting function
    with pytest.raises(ValueError, match="Please run `prepare_censored_data` first"):
        inspect_trend_data(raw_data, plot=True, plot_path="test.png")


def test_inspection_plot_cycle_generalization(tmp_path):
    """Test that matrix plots correctly use a non-year cycle."""
    # Create weekly data where the cycle should be the week number
    dates = pd.to_datetime(pd.date_range(start='2022-01-01', end='2022-12-31', freq='W-MON'))
    values = range(len(dates))
    data = pd.DataFrame({'t': dates, 'value': values})
    # Add dummy censored columns so it passes validation
    data['censored'] = False
    data['cen_type'] = 'not'

    plot_path = tmp_path / "weekly_inspection_plot.png"

    # Running inspect_trend_data should select 'weekly' as the best increment
    inspect_trend_data(
        data,
        plot=True,
        plot_path=str(plot_path)
    )

    # The existence of the plot confirms the logic didn't crash with a non-year cycle
    assert os.path.exists(plot_path)
    assert os.path.getsize(plot_path) > 0
