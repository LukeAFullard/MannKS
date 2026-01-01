
import pandas as pd
import pytest
from MannKS.inspection import inspect_trend_data

@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    dates = pd.to_datetime(pd.date_range(start='2010-01-01', end='2019-12-31', freq='ME'))
    values = range(len(dates))
    return pd.DataFrame({'t': dates, 'value': values})

def test_inspect_trend_data_filtering(sample_data):
    """Test if the data is filtered correctly by trend period."""
    filtered_data = inspect_trend_data(sample_data, trend_period=5, end_year=2015)
    assert filtered_data['t'].dt.year.min() == 2011
    assert filtered_data['t'].dt.year.max() == 2015

def test_inspect_trend_data_increment_selection(sample_data):
    """Test if the correct time increment is selected."""
    # This data should be sufficient for monthly analysis
    inspected_data = inspect_trend_data(sample_data, prop_year_tol=0.8, prop_incr_tol=0.8)
    # The 'time_increment' column should contain the month number if 'monthly' is selected
    assert inspected_data['time_increment'].nunique() > 1
    assert (inspected_data['time_increment'] == inspected_data['t'].dt.month).all()


def test_inspect_trend_data_no_suitable_increment():
    """Test the case where no increment meets the criteria."""
    dates = pd.to_datetime(['2010-01-01', '2012-05-01', '2015-11-01'])
    values = [1, 2, 3]
    data = pd.DataFrame({'t': dates, 'value': values})
    inspected_data = inspect_trend_data(data)
    assert (inspected_data['time_increment'] == 'none').all()

def test_inspect_trend_data_empty_filter_return():
    """Test that an empty DataFrame is returned when the filter finds no data."""
    dates = pd.to_datetime(['2010-01-01', '2012-05-01', '2015-11-01'])
    values = [1, 2, 3]
    data = pd.DataFrame({'t': dates, 'value': values})
    inspected_data = inspect_trend_data(data, trend_period=1, end_year=2011)
    assert inspected_data.empty
    assert 'time_increment' in inspected_data.columns

def test_inspect_trend_data_return_summary(sample_data):
    """Test the return_summary functionality."""
    result = inspect_trend_data(sample_data, return_summary=True)
    assert isinstance(result.summary, pd.DataFrame)
    assert 'increment' in result.summary.columns
    assert 'data_ok' in result.summary.columns

def test_inspect_trend_data_error_handling():
    """Test error handling for invalid input."""
    with pytest.raises(TypeError):
        inspect_trend_data("not a dataframe")

    with pytest.raises(ValueError):
        inspect_trend_data(pd.DataFrame({'a': [1]}), time_col='t')

    with pytest.raises(ValueError):
        inspect_trend_data(pd.DataFrame({'t': [1]}), value_col='value')

def test_inspect_trend_data_custom_increments(sample_data):
    """Test the custom_increments functionality."""
    custom_increments = {'bi-annually': 2, 'annually': 1}
    result = inspect_trend_data(sample_data, custom_increments=custom_increments, return_summary=True)
    assert len(result.summary) == 2
    assert set(result.summary['increment']) == {'bi-annually', 'annually'}
    assert (result.data['time_increment'] == (result.data['t'].dt.month - 1) // 6 + 1).all()

def test_inspect_trend_data_weekly_increment():
    """Test the new 'weekly' increment."""
    dates = pd.to_datetime(pd.date_range(start='2018-01-01', end='2019-12-31', freq='W'))
    values = range(len(dates))
    data = pd.DataFrame({'t': dates, 'value': values})
    inspected_data = inspect_trend_data(data)
    assert (inspected_data['time_increment'] == inspected_data['t'].dt.isocalendar().week).all()
