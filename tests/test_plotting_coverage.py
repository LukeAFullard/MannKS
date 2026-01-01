import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from collections import namedtuple

from MannKS.plotting import (
    plot_seasonal_distribution,
    plot_inspection_data,
    plot_trend
)
from MannKS.preprocessing import prepare_censored_data

@pytest.fixture
def temp_plot_path(tmpdir):
    """Fixture to create a temporary file path for plots."""
    return os.path.join(tmpdir, "test_plot.png")

def test_plot_seasonal_distribution_insufficient_data(capsys):
    """Test plot_seasonal_distribution with less than 2 data points."""
    result = plot_seasonal_distribution([1], [1])
    captured = capsys.readouterr()
    assert result is None
    assert "Not enough data to generate a plot" in captured.out

def test_plot_seasonal_distribution_numeric_seasons(temp_plot_path):
    """Test plot_seasonal_distribution with numeric season data."""
    x = np.random.rand(20)
    t = np.arange(20)
    result = plot_seasonal_distribution(x, t, period=4, plot_path=temp_plot_path)
    assert os.path.exists(result)

def test_plot_inspection_data_non_datetime(temp_plot_path):
    """Test plot_inspection_data with non-datetime time data."""
    data = pd.DataFrame({
        'value': [1, 2, 3],
        'time': [10, 20, 30],
        'censored': [False, False, False],
        'cen_type': ['not', 'not', 'not']
    })
    increment_map = {'yearly': 'year', 'monthly': 'month'}
    result = plot_inspection_data(data, temp_plot_path, 'value', 'time', 'monthly', increment_map)
    assert os.path.exists(result)

@patch('pandas.DataFrame.pivot_table')
def test_plot_inspection_data_matrix_exceptions(mock_pivot, temp_plot_path):
    """Test plot_inspection_data handles exceptions during matrix generation."""
    mock_pivot.side_effect = [
        pd.DataFrame(),
        Exception("Test Value Error"),
        Exception("Test Count Error")
    ]
    time_data = pd.to_datetime(['2022-01-01', '2022-02-01', '2023-01-01', '2023-02-01'])
    data = pd.DataFrame({
        'value': [1, 2, 3, 4],
        't': time_data,
        'censored': [False, False, False, False],
        'cen_type': ['not', 'not', 'not', 'not']
    })
    increment_map = {'yearly': 'year', 'monthly': 'month'}
    result_path = plot_inspection_data(data, temp_plot_path, 'value', 't', 'monthly', increment_map)
    assert os.path.exists(result_path)

def test_plot_trend_no_save_path():
    """Test plot_trend exits gracefully if save_path is None."""
    result = plot_trend(data=None, results=None, save_path=None, alpha=0.05)
    assert result is None

def test_plot_trend_with_numeric_time(temp_plot_path):
    """Test plot_trend with numeric time data."""
    TrendResult = namedtuple('TrendResult', ['trend', 'Tau', 'slope', 'intercept', 'lower_ci', 'upper_ci', 'p', 'slope_units'])
    results = TrendResult('increasing', 0.5, 1.0, 0.0, 0.5, 1.5, 0.01, 'units per year')
    data = pd.DataFrame({
        'value': [1, 2, 3, 4, 5],
        't': [1, 2, 3, 4, 5],
        'censored': [False] * 5
    })
    plot_trend(data, results, temp_plot_path, alpha=0.05)
    assert os.path.exists(temp_plot_path)
