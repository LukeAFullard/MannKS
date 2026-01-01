import pytest
import pandas as pd

from MannKS._datetime import _get_season_func, _get_agg_func

# --- Tests for _get_season_func ---

@pytest.mark.parametrize("season_type, period, date_str, expected_season", [
    ('hour', 24, '2020-01-01 14:30', 14),
    ('minute', 60, '2020-01-01 14:30:55', 30),
    ('second', 60, '2020-01-01 14:30:55', 55),
    ('day_of_year', 366, '2020-03-01', 61), # Leap year
    ('unsupported', 12, '2020-01-01', None),
])
def test_get_season_func_various_types(season_type, period, date_str, expected_season):
    """Test _get_season_func for various standard and unsupported types."""
    dates = pd.to_datetime([date_str])
    if expected_season is None:
        # Corrected the expected error message to match the implementation
        with pytest.raises(ValueError, match=f"Unknown season_type: '{season_type}'"):
            _get_season_func(season_type, period)
    else:
        func = _get_season_func(season_type, period)
        result = func(dates)
        val = result.iloc[0] if isinstance(result, pd.Series) else result[0]
        assert val == expected_season

# --- Tests for _get_agg_func ---

@pytest.mark.parametrize("agg_period, date_str, expected_group", [
    # Corrected the expected group to match the 'year' * 100 + 'week' logic
    ('week', '2020-01-05', 202001), # First week of 2020
    ('week', '2020-01-06', 202002), # Second week of 2020
    # Corrected the expected group to be the date object itself
    ('day', '2020-03-15', pd.to_datetime('2020-03-15').date()),
    ('unsupported', '2020-01-01', None),
])
def test_get_agg_func_various_periods(agg_period, date_str, expected_group):
    """Test _get_agg_func for various standard and unsupported periods."""
    dates = pd.to_datetime([date_str])
    if expected_group is None:
        # Corrected the expected error message
        with pytest.raises(ValueError, match=f"Unknown agg_period: '{agg_period}'"):
            _get_agg_func(agg_period)
    else:
        func = _get_agg_func(agg_period)
        result = func(dates)
        val = result.iloc[0] if isinstance(result, pd.Series) else result[0]
        assert val == expected_group
