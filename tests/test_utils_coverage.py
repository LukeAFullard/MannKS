"""
Tests for _utils.py to improve test coverage.
"""
import unittest
import pandas as pd
import numpy as np
from MannKS._datetime import (_is_datetime_like, _get_season_func,
                                 _get_cycle_identifier)
from MannKS._stats import (_rle_lengths, _z_score,
                             _mk_score_and_var_censored)
from MannKS._helpers import (_missing_values_analysis, _aggregate_censored_median)

class TestUtilsCoverage(unittest.TestCase):

    def test_get_season_func_invalid_season(self):
        """Test _get_season_func with an invalid season_type."""
        with self.assertRaises(ValueError):
            _get_season_func('invalid_season', 12)

    def test_get_season_func_invalid_period(self):
        """Test _get_season_func with an invalid period for a given season_type."""
        with self.assertRaises(ValueError):
            _get_season_func('month', 13)
        with self.assertRaises(ValueError):
            _get_season_func('week_of_year', 51)

    def test_rle_lengths_empty_input(self):
        """Test _rle_lengths with an empty array."""
        self.assertEqual(len(_rle_lengths(np.array([]))), 0)

    def test_get_cycle_identifier_edge_cases(self):
        """Test _get_cycle_identifier with less common season types."""
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=5, freq='D'))

        # Test day_of_week
        cycles_dow = _get_cycle_identifier(dates, 'day_of_week')
        self.assertEqual(len(cycles_dow), 5)

        # Test hour, minute, second
        dates_hourly = pd.to_datetime(pd.date_range(start='2023-01-01', periods=5, freq='h'))
        cycles_hour = _get_cycle_identifier(dates_hourly, 'hour')
        self.assertEqual(len(np.unique(cycles_hour)), 1) # All on the same day

    def test_mk_score_and_var_censored_no_variance(self):
        """Test _mk_score_and_var_censored with data that has no variance."""
        x = np.array([5, 5, 5])
        t = np.array([1, 2, 3])
        censored = np.array([False, False, False])
        cen_type = np.array(['not', 'not', 'not'])
        s, var_s, D, _ = _mk_score_and_var_censored(x, t, censored, cen_type)
        self.assertEqual(s, 0)
        self.assertGreaterEqual(var_s, 0)
        self.assertEqual(D, 0)

class TestAggregateCensoredMedian(unittest.TestCase):

    def test_empty_group(self):
        """Test that an empty group returns an empty DataFrame."""
        group = pd.DataFrame({'value': [], 'censored': [], 'cen_type': [], 't_original': [], 't': []})
        result = _aggregate_censored_median(group, is_datetime=False)
        self.assertTrue(result.empty)

    def test_nan_cen_type_mode(self):
        """Test that a group with only NaN cen_types for censored data is handled."""
        group = pd.DataFrame({
            'value': [1, 2, 3],
            'censored': [True, True, False],
            'cen_type': [np.nan, np.nan, 'not'],
            't_original': [1, 2, 3],
            't': [1, 2, 3]
        })
        # The median is 2, and the max censored value is 2, so `is_censored` would be true.
        # However, since the mode of cen_type for censored data is empty, it should
        # fall back to 'not' and `is_censored` should become False.
        result = _aggregate_censored_median(group, is_datetime=False)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result['value'].iloc[0], 2.0)
        self.assertFalse(result['censored'].iloc[0])
        self.assertEqual(result['cen_type'].iloc[0], 'not')

class TestNumericalStability(unittest.TestCase):

    def test_z_score_zero_variance(self):
        """Test that _z_score issues a warning for near-zero variance."""
        with self.assertWarnsRegex(UserWarning, "Variance near zero"):
            result = _z_score(s=10, var_s=1e-12)
            self.assertEqual(result, 0)

    def test_mk_score_and_var_censored_zero_denominator(self):
        """Test that _mk_score_and_var_censored warns for a near-zero denominator."""
        # Create a case with tied data that leads to D=0
        x = [1, 1, 1]
        t = [1, 2, 3]
        censored = [False, False, False]
        cen_type = ['not', 'not', 'not']
        with self.assertWarnsRegex(UserWarning, "Denominator near zero"):
            s, var_s, D, Tau = _mk_score_and_var_censored(x, t, censored, cen_type, tau_method='b')
            self.assertEqual(Tau, 0)


if __name__ == '__main__':
    unittest.main()
