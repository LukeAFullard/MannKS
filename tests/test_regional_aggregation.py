"""
Tests for the regional_test function.
"""
import unittest
import pandas as pd
import numpy as np
from MannKS import trend_test, regional_test

class TestRegionalAggregation(unittest.TestCase):
    def test_regional_test_basic(self):
        """
        Test the regional_test function with a synthetic dataset.
        """
        # --- 1. Create Synthetic Data ---
        # Three sites: A is increasing, B is decreasing, C is increasing
        dates = pd.to_datetime(pd.date_range(start='2000-01-01', periods=20, freq='YE'))
        sites = ['A', 'B', 'C']
        all_ts_data = []

        for site in sites:
            np.random.seed(hash(site) % (2**32 - 1))
            noise = np.random.normal(0, 0.1, 20) # Reduced noise
            if site == 'B':
                trend = -0.5 * np.arange(20) # Increased trend signal
            else:
                trend = 0.5 * np.arange(20) # Increased trend signal

            values = 10 + trend + noise

            df = pd.DataFrame({
                'time': dates,
                'value': values,
                'site': site
            })
            all_ts_data.append(df)

        time_series_data = pd.concat(all_ts_data, ignore_index=True)

        # --- 2. Run Single-Site Trend Analysis ---
        trend_results = []
        for site in sites:
            site_data = time_series_data[time_series_data['site'] == site]
            res = trend_test(x=site_data['value'], t=site_data['time'], min_size=None)

            # Assert that the new fields are present
            self.assertTrue(hasattr(res, 'classification'))
            self.assertTrue(hasattr(res, 'analysis_notes'))
            self.assertIsInstance(res.analysis_notes, list)


            # Convert namedtuple to a dictionary and add site
            res_dict = res._asdict()
            res_dict['site'] = site
            trend_results.append(res_dict)

        trend_results_df = pd.DataFrame(trend_results)

        # --- 3. Run Regional Trend Aggregation ---
        regional_res = regional_test(trend_results=trend_results_df,
                                     time_series_data=time_series_data,
                                     site_col='site',
                                     value_col='value',
                                     time_col='time',
                                     s_col='s',
                                     c_col='C')

        # --- 4. Assertions ---
        self.assertIsInstance(regional_res, tuple)
        self.assertEqual(regional_res.M, 3)

        # Modal direction should be 'Increasing' (2 of 3 sites are increasing)
        self.assertEqual(regional_res.DT, 'Increasing')

        # TAU should be 2/3
        self.assertAlmostEqual(regional_res.TAU, 2/3, places=5)

        # Check that variances and confidence are plausible floats
        self.assertIsInstance(regional_res.VarTAU, float)
        self.assertIsInstance(regional_res.CorrectedVarTAU, float)
        self.assertIsInstance(regional_res.CT, float)

        # Corrected variance should be less than or equal to the uncorrected
        self.assertLessEqual(regional_res.CorrectedVarTAU, regional_res.VarTAU)

    def test_regional_test_input_validation(self):
        """Test the input validation in regional_test."""
        # Create dummy data
        trend_results = pd.DataFrame({'site': ['A'], 's': [1], 'C': [0.9]})
        time_series_data = pd.DataFrame({'site': ['A'], 'value': [1], 'time': [pd.to_datetime('2020-01-01')]})

        # Test missing columns in trend_results
        with self.assertRaises(ValueError):
            regional_test(trend_results.drop(columns=['s']), time_series_data)

        # Test missing columns in time_series_data
        with self.assertRaises(ValueError):
            regional_test(trend_results, time_series_data.drop(columns=['value']))

    def test_regional_test_insufficient_data(self):
        """Test regional_test with no valid data."""
        trend_results = pd.DataFrame({'site': ['A'], 's': [np.nan], 'C': [np.nan]})
        time_series_data = pd.DataFrame({'site': ['A'], 'value': [1], 'time': [pd.to_datetime('2020-01-01')]})

        result = regional_test(trend_results, time_series_data)
        self.assertEqual(result.M, 0)
        self.assertEqual(result.DT, 'Insufficient Data')

    def test_regional_test_tied_direction(self):
        """Test regional_test with a tied modal direction."""
        trend_results = pd.DataFrame({
            'site': ['A', 'B'],
            's': [1, -1],  # One increasing, one decreasing
            'C': [0.9, 0.9]
        })
        time_series_data = pd.DataFrame({
            'site': ['A', 'B'],
            'value': [1, 1],
            'time': [pd.to_datetime('2020-01-01'), pd.to_datetime('2020-01-01')]
        })
        result = regional_test(trend_results, time_series_data)
        self.assertEqual(result.DT, 'No Clear Direction')
        self.assertEqual(result.TAU, 0.5)
        self.assertTrue(np.isnan(result.CT))


if __name__ == '__main__':
    unittest.main()
