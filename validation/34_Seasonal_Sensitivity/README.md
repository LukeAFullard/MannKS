# V-34: Seasonal Sensitivity Sweep

Verifies metrics for seasonal data with mixed censoring.

Total Tests: 99
Class Matches: 99 (100.0%)
Max Slope Diff: 0.000014
Max P-value Diff: 0.000000
Max Lower CI Diff: 0.000015
Max Upper CI Diff: 0.000014

|   iter |   slope_in |   censor_pct |   slope_diff |      p_diff |    lci_diff |    uci_diff | match_class   | py_class                    | r_class                     |
|-------:|-----------:|-------------:|-------------:|------------:|------------:|------------:|:--------------|:----------------------------|:----------------------------|
|      1 |       0    |          0.2 |  0           | 1.11022e-16 | 5.42101e-20 | 8.67362e-19 | True          | as likely as not increasing | as likely as not increasing |
|      2 |       0    |          0.1 |  1.73472e-18 | 0           | 2.77556e-17 | 0           | True          | as likely as not decreasing | as likely as not decreasing |
|      3 |       0.15 |          0.1 |  0           | 6.56332e-28 | 6.93889e-17 | 2.77556e-17 | True          | highly likely increasing    | highly likely increasing    |
|      4 |       0.1  |          0.3 |  0           | 5.55112e-17 | 0           | 0           | True          | very likely increasing      | very likely increasing      |
|      5 |       0    |          0   |  6.93889e-18 | 0           | 5.55112e-17 | 0           | True          | likely decreasing           | likely decreasing           |
|      6 |       0.1  |          0.3 |  1.38778e-17 | 1.98523e-23 | 2.08167e-17 | 0           | True          | highly likely increasing    | highly likely increasing    |
|      7 |       0.2  |          0.1 |  5.55112e-17 | 6.93889e-17 | 2.91434e-16 | 0           | True          | highly likely increasing    | highly likely increasing    |
|      8 |       0    |          0.3 |  8.92809e-06 | 5.55112e-17 | 2.76721e-06 | 0           | True          | likely increasing           | likely increasing           |
|      9 |       0.15 |          0.2 |  1.38778e-17 | 6.93889e-17 | 0           | 1.67514e-06 | True          | very likely increasing      | very likely increasing      |
|     10 |       0.05 |          0.1 |  0           | 0           | 6.93889e-18 | 1.38778e-17 | True          | as likely as not decreasing | as likely as not decreasing |
|     11 |       0.1  |          0.4 |  0           | 5.79026e-24 | 1.38778e-17 | 2.77556e-17 | True          | highly likely increasing    | highly likely increasing    |
|     12 |       0.2  |          0.3 |  2.24778e-06 | 3.46945e-18 | 8.67362e-17 | 2.77556e-17 | True          | highly likely increasing    | highly likely increasing    |
|     13 |       0.05 |          0.4 |  2.77556e-17 | 2.08167e-17 | 0           | 5.55112e-17 | True          | highly likely increasing    | highly likely increasing    |
|     14 |       0    |          0.1 |  0           | 0           | 1.52656e-16 | 0           | True          | as likely as not increasing | as likely as not increasing |
|     15 |       0.2  |          0.3 |  0           | 2.52435e-28 | 2.77556e-17 | 0           | True          | highly likely increasing    | highly likely increasing    |
|     16 |       0    |          0.4 |  0           | 2.22045e-16 | 1.80411e-16 | 4.44089e-16 | True          | likely increasing           | likely increasing           |
|     17 |       0.2  |          0.1 |  0           | 1.35525e-20 | 2.77556e-17 | 0           | True          | highly likely increasing    | highly likely increasing    |
|     18 |       0.05 |          0.3 |  6.93889e-18 | 1.16467e-21 | 0           | 0           | True          | highly likely increasing    | highly likely increasing    |
|     19 |       0.1  |          0.4 |  1.44034e-05 | 0           | 4.33862e-06 | 2.87391e-06 | True          | likely increasing           | likely increasing           |
|     20 |       0.15 |          0.4 |  0           | 3.46945e-18 | 6.93889e-18 | 1.28183e-06 | True          | highly likely increasing    | highly likely increasing    |