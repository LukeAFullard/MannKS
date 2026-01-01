# V-34: Seasonal Sensitivity Sweep

Verifies metrics for seasonal data with mixed censoring.

Total Tests: 99
Class Matches: 98 (99.0%)
Max Slope Diff: 0.000014
Max P-value Diff: 0.005821
Max Lower CI Diff: 0.004464
Max Upper CI Diff: 0.005016

|   iter |   slope_in |   censor_pct |   slope_diff |      p_diff |    lci_diff |    uci_diff | match_class   | py_class                    | r_class                     |
|-------:|-----------:|-------------:|-------------:|------------:|------------:|------------:|:--------------|:----------------------------|:----------------------------|
|      1 |       0    |          0.2 |  0           | 0.00582103  | 0.000603346 | 0.0003318   | True          | as likely as not increasing | as likely as not increasing |
|      2 |       0    |          0.1 |  1.73472e-18 | 0.00132473  | 0.00149767  | 0.00023762  | True          | as likely as not decreasing | as likely as not decreasing |
|      3 |       0.15 |          0.1 |  0           | 6.24618e-17 | 0.000246093 | 1.56249e-05 | True          | highly likely increasing    | highly likely increasing    |
|      4 |       0.1  |          0.3 |  0           | 2.77556e-17 | 0           | 9.6989e-05  | True          | very likely increasing      | very likely increasing      |
|      5 |       0    |          0   |  6.93889e-18 | 0           | 5.55112e-17 | 0           | True          | likely decreasing           | likely decreasing           |
|      6 |       0.1  |          0.3 |  1.38778e-17 | 2.8386e-17  | 0.000300647 | 0.00015062  | True          | highly likely increasing    | highly likely increasing    |
|      7 |       0.2  |          0.1 |  5.55112e-17 | 1.38778e-16 | 0.00117506  | 1.2039e-05  | True          | highly likely increasing    | highly likely increasing    |
|      8 |       0    |          0.3 |  8.92809e-06 | 0.00192418  | 0.000635846 | 0.00501553  | True          | likely increasing           | likely increasing           |
|      9 |       0.15 |          0.2 |  1.38778e-17 | 9.71445e-17 | 0           | 0.00144849  | True          | very likely increasing      | very likely increasing      |
|     10 |       0.05 |          0.1 |  0           | 0           | 6.93889e-18 | 1.38778e-17 | True          | as likely as not decreasing | as likely as not decreasing |
|     11 |       0.1  |          0.4 |  0           | 2.01907e-17 | 6.30514e-05 | 9.56617e-05 | True          | highly likely increasing    | highly likely increasing    |
|     12 |       0.2  |          0.3 |  2.24778e-06 | 3.90313e-17 | 0.000356721 | 0.000242348 | True          | highly likely increasing    | highly likely increasing    |
|     13 |       0.05 |          0.4 |  2.77556e-17 | 7.63278e-17 | 0           | 0.00037273  | True          | highly likely increasing    | highly likely increasing    |
|     14 |       0    |          0.1 |  0           | 1.11022e-16 | 0.000637522 | 0.000117257 | True          | as likely as not increasing | as likely as not increasing |
|     15 |       0.2  |          0.3 |  0           | 1.00683e-16 | 4.34586e-05 | 0.00010605  | True          | highly likely increasing    | highly likely increasing    |
|     16 |       0    |          0.4 |  0           | 2.22045e-16 | 0.00138148  | 0.00188579  | True          | likely increasing           | likely increasing           |
|     17 |       0.2  |          0.1 |  0           | 9.66973e-17 | 0.000130595 | 0.000339543 | True          | highly likely increasing    | highly likely increasing    |
|     18 |       0.05 |          0.3 |  6.93889e-18 | 3.99504e-08 | 0.00022303  | 5.77995e-05 | True          | highly likely increasing    | highly likely increasing    |
|     19 |       0.1  |          0.4 |  1.44034e-05 | 1.11022e-16 | 0.0001301   | 0.000982886 | True          | likely increasing           | likely increasing           |
|     20 |       0.15 |          0.4 |  0           | 8.19657e-17 | 2.98289e-05 | 8.4592e-05  | True          | highly likely increasing    | highly likely increasing    |