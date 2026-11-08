# V-33: Censored Sensitivity Sweep (Non-Seasonal)

Verifies Sen's slope, p-value, CIs, and classification for non-seasonal data with mixed censoring.

Total Tests: 99
Class Matches: 99 (100.0%)
Max Slope Diff: 0.000027
Max P-value Diff: 0.000000
Max Lower CI Diff: 0.000034
Max Upper CI Diff: 0.000014

## Detailed Results (Top 20)
|   iter |   slope_in |   noise_in |   censor_pct |   slope_diff |      p_diff |    lci_diff |    uci_diff | match_class   | py_class                    | r_class                     |
|-------:|-----------:|-----------:|-------------:|-------------:|------------:|------------:|------------:|:--------------|:----------------------------|:----------------------------|
|      1 |       0    |   0.1      |          0.6 |  0           | 0           | 0           | 0           | True          | as likely as not increasing | as likely as not increasing |
|      2 |       0    |   1.02627  |          0.6 |  0           | 0           | 0           | 0           | True          | as likely as not increasing | as likely as not increasing |
|      3 |      -0.05 |   0.1      |          0.2 |  0           | 1.21973e-19 | 5.29096e-07 | 4.66574e-06 | True          | highly likely decreasing    | highly likely decreasing    |
|      4 |       0.1  |   1.97577  |          0.4 |  0           | 0           | 0           | 1.21505e-06 | True          | likely increasing           | likely increasing           |
|      5 |       0    |   2.00886  |          0.1 |  6.93889e-18 | 1.11022e-16 | 5.55112e-17 | 9.15934e-16 | True          | likely decreasing           | likely decreasing           |
|      6 |       0.1  |   0.105364 |          0.4 |  0           | 1.41364e-26 | 2.4187e-06  | 1.35729e-06 | True          | highly likely increasing    | highly likely increasing    |
|      7 |      -0.1  |   1.00569  |          0.2 |  0           | 0           | 2.93812e-06 | 1.52536e-06 | True          | as likely as not decreasing | as likely as not decreasing |
|      8 |       0    |   1.04333  |          0.4 |  0           | 1.11022e-16 | 0           | 1.38778e-17 | True          | likely increasing           | likely increasing           |
|      9 |      -0.05 |   1.01374  |          0.3 |  0           | 0           | 2.88928e-06 | 3.60987e-06 | True          | as likely as not increasing | as likely as not increasing |
|     10 |       0.05 |   0.532122 |          0.2 |  0           | 5.55112e-17 | 4.68375e-17 | 1.38778e-17 | True          | likely increasing           | likely increasing           |
|     11 |       0.1  |   0.101259 |          0.6 |  3.65887e-08 | 8.27181e-25 | 1.38778e-17 | 0           | True          | highly likely increasing    | highly likely increasing    |
|     12 |      -0.1  |   0.500206 |          0.4 |  0           | 6.93889e-18 | 1.37949e-05 | 0           | True          | highly likely decreasing    | highly likely decreasing    |
|     13 |       0.05 |   1.97979  |          0.5 |  0           | 0           | 0           | 6.3145e-06  | True          | highly likely increasing    | highly likely increasing    |
|     14 |       0    |   0.549889 |          0.2 |  0           | 0           | 6.93889e-18 | 7.93074e-07 | True          | as likely as not increasing | as likely as not increasing |
|     15 |      -0.1  |   0.11531  |          0.4 |  5.55458e-07 | 5.493e-26   | 2.41453e-06 | 6.93889e-18 | True          | highly likely decreasing    | highly likely decreasing    |
|     16 |       0    |   1.01053  |          0.5 |  0           | 5.55112e-17 | 0           | 5.57189e-07 | True          | likely increasing           | likely increasing           |
|     17 |      -0.1  |   0.523831 |          0.2 |  5.73917e-07 | 0           | 2.30547e-06 | 1.5893e-06  | True          | likely increasing           | likely increasing           |
|     18 |       0.05 |   0.144574 |          0.6 |  3.46945e-18 | 2.57498e-19 | 0           | 2.08167e-17 | True          | highly likely increasing    | highly likely increasing    |
|     19 |       0.1  |   0.965414 |          0.5 |  0           | 5.55112e-17 | 0           | 1.67981e-06 | True          | likely increasing           | likely increasing           |
|     20 |      -0.05 |   0.471677 |          0.5 |  0           | 0           | 5.02169e-06 | 0           | True          | likely decreasing           | likely decreasing           |