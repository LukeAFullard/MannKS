# V-32: Classification Sensitivity Sweep

This validation runs a parameter sweep of slope and noise to verify that `MannKS` classification logic matches the LWP R script across the full range of confidence levels.

## Summary

Total Tests: 99
Matches: 99 (100.0%)

## Mismatches

No mismatches found.

## Full Results (Top 20)

|   slope_in |   noise_in |      py_p |       r_p |     py_C |      r_C | py_class                    | r_class                     | match_class   | match_C   |
|-----------:|-----------:|----------:|----------:|---------:|---------:|:----------------------------|:----------------------------|:--------------|:----------|
|       0    |        0.5 | 0.632405  | 0.632405  | 0.683798 | 0.683798 | Likely Increasing           | Likely Increasing           | True          | True      |
|       0    |        0.5 | 0.933921  | 0.933921  | 0.53304  | 0.53304  | As Likely as Not Decreasing | As Likely As Not Decreasing | True          | True      |
|       0    |        0.5 | 0.494962  | 0.494962  | 0.752519 | 0.752519 | Likely Increasing           | Likely Increasing           | True          | True      |
|       0    |        1   | 0.112263  | 0.112263  | 0.943869 | 0.943869 | Very Likely Decreasing      | Very Likely Decreasing      | True          | True      |
|       0    |        1   | 0.31054   | 0.31054   | 0.84473  | 0.84473  | Likely Decreasing           | Likely Decreasing           | True          | True      |
|       0    |        1   | 0.803563  | 0.803563  | 0.598219 | 0.598219 | As Likely as Not Increasing | As Likely As Not Increasing | True          | True      |
|       0    |        2   | 0.764358  | 0.764358  | 0.617821 | 0.617821 | As Likely as Not Decreasing | As Likely As Not Decreasing | True          | True      |
|       0    |        2   | 0.873316  | 0.873316  | 0.563342 | 0.563342 | As Likely as Not Increasing | As Likely As Not Increasing | True          | True      |
|       0    |        2   | 0.286824  | 0.286824  | 0.856588 | 0.856588 | Likely Increasing           | Likely Increasing           | True          | True      |
|       0.02 |        0.5 | 0.0260211 | 0.0260211 | 0.986989 | 0.986989 | Highly Likely Increasing    | Highly Likely Increasing    | True          | True      |
|       0.02 |        0.5 | 0.199855  | 0.199855  | 0.900072 | 0.900072 | Very Likely Increasing      | Very Likely Increasing      | True          | True      |
|       0.02 |        0.5 | 0.292635  | 0.292635  | 0.853682 | 0.853682 | Likely Increasing           | Likely Increasing           | True          | True      |
|       0.02 |        1   | 0.286824  | 0.286824  | 0.856588 | 0.856588 | Likely Decreasing           | Likely Decreasing           | True          | True      |
|       0.02 |        1   | 0.463277  | 0.463277  | 0.768362 | 0.768362 | Likely Increasing           | Likely Increasing           | True          | True      |
|       0.02 |        1   | 0.553083  | 0.553083  | 0.723459 | 0.723459 | Likely Decreasing           | Likely Decreasing           | True          | True      |
|       0.02 |        2   | 0.570282  | 0.570282  | 0.714859 | 0.714859 | Likely Increasing           | Likely Increasing           | True          | True      |
|       0.02 |        2   | 0.486934  | 0.486934  | 0.756533 | 0.756533 | Likely Increasing           | Likely Increasing           | True          | True      |
|       0.02 |        2   | 0.410649  | 0.410649  | 0.794676 | 0.794676 | Likely Decreasing           | Likely Decreasing           | True          | True      |
|       0.05 |        0.5 | 0.59655   | 0.59655   | 0.701725 | 0.701725 | Likely Decreasing           | Likely Decreasing           | True          | True      |
|       0.05 |        0.5 | 0.903548  | 0.903548  | 0.548226 | 0.548226 | As Likely as Not Increasing | As Likely As Not Increasing | True          | True      |