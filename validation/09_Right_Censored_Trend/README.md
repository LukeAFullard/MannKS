# Validation Report


    # V-09: Right-Censored Trend

    This validation case tests the package's ability to handle right-censored data (values reported as greater than a detection limit, e.g., `>5.0`).

    Three scenarios were tested:
    1. **Strong Increasing Trend**: Data with a clear positive slope, where higher values are censored.
    2. **Weak Decreasing Trend**: Data with a slight negative slope, where initial high values are censored.
    3. **Stable (No Trend)**: Random data with no trend, with some high values censored.
    

## Plots
### v09_combined.png
![v09_combined.png](v09_combined.png)

### v09_strong_right_censored.png
![v09_strong_right_censored.png](v09_strong_right_censored.png)

## Results
| Test ID                | Method            |      Slope |     P-Value |   Lower CI |   Upper CI |
|:-----------------------|:------------------|-----------:|------------:|-----------:|-----------:|
| V-09_strong_increasing | MannKS (Standard) |  1.70942   | 0.000285164 |   1.37246  |   2.18715  |
| V-09_strong_increasing | MannKS (LWP Mode) |  1.84793   | 4.80273e-09 |   1.51366  |   2.1891   |
| V-09_strong_increasing | LWP-TRENDS (R)    |  1.84793   | 4.80273e-09 |   1.51366  |   2.1891   |
| V-09_strong_increasing | MannKS (ATS)      |  1.70942   | 0.000285164 |   1.37246  |   2.18715  |
| V-09_strong_increasing | NADA2 (R)         |  1.282     | 0.000285164 | nan        | nan        |
| V-09_weak_decreasing   | MannKS (Standard) | -0.0256599 | 0.717695    |  -0.669716 |   0.598758 |
| V-09_weak_decreasing   | MannKS (LWP Mode) |  0         | 0.956435    |  -0.485501 |   0.509951 |
| V-09_weak_decreasing   | LWP-TRENDS (R)    |  0         | 0.956435    |  -0.485501 |   0.509951 |
| V-09_weak_decreasing   | MannKS (ATS)      | -0.0256599 | 0.717695    |  -0.669716 |   0.598758 |
| V-09_weak_decreasing   | NADA2 (R)         | -0.124127  | 0.717695    | nan        | nan        |
| V-09_stable            | MannKS (Standard) | -0.0279316 | 0.595899    |  -0.332116 |   0.27474  |
| V-09_stable            | MannKS (LWP Mode) |  0         | 0.848148    |  -0.332156 |   0.182018 |
| V-09_stable            | LWP-TRENDS (R)    |  0         | 0.848148    |  -0.332156 |   0.182018 |
| V-09_stable            | MannKS (ATS)      | -0.0279316 | 0.595899    |  -0.332116 |   0.27474  |
| V-09_stable            | NADA2 (R)         |  0.0653666 | 0.595899    | nan        | nan        |

## LWP Accuracy (Python vs R)
| Test ID                |   Slope Error |   Slope % Error |
|:-----------------------|--------------:|----------------:|
| V-09_strong_increasing |   2.22045e-16 |     1.11022e-14 |
| V-09_weak_decreasing   |   0           |    -0           |
| V-09_stable            |   0           |     0           |
