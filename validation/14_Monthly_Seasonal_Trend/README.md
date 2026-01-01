# Validation Report


**V-14: Monthly Seasonal Trend**

This test verifies the seasonal trend analysis functionality on a simple monthly dataset.
It compares the standard `MannKS` seasonal test against the LWP-TRENDS R script and NADA2.

**Scenarios:**
1.  **Strong Increasing:** Clear positive trend with seasonality.
2.  **Weak Decreasing:** Subtle negative trend with seasonality.
3.  **Stable:** No underlying trend, just seasonality.


## Plots
### V14_Trend_Analysis.png
![V14_Trend_Analysis.png](V14_Trend_Analysis.png)

## Results
| Test ID                | Method            |      Slope |     P-Value |    Lower CI |     Upper CI |
|:-----------------------|:------------------|-----------:|------------:|------------:|-------------:|
| V-14_strong_increasing | MannKS (Standard) |  2.02086   | 0           |   1.97257   |   2.08004    |
| V-14_strong_increasing | MannKS (LWP Mode) |  2.02086   | 0           |   1.97259   |   2.07975    |
| V-14_strong_increasing | LWP-TRENDS (R)    |  2.02086   | 2.99e-41    |   1.97689   |   2.0608     |
| V-14_strong_increasing | MannKS (ATS)      |  2.0212    | 0           |   2.00513   |   2.0394     |
| V-14_strong_increasing | NADA2 (R)         |  1.952     | 0.002       | nan         | nan          |
| V-14_weak_decreasing   | MannKS (Standard) | -0.466957  | 1.28317e-10 |  -0.61406   |  -0.33305    |
| V-14_weak_decreasing   | MannKS (LWP Mode) | -0.466957  | 1.28317e-10 |  -0.613986  |  -0.333163   |
| V-14_weak_decreasing   | LWP-TRENDS (R)    | -0.466957  | 1.28317e-10 |  -0.604165  |  -0.354746   |
| V-14_weak_decreasing   | MannKS (ATS)      | -0.467963  | 1.28317e-10 |  -0.528184  |  -0.429177   |
| V-14_weak_decreasing   | NADA2 (R)         | -0.5837    | 0.002       | nan         | nan          |
| V-14_stable            | MannKS (Standard) | -0.0102414 | 0.737128    |  -0.0578728 |   0.0618986  |
| V-14_stable            | MannKS (LWP Mode) | -0.0102414 | 0.737128    |  -0.0578719 |   0.0617467  |
| V-14_stable            | LWP-TRENDS (R)    | -0.0102414 | 0.737128    |  -0.0505001 |   0.0493427  |
| V-14_stable            | MannKS (ATS)      | -0.0101594 | 0.737128    |  -0.0256705 |   0.00548697 |
| V-14_stable            | NADA2 (R)         | -0.09259   | 0.698       | nan         | nan          |

## LWP Accuracy (Python vs R)
| Test ID                |   Slope Error |   Slope % Error |
|:-----------------------|--------------:|----------------:|
| V-14_strong_increasing |   0           |     0           |
| V-14_weak_decreasing   |  -5.55112e-17 |     1.11022e-14 |
| V-14_stable            |  -1.73472e-18 |     1.69383e-14 |
