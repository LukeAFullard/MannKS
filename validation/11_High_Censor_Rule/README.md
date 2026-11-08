# V-11: High Censor Rule

## Objective
Verify the implementation of the 'High Censor Rule' (`hicensor=True`). This rule, used in older LWP-TRENDS versions, sets all values below the highest detection limit to be censored at that highest limit. This prevents spurious trends caused solely by changing detection limits (e.g., detection limit improves from <5 to <1 over time).

## Plots
### v11_combined.png
![v11_combined.png](v11_combined.png)

## Results
| Test ID                | Method            |      Slope |     P-Value |    Lower CI |    Upper CI |
|:-----------------------|:------------------|-----------:|------------:|------------:|------------:|
| V-11_strong_increasing | MannKS (Standard) |  0.83495   | 1.07283e-22 |   0.75549   |   0.916887  |
| V-11_strong_increasing | MannKS (LWP Mode) |  0.791823  | 3.49718e-24 |   0.713381  |   0.862459  |
| V-11_strong_increasing | LWP-TRENDS (R)    |  0.791823  | 3.49718e-24 |   0.713381  |   0.862459  |
| V-11_strong_increasing | MannKS (ATS)      |  0.83495   | 1.07283e-22 |   0.75549   |   0.916887  |
| V-11_strong_increasing | NADA2 (R)         |  0.835051  | 0           | nan         | nan         |
| V-11_weak_decreasing   | MannKS (Standard) | -0.72042   | 1.30328e-16 |  -0.806404  |  -0.633845  |
| V-11_weak_decreasing   | MannKS (LWP Mode) | -0.699787  | 1.87109e-18 |  -0.78224   |  -0.570792  |
| V-11_weak_decreasing   | LWP-TRENDS (R)    | -0.699787  | 1.87109e-18 |  -0.78224   |  -0.570792  |
| V-11_weak_decreasing   | MannKS (ATS)      | -0.72042   | 1.30328e-16 |  -0.806404  |  -0.633845  |
| V-11_weak_decreasing   | NADA2 (R)         | -0.720507  | 2.22045e-16 | nan         | nan         |
| V-11_stable            | MannKS (Standard) |  0.0903797 | 0.147663    |  -0.0287511 |   0.199649  |
| V-11_stable            | MannKS (LWP Mode) |  0         | 0.155882    |   0         |   0.0535881 |
| V-11_stable            | LWP-TRENDS (R)    |  0         | 0.155882    |   0         |   0.0535881 |
| V-11_stable            | MannKS (ATS)      |  0.0903797 | 0.147663    |  -0.0287511 |   0.199649  |
| V-11_stable            | NADA2 (R)         |  0.0904053 | 0.147663    | nan         | nan         |

## LWP Accuracy (Python vs R)
| Test ID                |   Slope Error |   Slope % Error |
|:-----------------------|--------------:|----------------:|
| V-11_strong_increasing |   1.11022e-16 |     1.40211e-14 |
| V-11_weak_decreasing   |   0           |    -0           |
| V-11_stable            |   0           |     0           |
