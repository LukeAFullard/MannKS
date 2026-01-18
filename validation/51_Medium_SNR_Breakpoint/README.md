# Validation Report
## 1. Model Selection Accuracy (Correct N)
| Method | Accuracy | Mean Time (s) |
| :--- | :--- | :--- |
| Piecewise_Regression | 83.3% | 4.3994 |
| MannKS_Hybrid | 83.3% | 4.3579 |

## 2. Breakpoint Location Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 2.2408 | 1.8406 | 0.1109 | 6.8377 |
| MannKS_Hybrid | 2.2508 | 1.8527 | 0.1109 | 6.8377 |

## 3. Slope Estimation Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 0.1068 | 0.1092 | 0.0000 | 0.3806 |
| MannKS_Hybrid | 0.0110 | 0.0121 | 0.0004 | 0.0448 |

## 3. Confusion Matrix (True N vs Predicted N)

### Piecewise_Regression
| True N \ Pred N | 0 | 1 | 2 |
| :--- | --- | --- | --- |
| **0** | 8 | 0 | 0 |
| **1** | 0 | 11 | 1 |
| **2** | 0 | 4 | 6 |

### MannKS_Hybrid
| True N \ Pred N | 0 | 1 | 2 |
| :--- | --- | --- | --- |
| **0** | 8 | 0 | 0 |
| **1** | 0 | 11 | 1 |
| **2** | 0 | 4 | 6 |
