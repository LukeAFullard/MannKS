# Validation Report
## 1. Model Selection Accuracy (Correct N)
| Method | Accuracy | Mean Time (s) |
| :--- | :--- | :--- |
| Piecewise_Regression | 70.0% | 1.8215 |
| MannKS_Hybrid | 68.0% | 1.8236 |

## 2. Breakpoint Location Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 5.9180 | 5.3729 | 0.1353 | 22.4948 |
| MannKS_Hybrid | 6.1866 | 5.6887 | 0.1353 | 22.4948 |

## 3. Slope Estimation Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 0.1039 | 0.1002 | 0.0000 | 0.3325 |
| MannKS_Hybrid | 0.0110 | 0.0126 | 0.0004 | 0.0544 |

## 3. Confusion Matrix (True N vs Predicted N)

### Piecewise_Regression
| True N \ Pred N | 0 | 1 | 2 |
| :--- | --- | --- | --- |
| **0** | 12 | 0 | 0 |
| **1** | 3 | 18 | 2 |
| **2** | 0 | 10 | 5 |

### MannKS_Hybrid
| True N \ Pred N | 0 | 1 | 2 |
| :--- | --- | --- | --- |
| **0** | 12 | 0 | 0 |
| **1** | 3 | 17 | 3 |
| **2** | 0 | 10 | 5 |
