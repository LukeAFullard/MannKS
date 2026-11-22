# Validation Report
## 1. Model Selection Accuracy (Correct N)
| Method | Accuracy | Mean Time (s) |
| :--- | :--- | :--- |
| Piecewise_Regression | 83.3% | 6.5766 |
| MannKS_Hybrid | 96.7% | 2.2003 |

## 2. Breakpoint Location Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 1.1924 | 0.8328 | 0.0501 | 2.5670 |
| MannKS_Hybrid | 1.1516 | 0.7624 | 0.0501 | 2.6635 |

## 3. Slope Estimation Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 0.0114 | 0.0115 | 0.0000 | 0.0461 |
| MannKS_Hybrid | 0.0123 | 0.0127 | 0.0004 | 0.0451 |

## 3. Confusion Matrix (True N vs Predicted N)

### Piecewise_Regression
| True N \ Pred N | 0 | 1 | 2 |
| :--- | --- | --- | --- |
| **0** | 8 | 0 | 0 |
| **1** | 0 | 10 | 2 |
| **2** | 0 | 3 | 7 |

### MannKS_Hybrid
| True N \ Pred N | 0 | 1 | 2 |
| :--- | --- | --- | --- |
| **0** | 8 | 0 | 0 |
| **1** | 0 | 12 | 0 |
| **2** | 0 | 1 | 9 |
