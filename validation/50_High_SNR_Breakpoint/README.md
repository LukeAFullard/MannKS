# Validation Report
## 1. Model Selection Accuracy (Correct N)
| Method | Accuracy | Mean Time (s) |
| :--- | :--- | :--- |
| Piecewise_Regression | 83.3% | 3.6849 |
| MannKS_Hybrid | 83.3% | 3.5575 |

## 2. Breakpoint Location Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 1.1924 | 0.8328 | 0.0501 | 2.5670 |
| MannKS_Hybrid | 1.1799 | 0.8301 | 0.0501 | 2.5670 |

## 3. Slope Estimation Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 0.1785 | 0.1658 | 0.0000 | 0.4848 |
| MannKS_Hybrid | 0.0110 | 0.0107 | 0.0004 | 0.0374 |

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
| **1** | 0 | 10 | 2 |
| **2** | 0 | 3 | 7 |
