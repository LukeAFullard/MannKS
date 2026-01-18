# Validation Report
## 1. Model Selection Accuracy (Correct N)
| Method | Accuracy | Mean Time (s) |
| :--- | :--- | :--- |
| Piecewise_Regression | 71.0% | 4.3081 |
| MannKS_BIC | 71.0% | 1.1108 |
| MannKS_Hybrid | 67.7% | 4.2503 |

## 2. Breakpoint Location Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 6.0436 | 4.6929 | 0.5877 | 17.4792 |
| MannKS_BIC | 8.2531 | 6.3790 | 1.1458 | 19.7388 |
| MannKS_Hybrid | 6.4958 | 5.2928 | 0.5877 | 17.4792 |

## 3. Slope Estimation Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 0.0988 | 0.1092 | 0.0000 | 0.3325 |
| MannKS_BIC | 0.0330 | 0.1119 | 0.0004 | 0.5322 |
| MannKS_Hybrid | 0.0092 | 0.0104 | 0.0004 | 0.0422 |

## 3. Confusion Matrix (True N vs Predicted N)

### Piecewise_Regression
| True N \ Pred N | 0 | 1 | 2 |
| :--- | --- | --- | --- |
| **0** | 9 | 0 | 0 |
| **1** | 2 | 9 | 1 |
| **2** | 0 | 6 | 4 |

### MannKS_BIC
| True N \ Pred N | 0 | 1 | 2 |
| :--- | --- | --- | --- |
| **0** | 9 | 0 | 0 |
| **1** | 2 | 9 | 1 |
| **2** | 0 | 6 | 4 |

### MannKS_Hybrid
| True N \ Pred N | 0 | 1 | 2 |
| :--- | --- | --- | --- |
| **0** | 9 | 0 | 0 |
| **1** | 2 | 8 | 2 |
| **2** | 0 | 6 | 4 |
