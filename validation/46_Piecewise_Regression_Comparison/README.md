# Validation Report
## 1. Model Selection Accuracy (Correct N)
| Method | Accuracy | Mean Time (s) |
| :--- | :--- | :--- |
| Piecewise_Regression | 90.0% | 3.7422 |
| MannKS_Hybrid | 90.0% | 3.4944 |

## 2. Breakpoint Location Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 1.1464 | 0.8199 | 0.1974 | 2.5670 |
| MannKS_Hybrid | 1.1301 | 0.8152 | 0.1974 | 2.5670 |

## 3. Slope Estimation Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | N/A | N/A | N/A | N/A |
| MannKS_Hybrid | N/A | N/A | N/A | N/A |

## 3. Confusion Matrix (True N vs Predicted N)

### Piecewise_Regression
| True N \ Pred N | 0 | 1 | 2 |
| :--- | --- | --- | --- |
| **0** | 5 | 0 | 0 |
| **1** | 0 | 6 | 0 |
| **2** | 0 | 2 | 7 |

### MannKS_Hybrid
| True N \ Pred N | 0 | 1 | 2 |
| :--- | --- | --- | --- |
| **0** | 5 | 0 | 0 |
| **1** | 0 | 6 | 0 |
| **2** | 0 | 2 | 7 |
