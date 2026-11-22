# Validation Report
## 1. Model Selection Accuracy (Correct N)
| Method | Accuracy | Mean Time (s) |
| :--- | :--- | :--- |
| Piecewise_Regression | 96.7% | 3.6396 |
| MannKS_Hybrid | 96.7% | 3.3754 |

## 2. Breakpoint Location Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 1.2209 | 0.9503 | 0.1134 | 3.3974 |
| MannKS_Hybrid | 1.2209 | 0.9503 | 0.1134 | 3.3974 |

## 3. Slope Estimation Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 0.0338 | 0.0186 | 0.0038 | 0.0763 |
| MannKS_Hybrid | 0.0107 | 0.0055 | 0.0017 | 0.0210 |

## 3. Confusion Matrix (True N vs Predicted N)

### Piecewise_Regression
| True N \ Pred N | 1 | 2 |
| :--- | --- | --- |
| **1** | 29 | 1 |

### MannKS_Hybrid
| True N \ Pred N | 1 | 2 |
| :--- | --- | --- |
| **1** | 29 | 1 |
