# Validation Report
## 1. Model Selection Accuracy (Correct N)
| Method | Accuracy | Mean Time (s) |
| :--- | :--- | :--- |
| Piecewise_Regression | 96.7% | 4.1920 |
| MannKS_Hybrid | 93.3% | 4.0635 |

## 2. Breakpoint Location Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 1.2690 | 0.9547 | 0.1134 | 3.3974 |
| MannKS_Hybrid | 1.2419 | 0.9608 | 0.1134 | 3.3974 |

## 3. Slope Estimation Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 0.2765 | 0.0384 | 0.2198 | 0.3817 |
| MannKS_Hybrid | 0.0108 | 0.0056 | 0.0017 | 0.0210 |

## 3. Confusion Matrix (True N vs Predicted N)

### Piecewise_Regression
| True N \ Pred N | 1 | 2 |
| :--- | --- | --- |
| **1** | 29 | 1 |

### MannKS_Hybrid
| True N \ Pred N | 1 | 2 |
| :--- | --- | --- |
| **1** | 28 | 2 |
