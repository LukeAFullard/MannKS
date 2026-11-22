# Validation Report
## 1. Model Selection Accuracy (Correct N)
| Method | Accuracy | Mean Time (s) |
| :--- | :--- | :--- |
| Piecewise_Regression | 86.7% | 3.2691 |
| MannKS_Hybrid | 90.0% | 0.9562 |

## 2. Breakpoint Location Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 0.5815 | 0.4177 | 0.0786 | 1.4807 |
| MannKS_Hybrid | 0.5631 | 0.4314 | 0.0846 | 1.5394 |

## 3. Slope Estimation Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 0.0161 | 0.0106 | 0.0017 | 0.0399 |
| MannKS_Hybrid | 0.0170 | 0.0098 | 0.0024 | 0.0408 |

## 3. Confusion Matrix (True N vs Predicted N)

### Piecewise_Regression
| True N \ Pred N | 1 | 2 |
| :--- | --- | --- |
| **1** | 26 | 4 |

### MannKS_Hybrid
| True N \ Pred N | 1 | 2 |
| :--- | --- | --- |
| **1** | 27 | 3 |
