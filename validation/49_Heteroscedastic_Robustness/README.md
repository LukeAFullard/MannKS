# Validation Report
## 1. Model Selection Accuracy (Correct N)
| Method | Accuracy | Mean Time (s) |
| :--- | :--- | :--- |
| Piecewise_Regression | 86.7% | 3.8371 |
| MannKS_Hybrid | 86.7% | 3.9388 |

## 2. Breakpoint Location Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 0.5815 | 0.4177 | 0.0786 | 1.4807 |
| MannKS_Hybrid | 0.5815 | 0.4177 | 0.0786 | 1.4807 |

## 3. Slope Estimation Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 0.2650 | 0.0220 | 0.2406 | 0.3220 |
| MannKS_Hybrid | 0.0169 | 0.0098 | 0.0024 | 0.0408 |

## 3. Confusion Matrix (True N vs Predicted N)

### Piecewise_Regression
| True N \ Pred N | 1 | 2 |
| :--- | --- | --- |
| **1** | 26 | 4 |

### MannKS_Hybrid
| True N \ Pred N | 1 | 2 |
| :--- | --- | --- |
| **1** | 26 | 4 |
