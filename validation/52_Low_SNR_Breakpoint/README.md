# Validation Report
## 1. Model Selection Accuracy (Correct N)
| Method | Accuracy | Mean Time (s) |
| :--- | :--- | :--- |
| Piecewise_Regression | 70.0% | 4.4362 |
| MannKS_Hybrid | 70.0% | 4.1886 |

## 2. Breakpoint Location Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 6.1514 | 5.5607 | 0.1353 | 22.4948 |
| MannKS_Hybrid | 6.1579 | 5.5596 | 0.1353 | 22.4948 |

## 3. Slope Estimation Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 0.0117 | 0.0134 | 0.0000 | 0.0612 |
| MannKS_Hybrid | 0.0109 | 0.0124 | 0.0004 | 0.0544 |

## 3. Confusion Matrix (True N vs Predicted N)

### Piecewise_Regression
| True N \ Pred N | 0 | 1 | 2 |
| :--- | --- | --- | --- |
| **0** | 12 | 0 | 0 |
| **1** | 4 | 18 | 1 |
| **2** | 0 | 10 | 5 |

### MannKS_Hybrid
| True N \ Pred N | 0 | 1 | 2 |
| :--- | --- | --- | --- |
| **0** | 12 | 0 | 0 |
| **1** | 5 | 18 | 0 |
| **2** | 0 | 10 | 5 |
