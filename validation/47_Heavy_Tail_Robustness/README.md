# Validation Report
## 1. Model Selection Accuracy (Correct N)
| Method | Accuracy | Mean Time (s) |
| :--- | :--- | :--- |
| Piecewise_Regression | 93.3% | 4.0554 |
| MannKS_Hybrid | 93.3% | 3.8888 |

## 2. Breakpoint Location Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 0.8761 | 0.6390 | 0.0781 | 2.1517 |
| MannKS_Hybrid | 0.8761 | 0.6390 | 0.0781 | 2.1517 |

## 3. Slope Estimation Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 0.2636 | 0.0244 | 0.2182 | 0.3191 |
| MannKS_Hybrid | 0.0174 | 0.0087 | 0.0037 | 0.0381 |

## 3. Confusion Matrix (True N vs Predicted N)

### Piecewise_Regression
| True N \ Pred N | 1 | 2 |
| :--- | --- | --- |
| **1** | 28 | 2 |

### MannKS_Hybrid
| True N \ Pred N | 1 | 2 |
| :--- | --- | --- |
| **1** | 28 | 2 |
