# Validation Report
## 1. Model Selection Accuracy (Correct N)
| Method | Accuracy | Mean Time (s) |
| :--- | :--- | :--- |
| Piecewise_Regression | 93.3% | 4.0179 |
| MannKS_Hybrid | 90.0% | 1.3370 |

## 2. Breakpoint Location Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 0.8761 | 0.6390 | 0.0781 | 2.1517 |
| MannKS_Hybrid | 0.8438 | 0.6377 | 0.0781 | 2.1517 |

## 3. Slope Estimation Accuracy (MAE)
| Method | Mean Error | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Piecewise_Regression | 0.0201 | 0.0110 | 0.0025 | 0.0416 |
| MannKS_Hybrid | 0.0166 | 0.0087 | 0.0037 | 0.0381 |

## 3. Confusion Matrix (True N vs Predicted N)

### Piecewise_Regression
| True N \ Pred N | 1 | 2 |
| :--- | --- | --- |
| **1** | 28 | 2 |

### MannKS_Hybrid
| True N \ Pred N | 1 | 2 |
| :--- | --- | --- |
| **1** | 27 | 3 |
