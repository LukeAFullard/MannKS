# V-25: Regional Trend with High Inter-site Correlation

This test verifies that the Regional Kendall Test correctly calculates the variance inflation factor when sites are highly correlated.

In the 'High Correlation' scenario, sites share the same underlying noise pattern, which should result in a much higher `CorrectedVarTAU` compared to the uncorrected `VarTAU`. In the 'Uncorrelated' scenario, the corrected and uncorrected variances should be similar.

## Results Comparison

| Scenario               | Metric           | Python (MKS)   | R (LWP)    | Match?   |
|------------------------|------------------|----------------|------------|----------|
| High Correlation       | Sites (M)        | 4              | 4          | ✅        |
|                        | TAU              | 1.0000         | 1.0000     | ✅        |
|                        | VarTAU (Uncorr)  | 0.0000         | 0.0000     | ✅        |
|                        | Corrected VarTAU | 0.0000         | 0.0000     | ✅        |
|                        | Direction (DT)   | Increasing     | Increasing | ✅        |
|                        | Confidence (CT)  | 1.0000         | 1.0000     | ✅        |
| ---                    | ---              | ---            | ---        | ---      |
| Uncorrelated (Control) | Sites (M)        | 4              | 4          | ✅        |
|                        | TAU              | 1.0000         | 1.0000     | ✅        |
|                        | VarTAU (Uncorr)  | 0.0000         | 0.0000     | ✅        |
|                        | Corrected VarTAU | 0.0000         | 0.0000     | ✅        |
|                        | Direction (DT)   | Increasing     | Increasing | ✅        |
|                        | Confidence (CT)  | 1.0000         | 1.0000     | ✅        |
| ---                    | ---              | ---            | ---        | ---      |
