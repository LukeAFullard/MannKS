# V-24: Regional Trend with Mixed Directions

Verification of regional trend aggregation for sites with conflicting trend directions.

## Results Comparison

| Scenario                | Metric           | Python (MKS)   | R (LWP)    | Match?   |
|-------------------------|------------------|----------------|------------|----------|
| Aggregate Increasing    | Sites (M)        | 5              | 5          | ✅        |
|                         | TAU              | 0.8000         | 0.8000     | ✅        |
|                         | VarTAU (Uncorr)  | 0.0000         | 0.0000     | ✅        |
|                         | Corrected VarTAU | 0.0000         | 0.0000     | ✅        |
|                         | Direction (DT)   | Increasing     | Increasing | ✅        |
|                         | Confidence (CT)  | 1.0000         | 1.0000     | ✅        |
| ---                     | ---              | ---            | ---        | ---      |
| Aggregate Decreasing    | Sites (M)        | 5              | 5          | ✅        |
|                         | TAU              | 0.8000         | 0.8000     | ✅        |
|                         | VarTAU (Uncorr)  | 0.0000         | 0.0000     | ✅        |
|                         | Corrected VarTAU | 0.0000         | 0.0000     | ✅        |
|                         | Direction (DT)   | Decreasing     | Decreasing | ✅        |
|                         | Confidence (CT)  | 1.0000         | 1.0000     | ✅        |
| ---                     | ---              | ---            | ---        | ---      |
| Aggregate Indeterminate | Sites (M)        | 5              | 5          | ✅        |
|                         | TAU              | 0.6000         | 0.6000     | ✅        |
|                         | VarTAU (Uncorr)  | 0.0050         | 0.0050     | ✅        |
|                         | Corrected VarTAU | 0.0050         | 0.0050     | ✅        |
|                         | Direction (DT)   | Decreasing     | Decreasing | ✅        |
|                         | Confidence (CT)  | 0.9211         | 0.9211     | ✅        |
| ---                     | ---              | ---            | ---        | ---      |
