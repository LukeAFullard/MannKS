# V-23: Basic Regional Trend

Verification of regional trend aggregation for consistent site trends (Increasing, Decreasing, Stable).

## Results Comparison

| Scenario          | Metric           | Python (MKS)   | R (LWP)    | Match?   |
|-------------------|------------------|----------------|------------|----------|
| Strong Increasing | Sites (M)        | 5              | 5          | ✅        |
|                   | TAU              | 1.0000         | 1.0000     | ✅        |
|                   | VarTAU (Uncorr)  | 0.0000         | 0.0000     | ✅        |
|                   | Corrected VarTAU | 0.0000         | 0.0000     | ✅        |
|                   | Direction (DT)   | Increasing     | Increasing | ✅        |
|                   | Confidence (CT)  | 1.0000         | 1.0000     | ✅        |
| ---               | ---              | ---            | ---        | ---      |
| Weak Decreasing   | Sites (M)        | 5              | 5          | ✅        |
|                   | TAU              | 1.0000         | 1.0000     | ✅        |
|                   | VarTAU (Uncorr)  | 0.0000         | 0.0000     | ✅        |
|                   | Corrected VarTAU | 0.0000         | 0.0000     | ✅        |
|                   | Direction (DT)   | Decreasing     | Decreasing | ✅        |
|                   | Confidence (CT)  | 1.0000         | 1.0000     | ✅        |
| ---               | ---              | ---            | ---        | ---      |
| Stable            | Sites (M)        | 5              | 5          | ✅        |
|                   | TAU              | 0.6000         | 0.6000     | ✅        |
|                   | VarTAU (Uncorr)  | 0.0243         | 0.0243     | ✅        |
|                   | Corrected VarTAU | 0.0226         | 0.0226     | ✅        |
|                   | Direction (DT)   | Increasing     | Increasing | ✅        |
|                   | Confidence (CT)  | 0.7472         | 0.7472     | ✅        |
| ---               | ---              | ---            | ---        | ---      |
