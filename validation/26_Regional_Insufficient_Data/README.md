# V-26: Regional Trend with Insufficient Site Data

This test checks the robustness of the regional test when one site (SiteC) provides invalid or empty results (simulated as insufficient data).

The expected behavior is that the invalid site is excluded from the calculation, meaning the number of sites (M) should be 2, not 3. The regional statistics (TAU, etc.) should be calculated based only on the valid sites (A and B).

## Results Comparison

| Scenario               | Metric           | Python (MKS)   | R (LWP)    | Match?   |
|------------------------|------------------|----------------|------------|----------|
| Mixed Data Sufficiency | Sites (M)        | 2              | 2          | ✅        |
|                        | TAU              | 1.0000         | 1.0000     | ✅        |
|                        | VarTAU (Uncorr)  | 0.0000         | 0.0000     | ✅        |
|                        | Corrected VarTAU | 0.0000         | 0.0000     | ✅        |
|                        | Direction (DT)   | Increasing     | Increasing | ✅        |
|                        | Confidence (CT)  | 1.0000         | 1.0000     | ✅        |
| ---                    | ---              | ---            | ---        | ---      |
