# V-01: Simple Trend
...
**Results:**

| Scenario          | Method                  |   Sen's Slope (per year) |   p-value | Trend                    |
|:------------------|:------------------------|-------------------------:|----------:|:-------------------------|
| strong_increasing | `MannKS` (Standard) |                   0.5806 |    0      | Highly Likely Increasing |
|                   | `MannKS` (LWP Mode) |                   0.5806 |    0      | Highly Likely Increasing |
|                   | LWP-TRENDS R Script     |                   0.5806 |    0      | Increasing               |
| weak_decreasing   | `MannKS` (Standard) |                  -0.1085 |    0.1376 | No Trend                 |
|                   | `MannKS` (LWP Mode) |                  -0.1085 |    0.1376 | No Trend                 |
|                   | LWP-TRENDS R Script     |                  -0.1085 |    0.1376 | Decreasing               |
| stable_no_trend   | `MannKS` (Standard) |                  -0.0099 |    1      | No Trend                 |
|                   | `MannKS` (LWP Mode) |                  -0.0099 |    1      | No Trend                 |
|                   | LWP-TRENDS R Script     |                  -0.0099 |    1      | No Trend                 |
...
