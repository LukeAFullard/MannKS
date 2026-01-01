# V-21: Trend Classification Validation

This document validates the `classify_trend` helper function using various synthetic trend results.

## 1. Default Classification Map (IPCC-based)

| Scenario | Slope | P-Value | C (Confidence) | h (Signif) | Trend | Classification |
|---|---|---|---|---|---|---|
| Strong Increase | 1.5 | 0.001 | 0.9995 | True | increasing | Highly Likely Increasing |
| Likely Increase | 0.5 | 0.15 | 0.9250 | True | increasing | Very Likely Increasing |
| Possible Decrease (but h=True) | -0.2 | 0.6 | 0.7000 | True | decreasing | Likely Decreasing |
| Not Significant (h=False) | 0.01 | 0.9 | 0.5500 | False | no trend | As Likely as Not No trend |

## 2. Custom Classification Map

Map: `{0.90: 'High Confidence', 0.0: 'Low Confidence'}`

| Scenario | Classification |
|---|---|
| Strong Increase | High Confidence Increasing |
| Likely Increase | High Confidence Increasing |
| Possible Decrease (but h=True) | Low Confidence Decreasing |
| Not Significant (h=False) | Low Confidence No trend |
