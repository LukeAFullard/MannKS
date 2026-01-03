# Example 29: Block Bootstrap Validation

This example validates the integration of the Block Bootstrap method with other package features.

## Scenario 1: Aggregation + Bootstrap
Bootstrap is applied *after* temporal aggregation. This is crucial for high-frequency autocorrelated data.
```python
mk.trend_test(..., agg_method='median', agg_period='month', autocorr_method='block_bootstrap')
```
## Scenario 2: High Censoring + Bootstrap
Ensures `hicensor` preprocessing works correctly before the bootstrap resampling step.
```python
mk.trend_test(..., hicensor=True, autocorr_method='block_bootstrap')
```
## Scenario 3: Seasonal Bootstrap
Validates that the seasonal test bootstraps entire seasonal cycles (e.g., years) to preserve seasonality.
```python
mk.seasonal_trend_test(..., autocorr_method='block_bootstrap', block_size='auto')
```
## Scenario 4: ATS Slope + Bootstrap P-value
Combines the robust ATS estimator for the slope/CI with the Block Bootstrap for the p-value.
```python
mk.trend_test(..., sens_slope_method='ats', autocorr_method='block_bootstrap')
```

## Execution Output
```text

--- Scenario 1: Aggregation + Bootstrap ---
Original N: 730
Aggregated N (approx): 24.3
Block Size Used (on aggregated data): 4
P-value (Bootstrap): 0.0000
Trend: increasing

--- Scenario 2: High Censoring + Bootstrap ---
High Censor Rule Applied: ['Long run of single value', 'WARNING: Sen slope influenced by left-censored values.']
Block Size Used: 8
P-value: 0.0000

--- Scenario 3: Seasonal Bootstrap ---
Block Size Used (Cycles): 3
P-value: 0.0000
Trend: increasing

--- Scenario 4: ATS Slope + Bootstrap P-value ---
Slope (ATS): 0.0373
CI (ATS): [0.0146, 0.0618]
P-value (Bootstrap): 0.0200
```

## Conclusion
The successful execution of these scenarios confirms that the Block Bootstrap feature is robustly integrated into the package. It correctly handles:
- **Pre-processing steps** like temporal aggregation and high-censoring adjustments, applying the bootstrap to the processed data.
- **Seasonal structures**, by bootstrapping entire cycles (e.g., years) to preserve seasonal dependence.
- **Advanced estimators** like ATS, allowing users to combine robust slope estimation with robust significance testing.
