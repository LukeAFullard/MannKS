# Validation Case 44: Seasonal Bootstrap Validation

This validation case verifies that the Seasonal Block Bootstrap Mann-Kendall test correctly handles seasonal data with autocorrelation. Specifically, it checks if the method preserves the seasonal structure during resampling and controls Type I error rates better than the standard test.

## Simulation Setup
- **Data**: Monthly data with strong seasonality + AR(1) noise ($\rho=0.7$)
- **Duration**: 30 years
- **Simulations**: 100
- **Alpha**: 0.05

## Results
The table below shows the rejection rates (Type I Error) for Standard vs. Bootstrap Seasonal MK tests.

| Method                |   Type I Error Rate |
|:----------------------|--------------------:|
| Standard Seasonal MK  |                0.32 |
| Seasonal Bootstrap MK |                0.02 |

## Interpretation
- **Standard Method**: Likely to have inflated Type I error if the autocorrelation is significant, as it treats years as independent observations of the seasonal cycle.
- **Bootstrap Method**: Should have a rejection rate closer to the nominal alpha (0.05), as it resamples blocks of years to preserve the inter-annual dependence structure.

## Structural Verification
The implementation uses `moving_block_bootstrap` on the *indices of the cycles* (years). This ensures that:
1.  **Seasonality is Preserved**: Data for 'January' always remains in the 'January' slot relative to the cycle start.
2.  **Autocorrelation is Preserved**: Blocks of consecutive years are kept together, capturing the serial correlation.
