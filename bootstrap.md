# Statistical Methodology: Block Bootstrap for Autocorrelated Data

The `MannKS` package implements block bootstrap methods to robustly handle serial correlation (autocorrelation) in time series data. Standard Mann-Kendall tests assume independence; when this assumption is violated, the Type I error rate (false positive rate) can be significantly inflated.

We use two distinct bootstrap strategies for the two main goals of trend analysis: Hypothesis Testing (P-value) and Estimation (Confidence Intervals). This combination is statistically rigorous as it addresses the specific requirements of each statistical inference task.

## 1. Hypothesis Testing: Detrended Residual Block Bootstrap

To test the null hypothesis ($H_0$) that there is **no trend**, we must generate a null distribution of the Mann-Kendall $S$ statistic that preserves the data's autocorrelation structure but removes any existing trend.

**Procedure:**
1.  **Trend Estimation:** Estimate the trend using Sen's Slope ($\hat{\beta}$).
2.  **Detrending:** Calculate residuals: $r_i = x_i - \hat{\beta} t_i$.
    *   *Numerical Stability:* We use centered time ($t_{centered} = t - \text{median}(t)$) for this calculation. This prevents floating-point precision issues that can arise when working with large timestamp values (e.g., Unix epochs).
    *   *Note on Censored Data:* For censored values (e.g., $<5$), the residual is calculated using the numeric detection limit. While this is an approximation, the rank-based nature of the Mann-Kendall test makes it robust to the specific values of these residuals as long as the temporal structure is preserved.
3.  **Block Size Selection:**
    *   By default (`block_size='auto'`), the optimal block length is calculated based on the data's estimated **correlation length** (the lag at which autocorrelation drops below a significance threshold). This ensures blocks are large enough to capture the dependence structure but small enough to provide sufficient randomness.
    *   *Seasonal Data:* For seasonal tests, "blocks" effectively represent complete seasonal cycles (e.g., full years) to preserve the seasonal structure within the bootstrapped samples.
4.  **Resampling:** Apply a **Moving Block Bootstrap** to the residuals.
    *   Blocks of consecutive residuals are sampled with replacement to form a new series $r^*_{boot}$.
5.  **No Trend Reconstruction:** The bootstrap sample is simply the shuffled residuals ($x^*_{boot} = r^*_{boot}$). We do *not* add the trend back, because we are simulating $H_0$.
6.  **Test Statistic:** Calculate the Mann-Kendall $S$ statistic on $(x^*_{boot}, t_{original})$.
7.  **P-value:** Compare the observed $S$ against the distribution of bootstrap $S$ values.

**References:**
*   Politis, D. N., & White, H. (2004). Automatic block-length selection for the dependent bootstrap. *Econometric Reviews*, 23(1), 53-70.

---

## 2. Confidence Intervals: Pairs Block Bootstrap

For calculating confidence intervals around the Sen's Slope, we use a **Pairs Block Bootstrap**.

**Procedure:**
1.  **Block Sampling:** Instead of separating residuals, we sample blocks of time-value pairs $(x_i, t_i)$.
2.  **Resampling:** Concatenate these blocks to form a new dataset $(x^*_{boot}, t^*_{boot})$.
    *   *Note:* The bootstrapped time vector $t^*_{boot}$ will contain duplicate timestamps and will not be strictly increasing.
3.  **Sorting:** To ensure correct calculation of pairwise slopes (especially for censored data logic), we sort the bootstrapped sample by time: $(x^*_{sorted}, t^*_{sorted})$.
4.  **Estimation:** Calculate Sen's Slope on this sorted bootstrapped dataset.
4.  **Confidence Interval:** The 90% CI is determined from the 5th and 95th percentiles of the bootstrap distribution of slopes.

### Why Pairs Bootstrap? (The "Shifting Limit" Issue)

We previously attempted a residual-based bootstrap for confidence intervals ($x^*_{boot} = r^*_{boot} + \hat{\beta}t_{original}$). However, this method introduces a systematic bias when applied to **censored data**.

**The Problem:**
In a residual bootstrap, a "censored residual" is derived from a specific point in time relative to the trend line. If we move this residual to a different point in time (by shuffling) and add the trend back, the implied "detection limit" changes.

*   *Example:* Consider a decreasing trend. Data at the end of the series is often censored (low values). These have "high" residuals relative to the trend line. If we move such a residual to the beginning of the series (where values are high), and add the high trend value, the reconstructed data point becomes a very large number that is still marked as "censored". This effectively creates a "high detection limit" data point at the start of the series.
*   **Result:** This artificial mixing of detection limits distorts the slope calculation, typically biasing the slope magnitude downwards (towards zero) or creating asymmetric confidence intervals that do not cover the true parameter.

**The Solution:**
The **Pairs Bootstrap** avoids this entirely by keeping the value $x_i$, its censoring status, and its time $t_i$ together. We do not try to "reconstruct" values; we simply re-weight the original observations while preserving their local dependence.

**Additional Benefits:**
*   **Heteroscedasticity:** Pairs bootstrap is also robust to non-constant variance (heteroscedasticity) in the error terms, as it does not rely on the assumption that residuals are identically distributed across the entire time series.

**References:**
*   Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall/CRC. (Section 8.6 on Moving Block Bootstrap).
*   Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and their Application*. Cambridge University Press. (Discussion on model-based vs. pairs resampling).
