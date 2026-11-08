# Statistical Methodology: Block Bootstrap for Autocorrelated Data

The `MannKS` package implements block bootstrap methods to handle serial correlation (autocorrelation) in time series data. The standard Mann-Kendall test assumes independence; when this assumption is violated, the Type I error rate (false positive rate) can be substantially inflated.

Two bootstrap procedures are used, one for each inference task: hypothesis testing (the p-value) and estimation (the slope's confidence interval).

## 1. Hypothesis Testing: Detrended Residual Block Bootstrap

To test the null hypothesis ($H_0$) of **no trend**, we generate a null distribution of the Mann-Kendall $S$ statistic that preserves the data's short-range dependence but contains no trend.

**Procedure:**
1.  **Trend estimation:** Estimate the slope $\hat{\beta}$ with the selected estimator (`sens_slope_method`: Akritas-Theil-Sen by default for censored data, Sen's slope otherwise).
2.  **Detrending:** Compute residuals $r_i = x_i - \hat{\beta}\,(t_i - \text{median}(t))$. Centring time avoids precision problems with large timestamps (e.g. Unix epochs).
    *   *Censored data:* the residual of a censored value is computed from its detection limit and keeps its censoring flag. Detrending therefore shifts each detection limit by the removed trend, so resampled censored residuals carry limits that vary over time. This is an approximation; the rank-based $S$ is only affected where such limits overlap uncensored residuals.
3.  **Block length** (`block_size='auto'`): twice the first lag at which the autocorrelation of the **detrended** series drops below 0.1, bounded to $[3, \sqrt{n}]$. This is a simple heuristic, not an optimal block-length selector.
    *   *Seasonal data:* the bootstrap resamples whole cycles (e.g. years), preserving the seasonal structure. The block length (in cycles) is chosen from the autocorrelation of the detrended cycle means.
4.  **Resampling:** a **circular block bootstrap** (Politis & Romano, 1992). Blocks of consecutive residuals, with their censoring flags, start at random positions and wrap around the end of the series, so every observation is equally likely to be drawn.
5.  **No trend reconstruction:** the bootstrap series is the resampled residuals; the trend is not added back, because we are simulating $H_0$.
6.  **Test statistic:** compute $S^*$ on $(r^*, t_{\text{original}})$.
7.  **P-value:** $p = \dfrac{1 + \#\{|S^*_b| \ge |S|\}}{B + 1}$, which is never exactly zero.

`random_state` makes the procedure reproducible.

## 2. Confidence Intervals: Pairs Block Bootstrap

For the slope's confidence interval we use a **pairs block bootstrap**.

**Procedure:**
1.  **Block sampling:** draw circular blocks of observation indices, as above. Each observation keeps its value, time and censoring status ("pairs").
2.  **Resampled data set:** the drawn observations, with their original times. Repeated observations keep their multiplicity; pairs with equal times do not contribute to the slope.
3.  **Estimation:** compute the slope with the selected estimator.
4.  **Interval:** percentile interval of the bootstrap slopes at level $1-\alpha$ (the `alpha` argument; the default is a 95% interval).

### Why the pairs bootstrap? (The "shifting limit" issue)

A residual bootstrap for confidence intervals ($x^*_{boot} = r^*_{boot} + \hat{\beta}t_{original}$) is biased for **censored data**. A censored residual is tied to a particular point in time relative to the trend line. If it is moved to another time and the trend is added back, the implied detection limit changes. For example, with a decreasing trend the late, low (often censored) values have high residuals; moved to the start of the series and with the trend added back, they become large values that are still marked as censored, i.e. artificially high detection limits. This distorts the slope estimate, typically shrinking it towards zero.

The pairs bootstrap avoids this by keeping each value, its censoring status and its time together. It is also robust to heteroscedastic errors.

## Without autocorrelation correction

When `autocorr_method='none'`, the Akritas-Theil-Sen interval is obtained by inverting the censored Kendall test: the limits are the slopes $\beta$ at which $S(\beta) = \pm z_{1-\alpha/2}\sqrt{\operatorname{Var}(S)}$. This is the same construction that gives Sen's rank-based interval for uncensored data.

**References:**
*   Politis, D. N., & Romano, J. P. (1992). A circular block-resampling procedure for stationary data. In *Exploring the Limits of Bootstrap*, Wiley, 263-270.
*   Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
*   Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and their Application*. Cambridge University Press.
*   Akritas, M. G., Murphy, S. A., & LaValley, M. P. (1995). The Theil-Sen estimator with doubly censored data and applications to astronomy. *JASA*, 90(429), 170-177.
