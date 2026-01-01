# Validation Case V-18: `lt_mult` and `gt_mult` Parameters

## Objective
This validation case demonstrates the effect of the `lt_mult` and `gt_mult` parameters, which are used to replicate the LWP-TRENDS methodology for calculating Sen's slope with censored data.

## Data
A small, carefully crafted dataset of 3 points is used to isolate the effect of `lt_mult`. The dataset is `(<10, 2000)`, `(12, 2001)`, `(20, 2002)`.

This results in 3 pairwise slopes. By design, the median of these three slopes is the one calculated between the first and third points, making the final Sen's slope directly dependent on the numeric value substituted for `'<10'`.

## Analysis

The `sens_slope_method='lwp'` in `MannKS` emulates the LWP-TRENDS R script's behavior of substituting censored values before calculating pairwise slopes. For left-censored data (`<D`), the substitution is `D * lt_mult`. The R script hardcodes this multiplier to 0.5.

#### Case 1: `lt_mult=0.5` (The LWP-TRENDS Default)
- The value for `'<10'` becomes `10 * 0.5 = 5`.
- The three pairwise slopes are:
  - `(12 - 5) / 1 year = 7.0`
  - `(20 - 12) / 1 year = 8.0`
  - `(20 - 5) / 2 years = 7.5`
- The sorted slopes are `[7.0, 7.5, 8.0]`. The median (the Sen's slope) is **7.5**.

#### Case 2: `lt_mult=0.1`
- The value for `'<10'` becomes `10 * 0.1 = 1`.
- The three pairwise slopes are:
  - `(12 - 1) / 1 year = 11.0`
  - `(20 - 12) / 1 year = 8.0`
  - `(20 - 1) / 2 years = 9.5`
- The sorted slopes are `[8.0, 9.5, 11.0]`. The median (the Sen's slope) is **9.5**.

## Results Comparison

The following table shows the calculated Sen's slope for each run. To get a reliable result from the R script, its fragile high-level wrappers were bypassed, and the core internal slope calculation function (`GetInterObservationSlopes`) was called directly.

| Analysis                        | Sen's Slope (per year) |
|---------------------------------|------------------------|
| `MannKS` (`lt_mult=0.5`)      | 7.494870       |
| `MannKS` (`lt_mult=0.1`)      | 9.493502      |
| LWP-TRENDS R Script (Internal)  | 7.494870          |

## Conclusion
The results confirm that the `lt_mult` parameter in `MannKS` functions exactly as designed.

- By calling the R script's internal slope function, we confirm its core logic is equivalent to a hardcoded `lt_mult=0.5`. Its result now correctly matches the corresponding `MannKS` run.
- Changing `lt_mult` in `MannKS` correctly alters the final Sen's slope, providing the intended flexibility.
