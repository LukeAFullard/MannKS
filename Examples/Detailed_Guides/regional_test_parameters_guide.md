# A Comprehensive Guide to `regional_test` Parameters

The `regional_test` function is a powerful tool for aggregating trend analysis results from multiple sites to determine if there is a consistent, region-wide trend. It implements the Regional Kendall Test. This guide explains its parameters, usage, and interpretation.

---

### Core Parameters

These are the essential parameters you will always need to provide.

#### `trend_results`
-   **Type:** `pandas.DataFrame`
-   **Description:** A DataFrame containing the summary statistics for each site. This is typically constructed from the results of running `trend_test` or `seasonal_trend_test` on each individual site.
-   **Usefulness:** This input provides the core signal (trend direction and strength) for each location.
-   **Requirements:** Must contain columns for the site identifier, the Mann-Kendall score (S), and the confidence (C).

#### `time_series_data`
-   **Type:** `pandas.DataFrame`
-   **Description:** A DataFrame containing the raw time series data for all sites.
-   **Usefulness:** This is used to calculate the **inter-site correlation**. A key challenge in regional analysis is that sites in the same region are often correlated (e.g., a drought affects all of them). If this correlation is not accounted for, the significance of the regional trend will be overestimated. This data allows the test to correct for that correlation.
-   **Requirements:** Must contain columns for the site identifier, the time (or date), and the value.

---

### Column Mapping Parameters

These parameters tell the function which columns in your DataFrames correspond to the required inputs.

#### `site_col`
-   **Type:** `str`, **Default:** `'site'`
-   **Description:** The name of the column containing unique site identifiers.
-   **Usefulness:** Critical for linking the `trend_results` to the `time_series_data`. This column name must exist in **both** input DataFrames.

#### `value_col`
-   **Type:** `str`, **Default:** `'value'`
-   **Description:** The name of the column in `time_series_data` containing the observed values.

#### `time_col`
-   **Type:** `str`, **Default:** `'time'`
-   **Description:** The name of the column in `time_series_data` containing the timestamps.
-   **Usefulness:** Used to align data across sites to calculate pairwise correlations.

#### `agg_period`
-   **Type:** `str` or `None`, **Default:** `'month'`
-   **Description:** For datetime `time_col`, each site's series is first reduced to one value per time increment (`'year'`, `'quarter'`, `'month'`, `'week'` or `'day'`; the observation closest to the increment mid-point), and the sites are aligned on these increments before the correlations are computed, as LWP-TRENDS `getTAU` does.
-   **Usefulness:** Sites are rarely sampled at exactly the same times. Aligning by increment lets sites sampled on different days of the same month be correlated. Use the time increment of your site trend analyses (e.g. `'year'` for annual data). `None` correlates on exact timestamps. Ignored for numeric `time_col`.

#### `s_col`
-   **Type:** `str`, **Default:** `'s'`
-   **Description:** The name of the column in `trend_results` containing the Mann-Kendall S-statistic.

#### `c_col`
-   **Type:** `str`, **Default:** `'C'`
-   **Description:** The name of the column in `trend_results` containing the trend confidence value `C` (0.5 to 1).

---

### Output Interpretation

The function returns a `namedtuple` with the following fields:

-   **`M`**: The total number of sites included in the analysis.
-   **`TAU`**: The aggregate trend strength. It represents the consistency of the trend across the region.
-   **`VarTAU`**: The variance of TAU, assuming sites are independent.
-   **`CorrectedVarTAU`**: The variance of TAU, **corrected for inter-site correlation**. This is the value used for the final confidence. Site pairs whose correlation cannot be computed (too few common time increments) are omitted from the correction, with a warning, as in LWP-TRENDS.
-   **`DT`**: The aggregate trend direction ('Increasing' or 'Decreasing'). When exactly half the sites go each way, 'Increasing' is reported with TAU = 0.5 and CT = 0.5, as in LWP-TRENDS.
-   **`CT`**: The confidence in the aggregate trend direction (0.5 to 1). High confidence (e.g., >0.95) indicates a consistent regional trend.
-   **`warnings`**: Warnings raised during the calculation.
