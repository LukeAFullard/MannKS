# Example 36: Generalized Quantile-Kendall Trend Analysis

This example demonstrates the **Generalized Quantile-Kendall (QK) Framework** introduced in **MannKS v0.7.0**. 

Traditional monotonic trend tests (such as standard Mann-Kendall or Seasonal Mann-Kendall) evaluate trends only in the **central tendency** (mean or median). However, environmental and hydrological changes rarely affect all percentiles uniformly:
- Agricultural pollution flushes streamflow concentrations primarily during high-runoff storm events (upper quantiles), leaving summer baseflow unchanged.
- Climate warming often intensifies hydrological extremes, causing flood peaks to increase while drought baseflows diminish.
- Ambient groundwater monitoring networks collect sparse, irregular, and censored data where classical daily methods cannot apply.
- Historical shifts in sampling frequency (e.g., transition from monthly to weekly monitoring) introduce severe sample-size artifacts into tail quantiles if not properly diagnosed and equalized.

MannKS v0.7.0 addresses these challenges by extending the order-statistic trend concept of USGS EGRET (*Hirsch et al., 2015*) into a **fully generalized, frequency-agnostic, censor-aware, and drift-protected** framework.

---

## Table of Contents
1. [Case 1: Weekly Nitrate Streamflow (Winter Flush Degradation)](#case-1-weekly-nitrate-streamflow)
2. [Case 2: Daily Streamflow with 5-Year Blocks (Divergent Hydrological Regimes)](#case-2-daily-streamflow-with-5-year-blocks)
3. [Case 3: Sparse Irregular Censored Groundwater (Ambient Monitoring)](#case-3-sparse-irregular-censored-groundwater)
4. [Case 4: The Sampling Frequency Drift Trap & Equalization](#case-4-the-sampling-frequency-drift-trap--equalization)
5. [Summary of Quantile-Kendall Diagnostics](#summary-of-quantile-kendall-diagnostics)

---

## Case 1: Weekly Nitrate Streamflow

### Scientific Background
A 20-year weekly streamflow monitoring program (2005–2024, $N=1,044$) tracks dissolved nitrate concentrations ($mg/L$) in an agricultural catchment. Subsurface tile drainage delivers heavy nitrate pulses during winter runoff events, while summer baseflow remains dominated by deep groundwater with stable concentrations.

A global median test would dilute this critical seasonal pollution signal. The Quantile-Kendall test segments the series into annual blocks (`block_frequency='YE'`), standardizes each year into 52 weekly order statistics, and evaluates Sen's slope across all non-exceedance probabilities.

### Python Code
```python
import numpy as np
import pandas as pd
from MannKS import quantile_trend_test, plot_quantile_trend, summarize_quantile_regimes

# 1. Generate 20 years of weekly monitoring data
rng = np.random.default_rng(42)
dates = pd.date_range("2005-01-01", "2024-12-31", freq="W")
years = (dates - dates[0]).days / 365.25
day_of_year = dates.day_of_year if hasattr(dates, 'day_of_year') else dates.dayofyear
is_winter = np.cos(2 * np.pi * (day_of_year - 30) / 365.25) > 0.4
nitrate = 1.2 + np.where(is_winter, 2.5 * (1.0 + 0.06 * years), 0.2) + rng.lognormal(0.0, 0.25, size=len(dates))

# 2. Run Generalized Quantile-Kendall Test
result = quantile_trend_test(
    x=nitrate,
    t=dates,
    block_frequency='YE',
    quantiles='auto',
    slope_scaling='year',
    x_unit='mg/L',
)

# 3. Diagnostic Plot with Flow Duration Curve overlay
plot_quantile_trend(
    result,
    save_path='case1_weekly_nitrate_qk.png',
    show_ci=True,
    show_fdc=True,
    title='Case 1: Weekly Nitrate Streamflow (Winter Flush Trend)',
)
```

### Visual Diagnostics
![Case 1: Weekly Nitrate Streamflow](case1_weekly_nitrate_qk.png)

### Results & Regime Summary
The resulting quantile slopes show clear selective degradation in the upper percentiles:

#### Sample Results Across Ranks
| rank | quantile | plotting_position | slope   | p         | classification              |
| ---- | -------- | ----------------- | ------- | --------- | --------------------------- |
| 1    | 0.0000   | 0.0189            | -0.0022 | 0.4957    | Likely Decreasing           |
| 6    | 0.0980   | 0.1132            | 0.0001  | 0.9225    | As Likely as Not Increasing |
| 11   | 0.1961   | 0.2075            | 0.0014  | 0.4957    | Likely Increasing           |
| 16   | 0.2941   | 0.3019            | 0.0018  | 0.4957    | Likely Increasing           |
| 21   | 0.3922   | 0.3962            | 0.0001  | 0.8711    | As Likely as Not Increasing |
| 26   | 0.4902   | 0.4906            | -0.0014 | 0.6732    | As Likely as Not Decreasing |
| 31   | 0.5882   | 0.5849            | -0.0008 | 0.9741    | As Likely as Not Decreasing |
| 36   | 0.6863   | 0.6792            | 0.1496  | 1.303e-09 | Highly Likely Increasing    |
| 41   | 0.7843   | 0.7736            | 0.1521  | 1.303e-09 | Highly Likely Increasing    |
| 46   | 0.8824   | 0.8679            | 0.1523  | 1.947e-09 | Highly Likely Increasing    |
| 51   | 0.9804   | 0.9623            | 0.1529  | 1.365e-08 | Highly Likely Increasing    |

#### Quantile Regimes Breakdown
| regime | p_min  | p_max  | n_quantiles | median_slope | median_slope_pct | prop_significant | modal_classification        |
| ------ | ------ | ------ | ----------- | ------------ | ---------------- | ---------------- | --------------------------- |
| low    | 0.0000 | 0.2500 | 13          | 0.0004       | 0.0155           | 0.0000           | As Likely as Not Increasing |
| mid    | 0.2500 | 0.7500 | 26          | 0.0001       | 0.0055           | 0.2308           | As Likely as Not Decreasing |
| high   | 0.7500 | 1.0000 | 13          | 0.1521       | 2.4118           | 1.0000           | Highly Likely Increasing    |

**Key Takeaway:** The low and mid percentiles (summer baseflow) exhibit no significant trend ($p > 0.10$, slope $\approx 0$), whereas high quantiles ($p > 0.75$) exhibit highly significant increasing trends ($p < 0.001$, slope $> 0.15$ mg/L/year), capturing selective agricultural winter runoff pollution.

---

## Case 2: Daily Streamflow with 5-Year Blocks

### Scientific Background
30 years of daily streamflow ($m^3/s$) from 1990 to 2019 (~10,957 observations) are analyzed. In this watershed, climate change and land-use alterations have created **divergent hydrological regimes**:
1. Extreme precipitation events have intensified flood peaks (+3% per year in top quantiles).
2. Extended summer droughts and agricultural water extraction have caused baseflow to diminish (-2% per year in lower quantiles).

By setting `block_frequency='5YE'`, daily observations are grouped into 6 multi-year hydrological epochs of 1,826 order statistics each. Log-transformation (`transform='log'`) reports slopes directly as percentage change per year (`% per year`).

### Python Code
```python
# Daily streamflow with multi-year blocking and log transform
result2 = quantile_trend_test(
    x=daily_flow,
    t=daily_dates,
    block_frequency='5YE',
    quantiles='auto',
    transform='log',
    slope_scaling='year',
    x_unit='m3/s',
)

plot_quantile_trend(
    result2,
    save_path='case2_daily_flow_5y_qk.png',
    show_ci=True,
    show_fdc=True,
    title='Case 2: Daily Streamflow 5-Year Blocks (Divergent Flood vs Baseflow Trends)',
)

# Summarize custom hydrologic regimes
regimes2 = summarize_quantile_regimes(
    result2,
    regimes={
        'drought_baseflow': (0.0, 0.50),
        'moderate_flow': (0.50, 0.95),
        'flood_peaks': (0.95, 1.0),
    },
)
```

### Visual Diagnostics
![Case 2: Daily Streamflow 5-Year Blocks](case2_daily_flow_5y_qk.png)

### Results & Regime Summary
#### Sample Results Across Quantiles (%/year)
| rank | quantile | plotting_position | slope_pct | p      | classification              |
| ---- | -------- | ----------------- | --------- | ------ | --------------------------- |
| 1    | 0.0000   | 0.0005            | -2.3173   | 0.0242 | Highly Likely Decreasing    |
| 11   | 0.0055   | 0.0060            | -1.9370   | 0.0085 | Highly Likely Decreasing    |
| 21   | 0.0110   | 0.0115            | -1.9367   | 0.0085 | Highly Likely Decreasing    |
| 31   | 0.0164   | 0.0170            | -1.9864   | 0.0085 | Highly Likely Decreasing    |
| 41   | 0.0219   | 0.0224            | -1.9785   | 0.0085 | Highly Likely Decreasing    |
| 51   | 0.0274   | 0.0279            | -1.9636   | 0.0085 | Highly Likely Decreasing    |
| 61   | 0.0329   | 0.0334            | -1.9428   | 0.0085 | Highly Likely Decreasing    |
| 71   | 0.0384   | 0.0389            | -1.9468   | 0.0085 | Highly Likely Decreasing    |
| 81   | 0.0438   | 0.0443            | -1.9792   | 0.0085 | Highly Likely Decreasing    |
| 91   | 0.0493   | 0.0498            | -1.9656   | 0.0085 | Highly Likely Decreasing    |
| 101  | 0.0548   | 0.0553            | -1.9824   | 0.0085 | Highly Likely Decreasing    |
| 111  | 0.0603   | 0.0608            | -2.0048   | 0.0085 | Highly Likely Decreasing    |
| 121  | 0.0658   | 0.0662            | -2.0249   | 0.0085 | Highly Likely Decreasing    |
| 131  | 0.0712   | 0.0717            | -2.0138   | 0.0085 | Highly Likely Decreasing    |
| 141  | 0.0767   | 0.0772            | -1.9965   | 0.0085 | Highly Likely Decreasing    |
| 151  | 0.0822   | 0.0826            | -1.9889   | 0.0085 | Highly Likely Decreasing    |
| 161  | 0.0877   | 0.0881            | -2.0056   | 0.0085 | Highly Likely Decreasing    |
| 171  | 0.0932   | 0.0936            | -1.9889   | 0.0085 | Highly Likely Decreasing    |
| 181  | 0.0986   | 0.0991            | -1.9751   | 0.0085 | Highly Likely Decreasing    |
| 191  | 0.1041   | 0.1045            | -1.9693   | 0.0085 | Highly Likely Decreasing    |
| 201  | 0.1096   | 0.1100            | -1.9661   | 0.0085 | Highly Likely Decreasing    |
| 211  | 0.1151   | 0.1155            | -1.9920   | 0.0085 | Highly Likely Decreasing    |
| 221  | 0.1205   | 0.1210            | -1.9862   | 0.0085 | Highly Likely Decreasing    |
| 231  | 0.1260   | 0.1264            | -2.0093   | 0.0085 | Highly Likely Decreasing    |
| 241  | 0.1315   | 0.1319            | -2.0172   | 0.0085 | Highly Likely Decreasing    |
| 251  | 0.1370   | 0.1374            | -2.0224   | 0.0085 | Highly Likely Decreasing    |
| 261  | 0.1425   | 0.1429            | -2.0446   | 0.0085 | Highly Likely Decreasing    |
| 271  | 0.1479   | 0.1483            | -2.0395   | 0.0085 | Highly Likely Decreasing    |
| 281  | 0.1534   | 0.1538            | -2.0205   | 0.0085 | Highly Likely Decreasing    |
| 291  | 0.1589   | 0.1593            | -2.0184   | 0.0085 | Highly Likely Decreasing    |
| 301  | 0.1644   | 0.1648            | -2.0118   | 0.0085 | Highly Likely Decreasing    |
| 311  | 0.1699   | 0.1702            | -2.0034   | 0.0085 | Highly Likely Decreasing    |
| 321  | 0.1753   | 0.1757            | -2.0007   | 0.0085 | Highly Likely Decreasing    |
| 331  | 0.1808   | 0.1812            | -2.0062   | 0.0085 | Highly Likely Decreasing    |
| 341  | 0.1863   | 0.1866            | -2.0019   | 0.0085 | Highly Likely Decreasing    |
| 351  | 0.1918   | 0.1921            | -2.0138   | 0.0085 | Highly Likely Decreasing    |
| 361  | 0.1973   | 0.1976            | -2.0052   | 0.0085 | Highly Likely Decreasing    |
| 371  | 0.2027   | 0.2031            | -1.9854   | 0.0085 | Highly Likely Decreasing    |
| 381  | 0.2082   | 0.2085            | -1.9648   | 0.0085 | Highly Likely Decreasing    |
| 391  | 0.2137   | 0.2140            | -2.0022   | 0.0085 | Highly Likely Decreasing    |
| 401  | 0.2192   | 0.2195            | -1.9872   | 0.0085 | Highly Likely Decreasing    |
| 411  | 0.2247   | 0.2250            | -1.9996   | 0.0085 | Highly Likely Decreasing    |
| 421  | 0.2301   | 0.2304            | -1.9844   | 0.0085 | Highly Likely Decreasing    |
| 431  | 0.2356   | 0.2359            | -1.9865   | 0.0085 | Highly Likely Decreasing    |
| 441  | 0.2411   | 0.2414            | -1.9913   | 0.0085 | Highly Likely Decreasing    |
| 451  | 0.2466   | 0.2469            | -1.9844   | 0.0085 | Highly Likely Decreasing    |
| 461  | 0.2521   | 0.2523            | -1.9930   | 0.0085 | Highly Likely Decreasing    |
| 471  | 0.2575   | 0.2578            | -1.9861   | 0.0085 | Highly Likely Decreasing    |
| 481  | 0.2630   | 0.2633            | -1.9757   | 0.0085 | Highly Likely Decreasing    |
| 491  | 0.2685   | 0.2687            | -1.9574   | 0.0085 | Highly Likely Decreasing    |
| 501  | 0.2740   | 0.2742            | -1.9875   | 0.0085 | Highly Likely Decreasing    |
| 511  | 0.2795   | 0.2797            | -2.0087   | 0.0085 | Highly Likely Decreasing    |
| 521  | 0.2849   | 0.2852            | -2.0142   | 0.0085 | Highly Likely Decreasing    |
| 531  | 0.2904   | 0.2906            | -1.9693   | 0.0085 | Highly Likely Decreasing    |
| 541  | 0.2959   | 0.2961            | -1.9814   | 0.0085 | Highly Likely Decreasing    |
| 551  | 0.3014   | 0.3016            | -1.9981   | 0.0085 | Highly Likely Decreasing    |
| 561  | 0.3068   | 0.3071            | -1.9985   | 0.0085 | Highly Likely Decreasing    |
| 571  | 0.3123   | 0.3125            | -1.9902   | 0.0085 | Highly Likely Decreasing    |
| 581  | 0.3178   | 0.3180            | -1.9969   | 0.0085 | Highly Likely Decreasing    |
| 591  | 0.3233   | 0.3235            | -2.0126   | 0.0085 | Highly Likely Decreasing    |
| 601  | 0.3288   | 0.3290            | -1.9984   | 0.0085 | Highly Likely Decreasing    |
| 611  | 0.3342   | 0.3344            | -1.9903   | 0.0085 | Highly Likely Decreasing    |
| 621  | 0.3397   | 0.3399            | -2.0351   | 0.0085 | Highly Likely Decreasing    |
| 631  | 0.3452   | 0.3454            | -2.0310   | 0.0085 | Highly Likely Decreasing    |
| 641  | 0.3507   | 0.3508            | -2.0018   | 0.0085 | Highly Likely Decreasing    |
| 651  | 0.3562   | 0.3563            | -2.0134   | 0.0085 | Highly Likely Decreasing    |
| 661  | 0.3616   | 0.3618            | -2.0087   | 0.0085 | Highly Likely Decreasing    |
| 671  | 0.3671   | 0.3673            | -2.0380   | 0.0085 | Highly Likely Decreasing    |
| 681  | 0.3726   | 0.3727            | -2.0468   | 0.0085 | Highly Likely Decreasing    |
| 691  | 0.3781   | 0.3782            | -2.0322   | 0.0085 | Highly Likely Decreasing    |
| 701  | 0.3836   | 0.3837            | -2.0194   | 0.0085 | Highly Likely Decreasing    |
| 711  | 0.3890   | 0.3892            | -2.0231   | 0.0085 | Highly Likely Decreasing    |
| 721  | 0.3945   | 0.3946            | -2.0244   | 0.0085 | Highly Likely Decreasing    |
| 731  | 0.4000   | 0.4001            | -2.0243   | 0.0085 | Highly Likely Decreasing    |
| 741  | 0.4055   | 0.4056            | -2.0370   | 0.0085 | Highly Likely Decreasing    |
| 751  | 0.4110   | 0.4111            | -2.0558   | 0.0085 | Highly Likely Decreasing    |
| 761  | 0.4164   | 0.4165            | -2.0624   | 0.0085 | Highly Likely Decreasing    |
| 771  | 0.4219   | 0.4220            | -2.0638   | 0.0085 | Highly Likely Decreasing    |
| 781  | 0.4274   | 0.4275            | -2.0691   | 0.0085 | Highly Likely Decreasing    |
| 791  | 0.4329   | 0.4330            | -2.0730   | 0.0085 | Highly Likely Decreasing    |
| 801  | 0.4384   | 0.4384            | -2.0793   | 0.0085 | Highly Likely Decreasing    |
| 811  | 0.4438   | 0.4439            | -2.0696   | 0.0085 | Highly Likely Decreasing    |
| 821  | 0.4493   | 0.4494            | -2.0793   | 0.0085 | Highly Likely Decreasing    |
| 831  | 0.4548   | 0.4548            | -2.1041   | 0.0085 | Highly Likely Decreasing    |
| 841  | 0.4603   | 0.4603            | -2.0965   | 0.0085 | Highly Likely Decreasing    |
| 851  | 0.4658   | 0.4658            | -2.0666   | 0.0085 | Highly Likely Decreasing    |
| 861  | 0.4712   | 0.4713            | -2.0516   | 0.0085 | Highly Likely Decreasing    |
| 871  | 0.4767   | 0.4767            | -2.0565   | 0.0085 | Highly Likely Decreasing    |
| 881  | 0.4822   | 0.4822            | -2.0532   | 0.0085 | Highly Likely Decreasing    |
| 891  | 0.4877   | 0.4877            | -2.0435   | 0.0085 | Highly Likely Decreasing    |
| 901  | 0.4932   | 0.4932            | -2.0391   | 0.0085 | Highly Likely Decreasing    |
| 911  | 0.4986   | 0.4986            | -2.0571   | 0.0085 | Highly Likely Decreasing    |
| 921  | 0.5041   | 0.5041            | -2.0324   | 0.0085 | Highly Likely Decreasing    |
| 931  | 0.5096   | 0.5096            | -2.0421   | 0.0085 | Highly Likely Decreasing    |
| 941  | 0.5151   | 0.5151            | -2.0342   | 0.0085 | Highly Likely Decreasing    |
| 951  | 0.5205   | 0.5205            | -2.0307   | 0.0085 | Highly Likely Decreasing    |
| 961  | 0.5260   | 0.5260            | -2.0226   | 0.0085 | Highly Likely Decreasing    |
| 971  | 0.5315   | 0.5315            | -2.0068   | 0.0085 | Highly Likely Decreasing    |
| 981  | 0.5370   | 0.5369            | -2.0086   | 0.0085 | Highly Likely Decreasing    |
| 991  | 0.5425   | 0.5424            | -2.0019   | 0.0085 | Highly Likely Decreasing    |
| 1001 | 0.5479   | 0.5479            | -2.0020   | 0.0085 | Highly Likely Decreasing    |
| 1011 | 0.5534   | 0.5534            | -1.9946   | 0.0085 | Highly Likely Decreasing    |
| 1021 | 0.5589   | 0.5588            | -1.9968   | 0.0085 | Highly Likely Decreasing    |
| 1031 | 0.5644   | 0.5643            | -2.0006   | 0.0085 | Highly Likely Decreasing    |
| 1041 | 0.5699   | 0.5698            | -2.0091   | 0.0085 | Highly Likely Decreasing    |
| 1051 | 0.5753   | 0.5753            | -2.0019   | 0.0085 | Highly Likely Decreasing    |
| 1061 | 0.5808   | 0.5807            | -2.0048   | 0.0085 | Highly Likely Decreasing    |
| 1071 | 0.5863   | 0.5862            | -1.9896   | 0.0085 | Highly Likely Decreasing    |
| 1081 | 0.5918   | 0.5917            | -1.9744   | 0.0085 | Highly Likely Decreasing    |
| 1091 | 0.5973   | 0.5972            | -1.9814   | 0.0085 | Highly Likely Decreasing    |
| 1101 | 0.6027   | 0.6026            | -1.9984   | 0.0085 | Highly Likely Decreasing    |
| 1111 | 0.6082   | 0.6081            | -1.9667   | 0.0085 | Highly Likely Decreasing    |
| 1121 | 0.6137   | 0.6136            | -1.9837   | 0.0085 | Highly Likely Decreasing    |
| 1131 | 0.6192   | 0.6190            | -1.9779   | 0.0085 | Highly Likely Decreasing    |
| 1141 | 0.6247   | 0.6245            | -1.9690   | 0.0085 | Highly Likely Decreasing    |
| 1151 | 0.6301   | 0.6300            | -1.9898   | 0.0085 | Highly Likely Decreasing    |
| 1161 | 0.6356   | 0.6355            | -1.9630   | 0.0085 | Highly Likely Decreasing    |
| 1171 | 0.6411   | 0.6409            | -1.9545   | 0.0085 | Highly Likely Decreasing    |
| 1181 | 0.6466   | 0.6464            | -1.9685   | 0.0085 | Highly Likely Decreasing    |
| 1191 | 0.6521   | 0.6519            | -1.9463   | 0.0085 | Highly Likely Decreasing    |
| 1201 | 0.6575   | 0.6574            | -1.9570   | 0.0085 | Highly Likely Decreasing    |
| 1211 | 0.6630   | 0.6628            | -1.9525   | 0.0085 | Highly Likely Decreasing    |
| 1221 | 0.6685   | 0.6683            | -1.9698   | 0.0085 | Highly Likely Decreasing    |
| 1231 | 0.6740   | 0.6738            | -1.9582   | 0.0085 | Highly Likely Decreasing    |
| 1241 | 0.6795   | 0.6793            | -1.9667   | 0.0085 | Highly Likely Decreasing    |
| 1251 | 0.6849   | 0.6847            | -1.9613   | 0.0085 | Highly Likely Decreasing    |
| 1261 | 0.6904   | 0.6902            | -1.9669   | 0.0085 | Highly Likely Decreasing    |
| 1271 | 0.6959   | 0.6957            | -1.9435   | 0.0085 | Highly Likely Decreasing    |
| 1281 | 0.7014   | 0.7011            | -1.9387   | 0.0085 | Highly Likely Decreasing    |
| 1291 | 0.7068   | 0.7066            | -1.9553   | 0.0085 | Highly Likely Decreasing    |
| 1301 | 0.7123   | 0.7121            | -1.9539   | 0.0085 | Highly Likely Decreasing    |
| 1311 | 0.7178   | 0.7176            | -1.9597   | 0.0085 | Highly Likely Decreasing    |
| 1321 | 0.7233   | 0.7230            | -1.9440   | 0.0085 | Highly Likely Decreasing    |
| 1331 | 0.7288   | 0.7285            | -1.9148   | 0.0085 | Highly Likely Decreasing    |
| 1341 | 0.7342   | 0.7340            | -1.9184   | 0.0085 | Highly Likely Decreasing    |
| 1351 | 0.7397   | 0.7395            | -1.9037   | 0.0085 | Highly Likely Decreasing    |
| 1361 | 0.7452   | 0.7449            | -1.9036   | 0.0085 | Highly Likely Decreasing    |
| 1371 | 0.7507   | 0.7504            | -1.9309   | 0.0085 | Highly Likely Decreasing    |
| 1381 | 0.7562   | 0.7559            | -1.9153   | 0.0085 | Highly Likely Decreasing    |
| 1391 | 0.7616   | 0.7614            | -1.9126   | 0.0085 | Highly Likely Decreasing    |
| 1401 | 0.7671   | 0.7668            | -1.9058   | 0.0085 | Highly Likely Decreasing    |
| 1411 | 0.7726   | 0.7723            | -1.9038   | 0.0085 | Highly Likely Decreasing    |
| 1421 | 0.7781   | 0.7778            | -1.9020   | 0.0085 | Highly Likely Decreasing    |
| 1431 | 0.7836   | 0.7833            | -1.8954   | 0.0085 | Highly Likely Decreasing    |
| 1441 | 0.7890   | 0.7887            | -1.8786   | 0.0085 | Highly Likely Decreasing    |
| 1451 | 0.7945   | 0.7942            | -1.8691   | 0.0085 | Highly Likely Decreasing    |
| 1461 | 0.8000   | 0.7997            | -1.8890   | 0.0085 | Highly Likely Decreasing    |
| 1471 | 0.8055   | 0.8051            | -1.8981   | 0.0085 | Highly Likely Decreasing    |
| 1481 | 0.8110   | 0.8106            | -1.9119   | 0.0085 | Highly Likely Decreasing    |
| 1491 | 0.8164   | 0.8161            | -1.9031   | 0.0085 | Highly Likely Decreasing    |
| 1501 | 0.8219   | 0.8216            | -1.8773   | 0.0085 | Highly Likely Decreasing    |
| 1511 | 0.8274   | 0.8270            | -1.8842   | 0.0085 | Highly Likely Decreasing    |
| 1521 | 0.8329   | 0.8325            | -1.9423   | 0.0085 | Highly Likely Decreasing    |
| 1531 | 0.8384   | 0.8380            | -1.8391   | 0.0085 | Highly Likely Decreasing    |
| 1541 | 0.8438   | 0.8435            | -1.8211   | 0.0085 | Highly Likely Decreasing    |
| 1551 | 0.8493   | 0.8489            | -1.7987   | 0.0085 | Highly Likely Decreasing    |
| 1561 | 0.8548   | 0.8544            | -1.8161   | 0.0085 | Highly Likely Decreasing    |
| 1571 | 0.8603   | 0.8599            | -1.7922   | 0.0085 | Highly Likely Decreasing    |
| 1581 | 0.8658   | 0.8654            | -1.7969   | 0.0085 | Highly Likely Decreasing    |
| 1591 | 0.8712   | 0.8708            | -1.7965   | 0.0085 | Highly Likely Decreasing    |
| 1601 | 0.8767   | 0.8763            | -1.8076   | 0.0085 | Highly Likely Decreasing    |
| 1611 | 0.8822   | 0.8818            | -1.8060   | 0.0085 | Highly Likely Decreasing    |
| 1621 | 0.8877   | 0.8872            | -1.7806   | 0.0085 | Highly Likely Decreasing    |
| 1631 | 0.8932   | 0.8927            | -1.7732   | 0.0085 | Highly Likely Decreasing    |
| 1641 | 0.8986   | 0.8982            | -1.8230   | 0.0085 | Highly Likely Decreasing    |
| 1651 | 0.9041   | 0.9037            | -1.7663   | 0.0085 | Highly Likely Decreasing    |
| 1661 | 0.9096   | 0.9091            | -1.7870   | 0.0085 | Highly Likely Decreasing    |
| 1671 | 0.9151   | 0.9146            | -1.7787   | 0.0085 | Highly Likely Decreasing    |
| 1681 | 0.9205   | 0.9201            | -1.7187   | 0.0085 | Highly Likely Decreasing    |
| 1691 | 0.9260   | 0.9256            | -1.6658   | 0.0085 | Highly Likely Decreasing    |
| 1701 | 0.9315   | 0.9310            | -1.5931   | 0.0085 | Highly Likely Decreasing    |
| 1711 | 0.9370   | 0.9365            | -1.5165   | 0.0242 | Highly Likely Decreasing    |
| 1721 | 0.9425   | 0.9420            | -1.3251   | 0.0603 | Highly Likely Decreasing    |
| 1731 | 0.9479   | 0.9475            | -0.4820   | 0.7071 | As Likely as Not Decreasing |
| 1741 | 0.9534   | 0.9529            | 0.8348    | 0.1329 | Very Likely Increasing      |
| 1751 | 0.9589   | 0.9584            | 1.7088    | 0.1329 | Very Likely Increasing      |
| 1761 | 0.9644   | 0.9639            | 2.2741    | 0.0085 | Highly Likely Increasing    |
| 1771 | 0.9699   | 0.9693            | 2.4007    | 0.0085 | Highly Likely Increasing    |
| 1781 | 0.9753   | 0.9748            | 2.5843    | 0.0085 | Highly Likely Increasing    |
| 1791 | 0.9808   | 0.9803            | 2.4238    | 0.0242 | Highly Likely Increasing    |
| 1801 | 0.9863   | 0.9858            | 3.0808    | 0.0242 | Highly Likely Increasing    |
| 1811 | 0.9918   | 0.9912            | 2.0354    | 0.0603 | Highly Likely Increasing    |
| 1821 | 0.9973   | 0.9967            | 2.2726    | 0.0085 | Highly Likely Increasing    |

#### Quantile Regimes Breakdown
| regime           | p_min  | p_max  | n_quantiles | median_slope | median_slope_pct | prop_significant | modal_classification     |
| ---------------- | ------ | ------ | ----------- | ------------ | ---------------- | ---------------- | ------------------------ |
| drought_baseflow | 0.0000 | 0.5000 | 913         | -0.0203      | -2.0070          | 1.0000           | Highly Likely Decreasing |
| moderate_flow    | 0.5000 | 0.9500 | 822         | -0.0196      | -1.9392          | 0.9818           | Highly Likely Decreasing |
| flood_peaks      | 0.9500 | 1.0000 | 91          | 0.0238       | 2.4105           | 0.6593           | Highly Likely Increasing |

**Key Takeaway:** A standard overall Mann-Kendall test might register "No Trend" or a slight net decrease due to cancellation between opposite ends of the hydrograph. Quantile-Kendall explicitly uncovers the dual regime: **drought baseflow is drying up (-2%/year)** while **flood peak discharge is expanding (+3%/year)**.

---

## Case 3: Sparse Irregular Censored Groundwater

### Scientific Background
Ambient groundwater quality networks typically do not record daily or weekly data. Instead, wells are visited irregularly (8 to 18 times per year) over a 25-year period. In addition, organic contaminants and pesticides frequently fall below the analytical laboratory detection limit ($< 0.5$ μg/L).

MannKS solves this by:
1. Validating block coverage (`min_coverage=0.70`) to ensure visits span at least 70% of each annual block.
2. Standardizing irregular visit counts to explicit user-requested quantiles (`quantiles=[0.10, 0.25, 0.50, 0.75, 0.90]`).
3. Propagating left-censoring flags through the order statistics and computing robust Akritas-Theil-Sen (ATS) slopes without artificial substitution plateaus.

### Python Code
```python
# Prepare censored DataFrame
cens_df = pd.DataFrame({
    'value': visit_vals,
    'censored': visit_cens,
    'cen_type': visit_types,
})

# Quantile trend test on irregular, censored series
result3 = quantile_trend_test(
    x=cens_df,
    t=visit_dates,
    block_frequency='YE',
    quantiles=[0.10, 0.25, 0.50, 0.75, 0.90],
    min_coverage=0.70,
    min_size_per_block=8,
    slope_scaling='year',
    x_unit='ug/L',
)

plot_quantile_trend(
    result3,
    save_path='case3_sparse_irregular_qk.png',
    show_ci=True,
    x_scale='linear',
    title='Case 3: Sparse Irregular Censored Groundwater (Annual Blocks)',
)
```

### Visual Diagnostics
![Case 3: Sparse Irregular Censored Groundwater](case3_sparse_irregular_qk.png)

### Results Table
| rank | quantile | plotting_position | slope   | prop_censored | p         | classification           |
| ---- | -------- | ----------------- | ------- | ------------- | --------- | ------------------------ |
| 1    | 0.1000   | 0.1000            | -0.0662 | 0.4783        | 1.044e-07 | Highly Likely Decreasing |
| 2    | 0.2500   | 0.2500            | -0.0730 | 0.3913        | 4.684e-08 | Highly Likely Decreasing |
| 3    | 0.5000   | 0.5000            | -0.0800 | 0.3478        | 1.058e-08 | Highly Likely Decreasing |
| 4    | 0.7500   | 0.7500            | -0.0885 | 0.2174        | 1.293e-08 | Highly Likely Decreasing |
| 5    | 0.9000   | 0.9000            | -0.1015 | 0.0870        | 3.825e-09 | Highly Likely Decreasing |

**Key Takeaway:** The lower quantiles ($p = 0.10, 0.25$) are heavily censored (`prop_censored` > 0), yet MannKS accurately preserves interval bounds and detects remediation progress across all detectable percentiles.

---

## Case 4: The Sampling Frequency Drift Trap & Equalization

### The Sampling Drift Trap Explained
A notorious trap in non-parametric quantile analysis occurs when monitoring frequency increases over time (e.g., historical monthly sampling shifting to modern weekly sampling). Even if the underlying physical distribution is completely stationary:
- The expected maximum of a sample of size $N=52$ is statistically greater than the expected maximum of size $N=12$.
- The expected minimum of size $N=52$ is statistically smaller than the expected minimum of size $N=12$.

Consequently, naive order-statistic trend tests report **spurious upward trends in high quantiles** and **spurious downward trends in low quantiles**, generating false alarms of climate or environmental shifts.

MannKS automatically guards against this trap:
1. **Sampling Diagnostic:** Computes Kendall's $\tau$ and the max/min sample size ratio between blocks. If $p < 0.10$ or $N_{\max} / N_{\min} > 1.5$, a warning is raised and `result.sampling_diagnostic['flag'] == True`.
2. **Equalization (`equalize='thin'`):** Systematically sub-samples blocks to a common sample size, eliminating the sample-size expansion artifact.

### Python Comparison
```python
# Part A: Naive Raw Analysis (Suffers from Drift Artifact)
res_raw = quantile_trend_test(
    x=stationary_lognormal_values,
    t=dates,
    block_frequency='YE',
    quantiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
    slope_scaling='year',
)

# Part B: Equalized Analysis (Thinning Eliminates Drift Artifact)
res_eq = quantile_trend_test(
    x=stationary_lognormal_values,
    t=dates,
    block_frequency='YE',
    quantiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
    equalize='thin',
    random_state=42,
    slope_scaling='year',
)
```

### Visual Diagnostics
#### Case 4A: Raw Un-equalized Series (False Artifactual Trends)
![Case 4A: Sampling Frequency Drift Trap](case4a_sampling_drift_raw_qk.png)

#### Case 4B: Equalized Series via Thinning (Drift Artifact Removed)
![Case 4B: Equalized Blocks](case4b_sampling_drift_equalized_qk.png)

### Numerical Comparison
#### Raw Results (Case 4A - Severe False Positive in Tails)
| rank | quantile | slope   | p      | classification           |
| ---- | -------- | ------- | ------ | ------------------------ |
| 1    | 0.0500   | -0.0732 | 0.0086 | Highly Likely Decreasing |
| 2    | 0.1000   | -0.0640 | 0.0179 | Highly Likely Decreasing |
| 3    | 0.2500   | -0.0375 | 0.2300 | Likely Decreasing        |
| 4    | 0.5000   | -0.0367 | 0.3145 | Likely Decreasing        |
| 5    | 0.7500   | -0.0325 | 0.4555 | Likely Decreasing        |
| 6    | 0.9000   | 0.2030  | 0.1443 | Very Likely Increasing   |
| 7    | 0.9500   | 0.3344  | 0.0556 | Highly Likely Increasing |

#### Equalized Results (Case 4B - Drift Artifact Removed)
| rank | quantile | slope   | p      | classification           |
| ---- | -------- | ------- | ------ | ------------------------ |
| 1    | 0.0500   | -0.0945 | 0.0039 | Highly Likely Decreasing |
| 2    | 0.1000   | -0.0556 | 0.0350 | Highly Likely Decreasing |
| 3    | 0.2500   | -0.0405 | 0.1443 | Very Likely Decreasing   |
| 4    | 0.5000   | -0.0464 | 0.4957 | Likely Decreasing        |
| 5    | 0.7500   | -0.0378 | 0.6265 | Likely Decreasing        |
| 6    | 0.9000   | 0.1976  | 0.2843 | Likely Increasing        |
| 7    | 0.9500   | 0.3322  | 0.2300 | Likely Increasing        |

#### Diagnostic Output
- **Raw Diagnostic:** `flag = True`, ratio $N_{\max}/N_{\min} = 4.42$, $p = 2.1291e-04$
- **Equalized Diagnostic:** `flag = False`, ratio $N_{\max}/N_{\min} = 1.00$

---

## Summary of Quantile-Kendall Diagnostics

| Diagnostic / Feature | Parameter | Purpose |
| :--- | :--- | :--- |
| **Normal Z vs Linear Scale** | `x_scale='normal'` or `'linear'` | Spreads tail quantiles on probit scale ($Z = \Phi^{-1}(p)$) or uniform $[0, 1]$. |
| **Flow Duration Curve** | `show_fdc=True` | Overlays median flow/concentration across blocks on a secondary right y-axis. |
| **Log Slopes (%/year)** | `transform='log'` | Exponentiates Sen's slope as percentage change per time unit ($100 \times (e^b - 1)$). |
| **Coverage Filter** | `min_coverage=0.70` | Discards incomplete blocks where sampling span is less than 70% of block duration. |
| **Sampling Drift Guard** | `equalize='thin'` | Resamples blocks to equal size to eliminate tail order-statistic expansion artifacts. |
| **Regime Aggregator** | `summarize_quantile_regimes` | Summarizes low, mid, and high quantile trends into concise decision tables. |
