# A Comprehensive Guide to Trend Classification

The `MannKS` package translates a trend test result into a short, human-readable label such as "Very Likely Increasing". This guide explains how the label is built, what the categories mean, their uses and limitations, and how to customise them.

The classification is performed by the `classify_trend` function, which is called automatically by `trend_test` and `seasonal_trend_test` (field `classification`).

---

### How Classification Works

Two pieces of information from the test are combined:

1.  **Direction** (`trend`): taken from the sign of the Mann-Kendall statistic (S, or equivalently Z): `'increasing'`, `'decreasing'`, or `'indeterminate'` when Z = 0.

2.  **Confidence in that direction** (`C`):
    -   **What it is:** `C = 1 - p/2`, where `p` is the two-sided p-value of the Mann-Kendall test. It is the confidence that the trend goes in the reported direction, and ranges from 0.5 (no evidence either way) to 1. This is the "continuous confidence" measure of LWP-TRENDS (Fraser & Whitehead 2022).
    -   **Related output:** `Cd` is the confidence that the trend is *decreasing* (`Cd = C` for a decreasing trend and `1 - C` for an increasing one).

The strength word comes from `C` and the direction word from `trend`. The Sen's slope is not used.

---

### The Default Confidence Categories

The defaults follow the IPCC-style "Direction" categories of LWP-TRENDS (`AssignConfCat`). Classes are right-closed, exactly as in LWP-TRENDS: a value on a boundary belongs to the lower class.

| Confidence (`C`) | Equivalent p-value | Classification |
| :--------------- | :----------------- | :------------- |
| `C > 0.95`       | `p < 0.10`         | **Highly Likely** Increasing/Decreasing |
| `0.90 < C <= 0.95` | `0.10 <= p < 0.20` | **Very Likely** Increasing/Decreasing |
| `0.67 < C <= 0.90` | `0.20 <= p < 0.66` | **Likely** Increasing/Decreasing |
| `C <= 0.67`      | `p >= 0.66`        | **As Likely as Not** Increasing/Decreasing |

A test that is significant at the default `alpha = 0.05` has `C > 0.975` and is therefore "Highly Likely".

With `continuous_confidence=False`, the classical behaviour is used instead: the result is "Increasing" or "Decreasing" only if the test is significant at `alpha`, and "No Trend" otherwise.

See **[Example 16: Interpreting the Full Output](../16_Interpreting_Full_Output/README.md)** and **[Example 27](../27_Continuous_Confidence_Comparison/README.md)** for practical demonstrations.

### Usefulness and Limitations of Classification

-   **Usefulness:** Classification is very useful for **summarisation**. When you analyse hundreds of sites, a table of categories gives an immediate overview, and it provides a standard vocabulary for reporting.
-   **Limitations:** A category is a **simplification**. A p-value of `0.099` ("Highly Likely") and `0.101` ("Very Likely") are statistically almost identical but fall into different categories. **Never rely only on the category**; also look at `C`, the p-value and the slope with its confidence interval.

---

### Customising the Classification

Use the `category_map` parameter: a dictionary whose **keys are confidence thresholds** (floats) and whose **values are labels**. The direction word is appended automatically.

For example, a simpler three-level scheme:
```python
my_map = {
    0.975: "Significant",   # C > 0.975, i.e. p < 0.05
    0.95: "Suggestive",     # 0.95 < C <= 0.975
    0.0: "Indeterminate",   # everything else (catch-all)
}
mk.trend_test(x, t, category_map=my_map)   # e.g. "Suggestive Increasing"
```

**Rules for custom maps:**
1.  **Keys are confidence levels (`C`)**, between 0 and 1.
2.  **Highest threshold exceeded wins.** A label applies when `C` is *strictly greater* than its threshold (as in LWP-TRENDS); the lowest threshold is inclusive.
3.  **Include a catch-all** (e.g. `0.0: "Indeterminate"`), so that every result receives a label.

#### Reproducing the LWP-TRENDS "Decrease" categories

LWP-TRENDS can also report the confidence that a trend is decreasing, `Cd`, in five classes ("Very likely increasing" … "Very likely decreasing", with "Low confidence" in between). The same classes follow from `C` and the direction:

```python
lwp_like = {0.90: "Very Likely", 0.67: "Likely", 0.0: "Low confidence"}
```

("Low confidence Increasing/Decreasing" corresponds to LWP-TRENDS' "Low confidence". The classes agree except for values lying exactly on a class boundary.)

See **[Example 18](../18_Interpreting_Confidence_and_Classification/README.md)** for a hands-on guide to creating and using custom maps.
