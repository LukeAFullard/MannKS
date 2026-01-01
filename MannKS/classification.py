"""
This module provides functionality to classify trend analysis results into
descriptive categories based on confidence levels.
"""

DEFAULT_CATEGORY_MAP = {
    0.95: "Highly Likely",
    0.90: "Very Likely",
    0.67: "Likely",
    0.0:  "As Likely as Not"
}

import numpy as np
from typing import Optional, Dict, NamedTuple

def classify_trend(result: NamedTuple, category_map: Optional[Dict[float, str]] = None) -> str:
    """
    Classifies a trend result into a descriptive category.

    This function takes the namedtuple result from `original_test` or
    `seasonal_test` and classifies the trend based on its statistical
    significance and confidence. It uses a default, IPCC-style mapping for
    confidence levels, but a custom mapping can be provided.

    Args:
        result: The namedtuple returned by a trend test function.
        category_map (dict, optional): A dictionary mapping confidence
            thresholds (float) to descriptive category labels (str).
            If None, a default IPCC-style map is used.

    Returns:
        str: A string describing the trend category (e.g.,
             "Highly Likely Increasing", "No Trend").
    """
    if np.isnan(result.C):
        return "Insufficient Data"

    if category_map is None:
        category_map = DEFAULT_CATEGORY_MAP

    confidence = result.C
    # Capitalize the first letter of the trend direction ('increasing' -> 'Increasing')
    direction = result.trend.capitalize()

    # Sort thresholds in descending order to find the highest category met
    sorted_thresholds = sorted(category_map.keys(), reverse=True)

    category = ""
    for threshold in sorted_thresholds:
        if confidence >= threshold:
            category = category_map[threshold]
            break

    return f"{category} {direction}"
