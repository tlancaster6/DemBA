import pandas as pd
import numpy as np
from typing import Tuple

def phi_coefficient(series1: pd.Series, series2: pd.Series) -> float:
    """
    Calculate the phi coefficient between two binary pandas Series.

    The phi coefficient is equivalent to the Pearson correlation coefficient
    for binary variables. It ranges from -1 to 1, where:
    - 1 indicates perfect positive association
    - 0 indicates no association
    - -1 indicates perfect negative association

    Args:
        series1, series2: pandas Series with boolean values (True/False)

    Returns:
        float: phi coefficient
    """
    # Ensure series have the same length
    if len(series1) != len(series2):
        raise ValueError("Series must have the same length")

    # Convert to boolean if not already
    s1 = series1.astype(bool)
    s2 = series2.astype(bool)

    # Create contingency table
    # True=1, False=0
    a = (s1 & s2).sum()  # both True
    b = (s1 & ~s2).sum()  # s1 True, s2 False
    c = (~s1 & s2).sum()  # s1 False, s2 True
    d = (~s1 & ~s2).sum()  # both False

    # Calculate phi coefficient
    numerator = (a * d) - (b * c)
    denominator = np.sqrt(float(a + b) * float(c + d) * float(a + c) * float(b + d))

    # Handle edge case where denominator is 0
    if denominator == 0:
        return 0.0

    return numerator / denominator


def jaccard_index(series1: pd.Series, series2: pd.Series) -> float:
    """
    Calculate the Jaccard index between two binary pandas Series.

    The Jaccard index measures similarity between finite sample sets,
    defined as the size of the intersection divided by the size of the union.
    It ranges from 0 to 1, where:
    - 1 indicates perfect similarity (identical sets)
    - 0 indicates no similarity (no overlap)

    Args:
        series1, series2: pandas Series with boolean values (True/False)

    Returns:
        float: Jaccard index
    """
    # Ensure series have the same length
    if len(series1) != len(series2):
        raise ValueError("Series must have the same length")

    # Convert to boolean if not already
    s1 = series1.astype(bool)
    s2 = series2.astype(bool)

    # Calculate intersection and union
    intersection = (s1 & s2).sum()
    union = (s1 | s2).sum()

    # Handle edge case where union is 0 (both series are all False)
    if union == 0:
        return 1.0  # Perfect similarity when both are empty sets

    return intersection / union
