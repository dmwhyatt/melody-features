"""
This module contains a series of algorithms that may be used on the different input types
to aid in the calculation of features related to distributional properties.
"""

__author__ = "David Whyatt"

import numpy as np
from scipy import stats


def histogram_bins(values, num_bins: int) -> dict[str, int]:
    """Places data into histogram bins and counts occurrences in each bin.

    Parameters
    ----------
    values : list or numpy.ndarray
        List or array of numeric values to bin
    num_bins : int
        Number of equal-width bins to create

    Returns
    -------
    dict[str, int]
        Dictionary mapping bin range strings (e.g. '1.00-2.00') to counts.
        Returns empty dictionary for empty input.

    Raises
    ------
    ValueError
        If num_bins is less than 1
    """
    # Convert to numpy array if not already
    values = np.asarray(values)

    # Check if array is empty
    if values.size == 0:
        return {}

    if num_bins < 1:
        raise ValueError("Number of bins must be at least 1")

    # Calculate histogram
    counts, bin_edges = np.histogram(values, bins=num_bins)

    # Create dictionary with formatted bin ranges as keys
    result = {}
    for i, (count, edge) in enumerate(zip(counts, bin_edges)):
        bin_label = f"{edge:.2f}-{bin_edges[i+1]:.2f}"
        result[bin_label] = int(count)  # Convert count to integer

    return result





def kurtosis(values) -> float:
    """Calculate kurtosis of values.

    Parameters
    ----------
    values : list or numpy.ndarray
        List or array of values to analyze

    Returns
    -------
    float
        Kurtosis value
    """
    # Convert to numpy array if not already
    values = np.asarray(values)

    # Check if array is empty
    if values.size == 0:
        return 0.0

    # Calculate kurtosis using scipy
    return float(stats.kurtosis(values))


def skew(values) -> float:
    """Calculate skewness of values.

    Parameters
    ----------
    values : list or numpy.ndarray
        List or array of values to analyze

    Returns
    -------
    float
        Skewness value
    """
    # Convert to numpy array if not already
    values = np.asarray(values)

    # Check if array is empty
    if values.size == 0:
        return 0.0

    # Check if there are at least 2 unique values
    if np.unique(values).size < 2:
        return 0.0

    # Calculate skewness using scipy
    return float(stats.skew(values, bias=False))


def distribution_proportions(values: list[float]) -> dict[float, float]:
    """Calculates the proportion of each unique value in a list of numbers.

    Parameters
    ----------
    values : list[float]
        List of numeric values

    Returns
    -------
    dict[float, float]
        Dictionary mapping unique values to their proportions.
        Returns empty dict for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> distribution_proportions([1, 2, 2, 3])
    {1.0: 0.25, 2.0: 0.5, 3.0: 0.25}
    >>> distribution_proportions([])  # Empty list
    {}
    >>> distribution_proportions([1.5, 1.5, 2.5])
    {1.5: 0.6666666666666666, 2.5: 0.3333333333333333}
    """
    if not values:
        return {}
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Calculate frequencies of each unique value
    unique, counts = np.unique(values_array, return_counts=True)

    # Calculate proportions
    proportions = counts * (1.0 / len(values_array))
    return {float(u): float(p) for u, p in zip(unique, proportions)}
