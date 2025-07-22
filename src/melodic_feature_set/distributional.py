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

def standardize_distribution(values: list[float]) -> list[float]:
    """Converts a list of numbers to a normal distribution with mean 0 and std dev 1.

    Parameters
    ----------
    values : list[float]
        List of numeric values to normalize

    Returns
    -------
    list[float]
        List of normalized values with mean 0 and standard deviation 1.
        Returns empty list for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float
    ValueError
        If input has zero standard deviation

    Examples
    --------
    >>> standardize_distribution([1, 2, 3])
    [-1.224..., 0.0, 1.224...]
    >>> standardize_distribution([]) # Empty list
    []
    >>> standardize_distribution([10, 20, 30])
    [-1.224..., 0.0, 1.224...]
    """
    if not values:
        return []

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    mean_val = np.mean(values_array)
    std = np.std(values_array)

    if std == 0:
        raise ValueError("Cannot normalize - standard deviation is zero")

    normalized = (values_array - mean_val) / std
    return [float(x) for x in normalized]

def normalize_distribution(values: list[float]) -> tuple[list[float], float, float]:
    """Normalizes a list of numbers to a range between 0 and 1 using min-max normalization.

    Parameters
    ----------
    values : list[float]
        List of numeric values to normalize

    Returns
    -------
    tuple[list[float], float, float]
        Tuple containing:
        - List of normalized values between 0 and 1
        - Mean of normalized values
        - Standard deviation of normalized values
        Returns ([], 0.0, 0.0) for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float
    ValueError
        If all input values are identical

    Examples
    --------
    >>> normalize_distribution([1, 2, 3])  # doctest: +ELLIPSIS
    ([0.0, 0.5, 1.0], 0.5, 0.408...)
    >>> normalize_distribution([])  # Empty list
    ([], 0.0, 0.0)
    >>> normalize_distribution([10, 20, 30])  # doctest: +ELLIPSIS
    ([0.0, 0.5, 1.0], 0.5, 0.408...)
    """
    if not values:
        return [], 0.0, 0.0

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    min_val = np.min(values_array)
    max_val = np.max(values_array)

    if min_val == max_val:
        raise ValueError("Cannot normalize - all values are identical")

    normalized = (values_array - min_val) / (max_val - min_val)
    mean_val = np.mean(normalized)
    std_dev = np.std(normalized)

    return [float(x) for x in normalized], float(mean_val), float(std_dev)

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
    proportions = counts * (1.0/len(values_array))
    return {float(u): float(p) for u, p in zip(unique, proportions)}
