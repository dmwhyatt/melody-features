"""
This module contains a series of algorithms that may be used on the different input types
to aid in the calculation of features which use descriptive statistics.
"""
__author__ = "David Whyatt"

import numpy as np
import scipy.stats

def range_func(values) -> float:
    """Calculate range between highest and lowest values.
    
    Parameters
    ----------
    values : list or numpy.ndarray
        List or array of values to calculate range for
        
    Returns
    -------
    float
        Range between highest and lowest values
    """
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    return float(np.ptp(values))

def standard_deviation(values) -> float:
    """Calculate standard deviation of values.
    
    Parameters
    ----------
    values : list or numpy.ndarray
        List or array of values to calculate standard deviation for
        
    Returns
    -------
    float
        Standard deviation of values
    """
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    return float(np.std(values))

def shannon_entropy(values) -> float:
    """Calculate Shannon entropy of a distribution.
    
    Parameters
    ----------
    values : list or numpy.ndarray
        List or array of values to calculate entropy for
        
    Returns
    -------
    float
        Shannon entropy value
    """
    # Convert to numpy array if not already
    values = np.asarray(values)
    
    # Check if array is empty
    if values.size == 0:
        return 0.0
        
    # Get unique values and their counts
    unique, counts = np.unique(values, return_counts=True)
    
    # Calculate probabilities
    probs = counts / counts.sum()
    
    # Calculate entropy
    return -np.sum(probs * np.log2(probs))


def natural_entropy(values: list[float]) -> float:
    """Calculates the natural entropy (base-e) of a list of numbers.

    Parameters
    ----------
    values : list[float]
        List of numeric values to analyze

    Returns
    -------
    float
        The natural entropy of the values. Returns 0 for empty list.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> natural_entropy([1, 1, 2, 2])  # Equal probabilities
    0.693...
    >>> natural_entropy([1, 1, 1])  # All same value
    -0.0
    >>> natural_entropy([])  # Empty list
    0.0
    """
    if not values:
        return 0.0
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Calculate probabilities of each unique value
    _, counts = np.unique(values_array, return_counts=True)
    probabilities = counts / len(values_array)

    # Calculate entropy using the formula: -sum(p * ln(p))
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy)


def mean(values: list[float]) -> float:
    """Calculates the arithmetic mean of a list of numbers.

    Parameters
    ----------
    values : list[float]
        List of numeric values to analyze

    Returns
    -------
    float
        The arithmetic mean. Returns 0 for empty list.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> mean([1, 2, 3, 4, 5])
    3.0
    >>> mean([])  # Empty list
    0.0
    >>> mean([1.5, 2.5, 3.5])
    2.5
    """
    if not values:
        return 0.0

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    return float(np.mean(values_array))

def mode(values) -> float:
    """Find most common value.
    
    Parameters
    ----------
    values : list or numpy.ndarray
        List or array of values to find mode for
        
    Returns
    -------
    float
        Most common value
    """
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    return float(scipy.stats.mode(values, keepdims=False)[0])

def length(values: list[float]) -> float:
    """Returns the length (number of elements) of a list of numbers.

    Parameters
    ----------
    values : list[float]
        List of numeric values to analyze

    Returns
    -------
    float
        The length. Returns 0 for empty list.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> length([1, 2, 3, 4, 5])
    5.0
    >>> length([])  # Empty list
    0.0
    >>> length([1.5])  # Single element
    1.0
    """
    if not values:
        return 0.0

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    return float(len(values_array))
