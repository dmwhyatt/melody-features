"""
This module contains a series of algorithms that may be used on the different input types
to aid in the calculation of features related to correlations.
"""
__author__ = "David Whyatt"

import numpy as np
from scipy import stats, signal

def correlation(x: list[float], y: list[float]) -> float:
    """Calculates the Pearson-Bravais correlation coefficient between two lists of values.

    Parameters
    ----------
    x : list[float]
        First list of numeric values
    y : list[float] 
        Second list of numeric values. Must have same length as x.

    Returns
    -------
    float
        Correlation coefficient between -1 and 1.
        Returns None if input lists are empty or have different lengths.
        Returns 0 if there is no correlation (e.g. if one list has zero variance).

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> correlation([1, 2, 3], [2, 4, 6])  # Perfect positive correlation
    1.0
    >>> correlation([1, 2, 3], [3, 2, 1])  # Perfect negative correlation
    -1.0
    >>> correlation([1, 1, 1], [1, 2, 3])  # Zero variance in first list
    0.0
    """
    if not x or not y or len(x) != len(y):
        return None

    try:
        x_array = np.array(x, dtype=float)
        y_array = np.array(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Check for zero variance in either array
    if np.var(x_array) == 0 or np.var(y_array) == 0:
        return 0.0

    # Use numpy's built-in corrcoef function which implements Pearson correlation
    correlation_matrix = np.corrcoef(x_array, y_array)

    # corrcoef returns a 2x2 matrix, we want the off-diagonal element
    return float(correlation_matrix[0, 1])


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Calculate Spearman's rank correlation coefficient between two lists of numbers.

    Parameters
    ----------
    x : list[float]
        First list of numeric values
    y : list[float]
        Second list of numeric values

    Returns
    -------
    float
        Float value representing Spearman's correlation coefficient.
        Returns 0 if either list is empty or lists have different lengths.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> spearman_correlation([1, 2, 3], [4, 5, 6])  # Perfect monotonic relationship
    1.0
    >>> spearman_correlation([1, 2, 3], [9, 4, 1])  # Perfect negative monotonic
    -1.0
    >>> spearman_correlation([], [1, 2])  # Empty/unequal lists
    0.0
    """
    if not x or not y or len(x) != len(y):
        return 0.0

    try:
        x_array = np.array(x, dtype=float)
        y_array = np.array(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    corr, _ = stats.spearmanr(x_array, y_array)

    # Handle NaN result
    if np.isnan(corr):
        return 0.0

    return float(corr)

def kendall_tau(x: list[float], y: list[float]) -> float:
    """Calculate Kendall's tau correlation coefficient between two lists of numbers.

    Parameters
    ----------
    x : list[float]
        First list of numeric values
    y : list[float]
        Second list of numeric values

    Returns
    -------
    float
        Float value representing Kendall's tau correlation coefficient.
        Returns 0 if either list is empty or lists have different lengths.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> kendall_tau([1, 2, 3], [2, 4, 6])  # Perfect concordance
    1.0
    >>> kendall_tau([1, 2, 3], [3, 2, 1])  # Perfect discordance
    -1.0
    >>> kendall_tau([], [1, 2])  # Empty/unequal lists
    0.0
    """
    if not x or not y or len(x) != len(y):
        return 0.0

    try:
        x_array = np.array(x, dtype=float)
        y_array = np.array(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    tau, _ = stats.kendalltau(x_array, y_array)

    # Handle NaN result
    if np.isnan(tau):
        return 0.0

    return float(tau)


def cross_correlation(x: list[float], y: list[float]) -> list[float]:
    """Calculates the cross-correlation between two lists of numbers using scipy.signal.correlate.

    Parameters
    ----------
    x : list[float]
        First list of numeric values
    y : list[float]
        Second list of numeric values

    Returns
    -------
    list[float]
        List containing the cross-correlation values. Returns empty list for empty inputs.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> cross_correlation([1, 2, 3], [1, 2, 3])  # Auto-correlation
    [3.0, 8.0, 14.0, 8.0, 3.0]
    >>> cross_correlation([1, 1], [1, -1])  # Different signals
    [-1.0, 0.0, 1.0]
    >>> cross_correlation([], [1, 2])  # Empty input
    []
    """
    if not x or not y:
        return []

    try:
        x_array = np.array(x, dtype=float)
        y_array = np.array(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Calculate cross-correlation using scipy.signal.correlate
    corr = signal.correlate(x_array, y_array, mode='full')

    return corr.tolist()
