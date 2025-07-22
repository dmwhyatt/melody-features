"""
This module contains a series of algorithms that may be used on the different input types
to aid in the calculation of features related to similarity.
"""
__author__ = "David Whyatt"

import numpy as np

def diffexp(melody1: list[float], melody2: list[float]) -> float:
    """Calculates the differential expression score between two melodies
    based on their pitch intervals.
    
    Implements σ(μ₁,μ₂) = e^(-Δp/(N-1)) where Δp is the L1 norm (Manhattan distance) 
    between the pitch interval vectors of the two melodies.

    Parameters
    ----------
    melody1 : list[float]
        First list of numeric pitch values
    melody2 : list[float]
        Second list of numeric pitch values

    Returns
    -------
    float
        Value representing the differential expression score.
        Returns 0.0 if either melody has fewer than 2 notes (no intervals possible).

    Raises
    ------
    TypeError
        If inputs contain non-numeric values

    Examples
    --------
    >>> diffexp([60, 62, 64], [60, 62, 64])  # Identical melodies
    1.0
    >>> diffexp([60, 62, 64], [60, 63, 65])  # Similar contour, different intervals
    0.606...
    >>> diffexp([60], [60, 62])  # Too short
    0.0
    """
    # Need at least 2 notes to form intervals
    if len(melody1) < 2 or len(melody2) < 2:
        return 0.0

    try:
        # Convert melodies to numpy arrays
        m1 = np.array(melody1, dtype=float)
        m2 = np.array(melody2, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Melody inputs must contain numeric values") from exc

    # Calculate pitch intervals (differences between consecutive notes)
    intervals1 = np.diff(m1)
    intervals2 = np.diff(m2)

    # If interval vectors have different lengths, pad shorter one with zeros
    if len(intervals1) != len(intervals2):
        max_len = max(len(intervals1), len(intervals2))
        intervals1 = np.pad(intervals1, (0, max_len - len(intervals1)))
        intervals2 = np.pad(intervals2, (0, max_len - len(intervals2)))

    # Calculate Δp as L1 norm (sum of absolute differences) between interval vectors
    delta_p = np.sum(np.abs(intervals1 - intervals2))

    # n is the length of the longer melody
    n = max(len(melody1), len(melody2))

    # Calculate final score
    score = np.exp(-delta_p / (n - 1))

    return float(score)

def diff(melody1: list[float], melody2: list[float]) -> float:
    """Calculates the differential score between two melodies based on their pitch intervals.
    
    Implements σ(μ₁,μ₂) = 1 - Δp/((N-1)Δp∞) where:
    - Δp is the L1 norm (Manhattan distance) between the pitch interval vectors
    - Δp∞ is the maximum absolute interval difference across both melodies
    - N is the length of the longer melody

    Parameters
    ----------
    melody1 : list[float]
        First list of numeric pitch values
    melody2 : list[float]
        Second list of numeric pitch values

    Returns
    -------
    float
        Value representing the differential score.
        Returns 0.0 if either melody has fewer than 2 notes (no intervals possible).

    Raises
    ------
    TypeError
        If inputs contain non-numeric values
    ValueError
        If Δp∞ is zero (no pitch differences between melodies)

    Examples
    --------
    >>> diff([60, 62, 64], [60, 62, 64])  # Identical melodies
    1.0
    >>> diff([60, 62, 64], [60, 63, 65])  # Similar contour, different intervals
    0.8333...
    >>> diff([60], [60, 62])  # Too short
    0.0
    """
    # Need at least 2 notes to form intervals
    if len(melody1) < 2 or len(melody2) < 2:
        return 0.0

    try:
        # Convert melodies to numpy arrays
        m1 = np.array(melody1, dtype=float)
        m2 = np.array(melody2, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Melody inputs must contain numeric values") from exc

    # Calculate pitch intervals (differences between consecutive notes)
    intervals1 = np.diff(m1)
    intervals2 = np.diff(m2)

    # If interval vectors have different lengths, pad shorter one with zeros
    if len(intervals1) != len(intervals2):
        max_len = max(len(intervals1), len(intervals2))
        intervals1 = np.pad(intervals1, (0, max_len - len(intervals1)))
        intervals2 = np.pad(intervals2, (0, max_len - len(intervals2)))

    # Calculate Δp as L1 norm (sum of absolute differences) between interval vectors
    delta_p = np.sum(np.abs(intervals1 - intervals2))

    # Calculate Δp∞ as max of absolute intervals across both melodies
    delta_p_inf = max(np.max(np.abs(intervals1)), np.max(np.abs(intervals2)))

    if delta_p_inf == 0:
        raise ValueError("Cannot calculate diff score - no pitch differences between melodies")

    # n is the length of the longer melody
    n = max(len(melody1), len(melody2))

    # Calculate final score
    score = 1 - (delta_p / ((n - 1) * delta_p_inf))

    return float(score)


def edit_distance(list1: list[float], list2: list[float], insertion_cost: float=1,
                deletion_cost: float=1, substitution_cost: float=1) -> float:
    """Calculates the edit distance (Levenshtein distance) between two lists of numbers.

    Parameters
    ----------
    list1 : list[float]
        First list of numbers
    list2 : list[float]
        Second list of numbers
    insertion_cost : float, optional
        Cost of inserting an element, by default 1
    deletion_cost : float, optional
        Cost of deleting an element, by default 1
    substitution_cost : float, optional
        Cost of substituting an element, by default 1

    Returns
    -------
    float
        Weighted edit distance between the two lists

    Examples
    --------
    >>> edit_distance([1, 2, 3], [1, 2, 3])  # Identical lists
    0.0
    >>> edit_distance([1, 2], [1, 2, 3])  # One insertion
    1.0
    >>> edit_distance([1, 2, 3], [1, 4, 3])  # One substitution
    1.0
    """
    len1, len2 = len(list1), len(list2)

    # Create a matrix to store distances
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Initialize the matrix
    for i in range(len1 + 1):
        dp[i][0] = i * deletion_cost
    for j in range(len2 + 1):
        dp[0][j] = j * insertion_cost

    # Compute the edit distance
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + deletion_cost,     # Deletion
                              dp[i][j - 1] + insertion_cost,      # Insertion
                              dp[i - 1][j - 1] + substitution_cost) # Substitution

    return float(dp[len1][len2])
