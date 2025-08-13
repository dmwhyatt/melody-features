"""
This module contains a series of algorithms that may be used on the different input types
to aid in the calculation of features related to complexity.
"""

__author__ = "David Whyatt"

from collections import Counter

import numpy as np


def yules_k(ngram_counts: list[Counter]) -> float:
    """Calculates mean Yule's K statistic over m-type n-grams.

    Parameters
    ----------
    ngram_counts : list[Counter]
        List of Counter objects containing n-gram counts for each length n

    Returns
    -------
    float
        Mean Yule's K statistic across n-gram lengths.
        Returns 0 for empty input.

    Examples
    --------
    >>> from collections import Counter
    >>> # Sample melody [60, 62, 64, 62, 60] with equal note durations, with 1-grams, 2-grams, 3-grams, and 4-grams
    >>> counts = [Counter({
    ...     (('u2', 'e'),): 2,
    ...     (('d2', 'e'),): 1,
    ...     (('d2', None),): 1,
    ...     (('u2', 'e'), ('u2', 'e')): 1,
    ...     (('u2', 'e'), ('d2', 'e')): 1,
    ...     (('d2', 'e'), ('d2', None)): 1,
    ...     (('u2', 'e'), ('u2', 'e'), ('d2', 'e')): 1,
    ...     (('u2', 'e'), ('d2', 'e'), ('d2', None)): 1,
    ...     (('u2', 'e'), ('u2', 'e'), ('d2', 'e'), ('d2', None)): 1
    ... })]
    >>> yules_k(counts)  # High K indicates more repetition
    20.0

    >>> # Empty input
    >>> yules_k([])
    0.0

    >>> # Counter with no tokens
    >>> yules_k([Counter()])
    0.0
    """
    if not ngram_counts:
        return 0.0

    k_values = []
    for counts in ngram_counts:
        # Filter out n-grams longer than 5
        filtered_counts = Counter({k: v for k, v in counts.items() if len(k) <= 5})
        if not filtered_counts:
            continue

        # Get frequency of frequencies
        freq_spec = Counter(filtered_counts.values())

        # Calculate N (total tokens)
        N = sum(filtered_counts.values())
        if N == 0:
            continue

        # Calculate sum(vm * m²) where vm is frequency of value m
        vm_m2_sum = sum(freq * (count * count) for count, freq in freq_spec.items())

        # Calculate K with scaling factor of 1000
        K = 1000 * (vm_m2_sum - N) / (N * N)
        k_values.append(K)

    return float(np.mean(k_values)) if k_values else 0.0


def simpsons_d(ngram_counts: list[Counter]) -> float:
    """Compute mean Simpson's D diversity index over m-type n-grams.

    Parameters
    ----------
    ngram_counts : list[Counter]
        List of Counter objects containing n-gram counts for each n

    Returns
    -------
    float
        Mean Simpson's D value across n-gram lengths.
        Returns 0.0 for empty input.

    Examples
    --------
    >>> from collections import Counter
    >>> # Sample melody [60, 62, 64, 62, 60] with equal note durations
    >>> counts = counts = [Counter({
    ...     (('u2', 'e'),): 2,
    ...     (('d2', 'e'),): 1,
    ...     (('d2', None),): 1,
    ...     (('u2', 'e'), ('u2', 'e')): 1,
    ...     (('u2', 'e'), ('d2', 'e')): 1,
    ...     (('d2', 'e'), ('d2', None)): 1,
    ...     (('u2', 'e'), ('u2', 'e'), ('d2', 'e')): 1,
    ...     (('u2', 'e'), ('d2', 'e'), ('d2', None)): 1,
    ...     (('u2', 'e'), ('u2', 'e'), ('d2', 'e'), ('d2', None)): 1
    ... })]
    >>> simpsons_d(counts)  # Higher D indicates less diversity
    0.022...

    >>> # Empty input
    >>> simpsons_d([])
    0.0

    >>> # Counter with no tokens
    >>> simpsons_d([Counter()])
    0.0
    """

    if not ngram_counts:
        return 0.0

    d_values = []
    for counts in ngram_counts:
        # Filter out n-grams longer than 5
        filtered_counts = Counter({k: v for k, v in counts.items() if len(k) <= 5})
        if not filtered_counts:
            continue

        # Get counts
        count_values = list(filtered_counts.values())
        N = sum(count_values)  # total tokens

        if N <= 1:
            continue

        # Calculate D using the formula: sum(n_i * (n_i - 1)) / (N * (N - 1))
        d = sum(n * (n - 1) for n in count_values) / (N * (N - 1))
        d_values.append(d)

    return float(np.mean(d_values)) if d_values else 0.0


def sichels_s(ngram_counts: list[Counter]) -> float:
    """Compute mean Sichel's S statistic over m-type n-grams.

    Parameters
    ----------
    ngram_counts : list[Counter]
        List of Counter objects containing n-gram counts for each n

    Returns
    -------
    float
        Mean Sichel's S value across n-gram lengths.
        Returns 0.0 for empty input.

    Examples
    --------
    >>> from collections import Counter
    >>> # Sample melody [60, 62, 64, 62, 60] with equal note durations
    >>> counts = counts = [Counter({
    ...     (('u2', 'e'),): 2,
    ...     (('d2', 'e'),): 1,
    ...     (('d2', None),): 1,
    ...     (('u2', 'e'), ('u2', 'e')): 1,
    ...     (('u2', 'e'), ('d2', 'e')): 1,
    ...     (('d2', 'e'), ('d2', None)): 1,
    ...     (('u2', 'e'), ('u2', 'e'), ('d2', 'e')): 1,
    ...     (('u2', 'e'), ('d2', 'e'), ('d2', None)): 1,
    ...     (('u2', 'e'), ('u2', 'e'), ('d2', 'e'), ('d2', None)): 1
    ... })]
    >>> sichels_s(counts)  # Higher S indicates more doubles
    0.111...

    >>> # Empty input
    >>> sichels_s([])
    0.0

    >>> # Counter with no tokens
    >>> sichels_s([Counter()])
    0.0
    """
    if not ngram_counts:
        return 0.0

    s_values = []
    for counts in ngram_counts:
        # Filter out n-grams longer than 5
        filtered_counts = Counter({k: v for k, v in counts.items() if len(k) <= 5})
        if not filtered_counts:
            continue

        # Count how many n-grams occur exactly twice
        doubles = sum(1 for count in filtered_counts.values() if count == 2)

        # Total number of unique n-grams
        V = len(filtered_counts)

        if V == 0:
            continue

        # Calculate S value
        s = float(doubles) / V
        s_values.append(s)

    return float(np.mean(s_values)) if s_values else 0.0


def honores_h(ngram_counts: list[Counter]) -> float:
    """Compute mean Honore's H statistic over m-type n-grams.

    Parameters
    ----------
    ngram_counts : list[Counter]
        List of Counter objects containing n-gram counts for each n

    Returns
    -------
    float
        Mean Honore's H value across n-gram lengths.
        Returns 0.0 for empty input.

    Examples
    --------
    >>> from collections import Counter
    >>> # Sample melody [60, 62, 64, 62, 60] with equal note durations
    >>> counts = counts = [Counter({
    ...     (('u2', 'e'),): 2,
    ...     (('d2', 'e'),): 1,
    ...     (('d2', None),): 1,
    ...     (('u2', 'e'), ('u2', 'e')): 1,
    ...     (('u2', 'e'), ('d2', 'e')): 1,
    ...     (('d2', 'e'), ('d2', None)): 1,
    ...     (('u2', 'e'), ('u2', 'e'), ('d2', 'e')): 1,
    ...     (('u2', 'e'), ('d2', 'e'), ('d2', None)): 1,
    ...     (('u2', 'e'), ('u2', 'e'), ('d2', 'e'), ('d2', None)): 1
    ... })]
    >>> honores_h(counts)  # Higher H indicates more unique words  # doctest: +ELLIPSIS
    2072.326...

    >>> # Empty input
    >>> honores_h([])
    0.0

    >>> # Counter with no tokens
    >>> honores_h([Counter()])
    0.0
    """
    if not ngram_counts:
        return 0.0

    h_values = []
    for counts in ngram_counts:
        # Filter out n-grams longer than 5
        filtered_counts = Counter({k: v for k, v in counts.items() if len(k) <= 5})
        if not filtered_counts:
            continue

        # Get total tokens (N)
        N = sum(filtered_counts.values())

        # Get number of hapax legomena (V1)
        V1 = sum(1 for count in filtered_counts.values() if count == 1)

        # Get total types (V)
        V = len(filtered_counts)

        # Handle edge cases
        if V == 0 or V1 == 0 or V1 == V:
            continue

        # Calculate H value
        H = 100.0 * (np.log(N) / (1.0 - (float(V1) / V)))
        h_values.append(H)

    return float(np.mean(h_values)) if h_values else 0.0


def mean_entropy(ngram_counts: list[Counter]) -> float:
    """Compute mean entropy of m-type n-gram distribution.

    Parameters
    ----------
    ngram_counts : Counter
        List of Counter objects containing n-gram counts for each n

    Returns
    -------
    float
        Mean normalized entropy value across n-gram lengths.
        Returns 0.0 for empty input.

    Examples
    --------
    >>> from collections import Counter
    >>> # Sample melody [60, 62, 64, 62, 60] with equal note durations
    >>> counts = counts = [Counter({
    ...     (('u2', 'e'),): 2,
    ...     (('d2', 'e'),): 1,
    ...     (('d2', None),): 1,
    ...     (('u2', 'e'), ('u2', 'e')): 1,
    ...     (('u2', 'e'), ('d2', 'e')): 1,
    ...     (('d2', 'e'), ('d2', None)): 1,
    ...     (('u2', 'e'), ('u2', 'e'), ('d2', 'e')): 1,
    ...     (('u2', 'e'), ('d2', 'e'), ('d2', None)): 1,
    ...     (('u2', 'e'), ('u2', 'e'), ('d2', 'e'), ('d2', None)): 1
    ... })]
    >>> mean_entropy(counts)  # Higher entropy indicates more randomness
    0.939...

    >>> # Empty input
    >>> mean_entropy([])
    0.0

    >>> # Counter with no tokens
    >>> mean_entropy([Counter()])
    0.0
    """
    if not ngram_counts:
        return 0.0

    entropy_values = []
    for counts in ngram_counts:
        # Filter out n-grams longer than 5
        filtered_counts = Counter({k: v for k, v in counts.items() if len(k) <= 5})
        if not filtered_counts:
            continue

        # Get total tokens
        N = sum(filtered_counts.values())

        if N <= 1:
            continue

        # Calculate probabilities
        probs = [count / N for count in filtered_counts.values()]

        # Calculate entropy
        H = -np.sum(probs * np.log2(probs))

        # Normalize by log(N)
        H_norm = H / np.log2(N)
        entropy_values.append(H_norm)

    return float(np.mean(entropy_values)) if entropy_values else 0.0


def mean_productivity(ngram_counts: list[Counter]) -> float:
    """Compute mean productivity of m-type n-gram distribution.

    Parameters
    ----------
    ngram_counts : Counter
        List of Counter objects containing n-gram counts for each n

    Returns
    -------
    float
        Mean productivity value across n-gram lengths.
        Returns 0.0 for empty input.

    Examples
    --------
    >>> from collections import Counter
    >>> # Sample melody [60, 62, 64, 62, 60] with equal note durations
    >>> counts = counts = [Counter({
    ...     (('u2', 'e'),): 2,
    ...     (('d2', 'e'),): 1,
    ...     (('d2', None),): 1,
    ...     (('u2', 'e'), ('u2', 'e')): 1,
    ...     (('u2', 'e'), ('d2', 'e')): 1,
    ...     (('d2', 'e'), ('d2', None)): 1,
    ...     (('u2', 'e'), ('u2', 'e'), ('d2', 'e')): 1,
    ...     (('u2', 'e'), ('d2', 'e'), ('d2', None)): 1,
    ...     (('u2', 'e'), ('u2', 'e'), ('d2', 'e'), ('d2', None)): 1
    ... })]
    >>> mean_productivity(counts)  # Higher productivity indicates more hapax legomena
    0.8

    >>> # Empty input
    >>> mean_productivity([])
    0.0

    >>> # Counter with no tokens
    >>> mean_productivity([Counter()])
    0.0
    """
    if not ngram_counts:
        return 0.0

    productivity_values = []
    for counts in ngram_counts:
        # Filter out n-grams longer than 5
        filtered_counts = Counter({k: v for k, v in counts.items() if len(k) <= 5})
        if not filtered_counts:
            continue

        # Get total tokens
        N = sum(filtered_counts.values())

        if N == 0:
            continue

        # Count hapax legomena (types occurring once)
        V1N = sum(1 for count in filtered_counts.values() if count == 1)

        # Calculate productivity
        prod = V1N / N
        productivity_values.append(prod)

    return float(np.mean(productivity_values)) if productivity_values else 0.0


def repetition_rate(values) -> float:
    """Calculate rate of repetition in a sequence.

    Parameters
    ----------
    values : list or numpy.ndarray
        List or array of values to analyze

    Returns
    -------
    float
        Rate of repetition (0.0-1.0)
    """
    # Convert to numpy array if not already
    values = np.asarray(values)

    # Check if array is empty
    if values.size == 0:
        return 0.0

    # Count unique values
    unique_values = np.unique(values)

    # Calculate repetition rate
    return 1.0 - (len(unique_values) / values.size)


def repetition_count(values: list[float]) -> list[float]:
    """Counts the number of times each value repeats in the list.

    Parameters
    ----------
    values : list[float]
        List of numeric values to count repetitions

    Returns
    -------
    dict[int, float]
        Dictionary mapping values to their repetition counts for values that appear more than once.
        Returns empty dictionary for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> repetition_count([1, 2, 2, 3, 3, 3])  # Multiple repeats
    {2: 2.0, 3: 3.0}
    >>> repetition_count([1, 2, 3])  # No repeats
    {}
    >>> repetition_count([])  # Empty list
    {}
    """
    if not values:
        return {}

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Get counts of each unique value
    unique, counts = np.unique(values_array, return_counts=True)

    # Create list of indices where counts > 1
    repeat_indices = [i for i, count in enumerate(counts) if count > 1]

    # Return dictionary mapping values to their counts
    return {int(unique[i]): float(counts[i]) for i in repeat_indices}


def consecutive_repetition_count(values: list[float]) -> dict[float, float]:
    """Counts the number of times each value appears consecutively in the list.

    Parameters
    ----------
    values : list[float]
        List of numeric values to check for consecutive repetitions

    Returns
    -------
    dict[float, float]
        Dictionary mapping values to their consecutive repetition counts for values that appear
        consecutively more than once. Returns empty dictionary for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> consecutive_repetition_count([1, 1, 2, 2, 2, 3])  # Multiple consecutive repeats
    {1.0: 2.0, 2.0: 3.0}
    >>> consecutive_repetition_count([1, 2, 1, 2])  # No consecutive repeats
    {}
    >>> consecutive_repetition_count([])  # Empty list
    {}
    """
    if not values:
        return {}

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    repetitions = {}
    current_value = values_array[0]
    current_count = 1

    # Iterate through values starting from second element
    for val in values_array[1:]:
        if val == current_value:
            current_count += 1
        else:
            # When value changes, record count if > 1 and reset
            if current_count > 1:
                repetitions[float(current_value)] = float(current_count)
            current_value = val
            current_count = 1

    # Check final run
    if current_count > 1:
        repetitions[float(current_value)] = float(current_count)

    return repetitions


def consecutive_fifths(values: list[float]) -> dict[float, int]:
    """Checks the input list for consecutive values separated by perfect fifths.

    Parameters
    ----------
    values : list[float]
        List of numeric values to analyze

    Returns
    -------
    dict[float, int]
        Dictionary mapping starting values to their consecutive fifths counts.
        Returns empty dictionary for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> consecutive_fifths([1, 8, 15, 22])  # Consecutive fifths
    {1.0: 3}
    >>> consecutive_fifths([1, 2, 3, 4])  # No consecutive fifths
    {}
    >>> consecutive_fifths([])  # Empty list
    {}
    """
    # return sum([(j - i) % 12 == 7 for i, j in zip(values, values[1:])])
    if not values:
        return {}

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    fifths = {}
    current_value = values_array[0]
    current_count = 0

    # Iterate through values starting from second element
    for val in values_array[1:]:
        if (val - current_value) % 12 == 7:
            current_count += 1
            current_value = val
        else:
            # When value changes, record count if > 0 and reset
            if current_count > 0:
                fifths[float(values_array[0])] = current_count
            current_value = val
            current_count = 0

    # Check final run
    if current_count > 0:
        fifths[float(values_array[0])] = current_count

    return fifths
