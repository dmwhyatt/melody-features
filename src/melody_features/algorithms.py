"""
This module contains a series of algorithms that may be called on different input types
to calculate features. These algorithms are rather more specific than the others implemented
in the rest of the repository, and are placed here for clarity and organization,
since they do not group well with the others.
"""

__author__ = "David Whyatt"

from typing import Optional

import numpy as np


def ratio(x: list[float], y: list[float]) -> list[float]:
    """Calculates the ratio between corresponding elements in two lists.

    Parameters
    ----------
    x : list[float]
        First list of numeric values
    y : list[float]
        Second list of numeric values. Must have same length as x.

    Returns
    -------
    list[float]
        List containing ratios of corresponding elements (x[i]/y[i]).
        Returns empty list if input lists are empty or have different lengths.
        Returns None for any division by zero.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> ratio([2, 4, 6], [1, 2, 3])  # Simple ratios
    [2.0, 2.0, 2.0]
    >>> ratio([1, 2], [0, 2])  # Division by zero
    [None, 1.0]
    >>> ratio([], [1, 2])  # Empty/unequal lists
    []
    """
    if not x or not y or len(x) != len(y):
        return []

    try:
        x_array = np.array(x, dtype=float)
        y_array = np.array(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Handle division by zero by converting to None
    ratios = []
    for _, (x_val, y_val) in enumerate(zip(x_array, y_array)):
        if y_val == 0 or x_val == 0:
            ratios.append(None)
        else:
            ratios.append(float(x_val / y_val))

    return ratios


def rank_values(values: list[float], descending: bool = False) -> list[float]:
    """Ranks the input values from 1 to n. Ties get the same rank.

    Parameters
    ----------
    values : list[float]
        List of numeric values to rank
    descending : bool, optional
        If True, highest value gets rank 1.
        If False, lowest value gets rank 1.

    Returns
    -------
    list[float]
        List of ranks corresponding to the input values.
        Returns empty list for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> rank_values([3, 1, 4, 1, 5])  # Ascending order
    [2.0, 1.0, 3.0, 1.0, 4.0]
    >>> rank_values([3, 1, 4, 1, 5], descending=True)  # Descending order
    [3.0, 4.0, 2.0, 4.0, 1.0]
    >>> rank_values([])  # Empty input
    []
    """
    if not values:
        return []
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Get sorting order based on descending flag
    order = -1 if descending else 1

    # Sort values and get unique values
    sorted_vals = np.sort(order * values_array)
    unique_vals = np.unique(sorted_vals)

    # Create rank mapping for each unique value
    rank_map = {}
    for i, val in enumerate(unique_vals, 1):
        rank_map[val] = float(i)

    # Map each value to its rank
    ranks = [rank_map[order * val] for val in values_array]

    return ranks


def modulo_twelve(values: list[float]) -> list[float]:
    """Takes modulo 12 of each number in the input list.

    Parameters
    ----------
    values : list[float]
        List of numeric values

    Returns
    -------
    list[float]
        List with modulo 12 of each value. Returns empty list for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> modulo_twelve([14, 25, 36])  # Values > 12
    [2.0, 1.0, 0.0]
    >>> modulo_twelve([5, 7, 11])  # Values < 12
    [5.0, 7.0, 11.0]
    >>> modulo_twelve([])  # Empty input
    []
    """
    if not values:
        return []

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    return [float(x) for x in values_array % 12]


def absolute_values(values: list[float]) -> list[float]:
    """Returns a list with absolute values of all input numbers.

    Parameters
    ----------
    values : list[float]
        List of numeric values

    Returns
    -------
    list[float]
        List containing absolute values of input numbers. Returns empty list for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> absolute_values([-1, 2, -3])  # Mixed signs
    [1.0, 2.0, 3.0]
    >>> absolute_values([0, -0.5, 1.5])  # Decimal values
    [0.0, 0.5, 1.5]
    >>> absolute_values([])  # Empty input
    []
    """
    if not values:
        return []

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    return [float(x) for x in np.abs(values_array)]


def limit(values: list[float], x: float = None, y: float = None) -> list[float]:
    """Removes values larger than x or smaller than y from the input list.

    Parameters
    ----------
    values : list[float]
        List of numeric values
    x : float, optional
        Upper limit (inclusive). If None, no upper limit is applied.
    y : float, optional
        Lower limit (inclusive). If None, no lower limit is applied.

    Returns
    -------
    list[float]
        List containing only values within the specified limits. Returns empty list for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> limit([1, 2, 3, 4, 5], x=3)  # Upper limit only
    [1.0, 2.0, 3.0]
    >>> limit([1, 2, 3, 4, 5], y=3)  # Lower limit only
    [3.0, 4.0, 5.0]
    >>> limit([1, 2, 3, 4, 5], x=4, y=2)  # Both limits
    [2.0, 3.0, 4.0]
    """
    if not values:
        return []

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    mask = np.ones(len(values_array), dtype=bool)

    if x is not None:
        mask &= values_array <= x

    if y is not None:
        mask &= values_array >= y

    return [float(x) for x in values_array[mask]]


def nine_percent_significant_values(
    values: list[float], threshold: float = 0.09
) -> list[float]:
    """Returns values that appear more than a given proportion of times in the input list.

    Parameters
    ----------
    values : list[float]
        List of numeric values to analyze
    threshold : float, optional
        Minimum proportion (between 0 and 1) required for a value to be considered significant.
        Default is 0.09 (9%)

    Returns
    -------
    list[float]
        List of values that appear more than the threshold proportion of times.
        Returns empty list for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> nine_percent_significant_values([1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # >9% threshold
    [1.0]
    >>> nine_percent_significant_values([1, 1, 2, 2, 3, 3], threshold=0.3)  # 30% threshold
    [1.0, 2.0, 3.0]
    >>> nine_percent_significant_values([])  # Empty input
    []
    """
    if not values:
        return []

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Get counts of unique values
    unique, counts = np.unique(values_array, return_counts=True)

    # Calculate proportions
    proportions = counts / len(values_array)

    # Get values that exceed threshold
    significant = [
        float(val) for val, prop in zip(unique, proportions) if prop >= threshold
    ]

    return significant


def circle_of_fifths(pitches: list[float], counts: list[float]) -> dict[int, float]:
    """Reorganizes two lists of pitches and counts according to the circle of fifths pattern.

    Parameters
    ----------
    pitches : list[float]
        List of pitch classes (0-12)
    counts : list[float]
        List of counts corresponding to each pitch class

    Returns
    -------
    dict[int, float]
        Dictionary with counts mapped to pitches reordered according to circle of fifths pattern
        (0,7,2,9,4,11,6,1,8,3,10,5). Empty input returns empty dictionary.

    Raises
    ------
    ValueError
        If any pitch class is not between 0 and 12
        If input lists have different lengths

    Examples
    --------
    >>> circle_of_fifths([0, 4, 7, 9], [1, 2, 3, 2])  # Subset of pitches
    {0: 1.0, 7: 3.0, 9: 2.0, 4: 2.0}
    >>> circle_of_fifths([], [])  # Empty input
    {}
    """
    if not pitches or not counts:
        return {}

    if len(pitches) != len(counts):
        raise ValueError("Input lists must have same length")

    # Validate pitch values
    for pitch in pitches:
        if pitch < 0 or pitch > 12:
            raise ValueError("Pitch classes must be between 0 and 12")

    # Create initial dictionary from inputs
    pitch_counts = {int(p): float(c) for p, c in zip(pitches, counts)}

    # Define circle of fifths order
    fifths_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

    # Create new dictionary with circle of fifths ordering
    result = {}
    for pitch in fifths_order:
        if pitch in pitch_counts:
            result[pitch] = pitch_counts[pitch]

    return result


def contour_extrema(values: list[float]) -> list[tuple[int, float]]:
    """Finds all contour extremum notes in a sequence.

    Identifies:
    - First and last notes
    - Local maxima where surrounding notes are lower
    - Local minima where surrounding notes are higher
    - Handles plateaus by looking at extended context

    Parameters
    ----------
    values : list[float]
        List of numeric values

    Returns
    -------
    list[tuple[int, float]]
        List of tuples (index, value) for each extrema point.
        Returns empty list for empty input or lists shorter than 2 elements.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> contour_extrema([1, 3, 2, 4, 1])  # Simple peaks and valleys
    [(0, 1.0), (1, 3.0), (2, 2.0), (3, 4.0), (4, 1.0)]
    >>> contour_extrema([1, 2, 2, 2, 1])  # Plateau
    [(0, 1.0), (4, 1.0)]
    >>> contour_extrema([1])  # Too short
    []
    """
    if not values or len(values) < 2:
        return []

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    extrema = [(0, values_array[0])]  # First note is always included
    n = len(values_array)

    # Check each interior point
    for i in range(1, n - 1):
        # Get values for comparison
        prev = values_array[i - 1]
        curr = values_array[i]
        next_val = values_array[i + 1]

        # Simple case: clear maximum or minimum
        if (prev < curr and next_val < curr) or (prev > curr and next_val > curr):
            extrema.append((i, curr))
            continue

        # Handle plateaus by looking at extended context
        if prev == curr or next_val == curr:
            # Look back up to 2 positions if available
            back2 = values_array[i - 2] if i >= 2 else None

            # Look forward up to 2 positions if available
            forward2 = values_array[i + 2] if i < n - 2 else None

            # Check plateau cases
            if prev == curr and forward2 is not None:
                if (back2 is not None and back2 < curr and next_val < curr) or (
                    back2 is not None and back2 > curr and next_val > curr
                ):
                    extrema.append((i, curr))
            elif next_val == curr and back2 is not None:
                if (forward2 is not None and forward2 < curr and prev < curr) or (
                    forward2 is not None and forward2 > curr and prev > curr
                ):
                    extrema.append((i, curr))

    extrema.append((n - 1, values_array[n - 1]))  # Last note is always included
    return [(i, float(v)) for i, v in extrema]  # Ensure values are float


def contour_gradients(values: list[float]) -> list[float]:
    """Calculates gradients between consecutive contour extrema points.

    For each pair of consecutive extrema points (ti,pi) and (tj,pj),
    calculates the gradient m = (pj-pi)/(tj-ti)

    Parameters
    ----------
    values : list[float]
        List of numeric values

    Returns
    -------
    list[float]
        List of gradients between consecutive extrema points.
        Returns empty list for empty input or lists shorter than 2 elements.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> contour_gradients([1, 3, 2, 4, 1])  # Simple sequence
    [2.0, -1.0, 2.0, -3.0]
    >>> contour_gradients([1, 1, 1])  # Flat sequence
    [0.0]
    >>> contour_gradients([1])  # Too short
    []
    """
    if not values or len(values) < 2:
        return []

    # Get extrema points using existing function
    extrema = contour_extrema(values)

    # Calculate gradients between consecutive extrema
    gradients = []
    for i in range(len(extrema) - 1):
        t1, p1 = extrema[i]
        t2, p2 = extrema[i + 1]

        # Skip if time difference is zero (shouldn't happen with current extrema logic)
        if t2 - t1 == 0:
            continue

        # Calculate gradient
        gradient = (p2 - p1) / (t2 - t1)
        gradients.append(float(gradient))

    return gradients


def compute_tonality_vector(pitch_classes) -> list[tuple[str, float]]:
    """Compute tonality vector for a sequence of pitch classes.

    Parameters
    ----------
    pitch_classes : list or numpy.ndarray
        List or array of pitch classes (0-11)

    Returns
    -------
    list[tuple[str, float]]
        List of (key, correlation) tuples sorted by correlation value
    """
    # Convert to numpy array if not already
    pitch_classes = np.asarray(pitch_classes)

    # Check if array is empty
    if pitch_classes.size == 0:
        return [
            ("C major", 0.0),
            ("C# major", 0.0),
            ("D major", 0.0),
            ("D# major", 0.0),
            ("E major", 0.0),
            ("F major", 0.0),
            ("F# major", 0.0),
            ("G major", 0.0),
            ("G# major", 0.0),
            ("A major", 0.0),
            ("A# major", 0.0),
            ("B major", 0.0),
            ("c minor", 0.0),
            ("c# minor", 0.0),
            ("d minor", 0.0),
            ("d# minor", 0.0),
            ("e minor", 0.0),
            ("f minor", 0.0),
            ("f# minor", 0.0),
            ("g minor", 0.0),
            ("g# minor", 0.0),
            ("a minor", 0.0),
            ("a# minor", 0.0),
            ("b minor", 0.0),
        ]

    # Create key name lists
    major_keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    minor_keys = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]

    # Define major and minor key profiles (pre-normalized)
    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )

    # Normalize profiles once
    major_profile = major_profile / np.sum(major_profile)
    minor_profile = minor_profile / np.sum(minor_profile)

    # Count pitch class occurrences once
    pc_counts = np.bincount(pitch_classes, minlength=12)
    pc_dist = pc_counts / np.sum(pc_counts)

    # Pre-compute all rotated profiles
    rotated_major = np.array([np.roll(major_profile, i) for i in range(12)])
    rotated_minor = np.array([np.roll(minor_profile, i) for i in range(12)])

    # Compute correlations for all keys at once
    major_corrs = np.array([np.corrcoef(pc_dist, rm)[0, 1] for rm in rotated_major])
    minor_corrs = np.array([np.corrcoef(pc_dist, rm)[0, 1] for rm in rotated_minor])

    # Create list of (key, correlation) tuples
    key_correlations = []
    for i in range(12):
        key_correlations.append((major_keys[i] + " major", float(major_corrs[i])))
        key_correlations.append((minor_keys[i] + " minor", float(minor_corrs[i])))

    # Sort by correlation value, descending
    return sorted(key_correlations, key=lambda x: x[1], reverse=True)


def arpeggiation_proportion(pitch_values: list[float]) -> float:
    """Calculate the proportion of notes in the melody that constitute triadic movement.

    Examines consecutive pitch intervals and counts what proportion match common
    arpeggio intervals like thirds, fifths, octaves etc.

    Parameters
    ----------
    pitch_values : list[float]
        List of pitch values to analyze

    Returns
    -------
    float
        Proportion of intervals that match arpeggio patterns (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.

    Examples
    --------
    >>> arpeggiation_proportion([60, 64, 67])  # C major triad
    1.0
    >>> arpeggiation_proportion([60, 62, 64])  # Stepwise motion
    0.0
    >>> arpeggiation_proportion([60])  # Single note
    0.0
    >>> arpeggiation_proportion(None)  # Invalid input
    -1.0
    """
    if pitch_values is None:
        return -1.0

    # Calculate differences between consecutive pitches
    intervals = []
    for i in range(len(pitch_values) - 1):
        interval = abs(pitch_values[i + 1] - pitch_values[i])
        intervals.append(interval)

    if not intervals:
        return 0.0

    # Count occurrences of specific intervals
    target_intervals = [
        0,  # repeated notes
        3,  # minor thirds
        4,  # major thirds
        7,  # perfect fifths
        10,  # minor sevenths
        11,  # major sevenths
        12,  # octaves
        15,  # minor tenths
        16,  # major tenths
    ]
    # Count how many intervals match our target arpeggio intervals
    matching_intervals = sum(
        1 for interval in intervals if interval in target_intervals
    )
    return float(matching_intervals) / len(intervals)


def chromatic_motion_proportion(pitch_values: list[float]) -> float:
    """Calculate the proportion of notes in the melody that move by chromatic intervals.

    Parameters
    ----------
    pitch_values : list[float]
        List of pitch values to analyze

    Returns
    -------
    float
        Proportion of notes that move by chromatic intervals (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.

    Examples
    --------
    >>> chromatic_motion_proportion([60, 61, 62, 63, 64])  # All semitones
    1.0
    >>> chromatic_motion_proportion([60, 62, 64])  # Stepwise motion
    0.0
    >>> chromatic_motion_proportion([60, 61, 63, 64])  # Mix which includes chromatic motion
    0.666...
    >>> chromatic_motion_proportion([60])  # Single note
    0.0
    >>> chromatic_motion_proportion(None)  # Invalid input
    -1.0
    """
    if pitch_values is None:
        return -1.0

    # Calculate differences between consecutive pitches
    intervals = []
    for i in range(len(pitch_values) - 1):
        interval = abs(pitch_values[i + 1] - pitch_values[i])
        intervals.append(interval)

    if not intervals:
        return 0.0

    # Count how many intervals match our target chromatic intervals
    chromatic = sum(1 for interval in intervals if interval == 1)
    return float(chromatic) / len(intervals)


def stepwise_motion_proportion(pitch_values: list[float]) -> float:
    """Calculate the proportion of notes in the melody that move by stepwise intervals.

    Parameters
    ----------
    pitch_values : list[float]
        List of pitch values to analyze

    Returns
    -------
    float
        Proportion of notes that move by stepwise intervals (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.

    Examples
    --------
    >>> stepwise_motion_proportion([60, 61, 62, 63, 64])  # All stepwise
    1.0
    >>> stepwise_motion_proportion([60, 63, 66])  # No stepwise motion
    0.0
    >>> stepwise_motion_proportion([60, 61, 64, 65])  # Mix which includes stepwise motion
    0.666...
    >>> stepwise_motion_proportion([60])  # Single note
    0.0
    >>> stepwise_motion_proportion(None)  # Invalid input
    -1.0
    """
    if pitch_values is None:
        return -1.0

    # Calculate differences between consecutive pitches
    intervals = []
    for i in range(len(pitch_values) - 1):
        interval = abs(pitch_values[i + 1] - pitch_values[i])
        intervals.append(interval)

    if not intervals:
        return 0.0

    # Count how many intervals match our target stepwise intervals
    stepwise = sum(1 for interval in intervals if interval in [1, 2])
    return float(stepwise) / len(intervals)


def repeated_notes_proportion(pitch_values: list[float]) -> float:
    """Calculate the proportion of notes in the melody that are repeated.

    Parameters
    ----------
    pitch_values : list[float]
        List of pitch values to analyze

    Returns
    -------
    float
        Proportion of notes that are repeated (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.

    Examples
    --------
    >>> repeated_notes_proportion([60, 60, 62, 64])  # One note repeated
    0.333...
    >>> repeated_notes_proportion([60, 60, 60, 60])  # All notes repeated
    1.0
    >>> repeated_notes_proportion([60, 62, 64])  # No repeated notes
    0.0
    >>> repeated_notes_proportion([60])  # Single note
    0.0
    >>> repeated_notes_proportion(None)  # Invalid input
    -1.0
    """
    value = -1.0
    if pitch_values is not None:
        intervals = [
            abs(pitch_values[i + 1] - pitch_values[i])
            for i in range(len(pitch_values) - 1)
        ]
        if intervals:
            repeated = sum(1 for interval in intervals if interval == 0)
            value = float(repeated) / len(intervals)
        else:
            value = 0.0

    return value


def get_durations(starts: list[float], ends: list[float]) -> list[float]:
    """Calculate durations from start and end times.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    list[float]
        List of note durations
    """
    if not starts or not ends or len(starts) != len(ends):
        return []

    try:
        return [end - start for start, end in zip(starts, ends)]
    except (TypeError, ValueError):
        return []


def get_duration_ratios(starts: list[float], ends: list[float]) -> list[float]:
    """Calculate duration ratios between subsequent notes.
    
    Based on Steinbeck (1982) as implemented in FANTASTIC toolbox.
    Computes ri = ΔTi / ΔTi+1 where ΔTi is the duration of note i.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
        
    Returns
    -------
    list[float]
        List of duration ratios between subsequent notes.
        Returns empty list for melodies with fewer than 2 notes.
        
    Raises
    ------
    TypeError
        If any element cannot be converted to float
        
    Examples
    --------
    >>> get_duration_ratios([0, 1, 2, 3], [1, 2, 3, 4])  # Equal durations
    [1.0, 1.0, 1.0]
    >>> get_duration_ratios([0, 1, 3], [1, 3, 4])  # Varying durations
    [0.5, 2.0]
    >>> get_duration_ratios([0], [1])  # Single note
    []
    """
    if len(starts) < 2 or len(ends) < 2:
        return []
    
    durations = get_durations(starts, ends)
    if len(durations) < 2:
        return []
    
    # Calculate ratios ri = duration[i] / duration[i+1]
    ratios = []
    for i in range(len(durations) - 1):
        if durations[i + 1] != 0:  # Avoid division by zero
            ratios.append(durations[i] / durations[i + 1])
    
    return ratios


def melodic_embellishment_proportion(
    pitch_values: list[float], note_starts: list[float], note_ends: list[float]
) -> float:
    """Calculate the proportion of notes in the melody that are embellished.

    Identifies embellished notes by checking if they are surrounded by notes that are
    at least a third of their duration.

    Parameters
    ----------
    pitch_values : list[float]
        List of pitch values to analyze
    note_starts : list[float]
        List of note start times
    note_ends : list[float]
        List of note end times

    Returns
    -------
    float
        Proportion of notes that are embellished (0.0-1.0).
        Returns -1.0 if any input is None, 0.0 if fewer than 3 notes.

    Examples
    --------
    >>> melodic_embellishment_proportion([60, 62, 64], [0, 1, 2], [1, 2, 3])  # No embellishment
    0.0
    >>> melodic_embellishment_proportion([60, 62, 64, 66], [0, 0.25, 2, 3], [0.25, 2, 2.25, 4])  # Some embellishment
    0.25
    """
    if any([pitch_values is None, note_starts is None, note_ends is None]):
        return -1.0

    # Calculate the duration of each note
    note_durations = get_durations(note_starts, note_ends)
    # Count embellished notes (notes surrounded by shorter notes)
    embellished = 0
    for i in range(1, len(note_durations) - 1):
        if (
            note_durations[i - 1] < note_durations[i] / 3
            and note_durations[i + 1] < note_durations[i] / 3
        ):
            embellished += 1

    # Calculate proportion of embellished notes
    if len(note_durations) > 2:  # Need at least 3 notes to have embellishment
        return float(embellished) / len(note_durations)
    return 0.0


def longest_monotonic_conjunct_scalar_passage(
    pitches: list[int], key_correlations: Optional[list] = None
) -> int:
    """Find the longest sequence of consecutive notes that follow a scale pattern,
    moving in the same direction.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    key_correlations : Optional[list]
        Pre-computed key correlations from compute_tonality_vector.
        If None, will compute them.

    Returns
    -------
    int
        Length of longest monotonic scalar sequence found

    Examples
    --------
    Twinkle Twinkle Little Star is in C major, with a total of 14 notes.
    However, many are repeated notes, which are omitted from the calculation.
    There are 8 notes once the repetitions are removed, 6 of which form a scalic sequence.
    >>> longest_monotonic_conjunct_scalar_passage([60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60])
    6

    The lick is in D minor, with 4 of 7 notes as conjunct motions
    >>> longest_monotonic_conjunct_scalar_passage([62, 64, 65, 67, 64, 60, 62])
    4

    C major scale (one octave) scores 8
    >>> longest_monotonic_conjunct_scalar_passage([60, 62, 64, 65, 67, 69, 71, 72])
    8

    >>> # Test case with melody containing a descending scalar passage of length 5
    >>> pitches = [62, 67, 67, 69, 74, 72, 71, 69, 67, 71, 69, 66, 67, 69, 67, 66, 64, 62, 62, 67, 67, 69, 69, 74, 74, 74, 72, 71, 69, 67, 69, 71, 69, 67]
    >>> longest_monotonic_conjunct_scalar_passage(pitches)
    5
    """
    # Remove repeated notes
    deduped = []
    for pitch in pitches:
        if not deduped or pitch != deduped[-1]:
            deduped.append(pitch)

    if key_correlations is None:
        # Get key using KS algorithm
        pitch_classes = [p % 12 for p in deduped]
        key_correlations = compute_tonality_vector(pitch_classes)

    key = key_correlations[0][0].split()[0]

    root = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11,
        "c": 0,
        "c#": 1,
        "d": 2,
        "d#": 3,
        "e": 4,
        "f": 5,
        "f#": 6,
        "g": 7,
        "g#": 8,
        "a": 9,
        "a#": 10,
        "b": 11,
    }[key]
    if "minor" in key.lower():
        scale = [(root + i) % 12 for i in [0, 2, 3, 5, 7, 8, 10]]
    else:
        scale = [(root + i) % 12 for i in [0, 2, 4, 5, 7, 9, 11]]

    if len(pitches) < 3:
        return 0

    if len(deduped) < 3:
        return 0

    # Find longest monotonic scalar sequence
    longest_sequence = 0
    current_sequence = 1
    direction = None

    for i in range(1, len(deduped)):
        interval = deduped[i] - deduped[i - 1]

        # Check if notes are adjacent scale degrees
        curr_pc = deduped[i] % 12
        prev_pc = deduped[i - 1] % 12

        if curr_pc not in scale or prev_pc not in scale:
            current_sequence = 1
            direction = None
            continue

        curr_scale_pos = scale.index(curr_pc)
        prev_scale_pos = scale.index(prev_pc)

        # Check if they're adjacent in the scale
        scale_interval = (curr_scale_pos - prev_scale_pos) % len(scale)
        if scale_interval != 1 and scale_interval != len(scale) - 1:
            current_sequence = 1
            direction = None
            continue

        # Check direction
        curr_direction = 1 if interval > 0 else -1

        if direction is None:
            direction = curr_direction
            current_sequence = 2
        elif direction == curr_direction:
            current_sequence += 1
        else:
            current_sequence = 2
            direction = curr_direction

        longest_sequence = max(longest_sequence, current_sequence)

    return longest_sequence


def longest_conjunct_scalar_passage(
    pitches: list[int], key_correlations: Optional[list] = None
) -> int:
    """Find the longest sequence of consecutive notes that follow a scale pattern,
    allowing for changes in direction.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    key_correlations : Optional[list]
        Pre-computed key correlations from compute_tonality_vector.
        If None, will compute them.

    Returns
    -------
    int
        Length of longest scalar sequence found

    Examples
    --------
    >>> longest_conjunct_scalar_passage([60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60])
    7

    >>> longest_conjunct_scalar_passage([62, 64, 65, 67, 64, 60, 62])
    4
    """
    if len(pitches) < 3:
        return 0

    # Remove repeated notes
    deduped = []
    for pitch in pitches:
        if not deduped or pitch != deduped[-1]:
            deduped.append(pitch)

    if len(deduped) < 3:
        return 0

    # Get key using KS algorithm
    pitch_classes = [p % 12 for p in deduped]
    if key_correlations is None:
        key_correlations = compute_tonality_vector(pitch_classes)
    key = key_correlations[0][0]
    root = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11,
        "c": 0,
        "c#": 1,
        "d": 2,
        "d#": 3,
        "e": 4,
        "f": 5,
        "f#": 6,
        "g": 7,
        "g#": 8,
        "a": 9,
        "a#": 10,
        "b": 11,
    }[key.split()[0]]
    if "minor" in key.lower():
        scale = [(root + i) % 12 for i in [0, 2, 3, 5, 7, 8, 10]]
    else:
        scale = [(root + i) % 12 for i in [0, 2, 4, 5, 7, 9, 11]]

    # Find longest scalar sequence
    longest_sequence = 0
    current_sequence = 1

    for i in range(1, len(deduped)):
        curr_pc = deduped[i] % 12
        prev_pc = deduped[i - 1] % 12

        if curr_pc not in scale or prev_pc not in scale:
            current_sequence = 1
            continue

        curr_scale_pos = scale.index(curr_pc)
        prev_scale_pos = scale.index(prev_pc)

        # Check if they're adjacent in the scale (in either direction)
        scale_interval = (curr_scale_pos - prev_scale_pos) % len(scale)
        if scale_interval == 1 or scale_interval == len(scale) - 1:
            current_sequence += 1
        else:
            current_sequence = 1

        longest_sequence = max(longest_sequence, current_sequence)

    return longest_sequence


def proportion_conjunct_scalar(
    pitches: list[int], key_correlations: Optional[list] = None
) -> float:
    """Calculate the proportion of notes that form conjunct scalar sequences.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    key_correlations : Optional[list]
        Pre-computed key correlations from compute_tonality_vector.
        If None, will compute them.

    Returns
    -------
    float
        Proportion of notes in conjunct scalar sequences (0.0-1.0)

    Examples
    --------
    >>> proportion_conjunct_scalar([60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60])
    0.5

    >>> proportion_conjunct_scalar([62, 64, 65, 67, 64, 60, 62])
    0.571...
    """
    if len(pitches) < 3:
        return 0.0

    scalar_length = longest_conjunct_scalar_passage(pitches, key_correlations)
    return scalar_length / len(pitches)


def proportion_scalar(
    pitches: list[int], key_correlations: Optional[list] = None
) -> float:
    """Calculate the proportion of notes that form scalar sequences.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    key_correlations : Optional[list]
        Pre-computed key correlations from compute_tonality_vector.
        If None, will compute them.

    Returns
    -------
    float
        Proportion of notes in scalar sequences (0.0-1.0)

    Examples
    --------
    >>> proportion_scalar([60, 62, 64, 65, 67, 69, 71, 72])
    1.0

    >>> proportion_scalar([62, 64, 65, 67, 64, 60, 62])
    0.571...

    >>> proportion_scalar([60, 62, 64, 65, 67, 68, 71, 72])
    0.625
    """
    if len(pitches) < 3:
        return 0.0

    scalar_length = longest_monotonic_conjunct_scalar_passage(pitches, key_correlations)
    return scalar_length / len(pitches)
