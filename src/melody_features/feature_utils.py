"""Shared helpers for feature definitions."""

from collections import Counter

import numpy as np


def ioi_from_starts(starts: list[float]) -> list[float]:
    """Inter-onset intervals from consecutive onset times."""
    if len(starts) < 2:
        return []
    return [float(starts[i] - starts[i - 1]) for i in range(1, len(starts))]


def mean_and_std(values: list[float]) -> tuple[float, float]:
    """Sample mean and standard deviation (ddof=1) for a numeric series."""
    if not values:
        return 0.0, 0.0
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    return mean, float(np.std(values, ddof=1))


def prevalence_of_mode(values: list) -> float:
    """Proportion of elements equal to the most common value."""
    if not values:
        return 0.0
    _, count = Counter(values).most_common(1)[0]
    return float(count / len(values))


def relative_prevalence_top_two(values: list) -> float:
    """Ratio of second-most-common to most-common relative frequencies."""
    if len(values) < 2:
        return 0.0
    counts = Counter(values)
    if len(counts) < 2:
        return 0.0
    first_freq, second_freq = (
        freq / len(values)
        for _, freq in counts.most_common(2)
    )
    if first_freq == 0.0:
        return 0.0
    return float(second_freq / first_freq)


def _get_durations(starts: list[float], ends: list[float], tempo: float = 120.0) -> list[float]:
    """Safely calculate durations from start and end times, converted to quarter notes.

    Parameters
    ----------
    starts : list[float]
        List of note start times (in seconds)
    ends : list[float]
        List of note end times (in seconds)
    tempo : float
        Tempo in BPM (beats per minute), default 120.0

    Returns
    -------
    list[float]
        List of durations in quarter notes, or empty list if calculation fails
    """
    if not starts or not ends or len(starts) != len(ends):
        return []
    try:
        durations_seconds = [float(end - start) for start, end in zip(starts, ends)]
        durations_quarter_notes = [duration * (tempo / 60.0) for duration in durations_seconds]
        return durations_quarter_notes
    except (TypeError, ValueError):
        return []
