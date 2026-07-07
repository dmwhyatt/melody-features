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


def midi_toolbox_entropy(distribution) -> float:
    """Relative entropy per MIDI Toolbox `entropy.m` (natural log, / log(n))."""
    d = np.asarray(distribution, dtype=float).ravel()
    if d.size == 0:
        return 0.0
    d = d / (d.sum() + 1e-12)
    logd = np.log(d + 1e-12)
    return float(-np.sum(d * logd) / np.log(d.size))


def distribution_entropy(weights) -> float:
    """Base-2 Shannon entropy of a discrete distribution from non-negative weights.

    Weights are normalized to sum to 1; zero weights are omitted. Returns `0.0`
    if all weights are zero or the input is empty.

    Parameters
    ----------
    weights : list or numpy.ndarray
        Non-negative masses (e.g. key-correlation magnitudes), not raw samples.

    Returns
    -------
    float
        Shannon entropy in bits.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return 0.0
    weights = weights[weights > 0]
    if weights.size == 0:
        return 0.0
    probs = weights / weights.sum()
    return float(-np.sum(probs * np.log2(probs)))


def get_mode(values) -> float:
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


def time_to_ticks(time_seconds: float, tempo: float = 120.0, ppqn: int = 480) -> int:
    """Convert time in seconds to MIDI ticks.
    
    This is the standard tick conversion used throughout the codebase for
    tempo-aware timing calculations.
    
    Parameters
    ----------
    time_seconds : float
        Time in seconds to convert
    tempo : float, default=120.0
        Tempo in beats per minute
    ppqn : int, default=480
        Pulses per quarter note (MIDI resolution)
        
    Returns
    -------
    int
        Time converted to MIDI ticks
    """
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    return int(round(float(time_seconds) / seconds_per_tick))


def times_to_ticks(times: list[float], tempo: float = 120.0, ppqn: int = 480) -> list[int]:
    """Convert a list of times in seconds to MIDI ticks.
    
    Parameters
    ----------
    times : list[float]
        List of times in seconds to convert
    tempo : float, default=120.0
        Tempo in beats per minute
    ppqn : int, default=480
        Pulses per quarter note (MIDI resolution)
        
    Returns
    -------
    list[int]
        List of times converted to MIDI ticks
    """
    return [time_to_ticks(t, tempo, ppqn) for t in times]
