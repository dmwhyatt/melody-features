"""
This module contains functions to compute features from melodies.
Features are the product of an input list and at least one algorithm.
"""
__author__ = "David Whyatt"

import json
import math
import csv
import warnings
from random import choices
from typing import Dict, List
from multiprocessing import Pool, cpu_count
from melodic_feature_set.algorithms import (
    rank_values, nine_percent_significant_values, circle_of_fifths,
    compute_tonality_vector, arpeggiation_proportion,
    chromatic_motion_proportion, stepwise_motion_proportion,
    repeated_notes_proportion, melodic_embellishment_proportion,
    longest_monotonic_conjunct_scalar_passage, longest_conjunct_scalar_passage,
    proportion_conjunct_scalar, proportion_scalar
)
from melodic_feature_set.complexity import (
    consecutive_fifths, repetition_rate
)
from melodic_feature_set.distributional import distribution_proportions, histogram_bins, kurtosis, skew
from melodic_feature_set.idyom_interface import run_idyom
from melodic_feature_set.import_mid import import_midi
from melodic_feature_set.interpolation_contour import InterpolationContour
from melodic_feature_set.melody_tokenizer import FantasticTokenizer
from melodic_feature_set.ngram_counter import NGramCounter
from melodic_feature_set.narmour import (
    proximity, closure, registral_direction, registral_return, intervallic_difference)
from melodic_feature_set.representations import Melody
from melodic_feature_set.stats import range_func, standard_deviation, shannon_entropy, mode
from melodic_feature_set.step_contour import StepContour
from melodic_feature_set.corpus import make_corpus_stats, load_corpus_stats
import numpy as np
import scipy
import pandas as pd
import tempfile
from pathlib import Path
import os
import glob
from natsort import natsorted
import time
import threading

# Pitch Features
def pitch_range(pitches: list[int]) -> int:
    """Calculate the range between the highest and lowest pitches.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Range between highest and lowest pitch in semitones
    """
    return int(range_func(pitches))

def pitch_standard_deviation(pitches: list[int]) -> float:
    """Calculate the standard deviation of pitch values.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of pitches
    """
    return float(standard_deviation(pitches))

def pitch_entropy(pitches: list[int]) -> float:
    """Calculate the Shannon entropy of pitch values.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Shannon entropy of pitch distribution
    """
    return float(shannon_entropy(pitches))

def pcdist1(pitches: list[int], starts: list[float], ends: list[float]) -> float:
    """Calculate duration-weighted distribution of pitch classes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Duration-weighted distribution proportion of pitch classes
    """
    if not pitches or not starts or not ends:
        return 0.0

    durations = [ends[i] - starts[i] for i in range(len(starts))]
    # Create weighted list by repeating each pitch class according to its duration
    weighted_pitch_classes = []
    for pitch, duration in zip(pitches, durations):
        # Convert pitch to pitch class (0-11)
        pitch_class = pitch % 12
        # Convert duration to integer number of repetitions (e.g. duration 2.5 -> 25 repetitions)
        repetitions = max(1, int(duration * 10))  # Ensure at least 1 repetition
        weighted_pitch_classes.extend([pitch_class] * repetitions)

    if not weighted_pitch_classes:
        return 0.0

    return distribution_proportions(weighted_pitch_classes)

def basic_pitch_histogram(pitches: list[int]) -> dict:
    """Create histogram of pitch values within range of input pitches.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    dict
        Dictionary mapping pitch values to counts
    """
    if not pitches:
        return {}

    # Use number of unique pitches as number of bins, with minimum of 1
    num_midi_notes = max(1, len(set(pitches)))
    return histogram_bins(pitches, num_midi_notes)

def pitch_ranking(pitches: list[int]) -> float:
    """Calculate ranking of pitches in descending order.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Ranking value of pitches
    """
    return rank_values(pitches, descending=True)

def melodic_pitch_variety(pitches: list[int]) -> float:
    """Calculate rate of pitch repetition.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Rate of pitch repetition
    """
    return repetition_rate(pitches)

def dominant_spread(pitches: list[int]) -> float:
    """Find longest sequence of pitch classes separated by perfect 5ths that each appear >9% of the time.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Length of longest sequence of significant pitch classes separated by perfect 5ths
    """
    pcs = [pitch % 12 for pitch in pitches]
    longest_sequence = []
    nine_percent_significant = nine_percent_significant_values(pcs)

    for i, pc in enumerate(pcs):
        if pc in nine_percent_significant:
            consecutive_fifth_pcs = consecutive_fifths(pcs[i:])
            if len(consecutive_fifth_pcs) > len(longest_sequence):
                longest_sequence = consecutive_fifth_pcs

    return len(longest_sequence)

def mean_pitch(pitches: list[int]) -> float:
    """Calculate mean pitch value.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Mean pitch value
    """
    return np.mean(pitches)

def most_common_pitch(pitches: list[int]) -> int:
    """Find most frequently occurring pitch.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Most common pitch value
    """
    return int(mode(pitches))

def number_of_common_pitches(pitches: list[int]) -> int:
    """Count pitch classes that appear in at least 9% of notes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of significant pitch classes
    """
    pcs = [pitch % 12 for pitch in pitches]
    significant_pcs = nine_percent_significant_values(pcs)
    return int(len(significant_pcs))

def number_of_pitches(pitches: list[int]) -> int:
    """Count number of unique pitches.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of unique pitches
    """
    return int(len(set(pitches)))

def folded_fifths_pitch_class_histogram(pitches: list[int]) -> dict:
    """Create histogram of pitch classes arranged in circle of fifths.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    dict
        Dictionary mapping pitch classes to counts, arranged by circle of fifths
    """
    # Get pitch classes and count occurrences
    pcs = [pitch % 12 for pitch in pitches]
    # Count occurrences of each pitch class
    unique = []
    counts = []
    for pc in set(pcs):
        unique.append(pc)
        counts.append(pcs.count(pc))
    return circle_of_fifths(unique, counts)

def pitch_class_kurtosis_after_folding(pitches: list[int]) -> float:
    """Calculate kurtosis of folded fifths pitch class histogram.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Kurtosis of folded fifths histogram values, or 0 for empty input
    """
    pitches = [pitch % 12 for pitch in pitches]
    histogram = folded_fifths_pitch_class_histogram(pitches)
    if not histogram:
        return 0.0
    return float(kurtosis(list(histogram.keys())))

def pitch_class_skewness_after_folding(pitches: list[int]) -> float:
    """Calculate skewness of folded fifths pitch class histogram.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Skewness of folded fifths histogram values, or 0 for empty input
    """
    pitches = [pitch % 12 for pitch in pitches]
    histogram = folded_fifths_pitch_class_histogram(pitches)
    if not histogram:
        return 0.0
    return float(skew(list(histogram.keys())))

def pitch_class_variability_after_folding(pitches: list[int]) -> float:
    """Calculate standard deviation of folded fifths pitch class histogram.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of folded fifths histogram values, or 0 for empty input
    """
    pitches = [pitch % 12 for pitch in pitches]
    histogram = folded_fifths_pitch_class_histogram(pitches)
    if not histogram:
        return 0.0
    return float(standard_deviation(list(histogram.keys())))

# Interval Features

def pitch_interval(pitches: list[int]) -> list[int]:
    """Calculate intervals between consecutive pitches.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    list[int]
        List of intervals between consecutive pitches in semitones
    """
    return [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]

def absolute_interval_range(pitches: list[int]) -> int:
    """Calculate range between largest and smallest absolute interval size.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Range between largest and smallest absolute interval in semitones
    """
    return int(range_func([abs(x) for x in pitch_interval(pitches)]))

def mean_absolute_interval(pitches: list[int]) -> float:
    """Calculate mean absolute interval size.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Mean absolute interval size in semitones
    """
    return float(np.mean([abs(x) for x in pitch_interval(pitches)]))

# Alias for mean_absolute_interval / FANTASTIC vs jSymbolic
mean_melodic_interval = mean_absolute_interval

def standard_deviation_absolute_interval(pitches: list[int]) -> float:
    """Calculate standard deviation of absolute interval sizes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of absolute interval sizes in semitones
    """
    return float(np.std([abs(x) for x in pitch_interval(pitches)]))

def modal_interval(pitches: list[int]) -> int:
    """Find most common interval size.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Most frequent interval size in semitones
    """
    return int(mode(pitch_interval(pitches)))

# Alias for modal_interval / FANTASTIC vs jSymbolic
most_common_interval = modal_interval

def interval_entropy(pitches: list[int]) -> float:
    """Calculate Shannon entropy of interval distribution.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Shannon entropy of interval sizes
    """
    return float(shannon_entropy(pitch_interval(pitches)))

def get_durations(starts: list[float], ends: list[float]) -> list[float]:
    """Safely calculate durations from start and end times.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
        
    Returns
    -------
    list[float]
        List of durations, or empty list if calculation fails
    """
    if not starts or not ends or len(starts) != len(ends):
        return []
    try:
        return [float(end - start) for start, end in zip(starts, ends)]
    except (TypeError, ValueError):
        return []

def ivdist1(pitches: list[int], starts: list[float], ends: list[float]) -> dict:
    """Calculate duration-weighted distribution of intervals.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Duration-weighted distribution proportion of intervals
    """
    if not pitches or not starts or not ends or len(pitches) < 2:
        return 0.0

    intervals = pitch_interval(pitches)
    durations = get_durations(starts, ends)

    if not intervals or not durations:
        return 0.0

    weighted_intervals = []
    for interval, duration in zip(intervals, durations[:-1]):
        repetitions = max(1, int(duration * 10))  # Ensure at least 1 repetition
        weighted_intervals.extend([interval] * repetitions)

    if not weighted_intervals:
        return 0.0

    return distribution_proportions(weighted_intervals)

def interval_direction(pitches: list[int]) -> tuple[float, float]:
    """Determine direction of each interval and calculate mean and standard deviation.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of interval directions, where:
        1 represents upward motion
        0 represents same pitch
        -1 represents downward motion
    """
    directions = [1 if pitches[i + 1] > pitches[i]
                 else 0 if pitches[i + 1] == pitches[i]
                 else -1
            for i in range(len(pitches) - 1)]

    if not directions:
        return 0.0, 0.0

    mean = sum(directions) / len(directions)
    variance = sum((x - mean) ** 2 for x in directions) / len(directions)
    std_dev = math.sqrt(variance)

    return float(mean), float(std_dev)

def average_interval_span_by_melodic_arcs(pitches: list[int]) -> float:
    """Calculate average interval span of melodic arcs.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Average interval span of melodic arcs, or 0.0 if no arcs found
    """
    total_intervals = 0
    number_intervals = 0

    intervals = pitch_interval(pitches)
    direction = 0
    interval_so_far = 0

    for interval in intervals:
        if direction == -1:  # Arc is currently descending
            if interval < 0:
                interval_so_far += abs(interval)
            elif interval > 0:
                total_intervals += interval_so_far
                number_intervals += 1
                interval_so_far = abs(interval)
                direction = 1

        elif direction == 1:  # Arc is currently ascending
            if interval > 0:
                interval_so_far += abs(interval)
            elif interval < 0:
                total_intervals += interval_so_far
                number_intervals += 1
                interval_so_far = abs(interval)
                direction = -1

        elif direction == 0:  # Arc is currently stationary
            if interval > 0:
                direction = 1
                interval_so_far += abs(interval)
            elif interval < 0:
                direction = -1
                interval_so_far += abs(interval)

    if number_intervals == 0:
        value = 0.0
    else:
        value = total_intervals / number_intervals

    return float(value)

def distance_between_most_prevalent_melodic_intervals(pitches: list[int]) -> float:
    """Calculate absolute difference between two most common interval sizes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Absolute difference between two most common intervals, or 0.0 if fewer than 2 intervals
    """
    if len(pitches) < 2:
        return 0.0

    intervals = pitch_interval(pitches)

    interval_counts = {}
    for interval in intervals:
        interval_counts[interval] = interval_counts.get(interval, 0) + 1

    if len(interval_counts) < 2:
        return 0.0

    sorted_intervals = sorted(interval_counts.items(), key=lambda x: x[1], reverse=True)
    most_common = sorted_intervals[0][0]
    second_most_common = sorted_intervals[1][0]
    return float(abs(most_common - second_most_common))

def melodic_interval_histogram(pitches: list[int]) -> dict:
    """Create histogram of interval sizes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    dict
        Dictionary mapping interval sizes to counts
    """
    intervals = pitch_interval(pitches)
    num_intervals = max(1, int(range_func(intervals)))
    return histogram_bins(intervals, num_intervals)

def melodic_large_intervals(pitches: list[int]) -> float:
    """Calculate proportion of intervals >= 13 semitones.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of large intervals, or -1.0 if no intervals
    """
    intervals = pitch_interval(pitches)
    if not intervals:
        return -1.0
    large_intervals = sum(1 for interval in intervals if abs(interval) >= 13)
    return float(large_intervals / len(intervals) if intervals else 0.0)

def variable_melodic_intervals(pitches: list[int], interval_level: int) -> float:
    """Calculate proportion of intervals >= specified size.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    interval_level : int
        Minimum interval size in semitones

    Returns
    -------
    float
        Proportion of intervals >= interval_level, or -1.0 if no intervals
    """
    intervals = pitch_interval(pitches)
    if not intervals:
        return -1.0
    large_intervals = sum(1 for interval in intervals if abs(interval) >= interval_level)
    return float(large_intervals / len(intervals) if intervals else 0.0)

def number_of_common_melodic_intervals(pitches: list[int]) -> int:
    """Count intervals that appear in at least 9% of melodic transitions.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of significant intervals
    """
    if len(pitches) < 2:
        return 0

    intervals = pitch_interval(pitches)
    significant_intervals = nine_percent_significant_values(intervals)

    return int(len(significant_intervals))

def prevalence_of_most_common_melodic_interval(pitches: list[int]) -> float:
    """Calculate proportion of most common interval.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of most common interval, or 0 if no intervals
    """
    intervals = pitch_interval(pitches)
    if not intervals:
        return 0

    interval_counts = {}
    for interval in intervals:
        interval_counts[interval] = interval_counts.get(interval, 0) + 1

    return float(max(interval_counts.values()) / len(intervals))

# Contour Features
def get_step_contour_features(pitches: list[int], starts: list[float], ends: list[float]) -> StepContour:
    """Calculate step contour features.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    StepContour
        StepContour object with global variation, direction and local variation
    """
    if not pitches or not starts or not ends or len(pitches) < 2:
        return 0.0, 0.0, 0.0

    durations = get_durations(starts, ends)
    if not durations:
        return 0.0, 0.0, 0.0

    sc = StepContour(pitches, durations)
    return sc.global_variation, sc.global_direction, sc.local_variation

def get_interpolation_contour_features(pitches: list[int], starts: list[float]) -> InterpolationContour:
    """Calculate interpolation contour features.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times

    Returns
    -------
    InterpolationContour
        InterpolationContour object with direction, gradient and class features
    """
    ic = InterpolationContour(pitches, starts)
    return (ic.global_direction, ic.mean_gradient, ic.gradient_std,
            ic.direction_changes, ic.class_label)

# Duration Features

def get_tempo(melody: Melody) -> float:
    """Access tempo of melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Tempo of melody in bpm
    
    """
    return melody.tempo

def duration_range(starts: list[float], ends: list[float]) -> float:
    """Calculate range between longest and shortest note duration.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Range between longest and shortest duration
    """
    durations = get_durations(starts, ends)
    if not durations:
        return 0.0
    return float(range_func(durations))

def mean_duration(starts: list[float], ends: list[float]) -> float:
    """Calculate mean note duration.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Mean note duration
    """
    durations = get_durations(starts, ends)
    if not durations:
        return 0.0
    return float(np.mean(durations))

def duration_standard_deviation(starts: list[float], ends: list[float]) -> float:
    """Calculate standard deviation of note durations.

    Parameters
    ---------- 
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Standard deviation of note durations
    """
    durations = get_durations(starts, ends)
    if not durations:
        return 0.0
    return float(np.std(durations))

def modal_duration(starts: list[float], ends: list[float]) -> float:
    """Find most common note duration.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Most frequent note duration
    """
    durations = get_durations(starts, ends)
    if not durations:
        return 0.0
    return float(mode(durations))

def duration_entropy(starts: list[float], ends: list[float]) -> float:
    """Calculate Shannon entropy of duration distribution.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Shannon entropy of note durations
    """
    durations = get_durations(starts, ends)
    if not durations:
        return 0.0
    return float(shannon_entropy(durations))

def length(starts: list[float]) -> float:
    """Count total number of notes.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Total number of notes
    """
    return len(starts)

def number_of_durations(starts: list[float], ends: list[float]) -> int:
    """Count number of unique note durations.

    Parameters
    ----------
    starts : list[float]
        List of note start times    
    ends : list[float]
        List of note end times

    Returns
    -------
    int
        Number of unique note durations
    """
    durations = get_durations(starts, ends)
    if not durations:
        return 0
    return int(len(set(durations)))

def global_duration(starts: list[float], ends: list[float]) -> float:
    """Calculate total duration from first note start to last note end.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Total duration of melody
    """
    if not starts or not ends or len(starts) == 0 or len(ends) == 0:
        return 0.0
    return float(ends[-1] - starts[0])

def note_density(starts: list[float], ends: list[float]) -> float:
    """Calculate average number of notes per unit time.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Note density (notes per unit time)
    """
    if not starts or not ends or len(starts) == 0 or len(ends) == 0:
        return 0.0
    total_duration = global_duration(starts, ends)
    if total_duration == 0:
        return 0.0
    return float(len(starts) / total_duration)

def ioi(starts: list[float]) -> tuple[float, float]:
    """Calculate mean and standard deviation of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of inter-onset intervals
    """
    intervals = [starts[i] - starts[i-1] for i in range(1, len(starts))]
    if not intervals:
        return 0.0, 0.0
    return float(np.mean(intervals)), float(np.std(intervals))

def ioi_ratio(starts: list[float]) -> tuple[float, float]:
    """Calculate mean and standard deviation of inter-onset interval ratios.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
        
    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of IOI ratios
    """
    if len(starts) < 2:
        return 0.0, 0.0

    # Calculate intervals first
    intervals = [starts[i] - starts[i-1] for i in range(1, len(starts))]

    if len(intervals) < 2:
        return 0.0, 0.0

    # Calculate ratios between consecutive intervals
    ratios = [intervals[i]/intervals[i-1] for i in range(1, len(intervals))]
    return float(np.mean(ratios)), float(np.std(ratios))

def ioi_range(starts: list[float]) -> float:
    """Calculate range of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Range of inter-onset intervals
    """
    intervals = [starts[i] - starts[i-1] for i in range(1, len(starts))]
    return max(intervals) - min(intervals)

def ioi_mean(starts: list[float]) -> float:
    """Calculate mean of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Mean of inter-onset intervals
    """
    intervals = [starts[i] - starts[i-1] for i in range(1, len(starts))]
    return float(np.mean(intervals))

def ioi_standard_deviation(starts: list[float]) -> float:
    """Calculate standard deviation of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Standard deviation of inter-onset intervals
    """
    intervals = [starts[i] - starts[i-1] for i in range(1, len(starts))]
    return float(np.std(intervals))

def ioi_contour(starts: list[float]) -> tuple[float, float]:
    """Calculate mean and standard deviation of IOI contour.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of contour values (-1: shorter, 0: same, 1: longer)
    """
    intervals = [starts[i] - starts[i-1] for i in range(1, len(starts))]
    if len(intervals) < 2:
        return 0.0, 0.0

    ratios = [intervals[i]/intervals[i-1] for i in range(1, len(intervals))]
    contour = [int(np.sign(ratio - 1)) for ratio in ratios]
    return float(np.mean(contour)), float(np.std(contour))

def duration_histogram(starts: list[float], ends: list[float]) -> dict:
    """Calculate histogram of note durations.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    dict
        Histogram of note durations
    """
    durations = get_durations(starts, ends)
    if not durations:
        return {}
    num_durations = max(1, len(set(durations)))
    return histogram_bins(durations, num_durations)

def ioi_histogram(starts: list[float]) -> dict:
    """Calculate histogram of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    dict
        Histogram of inter-onset intervals
    """
    intervals = [starts[i] - starts[i-1] for i in range(1, len(starts))]
    num_intervals = len(set(intervals))
    return histogram_bins(intervals, num_intervals)

# Tonality Features
def tonalness(pitches: list[int]) -> float:
    """Calculate tonalness as magnitude of highest key correlation.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Magnitude of highest key correlation value
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlation = compute_tonality_vector(pitch_classes)
    return correlation[0][1]

def tonal_clarity(pitches: list[int]) -> float:
    """Calculate ratio between top two key correlation values.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Ratio between highest and second highest key correlation values.
        Returns 1.0 if fewer than 2 correlation values.
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)

    if len(correlations) < 2:
        return -1.0

    # Get top 2 correlation values
    top_corr = abs(correlations[0][1])
    second_corr = abs(correlations[1][1])

    # Avoid division by zero
    if second_corr == 0:
        return 1.0

    return top_corr / second_corr

def tonal_spike(pitches: list[int]) -> float:
    """Calculate ratio between highest key correlation and sum of all others.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Ratio between highest correlation value and sum of all others.
        Returns 1.0 if fewer than 2 correlation values or sum is zero.
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)

    if len(correlations) < 2:
        return -1.0

    # Get highest correlation and sum of rest
    top_corr = abs(correlations[0][1])
    other_sum = sum(abs(corr[1]) for corr in correlations[1:])

    # Avoid division by zero
    if other_sum == 0:
        return 1.0

    return top_corr / other_sum

def tonal_entropy(pitches: list[int]) -> float:
    """Calculate tonal entropy as the entropy across the key correlations.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Entropy of the tonality vector correlation distribution
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)
    if not correlations:
        return -1.0

    # Calculate entropy of correlation distribution
    # Extract just the correlation values and normalize them to positive values
    corr_values = [abs(corr[1]) for corr in correlations]

    # Calculate entropy of the correlation distribution
    return shannon_entropy(corr_values)

def get_key_distances() -> dict[str, int]:
    """Returns a dictionary mapping key names to their semitone distances from C.
    
    Returns
    -------
    dict[str, int]
        Dictionary mapping key names (both major and minor) to semitone distances from C.
    """
    return {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11,
        'c': 0, 'c#': 1, 'd': 2, 'd#': 3, 'e': 4, 'f': 5,
        'f#': 6, 'g': 7, 'g#': 8, 'a': 9, 'a#': 10, 'b': 11
    }

def referent(pitches: list[int]) -> int:
    '''
    Feature that describes the chromatic interval of the key centre from C.
    '''
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)

    if not correlations:
        return -1

    # Get the key name from the highest correlation
    key_name = correlations[0][0].split()[0]  # Take first word (key name without major/minor)

    # Map key names to semitone distances from C
    key_distances = get_key_distances()

    return key_distances[key_name]

def inscale(pitches: list[int]) -> int:
    '''
    Captures whether the melody contains any notes which deviate from the estimated key.
    '''
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)[0]
    key_centre = correlations[0]

    # Get major/minor scales based on key
    if 'major' in key_centre:
        # Major scale pattern: W-W-H-W-W-W-H (W=2 semitones, H=1 semitone)
        scale = [0, 2, 4, 5, 7, 9, 11]
    else:
        # Natural minor scale pattern: W-H-W-W-H-W-W
        scale = [0, 2, 3, 5, 7, 8, 10]

    # Get key root pitch class
    key_name = key_centre.split()[0]
    key_distances = get_key_distances()
    root = key_distances[key_name]

    # Transpose scale to key
    scale = [(note + root) % 12 for note in scale]

    # Check if any pitch classes are outside the scale
    for pc in pitch_classes:
        if pc not in scale:
            return 0

    return 1

def temperley_likelihood(pitches: list[int]) -> float:
    '''
    Calculates the likelihood of a melody using Bayesian reasoning,
    according to David Temperley's model 
    (http://davidtemperley.com/wp-content/uploads/2015/11/temperley-cs08.pdf).
    '''
    # represent all possible notes as int
    notes_ints = np.arange(0, 120, 1)

    # Calculate central pitch profile
    central_pitch_profile = scipy.stats.norm.pdf(notes_ints, loc=68, scale=np.sqrt(5.0))
    central_pitch = choices(notes_ints, central_pitch_profile)
    range_profile = scipy.stats.norm.pdf(notes_ints, loc=central_pitch, scale=np.sqrt(23.0))

    # Get key probabilities
    rpk_major = [0.184, 0.001, 0.155, 0.003, 0.191, 0.109,
                0.005, 0.214, 0.001, 0.078, 0.004, 0.055] * 10
    rpk_minor = [0.192, 0.005, 0.149, 0.179, 0.002, 0.144,
                0.002, 0.201, 0.038, 0.012, 0.053, 0.022] * 10

    # Calculate total probability
    total_prob = 1.0
    for i in range(1, len(pitches)):
        # Calculate proximity profile centered on previous note
        prox_profile = scipy.stats.norm.pdf(notes_ints, loc=pitches[i-1], scale=np.sqrt(10))
        rp = range_profile * prox_profile

        # Apply key profile based on major/minor
        if 'major' in compute_tonality_vector([p % 12 for p in pitches])[0][0]:
            rpk = rp * rpk_major
        else:
            rpk = rp * rpk_minor

        # Normalize probabilities
        rpk_normed = rpk / np.sum(rpk)

        # Get probability of current note
        note_prob = rpk_normed[pitches[i]]
        total_prob *= note_prob

    return total_prob

def tonalness_histogram(pitches: list[int]) -> dict:
    '''
    Calculates the histogram of KS correlation values.
    '''
    p = [p % 12 for p in pitches]
    return histogram_bins(compute_tonality_vector(p)[0][1], 24)

def get_narmour_features(melody: Melody) -> Dict:
    """Calculate Narmour's implication-realization features.

    Parameters
    ----------
    melody : Melody
        The melody to analyze as a Melody object

    Returns
    -------
    Dict
        Dictionary containing scores for:
        - Registral direction (0 or 1)
        - Proximity (0-6)
        - Closure (0-2)
        - Registral return (0-3)
        - Intervallic difference (0 or 1)

    Notes
    -----
    Features represent:
    - Registral direction: Large intervals followed by direction change
    - Proximity: Closeness of consecutive pitches
    - Closure: Direction changes and interval size changes
    - Registral return: Return to previous pitch level
    - Intervallic difference: Relationship between consecutive intervals
    """
    pitches = melody.pitches
    return {
        'registral_direction': registral_direction(pitches),
        'proximity': proximity(pitches),
        'closure': closure(pitches),
        'registral_return': registral_return(pitches),
        'intervallic_difference': intervallic_difference(pitches)
    }

# Melodic Movement Features
def amount_of_arpeggiation(pitches: list[int]) -> float:
    """Calculate the proportion of notes in the melody that constitute triadic movement.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of intervals that match arpeggio patterns (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.
    """
    return arpeggiation_proportion(pitches)


def chromatic_motion(pitches: list[int]) -> float:
    """Calculate the proportion of chromatic motion in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of intervals that are chromatic (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.
    """
    return chromatic_motion_proportion(pitches)

def melodic_embellishment(pitches: list[int], starts: list[float], ends: list[float]) -> float:
    """Calculate proportion of melodic embellishments (e.g. trills, turns, neighbor tones).

    Melodic embellishments are identified by looking for notes with a duration 1/3rd of the
    adjacent note's duration that move away from and return to a pitch level, or oscillate
    between two pitches.
    

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Proportion of intervals that are embellishments (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.
    """
    return melodic_embellishment_proportion(pitches, starts, ends)

def repeated_notes(pitches: list[int]) -> float:
    """Calculate the proportion of repeated notes in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of intervals that are repeated notes (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.
    """
    return repeated_notes_proportion(pitches)

def stepwise_motion(pitches: list[int]) -> float:
    """Calculate the proportion of stepwise motion in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of intervals that are stepwise (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.
    """
    return stepwise_motion_proportion(pitches)

def check_is_monophonic(melody: Melody) -> bool:
    """Check if the melody is monophonic.

    This function determines if a melody is monophonic by ensuring that no
    notes overlap in time. It assumes the notes within the Melody object are
    sorted by their start times. A melody is considered polyphonic if any
    note starts before the previous note has ended.
    Parameters
    ----------
    melody : Melody
        The melody to analyze as a Melody object.

    Returns
    -------
    bool
        True if the melody is monophonic, False otherwise.
    """
    starts = melody.starts
    ends = melody.ends

    # A melody with 0 or 1 notes can only be monophonic.
    if len(starts) < 2:
        return True

    # otherwise, if start time of current note is less than end time of previous note,
    # the melody cannot be monophonic.
    for i in range(1, len(starts)):
        if starts[i] < ends[i-1]:
            return False

    return True

def get_mtype_features(melody: Melody) -> dict:
    """Calculate various n-gram statistics for the melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze as a Melody object

    Returns
    -------
    dict
        Dictionary containing complexity measures averaged across n-gram lengths
    """
    # Initialize tokenizer and get M-type tokens
    tokenizer = FantasticTokenizer()

    # Segment the melody first, using quarters as the time unit
    segments = tokenizer.segment_melody(melody, phrase_gap=1.5, units="quarters")

    # Get tokens for each segment
    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, 
            segment.starts, 
            segment.ends
        )
        all_tokens.extend(segment_tokens)

    # Create a fresh counter for this melody
    ngram_counter = NGramCounter()
    ngram_counter.ngram_counts = {}  # Explicitly reset the counter

    ngram_counter.count_ngrams(all_tokens)

    # Calculate complexity measures for each n-gram length
    mtype_features = {}

    # Initialize all features to NaN
    mtype_features['yules_k'] = float('nan')
    mtype_features['simpsons_d'] = float('nan')
    mtype_features['sichels_s'] = float('nan')
    mtype_features['honores_h'] = float('nan')
    mtype_features['mean_entropy'] = float('nan')
    mtype_features['mean_productivity'] = float('nan')

    # Try to calculate each feature individually
    if ngram_counter.ngram_counts:
        try:
            mtype_features['yules_k'] = ngram_counter.yules_k
        except Exception as e:
            warnings.warn(f"Error calculating Yule's K: {str(e)}")
        try:
            mtype_features['simpsons_d'] = ngram_counter.simpsons_d
        except Exception as e:
            warnings.warn(f"Error calculating Simpson's D: {str(e)}")

        try:
            mtype_features['sichels_s'] = ngram_counter.sichels_s
        except Exception as e:
            warnings.warn(f"Error calculating Sichel's S: {str(e)}")

        try:
            mtype_features['honores_h'] = ngram_counter.honores_h
        except Exception as e:
            warnings.warn(f"Error calculating Honoré's H: {str(e)}")

        try:
            mtype_features['mean_entropy'] = ngram_counter.mean_entropy
        except Exception as e:
            warnings.warn(f"Error calculating mean entropy: {str(e)}")

        try:
            mtype_features['mean_productivity'] = ngram_counter.mean_productivity
        except Exception as e:
            warnings.warn(f"Error calculating mean productivity: {str(e)}")

    return mtype_features


def get_ngram_document_frequency(ngram: tuple, corpus_stats: dict) -> int:
    """Retrieve the document frequency for a given n-gram from the corpus statistics.
    
    Parameters
    ----------
    ngram : tuple
        The n-gram to look up
    corpus_stats : dict
        Dictionary containing corpus statistics
        
    Returns
    -------
    int
        Document frequency count for the n-gram
    """
    # Get document frequencies dictionary once
    doc_freqs = corpus_stats.get('document_frequencies', {})

    # Convert ngram to string only once
    ngram_str = str(ngram)

    # Look up the count directly
    return doc_freqs.get(ngram_str, {}).get('count', 0)

def get_corpus_features(melody: Melody, corpus_stats: dict) -> Dict:
    """Compute all corpus-based features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
        
    Returns
    -------
    Dict
        Dictionary of corpus-based feature values
    """
    # Pre-compute tokenization and n-gram counts once
    tokenizer = FantasticTokenizer()

    # Segment the melody first
    segments = tokenizer.segment_melody(melody, phrase_gap=1.5, units="quarters")

    # Get tokens for each segment
    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, 
            segment.starts, 
            segment.ends
        )
        all_tokens.extend(segment_tokens)
    
    # Get document frequencies from corpus stats upfront
    doc_freqs = corpus_stats.get('document_frequencies', {})
    total_docs = len(doc_freqs)

    # Pre-compute n-gram counts and document frequencies for all n-gram lengths
    ngram_data = []

    for n in range(1, 6):
        # Count n-grams in the combined tokens
        ngram_counts = {}
        for i in range(len(all_tokens) - n + 1):
            ngram = tuple(all_tokens[i:i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        if not ngram_counts:
            continue

        # Get document frequencies for all n-grams at once
        ngram_df_data = {
            'counts': ngram_counts,
            'total_tf': sum(ngram_counts.values()),
            'df_values': [],
            'tf_values': [],
            'ngrams': []
        }
 
        # Batch lookup document frequencies
        for ngram, tf in ngram_counts.items():
            ngram_str = str(ngram)
            df = doc_freqs.get(ngram_str, {}).get('count', 0)
            if df > 0:
                ngram_df_data['df_values'].append(df)
                ngram_df_data['tf_values'].append(tf)
                ngram_df_data['ngrams'].append(ngram)

        if ngram_df_data['df_values']:
            ngram_data.append(ngram_df_data)

    features = {}

    # Compute correlation features using pre-computed values
    if ngram_data:
        all_tf = []
        all_df = []
        for data in ngram_data:
            all_tf.extend(data['tf_values'])
            all_df.extend(data['df_values'])

        if len(all_tf) >= 2:
            try:
                spearman = scipy.stats.spearmanr(all_tf, all_df)[0]
                kendall = scipy.stats.kendalltau(all_tf, all_df)[0]
                features['tfdf_spearman'] = float(spearman if not np.isnan(spearman) else 0.0)
                features['tfdf_kendall'] = float(kendall if not np.isnan(kendall) else 0.0)
            except:
                features['tfdf_spearman'] = 0.0
                features['tfdf_kendall'] = 0.0
        else:
            features['tfdf_spearman'] = 0.0
            features['tfdf_kendall'] = 0.0
    else:
        features['tfdf_spearman'] = 0.0
        features['tfdf_kendall'] = 0.0

    # Compute TFDF and distance features
    tfdf_values = []
    distances = []
    max_df = 0
    min_df = float('inf')
    total_log_df = 0.0
    df_count = 0

    for data in ngram_data:
        # TFDF calculation
        tf_array = np.array(data['tf_values'])
        df_array = np.array(data['df_values'])
        if len(tf_array) > 0:
            # Normalize vectors
            tf_norm = tf_array / data['total_tf']
            df_norm = df_array / total_docs
            tfdf = np.dot(tf_norm, df_norm)
            tfdf_values.append(tfdf)

            # Distance calculation
            distances.extend(np.abs(tf_norm - df_norm))

            # Track max/min/total log DF
            max_df = max(max_df, max(data['df_values']))
            min_df = min(min_df, min(x for x in data['df_values'] if x > 0))
            total_log_df += np.sum(np.log1p(df_array))
            df_count += len(df_array)

    features['mean_log_tfdf'] = float(np.mean(tfdf_values) if tfdf_values else 0.0)
    features['norm_log_dist'] = float(np.mean(distances) if distances else 0.0)
    features['max_log_df'] = float(np.log1p(max_df) if max_df > 0 else 0.0)
    features['min_log_df'] = float(np.log1p(min_df) if min_df < float('inf') else 0.0)
    features['mean_log_df'] = float(total_log_df / df_count if df_count > 0 else 0.0)

    return features

def get_pitch_features(melody: Melody) -> Dict:
    """Compute all pitch-based features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of pitch-based feature values
    
    """
    pitch_features = {}

    pitch_features['pitch_range'] = pitch_range(melody.pitches)
    pitch_features['pitch_standard_deviation'] = pitch_standard_deviation(melody.pitches)
    pitch_features['pitch_entropy'] = pitch_entropy(melody.pitches)
    pitch_features['pcdist1'] = pcdist1(melody.pitches, melody.starts, melody.ends)
    pitch_features['basic_pitch_histogram'] = basic_pitch_histogram(melody.pitches)
    pitch_features['mean_pitch'] = mean_pitch(melody.pitches)
    pitch_features['most_common_pitch'] = most_common_pitch(melody.pitches)
    pitch_features['number_of_pitches'] = number_of_pitches(melody.pitches)
    pitch_features['melodic_pitch_variety'] = melodic_pitch_variety(melody.pitches)
    pitch_features['dominant_spread'] = dominant_spread(melody.pitches)
    pitch_features['folded_fifths_pitch_class_histogram'] = folded_fifths_pitch_class_histogram(melody.pitches)
    pitch_features['pitch_class_kurtosis_after_folding'] = pitch_class_kurtosis_after_folding(melody.pitches)
    pitch_features['pitch_class_skewness_after_folding'] = pitch_class_skewness_after_folding(melody.pitches)
    pitch_features['pitch_class_variability_after_folding'] = pitch_class_variability_after_folding(melody.pitches)

    return pitch_features

def get_interval_features(melody: Melody) -> Dict:
    """Compute all interval-based features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of interval-based feature values
    
    """
    interval_features = {}

    interval_features['pitch_interval'] = pitch_interval(melody.pitches)
    interval_features['absolute_interval_range'] = absolute_interval_range(melody.pitches)
    interval_features['mean_absolute_interval'] = mean_absolute_interval(melody.pitches)
    interval_features['modal_interval'] = modal_interval(melody.pitches)
    interval_features['interval_entropy'] = interval_entropy(melody.pitches)
    interval_features['ivdist1'] = ivdist1(melody.pitches, melody.starts, melody.ends)
    direction_mean, direction_sd = interval_direction(melody.pitches)
    interval_features['interval_direction_mean'] = direction_mean
    interval_features['interval_direction_sd'] = direction_sd
    interval_features['average_interval_span_by_melodic_arcs'] = average_interval_span_by_melodic_arcs(melody.pitches)
    interval_features['distance_between_most_prevalent_melodic_intervals'] = distance_between_most_prevalent_melodic_intervals(melody.pitches)
    interval_features['melodic_interval_histogram'] = melodic_interval_histogram(melody.pitches)
    interval_features['melodic_large_intervals'] = melodic_large_intervals(melody.pitches)
    interval_features['variable_melodic_intervals'] = variable_melodic_intervals(melody.pitches, 7)  # TODO: Add more interval levels
    interval_features['number_of_common_melodic_intervals'] = number_of_common_melodic_intervals(melody.pitches)
    interval_features['prevalence_of_most_common_melodic_interval'] = prevalence_of_most_common_melodic_interval(melody.pitches)

    return interval_features

def get_contour_features(melody: Melody) -> Dict:
    """Compute all contour-based features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of contour-based feature values
    
    """
    contour_features = {}

    # Calculate step contour features
    step_contour = get_step_contour_features(melody.pitches, melody.starts, melody.ends)
    contour_features['step_contour_global_variation'] = step_contour[0]
    contour_features['step_contour_global_direction'] = step_contour[1]
    contour_features['step_contour_local_variation'] = step_contour[2]

    # Calculate interpolation contour features
    interpolation_contour = get_interpolation_contour_features(melody.pitches, melody.starts)
    contour_features['interpolation_contour_global_direction'] = interpolation_contour[0]
    contour_features['interpolation_contour_mean_gradient'] = interpolation_contour[1]
    contour_features['interpolation_contour_gradient_std'] = interpolation_contour[2]
    contour_features['interpolation_contour_direction_changes'] = interpolation_contour[3]
    contour_features['interpolation_contour_class_label'] = interpolation_contour[4]

    return contour_features

def get_duration_features(melody: Melody) -> Dict:
    """Compute all duration-based features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of duration-based feature values
    
    """
    duration_features = {}
    duration_features['tempo'] = get_tempo(melody)
    duration_features['duration_range'] = duration_range(melody.starts, melody.ends)
    duration_features['modal_duration'] = modal_duration(melody.starts, melody.ends)
    duration_features['mean_duration'] = mean_duration(melody.starts, melody.ends)
    duration_features['duration_standard_deviation'] = duration_standard_deviation(melody.starts, melody.ends)
    duration_features['number_of_durations'] = number_of_durations(melody.starts, melody.ends)
    duration_features['global_duration'] = global_duration(melody.starts, melody.ends)
    duration_features['note_density'] = note_density(melody.starts, melody.ends)
    duration_features['duration_entropy'] = duration_entropy(melody.starts, melody.ends)
    duration_features['length'] = length(melody.starts)
    duration_features['note_density'] = note_density(melody.starts, melody.ends)
    duration_features['ioi_mean'] = ioi_mean(melody.starts)
    duration_features['ioi_std'] = ioi_standard_deviation(melody.starts)
    ioi_ratio_mean, ioi_ratio_std = ioi_ratio(melody.starts)
    duration_features['ioi_ratio_mean'] = ioi_ratio_mean
    duration_features['ioi_ratio_std'] = ioi_ratio_std
    ioi_contour_mean, ioi_contour_std = ioi_contour(melody.starts)
    duration_features['ioi_contour_mean'] = ioi_contour_mean
    duration_features['ioi_contour_std'] = ioi_contour_std
    duration_features['ioi_range'] = ioi_range(melody.starts)
    duration_features['ioi_histogram'] = ioi_histogram(melody.starts)
    duration_features['duration_histogram'] = duration_histogram(melody.starts, melody.ends)
    return duration_features

def get_tonality_features(melody: Melody) -> Dict:
    """Compute all tonality-based features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of tonality-based feature values
    
    """
    tonality_features = {}

    # Pre-compute pitch classes and tonality vector once
    pitches = melody.pitches
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)

    # Pre-compute absolute correlation values
    abs_correlations = [(key, abs(val)) for key, val in correlations]
    abs_corr_values = [val for _, val in abs_correlations]

    # Basic tonality features using cached correlations
    tonality_features['tonalness'] = abs_corr_values[0]

    if len(correlations) >= 2:
        tonality_features['tonal_clarity'] = abs_corr_values[0] / abs_corr_values[1] if abs_corr_values[1] != 0 else 1.0
        other_sum = sum(abs_corr_values[1:])
        tonality_features['tonal_spike'] = abs_corr_values[0] / other_sum if other_sum != 0 else 1.0
    else:
        tonality_features['tonal_clarity'] = -1.0
        tonality_features['tonal_spike'] = -1.0

    # Entropy using cached values
    tonality_features['tonal_entropy'] = shannon_entropy(abs_corr_values) if correlations else -1.0

    # Key-based features using cached correlations
    if correlations:
        key_name = correlations[0][0].split()[0]
        key_distances = get_key_distances()
        root = key_distances[key_name]
        tonality_features['referent'] = root

        # Determine scale type and pattern
        is_major = 'major' in correlations[0][0]
        scale = [0, 2, 4, 5, 7, 9, 11] if is_major else [0, 2, 3, 5, 7, 8, 10]
        scale = [(note + root) % 12 for note in scale]

        # Check if all notes are in scale
        tonality_features['inscale'] = int(all(pc in scale for pc in pitch_classes))
    else:
        tonality_features['referent'] = -1
        tonality_features['inscale'] = 0

    # Optimize temperley_likelihood calculation
    if len(pitches) > 1:
        # Pre-compute constant arrays
        notes_ints = np.arange(0, 120)
        central_pitch_profile = scipy.stats.norm.pdf(notes_ints, loc=68, scale=np.sqrt(5.0))
        central_pitch = np.random.choice(notes_ints, p=central_pitch_profile/central_pitch_profile.sum())
        range_profile = scipy.stats.norm.pdf(notes_ints, loc=central_pitch, scale=np.sqrt(23.0))

        # Pre-compute key profile
        rpk = (np.array([0.184, 0.001, 0.155, 0.003, 0.191, 0.109, 0.005, 0.214, 0.001, 0.078, 0.004, 0.055] * 10) 
               if is_major else
               np.array([0.192, 0.005, 0.149, 0.179, 0.002, 0.144, 0.002, 0.201, 0.038, 0.012, 0.053, 0.022] * 10))

        # Vectorize probability calculation
        total_prob = 1.0
        prev_pitches = np.array(pitches[:-1])
        curr_pitches = np.array(pitches[1:])

        # Calculate all proximity profiles at once
        prox_profiles = scipy.stats.norm.pdf(notes_ints[:, np.newaxis], 
                                           loc=prev_pitches, 
                                           scale=np.sqrt(10))

        # Calculate probabilities for each note transition
        for i in range(len(prev_pitches)):
            rp = range_profile * prox_profiles[:, i]
            rpk_combined = rp * rpk
            rpk_normed = rpk_combined / np.sum(rpk_combined)
            total_prob *= rpk_normed[curr_pitches[i]]

        tonality_features['temperley_likelihood'] = total_prob
    else:
        tonality_features['temperley_likelihood'] = 0.0

    # Scalar passage features
    tonality_features['longest_monotonic_conjunct_scalar_passage'] = longest_monotonic_conjunct_scalar_passage(pitches)
    tonality_features['longest_conjunct_scalar_passage'] = longest_conjunct_scalar_passage(pitches)
    tonality_features['proportion_conjunct_scalar'] = proportion_conjunct_scalar(pitches)
    tonality_features['proportion_scalar'] = proportion_scalar(pitches)

    # Histogram using cached correlations
    tonality_features['tonalness_histogram'] = histogram_bins(correlations[0][1], 24)

    return tonality_features

def get_melodic_movement_features(melody: Melody) -> Dict:
    """Compute all melodic movement-based features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of melodic movement-based feature values
    
    """
    movement_features = {}

    movement_features['amount_of_arpeggiation'] = amount_of_arpeggiation(melody.pitches)
    movement_features['chromatic_motion'] = chromatic_motion(melody.pitches)
    movement_features['melodic_embellishment'] = melodic_embellishment(melody.pitches, melody.starts, melody.ends)
    movement_features['repeated_notes'] = repeated_notes(melody.pitches)
    movement_features['stepwise_motion'] = stepwise_motion(melody.pitches)

    return movement_features

def process_melody(args):
    """Process a single melody and return its features.
    
    Parameters
    ----------
    args : tuple
        Tuple containing (melody_data, corpus_stats, idyom_features)
    
    Returns
    -------
    tuple
        Tuple containing (melody_id, feature_dict, timings)
    """
    import time
    start_total = time.time()

    melody_data, corpus_stats, idyom_features = args
    mel = Melody(melody_data, tempo=100)

    # Time each feature category
    timings = {}

    start = time.time()
    pitch_features = get_pitch_features(mel)
    timings['pitch'] = time.time() - start

    start = time.time()
    interval_features = get_interval_features(mel)
    timings['interval'] = time.time() - start

    start = time.time()
    contour_features = get_contour_features(mel)
    timings['contour'] = time.time() - start

    start = time.time()
    duration_features = get_duration_features(mel)
    timings['duration'] = time.time() - start

    start = time.time()
    tonality_features = get_tonality_features(mel)
    timings['tonality'] = time.time() - start

    start = time.time()
    narmour_features = get_narmour_features(mel)
    timings['narmour'] = time.time() - start

    start = time.time()
    melodic_movement_features = get_melodic_movement_features(mel)
    timings['melodic_movement'] = time.time() - start

    start = time.time()
    mtype_features = get_mtype_features(mel)
    timings['mtype'] = time.time() - start

    melody_features = {
        'pitch_features': pitch_features,
        'interval_features': interval_features,
        'contour_features': contour_features,
        'duration_features': duration_features,
        'tonality_features': tonality_features,
        'narmour_features': narmour_features,
        'melodic_movement_features': melodic_movement_features,
        'mtype_features': mtype_features
    }

    # Add corpus features only if corpus stats are available
    if corpus_stats:
        start = time.time()
        melody_features['corpus_features'] = get_corpus_features(mel, corpus_stats)
        timings['corpus'] = time.time() - start

    # Add pre-computed IDyOM features if available for this melody's ID
    melody_id_str = str(melody_data['ID'])
    if idyom_features and melody_id_str in idyom_features:
        melody_features['idyom_features'] = idyom_features[melody_id_str]

    timings['total'] = time.time() - start_total

    return melody_data['ID'], melody_features, timings

def get_idyom_results(input_path, corpus_path=None) -> dict:
    """Run IDyOM on the input MIDI directory and return mean information content for each melody.

    Parameters
    ----------
    input_path : str
        Path to input MIDI directory
    
    Returns
    -------
    dict
        A dictionary mapping melody IDs to their mean information content.
    """
    dat_file_path = run_idyom(input_path,
            pretraining_path=corpus_path,
            output_dir='.',
            description="IDyOM_Feature_Set_Results",
            target_viewpoints=['cpitch'],
            source_viewpoints=[('cpint', 'cpintfref')],
            models=':both',
            detail=2)

    if not dat_file_path:
        print("Warning: run_idyom did not produce an output file. Skipping IDyOM features.")
        return {}

    # Get a naturally sorted list of MIDI files to match IDyOM's processing order.
    midi_files = natsorted(glob.glob(os.path.join(input_path, '*.mid')))
    
    idyom_results = {}
    try:
        with open(dat_file_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue # Skip malformed lines

                try:
                    # IDyOM's melody ID is a 1-based index.
                    melody_idx = int(parts[0]) - 1
                    mean_ic = float(parts[2])
                    
                    if 0 <= melody_idx < len(midi_files):
                        # Map the index to the actual filename.
                        melody_id = os.path.basename(midi_files[melody_idx])
                        idyom_results[melody_id] = {'mean_information_content': mean_ic}
                    else:
                        print(f"Warning: IDyOM returned an out-of-bounds index: {parts[0]}")
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line in IDyOM output: '{line.strip()}'. Error: {e}")

        os.remove(dat_file_path)

    except FileNotFoundError:
        print(f"Warning: IDyOM output file not found at {dat_file_path}. Skipping IDyOM features.")
        return {}
    except Exception as e:
        print(f"Error parsing IDyOM output file: {e}. Skipping IDyOM features.")
        if os.path.exists(dat_file_path):
            os.remove(dat_file_path)
        return {}
            
    return idyom_results

def get_all_features(input_path, output_path, corpus_path=None) -> None:
    """Calculate a multitude of features from across the computational melody analysis field.
    This function generates a CSV file with a row for every melody in the supplied input 
    directory of MIDI files. 
    If a path to a corpus of MIDI files is provided, corpus statistics will be computed following
    FANTASTIC's n-gram document frequency model (Müllensiefen, 2009). If not, this will be skipped.
    This function will also run IDyOM (Pearce, 2009) on the input directory of MIDI files. If a corpus
    of MIDI files is provided, IDyOM will be run with pretraining on the corpus. If not, it will be
    run without pretraining.

    Parameters
    ----------
    input_path : str
        Path to input MIDI directory
    output_path : str
        Name for output CSV file. If no extension is provided, .csv will be added.
    corpus_path : str, optional
        Path to corpus of MIDI files. If not provided, corpus statistics will not be computed
        and IDyOM will not be run with pretraining.

    Returns
    -------
    A CSV file with a row for every melody in the input directory.
    
    """
    # Ensure output_path has .csv extension
    if not output_path.endswith('.csv'):
        output_path = output_path + '.csv'
        
    import threading
    import time

    # Spinner animation thread
    class SpinnerThread(threading.Thread):
        def __init__(self):
            super().__init__()
            self.stop_event = threading.Event()
            self.spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            self.idx = 0

        def run(self):
            while not self.stop_event.is_set():
                print(f"\r{self.spinner[self.idx]} Processing melodies...", end='', flush=True)
                self.idx = (self.idx + 1) % len(self.spinner)
                time.sleep(0.1)  # Control animation speed

        def stop(self):
            self.stop_event.set()
            self.join()
            print("\r", end='', flush=True)  # Clear the spinner line
            print("\n")

    print("Starting job...\n")

    # --- Corpus Statistics Generation ---
    corpus_stats = None
    if corpus_path:
        if not Path(corpus_path).is_dir():
            raise FileNotFoundError(f"Corpus path is not a valid directory: {corpus_path}")

        print(f"--- Generating corpus statistics from: {corpus_path} ---")

        # Define a persistent path for the corpus stats file.
        corpus_name = Path(corpus_path).name
        corpus_stats_path = Path(output_path).parent / f"{corpus_name}_corpus_stats.json"
        print(f"Corpus statistics file will be at: {corpus_stats_path}")

        # Generate and load corpus stats.
        if not corpus_stats_path.exists():
            print("Corpus statistics file not found. Generating a new one...")
            make_corpus_stats(corpus_path, str(corpus_stats_path))
            print("Corpus statistics generated.")
        else:
            print("Existing corpus statistics file found.")
            
        corpus_stats = load_corpus_stats(str(corpus_stats_path))
        print("Corpus statistics loaded successfully.")
    else:
        print("No corpus path provided, corpus-dependent features will not be computed.")

    # --- Load Melody Data ---
    melody_data_list = []
    import os

    if os.path.isdir(input_path):
        # Handle directory of MIDI files
        import glob
        midi_files = glob.glob(os.path.join(input_path, '*.mid'))
        midi_files.extend(glob.glob(os.path.join(input_path, '*.midi')))

        if not midi_files:
            raise FileNotFoundError(f"No MIDI files found in the specified directory: {input_path}")

        # Sort MIDI files in natural order
        from natsort import natsorted
        midi_files = natsorted(midi_files)

        # Process MIDI files in parallel
        with Pool(cpu_count()) as pool:
            for midi_file in midi_files:
                try:
                    midi_data = import_midi(midi_file)
                    if midi_data:
                        # Perform monophonic check before adding to the list.
                        temp_mel = Melody(midi_data)
                        if check_is_monophonic(temp_mel):
                            melody_data_list.append(midi_data)
                        else:
                            print(f"Warning: Skipping polyphonic file: {midi_file}")
                except Exception as e:
                    print(f"Error importing {midi_file}: {str(e)}")
                    continue
    elif input_path.endswith('.json'):
        # Handle JSON file
        with open(input_path, encoding='utf-8') as f:
            all_data = json.load(f)
        
        # Filter for monophonic melodies from the JSON data.
        for melody_data in all_data:
            if melody_data:
                temp_mel = Melody(melody_data)
                if check_is_monophonic(temp_mel):
                    melody_data_list.append(melody_data)
                else:
                    print(f"Warning: Skipping polyphonic melody from JSON: {melody_data.get('ID', 'Unknown ID')}")

    else:
        raise ValueError(f"Input path must be either a directory containing MIDI files or a JSON file. Got: {input_path}")
    
    melody_data_list = [m for m in melody_data_list if m is not None]
    print(f"Processing {len(melody_data_list)} melodies")

    if not melody_data_list:
        print("No valid monophonic melodies found to process.")
        return

    # Assign unique melody_num to each melody (in sorted order)
    for idx, melody_data in enumerate(melody_data_list, 1):
        melody_data['melody_num'] = idx

    # --- Run IDyOM Analysis (must happen BEFORE parallel processing setup) ---
    idyom_results = {} # Initialize as an empty dict
    try:
        if corpus_path:
            print("\nRunning IDyOM analysis with pretraining on corpus")
            idyom_results = get_idyom_results(input_path, corpus_path)
        else:
            print("\nRunning IDyOM analysis without pretraining")
            idyom_results = get_idyom_results(input_path)

    
    except Exception as e:
        print(f"\n--- IDyOM analysis failed ---")
        print(f"Error during IDyOM processing: {e}")
        print("Skipping IDyOM features.")

    start_time = time.time()

    # --- Setup for Parallel Processing ---
    # Process first melody to get header structure
    mel = Melody(melody_data_list[0], tempo=100)
    first_features = {
        'pitch_features': get_pitch_features(mel),
        'interval_features': get_interval_features(mel),
        'contour_features': get_contour_features(mel),
        'duration_features': get_duration_features(mel),
        'tonality_features': get_tonality_features(mel),
        'narmour_features': get_narmour_features(mel),
        'melodic_movement_features': get_melodic_movement_features(mel),
        'mtype_features': get_mtype_features(mel)
    }

    # Add corpus features only if corpus stats are available
    if corpus_stats:
        first_features['corpus_features'] = get_corpus_features(mel, corpus_stats)
    
    # Add IDyOM features to the header if they were generated
    if idyom_results:
        # Get a sample of the features to build the header
        sample_id = next(iter(idyom_results))
        first_features['idyom_features'] = idyom_results[sample_id]

    # Create header by flattening feature names
    headers = ['melody_num', 'melody_id']
    for category, features in first_features.items():
        headers.extend(f"{category}.{feature}" for feature in features.keys())

    print("Starting parallel processing...\n")
    # Create pool of workers
    n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores")

    # Prepare arguments for parallel processing
    melody_args = [(melody_data, corpus_stats, idyom_results) for melody_data in melody_data_list]

    # Process melodies in parallel with chunking for better performance
    chunk_size = max(1, len(melody_args) // (n_cores * 4))  # Adjust chunk size based on number of cores
    all_features = []

    # Track timing statistics
    timing_stats = {
        'pitch': [],
        'interval': [],
        'contour': [],
        'duration': [],
        'tonality': [],
        'narmour': [],
        'melodic_movement': [],
        'mtype': [],
        'corpus': [],
        'total': []
    }

    # Start spinner thread
    spinner = SpinnerThread()
    spinner.start()

    try:
        with Pool(n_cores) as pool:
            # Use imap for better memory efficiency and progress tracking
            for i, result in enumerate(pool.imap(process_melody, melody_args, chunksize=chunk_size)):
                try:
                    melody_id, melody_features, timings = result
                    # Find the melody_num for this melody_id
                    melody_num = None
                    for m in melody_data_list:
                        if str(m['ID']) == str(melody_id):
                            melody_num = m.get('melody_num', None)
                            break
                    row = [melody_num, melody_id]
                    # Loop through the headers to ensure correct order and handle missing data
                    for header in headers[2:]: # Skip melody_num and melody_id headers
                        category, feature_name = header.split('.', 1)
                        value = melody_features.get(category, {}).get(feature_name, 0.0)
                        row.append(value)
                    all_features.append(row)

                    # Update timing statistics
                    for category, duration in timings.items():
                        timing_stats[category].append(duration)
                except Exception as e:
                    print(f"\nError processing melody {i}: {str(e)}")
                    continue
    finally:
        # Stop spinner thread
        spinner.stop()

    if not all_features:
        print("No features were successfully extracted from any melodies")
        return

    print("Processing complete")

    # Sort results by melody_id
    all_features.sort(key=lambda x: x[0])

    # Write results to CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_features)

    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
    print(f"Results written to {output_path}")

    # Print timing statistics
    print("\nTiming Statistics (average milliseconds per melody):")
    for category, times in timing_stats.items():
        if times:  # Only print if we have timing data
            avg_time = sum(times) / len(times) * 1000  # Convert to milliseconds
            print(f"{category:15s}: {avg_time:8.2f}ms")

