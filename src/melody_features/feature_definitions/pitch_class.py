"""Pitch class feature definitions."""

import numpy as np

from ..algorithms import circle_of_fifths, n_percent_significant_values
from ..feature_decorators import jsymbolic, midi_toolbox, pitch, pitch_class
from ..feature_histogram import PitchClassHistogram
from ..meter_estimation import duration_accent
from ..stats import get_mode


__all__ = [
    "pitch_class_variability",
    "pitch_class_variability_after_folding",
    "pcdist1",
    "first_pitch_class",
    "last_pitch_class",
    "dominant_spread",
    "mean_pitch_class",
    "most_common_pitch_class",
    "number_of_unique_pitch_classes",
    "number_of_common_pitch_classes",
    "number_of_common_pitches_classes",
    "prevalence_of_most_common_pitch_class",
    "relative_prevalence_of_top_pitch_classes",
    "interval_between_most_prevalent_pitch_classes",
    "folded_fifths_pitch_class_histogram",
    "pitch_class_skewness",
    "pitch_class_kurtosis",
    "pitch_class_skewness_after_folding",
    "pitch_class_kurtosis_after_folding",
    "strong_tonal_centres",
]


@jsymbolic
@pitch_class
@pitch
def pitch_class_variability(pitches: list[int]) -> float:
    """Standard deviation of all pitch classes in the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of pitch class values
    """
    if not pitches or len(pitches) < 2:
        return 0.0
    pcs = [int(p % 12) for p in pitches]
    return float(np.std(pcs, ddof=1))

@jsymbolic
@pitch_class
@pitch
def pitch_class_variability_after_folding(pitches: list[int]) -> float:
    """Standard deviation of all pitch classes after arranging the pitch classes by perfect fifths.
    Provides a measure of how close the pitch classes are as a whole from the mean pitch class from a 
    dominant-tonic perspective.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of folded pitch class values
    """
    if not pitches:
        return 0.0
    
    if not pitches or len(pitches) < 2:
        return 0.0
    folded_pcs = [int((7 * (p % 12)) % 12) for p in pitches]
    return float(np.std(folded_pcs, ddof=1))

def _pcdist1_vector(pitches: list[int], starts: list[float], ends: list[float]) -> np.ndarray:
    pcd = np.zeros(12, dtype=float)
    if not pitches or not starts or not ends:
        return pcd
    accents = duration_accent(starts, ends)
    n = min(len(pitches), len(accents))
    for pitch, acc in zip(pitches[:n], accents[:n]):
        pcd[int(pitch) % 12] += acc
    return pcd / (pcd.sum() + 1e-12)

@midi_toolbox
@pitch_class
@pitch
def pcdist1(pitches: list[int], starts: list[float], ends: list[float]) -> dict:
    """Pitch-class distribution weighted by Parncutt duration accent.

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
    dict
        Map from pitch class (0–11) to proportion
    """
    if not pitches or not starts or not ends:
        return {}
    vec = _pcdist1_vector(pitches, starts, ends)
    return {i: float(vec[i]) for i in range(12) if vec[i] > 0}

@jsymbolic
@pitch_class
@pitch
def first_pitch_class(pitches: list[int]) -> int:
    """The first pitch class in the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    
    Returns
    -------
    int - between 0 and 11
        First pitch class in the melody
    """
    if not pitches:
        return 0
    return int(pitches[0] % 12)

@jsymbolic
@pitch_class
@pitch
def last_pitch_class(pitches: list[int]) -> int:
    """The last pitch class in the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    
    Returns
    -------
    int - between 0 and 11
    """
    if not pitches:
        return 0
    return int(pitches[-1] % 12)

def _consecutive_fifths(pitch_classes: list[int]) -> list[int]:
    """Find longest sequence of pitch classes separated by perfect fifths.

    Parameters
    ----------
    pitch_classes : list[int]
        List of pitch classes (0-11)

    Returns
    -------
    list[int]
        Longest sequence of consecutive pitch classes separated by perfect fifths
    """
    if not pitch_classes:
        return []

    circle_of_fifths_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    
    longest_sequence = [pitch_classes[0]]
    current_sequence = [pitch_classes[0]]
    
    for i in range(1, len(pitch_classes)):
        pc = pitch_classes[i]
        last_pc = current_sequence[-1]
        
        # Check if current PC is a fifth away from the last PC with wraparound
        if (circle_of_fifths_order.index(pc) - circle_of_fifths_order.index(last_pc)) % 12 == 1:
            current_sequence.append(pc)
        else:
            if len(current_sequence) > len(longest_sequence):
                longest_sequence = current_sequence[:]
            current_sequence = [pc]

    if len(current_sequence) > len(longest_sequence):
        longest_sequence = current_sequence[:]
    
    return longest_sequence

@jsymbolic
@pitch_class
@pitch
def dominant_spread(pitches: list[int]) -> int:
    """The longest sequence of pitch classes separated by perfect 5ths that each appear >9% of the time.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Length of longest sequence of significant pitch classes separated by perfect 5ths
    """
    pcs = [pitch % 12 for pitch in pitches]
    pc_counts = {}
    for pc in pcs:
        pc_counts[pc] = pc_counts.get(pc, 0) + 1

    total_notes = len(pcs)
    threshold = 0.09

    significant_pcs = []
    for pc, count in pc_counts.items():
        if count / total_notes >= threshold:
            significant_pcs.append(pc)

    if not significant_pcs:
        return 0

    circle_of_fifths_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

    test_sequence = []
    for pc in circle_of_fifths_order:
        if pc in significant_pcs:
            test_sequence.append(pc)

    if test_sequence:
        test_sequence = test_sequence * 2

    longest_sequence = _consecutive_fifths(test_sequence)

    return len(longest_sequence)

@jsymbolic
@pitch_class
@pitch
def mean_pitch_class(pitches: list[int]) -> float:
    """The arithmetic mean of the pitch classes in the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Linear arithmetic mean pitch class value (between 0 and 11)

    Note
    ----
    This is a linear (non-circular) mean over pitch classes. For example, pitch
    classes near the wraparound boundary (e.g., 11 and 0) are averaged
    numerically rather than on the unit circle.
    """
    if not pitches:
        return 0.0
    return float(np.mean([pitch % 12 for pitch in pitches]))

@jsymbolic
@pitch_class
@pitch
def most_common_pitch_class(pitches: list[int]) -> int:
    """The most frequently occurring pitch class in the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Most common pitch class value
    """
    if not pitches:
        return 0
    return int(get_mode([pitch % 12 for pitch in pitches]))

@jsymbolic
@pitch_class
@pitch
def number_of_unique_pitch_classes(pitches: list[int]) -> int:
    """The number of unique pitch classes in the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of unique pitch classes
    """
    return int(len(set([pitch % 12 for pitch in pitches])))

@jsymbolic
@pitch_class
@pitch
def number_of_common_pitch_classes(pitches: list[int]) -> int:
    """The number of pitch classes that appear in at least 20% of total notes.

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
    significant_pcs = n_percent_significant_values(pcs, threshold=0.2)
    return int(len(significant_pcs))

number_of_common_pitches_classes = number_of_common_pitch_classes

@jsymbolic
@pitch_class
@pitch
def prevalence_of_most_common_pitch_class(pitches: list[int]) -> float:
    """The proportion of pitch classes that are the most common pitch class with regards to the
    total number of pitch classes in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of most common pitch class
    """
    if not pitches:
        return 0.0
    pcs = [pitch % 12 for pitch in pitches]
    return float(pcs.count(most_common_pitch_class(pcs)) / len(pcs))

@jsymbolic
@pitch_class
@pitch
def relative_prevalence_of_top_pitch_classes(pitches: list[int]) -> float:
    """The ratio of the frequency of the second most common pitch class to the frequency of the most common pitch class.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Ratio of second most common pitch class frequency to most common pitch class frequency
    """
    if len(pitches) < 2:
        return 0.0

    pcs = [pitch % 12 for pitch in pitches]
    if len(pcs) < 2:
        return 0.0

    pc_counts = {}
    for pc in pcs:
        pc_counts[pc] = pc_counts.get(pc, 0) + 1

    if len(pc_counts) < 2:
        return 0.0

    sorted_pcs = sorted(pc_counts.items(), key=lambda x: x[1], reverse=True)
    most_common_freq = sorted_pcs[0][1] / len(pcs)
    second_most_freq = sorted_pcs[1][1] / len(pcs)

    return float(second_most_freq / most_common_freq)

@jsymbolic
@pitch_class
@pitch
def interval_between_most_prevalent_pitch_classes(pitches: list[int]) -> int:
    """The number of semitones between the two most prevalent pitch classes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of semitones between the most prevalent pitch classes
    """
    if not pitches:
        return 0

    pch = PitchClassHistogram(pitches)
    histogram = pch.histogram
    if not histogram or sum(1 for v in histogram.values() if v > 0) < 2:
        return 0

    max_index = max(histogram, key=lambda k: histogram[k])
    tmp = dict(histogram)
    tmp.pop(max_index, None)
    if not tmp:
        return 0
    second_max_index = max(tmp, key=lambda k: tmp[k])

    diff = abs(int(max_index) - int(second_max_index))
    return int(diff)

@jsymbolic
@pitch_class
@pitch
def folded_fifths_pitch_class_histogram(pitches: list[int]) -> dict:
    """A histogram of pitch classes arranged according to the circle of fifths.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    dict
        Dictionary mapping pitch classes to counts, arranged according to the circle of fifths
        (using jSymbolic's Folded Fifths Pitch Class Histogram)
    """
    # again, we don't use the histogram object for this one to simplify the output
    pcs = [pitch % 12 for pitch in pitches]
    unique = []
    counts = []
    for pc in set(pcs):
        unique.append(pc)
        counts.append(pcs.count(pc))
    return circle_of_fifths(unique, counts)

@jsymbolic
@pitch_class
@pitch
def pitch_class_skewness(pitches: list[int]) -> float:
    """The skewness of the pitch class histogram, using Pearson's median skewness formula.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Skewness of pitch class histogram values, or 0 for empty input
    """
    if not pitches:
        return 0.0
    
    histogram = PitchClassHistogram(pitches, folded=False)
    return histogram.skewness

@jsymbolic
@pitch_class
@pitch
def pitch_class_kurtosis(pitches: list[int]) -> float:
    """The sample excess kurtosis of the pitch class histogram.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Kurtosis of pitch class histogram values, or 0 for empty input
    """
    if not pitches:
        return 0.0

    histogram = PitchClassHistogram(pitches, folded=False)
    return histogram.kurtosis

@jsymbolic
@pitch_class
@pitch
def pitch_class_skewness_after_folding(pitches: list[int]) -> float:
    """The skewness of the pitch class histogram, using Pearson's median skewness formula, 
    after arranging the pitch classes according to the circle of fifths.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Skewness of folded fifths histogram values, or 0 for empty input
    """
    if not pitches:
        return 0.0
    
    histogram = PitchClassHistogram(pitches, folded=True)
    return histogram.skewness

@jsymbolic
@pitch_class
@pitch
def pitch_class_kurtosis_after_folding(pitches: list[int]) -> float:
    """The sample excess kurtosis of the pitch class histogram, after arranging 
    the pitch classes according to the circle of fifths.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Kurtosis of folded fifths histogram values, or 0 for empty input
    """
    if not pitches:
        return 0.0
    
    histogram = PitchClassHistogram(pitches, folded=True)
    return histogram.kurtosis

@jsymbolic
@pitch_class
@pitch
def strong_tonal_centres(pitches: list[int]) -> float:
    """Counts the number of isolated peaks in the pitch class histogram that each account for at least 9% of notes, 
    arranged according to the circle of fifths.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Number of strong tonal centres (peaks >= 9% in fifths histogram)
    """
    if not pitches:
        return 0.0

    fifths_histogram = PitchClassHistogram(pitches, folded=True)
    fifths_hist = fifths_histogram.histogram

    total_notes = sum(fifths_hist.values())
    if total_notes == 0:
        return 0.0

    normalized_fifths = [fifths_hist[i] / total_notes for i in range(12)]

    peaks = 0
    for bin in range(12):
        if normalized_fifths[bin] >= 0.09:
            left = (bin - 1) % 12
            right = (bin + 1) % 12

            if (normalized_fifths[bin] > normalized_fifths[left] and 
                normalized_fifths[bin] > normalized_fifths[right]):
                peaks += 1

    return float(peaks)
