"""Pitch interval feature definitions."""

import math
from typing import Union

import numpy as np

from ..algorithms import (
    arpeggiation_proportion,
    chromatic_motion_proportion,
    n_percent_significant_values,
)
from ..feature_utils import prevalence_of_mode, relative_prevalence_top_two
from ..utils.distributional import distribution_proportions
from ..feature_decorators import fantastic, interval, jsymbolic, midi_toolbox, novel, pitch, simile
from ..feature_histogram import create_melodic_interval_histogram
from ..algorithms.meter_estimation import duration_accent
from ..utils.stats import get_mode, range_func


__all__ = [
    "pitch_interval",
    "absolute_interval_range",
    "mean_absolute_interval",
    "mean_melodic_interval",
    "standard_deviation_absolute_interval",
    "modal_interval",
    "most_common_interval",
    "ivdist1",
    "ivdirdist1",
    "ivsizedist1",
    "interval_direction",
    "interval_direction_mean",
    "interval_direction_std",
    "average_length_of_melodic_arcs",
    "average_interval_span_by_melodic_arcs",
    "distance_between_most_prevalent_melodic_intervals",
    "melodic_interval_histogram",
    "melodic_large_intervals",
    "variable_melodic_intervals",
    "melodic_thirds",
    "melodic_perfect_fourths",
    "melodic_tritones",
    "melodic_perfect_fifths",
    "melodic_sixths",
    "melodic_sevenths",
    "melodic_octaves",
    "minor_major_third_ratio",
    "direction_of_melodic_motion",
    "number_of_common_melodic_intervals",
    "prevalence_of_most_common_melodic_interval",
    "relative_prevalence_of_most_common_melodic_intervals",
    "amount_of_arpeggiation",
    "chromatic_motion",
]


def _ivdist1_vector(pitches: list[int], starts: list[float], ends: list[float]) -> np.ndarray:
    ivd = np.zeros(25, dtype=float)
    if len(pitches) < 2 or not starts or not ends:
        return ivd
    accents = duration_accent(starts, ends)
    intervals = np.diff(np.asarray(pitches, dtype=float))
    for m, iv in enumerate(intervals.astype(int)):
        if abs(iv) <= 12 and m + 1 < len(accents):
            ivd[iv + 12] += accents[m] + accents[m + 1]
    return ivd / (ivd.sum() + 1e-12)

@simile
@interval
@pitch
def pitch_interval(pitches: list[int]) -> list[int]:
    """The intervals (in semitones) between consecutive pitches in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    list[int]
        List of intervals between consecutive pitches in semitones
    """
    return [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]

@fantastic
@interval
@pitch
def absolute_interval_range(pitches: list[int]) -> int:
    """The range between the largest and smallest absolute interval size.

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

@fantastic
@jsymbolic
@interval
@pitch
def mean_absolute_interval(pitches: list[int]) -> float:
    """The arithmetic mean of the absolute intervals in the melody.

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

mean_melodic_interval = mean_absolute_interval

@fantastic
@interval
@pitch
def standard_deviation_absolute_interval(pitches: list[int]) -> float:
    """The standard deviation of the absolute intervals in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of absolute interval sizes in semitones
    """
    return float(np.std([abs(x) for x in pitch_interval(pitches)], ddof=1))

@fantastic
@jsymbolic
@interval
@pitch
def modal_interval(pitches: list[int]) -> int:
    """The most common interval size in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Most frequent interval size in semitones

    """

    intervals_abs = [abs(x) for x in pitch_interval(pitches)]
    if not intervals_abs:
        return 0
    return int(get_mode(intervals_abs))

most_common_interval = modal_interval

@midi_toolbox
@interval
@pitch
def ivdist1(pitches: list[int], starts: list[float], ends: list[float], tempo: float = 120.0) -> dict:
    """Interval distribution weighted by duration accent.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
    tempo : float
        Tempo in BPM (beats per minute), default 120.0

    Returns
    -------
    dict
        Map from interval in semitones (-12..12) to proportion
    """
    del tempo
    if not pitches or not starts or not ends or len(pitches) < 2:
        return {}
    vec = _ivdist1_vector(pitches, starts, ends)
    return {i - 12: float(vec[i]) for i in range(25) if vec[i] > 0}

@midi_toolbox
@interval
@pitch
def ivdirdist1(pitches: list[int]) -> dict[int, float]:
    """Directional interval bias for each interval size (1-12 semitones).

    For each absolute interval size n, this computes:
    ``(p(+n) - p(-n)) / (p(+n) + p(-n))``.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    dict[int, float]
        Dictionary mapping interval sizes (1-12 semitones) to directional bias values.
        Each value ranges from -1.0 (all downward) to 1.0 (all upward), with 0.0 being equal.
        Keys: 1=minor second, 2=major second, ..., 12=octave
    """
    if not pitches or len(pitches) < 2:
        return {interval_size: 0.0 for interval_size in range(1, 13)}
    
    intervals = pitch_interval(pitches)
    if not intervals:
        return {interval_size: 0.0 for interval_size in range(1, 13)}
    
    interval_distribution = distribution_proportions(intervals)
    
    interval_direction_distribution = {}
    
    for interval_size in range(1, 13):
        upward_proportion = interval_distribution.get(float(interval_size), 0.0)
        downward_proportion = interval_distribution.get(float(-interval_size), 0.0)
        
        total_proportion = upward_proportion + downward_proportion
        
        if total_proportion > 0:
            directional_bias = (upward_proportion - downward_proportion) / total_proportion
            interval_direction_distribution[interval_size] = directional_bias
        else:
            interval_direction_distribution[interval_size] = 0.0
    
    return interval_direction_distribution

@midi_toolbox
@interval
@pitch
def ivsizedist1(pitches: list[int]) -> dict[int, float]:
    """The distribution of interval sizes (0-12 semitones). Returns the distribution 
    of interval sizes by combining upward and downward intervals of the 
    same absolute size. The first component represents a unison (0)
    and the last component represents an octave (12).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    dict[int, float]
        Dictionary mapping interval sizes (0-12 semitones) to their proportions.
        Keys: 0=unison, 1=minor second, 2=major second, ..., 12=octave
    """
    if not pitches or len(pitches) < 2:
        return {interval_size: 0.0 for interval_size in range(13)}
    
    intervals = pitch_interval(pitches)
    if not intervals:
        return {interval_size: 0.0 for interval_size in range(13)}
    
    interval_distribution = distribution_proportions(intervals)
    
    interval_size_distribution = {}
    
    for interval_size in range(13):
        if interval_size == 0:
            size_proportion = interval_distribution.get(0.0, 0.0)
        else:
            # Combine upward and downward intervals of same absolute size
            upward_proportion = interval_distribution.get(float(interval_size), 0.0)
            downward_proportion = interval_distribution.get(float(-interval_size), 0.0)
            size_proportion = upward_proportion + downward_proportion
        
        interval_size_distribution[interval_size] = size_proportion
    
    return interval_size_distribution

@simile
@interval
@pitch
def interval_direction(pitches: list[int]) -> list[int]:
    """The sequence of interval directions in the melody, 
    where 1 represents upward motion, 0 represents no motion, and -1 represents downward motion.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    list[int]
        Sequence of interval directions, where:
        1 represents upward motion
        0 represents same pitch
        -1 represents downward motion
    """
    return [
        1 if pitches[i + 1] > pitches[i] else 0 if pitches[i + 1] == pitches[i] else -1
        for i in range(len(pitches) - 1)
    ]

@novel
@interval
@pitch
def interval_direction_mean(pitches: list[int]) -> float:
    """The mean of the direction of each interval in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Mean of interval directions

    Note
    ----
    Unisons contribute 0 to both numerator and denominator because the mean is
    taken over the full direction sequence {-1, 0, 1}.
    """
    directions = interval_direction(pitches)
    
    if not directions:
        return 0.0
    
    return float(sum(directions) / len(directions))

@novel
@interval
@pitch
def interval_direction_std(pitches: list[int]) -> float:
    """The standard deviation of the direction of each interval in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Population standard deviation of interval directions

    Note
    ----
    Uses population variance (divide by N)
    """
    directions = interval_direction(pitches)
    
    if not directions:
        return 0.0
    
    mean = sum(directions) / len(directions)
    variance = sum((x - mean) ** 2 for x in directions) / len(directions)
    std_dev = math.sqrt(variance)
    
    return float(std_dev)

@jsymbolic
@interval
@pitch
def average_length_of_melodic_arcs(pitches: list[int]) -> float:
    """The average number of notes that separate peaks and troughs in melodic arcs.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Average number of notes that separate peaks and troughs in melodic arcs
    """
    if not pitches:
        return 0.0

    intervals = pitch_interval(pitches)

    total_intervening_intervals = 0
    number_arcs = 0
    direction = 0

    for interval in intervals:
        if direction == -1:
            if interval < 0:
                total_intervening_intervals += 1
            elif interval > 0:
                total_intervening_intervals += 1
                number_arcs += 1
                direction = 1

        elif direction == 1:
            if interval > 0:
                total_intervening_intervals += 1
            elif interval < 0:
                total_intervening_intervals += 1
                number_arcs += 1
                direction = -1

        else:
            if interval > 0:
                direction = 1
                total_intervening_intervals += 1
            elif interval < 0:
                direction = -1
                total_intervening_intervals += 1

    if number_arcs == 0:
        return 0.0

    return float(total_intervening_intervals) / float(number_arcs)

@jsymbolic
@interval
@pitch
def average_interval_span_by_melodic_arcs(pitches: list[int]) -> float:
    """The average interval span of melodic arcs.

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
        if direction == -1:
            if interval < 0:
                interval_so_far += abs(interval)
            elif interval > 0:
                total_intervals += interval_so_far
                number_intervals += 1
                interval_so_far = abs(interval)
                direction = 1

        elif direction == 1:
            if interval > 0:
                interval_so_far += abs(interval)
            elif interval < 0:
                total_intervals += interval_so_far
                number_intervals += 1
                interval_so_far = abs(interval)
                direction = -1

        elif direction == 0:
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

@jsymbolic
@interval
@pitch
def distance_between_most_prevalent_melodic_intervals(pitches: list[int]) -> float:
    """The absolute difference between the two most common interval sizes.

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
    
    interval_hist = create_melodic_interval_histogram(intervals, use_absolute=True)
    
    histogram = interval_hist.histogram
    
    max_value = 0.0
    max_index = 0
    for interval, count in histogram.items():
        if count > max_value:
            max_value = count
            max_index = interval
    
    second_max_value = 0.0
    second_max_index = 0
    for interval, count in histogram.items():
        if count > second_max_value and interval != max_index:
            second_max_value = count
            second_max_index = interval
    
    if second_max_value == 0.0:
        return 0.0
    
    return float(abs(max_index - second_max_index))

@jsymbolic
@interval
@pitch
def melodic_interval_histogram(pitches: list[int]) -> dict[int, int]:
    """Histogram of absolute melodic interval sizes in semitones.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    dict[int, int]
        Mapping from interval size (0–127 semitones) to count. Only sizes with count > 0 are included.

    Note
    ----
    We only return bins for intervals that have a count > 0. An implementation that is truer to the original jSymbolic 
    implementation would return 128 bins (0-127) regardless of how any different intervals are present.
    """
    intervals = pitch_interval(pitches)
    if not intervals:
        return {}

    histogram: dict[int, int] = {}
    for interval in intervals:
        size = abs(interval)
        if 0 <= size <= 127:
            histogram[size] = histogram.get(size, 0) + 1
    return histogram

@jsymbolic
@interval
@pitch
def melodic_large_intervals(pitches: list[int]) -> float:
    """The proportion of intervals >= 13 semitones.

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

def variable_melodic_intervals(pitches: list[int], interval_level: Union[int, list[int]]) -> float:
    """The proportion of intervals >= specified size.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    interval_level : Union[int, list[int]]
        Minimum interval size in semitones

    Returns
    -------
    float
        Proportion of intervals == interval_level, or -1.0 if no intervals
    """
    intervals = pitch_interval(pitches)
    if not intervals:
        return -1.0
    if isinstance(interval_level, int):
        target_intervals = sum(
            1 for interval in intervals if abs(interval) == interval_level
        )
        return float(target_intervals / len(intervals) if intervals else 0.0)
    else:
        target_intervals = sum(
            1 for interval in intervals if abs(interval) in interval_level
        )
        return float(target_intervals / len(intervals) if intervals else 0.0)

@jsymbolic
@interval
@pitch
def melodic_thirds(pitches: list[int]) -> float:
    """The proportion of intervals that are thirds (3 or 4 semitones).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Proportion of intervals that are thirds (3 or 4 semitones)
    """
    
    return variable_melodic_intervals(pitches, [3, 4])

@jsymbolic
@interval
@pitch
def melodic_perfect_fourths(pitches: list[int]) -> float:
    """The proportion of intervals that are perfect fourths (5 semitones).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Proportion of intervals that are perfect fourths (5 semitones)
    """
    return variable_melodic_intervals(pitches, 5)

@jsymbolic
@interval
@pitch
def melodic_tritones(pitches: list[int]) -> float:
    """The proportion of intervals that are tritones (6 semitones).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Proportion of intervals that are tritones (6 semitones)
    """
    return variable_melodic_intervals(pitches, 6)

@jsymbolic
@interval
@pitch
def melodic_perfect_fifths(pitches: list[int]) -> float:
    """The proportion of intervals that are perfect fifths (7 semitones).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Proportion of intervals that are perfect fifths (7 semitones)
    """
    return variable_melodic_intervals(pitches, 7)

@jsymbolic
@interval
@pitch
def melodic_sixths(pitches: list[int]) -> float:
    """The proportion of intervals that are sixths (8 or 9 semitones).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of intervals that are sixths (8 or 9 semitones)
    """
    return variable_melodic_intervals(pitches, [8, 9])

@jsymbolic
@interval
@pitch
def melodic_sevenths(pitches: list[int]) -> float:
    """The proportion of intervals that are sevenths (10 or 11 semitones).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Proportion of intervals that are sevenths (10 or 11 semitones)
    """
    return variable_melodic_intervals(pitches, [10, 11])

@jsymbolic
@interval
@pitch
def melodic_octaves(pitches: list[int]) -> float:
    """The proportion of intervals that are octaves (12 semitones).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Proportion of intervals that are octaves (12 semitones)
    """
    return variable_melodic_intervals(pitches, 12)

@jsymbolic
@interval
@pitch
def minor_major_third_ratio(pitches: list[int]) -> float:
    """The ratio of minor thirds to major thirds.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Ratio of minor thirds to major thirds, or NaN if no major thirds exist

    Note
    ----
    Instead of matching jSymbolic behavior (returning 0.0 when there are no major thirds), 
    this returns NaN when there are no major thirds (including cases where minor thirds are present).
    """
    minor_thirds = variable_melodic_intervals(pitches, 3)
    major_thirds = variable_melodic_intervals(pitches, 4)

    if major_thirds == 0:
        return float('nan')

    return minor_thirds / major_thirds

@jsymbolic
@interval
@pitch
def direction_of_melodic_motion(pitches: list[int]) -> float:
    """The proportion of upward melodic motions with regards to the total number of melodic motions.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Proportion of upward melodic motion (0.0 to 1.0), or -1.0 if no intervals

    Note
    ----
    This feature excludes unisons from its denominator and maps only to [0, 1].
    """
    intervals = pitch_interval(pitches)
    if not intervals:
        return -1.0

    ups = 0
    downs = 0

    for interval in intervals:
        if interval > 0:
            ups += 1
        elif interval < 0:
            downs += 1

    if (ups + downs) == 0:
        return 0.0

    return float(ups) / float(ups + downs)

@jsymbolic
@interval
@pitch
def number_of_common_melodic_intervals(pitches: list[int]) -> int:
    """The number of intervals that appear in at least 9% of melodic transitions.

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
    absolute_intervals = [abs(iv) for iv in intervals]
    significant_intervals = n_percent_significant_values(absolute_intervals, threshold=0.09)

    return int(len(significant_intervals))

@jsymbolic
@interval
@pitch
def prevalence_of_most_common_melodic_interval(pitches: list[int]) -> float:
    """The proportion of intervals that are the most common interval.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of intervals that are the most common interval, or 0 if no intervals
    """
    absolute_intervals = [abs(iv) for iv in pitch_interval(pitches)]
    return prevalence_of_mode(absolute_intervals)

@jsymbolic
@interval
@pitch
def relative_prevalence_of_most_common_melodic_intervals(pitches: list[int]) -> float:
    """The ratio of the frequency of the second most common interval to the frequency of the most common interval.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Ratio of second most common interval frequency to most common interval frequency.
        Returns 0.0 if fewer than 2 intervals or only one unique interval.
    """
    absolute_intervals = [abs(iv) for iv in pitch_interval(pitches)]
    return relative_prevalence_top_two(absolute_intervals)

@jsymbolic
@pitch
@interval
def amount_of_arpeggiation(pitches: list[int]) -> float:
    """The proportion of pitch intervals in the melody that constitute triadic movements.

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

@jsymbolic
@pitch
@interval
def chromatic_motion(pitches: list[int]) -> float:
    """The proportion of chromatic motion in the melody. Chromatic motion is defined as a melodic interval of 1 semitone.

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
