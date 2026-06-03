"""Absolute pitch feature definitions."""

import numpy as np

from ..algorithms import (
    n_percent_significant_values,
    repeated_notes_proportion,
    stepwise_motion_proportion,
)
from ..feature_decorators import absolute, fantastic, jsymbolic, midi_toolbox, partitura, pitch
from ..feature_histogram import PitchHistogram
from ..algorithms.pitch_spelling import estimate_spelling_from_melody as _estimate_spelling_from_melody
from ..core.representations import Melody
from ..utils.stats import get_mode, range_func


__all__ = [
    "pitch_range",
    "ambitus",
    "pitch_standard_deviation",
    "pitch_variability",
    "first_pitch",
    "last_pitch",
    "basic_pitch_histogram",
    "melodic_pitch_variety",
    "mean_pitch",
    "most_common_pitch",
    "number_of_unique_pitches",
    "number_of_common_pitches",
    "tessitura",
    "mean_tessitura",
    "tessitura_std",
    "prevalence_of_most_common_pitch",
    "relative_prevalence_of_top_pitches",
    "interval_between_most_prevalent_pitches",
    "pitch_skewness",
    "pitch_kurtosis",
    "importance_of_bass_register",
    "importance_of_middle_register",
    "importance_of_high_register",
    "pitch_spelling",
    "repeated_notes",
    "stepwise_motion",
]


@fantastic
@jsymbolic
@midi_toolbox
@absolute
@pitch
def pitch_range(pitches: list[int]) -> int:
    """
    Subtract the lowest pitch number in the melody from the highest.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    
    Returns
    -------
    int
        Range between highest and lowest pitch in semitones
    
    
    Note
    -----
    This feature is named `ambitus` in MIDI Toolbox.
    """
    return int(range_func(pitches))

ambitus = pitch_range

@fantastic
@jsymbolic
@absolute
@pitch
def pitch_standard_deviation(pitches: list[int]) -> float:
    """Standard deviation of all pitch numbers in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of pitches
    """
    if not pitches or len(pitches) < 2:
        return 0.0
    return float(np.std(pitches, ddof=1))

pitch_variability = pitch_standard_deviation

@jsymbolic
@absolute
@pitch
def first_pitch(pitches: list[int]) -> int:
    """The first pitch number in the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    
    Returns
    -------
    int
        First pitch in the melody
    """
    if not pitches:
        return 0
    return int(pitches[0])

@jsymbolic
@absolute
@pitch
def last_pitch(pitches: list[int]) -> int:
    """The last pitch number in the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    
    Returns
    -------
    int
        Last pitch in the melody
    """
    if not pitches:
        return 0
    return int(pitches[-1])

@jsymbolic
@absolute
@pitch
def basic_pitch_histogram(pitches: list[int]) -> dict[int, int]:
    """A histogram of pitch values and their counts, with one pitch bin per non-zero count.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    dict[int, int]
        Mapping from MIDI pitch (0–127) to note count. Only pitches with count > 0 are included.

    Note
    ----
    We only return bins for pitches that have a count > 0. An implementation that is truer to the original jSymbolic 
    implementation would return 128 bins (0-127) regardless of how any different pitches are present.
    However, we believe our approach is more concise and easier to understand for many purposes.
    """
    if not pitches:
        return {}

    histogram = PitchHistogram(pitches).histogram
    return {pitch: count for pitch, count in histogram.items() if count > 0}

@jsymbolic
@absolute
@pitch
def melodic_pitch_variety(pitches: list[int], starts: list[float], tempo: float = 120.0, ppqn: int = 480) -> float:
    """The average number of onset positions before a pitch is repeated.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times
    tempo : float, default=120.0
        Tempo in beats per minute
    ppqn : int, default=480
        Pulses per quarter note (MIDI resolution)

    Returns
    -------
    float
        Average number of distinct onset positions before pitch repetition.
    """
    if not pitches or len(pitches) < 2:
        return 0.0

    from ..utils.stats import time_to_ticks
    
    note_sequence = sorted(zip(starts, pitches))
    starts_ordered, pitches_ordered = zip(*note_sequence)
    
    # Convert to ticks
    tick_pitch_map = {}
    for start, pitch in zip(starts_ordered, pitches_ordered):
        tick = time_to_ticks(start, tempo, ppqn)
        if tick not in tick_pitch_map:
            tick_pitch_map[tick] = []
        tick_pitch_map[tick].append(pitch)

    sorted_ticks = sorted(tick_pitch_map.keys())
    
    repeated_notes_count = 0
    total_notes_before_repetition = 0
    max_notes_that_can_go_by = 16

    for tick_idx, tick in enumerate(sorted_ticks):
        notes_at_tick = tick_pitch_map[tick]

        for pitch in notes_at_tick:
            found_repeated_pitch = False
            notes_gone_by_with_different_pitch = 0
            last_tick_examined = tick

            for future_tick_idx in range(tick_idx + 1, len(sorted_ticks)):
                if found_repeated_pitch or notes_gone_by_with_different_pitch > max_notes_that_can_go_by:
                    break

                future_tick = sorted_ticks[future_tick_idx]

                if future_tick != last_tick_examined:
                    notes_gone_by_with_different_pitch += 1
                    last_tick_examined = future_tick

                future_notes = tick_pitch_map[future_tick]

                for future_pitch in future_notes:
                    if future_pitch == pitch and not found_repeated_pitch and notes_gone_by_with_different_pitch <= max_notes_that_can_go_by:
                        found_repeated_pitch = True
                        repeated_notes_count += 1
                        total_notes_before_repetition += notes_gone_by_with_different_pitch
                        break

    if repeated_notes_count == 0:
        return 0.0

    return float(total_notes_before_repetition / repeated_notes_count)

@jsymbolic
@absolute
@pitch
def mean_pitch(pitches: list[int]) -> float:
    """The arithmetic mean of the pitch numbers in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Mean pitch value, or 0.0 for empty input
    """
    if not pitches:
        return 0.0
    return float(np.mean(pitches))

@jsymbolic
@absolute
@pitch
def most_common_pitch(pitches: list[int]) -> int:
    """The most frequently occurring pitch number in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Most common pitch value
    """
    return int(get_mode(pitches))

@jsymbolic
@absolute
@pitch
def number_of_unique_pitches(pitches: list[int]) -> int:
    """The number of unique pitch numbers in the melody.

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

@jsymbolic
@absolute
@pitch
def number_of_common_pitches(pitches: list[int]) -> int:
    """The number of unique pitch numbers that appear in at least 9% of total notes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of unique pitches that appear in at least 9% of notes
    """

    significant_pitches = n_percent_significant_values(pitches, threshold=0.09)
    return int(len(set(significant_pitches)))

@midi_toolbox
@absolute
@pitch
def tessitura(pitches: list[int]) -> list[float]:
    """
    Tessitura is based on standard deviation from median pitch height. The median range 
    of the melody tends to be favoured and thus more expected. Tessitura predicts 
    whether listeners expect tones close to median pitch height. Higher `tessitura` values
    correspond to melodies that have a wider range of pitches.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    list[float]
        Absolute local tessitura value for each note in sequence
    
    Citation
    ---------
    von Hippel (2000).
    """
    if len(pitches) < 2:
        return [0.0] if len(pitches) == 1 else []
    
    tessitura_values = [0.0]
    
    for i in range(2, len(pitches) + 1):
        median_prev = np.median(pitches[:i-1])
        
        if i == 2:
            tessitura_values.append(0.0)
            continue
            
        std_prev = np.std(pitches[:i-1], ddof=1)
        
        if std_prev == 0:
            tessitura_values.append(0.0)
        else:
            current_pitch = pitches[i-1]
            tessitura_val = (current_pitch - median_prev) / std_prev
            tessitura_values.append(abs(tessitura_val))
    
    tessitura_values = [float(val) for val in tessitura_values]
    return tessitura_values

@midi_toolbox
@absolute
@pitch
def mean_tessitura(pitches: list[int]) -> float:
    """
    The arithmetic mean of local `tessitura` values.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Mean tessitura value
    """
    tess_values = tessitura(pitches)
    if not tess_values:
        return 0.0
    return float(np.mean(tess_values))

@midi_toolbox
@absolute
@pitch
def tessitura_std(pitches: list[int]) -> float:
    """
    The standard deviation of the sequence of `tessitura` values.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Standard deviation of tessitura values
    """
    tess_values = tessitura(pitches)
    if len(tess_values) < 2:
        return 0.0
    return float(np.std(tess_values, ddof=1))

@jsymbolic
@absolute
@pitch
def prevalence_of_most_common_pitch(pitches: list[int]) -> float:
    """The proportion of pitches that are the most common pitch with regards to the
    total number of pitches in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of most common pitch (0.0 if there are no pitches)
    """
    if not pitches:
        return 0.0
    return float(pitches.count(most_common_pitch(pitches)) / len(pitches))

@jsymbolic
@absolute
@pitch
def relative_prevalence_of_top_pitches(pitches: list[int]) -> float:
    """The ratio of the frequency of the second most common pitch to the frequency of the most common pitch.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Ratio of second most common pitch frequency to most common pitch frequency
    """
    if len(pitches) < 2:
        return 0.0

    pitch_counts = {}
    for pitch in pitches:
        pitch_counts[pitch] = pitch_counts.get(pitch, 0) + 1

    if len(pitch_counts) < 2:
        return 0.0

    sorted_pitches = sorted(pitch_counts.items(), key=lambda x: x[1], reverse=True)
    most_common_freq = sorted_pitches[0][1] / len(pitches)
    second_most_freq = sorted_pitches[1][1] / len(pitches)

    return float(second_most_freq / most_common_freq)

@jsymbolic
@absolute
@pitch
def interval_between_most_prevalent_pitches(pitches: list[int]) -> int:
    """The number of semitones between the two most prevalent pitches.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of semitones between the most prevalent pitches
    """
    if not pitches:
        return 0

    pitch_hist = PitchHistogram(pitches)
    histogram = pitch_hist.histogram
    if not histogram or sum(1 for v in histogram.values() if v > 0) < 2:
        return 0

    max_index = max(histogram, key=lambda k: histogram[k])
    tmp = dict(histogram)
    tmp.pop(max_index, None)
    if not tmp:
        return 0
    second_max_index = max(tmp, key=lambda k: tmp[k])

    return int(abs(int(max_index) - int(second_max_index)))

@jsymbolic
@absolute
@pitch
def pitch_skewness(pitches: list[int]) -> float:
    """The skewness of the pitch histogram, using Pearson's median skewness formula.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Median skewness of pitch values, or 0 for empty input or when std dev is 0
    """
    if not pitches:
        return 0.0
    
    histogram = PitchHistogram(pitches)
    return histogram.skewness

@jsymbolic
@absolute
@pitch
def pitch_kurtosis(pitches: list[int]) -> float:
    """The sample excess kurtosis of the pitch histogram.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Kurtosis of regular pitch histogram values, or 0 for empty input
    """
    if not pitches:
        return 0.0
    histogram = PitchHistogram(pitches)
    return histogram.kurtosis

@jsymbolic
@absolute
@pitch
def importance_of_bass_register(pitches: list[int]) -> float:
    """The proportion of pitch numbers in the melody that are between 0 and 54. 
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of MIDI pitch numbers that are between 0 and 54 (0.0 if there are no pitches)
    """
    if not pitches:
        return 0.0
    return float(sum(1 for pitch in pitches if 0 <= pitch <= 54) / len(pitches))

@jsymbolic
@absolute
@pitch
def importance_of_middle_register(pitches: list[int]) -> float:
    """The proportion of pitch numbers in the melody that are between 55 and 72. 

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of MIDI pitch numbers that are between 55 and 72 (0.0 if there are no pitches)
    """
    if not pitches:
        return 0.0
    return float(sum(1 for pitch in pitches if 55 <= pitch <= 72) / len(pitches))

@jsymbolic
@absolute
@pitch
def importance_of_high_register(pitches: list[int]) -> float:
    """The proportion of pitch numbers in the melody that are between 73 and 127. 
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of MIDI pitch numbers that are between 73 and 127 (0.0 if there are no pitches)
    """
    if not pitches:
        return 0.0
    return float(sum(1 for pitch in pitches if 73 <= pitch <= 127) / len(pitches))

@partitura
@absolute
@pitch
def pitch_spelling(melody: Melody) -> list[str]:
    """Pitch spelling using the ps13s1 algorithm.
    
    Parameters
    ----------
    melody : Melody
        A melody-features Melody object.

    Returns
    -------
    list[str]
        List of pitch spellings.

    Citation
    ----------
    Meredith (2006)
    """
    return _estimate_spelling_from_melody(melody)

@jsymbolic
@pitch
@absolute
def repeated_notes(pitches: list[int]) -> float:
    """The proportion of repeated notes in the melody.

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

@jsymbolic
@pitch
@absolute
def stepwise_motion(pitches: list[int]) -> float:
    """The proportion of stepwise motion in the melody. Stepwise motion is defined as a melodic interval of 1 or 2 semitones.

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
