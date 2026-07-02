"""Timing feature definitions."""

import inspect
from functools import lru_cache
from typing import List, Optional

import numpy as np

from ..algorithms import get_duration_ratios
from ..utils.distributional import histogram_bins, transition_matrix_to_dict
from ..feature_decorators import fantastic, jsymbolic, midi_toolbox, novel, rhythm, timing
from ..feature_histogram import create_beat_histogram, create_rhythmic_value_histogram
from ..feature_utils import _get_durations
from ..algorithms.meter_estimation import compute_onset_autocorrelation, duration_accent as _duration_accent
from ..core.representations import Melody
from ..utils.stats import get_mode, range_func


__all__ = [
    "durdist1",
    "durdist2",
    "initial_tempo",
    "mean_tempo",
    "tempo_variability",
    "duration_range",
    "mean_duration",
    "average_note_duration",
    "duration_standard_deviation",
    "variability_of_note_durations",
    "modal_duration",
    "length",
    "total_number_of_notes",
    "number_of_unique_durations",
    "global_duration",
    "duration_in_seconds",
    "note_density",
    "note_density_variability",
    "note_density_per_quarter_note",
    "note_density_per_quarter_note_variability",
    "duration_histogram",
    "range_of_rhythmic_values",
    "number_of_different_rhythmic_values_present",
    "number_of_common_rhythmic_values_present",
    "prevalence_of_very_short_rhythmic_values",
    "prevalence_of_short_rhythmic_values",
    "prevalence_of_medium_rhythmic_values",
    "prevalence_of_long_rhythmic_values",
    "prevalence_of_very_long_rhythmic_values",
    "prevalence_of_dotted_notes",
    "shortest_rhythmic_value",
    "longest_rhythmic_value",
    "mean_rhythmic_value",
    "most_common_rhythmic_value",
    "prevalence_of_most_common_rhythmic_value",
    "relative_prevalence_of_most_common_rhythmic_values",
    "difference_between_most_common_rhythmic_values",
    "mean_rhythmic_value_run_length",
    "median_rhythmic_value_run_length",
    "variability_in_rhythmic_value_run_lengths",
    "mean_rhythmic_value_offset",
    "median_rhythmic_value_offset",
    "variability_of_rhythmic_value_offsets",
    "complete_rests_fraction",
    "longest_complete_rest",
    "mean_complete_rest_duration",
    "median_complete_rest_duration",
    "variability_of_complete_rest_durations",
    "strongest_rhythmic_pulse",
    "strongest_rhythmic_pulse_tempo_standardized",
    "second_strongest_rhythmic_pulse",
    "second_strongest_rhythmic_pulse_tempo_standardized",
    "harmonicity_of_two_strongest_rhythmic_pulses",
    "harmonicity_of_two_strongest_rhythmic_pulses_tempo_standardized",
    "strength_of_strongest_rhythmic_pulse",
    "strength_of_strongest_rhythmic_pulse_tempo_standardized",
    "strength_of_second_strongest_rhythmic_pulse",
    "strength_of_second_strongest_rhythmic_pulse_tempo_standardized",
    "strength_ratio_of_two_strongest_rhythmic_pulses",
    "strength_ratio_of_two_strongest_rhythmic_pulses_tempo_standardized",
    "combined_strength_of_two_strongest_rhythmic_pulses",
    "combined_strength_of_two_strongest_rhythmic_pulses_tempo_standardized",
    "rhythmic_variability",
    "rhythmic_variability_tempo_standardized",
    "rhythmic_looseness",
    "rhythmic_looseness_tempo_standardized",
    "polyrhythms",
    "polyrhythms_tempo_standardized",
    "number_of_strong_rhythmic_pulses",
    "number_of_strong_rhythmic_pulses_tempo_standardized",
    "number_of_moderate_rhythmic_pulses",
    "number_of_moderate_rhythmic_pulses_tempo_standardized",
    "number_of_relatively_strong_rhythmic_pulses",
    "number_of_relatively_strong_rhythmic_pulses_tempo_standardized",
    "minimum_note_duration",
    "maximum_note_duration",
    "equal_duration_transitions",
    "half_duration_transitions",
    "dotted_duration_transitions",
    "amount_of_staccato",
    "short_note_fraction",
    "npvi",
    "onset_autocorrelation",
    "onset_autocorr_peak",
]


def _durdist1_categories(
    starts: list[float], ends: list[float], tempo: float = 120.0
) -> np.ndarray:
    """Return 1-indexed MIDI Toolbox duration bins (1..9) for positive beat durations."""
    if not starts or not ends:
        return np.array([], dtype=int)
    beat_durs = [
        (float(end) - float(start)) * (tempo / 60.0)
        for start, end in zip(starts, ends)
        if float(end) > float(start)
    ]
    if not beat_durs:
        return np.array([], dtype=int)
    du = np.round(2 * np.log2(np.asarray(beat_durs, dtype=float))).astype(int)
    du = du[np.abs(du) <= 4] + 5
    return du


def _durdist1_vector(
    starts: list[float], ends: list[float], tempo: float = 120.0
) -> np.ndarray:
    du = _durdist1_categories(starts, ends, tempo)
    if du.size == 0:
        return np.zeros(9, dtype=float)
    hist = np.zeros(9, dtype=float)
    for bin_index in du:
        if 1 <= bin_index <= 9:
            hist[bin_index - 1] += 1.0
    return hist / (hist.sum() + 1e-12)


def _durdist2_matrix(
    starts: list[float], ends: list[float], tempo: float = 120.0
) -> np.ndarray:
    """Second-order duration transition matrix (MIDI Toolbox ``durdist2.m``)."""
    durd = np.zeros((9, 9), dtype=float)
    du = _durdist1_categories(starts, ends, tempo)
    if du.size < 2:
        return durd
    for k in range(1, du.size):
        from_bin = int(du[k - 1])
        to_bin = int(du[k])
        if 1 <= from_bin <= 9 and 1 <= to_bin <= 9:
            durd[from_bin - 1, to_bin - 1] += 1.0
    return durd / (durd.sum() + 1e-12)

@midi_toolbox
@rhythm
@timing
def durdist1(
    starts: list[float], ends: list[float], tempo: float = 120.0
) -> dict[int, float]:
    """Note duration distribution in nine log-spaced beat bins.

    Bin centers (in beats): 1/4, √2/4, 1/2, √2/2, 1, √2, 2, 2√2, 4.

    Parameters
    ----------
    starts : list[float]
        Note onset times in seconds
    ends : list[float]
        Note offset times in seconds
    tempo : float
        Tempo in BPM (default 120)

    Returns
    -------
    dict[int, float]
        Map from bin index (1–9) to proportion
    """
    if not starts or not ends:
        return {}
    vec = _durdist1_vector(starts, ends, tempo)
    return {i + 1: float(vec[i]) for i in range(9) if vec[i] > 0}

@midi_toolbox
@rhythm
@timing
def durdist2(
    starts: list[float], ends: list[float], tempo: float = 120.0
) -> dict:
    """Second-order duration transition distribution (MIDI Toolbox ``durdist2.m``).

    Returns a 9×9 matrix of transition probabilities between log-spaced duration
    bins (same bin centres as ``durdist1``). Keys are ``(from_bin, to_bin)`` with
    bin indices 1–9.

    Parameters
    ----------
    starts : list[float]
        Note onset times in seconds
    ends : list[float]
        Note offset times in seconds
    tempo : float
        Tempo in BPM (default 120)

    Returns
    -------
    dict[tuple[int, int], float]
        Map from duration-bin transition to proportion
    """
    if not starts or not ends:
        return {}
    matrix = _durdist2_matrix(starts, ends, tempo)
    return transition_matrix_to_dict(matrix, list(range(1, 10)))

@fantastic
@jsymbolic
@rhythm
@timing
def initial_tempo(melody: Melody) -> float:
    """The first tempo of the melody.

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

def _get_tempo(melody: Melody) -> float:
    return initial_tempo(melody)

@jsymbolic
@rhythm
@timing
def mean_tempo(melody: Melody) -> float:
    """The mean tempo of the melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Mean tempo of melody in bpm
    """
    if not melody.tempo_changes:
        return melody.tempo

    total_duration = max(melody.ends) if melody.ends else 0
    if total_duration == 0:
        return melody.tempo

    weighted_sum = 0.0
    last_time = 0.0
    last_tempo = melody.tempo

    for time, tempo in melody.tempo_changes:
        duration = time - last_time
        weighted_sum += last_tempo * duration
        last_time = time
        last_tempo = tempo

    final_duration = total_duration - last_time
    weighted_sum += last_tempo * final_duration

    return float(weighted_sum / total_duration)

@jsymbolic
@rhythm
@timing
def tempo_variability(melody: Melody) -> float:
    """The duration-weighted variability of tempo across the melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Weighted population standard deviation (BPM) of tempo segments.
    """
    if not melody.tempo_changes:
        return 0.0
    if not melody.ends:
        return 0.0

    total_duration = max(melody.ends)
    if total_duration <= 0.0:
        return 0.0

    segments: list[tuple[float, float]] = []
    current_time = 0.0
    current_tempo = float(melody.tempo)

    for change_time, new_tempo in melody.tempo_changes:
        change_time = float(change_time)
        if change_time <= current_time:
            current_tempo = float(new_tempo)
            continue
        segment_end = min(change_time, total_duration)
        segment_duration = segment_end - current_time
        if segment_duration > 0.0:
            segments.append((segment_duration, current_tempo))
        current_time = segment_end
        current_tempo = float(new_tempo)
        if current_time >= total_duration:
            break

    if current_time < total_duration:
        segments.append((total_duration - current_time, current_tempo))

    if not segments:
        return 0.0

    weighted_mean = sum(duration * tempo for duration, tempo in segments) / total_duration
    weighted_variance = (
        sum(duration * ((tempo - weighted_mean) ** 2) for duration, tempo in segments)
        / total_duration
    )
    return float(np.sqrt(weighted_variance))

@fantastic
@rhythm
@timing
def duration_range(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The range between the longest and shortest note duration in quarter notes.

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
    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return 0.0
    return float(range_func(durations))

@novel
@rhythm
@timing
def mean_duration(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The mean note duration in quarter notes, computed from the raw durations.

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
    float
        Mean raw note duration in quarter notes

    Note
    ----
    We use raw durations here (in the style of FANTASTIC), rather than jSymbolic's quantized rhythmic-value bins.
    """
    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return 0.0

    return float(np.mean(durations))

@jsymbolic
@rhythm
@timing
def average_note_duration(starts: list[float], ends: list[float]) -> float:
    """
    The average note duration in seconds.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Average note duration in seconds

    Note
    ----
    This feature reports duration in seconds, unlike quarter-note duration means
    such as `mean_duration` and `mean_rhythmic_value`.
    """
    durations = [end - start for start, end in zip(starts, ends)]
    if not durations:
        return 0.0
    return float(np.mean(durations))

@novel
@rhythm
@timing
def duration_standard_deviation(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The standard deviation of note durations in quarter notes.

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
    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return 0.0
    return float(np.std(durations, ddof=1))

@jsymbolic
@rhythm
@timing
def variability_of_note_durations(starts: list[float], ends: list[float]) -> float:
    """The standard deviation of note durations in seconds.

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
    durations = [end - start for start, end in zip(starts, ends)]
    if not durations:
        return 0.0
    return float(np.std(durations, ddof=1))

@fantastic
@jsymbolic
@rhythm
@timing
def modal_duration(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """
    The modal raw note duration in quarter notes.

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
    float
        Most frequent raw note duration in quarter notes

    Note
    ----
    This computes the mode of raw quarter-note durations, so differs
    from jSymbolic `most_common_rhythmic_value`, which uses the modal
    bin in a 12-bin rhythmic-value histogram.
    """
    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return 0.0

    return float(get_mode(durations))

@fantastic
@jsymbolic
@rhythm
@timing
def length(starts: list[float]) -> int:
    """The total number of notes.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    int
        Total number of notes

    """
    return len(starts)

total_number_of_notes = length

@novel
@rhythm
@timing
def number_of_unique_durations(starts: list[float], ends: list[float], tempo: float = 120.0) -> int:
    """The number of unique note durations, measured in quarter notes.

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
    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return 0
    return int(len(set(durations)))

@fantastic
@jsymbolic
@rhythm
@timing
def global_duration(melody: Melody) -> float:
    """The total duration in seconds of the melody.

    Parameters
    ----------
    melody : Melody
        Melody object containing MIDI data

    Returns
    -------
    float
        Total duration of the MIDI sequence in seconds

    """
    return melody.total_duration

duration_in_seconds = global_duration

@fantastic
@jsymbolic
@rhythm
@timing
def note_density(melody: Melody) -> float:
    """The average number of notes per second.

    Parameters
    ----------
    melody : Melody
        Melody object containing MIDI data

    Returns
    -------
    float
        Note density (notes per unit time)
    """
    if not melody.starts or not melody.ends or len(melody.starts) == 0 or len(melody.ends) == 0:
        return 0.0
    total_duration = melody.total_duration
    if total_duration == 0:
        return 0.0
    return float(len(melody.starts) / total_duration)

@jsymbolic
@rhythm
@timing
def note_density_variability(melody: Melody) -> float:
    """The standard deviation of note density across 5-second windows.

    Parameters
    ----------
    melody : Melody
        Melody object containing MIDI data

    Returns
    -------
    float
        Standard deviation of note density using 5-second windows

    Note
    ----

    Our tests indicate a certain discrepancy between our outputs and JSymbolic's outputs,
    which may be a consequence of JSymbolic's tick-based approach, or perhaps its
    idiosyncratic windowing approach.

    """
    if not melody.starts or not melody.ends or len(melody.starts) < 2:
        return 0.0

    # Create 5-second windows and calculate note density for each
    window_duration = 5.0
    window_densities = []

    # Start from 0 and create non-overlapping 5-second windows
    start_time = 0.0
    while start_time < melody.total_duration:
        end_time = min(start_time + window_duration, melody.total_duration)

        # Count notes that start within this window
        notes_in_window = sum(1.0 for start in melody.starts if start_time <= start < end_time)

        # we tried this too, but it just exacerbatated the discrepancy
        # last_onset_in_window = max(start for start in melody.starts if start_time <= start < end_time)
        # last_offset_in_window = max(end for end in melody.ends if start_time <= end < end_time)
        # last_event_in_window = max(last_onset_in_window, last_offset_in_window)

        # window_duration_actual = last_event_in_window - start_time
        window_duration_actual = end_time - start_time


        if window_duration_actual > 0:
            density = notes_in_window / window_duration_actual
            window_densities.append(density)

        start_time += window_duration

    if len(window_densities) < 2:
        return 0.0

    return np.std(window_densities, ddof=1)

@jsymbolic
@rhythm
@timing
def note_density_per_quarter_note(melody: Melody) -> float:
    """The average number of note onsets per unit of time corresponding to an
    idealized quarter note duration based on the tempo.

    Parameters
    ----------
    melody : Melody
        Melody object containing MIDI data

    Returns
    -------
    float
        Average number of notes per quarter note duration
    """
    if not melody.starts:
        return 0.0

    quarter_note_duration = 60.0 / melody.tempo
    total_duration_seconds = melody.total_duration
    total_duration_quarter_notes = total_duration_seconds / quarter_note_duration

    if total_duration_quarter_notes == 0:
        return 0.0

    return float(len(melody.starts) / total_duration_quarter_notes)

@jsymbolic
@rhythm
@timing
def note_density_per_quarter_note_variability(melody: Melody) -> float:
    """The standard deviation of note density per quarter note.

    Divides the melody into 8-quarter-note windows and calculates the standard deviation
    of note density across these windows.

    Parameters
    ----------
    melody : Melody
        Melody object containing MIDI data

    Returns
    -------
    float
        Standard deviation of note density across windows

    Note
    ----

    Our tests indicate a certain discrepancy between our outputs and JSymbolic's outputs,
    which may be a consequence of JSymbolic's tick-based approach, or perhaps its
    idiosyncratic windowing approach.
    """
    if not melody.starts or not melody.ends or len(melody.starts) < 2:
        return 0.0

    # Use 8-quarter-note windows (matching jSymbolic)
    window_size_quarter_notes = 8.0
    quarter_note_duration = 60.0 / melody.tempo
    window_size_seconds = window_size_quarter_notes * quarter_note_duration
    window_densities = []

    # Start from 0 and create non-overlapping 8-quarter-note windows
    start_time = 0.0
    while start_time < melody.total_duration:
        end_time = min(start_time + window_size_seconds, melody.total_duration)

        # Count notes that start within this window
        notes_in_window = sum(1 for start in melody.starts if start_time <= start < end_time)

        window_duration_seconds = end_time - start_time
        window_duration_quarter_notes = window_duration_seconds / quarter_note_duration

        if window_duration_quarter_notes > 0:
            # Calculate note density per quarter note for this window
            density_per_quarter_note = float(notes_in_window) / window_duration_quarter_notes
            window_densities.append(density_per_quarter_note)

        start_time += window_size_seconds

    if len(window_densities) < 2:
        return 0.0

    return np.std(window_densities, ddof=1)

@jsymbolic
@rhythm
@timing
def duration_histogram(starts: list[float], ends: list[float], tempo: float = 120.0) -> dict:
    """A histogram of note durations in quarter notes.

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
    # we use the simplified output once more
    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return {}
    num_durations = max(1, len(set(durations)))
    return histogram_bins(durations, num_durations)

@jsymbolic
@rhythm
@timing
def range_of_rhythmic_values(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The range of rhythmic values located within the 12-bin PPQN-based histogram. Durations are
    converted to quarter notes and mapped to 12 fixed rhythmic bins using midpoints. The
    returned value is the difference between the highest and lowest non-empty bins.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Range in bins (int cast to float), 0 if no durations present
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)

    hist = rhythmic_value_histogram_object.histogram
    lowest = None
    highest = None
    for i in range(12):
        if hist.get(i, 0.0) > 0.0:
            lowest = i
            break
    for i in range(11, -1, -1):
        if hist.get(i, 0.0) > 0.0:
            highest = i
            break

    if lowest is None or highest is None:
        return 0.0

    return float(highest - lowest)

@jsymbolic
@rhythm
@timing
def number_of_different_rhythmic_values_present(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The number of distinct rhythmic value bins that are present in the melody (non-zero).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Count of non-zero bins as a float (0.0 if no durations)
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    count = 0
    for i in range(12):
        if hist.get(i, 0.0) > 0.0:
            count += 1

    return float(count)

@jsymbolic
@rhythm
@timing
def number_of_common_rhythmic_values_present(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The number of rhythmic value bins with normalized proportion >= 0.15.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Count of bins with mass >= 0.15 as a float (0.0 if no durations)
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    count = 0
    for i in range(12):
        if hist.get(i, 0.0) >= 0.15:
            count += 1

    return float(count)

@jsymbolic
@rhythm
@timing
def prevalence_of_very_short_rhythmic_values(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The sum of the two shortest rhythmic bins (indexes 0 and 1).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Proportion in [0, 1] for bins 0 and 1 combined (0.0 if no durations)
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    return float(hist.get(0, 0.0) + hist.get(1, 0.0))

@jsymbolic
@rhythm
@timing
def prevalence_of_short_rhythmic_values(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The sum of the three shortest rhythmic bins (indexes 0, 1, and 2).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Proportion in [0, 1] for bins 0, 1 and 2 combined (0.0 if no durations)

    Note
    ----
    Rhythmic-bin families overlap by construction in jSymbolic
    (e.g., short/medium/long), so these prevalence values are not mutually
    exclusive and can sum to more than 1.0.
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    return float(hist.get(0, 0.0) + hist.get(1, 0.0) + hist.get(2, 0.0))

@jsymbolic
@rhythm
@timing
def prevalence_of_medium_rhythmic_values(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The sum of rhythmic bins 2 to 6 (8th notes to half notes).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Proportion in [0, 1] for bins 2..6 combined (0.0 if no durations)

    Note
    ----
    Rhythmic-bin families overlap by construction in jSymbolic
    (e.g., short/medium/long), so these prevalence values are not mutually
    exclusive and can sum to more than 1.0.
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    return float(
        hist.get(2, 0.0)
        + hist.get(3, 0.0)
        + hist.get(4, 0.0)
        + hist.get(5, 0.0)
        + hist.get(6, 0.0)
    )

@jsymbolic
@rhythm
@timing
def prevalence_of_long_rhythmic_values(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The sum of rhythmic bins 6 to 11 (half notes to dotted double whole notes or more).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Proportion in [0, 1] for bins 6 to 11 combined (0.0 if no durations)

    Note
    ----
    Rhythmic-bin families overlap by construction in jSymbolic
    (e.g., short/medium/long), so these prevalence values are not mutually
    exclusive and can sum to more than 1.0.
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    return float(
        hist.get(6, 0.0)
        + hist.get(7, 0.0)
        + hist.get(8, 0.0)
        + hist.get(9, 0.0)
        + hist.get(10, 0.0)
        + hist.get(11, 0.0)
    )

@jsymbolic
@rhythm
@timing
def prevalence_of_very_long_rhythmic_values(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The sum of rhythmic bins 9 to 11 (dotted whole notes to dotted double whole notes or more).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Proportion in [0, 1] for bins 9..11 combined (0.0 if no durations)
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    return float(
        hist.get(9, 0.0)
        + hist.get(10, 0.0)
        + hist.get(11, 0.0)
    )

@jsymbolic
@rhythm
@timing
def prevalence_of_dotted_notes(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The sum of dotted rhythmic bins: 3, 5, 7, 9, 11.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Proportion in [0, 1] for dotted bins combined (0.0 if no durations)
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    return float(
        hist.get(3, 0.0)
        + hist.get(5, 0.0)
        + hist.get(7, 0.0)
        + hist.get(9, 0.0)
        + hist.get(11, 0.0)
    )

@jsymbolic
@rhythm
@timing
def shortest_rhythmic_value(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The shortest quantized (non-zero) rhythmic-bin value (in quarter notes).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Shortest quantized rhythmic-bin ideal value in quarter notes (0.0 if empty)

    Note
    ----
    This returns the ideal value of the shortest occupied histogram bin, not
    the raw minimum note duration.
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rvh.histogram
    ideals = rvh.bin_values_quarter_notes()
    for i in range(12):
        if hist.get(i, 0.0) > 0.0:
            return float(ideals[i])
    return 0.0

@jsymbolic
@rhythm
@timing
def longest_rhythmic_value(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The longest quantized rhythmic-bin value (in quarter notes).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Longest quantized rhythmic-bin ideal value in quarter notes (0.0 if empty)

    Note
    ----
    This returns the ideal value of the longest occupied histogram bin, not the
    raw maximum note duration.
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rvh.histogram
    ideals = rvh.bin_values_quarter_notes()
    for i in range(11, -1, -1):
        if hist.get(i, 0.0) > 0.0:
            return float(ideals[i])
    return 0.0

@jsymbolic
@rhythm
@timing
def mean_rhythmic_value(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The mean quantized rhythmic value in quarter notes.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Weighted mean rhythmic-bin ideal value in quarter notes (0.0 if empty)

    Note
    ----
    Uses histogram-bin ideal values rather than raw note durations.
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rvh.histogram
    ideals = rvh.bin_values_quarter_notes()
    weights = [hist.get(i, 0.0) for i in range(12)]
    total = sum(weights)
    if total == 0.0:
        return 0.0
    mean_val = sum(ideals[i] * w for i, w in enumerate(weights)) / total
    return float(mean_val)

@jsymbolic
@rhythm
@timing
def most_common_rhythmic_value(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """
    The modal quantized rhythmic value (in quarter notes).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Modal rhythmic-bin ideal value in quarter notes (0.0 if empty or all-zero)

    Note
    ----
    Uses rhythmic-value mode from a 12-bin histogram. Differs from `modal_duration`, which computes a raw-duration mode in quarter-note units.
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rvh.histogram
    ideals = rvh.bin_values_quarter_notes()
    # Choose the smallest index in case of ties
    max_val = -1.0
    max_idx = 0
    for i in range(12):
        val = hist.get(i, 0.0)
        if val > max_val:
            max_val = val
            max_idx = i
    return float(ideals[max_idx]) if max_val > 0.0 else 0.0

@jsymbolic
@rhythm
@timing
def prevalence_of_most_common_rhythmic_value(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The proportion (0.0 - 1.0) of the modal rhythmic bin.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Proportion (0.0 - 1.0) of the modal rhythmic bin (0.0 if empty)
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rvh.histogram
    max_val = 0.0
    for i in range(12):
        max_val = max(max_val, hist.get(i, 0.0))
    return float(max_val)

@jsymbolic
@rhythm
@timing
def relative_prevalence_of_most_common_rhythmic_values(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The ratio of the second-most-common rhythmic bin to the most common bin.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Ratio of the second-most-common rhythmic bin to the most common bin (0.0 if empty)
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rvh.histogram

    # Convert to ordered list for deterministic tie-breaking by smaller index
    values = [hist.get(i, 0.0) for i in range(12)]
    if not values:
        return 0.0

    most_idx = 0
    for i in range(1, 12):
        if values[i] > values[most_idx]:
            most_idx = i
    second_idx = None
    for i in range(12):
        if i == most_idx:
            continue
        if second_idx is None or values[i] > values[second_idx]:
            second_idx = i

    most_val = values[most_idx]
    second_val = 0.0 if second_idx is None else values[second_idx]

    if most_val == 0.0:
        return 0.0
    return float(second_val / most_val)

@jsymbolic
@rhythm
@timing
def difference_between_most_common_rhythmic_values(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The absolute difference in bins between most and second most common rhythmic values.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Absolute difference in bins between most and second most common rhythmic values (0.0 if empty)
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rvh.histogram
    values = [hist.get(i, 0.0) for i in range(12)]

    most_idx = 0
    for i in range(1, 12):
        if values[i] > values[most_idx]:
            most_idx = i

    second_idx = None
    for i in range(12):
        if i == most_idx:
            continue
        if second_idx is None or values[i] > values[second_idx]:
            second_idx = i

    if values[most_idx] == 0.0 or second_idx is None:
        return 0.0

    return float(abs(most_idx - second_idx))

def _rhythmic_run_lengths(starts: list[float], ends: list[float], tempo: float = 120.0) -> List[int]:
    """Helper function to compute run lengths of identical rhythmic bins for a melody."""
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return []
    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    bin_sequence: List[int] = [rvh.map_quarter_notes_to_bin_index(d) for d in durations_qn]
    if not bin_sequence:
        return []
    run_lengths: List[int] = []
    current_run = 1
    for i in range(1, len(bin_sequence)):
        if bin_sequence[i] == bin_sequence[i - 1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    return run_lengths

@jsymbolic
@rhythm
@timing
def mean_rhythmic_value_run_length(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The mean run length of identical rhythmic values across the melody. Run length is the number of consecutive
    notes with the same rhythmic value.

    Returns 0.0 if there are fewer than 1 notes.
    """
    runs = _rhythmic_run_lengths(starts, ends, tempo)
    if not runs:
        return 0.0
    return float(np.mean(runs))

@jsymbolic
@rhythm
@timing
def median_rhythmic_value_run_length(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The median run length of identical rhythmic values across the melody. Run length is the number of consecutive
    notes with the same rhythmic value."""
    runs = _rhythmic_run_lengths(starts, ends, tempo)
    if not runs:
        return 0.0
    return float(np.median(runs))

@jsymbolic
@rhythm
@timing
def variability_in_rhythmic_value_run_lengths(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The standard deviation of rhythmic value run lengths. Run length is the number of consecutive
    notes with the same rhythmic value."""
    runs = _rhythmic_run_lengths(starts, ends, tempo)
    if not runs or len(runs) == 1:
        return 0.0
    return float(np.std(runs, ddof=1))

def _rhythmic_value_offsets(starts: list[float], ends: list[float], tempo: float = 120.0) -> List[float]:
    """Helper function to compute absolute offsets (in quarter notes) from nearest ideal value for each note."""
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return []
    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    ideals = rvh.bin_values_quarter_notes()
    offsets: List[float] = []
    for d in durations_qn:
        bin_idx = rvh.map_quarter_notes_to_bin_index(d)
        ideal = ideals[bin_idx]
        offsets.append(abs(float(d) - float(ideal)))
    return offsets

@jsymbolic
@rhythm
@timing
def mean_rhythmic_value_offset(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The mean quantized offset from the nearest ideal rhythmic value (in quarter notes).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Mean quantized offset from the nearest ideal rhythmic value (in quarter notes) (0.0 if no durations)
    """
    offsets = _rhythmic_value_offsets(starts, ends, tempo)
    if not offsets:
        return 0.0
    return float(np.mean(offsets))

@jsymbolic
@rhythm
@timing
def median_rhythmic_value_offset(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The median quantized offset from the nearest ideal rhythmic value (in quarter notes).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Median quantized offset from the nearest ideal rhythmic value (in quarter notes) (0.0 if no durations)
    """
    offsets = _rhythmic_value_offsets(starts, ends, tempo)
    if not offsets:
        return 0.0
    return float(np.median(offsets))

@jsymbolic
@rhythm
@timing
def variability_of_rhythmic_value_offsets(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The standard deviation of rhythmic value offsets (in quarter notes).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Standard deviation of rhythmic value offsets (in quarter notes) (0.0 if no durations)
    """
    offsets = _rhythmic_value_offsets(starts, ends, tempo)
    if not offsets or len(offsets) == 1:
        return 0.0
    return float(np.std(offsets, ddof=1))

def _silent_run_lengths_qn(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
    min_qn_threshold: float = 0.1,
) -> list[float]:
    """Return list of complete rest lengths in quarter notes, filtered by threshold. By complete rest,
    we mean that there is nothing sounding at all at any time during the rest.

    Discretizes time to ticks (constant tempo), builds a per-tick pitched-activity mask,
    adds a 1-quarter-note silent tail, collects silent run lengths, converts to quarter
    notes, and filters out runs shorter than min_qn_threshold.
    """
    if not starts or not ends or len(starts) != len(ends):
        return []

    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))

    start_ticks = [to_ticks(s) for s in starts]
    end_ticks = [to_ticks(e) for e in ends]
    if not end_ticks:
        return []

    duration_in_ticks = max(0, max(end_ticks))
    if duration_in_ticks <= 0:
        return []

    total_ticks = duration_in_ticks + int(ppqn)

    active = [False] * total_ticks
    for s_tick, e_tick in zip(start_ticks, end_ticks):
        if e_tick <= s_tick:
            continue
        a = max(0, min(total_ticks - 1, s_tick))
        b = max(0, min(total_ticks, e_tick))
        for t in range(a, b):
            active[t] = True

    runs_ticks: list[int] = []
    current = 0
    for t in range(total_ticks):
        if not active[t]:
            current += 1
        else:
            if current > 0:
                runs_ticks.append(current)
                current = 0
    if current > 0:
        runs_ticks.append(current)

    if not runs_ticks:
        return []

    qn_per_tick = seconds_per_tick / (60.0 / float(tempo))
    runs_qn = [(rl * qn_per_tick) for rl in runs_ticks]
    return [rl for rl in runs_qn if rl >= float(min_qn_threshold)]

@jsymbolic
@rhythm
@timing
def complete_rests_fraction(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The fraction of the total duration during which no pitched notes are sounding.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Fraction of total duration during which no pitched notes are sounding (0.0 if no durations)

    Note
    ----
    This feature includes all complete silent runs (including those shorter than
    0.1 quarter notes), whereas other complete-rest summary statistics apply a
    minimum 0.1 quarter-note threshold.
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    runs_qn = _silent_run_lengths_qn(starts, ends, tempo=tempo, ppqn=ppqn, min_qn_threshold=0.0)

    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    end_ticks = [to_ticks(e) for e in ends]
    if not end_ticks:
        return 0.0
    duration_in_ticks = max(0, max(end_ticks))
    if duration_in_ticks <= 0:
        return 0.0
    total_ticks = duration_in_ticks + int(ppqn)
    qn_per_tick = seconds_per_tick / (60.0 / float(tempo))
    total_qn = total_ticks * qn_per_tick

    rest_qn = sum(runs_qn) if runs_qn else 0.0
    if total_qn <= 0.0:
        return 0.0
    return float(rest_qn / total_qn)

@jsymbolic
@rhythm
@timing
def longest_complete_rest(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The longest uninterrupted complete rest in quarter-note units (ignoring rests shorter than 0.1 QN).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Longest uninterrupted complete rest in quarter-note units (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    runs_qn = _silent_run_lengths_qn(starts, ends, tempo=tempo, ppqn=ppqn, min_qn_threshold=0.1)
    if not runs_qn:
        return 0.0
    return float(max(runs_qn))

@jsymbolic
@rhythm
@timing
def mean_complete_rest_duration(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The mean duration of complete rests in quarter-note units (ignoring rests shorter than 0.1 QN).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Mean duration of complete rests in quarter-note units (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    runs_qn = _silent_run_lengths_qn(starts, ends, tempo=tempo, ppqn=ppqn, min_qn_threshold=0.1)
    if not runs_qn:
        return 0.0
    return float(np.mean(runs_qn))

@jsymbolic
@rhythm
@timing
def median_complete_rest_duration(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The median duration of complete rests in quarter-note units (ignoring rests shorter than 0.1 QN).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Median duration of complete rests in quarter-note units (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    runs_qn = _silent_run_lengths_qn(starts, ends, tempo=tempo, ppqn=ppqn, min_qn_threshold=0.1)
    if not runs_qn:
        return 0.0
    return float(np.median(runs_qn))

@jsymbolic
@rhythm
@timing
def variability_of_complete_rest_durations(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The standard deviation of complete rest durations in quarter notes (ignoring rests shorter than 0.1 QN).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Standard deviation of complete rest durations in quarter notes (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    runs_qn = _silent_run_lengths_qn(starts, ends, tempo=tempo, ppqn=ppqn, min_qn_threshold=0.1)
    if len(runs_qn) < 2:
        return 0.0
    return float(np.std(runs_qn, ddof=1))

def _calculate_thresholded_peak_table(values: list[float]) -> list[list[float]]:
    """Build jSymbolic-style thresholded peak table (n x 3) from a histogram array.

    Columns thresholds:
    - col 0: > 0.1
    - col 1: > 0.01
    - col 2: > 0.3 * max(hist)
    Then suppress adjacent peaks keeping only the larger in any adjacent pair per column.
    """
    n = len(values)
    if n == 0:
        return []
    table = [[0.0, 0.0, 0.0] for _ in range(n)]
    highest = values[int(np.argmax(values))] if n > 0 else 0.0
    for i in range(n):
        v = float(values[i])
        if v > 0.1:
            table[i][0] = v
        if v > 0.01:
            table[i][1] = v
        if highest > 0.0 and v > 0.3 * highest:
            table[i][2] = v
    for i in range(1, n):
        for j in range(3):
            if table[i][j] > 0.0 and table[i - 1][j] > 0.0:
                if table[i][j] > table[i - 1][j]:
                    table[i - 1][j] = 0.0
                else:
                    table[i][j] = 0.0
    return table

@lru_cache(maxsize=256)
def _get_beat_histogram_values_from_ticks(
    start_ticks: tuple[int, ...],
    end_ticks: tuple[int, ...],
    tempo: float,
    ppqn: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """LRU-cached beat histogram arrays (normal, standardized) from tick inputs.
    This is cached to avoid recomputing the beat histogram for the same start and end ticks, or worse, the autocorrelation.
    We've optimised the beat histogram computation to be more efficient, but caching still makes sense to me at this time."""
    if not end_ticks:
        return tuple(), tuple()
    duration_in_ticks = max(0, max(end_ticks))
    if duration_in_ticks <= 0:
        return tuple(), tuple()
    total_ticks = duration_in_ticks + int(ppqn)

    rhythm_score: list[int] = [0] * (total_ticks + 1)
    for tick in start_ticks:
        if 0 <= tick < len(rhythm_score):
            rhythm_score[tick] += 1

    mean_ticks_per_second = float(ppqn) * (float(tempo) / 60.0)
    bh = create_beat_histogram(
        rhythm_score=rhythm_score,
        mean_ticks_per_second=mean_ticks_per_second,
        ppqn=ppqn,
    )
    return tuple(bh.beat_histogram), tuple(bh.beat_histogram_120_bpm_standardized)

@lru_cache(maxsize=256)
def _compute_beat_histogram_tables(
    starts: tuple[float, ...], ends: tuple[float, ...], tempo: float, ppqn: int
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Compute thresholded peak tables for normal and 120-BPM-standardized beat histograms."""
    if not starts or not ends or len(starts) != len(ends):
        return (), ()

    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))

    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)

    normal_vals, std_vals = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    normal_table = _calculate_thresholded_peak_table(list(normal_vals))
    std_table = _calculate_thresholded_peak_table(list(std_vals))
    return tuple(tuple(row) for row in normal_table), tuple(tuple(row) for row in std_table)

def _count_strong_pulses(table: list[list[float]], column_index: int = 0) -> float:
    """Count peaks in BPM bins 40-200 whose thresholded value in the given column is non-zero."""
    if not table:
        return 0.0
    n = len(table)
    min_bpm = 40
    max_bpm = min(200, n - 1)
    count = 0
    for b in range(min_bpm, max_bpm + 1):
        if table[b][column_index] > 0.0:
            count += 1
    return float(count)

@jsymbolic
@rhythm
@timing
def strongest_rhythmic_pulse(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The bin index (BPM) of the maximum beat histogram magnitude.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Bin index (BPM) of the maximum beat histogram magnitude (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    values, _ = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values:
        return 0.0

    return float(int(np.argmax(values)))

@jsymbolic
@rhythm
@timing
def strongest_rhythmic_pulse_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The bin index (BPM) of the maximum in the 120-BPM standardized beat histogram.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Bin index (BPM) of the maximum beat histogram magnitude (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    _, values = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values:
        return 0.0

    return float(int(np.argmax(values)))

@jsymbolic
@rhythm
@timing
def second_strongest_rhythmic_pulse(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The bin index (BPM) of the second-highest magnitude in the beat histogram.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Bin index (BPM) of the second-highest magnitude in the beat histogram (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    values, _ = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values or len(values) < 2:
        return 0.0

    max_idx = int(np.argmax(values))
    max_val = values[max_idx]

    values_list = list(values)
    values_list[max_idx] = 0.0
    second_max_idx = int(np.argmax(values_list))

    return float(second_max_idx)

@jsymbolic
@rhythm
@timing
def second_strongest_rhythmic_pulse_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The bin index (BPM) of the second-highest magnitude in the 120-BPM standardized beat histogram.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Bin index (BPM) of the second-highest magnitude in the 120-BPM standardized beat histogram (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    _, values = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values or len(values) < 2:
        return 0.0

    max_idx = int(np.argmax(values))
    max_val = values[max_idx]

    values_list = list(values)
    values_list[max_idx] = 0.0
    second_max_idx = int(np.argmax(values_list))

    return float(second_max_idx)

@jsymbolic
@rhythm
@timing
def harmonicity_of_two_strongest_rhythmic_pulses(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The ratio of higher to lower bin index of the two strongest rhythmic pulses.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Ratio of higher to lower bin index of the two strongest rhythmic pulses
        (0.0 if no durations).

    Note
    ----
    The first peak is selected from the raw beat histogram
    and the second peak is selected from the thresholded
    peak table (column 1), excluding the first peak bin.
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0

    values, _ = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values:
        return 0.0

    normal_table, _ = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    if not normal_table:
        return 0.0

    # Find the bin with the highest magnitude from regular beat histogram
    max_value = 0.0
    max_idx = 1
    for bin in range(len(values)):
        if values[bin] > max_value:
            max_value = values[bin]
            max_idx = bin

    # Find the bin with the second highest magnitude from thresholded table column 1
    second_highest_bin_magnitude = 0.0
    second_max_idx = 1
    for bin in range(len(normal_table)):
        if (len(normal_table[bin]) > 1 and
            normal_table[bin][1] > second_highest_bin_magnitude and
            bin != max_idx):
            second_highest_bin_magnitude = normal_table[bin][1]
            second_max_idx = bin

    # Calculate the feature value
    if second_max_idx == 0 or max_idx == 0:
        value = 0.0
    elif max_idx > second_max_idx:
        value = float(max_idx) / float(second_max_idx)
    else:
        value = float(second_max_idx) / float(max_idx)

    return value

@jsymbolic
@rhythm
@timing
def harmonicity_of_two_strongest_rhythmic_pulses_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The ratio of higher to lower bin index of the two strongest rhythmic pulses (120-BPM standardized histogram).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Ratio of higher to lower bin index of the two strongest rhythmic pulses
        (120-BPM standardized histogram) (0.0 if no durations).

    Note
    ----
    The first peak is selected from the standardized histogram values
    from the standardized histogram values and the second peak is selected from the
    standardized thresholded peak table (column 1), excluding the first peak bin.
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0

    _, values = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values:
        return 0.0

    _, std_table = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    if not std_table:
        return 0.0

    # Find the bin with the highest magnitude from tempo standardized beat histogram
    max_value = 0.0
    max_idx = 1
    for bin in range(len(values)):
        if values[bin] > max_value:
            max_value = values[bin]
            max_idx = bin

    # Find the bin with the second highest magnitude from thresholded table column 1
    second_highest_bin_magnitude = 0.0
    second_max_idx = 1
    for bin in range(len(std_table)):
        if (len(std_table[bin]) > 1 and
            std_table[bin][1] > second_highest_bin_magnitude and
            bin != max_idx):
            second_highest_bin_magnitude = std_table[bin][1]
            second_max_idx = bin

    # Calculate the feature value
    if second_max_idx == 0 or max_idx == 0:
        value = 0.0
    elif max_idx > second_max_idx:
        value = float(max_idx) / float(second_max_idx)
    else:
        value = float(second_max_idx) / float(max_idx)

    return value

@jsymbolic
@rhythm
@timing
def strength_of_strongest_rhythmic_pulse(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The magnitude of the beat histogram bin with the highest magnitude.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Magnitude of the beat histogram bin with the highest magnitude (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    values, _ = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values:
        return 0.0
    return float(max(values))

@jsymbolic
@rhythm
@timing
def strength_of_strongest_rhythmic_pulse_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The magnitude of the tempo-standardized beat histogram bin with the highest magnitude.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Magnitude of the tempo-standardized beat histogram bin with the highest magnitude (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    _, values = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values:
        return 0.0
    return float(max(values))

@jsymbolic
@rhythm
@timing
def strength_of_second_strongest_rhythmic_pulse(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The magnitude of the beat histogram bin with the second-highest magnitude.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Magnitude of the beat histogram bin with the second-highest magnitude (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    values, _ = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values or len(values) < 2:
        return 0.0

    # Find the two highest values
    max_val = max(values)
    values_list = list(values)
    values_list[values_list.index(max_val)] = 0.0
    second_max_val = max(values_list)

    return float(second_max_val)

@jsymbolic
@rhythm
@timing
def strength_of_second_strongest_rhythmic_pulse_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The magnitude of the tempo-standardized beat histogram bin with the second-highest magnitude.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Magnitude of the tempo-standardized beat histogram bin with the second-highest magnitude (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    _, values = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values or len(values) < 2:
        return 0.0

    # Find the two highest values
    max_val = max(values)
    values_list = list(values)
    values_list[values_list.index(max_val)] = 0.0
    second_max_val = max(values_list)

    return float(second_max_val)

@jsymbolic
@rhythm
@timing
def strength_ratio_of_two_strongest_rhythmic_pulses(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """Ratio of the magnitude of the strongest to second-strongest rhythmic pulse.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Ratio of the magnitude of the strongest to second-strongest rhythmic pulse (0.0 if no durations)
    """
    strongest_strength = strength_of_strongest_rhythmic_pulse(starts, ends, tempo, ppqn)
    second_strongest_strength = strength_of_second_strongest_rhythmic_pulse(starts, ends, tempo, ppqn)

    if second_strongest_strength == 0:
        return 0.0
    return float(strongest_strength) / float(second_strongest_strength)

@jsymbolic
@rhythm
@timing
def strength_ratio_of_two_strongest_rhythmic_pulses_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """Ratio of the magnitude of the strongest to second-strongest rhythmic pulse (120-BPM standardized histogram).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Ratio of the magnitude of the strongest to second-strongest rhythmic pulse (120-BPM standardized histogram) (0.0 if no durations)
    """
    strongest_strength = strength_of_strongest_rhythmic_pulse_tempo_standardized(starts, ends, tempo, ppqn)
    second_strongest_strength = strength_of_second_strongest_rhythmic_pulse_tempo_standardized(starts, ends, tempo, ppqn)

    if second_strongest_strength == 0:
        return 0.0
    return float(strongest_strength) / float(second_strongest_strength)

@jsymbolic
@rhythm
@timing
def combined_strength_of_two_strongest_rhythmic_pulses(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """Sum of the magnitudes of the two strongest rhythmic pulses.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Sum of the magnitudes of the two strongest rhythmic pulses (0.0 if no durations)
    """
    strongest_strength = strength_of_strongest_rhythmic_pulse(starts, ends, tempo, ppqn)
    second_strongest_strength = strength_of_second_strongest_rhythmic_pulse(starts, ends, tempo, ppqn)

    return float(strongest_strength) + float(second_strongest_strength)

@jsymbolic
@rhythm
@timing
def combined_strength_of_two_strongest_rhythmic_pulses_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """Sum of the magnitudes of the two strongest rhythmic pulses using tempo-standardized histogram.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Sum of the magnitudes of the two strongest rhythmic pulses using tempo-standardized histogram (0.0 if no durations)
    """
    strongest_strength = strength_of_strongest_rhythmic_pulse_tempo_standardized(starts, ends, tempo, ppqn)
    second_strongest_strength = strength_of_second_strongest_rhythmic_pulse_tempo_standardized(starts, ends, tempo, ppqn)

    return float(strongest_strength) + float(second_strongest_strength)

@jsymbolic
@rhythm
@timing
def rhythmic_variability(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The standard deviation of the beat histogram bin magnitudes, excluding the first 40 bins.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Standard deviation of the beat histogram bin magnitudes, excluding the first 40 bins (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    values, _ = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values or len(values) <= 40:
        return 0.0

    # Exclude the first 40 bins (BPM 0-39)
    reduced_values = values[40:]
    if len(reduced_values) < 2:
        return 0.0

    return float(np.std(reduced_values, ddof=1))

@jsymbolic
@rhythm
@timing
def rhythmic_variability_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The standard deviation of the tempo-standardized beat histogram bin magnitudes, excluding the first 40 bins.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Standard deviation of the tempo-standardized beat histogram bin magnitudes, excluding the first 40 bins (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    _, values = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values or len(values) <= 40:
        return 0.0

    # Exclude the first 40 bins (BPM 0-39)
    reduced_values = values[40:]
    if len(reduced_values) < 2:
        return 0.0

    return float(np.std(reduced_values, ddof=1))

@jsymbolic
@rhythm
@timing
def rhythmic_looseness(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The average width of beat histogram peaks. Width is defined as the distance between points at 30% of the peak height.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Average width of beat histogram peaks (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    table, _ = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    if not table:
        return 0.0

    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    values, _ = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values:
        return 0.0

    # Find peaks with magnitude >= 30% of highest peak (column 2 in thresholded table)
    peak_bins = []
    for bin_idx in range(len(table)):
        if table[bin_idx][2] > 0.001:
            peak_bins.append(bin_idx)

    if not peak_bins:
        return 0.0

    widths = []
    for peak_bin in peak_bins:
        if peak_bin >= len(values):
            continue

        # 30% of this peak's height
        limit_value = 0.3 * values[peak_bin]

        # Find left limit
        left_index = 0
        i = peak_bin
        while i >= 0:
            if values[i] < limit_value:
                break
            left_index = i
            i -= 1

        # Find right limit
        right_index = len(values) - 1
        i = peak_bin
        while i < len(values):
            if values[i] < limit_value:
                break
            right_index = i
            i += 1

        # Calculate width (in BPM bins)
        width = float(right_index - left_index)
        widths.append(width)

    if not widths:
        return 0.0

    return float(np.mean(widths))

@jsymbolic
@rhythm
@timing
def rhythmic_looseness_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The average width of beat histogram peaks using tempo-standardized histogram. Width is defined as the distance between points at 30% of the peak height.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Average width of beat histogram peaks using tempo-standardized histogram (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    _, table = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    if not table:
        return 0.0

    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    _, values = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values:
        return 0.0

    # Find peaks with magnitude >= 30% of highest peak (column 2 in thresholded table)
    peak_bins = []
    for bin_idx in range(len(table)):
        if table[bin_idx][2] > 0.001:
            peak_bins.append(bin_idx)

    if not peak_bins:
        return 0.0

    widths = []
    for peak_bin in peak_bins:
        if peak_bin >= len(values):
            continue

        # 30% of this peak's height
        limit_value = 0.3 * values[peak_bin]

        # Find left limit
        left_index = 0
        i = peak_bin
        while i >= 0:
            if values[i] < limit_value:
                break
            left_index = i
            i -= 1

        # Find right limit
        right_index = len(values) - 1
        i = peak_bin
        while i < len(values):
            if values[i] < limit_value:
                break
            right_index = i
            i += 1

        # Calculate width (in BPM bins)
        width = float(right_index - left_index)
        widths.append(width)

    if not widths:
        return 0.0

    return float(np.mean(widths))

def _is_factor_or_multiple(bin_idx: int, highest_bin: int, multipliers: list[int]) -> bool:
    """Check if bin_idx is a factor or multiple of highest_bin using given multipliers with +/-3 tolerance."""
    for mult in multipliers:
        # Check if bin_idx is a multiple of highest_bin * mult (within tolerance)
        expected = highest_bin * mult
        if abs(bin_idx - expected) <= 3:
            return True
        # Check if bin_idx is a factor of highest_bin (within tolerance)
        if highest_bin % mult == 0:
            expected = highest_bin // mult
            if abs(bin_idx - expected) <= 3:
                return True
        # Also check if highest_bin is a multiple of bin_idx * mult (within tolerance)
        expected = bin_idx * mult
        if abs(highest_bin - expected) <= 3:
            return True
        # And if highest_bin is a factor of bin_idx (within tolerance)
        if bin_idx % mult == 0:
            expected = bin_idx // mult
            if abs(highest_bin - expected) <= 3:
                return True
    return False

@jsymbolic
@rhythm
@timing
def polyrhythms(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The fraction of strong beat-histogram peaks related to the strongest peak.

    Among peaks at least 30% as tall as the maximum, returns the proportion whose bin is
    an integer multiple/factor of the strongest (multipliers 1, 2, 3, 4, 6, 8; ±3 bins).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        ``hits / n_peaks``, or ``0.0`` if there are no qualifying peaks.
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    # Get thresholded peak table
    table, _ = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    if not table:
        return 0.0

    # Find peaks with magnitude >= 30% of highest peak (column 2 in thresholded table)
    peak_bins = []
    for bin_idx in range(len(table)):
        if table[bin_idx][2] > 0.001:
            peak_bins.append(bin_idx)

    if not peak_bins:
        return 0.0

    # Find the highest peak
    highest_index = 0
    max_magnitude = 0.0
    for peak_bin in peak_bins:
        if table[peak_bin][2] > max_magnitude:
            max_magnitude = table[peak_bin][2]
            highest_index = peak_bin

    # Count peaks that are multiples/factors of the highest peak
    multipliers = [1, 2, 3, 4, 6, 8]
    hits = 0

    for peak_bin in peak_bins:
        if _is_factor_or_multiple(peak_bin, highest_index, multipliers):
            hits += 1

    return float(hits) / float(len(peak_bins))

@jsymbolic
@rhythm
@timing
def polyrhythms_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The fraction of strong beat-histogram peaks related to the strongest peak using the tempo-standardized beat histogram.

    Among peaks at least 30% as tall as the maximum, returns the proportion whose bin is
    an integer multiple/factor of the strongest (multipliers 1, 2, 3, 4, 6, 8; ±3 bins).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        ``hits / n_peaks``, or ``0.0`` if there are no qualifying peaks.
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    _, table = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    if not table:
        return 0.0

    # Find peaks with magnitude >= 30% of highest peak (column 2 in thresholded table)
    peak_bins = []
    for bin_idx in range(len(table)):
        if table[bin_idx][2] > 0.001:
            peak_bins.append(bin_idx)

    if not peak_bins:
        return 0.0

    # Find the highest peak
    highest_index = 0
    max_magnitude = 0.0
    for peak_bin in peak_bins:
        if table[peak_bin][2] > max_magnitude:
            max_magnitude = table[peak_bin][2]
            highest_index = peak_bin

    # Count peaks that are multiples/factors of the highest peak
    multipliers = [1, 2, 3, 4, 6, 8]
    hits = 0

    for peak_bin in peak_bins:
        if _is_factor_or_multiple(peak_bin, highest_index, multipliers):
            hits += 1

    return float(hits) / float(len(peak_bins))

@jsymbolic
@rhythm
@timing
def number_of_strong_rhythmic_pulses(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The count of BPM bins with strong rhythmic pulses (> 0.1 in the underlying beat histogram).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Count of BPM bins 40-200 with a thresholded strong-pulse value (> 0.1 in the
        underlying beat histogram), or 0.0 if no durations.
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    # Use cached beat histogram tables
    table, _ = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    return _count_strong_pulses(list(table), column_index=0)

@jsymbolic
@rhythm
@timing
def number_of_strong_rhythmic_pulses_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The count of BPM bins with strong pulses (> 0.1 in the underlying standardized beat histogram).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Count of BPM bins 40-200 with a thresholded strong-pulse value (> 0.1 in the
        underlying standardized histogram), or 0.0 if no durations.
    """
    _, std_table = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    return _count_strong_pulses(list(std_table), column_index=0)

@jsymbolic
@rhythm
@timing
def number_of_moderate_rhythmic_pulses(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The number of beat histogram peaks with normalized magnitudes over 0.01.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Number of beat histogram peaks with normalized magnitudes over 0.01 (0.0 if no durations)
    """
    table, _ = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    return _count_strong_pulses(list(table), column_index=1)

@jsymbolic
@rhythm
@timing
def number_of_moderate_rhythmic_pulses_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The number of tempo-standardized beat histogram peaks with normalized magnitudes over 0.01.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Number of tempo-standardized beat histogram peaks with normalized magnitudes over 0.01 (0.0 if no durations)
    """
    _, std_table = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    return _count_strong_pulses(list(std_table), column_index=1)

@jsymbolic
@rhythm
@timing
def number_of_relatively_strong_rhythmic_pulses(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The number of peaks at least 30% of the max magnitude.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Number of peaks at least 30% of the max magnitude (0.0 if no durations)
    """
    table, _ = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    return _count_strong_pulses(list(table), column_index=2)

@jsymbolic
@rhythm
@timing
def number_of_relatively_strong_rhythmic_pulses_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The number of tempo-standardized peaks at least 30% of the max magnitude.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Number of tempo-standardized peaks at least 30% of the max magnitude (0.0 if no durations)
    """
    _, std_table = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    return _count_strong_pulses(list(std_table), column_index=2)

@jsymbolic
@rhythm
@timing
def minimum_note_duration(starts: list[float], ends: list[float]) -> float:
    """The minimum note duration in seconds.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Minimum note duration in seconds (0.0 if there are no notes)
    """
    if not starts or not ends:
        return 0.0
    durations = [end - start for start, end in zip(starts, ends)]
    return float(min(durations)) if durations else 0.0

@jsymbolic
@rhythm
@timing
def maximum_note_duration(starts: list[float], ends: list[float]) -> float:
    """The maximum note duration in seconds.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Maximum note duration in seconds
    """
    return max([end - start for start, end in zip(starts, ends)])

@fantastic
@rhythm
@timing
def equal_duration_transitions(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The proportion of duration transitions that are equal in length.


    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
    tempo : float, optional
        Included for API consistency; not used in this calculation.

    Returns
    -------
    float
        Proportion of equal duration transitions (0.0 to 1.0)

    Citation
    --------
    Steinbeck (1982)
    """
    ratios = get_duration_ratios(starts, ends)
    if not ratios:
        return 0.0

    # Count ratios that equal 1.0 (equal durations)
    equal_count = sum(1 for ratio in ratios if ratio == 1.0)

    return equal_count / len(ratios)

@fantastic
@rhythm
@timing
def half_duration_transitions(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The proportion of duration transitions that are halved or doubled.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
    tempo : float, optional
        Included for API consistency; not used in this calculation.

    Returns
    -------
    float
        Proportion of half/double duration transitions (0.0 to 1.0)

    Citation
    --------
    Steinbeck (1982)
    """
    ratios = get_duration_ratios(starts, ends)
    if not ratios:
        return 0.0

    # Count ratios that equal 0.5 or round to 2
    half_count = sum(1 for ratio in ratios if ratio == 0.5)
    double_count = sum(1 for ratio in ratios if round(ratio) == 2)

    return (half_count + double_count) / len(ratios)

@fantastic
@rhythm
@timing
def dotted_duration_transitions(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The proportion of duration transitions that are dotted.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
    tempo : float, optional
        Included for API consistency; not used in this calculation.

    Returns
    -------
    float
        Proportion of dotted duration transitions (0.0 to 1.0)

    Citation
    --------
    Steinbeck (1982)
    """
    ratios = get_duration_ratios(starts, ends)
    if not ratios:
        return 0.0

    # Count ratios that equal 1/3 or round to 3
    one_third_count = sum(1 for ratio in ratios if abs(ratio - (1/3)) < 1e-10)
    triple_count = sum(1 for ratio in ratios if round(ratio) == 3)

    return (one_third_count + triple_count) / len(ratios)

@jsymbolic
@rhythm
@timing
def amount_of_staccato(starts: list[float], ends: list[float]) -> float:
    """The proportion of notes with a duration shorter than 0.1 seconds.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Fraction of notes shorter than 0.1 seconds

    Note
    ----
    Though this feature is named `Amount Of Staccato`, it is a
    fixed-duration cutoff statistic rather than symbolic articulation parsing.
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    durations_seconds = [float(end - start) for start, end in zip(starts, ends)]
    if not durations_seconds:
        return 0.0
    short_count = sum(1 for d in durations_seconds if d < 0.1)
    return float(short_count / len(durations_seconds))

short_note_fraction = amount_of_staccato

@midi_toolbox
@rhythm
@timing
def npvi(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The normalized Pairwise Variability Index (nPVI) of note durations in quarter notes.
    The nPVI measures the durational variability of events, originally developed for
    language research to distinguish stress-timed vs. syllable-timed languages.
    Applied to music by Patel & Daniele (2003) to study the prosodic
    influences on musical rhythm.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        nPVI index value (higher values indicate greater durational variability)

    Citation
    --------
    Patel & Daniele (2003)
    """
    durations = _get_durations(starts, ends, tempo)
    if len(durations) < 2:
        return 0.0

    normalized_diffs = []
    for i in range(1, len(durations)):
        prev_dur = durations[i-1]
        curr_dur = durations[i]

        if prev_dur + curr_dur == 0:
            normalized_diffs.append(0.0)
        else:
            # Normalized difference: (d1 - d2) / ((d1 + d2) / 2)
            mean_duration = (prev_dur + curr_dur) / 2
            normalized_diff = (prev_dur - curr_dur) / mean_duration
            normalized_diffs.append(abs(normalized_diff))

    if not normalized_diffs:
        return 0.0

    npvi_value = (100 / len(normalized_diffs)) * sum(normalized_diffs)
    return float(npvi_value)

@midi_toolbox
@rhythm
@timing
def onset_autocorrelation(
    starts: list[float],
    ends: list[float],
    divisions_per_quarter: int = 4,
    max_lag_quarters: int = 8,
    tempo: float = 120.0,
    tempo_changes: Optional[list[tuple[float, float]]] = None,
) -> list[float]:
    """The autocorrelation function of onset times weighted by duration accents.
    This is calculated by weighting the onset times by the duration accents,
    as defined by Parncutt (1994).

    Onsets are converted to quarter-note beats using
    tempo before grid quantization.

    Parameters
    ----------
    starts : list[float]
        Note onset times in seconds
    ends : list[float]
        Note offset times in seconds
    divisions_per_quarter : int, optional
        Grid divisions per quarter note (default 4)
    max_lag_quarters : int, optional
        Maximum lag in quarter notes (default 8)
    tempo : float, optional
        Tempo in BPM (default 120)

    Returns
    -------
    list[float]
        Normalized autocorrelation from lag 0 through ``max_lag_quarters`` quarters

    Citation
    --------
    Parncutt (1994)
    """
    return compute_onset_autocorrelation(
        starts,
        ends,
        divisions_per_quarter=divisions_per_quarter,
        max_lag_quarters=max_lag_quarters,
        tempo=tempo,
        tempo_changes=tempo_changes,
    )

@midi_toolbox
@rhythm
@timing
def onset_autocorr_peak(
    starts: list[float],
    ends: list[float],
    divisions_per_quarter: int = 4,
    max_lag_quarters: int = 8,
    tempo: float = 120.0,
    tempo_changes: Optional[list[tuple[float, float]]] = None,
) -> float:
    """Maximum onset autocorrelation excluding lag 0."""
    autocorr_values = onset_autocorrelation(
        starts,
        ends,
        divisions_per_quarter,
        max_lag_quarters,
        tempo,
        tempo_changes,
    )
    if len(autocorr_values) <= 1:
        return 0.0
    return float(max(autocorr_values[1:]))

def _is_beat_histogram_function(func) -> bool:
    """Check if a function uses beat histogram computations."""
    import inspect
    try:
        source = inspect.getsource(func)
        return '_get_beat_histogram_values_from_ticks' in source or 'create_beat_histogram' in source
    except:
        return False

def _precompute_beat_histogram_data(melody: Melody) -> tuple:
    """Pre-compute beat histogram data for reuse across multiple functions.

    Returns
    -------
    tuple
        (normal_values, standardized_values, start_ticks, end_ticks, tempo, ppqn)
    """
    seconds_per_tick = (60.0 / float(melody.tempo)) / float(480)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in melody.starts)
    end_ticks = tuple(to_ticks(e) for e in melody.ends)

    if not end_ticks:
        return tuple(), tuple(), start_ticks, end_ticks, melody.tempo, 480

    normal_values, standardized_values = _get_beat_histogram_values_from_ticks(
        start_ticks, end_ticks, float(melody.tempo), 480
    )

    return normal_values, standardized_values, start_ticks, end_ticks, melody.tempo, 480
