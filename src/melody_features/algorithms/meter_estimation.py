"""
Meter estimation functions based on MIDI toolbox "meter.m"

This module provides autocorrelation-based meter estimation functions that can be 
used as fallbacks when MIDI files don't contain explicit time signature information.
"""

from typing import Optional, Union

import numpy as np
from scipy.signal import correlate


def _matlab_round(values: Union[np.ndarray, float]) -> np.ndarray:
    """Round like MATLAB ``round`` (half away from zero for positive values)."""
    arr = np.asarray(values, dtype=float)
    return np.floor(arr + 0.5).astype(int)


def duration_accent(starts: list[float], ends: list[float], tau: float = 0.5, accent_index: float = 2.0) -> list[float]:
    """Calculate duration accent for each note based on Parncutt (1994).
    
    Duration accent represents the perceptual salience of notes based on their duration.
    The MIDI toolbox implementation uses defaults of 0.5 for tau (saturation duration) 
    and 2.0 for accent_index (minimum discriminable duration).
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
    tau : float, optional
        Saturation duration in seconds, by default 0.5
    accent_index : float, optional
        Minimum discriminable duration parameter, by default 2.0
        
    Returns
    -------
    list[float]
        List of duration accent values for each note
    """
    if not starts or not ends or len(starts) != len(ends):
        return []
    
    durations = [end - start for start, end in zip(starts, ends)]
    if not durations:
        return []

    accents = []
    for dur in durations:
        if dur <= 0:
            accents.append(0.0)
        else:
            accent = (1 - np.exp(-dur / tau)) ** accent_index
            accents.append(float(accent))
    
    return accents


def melodic_accent(pitches: list[int]) -> list[float]:
    """Calculate melodic accent salience according to Thomassen's model.
    Implementation based on MIDI toolbox "melaccent.m"

    "Thomassen's model assigns melodic accents according to the possible
    melodic contours arising in 3-pitch windows. Accent values vary between
    0 (no salience) and 1 (maximum salience)."
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    list[float]
        List of accent values for each note
    """
    if len(pitches) < 3:
        # Return default accents for short melodies
        if len(pitches) == 0:
            return []
        elif len(pitches) == 1:
            return [1.0]
        elif len(pitches) == 2:
            return [1.0, 0.0]
    
    accent_values = np.zeros(len(pitches))
    
    # using 3-note windows
    accent_pairs = np.zeros((len(pitches) - 2, 2))
    
    for i in range(len(pitches) - 2):
        # make 3-note window
        current_window = pitches[i:i+3]
        
        # Calculate motions between adjacent notes
        first_interval = current_window[1] - current_window[0]
        second_interval = current_window[2] - current_window[1]
        
        # Assign accent values based on melodic contour
        if first_interval == 0 and second_interval == 0:
            # No motion
            current_accents = [0.00001, 0.0]
        elif first_interval != 0 and second_interval == 0:
            # Motion then stationary
            current_accents = [1.0, 0.0]
        elif first_interval == 0 and second_interval != 0:
            # Stationary then motion
            current_accents = [0.00001, 1.0]
        elif first_interval > 0 and second_interval < 0:
            # Up then down (peak)
            current_accents = [0.83, 0.17]
        elif first_interval < 0 and second_interval > 0:
            # Down then up (valley)
            current_accents = [0.71, 0.29]
        elif first_interval > 0 and second_interval > 0:
            # Continuous upward motion
            current_accents = [0.33, 0.67]
        elif first_interval < 0 and second_interval < 0:
            # Continuous downward motion
            current_accents = [0.5, 0.5]
        else:
            current_accents = [0.0, 0.0]
            
        accent_pairs[i, :] = current_accents
    
    # Combine overlapping accent values
    accent_values[0] = 1.0  # First note gets accent of 1
    accent_values[1] = accent_pairs[0, 0]  # Second note
    
    # For middle notes, multiply overlapping accent values
    for note_idx in range(2, len(pitches) - 1):
        overlapping_accents = [accent_pairs[note_idx-2, 1], accent_pairs[note_idx-1, 0]]
        # Product of non-zero values
        non_zero_accents = [x for x in overlapping_accents if x != 0]
        if non_zero_accents:
            accent_values[note_idx] = np.prod(non_zero_accents)
        else:
            accent_values[note_idx] = 0.0

    accent_values[len(pitches) - 1] = accent_pairs[-1, 1]
    
    return accent_values.tolist()


def _beat_onsets_and_durations(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    tempo_changes: Optional[list[tuple[float, float]]] = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Onsets/durations in quarter-note beats (MIDI Toolbox ``onset`` / ``dur``)."""
    from ..algorithms.pitch_spelling import _seconds_to_beats

    if tempo_changes is None:
        tempo_changes = [(0.0, tempo)]
    onsets_beats = _seconds_to_beats(starts, tempo_changes)
    beat_origin = float(onsets_beats[0]) if len(onsets_beats) else 0.0
    onsets_beats = onsets_beats - beat_origin
    end_beats = _seconds_to_beats(ends, tempo_changes) - beat_origin
    durations_beats = end_beats - onsets_beats
    durations_sec = [end - start for start, end in zip(starts, ends)]
    return onsets_beats, durations_beats, durations_sec


def _onset_function_grid(
    onsets_beats: np.ndarray,
    accent_values: list[float],
    divisions_per_quarter: int = 4,
    max_lag_quarters: int = 8,
) -> np.ndarray:
    """``onsetfunc.m``: delta grid weighted by accent values."""
    if len(onsets_beats) == 0:
        return np.zeros(1)

    max_beat = float(np.max(onsets_beats))
    grid_length = divisions_per_quarter * max(
        2 * max_lag_quarters, int(np.ceil(max_beat)) + 1
    )
    onset_grid = np.zeros(grid_length)
    for onset_beat, weight in zip(onsets_beats, accent_values):
        grid_index = int(_matlab_round(onset_beat * divisions_per_quarter) % len(onset_grid))
        onset_grid[grid_index] += weight
    return onset_grid


def _ofacorr(onset_grid: np.ndarray, max_lag_quarters: int = 8, divisions_per_quarter: int = 4) -> np.ndarray:
    """``ofacorr.m``: subsampled onset-function autocorrelation for meter estimation."""
    full_autocorr = correlate(onset_grid, onset_grid, mode="full")
    center_index = len(full_autocorr) // 2
    end_index = min(len(full_autocorr), center_index + max_lag_quarters * divisions_per_quarter)
    autocorr = np.zeros(max_lag_quarters * divisions_per_quarter + 1)
    segment = full_autocorr[center_index : end_index + 1]
    autocorr[: len(segment)] = segment
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
    return autocorr[2::2]


def compute_onset_autocorrelation(
    starts: list[float],
    ends: list[float],
    pitches: Optional[list[int]] = None,
    accent_type: str = "duration",
    divisions_per_quarter: int = 4,
    max_lag_quarters: int = 8,
    tempo: float = 120.0,
    tempo_changes: Optional[list[tuple[float, float]]] = None,
) -> list[float]:
    """Autocorrelation of onset times weighted by accents (MIDI Toolbox ``onsetacorr.m``)."""
    expected_length = max_lag_quarters * divisions_per_quarter + 1

    if not starts or not ends or len(starts) != len(ends):
        return [0.0] * expected_length

    onsets_beats, _durations_beats, durations_sec = _beat_onsets_and_durations(
        starts, ends, tempo=tempo, tempo_changes=tempo_changes
    )
    if accent_type == "melodic" and pitches and len(pitches) == len(starts):
        accent_values = melodic_accent(pitches)
    else:
        accent_values = duration_accent(starts, ends)

    if not accent_values:
        return [0.0] * expected_length

    max_beat = float(np.max(onsets_beats)) if len(onsets_beats) else 0.0
    grid_length = divisions_per_quarter * max(
        2 * max_lag_quarters, int(np.ceil(max_beat)) + 1
    )
    onset_grid = np.zeros(grid_length)
    for onset_beat, weight in zip(onsets_beats, accent_values):
        grid_index = int(_matlab_round(onset_beat * divisions_per_quarter) % len(onset_grid))
        onset_grid[grid_index] += weight

    full_autocorr = correlate(onset_grid, onset_grid, mode="full")
    center_index = len(full_autocorr) // 2
    end_index = center_index + max_lag_quarters * divisions_per_quarter
    autocorr_result = full_autocorr[center_index : end_index + 1]

    if autocorr_result[0] != 0:
        autocorr_result = autocorr_result / autocorr_result[0]
    else:
        autocorr_result = np.zeros_like(autocorr_result)

    return autocorr_result.tolist()


def onset_autocorrelation_with_accents(
    starts: list[float],
    ends: list[float],
    pitches: list[int],
    accent_type: str = "duration",
    divisions_per_quarter: int = 4,
    max_lag_quarters: int = 4,
    tempo: float = 120.0,
) -> list[float]:
    """Autocorrelation of onsets weighted by duration or melodic accents (meter estimation)."""
    return compute_onset_autocorrelation(
        starts,
        ends,
        pitches=pitches,
        accent_type=accent_type,
        divisions_per_quarter=divisions_per_quarter,
        max_lag_quarters=max_lag_quarters,
        tempo=tempo,
    )


def estimate_meter_simple(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    tempo_changes: Optional[list[tuple[float, float]]] = None,
) -> int:
    """Simple meter estimation using duration accents only (``meter.m`` default)."""
    if not starts or not ends or len(starts) != len(ends):
        return 2

    onsets_beats, _durations_beats, _durations_sec = _beat_onsets_and_durations(
        starts, ends, tempo=tempo, tempo_changes=tempo_changes
    )
    accent_values = duration_accent(starts, ends)
    onset_grid = _onset_function_grid(onsets_beats, accent_values)
    autocorr_values = _ofacorr(onset_grid)

    if len(autocorr_values) < 6:
        return 2

    quarter_note_corr = float(autocorr_values[3])
    dotted_quarter_corr = float(autocorr_values[5])
    return 2 if quarter_note_corr >= dotted_quarter_corr else 3


def estimate_meter_optimal(
    starts: list[float],
    ends: list[float],
    pitches: list[int],
    tempo: float = 120.0,
    tempo_changes: Optional[list[tuple[float, float]]] = None,
) -> int:
    """Optimal meter estimation (``meter.m`` with ``'optimal'``)."""
    if not pitches or len(pitches) != len(starts):
        return estimate_meter_simple(starts, ends, tempo=tempo, tempo_changes=tempo_changes)

    onsets_beats, _durations_beats, _durations_sec = _beat_onsets_and_durations(
        starts, ends, tempo=tempo, tempo_changes=tempo_changes
    )
    duration_grid = _onset_function_grid(
        onsets_beats, duration_accent(starts, ends)
    )
    melodic_grid = _onset_function_grid(
        onsets_beats, melodic_accent(pitches)
    )
    duration_autocorr = _ofacorr(duration_grid)
    melodic_autocorr = _ofacorr(melodic_grid)

    if len(duration_autocorr) < 16 or len(melodic_autocorr) < 16:
        return estimate_meter_simple(starts, ends, tempo=tempo, tempo_changes=tempo_changes)

    ac1_3 = float(duration_autocorr[2])
    ac1_4 = float(duration_autocorr[3])
    ac1_6 = float(duration_autocorr[5])
    ac1_8 = float(duration_autocorr[7])
    ac1_12 = float(duration_autocorr[11])
    ac1_16 = float(duration_autocorr[15])

    ac2_3 = float(melodic_autocorr[2])
    ac2_4 = float(melodic_autocorr[3])
    ac2_6 = float(melodic_autocorr[5])
    ac2_8 = float(melodic_autocorr[7])
    ac2_12 = float(melodic_autocorr[11])
    ac2_16 = float(melodic_autocorr[15])

    discriminant = (
        -1.042
        + 0.318 * ac1_3
        + 5.240 * ac1_4
        - 0.63 * ac1_6
        + 0.745 * ac1_8
        - 8.122 * ac1_12
        + 4.160 * ac1_16
        - 0.978 * ac2_3
        + 1.018 * ac2_4
        - 1.657 * ac2_6
        + 1.419 * ac2_8
        - 2.205 * ac2_12
        + 1.568 * ac2_16
    )
    return 2 if discriminant >= 0 else 3


def estimate_meter(
    starts: list[float],
    ends: list[float],
    pitches: list[int] = None,
    use_optimal: bool = False,
    tempo: float = 120.0,
    tempo_changes: Optional[list[tuple[float, float]]] = None,
) -> int:
    """Estimate meter using autocorrelation-based method from MIDI toolbox.
    Implementation based on MIDI toolbox "meter.m"
    
    Uses autocorrelation of onset functions to distinguish between duple (2) and 
    triple/compound (3) meters. The optimal version uses a weighted combination
    of duration and melodic accents with a discriminant function trained on 
    12,000 folk melodies.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times in seconds
    ends : list[float]
        List of note end times in seconds
    pitches : list[int], optional
        List of MIDI pitch values (needed for optimal version), by default None
    use_optimal : bool, optional
        Whether to use the optimal weighted method, by default False
        
    Returns
    -------
    int
        Estimated meter: 2 for simple duple, 3 for simple triple or compound
        
    Notes
    -----
    - Returns 2 (duple) for melodies with fewer than 2 notes
    - The optimal version requires pitch data and works best with monophonic melodies
    - Falls back to simple method if optimal method fails
    """
    if not starts or len(starts) < 2:
        return 2  # Default to duple meter for short melodies
    
    if use_optimal and pitches and len(pitches) == len(starts):
        try:
            return estimate_meter_optimal(
                starts, ends, pitches, tempo=tempo, tempo_changes=tempo_changes
            )
        except Exception:
            return estimate_meter_simple(
                starts, ends, tempo=tempo, tempo_changes=tempo_changes
            )
    return estimate_meter_simple(
        starts, ends, tempo=tempo, tempo_changes=tempo_changes
    )


def _onset_mod_meter(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    pitches: list[int] = None,
    tempo_changes: Optional[list[tuple[float, float]]] = None,
) -> list[float]:
    """Onset times modulo estimated meter (``onsetmodmeter.m``)."""
    if not starts:
        return []

    if len(starts) != len(ends):
        raise ValueError("starts and ends must have the same length")

    onsets_beats, durations_beats, _durations_sec = _beat_onsets_and_durations(
        starts, ends, tempo=tempo, tempo_changes=tempo_changes
    )
    meter_length = float(
        estimate_meter(
            starts,
            ends,
            pitches=pitches,
            use_optimal=False,
            tempo=tempo,
            tempo_changes=tempo_changes,
        )
    )
    onsets_mod = np.mod(onsets_beats, meter_length)
    grid_size = int(4 * meter_length)
    grid_weights = np.zeros(grid_size)
    grid_indices = _matlab_round(onsets_mod * 4) % grid_size
    for grid_index, duration in zip(grid_indices, durations_beats):
        grid_weights[grid_index] += duration
    strongest_beat = int(np.argmax(grid_weights))
    beat_offset = strongest_beat * 0.25
    wrapped = np.mod(onsets_beats - beat_offset, meter_length)
    return wrapped.tolist()


def metric_hierarchy(
    starts: list[float],
    ends: list[float],
    time_signature: tuple[int, int] = None,
    tempo: float = 120.0,
    pitches: list[int] = None,
    tempo_changes: Optional[list[tuple[float, float]]] = None,
) -> list[int]:
    """Metric hierarchy per note (``metrichierarchy.m``)."""
    del time_signature  # MIDI Toolbox always estimates meter from autocorrelation.

    if not starts:
        return []

    onset_mod = np.asarray(
        _onset_mod_meter(
            starts,
            ends,
            tempo=tempo,
            pitches=pitches,
            tempo_changes=tempo_changes,
        ),
        dtype=float,
    )
    hierarchy = (np.abs(onset_mod) < 1e-6).astype(int)
    hierarchy = hierarchy + (np.abs(onset_mod - np.round(onset_mod)) < 1e-6).astype(int)
    ob = onset_mod.copy()
    for _ in range(3):
        ob = ob * 2.0
        hierarchy = hierarchy + (np.abs(ob - np.round(ob)) < 1e-6).astype(int)
    return hierarchy.tolist()


def meter_to_time_signature(estimated_meter: int) -> tuple[int, int]:
    """Convert estimated meter to a time signature tuple.
    
    Parameters
    ----------
    estimated_meter : int
        Estimated meter from meter estimation (2 or 3)
        
    Returns
    -------
    tuple[int, int]
        Time signature as (numerator, denominator)
        - 2 (duple) -> (4, 4) 
        - 3 (triple/compound) -> (3, 4)
        
    Raises
    ------
    ValueError
        If estimated_meter is not 2 or 3
    """
    if estimated_meter == 2:
        return (4, 4)  # Simple duple meter
    elif estimated_meter == 3:
        return (3, 4)  # Simple triple or compound meter
    else:
        raise ValueError(f"Invalid meter value: {estimated_meter}. Expected 2 or 3.")