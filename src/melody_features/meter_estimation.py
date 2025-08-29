"""
Meter estimation functions based on MIDI toolbox "meter.m"

This module provides autocorrelation-based meter estimation functions that can be 
used as fallbacks when MIDI files don't contain explicit time signature information.
"""

import numpy as np
from scipy.signal import correlate


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


def onset_autocorrelation_with_accents(starts: list[float], ends: list[float], pitches: list[int], 
                                     accent_type: str = 'duration', divisions_per_quarter: int = 4, 
                                     max_lag_quarters: int = 4) -> list[float]:
    """Calculate autocorrelation function of onset times weighted by accents.
    
    This function calculates the autocorrelation of onset times weighted by either
    duration accents or melodic accents, useful for meter estimation.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times in seconds
    ends : list[float]
        List of note end times in seconds
    pitches : list[int]
        List of MIDI pitch values (needed for melodic accents)
    accent_type : str, optional
        Type of accent to use: 'duration' or 'melodic', by default 'duration'
    divisions_per_quarter : int, optional
        Divisions per quarter note, by default 4
    max_lag_quarters : int, optional
        Maximum lag in quarter notes, by default 4
        
    Returns
    -------
    list[float]
        Autocorrelation values from lag 0 to max_lag_quarters quarter notes
    """
    expected_length = max_lag_quarters * divisions_per_quarter + 1
    
    if not starts or not ends or len(starts) != len(ends):
        return [0.0] * expected_length
    
    if len(starts) == 0:
        return [0.0] * expected_length
    
    # Get appropriate accent values
    if accent_type == 'melodic' and pitches and len(pitches) == len(starts):
        accent_values = melodic_accent(pitches)
    else:
        accent_values = duration_accent(starts, ends)
    
    if not accent_values:
        return [0.0] * expected_length
    
    # Create onset time grid
    max_onset_time = max(starts) if starts else 0
    grid_length = divisions_per_quarter * max(2 * max_lag_quarters, int(np.ceil(max_onset_time)) + 1)
    onset_grid = np.zeros(grid_length)
    
    # Place accents at quantized onset positions
    for note_idx, onset_time in enumerate(starts):
        if note_idx < len(accent_values):
            # Quantize onset time to grid divisions
            grid_index = int(np.round(onset_time * divisions_per_quarter)) % len(onset_grid)
            onset_grid[grid_index] += accent_values[note_idx]
    
    # Calculate autocorrelation using scipy's cross-correlation function
    full_autocorr = correlate(onset_grid, onset_grid, mode='full')
    
    # Extract the positive lags up to max_lag_quarters
    center_index = len(full_autocorr) // 2
    autocorr_result = full_autocorr[center_index:center_index + expected_length]
    
    # Normalize by the zero-lag value
    if autocorr_result[0] != 0:
        autocorr_result = autocorr_result / autocorr_result[0]
    else:
        autocorr_result = np.zeros_like(autocorr_result)
    
    return autocorr_result.tolist()


def estimate_meter_simple(starts: list[float], ends: list[float]) -> int:
    """Simple meter estimation using duration accents only.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times in seconds
    ends : list[float]
        List of note end times in seconds
        
    Returns
    -------
    int
        Estimated meter: 2 for simple duple, 3 for simple triple or compound
    """
    # Get autocorrelation of onset function weighted by duration accents
    # Using max_lag_quarters=4 to get up to 16 divisions (4 quarter notes)
    autocorr_values = onset_autocorrelation_with_accents(
        starts, ends, [], accent_type='duration', 
        divisions_per_quarter=4, max_lag_quarters=4
    )
    
    if len(autocorr_values) < 7:  # Need at least indices 0-6
        return 2
    
    # Compare autocorrelation at lag 4 (quarter note) vs lag 6 (dotted quarter)
    # MATLAB uses 1-based indexing: ac(4) vs ac(6)
    # Python uses 0-based indexing: ac[3] vs ac[5]
    quarter_note_corr = autocorr_values[3] if len(autocorr_values) > 3 else 0.0
    dotted_quarter_corr = autocorr_values[5] if len(autocorr_values) > 5 else 0.0
    
    if quarter_note_corr >= dotted_quarter_corr:
        return 2  # Simple duple meter
    else:
        return 3  # Simple triple or compound meter


def estimate_meter_optimal(starts: list[float], ends: list[float], pitches: list[int]) -> int:
    """Optimal meter estimation using weighted duration and melodic accents.
    
    Uses discriminant function trained on 12,000 folk melodies from the MIDI toolbox.
    Only works with monophonic melodies.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times in seconds
    ends : list[float]
        List of note end times in seconds
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    int
        Estimated meter: 2 for simple duple, 3 for simple triple or compound
    """
    if not pitches or len(pitches) != len(starts):
        # Fall back to simple method if no pitch data
        return estimate_meter_simple(starts, ends)
    
    # Get autocorrelation with duration accents
    duration_autocorr = onset_autocorrelation_with_accents(
        starts, ends, pitches, accent_type='duration', 
        divisions_per_quarter=4, max_lag_quarters=4
    )
    
    # Get autocorrelation with melodic accents
    melodic_autocorr = onset_autocorrelation_with_accents(
        starts, ends, pitches, accent_type='melodic', 
        divisions_per_quarter=4, max_lag_quarters=4
    )
    
    # Ensure we have enough values for the discriminant function
    if len(duration_autocorr) < 17 or len(melodic_autocorr) < 17:
        return estimate_meter_simple(starts, ends)
    
    # Extract specific autocorrelation values (converting from MATLAB 1-based to Python 0-based)
    # MATLAB indices: ac1(3), ac1(4), ac1(6), ac1(8), ac1(12), ac1(16)
    # Python indices: ac1[2], ac1[3], ac1[5], ac1[7], ac1[11], ac1[15]
    
    # Duration-based autocorrelation values
    ac1_3 = duration_autocorr[2] if len(duration_autocorr) > 2 else 0.0
    ac1_4 = duration_autocorr[3] if len(duration_autocorr) > 3 else 0.0
    ac1_6 = duration_autocorr[5] if len(duration_autocorr) > 5 else 0.0
    ac1_8 = duration_autocorr[7] if len(duration_autocorr) > 7 else 0.0
    ac1_12 = duration_autocorr[11] if len(duration_autocorr) > 11 else 0.0
    ac1_16 = duration_autocorr[15] if len(duration_autocorr) > 15 else 0.0
    
    # Melodic accent-based autocorrelation values
    ac2_3 = melodic_autocorr[2] if len(melodic_autocorr) > 2 else 0.0
    ac2_4 = melodic_autocorr[3] if len(melodic_autocorr) > 3 else 0.0
    ac2_6 = melodic_autocorr[5] if len(melodic_autocorr) > 5 else 0.0
    ac2_8 = melodic_autocorr[7] if len(melodic_autocorr) > 7 else 0.0
    ac2_12 = melodic_autocorr[11] if len(melodic_autocorr) > 11 else 0.0
    ac2_16 = melodic_autocorr[15] if len(melodic_autocorr) > 15 else 0.0
    
    # Discriminant function from MATLAB code
    # df = -1.042+0.318*ac1(3)+5.240*ac1(4)-0.63*ac1(6)+0.745*ac1(8)-8.122*ac1(12)+4.160*ac1(16);
    # df=df-0.978*ac2(3)+1.018*ac2(4)-1.657*ac2(6)+1.419*ac2(8)-2.205*ac2(12)+1.568*ac2(16);
    discriminant = (-1.042 + 0.318*ac1_3 + 5.240*ac1_4 - 0.63*ac1_6 + 
                   0.745*ac1_8 - 8.122*ac1_12 + 4.160*ac1_16 - 
                   0.978*ac2_3 + 1.018*ac2_4 - 1.657*ac2_6 + 
                   1.419*ac2_8 - 2.205*ac2_12 + 1.568*ac2_16)
    
    if discriminant >= 0:
        return 2  # Simple duple meter
    else:
        return 3  # Simple triple or compound meter


def estimate_meter(starts: list[float], ends: list[float], pitches: list[int] = None, 
                  use_optimal: bool = False) -> int:
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
            return estimate_meter_optimal(starts, ends, pitches)
        except Exception:
            # Fall back to simple method if optimal fails
            return estimate_meter_simple(starts, ends)
    else:
        return estimate_meter_simple(starts, ends)


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