import logging
import os
import warnings

import pretty_midi
from mido.midifiles.meta import KeySignatureError

from .meter_estimation import estimate_meter, meter_to_time_signature

# Suppress warnings from external libraries
warnings.filterwarnings("ignore", category=UserWarning, module="pretty_midi")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*pkg_resources is deprecated.*"
)


def import_midi(midi_file: str) -> dict:
    """Import a MIDI file and return a dictionary with melody data.

    Parameters
    ----------
    midi_file : str
        Path to the MIDI file

    Returns
    -------
    dict or None
        Dictionary containing:
        - ID: Filename of the MIDI file
        - MIDI Sequence: String representation of the melody
        - pitches: List of MIDI pitch values
        - starts: List of note start times
        - ends: List of note end times
        Returns None if the file cannot be imported
    """
    logger = logging.getLogger("melody_features")

    try:
        # Parse the MIDI file
        midi_data = pretty_midi.PrettyMIDI(midi_file)

        # Get the first instrument with notes
        melody_track = None
        for instrument in midi_data.instruments:
            if len(instrument.notes) > 0:
                melody_track = instrument
                break

        if melody_track is None:
            logger.warning(f"No melody track found in {midi_file}")
            return None

        # Extract note data
        pitches = [note.pitch for note in melody_track.notes]
        starts = [note.start for note in melody_track.notes]
        ends = [note.end for note in melody_track.notes]

        # Create MIDI sequence string
        midi_sequence = "Note(" + "Note(".join(
            [
                f"pitch={p}, start={s}, end={e})"
                for p, s, e in zip(pitches, starts, ends)
            ]
        )

        # Extract tempo information
        tempo = extract_tempo_from_midi(midi_data)
        tempo_changes = extract_tempo_changes_from_midi(midi_data)
        
        # Extract time signature information (with meter estimation fallback)
        time_sig_info = extract_time_signatures_from_midi(midi_data, starts, ends, pitches)

        return {
            "ID": os.path.basename(midi_file),
            "MIDI Sequence": midi_sequence,
            "pitches": pitches,
            "starts": starts,
            "ends": ends,
            "tempo": tempo,
            "tempo_changes": tempo_changes,
            "time_signature_info": time_sig_info,
        }

    except (KeySignatureError, ValueError, IOError) as e:
        logger.warning(f"Could not import {midi_file}: {str(e)}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error importing {midi_file}: {str(e)}")
        return None


def extract_time_signatures_from_midi(midi_data: pretty_midi.PrettyMIDI, starts: list[float] = None, 
                                     ends: list[float] = None, pitches: list[int] = None) -> dict:
    """Extract time signature information from MIDI data with meter estimation fallback.
    
    Parameters
    ----------
    midi_data : pretty_midi.PrettyMIDI
        The MIDI data object
    starts : list[float], optional
        Note start times for meter estimation fallback
    ends : list[float], optional  
        Note end times for meter estimation fallback
    pitches : list[int], optional
        MIDI pitch values for optimal meter estimation
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'first_time_signature': tuple of (numerator, denominator) for first time sig
        - 'all_time_signatures': list of (time, numerator, denominator) for all time sigs
        - 'metric_stability': proportion (0.0-1.0) that first time sig comprises of total
        - 'is_estimated': bool indicating if meter was estimated vs read from file
    """
    time_signatures = midi_data.time_signature_changes
    
    if not time_signatures:
        # use meter estimation as fallback
        if starts and ends:
            estimated_meter = estimate_meter(starts, ends, pitches, use_optimal=True)
            estimated_time_sig = meter_to_time_signature(estimated_meter)
        else:
            raise ValueError("Cannot estimate meter: no note timing data (starts/ends) provided and no time signature found in MIDI file")
            
        return {
            'first_time_signature': estimated_time_sig,
            'all_time_signatures': [(0.0, estimated_time_sig[0], estimated_time_sig[1])],
            'metric_stability': 0.0,  # No stability when meter is estimated
            'is_estimated': True
        }
    
    # Time signatures found in MIDI file
    first_ts = time_signatures[0]
    first_time_sig = (first_ts.numerator, first_ts.denominator)
    
    # Get all time signatures with their times
    all_time_sigs = []
    for ts in time_signatures:
        all_time_sigs.append((ts.time, ts.numerator, ts.denominator))
    
    # Calculate metric stability
    if len(time_signatures) == 1:
        # Only one time signature = perfectly stable
        metric_stability = 1.0
    else:
        # Calculate proportion of time spent in first time signature
        total_duration = midi_data.get_end_time()
        if total_duration == 0:
            metric_stability = 1.0
        else:
            # Find when first time signature ends (when next one begins)
            if len(time_signatures) > 1:
                first_ts_duration = time_signatures[1].time - time_signatures[0].time
            else:
                first_ts_duration = total_duration
            
            # Calculate what proportion of total time is in first time signature
            metric_stability = min(1.0, first_ts_duration / total_duration)
    
    return {
        'first_time_signature': first_time_sig,
        'all_time_signatures': all_time_sigs,
        'metric_stability': metric_stability,
        'is_estimated': False
    }


def extract_tempo_from_midi(midi_data: pretty_midi.PrettyMIDI) -> float:
    """Extract tempo information from a MIDI file.

    Parameters
    ----------
    midi_data : pretty_midi.PrettyMIDI
        Parsed MIDI data object

    Returns
    -------
    float
        Tempo in beats per minute (BPM). Returns 100.0 as fallback if no tempo found.
    """
    logger = logging.getLogger("melody_features")

    try:
        # Try to get tempo changes from the MIDI file
        tempo_changes = midi_data.get_tempo_changes()

        if len(tempo_changes[0]) > 0:
            # Use the first tempo change as the main tempo
            tempo = tempo_changes[1][0]
            logger.debug(f"Extracted tempo from MIDI: {tempo} BPM")
            return tempo
        else:
            # If no tempo changes found, try using the estimate_tempo method
            try:
                estimated_tempo = midi_data.estimate_tempo()
                if estimated_tempo > 0:
                    logger.debug(f"Estimated tempo from MIDI: {estimated_tempo} BPM")
                    return estimated_tempo
            except:
                pass

            # If all else fails, check for a single tempo in the first track
            for instrument in midi_data.instruments:
                if hasattr(instrument, 'control_changes'):
                    for control in instrument.control_changes:
                        if control.number == 0x51:  # Tempo change control
                            # Convert from microseconds per quarter note to BPM
                            # MIDI tempo is stored in microseconds per quarter note
                            if hasattr(control, 'value') and control.value > 0:
                                tempo_bpm = 60000000 / control.value
                                logger.debug(f"Found tempo in control changes: {tempo_bpm} BPM")
                                return tempo_bpm

    except Exception as e:
        logger.warning(f"Could not extract tempo from MIDI, using default: {str(e)}")

    # Default fallback tempo
    logger.debug("Using default tempo: 100.0 BPM")
    return 100.0


def extract_tempo_changes_from_midi(midi_data: pretty_midi.PrettyMIDI) -> list[tuple[float, float]]:
    """Extract all tempo changes from a MIDI file.

    Parameters
    ----------
    midi_data : pretty_midi.PrettyMIDI
        Parsed MIDI data object

    Returns
    -------
    list[tuple[float, float]]
        List of (time_in_seconds, tempo_in_bpm) tuples representing tempo changes
    """
    logger = logging.getLogger("melody_features")
    
    try:
        tempo_changes = midi_data.get_tempo_changes()
        
        if len(tempo_changes[0]) > 0:
            # Convert from ticks to seconds
            tempo_times_ticks = tempo_changes[0]
            tempo_values_bpm = tempo_changes[1]
            
            # Convert tick times to seconds
            tempo_times_seconds = []
            for tick_time in tempo_times_ticks:
                seconds = midi_data.tick_to_time(tick_time)
                tempo_times_seconds.append(seconds)
            
            tempo_changes_list = list(zip(tempo_times_seconds, tempo_values_bpm))
            logger.debug(f"Extracted {len(tempo_changes_list)} tempo changes from MIDI")
            return tempo_changes_list
        else:
            single_tempo = extract_tempo_from_midi(midi_data)
            return [(0.0, single_tempo)]
            
    except Exception as e:
        logger.warning(f"Could not extract tempo changes from MIDI: {str(e)}")
        # Fallback to single tempo
        single_tempo = extract_tempo_from_midi(midi_data)
        return [(0.0, single_tempo)]
