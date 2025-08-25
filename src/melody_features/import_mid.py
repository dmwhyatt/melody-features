import logging
import os
import warnings

import pretty_midi
from mido.midifiles.meta import KeySignatureError

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

        return {
            "ID": os.path.basename(midi_file),
            "MIDI Sequence": midi_sequence,
            "pitches": pitches,
            "starts": starts,
            "ends": ends,
            "tempo": tempo,
        }

    except (KeySignatureError, ValueError, IOError) as e:
        logger.warning(f"Could not import {midi_file}: {str(e)}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error importing {midi_file}: {str(e)}")
        return None


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
