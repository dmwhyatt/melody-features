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

        return {
            "ID": os.path.basename(midi_file),
            "MIDI Sequence": midi_sequence,
            "pitches": pitches,
            "starts": starts,
            "ends": ends,
        }

    except (KeySignatureError, ValueError, IOError) as e:
        logger.warning(f"Could not import {midi_file}: {str(e)}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error importing {midi_file}: {str(e)}")
        return None
