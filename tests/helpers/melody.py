"""Shared Melody builders for tests."""

from melody_features.core.representations import build_midi_sequence_string


def make_melody(
    pitches,
    starts,
    ends,
    *,
    melody_id="test",
    tempo=120.0,
    time_signature="4/4",
):
    """Build a melody_data dict compatible with ``Melody``."""
    pitch_list = [int(p) for p in pitches]
    start_list = [float(s) for s in starts]
    end_list = [float(e) for e in ends]
    return {
        "ID": melody_id,
        "pitches": pitch_list,
        "starts": start_list,
        "ends": end_list,
        "MIDI Sequence": build_midi_sequence_string(
            pitch_list, start_list, end_list
        ),
        "tempo": tempo,
        "time_signature": time_signature,
    }
