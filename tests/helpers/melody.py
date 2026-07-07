"""Shared Melody builders for tests."""

from melody_features.core.representations import build_midi_data


def make_melody(
    pitches,
    starts,
    ends,
    *,
    melody_id="test",
    tempo=120.0,
    time_signature="4/4",
):
    """Build a melody_data dict compatible with `Melody`."""
    return build_midi_data(
        pitches,
        starts,
        ends,
        melody_id=melody_id,
        tempo=tempo,
        time_signature=time_signature,
    )
