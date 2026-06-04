"""Tests for Melody note loading (structured lists vs legacy strings)."""

from melody_features.core.representations import Melody, build_midi_sequence_string
from tests.helpers.melody import make_melody


def test_melody_prefers_structured_note_lists():
    data = make_melody([60, 62], [0.0, 1.0], [0.5, 1.5])
    melody = Melody(data)
    assert melody.pitches == [60, 62]
    assert melody.starts == [0.0, 1.0]
    assert melody.ends == [0.5, 1.5]


def test_melody_falls_back_to_midi_sequence_string():
    sequence = build_midi_sequence_string([64], [0.0], [1.0])
    melody = Melody({"MIDI Sequence": sequence})
    assert melody.pitches == [64]
    assert melody.starts == [0.0]
    assert melody.ends == [1.0]


def test_structured_lists_ignore_mismatched_legacy_string():
    data = make_melody([60], [0.0], [1.0])
    data["MIDI Sequence"] = build_midi_sequence_string([99], [2.0], [3.0])
    melody = Melody(data)
    assert melody.pitches == [60]
