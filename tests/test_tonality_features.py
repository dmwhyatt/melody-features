import os
import tempfile
from typing import Optional

import mido

from melody_features.algorithms import longest_monotonic_conjunct_scalar_passage
from melody_features.features import (
    get_tonality_features,
    inscale,
    key,
    proportion_inscale,
    referent,
    tonal_clarity,
    tonal_spike,
    tonalness,
)
from melody_features.import_mid import import_midi
from melody_features.core.representations import Melody


def _build_test_melody(
    pitches: list[int],
    key_signature: Optional[str] = None,
    tempo_bpm: int = 120,
) -> Melody:
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    if key_signature is not None:
        track.append(mido.MetaMessage("key_signature", key=key_signature, time=0))

    ticks_per_note = 480
    for pitch in pitches:
        track.append(mido.Message("note_on", note=pitch, velocity=64, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=64, time=ticks_per_note))

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
        midi.save(handle.name)
        path = handle.name
    try:
        imported = import_midi(path)
        return Melody(imported)
    finally:
        os.unlink(path)


def test_key_format_is_canonical():
    melody = _build_test_melody([60, 64, 67, 71], key_signature="G#m")
    assert key(melody, key_estimation="always_read_from_file") == "G# minor"


def test_key_estimation_affects_referent_and_inscale():
    # C-major melody annotated with F# minor key signature.
    melody = _build_test_melody([60, 62, 64, 65, 67, 69, 71], key_signature="F#m")

    referent_from_file = referent(melody, key_estimation="always_read_from_file")
    referent_inferred = referent(melody, key_estimation="always_infer")
    assert referent_from_file == 6
    assert referent_inferred == 0

    inscale_from_file = inscale(melody, key_estimation="always_read_from_file")
    inscale_inferred = inscale(melody, key_estimation="always_infer")
    assert inscale_from_file != inscale_inferred
    assert proportion_inscale(melody, key_estimation="always_read_from_file") < 1.0
    assert proportion_inscale(melody, key_estimation="always_infer") == 1.0


def test_tonal_sentinels_for_empty_input():
    assert tonalness([]) == 0.0
    assert tonal_clarity([]) == 0.0
    assert tonal_spike([]) == 0.0


def test_batch_and_standalone_tonalness_match():
    melody = _build_test_melody([60, 62, 64, 65, 67, 69, 71, 72], key_signature="C")
    standalone = tonalness(melody.pitches)
    batched = get_tonality_features(melody)["tonalness"]
    assert standalone == batched


def test_scalar_passage_with_flat_key_signature():
    """Scalar helpers must accept flat key roots from MIDI key signatures (e.g. Bb)."""
    melody = _build_test_melody([70, 72, 74, 75, 77], key_signature="Bb")
    features = get_tonality_features(melody)
    assert features["key"] == "Bb major"
    assert features["longest_monotonic_conjunct_scalar_passage"] >= 1
    assert longest_monotonic_conjunct_scalar_passage(
        melody.pitches, [("Bb major", 1.0)]
    ) >= 1
