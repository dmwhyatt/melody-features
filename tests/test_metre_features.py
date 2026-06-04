import math
from typing import Optional

from melody_features.features import (
    number_of_unique_time_signatures,
    syncopicity,
    tempo_variability,
)
from melody_features.core.representations import Melody
from tests.helpers.melody import make_melody


def _build_melody(
    starts: list[float],
    ends: list[float],
    pitches: list[int],
    *,
    tempo: float = 120.0,
    tempo_changes: Optional[list[tuple[float, float]]] = None,
    all_time_signatures: Optional[list[tuple[float, int, int]]] = None,
) -> Melody:
    midi_data = make_melody(pitches, starts, ends, tempo=tempo)
    if tempo_changes is not None:
        midi_data["tempo_changes"] = tempo_changes
    if all_time_signatures is not None:
        midi_data["time_signature_info"] = {
            "first_time_signature": (
                all_time_signatures[0][1],
                all_time_signatures[0][2],
            ),
            "all_time_signatures": all_time_signatures,
        }
    return Melody(midi_data)


def test_number_of_unique_time_signatures_counts_meter_pairs_only():
    melody = _build_melody(
        starts=[0.0, 1.0, 2.0],
        ends=[0.5, 1.5, 2.5],
        pitches=[60, 62, 64],
        all_time_signatures=[
            (0.0, 4, 4),
            (8.0, 4, 4),
            (16.0, 3, 4),
        ],
    )

    assert number_of_unique_time_signatures(melody) == 2


def test_tempo_variability_uses_duration_weighted_segments():
    melody = _build_melody(
        starts=[0.0, 1.0, 2.0],
        ends=[1.0, 2.0, 3.0],
        pitches=[60, 62, 64],
        tempo=120.0,
        tempo_changes=[(0.0, 120.0), (1.0, 240.0)],
    )

    # Segments: 1s at 120 BPM and 2s at 240 BPM.
    expected_std = math.sqrt(3200.0)
    assert math.isclose(tempo_variability(melody), expected_std, rel_tol=1e-9)


def test_syncopicity_returns_sum_of_level_proportions():
    melody = _build_melody(
        starts=[0.0, 0.75, 1.0, 1.75],
        ends=[0.5, 0.95, 1.5, 2.25],
        pitches=[60, 62, 64, 65],
    )

    value = syncopicity(melody)
    assert value >= 0.0
    # Three tested levels each contribute count / n_notes.
    assert value <= 3.0
