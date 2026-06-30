"""Validate MUST feature implementations against reference values."""

import csv
from pathlib import Path

import pytest

from melody_features.core.representations import Melody
from melody_features.features import get_must_features
from melody_features.io.midi import import_midi

REFERENCE_CSV = Path(__file__).parent / "must_reference_values.csv"

MUST_FEATURE_MAPPING = {
    "bisectUnbalance": "bisect_unbalance",
    "centerMassOffset": "center_mass_offset",
    "eventHeterogeneity": "event_heterogeneity",
    "avAbsInterval": "av_abs_interval",
    "melAbruptness": "mel_abruptness",
    "durAbruptness": "dur_abruptness",
    "rhythmAbruptness": "rhythm_abruptness",
    "asymTotal": "asym_total",
    "asymIndex": "asym_index",
    "eventDensity": "event_density",
    "avLocalp1entropy": "av_local_p1_entropy",
    "p1entropy": "p1_entropy",
    "p2entropy": "p2_entropy",
    "p3entropy": "p3_entropy",
    "i1entropy": "i1_entropy",
    "i2entropy": "i2_entropy",
    "d1entropy": "d1_entropy",
    "d2entropy": "d2_entropy",
    "d3entropy": "d3_entropy",
    "wpEntropy": "wp_entropy",
}


def load_reference_values() -> dict[str, dict[str, float]]:
    """Load reference MUST values from CSV."""
    reference_data: dict[str, dict[str, float]] = {}
    with open(REFERENCE_CSV, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            midi_path = row["midi_path"]
            reference_data[midi_path] = {
                feature: float(row[feature])
                for feature in MUST_FEATURE_MAPPING
            }
    return reference_data


def create_melody_from_file(file_path: str) -> Melody:
    """Create a melody from a MIDI file."""
    midi_data = import_midi(file_path)
    return Melody(midi_data)


@pytest.fixture(scope="module")
def reference_data():
    return load_reference_values()


def test_reference_csv_columns_match_mapping():
    with open(REFERENCE_CSV, encoding="utf-8") as handle:
        fieldnames = set(csv.DictReader(handle).fieldnames or [])
    assert set(MUST_FEATURE_MAPPING.keys()).issubset(fieldnames)


def test_all_must_features_are_mapped():
    """Every CSV feature column has a corresponding implementation."""
    assert set(MUST_FEATURE_MAPPING.values())
    assert len(MUST_FEATURE_MAPPING) == 20


@pytest.mark.parametrize("midi_path", load_reference_values().keys())
def test_must_features_match_reference(midi_path: str, reference_data):
    """Each MUST feature matches its reference value within 1%."""
    melody = create_melody_from_file(midi_path)
    computed = get_must_features(melody)
    expected_row = reference_data[midi_path]

    assert set(computed.keys()) == set(MUST_FEATURE_MAPPING.values())
    assert len(computed) == 20

    failures = []
    for csv_name, func_name in MUST_FEATURE_MAPPING.items():
        expected = expected_row[csv_name]
        actual = computed[func_name]
        tolerance = 1e-10 if expected == 0.0 else abs(expected) * 0.01
        if abs(expected - actual) >= tolerance:
            failures.append(
                f"{csv_name}: expected {expected}, got {actual}, "
                f"diff {abs(expected - actual):.6g}, tolerance {tolerance:.6g}"
            )

    assert not failures, "\n".join(failures)
