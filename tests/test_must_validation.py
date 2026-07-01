"""Validate MUST feature implementations against reference values."""

import csv
import json
from pathlib import Path
from typing import Dict, Hashable, List

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

MUST_DISTRIBUTION_FEATURES = (
    "pdist1",
    "pdist2",
    "pdist3",
    "idist1",
    "idist2",
    "ddist1",
    "ddist2",
    "ddist3",
)


def encode_distribution_key(key: Hashable) -> str:
    """Serialize a distribution category label for reference comparison."""
    if isinstance(key, tuple):
        return json.dumps(list(key), separators=(",", ":"))
    return json.dumps(key)


def encode_distribution(dist: Dict[Hashable, float]) -> Dict[str, float]:
    """Convert a computed distribution dict to reference form."""
    return {
        encode_distribution_key(key): float(weight)
        for key, weight in dist.items()
    }


def _parse_distribution_cell(cell: str) -> Dict[str, float]:
    if not cell or not str(cell).strip():
        return {}
    parsed = json.loads(cell)
    return {str(key): float(weight) for key, weight in parsed.items()}


def load_reference_values() -> Dict[str, Dict[str, float]]:
    """Load reference MUST scalar values from CSV."""
    reference_data: Dict[str, Dict[str, float]] = {}
    with open(REFERENCE_CSV, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            midi_path = row["midi_path"]
            reference_data[midi_path] = {
                feature: float(row[feature])
                for feature in MUST_FEATURE_MAPPING
            }
    return reference_data


def load_distribution_reference_values() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load reference MUST distribution values from CSV JSON columns."""
    reference_data: Dict[str, Dict[str, Dict[str, float]]] = {}
    with open(REFERENCE_CSV, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            midi_path = row["midi_path"]
            reference_data[midi_path] = {
                feature: _parse_distribution_cell(row[feature])
                for feature in MUST_DISTRIBUTION_FEATURES
            }
    return reference_data


def create_melody_from_file(file_path: str) -> Melody:
    """Create a melody from a MIDI file."""
    midi_data = import_midi(file_path)
    return Melody(midi_data)


def _distribution_failures(
    feature_name: str,
    expected: Dict[str, float],
    actual: Dict[str, float],
) -> List[str]:
    failures: List[str] = []
    missing = set(expected.keys()) - set(actual.keys())
    extra = set(actual.keys()) - set(expected.keys())
    if missing:
        failures.append(f"{feature_name}: missing keys {sorted(missing)[:5]}")
    if extra:
        failures.append(f"{feature_name}: unexpected keys {sorted(extra)[:5]}")
    for key, expected_weight in expected.items():
        if key not in actual:
            continue
        actual_weight = actual[key]
        tolerance = 1e-10 if expected_weight == 0.0 else abs(expected_weight) * 0.01
        if abs(expected_weight - actual_weight) >= tolerance:
            failures.append(
                f"{feature_name}[{key}]: expected {expected_weight}, "
                f"got {actual_weight}, diff {abs(expected_weight - actual_weight):.6g}, "
                f"tolerance {tolerance:.6g}"
            )
    return failures


@pytest.fixture(scope="module")
def reference_data():
    return load_reference_values()


@pytest.fixture(scope="module")
def distribution_reference_data():
    return load_distribution_reference_values()


def test_reference_csv_columns_match_mapping():
    with open(REFERENCE_CSV, encoding="utf-8") as handle:
        fieldnames = set(csv.DictReader(handle).fieldnames or [])
    assert set(MUST_FEATURE_MAPPING.keys()).issubset(fieldnames)
    assert set(MUST_DISTRIBUTION_FEATURES).issubset(fieldnames)


def test_distribution_reference_paths_match_scalars():
    scalar_paths = set(load_reference_values().keys())
    distribution_paths = set(load_distribution_reference_values().keys())
    assert distribution_paths == scalar_paths


def test_distribution_reference_features_complete():
    reference = load_distribution_reference_values()
    for midi_path, features in reference.items():
        assert set(features.keys()) == set(MUST_DISTRIBUTION_FEATURES), midi_path


def test_all_must_features_are_mapped():
    """Every CSV feature column has a corresponding scalar implementation."""
    assert set(MUST_FEATURE_MAPPING.values())
    assert len(MUST_FEATURE_MAPPING) == 20


@pytest.mark.parametrize("midi_path", load_reference_values().keys())
def test_must_features_match_reference(midi_path: str, reference_data):
    """Each MUST scalar feature matches its reference value within 1%."""
    melody = create_melody_from_file(midi_path)
    computed = get_must_features(melody)
    expected_row = reference_data[midi_path]
    scalar_features = {
        name: value
        for name, value in computed.items()
        if name in MUST_FEATURE_MAPPING.values()
    }

    assert set(scalar_features.keys()) == set(MUST_FEATURE_MAPPING.values())
    assert len(scalar_features) == 20

    failures = []
    for csv_name, func_name in MUST_FEATURE_MAPPING.items():
        expected = expected_row[csv_name]
        actual = scalar_features[func_name]
        tolerance = 1e-10 if expected == 0.0 else abs(expected) * 0.01
        if abs(expected - actual) >= tolerance:
            failures.append(
                f"{csv_name}: expected {expected}, got {actual}, "
                f"diff {abs(expected - actual):.6g}, tolerance {tolerance:.6g}"
            )

    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("midi_path", load_reference_values().keys())
def test_must_distributions_match_reference(
    midi_path: str,
    distribution_reference_data,
):
    """Each MUST distribution matches its reference weights within 1%."""
    melody = create_melody_from_file(midi_path)
    computed = get_must_features(melody)
    expected_row = distribution_reference_data[midi_path]

    failures = []
    for feature_name in MUST_DISTRIBUTION_FEATURES:
        actual = encode_distribution(computed[feature_name])
        expected = expected_row[feature_name]
        failures.extend(_distribution_failures(feature_name, expected, actual))

    assert not failures, "\n".join(failures)
