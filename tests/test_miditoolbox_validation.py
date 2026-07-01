"""Validate MIDI Toolbox feature implementations against reference values."""

import csv
import json
import math
from pathlib import Path
from typing import Dict, Hashable, List

import numpy as np
import pytest

from melody_features.algorithms.meter_estimation import duration_accent
from melody_features.core.representations import Melody
from melody_features.feature_definitions.pitch_class import _pcdist2_matrix, pcdist2
from melody_features.feature_definitions.pitch_interval import (
    _ivdist2_categories,
    _ivdist2_matrix,
    ivdist2,
)
from melody_features.feature_definitions.timing import _durdist2_matrix, durdist2
from melody_features.features import get_midi_toolbox_features
from melody_features.io.midi import import_midi

REFERENCE_CSV = Path(__file__).parent / "miditoolbox_reference_values.csv"

EXCLUDED_FEATURES = frozenset({"comb_contour_matrix", "get_comb_contour_matrix"})

SCALAR_FEATURES = (
    "ambitus",
    "pitch_range",
    "complebm_optimal",
    "complebm_pitch",
    "complebm_rhythm",
    "compltrans",
    "duration_accent_std",
    "gradus",
    "mean_duration_accent",
    "mean_melodic_accent",
    "mean_melodic_attraction",
    "mean_mobility",
    "mean_tessitura",
    "melodic_accent_std",
    "melodic_attraction_std",
    "meter_accent",
    "mobility_std",
    "narmour_closure",
    "narmour_intervallic_difference",
    "narmour_proximity",
    "narmour_registral_direction",
    "narmour_registral_return",
    "npvi",
    "onset_autocorr_peak",
    "tessitura_std",
)

DISTRIBUTION_FEATURES = (
    "pcdist1",
    "ivdist1",
    "ivdirdist1",
    "ivsizedist1",
    "durdist1",
)

SEQUENCE_FEATURES = (
    "duration_accent",
    "mobility",
    "melodic_accent",
    "melodic_attraction",
    "tessitura",
    "metric_hierarchy",
    "onset_autocorrelation",
)


def _parse_float_cell(cell: str) -> float:
    text = str(cell).strip()
    if not text:
        return 0.0
    lowered = text.lower()
    if lowered in {"nan", "null"}:
        return float("nan")
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if lowered in {"-inf", "-infinity"}:
        return float("-inf")
    return float(text)


def _parse_json_array_cell(cell: str) -> List[float]:
    if not cell or not str(cell).strip():
        return []
    text = str(cell)
    # MATLAB exports bare Inf; avoid touching Infinity/NaN from Python json.dumps.
    if "Infinity" not in text:
        text = text.replace("-Inf", "-Infinity").replace("Inf", "Infinity")
    parsed = json.loads(text)
    result: List[float] = []
    for value in parsed:
        if value is None:
            result.append(float("nan"))
        else:
            result.append(float(value))
    return result


def _values_match(expected: float, actual: float, tolerance: float) -> bool:
    if math.isnan(expected):
        return math.isnan(actual)
    if math.isinf(expected):
        return math.isinf(actual) and math.copysign(1.0, expected) == math.copysign(
            1.0, actual
        )
    return abs(expected - actual) < tolerance


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


def _parse_sequence_cell(cell: str) -> List[float]:
    return _parse_json_array_cell(cell)


def load_scalar_reference_values() -> Dict[str, Dict[str, float]]:
    reference_data: Dict[str, Dict[str, float]] = {}
    with open(REFERENCE_CSV, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            midi_path = row["midi_path"]
            reference_data[midi_path] = {
                feature: _parse_float_cell(row[feature]) for feature in SCALAR_FEATURES
            }
    return reference_data


def load_distribution_reference_values() -> Dict[str, Dict[str, Dict[str, float]]]:
    reference_data: Dict[str, Dict[str, Dict[str, float]]] = {}
    with open(REFERENCE_CSV, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            midi_path = row["midi_path"]
            reference_data[midi_path] = {
                feature: _parse_distribution_cell(row[feature])
                for feature in DISTRIBUTION_FEATURES
            }
    return reference_data


def load_sequence_reference_values() -> Dict[str, Dict[str, List[float]]]:
    reference_data: Dict[str, Dict[str, List[float]]] = {}
    with open(REFERENCE_CSV, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            midi_path = row["midi_path"]
            reference_data[midi_path] = {
                feature: _parse_sequence_cell(row[feature])
                for feature in SEQUENCE_FEATURES
            }
    return reference_data


def create_melody_from_file(file_path: str) -> Melody:
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


def _sequence_failures(
    feature_name: str,
    expected: List[float],
    actual: List[float],
) -> List[str]:
    failures: List[str] = []
    if len(expected) != len(actual):
        failures.append(
            f"{feature_name}: length expected {len(expected)}, got {len(actual)}"
        )
        return failures
    for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
        tolerance = 1e-10 if expected_value == 0.0 else abs(expected_value) * 0.01
        if not _values_match(expected_value, actual_value, tolerance):
            failures.append(
                f"{feature_name}[{index}]: expected {expected_value}, "
                f"got {actual_value}, diff {abs(expected_value - actual_value):.6g}, "
                f"tolerance {tolerance:.6g}"
            )
            if len(failures) >= 5:
                failures.append(f"{feature_name}: ... additional mismatches omitted")
                break
    return failures


@pytest.fixture(scope="module")
def scalar_reference_data():
    return load_scalar_reference_values()


@pytest.fixture(scope="module")
def distribution_reference_data():
    return load_distribution_reference_values()


@pytest.fixture(scope="module")
def sequence_reference_data():
    return load_sequence_reference_values()


def test_reference_csv_columns_complete():
    with open(REFERENCE_CSV, encoding="utf-8") as handle:
        fieldnames = set(csv.DictReader(handle).fieldnames or [])
    expected = (
        {"stimulus", "midi_path"}
        | set(SCALAR_FEATURES)
        | set(DISTRIBUTION_FEATURES)
        | set(SEQUENCE_FEATURES)
    )
    assert expected.issubset(fieldnames)


def test_reference_paths_consistent():
    scalar_paths = set(load_scalar_reference_values().keys())
    distribution_paths = set(load_distribution_reference_values().keys())
    sequence_paths = set(load_sequence_reference_values().keys())
    assert distribution_paths == scalar_paths == sequence_paths


def test_excluded_contour_matrix_not_in_reference():
    with open(REFERENCE_CSV, encoding="utf-8") as handle:
        fieldnames = set(csv.DictReader(handle).fieldnames or [])
    assert not EXCLUDED_FEATURES.intersection(fieldnames)


@pytest.mark.parametrize("midi_path", load_scalar_reference_values().keys())
def test_miditoolbox_scalars_match_reference(midi_path: str, scalar_reference_data):
    melody = create_melody_from_file(midi_path)
    computed = get_midi_toolbox_features(melody)
    expected_row = scalar_reference_data[midi_path]

    failures = []
    for feature_name in SCALAR_FEATURES:
        expected = expected_row[feature_name]
        actual = float(computed[feature_name])
        tolerance = 1e-10 if expected == 0.0 else abs(expected) * 0.01
        if not _values_match(expected, actual, tolerance):
            failures.append(
                f"{feature_name}: expected {expected}, got {actual}, "
                f"diff {abs(expected - actual):.6g}, tolerance {tolerance:.6g}"
            )

    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("midi_path", load_scalar_reference_values().keys())
def test_miditoolbox_distributions_match_reference(
    midi_path: str,
    distribution_reference_data,
):
    melody = create_melody_from_file(midi_path)
    computed = get_midi_toolbox_features(melody)
    expected_row = distribution_reference_data[midi_path]

    failures = []
    for feature_name in DISTRIBUTION_FEATURES:
        actual = encode_distribution(computed[feature_name])
        expected = expected_row[feature_name]
        failures.extend(_distribution_failures(feature_name, expected, actual))

    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("midi_path", load_scalar_reference_values().keys())
def test_miditoolbox_sequences_match_reference(
    midi_path: str,
    sequence_reference_data,
):
    melody = create_melody_from_file(midi_path)
    computed = get_midi_toolbox_features(melody)
    expected_row = sequence_reference_data[midi_path]

    failures = []
    for feature_name in SEQUENCE_FEATURES:
        actual = [float(value) for value in computed[feature_name]]
        expected = expected_row[feature_name]
        failures.extend(_sequence_failures(feature_name, expected, actual))

    assert not failures, "\n".join(failures)


def test_ivdist2_categories_match_matlab_encoding():
    pitches = [60, 62, 74, 62]
    categories = _ivdist2_categories(pitches)
    assert categories.tolist() == [0, 2, 0, 0]


def test_ivdist2_transition_weights_use_duration_accent_sum():
    pitches = [60, 62, 64]
    starts = [0.0, 1.0, 2.0]
    ends = [1.0, 2.0, 3.0]
    accents = duration_accent(starts, ends)
    matrix = _ivdist2_matrix(pitches, starts, ends)

    expected_weight = accents[0] + accents[1]
    assert matrix[12, 14] == pytest.approx(expected_weight / (2 * expected_weight + 1e-12))
    assert matrix[14, 14] == pytest.approx(expected_weight / (2 * expected_weight + 1e-12))
    assert matrix.sum() == pytest.approx(1.0)


def test_pcdist2_transition_weights_use_duration_accent_product():
    pitches = [60, 62, 64]
    starts = [0.0, 1.0, 2.0]
    ends = [1.0, 2.0, 3.0]
    accents = duration_accent(starts, ends)
    matrix = _pcdist2_matrix(pitches, starts, ends)

    w01 = accents[0] * accents[1]
    w12 = accents[1] * accents[2]
    assert matrix[0, 2] == pytest.approx(w01 / (w01 + w12 + 1e-12))
    assert matrix[2, 4] == pytest.approx(w12 / (w01 + w12 + 1e-12))


def test_durdist2_constant_durations_single_transition():
    starts = [0.0, 0.5, 1.0]
    ends = [0.5, 1.0, 1.5]
    matrix = _durdist2_matrix(starts, ends, tempo=120.0)

    assert matrix.shape == (9, 9)
    assert matrix[4, 4] == pytest.approx(1.0)
    result = durdist2(starts, ends, tempo=120.0)
    assert list(result.keys()) == [(5, 5)]
    assert list(result.values())[0] == pytest.approx(1.0)


def test_dist2_empty_inputs_return_empty_dicts():
    assert ivdist2([], [], []) == {}
    assert pcdist2([], [], []) == {}
    assert durdist2([], [], []) == {}


def test_dist2_matrices_normalize():
    pitches = [60, 62, 64, 67, 65]
    starts = [0.0, 0.5, 1.0, 1.5, 2.0]
    ends = [0.5, 1.0, 1.5, 2.0, 2.5]

    assert _ivdist2_matrix(pitches, starts, ends).sum() == pytest.approx(1.0)
    assert _pcdist2_matrix(pitches, starts, ends).sum() == pytest.approx(1.0)
    assert _durdist2_matrix(starts, ends, tempo=120.0).sum() == pytest.approx(1.0)

    iv_dict = ivdist2(pitches, starts, ends)
    pc_dict = pcdist2(pitches, starts, ends)
    du_dict = durdist2(starts, ends, tempo=120.0)
    assert np.isclose(sum(iv_dict.values()), 1.0)
    assert np.isclose(sum(pc_dict.values()), 1.0)
    assert np.isclose(sum(du_dict.values()), 1.0)
