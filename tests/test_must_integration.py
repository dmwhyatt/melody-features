"""Integration tests for MUST feature wiring."""

import tempfile
from pathlib import Path

import pytest

import melody_features.features as features_module
from melody_features.core.representations import Melody
from melody_features.features import get_all_features, get_complexity_features, get_must_features
from melody_features.feature_registry import get_features_by_source
from melody_features.io.midi import import_midi
from tests.helpers.midi import create_test_midi_file

CORPUS_MIDI = (
    Path(__file__).parent.parent
    / "src/melody_features/corpora/essen_folksong_collection/appenzel.mid"
)

MUST_FEATURE_NAMES = {
    "bisect_unbalance",
    "center_mass_offset",
    "event_heterogeneity",
    "av_abs_interval",
    "mel_abruptness",
    "dur_abruptness",
    "rhythm_abruptness",
    "asym_total",
    "asym_index",
    "event_density",
    "av_local_p1_entropy",
    "p1_entropy",
    "p2_entropy",
    "p3_entropy",
    "i1_entropy",
    "i2_entropy",
    "d1_entropy",
    "d2_entropy",
    "d3_entropy",
    "wp_entropy",
}


@pytest.fixture
def appenzel_melody() -> Melody:
    return Melody(import_midi(str(CORPUS_MIDI)))


def test_must_features_registered_by_source():
    must_features = get_features_by_source(features_module, "must")
    assert set(must_features.keys()) == MUST_FEATURE_NAMES
    for func in must_features.values():
        assert "must" in func._feature_sources


def test_must_features_match_complexity_subset(appenzel_melody):
    must_values = get_must_features(appenzel_melody)
    complexity_values = get_complexity_features(appenzel_melody)

    assert set(must_values.keys()).issubset(complexity_values.keys())
    for name, expected in must_values.items():
        assert complexity_values[name] == expected


def test_must_features_in_pipeline_output():
    with tempfile.TemporaryDirectory() as temp_dir:
        midi_path = Path(temp_dir) / "test.mid"
        create_test_midi_file([60, 62, 64], [0, 1, 2], [1, 2, 3], filepath=str(midi_path))

        results = get_all_features(str(midi_path), skip_idyom=True)
        columns = set(results.columns)

    for feature_name in ("bisect_unbalance", "wp_entropy", "p1_entropy"):
        assert f"complexity.{feature_name}" in columns
