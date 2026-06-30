"""Unit tests for MUST algorithm edge cases and helpers."""

import math

import numpy as np
import pytest

from melody_features.algorithms import must as must_algorithms
from melody_features.core.representations import Melody
from melody_features.features import get_must_features
from tests.helpers.melody import make_melody

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


def _melody(pitches, starts, ends, tempo=120.0) -> Melody:
    return Melody(make_melody(pitches, starts, ends, tempo=tempo))


@pytest.mark.parametrize(
    "pitches,starts,ends",
    [
        ([], [], []),
        ([60], [0.0], [1.0]),
        ([60, 62], [0.0, 1.0], [1.0, 2.0]),
        ([60] * 5, [0, 1, 2, 3, 4], [1, 2, 3, 4, 5]),
        ([60, 62, 64], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ],
    ids=["empty", "single", "two_notes", "constant_pitch", "zero_span"],
)
def test_get_must_features_returns_twenty_finite_values(pitches, starts, ends):
    melody = _melody(pitches, starts, ends)
    computed = get_must_features(melody)

    assert set(computed.keys()) == MUST_FEATURE_NAMES
    for name, value in computed.items():
        assert isinstance(value, (int, float)), name
        assert math.isfinite(float(value)), name


def test_empty_melody_returns_zeros():
    melody = _melody([], [], [])
    computed = get_must_features(melody)
    assert all(value == 0.0 for value in computed.values())


@pytest.mark.parametrize(
    "a,b,c,expected",
    [
        (1, 2, 3, 1),
        (1, 2, 2, 2),
        (1, 3, 2, 3),
        (2, 3, 2, 4),
        (5, 10, 3, 5),
        (2, 2, 3, 6),
        (2, 2, 2, 7),
        (3, 3, 1, 8),
        (2, 1, 3, 9),
        (3, 1, 3, 10),
        (5, 2, 4, 11),
        (5, 2, 2, 12),
        (5, 3, 2, 13),
    ],
)
def test_order_sign_cases(a, b, c, expected):
    assert must_algorithms._order_sign(a, b, c) == expected


def test_must_shannon_entropy_zero_distribution():
    assert must_algorithms.must_shannon_entropy(np.array([0.0, 0.0])) == 0.0


def test_duration3_distribution_short_sequence():
    melody = _melody([60, 62], [0.0, 1.0], [1.0, 2.0])
    distribution = must_algorithms._duration3_distribution(melody)
    assert np.allclose(distribution, np.array([1.0]))


def test_local_unbalance_single_note():
    melody = _melody([60], [0.0], [1.0])
    densities, center_weights = must_algorithms._local_unbalance(melody)
    assert np.allclose(densities, np.array([1.0]))
    assert np.allclose(center_weights, np.array([0.0]))


def test_internal_helpers_handle_empty_melody():
    melody = _melody([], [], [])
    assert must_algorithms._onsets_beats(melody).size == 0
    assert np.allclose(must_algorithms._pitch_distribution(np.array([])), np.array([0.0]))
    densities, center_weights = must_algorithms._local_unbalance(melody)
    assert np.allclose(densities, np.array([1.0]))
    assert np.allclose(center_weights, np.array([0.0]))
    assert must_algorithms._mirror_pitch_series(melody).size == 0
