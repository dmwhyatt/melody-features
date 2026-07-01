"""Tests for MUST distribution tokenization."""

import numpy as np
import pytest

from melody_features.algorithms import must as must_algorithms
from melody_features.core.representations import Melody
from melody_features.melody_tokenizer import MustDistribution, MustTokenizer
from tests.helpers.melody import make_melody


def _melody(pitches, starts, ends, tempo=120.0) -> Melody:
    return Melody(make_melody(pitches, starts, ends, tempo=tempo))


def test_pdist1_matches_p1_entropy():
    melody = _melody([60, 62, 64, 62], [0, 1, 2, 3], [1, 2, 3, 4])
    tokenizer = MustTokenizer()

    assert tokenizer.pdist1(melody).entropy() == pytest.approx(
        must_algorithms.p1_entropy(melody)
    )


def test_distribution_entropy_matches_scalar_features():
    melody = _melody([60, 62, 64, 67], [0, 1, 2, 3], [1, 2, 3, 4])
    tokenizer = MustTokenizer()

    assert tokenizer.pdist2(melody).entropy() == pytest.approx(must_algorithms.p2_entropy(melody))
    assert tokenizer.pdist3(melody).entropy() == pytest.approx(must_algorithms.p3_entropy(melody))
    assert tokenizer.idist1(melody).entropy() == pytest.approx(must_algorithms.i1_entropy(melody))
    assert tokenizer.idist2(melody).entropy() == pytest.approx(must_algorithms.i2_entropy(melody))
    assert tokenizer.ddist1(melody).entropy() == pytest.approx(must_algorithms.d1_entropy(melody))
    assert tokenizer.ddist2(melody).entropy() == pytest.approx(must_algorithms.d2_entropy(melody))
    assert tokenizer.ddist3(melody).entropy() == pytest.approx(must_algorithms.d3_entropy(melody))


def test_pdist1_as_dict():
    melody = _melody([60, 60, 62], [0, 1, 2], [1, 2, 3])
    distribution = MustTokenizer().pdist1(melody).as_dict()

    assert set(distribution.keys()) == {60, 62}
    assert np.isclose(sum(distribution.values()), 1.0)


def test_empty_melody_distributions():
    melody = _melody([], [], [])
    tokenizer = MustTokenizer()

    assert tokenizer.pdist1(melody).as_dict() == {}
    assert tokenizer.pdist1(melody).entropy() == 0.0
    assert tokenizer.pdist2(melody).as_dict() == {}
    assert tokenizer.ddist3(melody).weights.tolist() == [1.0]


def test_ddist3_short_sequence():
    melody = _melody([60, 62], [0.0, 1.0], [1.0, 2.0])
    distribution = MustTokenizer().ddist3(melody)
    assert np.allclose(distribution.weights, np.array([1.0]))


def test_must_distribution_dataclass():
    distribution = MustDistribution(
        values=np.array([60, 62]),
        weights=np.array([0.5, 0.5]),
    )
    assert distribution.as_dict() == {60: 0.5, 62: 0.5}
    assert distribution.entropy() == pytest.approx(np.log(2.0))
