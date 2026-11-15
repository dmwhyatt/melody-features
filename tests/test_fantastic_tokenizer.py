"""FANTASTIC Table 1 classes and within-phrase m-type regressions."""

import numpy as np
import pytest

from melody_features.core.representations import Melody
from melody_features.corpus import process_melody_ngrams
from melody_features.feature_definitions.corpus import _compute_corpus_feature_bundle
from melody_features.feature_definitions.lexical_diversity import get_mtype_features
from melody_features.melody_tokenizer import FantasticTokenizer, MType
from melody_features.ngram_counter import NGramCounter


# Müllensiefen (2009), Table 1: unison, seconds, thirds, fourth,
# tritone, fifth, sixths, sevenths and octave; section 6.2.1 adds >octave.
@pytest.mark.parametrize("semitones,code", [
    (-24, -9), (-13, -9), (-12, -8), (-11, -7), (-10, -7),
    (-9, -6), (-8, -6), (-7, -5), (-6, -4), (-5, -3),
    (-4, -2), (-3, -2), (-2, -1), (-1, -1), (0, 0),
    (1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 4),
    (7, 5), (8, 6), (9, 6), (10, 7), (11, 7), (12, 8),
    (13, 9), (24, 9),
])
def test_fantastic_signed_interval_classes(semitones, code):
    assert FantasticTokenizer()._classify_pitch_interval(semitones) == code


def test_fantastic_has_19_classes_and_distinguishes_return_from_ascent():
    tok = FantasticTokenizer()
    assert len({tok._classify_pitch_interval(i) for i in range(-127, 128)}) == 19
    ascent = tok.tokenize_melody([60, 62, 64], [0, .5, 1], [.5, 1, 1.5])
    returning = tok.tokenize_melody([60, 62, 60], [0, .5, 1], [.5, 1, 1.5])
    assert ascent == [MType(1, 0), MType(1, 2)]
    assert returning == [MType(1, 0), MType(-1, 2)]
    assert ascent != returning
    # Transposition and uniform time scaling do not change the vocabulary.
    assert ascent == tok.tokenize_melody([65, 67, 69], [0, 1, 2], [1, 2, 3])


@pytest.mark.parametrize("interval,expected", [(-12, -4), (-7, -3), (-3, -2),
    (-1, -1), (0, 0), (2, 1), (4, 2), (7, 3), (12, 4)])
def test_simile_classes_are_unchanged(interval, expected):
    assert FantasticTokenizer("SIMILE")._classify_pitch_interval(interval) == expected


def two_phrases():
    return Melody.from_notes(
        [60, 62, 64, 67, 65, 64],
        [0, .5, 1, 3, 3.5, 4], [.4, .9, 1.4, 3.4, 3.9, 4.4], tempo=120,
    )


def test_segmentation_splits_before_note_after_gap():
    tok = FantasticTokenizer()
    segments = tok.segment_melody(two_phrases(), phrase_gap=1.5)
    assert [s.pitches for s in segments] == [[60, 62, 64], [67, 65, 64]]
    assert [s.starts for s in segments] == [[0, .5, 1], [0, .5, 1]]
    assert tok.tokenize_phrases(two_phrases()) == [
        [MType(1, 0), MType(1, 2)], [MType(-1, 0), MType(-1, 2)]
    ]


def test_phrase_gap_uses_requested_units_and_tempo_map():
    tok = FantasticTokenizer()
    melody = Melody.from_notes([60, 62, 64], [0, .75, 1.75], [.5, 1, 2], tempo=120)
    # The first gap equals 1.5 quarters: split only on a strictly larger gap.
    assert [s.pitches for s in tok.segment_melody(melody)] == [[60, 62], [64]]
    assert len(tok.segment_melody(melody, units="seconds")) == 1
    slower = Melody.from_notes([60, 62, 64], [0, 1.5, 3.5], [1, 2, 4], tempo=60)
    assert [s.pitches for s in tok.segment_melody(slower)] == [[60, 62], [64]]
    changing = Melody.from_notes([60, 62, 64], [0, .5, 2], [.4, 1, 2.5],
                                tempo=60, tempo_changes=[(0, 60), (1, 120)])
    assert [s.pitches for s in tok.segment_melody(changing)] == [[60, 62], [64]]


def test_empty_and_single_note_phrases():
    tok = FantasticTokenizer()
    assert tok.tokenize_phrases(Melody.from_notes([], [], [])) == []
    assert tok.tokenize_phrases(Melody.from_notes([60], [0], [1])) == [[]]


@pytest.mark.parametrize("bpm", [60, 100, 108, 120])
@pytest.mark.parametrize("excess", [0, 1e-6])
def test_phrase_gap_boundary_is_stable_across_tempi(bpm, excess):
    starts = [i * (1.5 + excess) * 60 / bpm for i in range(16)]
    melody = Melody.from_notes([60 + i % 3 for i in range(16)], starts,
                              [s + .1 for s in starts], tempo=bpm)
    sizes = [len(s.pitches) for s in FantasticTokenizer().segment_melody(melody)]
    assert sizes == ([16] if excess == 0 else [1] * 16)


def test_phrase_tokenization_updates_all_counts_without_boundary_windows():
    tok = FantasticTokenizer()
    tok.tokenize_phrases(two_phrases())
    up0, up, down0, down = MType(1, 0), MType(1, 2), MType(-1, 0), MType(-1, 2)
    assert tok.ngram_counts() == {
        (up0,): 1, (up,): 1, (down0,): 1, (down,): 1,
        (up0, up): 1, (down0, down): 1,
    }
    assert tok.phrases == [up0, up, down0, down]
    # Switching back to a single phrase replaces the multi-phrase state.
    tok.tokenize_melody([60, 61], [0, .5], [.4, .9])
    assert tok.ngram_counts() == {(up0,): 1}


@pytest.mark.parametrize("pitches,starts,ends", [([], [], []), ([60], [0], [1])])
@pytest.mark.parametrize("method", ["tokenize_melody", "tokenize_phrases"])
def test_short_inputs_clear_previous_tokenizer_state(pitches, starts, ends, method):
    tok = FantasticTokenizer()
    tok.tokenize_phrases(two_phrases())
    # Populate derived caches as well as the raw counts.
    assert tok.ngram_counter.total_tokens == 4
    assert tok.ngram_counter.freq_spec
    assert tok.ngram_counter.count_values
    if method == "tokenize_melody":
        tok.tokenize_melody(pitches, starts, ends)
    else:
        tok.tokenize_phrases(Melody.from_notes(pitches, starts, ends))
    assert tok.ngram_counts() == {} and tok.phrases == []
    assert tok.ngram_counter.total_tokens == 0
    assert tok.ngram_counter.freq_spec == {}
    assert tok.ngram_counter.count_values == []


def test_count_sequences_does_not_bridge_phrases_and_resets_caches():
    counter = NGramCounter()
    counter.count_sequences([["a", "b"], [], ["a", "b"]], max_order=3)
    assert counter.get_counts() == {("a",): 2, ("b",): 2, ("a", "b"): 2}
    assert counter.total_tokens == 4
    assert counter.count_values == [2, 2, 2]
    assert counter.freq_spec == {2: 3}
    assert counter.mean_entropy == pytest.approx(.5)
    counter.count_ngrams(["x"], max_order=2)
    assert counter.get_counts() == {("x",): 1}
    assert counter.total_tokens == 1
    assert counter.count_values == [1]
    assert counter.freq_spec == {1: 1}
    counter.count_sequences([], max_order=2)
    assert counter.get_counts() == {} and counter.total_tokens == 0


def test_lexical_and_corpus_paths_use_only_within_phrase_ngrams():
    melody = two_phrases()
    up0, up, down0, down = MType(1, 0), MType(1, 2), MType(-1, 0), MType(-1, 2)
    expected = {(up0,), (up,), (down0,), (down,), (up0, up), (down0, down)}
    assert process_melody_ngrams((melody, (1, 5), 1.5)) == expected
    assert process_melody_ngrams((melody, (2, 5), 1.5)) == {(up0, up), (down0, down)}
    # Four equally frequent unigrams (entropy 2), two bigrams (entropy 1).
    assert get_mtype_features(melody, 1.5, 5)["mean_entropy"] == pytest.approx(1.5)
    # A deliberately tempting DF entry for the artificial boundary bigram.
    stats = {"tokenizer_version": 2, "phrase_gap": 1.5, "n_range": [1, 5],
             "document_frequencies": {str((up, down0)): {"count": 1024}}}
    features = _compute_corpus_feature_bundle(melody, stats, 1.5, 5)
    assert features["max_log_df"] == 0
    # Real within-phrase matches are still used by both TF/DF and weighting.
    stats["document_frequencies"].update({str(k): {"count": 8} for k in expected})
    features = _compute_corpus_feature_bundle(melody, stats, 1.5, 5)
    assert features["max_log_df"] == 3
    assert np.isfinite(features["mean_global_local_weight"])
