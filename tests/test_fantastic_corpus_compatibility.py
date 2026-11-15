"""Prevent old integer token codes from being matched to the new vocabulary."""

from types import SimpleNamespace

import pytest

from melody_features.core.representations import Melody
from melody_features.corpus import load_corpus_stats, save_corpus_stats
from melody_features.features import get_fantastic_features
from melody_features.feature_definitions.corpus import (
    InverseEntropyWeighting,
    _setup_corpus_statistics,
    get_corpus_features,
    get_ngram_document_frequency,
    max_log_df,
)
from melody_features.melody_tokenizer import FANTASTIC_TOKENIZER_VERSION, MType


def second_corpus():
    return {
        "tokenizer_version": FANTASTIC_TOKENIZER_VERSION,
        "phrase_gap": 1.5, "n_range": (1, 5), "corpus_size": 8,
        "document_frequencies": {
            "((1, 0),)": {"count": 8}, "((1, 2),)": {"count": 8},
            "((1, 0), (1, 2))": {"count": 8},
        },
    }


def melody(pitches=(60, 62, 64)):
    return Melody.from_notes(list(pitches), [0, .5, 1], [.4, .9, 1.4], tempo=120)


@pytest.mark.parametrize("version", [None, 1, 999])
@pytest.mark.parametrize("consumer", [get_corpus_features, max_log_df, get_fantastic_features])
def test_legacy_or_unknown_versions_fail_at_public_feature_entry_points(version, consumer):
    stats = second_corpus()
    # Old code 2 denotes major seconds, but the corrected code 2 denotes thirds.
    stats["document_frequencies"] = {
        k.replace("(1,", "(2,"): v for k, v in stats["document_frequencies"].items()
    }
    if version is None:
        stats.pop("tokenizer_version")
    else:
        stats["tokenizer_version"] = version
    with pytest.raises(ValueError, match="Regenerate with make_corpus_stats"):
        consumer(melody((60, 64, 68)), corpus_stats=stats, phrase_gap=1.5, max_ngram_order=2)


@pytest.mark.parametrize("key,value", [
    ("phrase_gap", None), ("phrase_gap", 2.0), ("phrase_gap", float("nan")),
    ("n_range", None), ("n_range", [1, 1]), ("n_range", [2, 5]),
])
def test_missing_or_incompatible_settings_are_rejected(key, value):
    stats = second_corpus()
    if value is None:
        stats.pop(key)
    else:
        stats[key] = value
    with pytest.raises(ValueError, match=key):
        get_corpus_features(melody(), stats, 1.5, 2)


def test_compatible_roundtrip_and_larger_ngram_range_are_accepted(tmp_path):
    path = str(tmp_path / "current")
    save_corpus_stats(second_corpus(), path)
    stats = load_corpus_stats(path)
    assert stats["n_range"] == [1, 5]  # JSON changes tuples to lists.
    assert get_corpus_features(melody(), stats, 1.5, 2)["max_log_df"] == 3
    assert max_log_df(melody(), stats, 1.5, 2) == 3
    assert get_corpus_features(melody((60, 64, 68)), stats, 1.5, 2)["max_log_df"] == 0
    assert get_ngram_document_frequency((MType(1, 0),), stats) == 8
    assert len(InverseEntropyWeighting({(MType(1, 0),): 1}, stats).combined_weights) == 1


def test_legacy_json_is_inspectable_but_cannot_be_used(tmp_path):
    stats = second_corpus()
    stats.pop("tokenizer_version")
    path = str(tmp_path / "legacy")
    save_corpus_stats(stats, path)
    loaded = load_corpus_stats(path)
    assert loaded["document_frequencies"] == stats["document_frequencies"]
    with pytest.raises(ValueError, match="tokenizer_version"):
        get_corpus_features(melody(), loaded, 1.5, 2)
    with pytest.raises(ValueError, match="tokenizer_version"):
        get_ngram_document_frequency((MType(1, 0),), loaded)
    with pytest.raises(ValueError, match="tokenizer_version"):
        InverseEntropyWeighting({(MType(1, 0),): 1}, loaded)


def test_versioned_filename_does_not_override_incompatible_metadata(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    config = SimpleNamespace(corpus=None, fantastic=SimpleNamespace(
        corpus=corpus_dir, phrase_gap=1.5, max_ngram_order=2,
    ))
    path = tmp_path / f"corpus_corpus_stats_v{FANTASTIC_TOKENIZER_VERSION}_pg1p5_n2.json"
    stats = second_corpus()
    stats["tokenizer_version"] = 1
    save_corpus_stats(stats, str(path))
    with pytest.raises(ValueError, match="tokenizer_version"):
        _setup_corpus_statistics(config, str(tmp_path / "features.csv"))
