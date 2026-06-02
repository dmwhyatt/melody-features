"""Tests for taxonomy-aligned timing statistics."""

from melody_features.features import (
    TIMING_STAT_CATEGORIES,
    _get_category_display_name,
    _init_timing_stats,
    process_melody,
)
from melody_features.import_mid import import_midi
from melody_features.corpus import get_corpus_files


def test_timing_stat_categories_cover_taxonomy():
    """Every FeatureType category used in batch extraction has a timing bucket."""
    expected = {
        "absolute_pitch",
        "pitch_class",
        "pitch_interval",
        "contour",
        "timing",
        "inter_onset_interval",
        "tonality",
        "metre",
        "expectation",
        "complexity",
        "lexical_diversity",
        "corpus",
        "total",
    }
    assert set(TIMING_STAT_CATEGORIES) == expected
    assert _init_timing_stats().keys() == expected


def test_process_melody_returns_all_timing_keys():
    path = get_corpus_files("essen", max_files=1)[0]
    data = import_midi(path)
    data["melody_num"] = 1
    _, _, timings = process_melody((data, None, {}, 1.5, 5, "infer_if_necessary"))
    for category in TIMING_STAT_CATEGORIES:
        assert category in timings
        assert timings[category] >= 0.0
    assert timings["lexical_diversity"] > 0.0
    assert timings["corpus"] == 0.0  # no corpus_stats in this call


def test_timing_display_names_match_taxonomy():
    assert _get_category_display_name("timing") == "Timing"
    assert _get_category_display_name("inter_onset_interval") == "Inter-Onset Interval"
    assert _get_category_display_name("lexical_diversity") == "Lexical Diversity"
    assert _get_category_display_name("absolute_pitch") == "Absolute Pitch"
