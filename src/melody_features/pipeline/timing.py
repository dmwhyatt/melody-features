"""Timing-statistics helpers for the feature extraction pipeline."""

from typing import Dict, List


TIMING_STAT_CATEGORIES = (
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
)


def _init_timing_stats() -> Dict[str, List[float]]:
    """Return an empty timing accumulator for all taxonomy categories."""
    return {category: [] for category in TIMING_STAT_CATEGORIES}
