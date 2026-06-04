"""Output-column helpers for feature extraction results."""

import logging
from typing import Dict, List

from .timing import TIMING_STAT_CATEGORIES


def _get_category_display_name(category: str, feature_name: str = None) -> str:
    """Map internal category names to display names.

    Parameters
    ----------
    category : str
        Internal category name (e.g., "pitch_features", "pitch_class_features", "pitch")
    feature_name : str, optional
        Feature name to help determine category (e.g., for mtype features or IOI features)

    Returns
    -------
    str
        Display name for the category (e.g., "Absolute Pitch", "Pitch Class")
    """
    mtype_features = {"yules_k", "simpsons_d", "sichels_s", "honores_h", "mean_entropy", "mean_productivity"}

    if feature_name and feature_name in mtype_features:
        return "Lexical Diversity"

    if category == "rhythm_features" and feature_name and "ioi" in feature_name.lower():
        return "Inter-Onset Interval"

    category_mapping = {
        "pitch_features": "Absolute Pitch",
        "pitch_class_features": "Pitch Class",
        "interval_features": "Pitch Interval",
        "contour_features": "Contour",
        "rhythm_features": "Timing",
        "tonality_features": "Tonality",
        "metre_features": "Metre",
        "expectation_features": "Expectation",
        "complexity_features": "Complexity",
        "corpus_features": "Corpus",
    }

    # mapping for timing_stats keys (aligned with FeatureType taxonomy)
    timing_mapping = {
        "absolute_pitch": "Absolute Pitch",
        "pitch": "Absolute Pitch",  # legacy
        "pitch_class": "Pitch Class",
        "pitch_interval": "Pitch Interval",
        "interval": "Pitch Interval",  # legacy
        "contour": "Contour",
        "timing": "Timing",
        "rhythm": "Timing",  # legacy
        "inter_onset_interval": "Inter-Onset Interval",
        "tonality": "Tonality",
        "metre": "Metre",
        "expectation": "Expectation",
        "complexity": "Complexity",
        "lexical_diversity": "Lexical Diversity",
        "corpus": "Corpus",
        "total": "Total",
    }

    # Handle IDyOM features
    if category.startswith("idyom_"):
        return "IDyOM"

    # try timing_stats mapping first
    if category in timing_mapping:
        return timing_mapping[category]

    # try DataFrame category mapping
    if category in category_mapping:
        return category_mapping[category]

    # fallback: format the category name
    return category.replace("_features", "").replace("_", " ").title()


def rename_feature_columns(df):
    """Rename internal feature columns to display-oriented category names."""
    column_rename_map = {}
    for col in df.columns:
        if col in ["melody_num", "melody_id"]:
            continue
        if col.startswith("idyom_"):
            if "_features" in col:
                category = col.rsplit("_features", 1)[0]
            else:
                category = col
            display_name = _get_category_display_name(category)
            display_name_lower = display_name.lower().replace(" ", "_").replace("-", "_")
            if "." in col:
                _, feature_name = col.split(".", 1)
                if category.startswith("idyom_"):
                    config_name = category[6:]
                    new_col_name = f"{display_name_lower}.{config_name}_{feature_name}"
                else:
                    new_col_name = f"{display_name_lower}.{feature_name}"
            else:
                new_col_name = col
            column_rename_map[col] = new_col_name
        elif "." in col:
            category, feature_name = col.split(".", 1)
            display_name = _get_category_display_name(category, feature_name)
            display_name_lower = display_name.lower().replace(" ", "_").replace("-", "_")
            column_rename_map[col] = f"{display_name_lower}.{feature_name}"

    return df.rename(columns=column_rename_map)


def log_timing_statistics(
    logger: logging.Logger,
    timing_stats: Dict[str, List[float]],
    total_seconds: float,
) -> None:
    """Log aggregate timing statistics for processed melodies."""
    logger.info(f"Total processing time: {total_seconds:.2f} seconds")
    logger.info("Timing Statistics (average milliseconds per melody):")
    for category in TIMING_STAT_CATEGORIES:
        times = timing_stats.get(category, [])
        if times:
            avg_time = sum(times) / len(times) * 1000
            display_name = _get_category_display_name(category)
            logger.info(f"{display_name:22s}: {avg_time:8.2f}ms")
