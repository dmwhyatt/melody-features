"""
This module contains functions to compute features from melodies.
Features are the product of an input list and at least one algorithm.
"""

__author__ = "David Whyatt"

import warnings
from importlib import resources

from .feature_decorators import (
    fantastic, idyom, midi_toolbox, melsim, jsymbolic, novel, simile, partitura, must,
    FeatureType, feature_type, interval, pitch_class, contour, tonality, metre, absolute, timing,
    lexical_diversity, expectation, complexity,
    pitch, rhythm, both
)
from .feature_dispatch import collect_feature_values as _dispatch_collect_feature_values
from .feature_dispatch import invoke_feature as _dispatch_invoke_feature
from .feature_registry import get_features_by_domain as _registry_get_features_by_domain
from .feature_registry import (
    get_features_by_domain_and_types as _registry_get_features_by_domain_and_types,
)
from .feature_registry import get_features_by_source as _registry_get_features_by_source
from .feature_registry import get_features_by_type as _registry_get_features_by_type
from .feature_registry import list_available_features
from .feature_definitions.absolute_pitch import (
    pitch_range,
    ambitus,
    pitch_standard_deviation,
    pitch_variability,
    first_pitch,
    last_pitch,
    basic_pitch_histogram,
    melodic_pitch_variety,
    mean_pitch,
    most_common_pitch,
    number_of_unique_pitches,
    number_of_common_pitches,
    tessitura,
    mean_tessitura,
    tessitura_std,
    prevalence_of_most_common_pitch,
    relative_prevalence_of_top_pitches,
    interval_between_most_prevalent_pitches,
    pitch_skewness,
    pitch_kurtosis,
    importance_of_bass_register,
    importance_of_middle_register,
    importance_of_high_register,
    pitch_spelling,
    repeated_notes,
    stepwise_motion,
)
from .feature_definitions.pitch_class import (
    _pcdist1_vector,
    _consecutive_fifths,
    pitch_class_variability,
    pitch_class_variability_after_folding,
    pcdist1,
    first_pitch_class,
    last_pitch_class,
    dominant_spread,
    mean_pitch_class,
    most_common_pitch_class,
    number_of_unique_pitch_classes,
    number_of_common_pitch_classes,
    number_of_common_pitches_classes,
    prevalence_of_most_common_pitch_class,
    relative_prevalence_of_top_pitch_classes,
    interval_between_most_prevalent_pitch_classes,
    folded_fifths_pitch_class_histogram,
    pitch_class_skewness,
    pitch_class_kurtosis,
    pitch_class_skewness_after_folding,
    pitch_class_kurtosis_after_folding,
    strong_tonal_centres,
)
from .feature_definitions.pitch_interval import (
    _ivdist1_vector,
    pitch_interval,
    absolute_interval_range,
    mean_absolute_interval,
    mean_melodic_interval,
    standard_deviation_absolute_interval,
    modal_interval,
    most_common_interval,
    ivdist1,
    ivdirdist1,
    ivsizedist1,
    interval_direction,
    interval_direction_mean,
    interval_direction_std,
    average_length_of_melodic_arcs,
    average_interval_span_by_melodic_arcs,
    distance_between_most_prevalent_melodic_intervals,
    melodic_interval_histogram,
    melodic_large_intervals,
    variable_melodic_intervals,
    melodic_thirds,
    melodic_perfect_fourths,
    melodic_tritones,
    melodic_perfect_fifths,
    melodic_sixths,
    melodic_sevenths,
    melodic_octaves,
    minor_major_third_ratio,
    direction_of_melodic_motion,
    number_of_common_melodic_intervals,
    prevalence_of_most_common_melodic_interval,
    relative_prevalence_of_most_common_melodic_intervals,
    amount_of_arpeggiation,
    chromatic_motion,
)
from .feature_definitions.expectation import (
    _get_key_distances,
    get_narmour_features,
    _stability_distance,
    _get_simonton_transition_matrix,
    narmour_registral_direction,
    narmour_proximity,
    narmour_closure,
    narmour_registral_return,
    narmour_intervallic_difference,
    melodic_embellishment,
    mobility,
    mean_mobility,
    mobility_std,
    melodic_attraction,
    mean_melodic_attraction,
    melodic_attraction_std,
    melodic_accent,
    mean_melodic_accent,
    melodic_accent_std,
    compltrans,
    pitch_stm_mean_information_content,
    pitch_ltm_mean_information_content,
    rhythm_stm_mean_information_content,
    rhythm_ltm_mean_information_content,
)
from .feature_definitions.metre import (
    _meter_accent_mean,
    metric_hierarchy,
    meter_accent,
    meter_numerator,
    meter_denominator,
    proportion_of_time_in_first_meter,
    number_of_unique_time_signatures,
    syncopation,
    syncopicity,
)
from .feature_definitions.corpus import (
    _fantastic_melody_tokens,
    _fantastic_melody_tf_df,
    _fantastic_log_normalized_tf_df,
    _fantastic_melody_ngram_counts,
    _fantastic_min_tie_ranks,
    _compute_corpus_feature_bundle,
    _setup_corpus_statistics,
    get_ngram_document_frequency,
    InverseEntropyWeighting,
    tfdf_spearman,
    tfdf_kendall,
    mean_log_tfdf,
    norm_log_dist,
    max_log_df,
    min_log_df,
    mean_log_df,
    mean_global_local_weight,
    std_global_local_weight,
    mean_global_weight,
    std_global_weight,
    get_corpus_features,
)
from .feature_utils import _get_durations
from .utils.warnings import suppress_common_melody_warnings
from .feature_definitions.contour import (
    get_step_contour_features,
    get_interpolation_contour_features,
    comb_contour_matrix,
    get_comb_contour_matrix,
    get_polynomial_contour_features,
    get_huron_contour_features,
    get_contour_features,
)
from .feature_definitions.tonality import (
    _normalize_key_root,
    _canonical_key_string,
    _resolve_key_for_melody,
    _tonality_correlations_for_key,
    infer_key_from_pitches,
    key,
    keyname,
    tonalness,
    tonal_clarity,
    tonal_spike,
    referent,
    tonal_tension,
    mean_cloud_diameter,
    std_cloud_diameter,
    mean_cloud_momentum,
    std_cloud_momentum,
    mean_tensile_strain,
    std_tensile_strain,
    tonalness_histogram,
    inscale,
    proportion_inscale,
    longest_monotonic_conjunct_scalar_passage,
    longest_conjunct_scalar_passage,
    proportion_conjunct_scalar,
    proportion_scalar,
    mode,
    get_tonality_features,
)
from .feature_definitions.timing import (
    _durdist1_vector,
    _get_tempo,
    _rhythmic_run_lengths,
    _rhythmic_value_offsets,
    _silent_run_lengths_qn,
    _calculate_thresholded_peak_table,
    _get_beat_histogram_values_from_ticks,
    _compute_beat_histogram_tables,
    _count_strong_pulses,
    _is_factor_or_multiple,
    durdist1,
    initial_tempo,
    mean_tempo,
    tempo_variability,
    duration_range,
    mean_duration,
    average_note_duration,
    duration_standard_deviation,
    variability_of_note_durations,
    modal_duration,
    length,
    total_number_of_notes,
    number_of_unique_durations,
    global_duration,
    duration_in_seconds,
    note_density,
    note_density_variability,
    note_density_per_quarter_note,
    note_density_per_quarter_note_variability,
    duration_histogram,
    range_of_rhythmic_values,
    number_of_different_rhythmic_values_present,
    number_of_common_rhythmic_values_present,
    prevalence_of_very_short_rhythmic_values,
    prevalence_of_short_rhythmic_values,
    prevalence_of_medium_rhythmic_values,
    prevalence_of_long_rhythmic_values,
    prevalence_of_very_long_rhythmic_values,
    prevalence_of_dotted_notes,
    shortest_rhythmic_value,
    longest_rhythmic_value,
    mean_rhythmic_value,
    most_common_rhythmic_value,
    prevalence_of_most_common_rhythmic_value,
    relative_prevalence_of_most_common_rhythmic_values,
    difference_between_most_common_rhythmic_values,
    mean_rhythmic_value_run_length,
    median_rhythmic_value_run_length,
    variability_in_rhythmic_value_run_lengths,
    mean_rhythmic_value_offset,
    median_rhythmic_value_offset,
    variability_of_rhythmic_value_offsets,
    complete_rests_fraction,
    longest_complete_rest,
    mean_complete_rest_duration,
    median_complete_rest_duration,
    variability_of_complete_rest_durations,
    strongest_rhythmic_pulse,
    strongest_rhythmic_pulse_tempo_standardized,
    second_strongest_rhythmic_pulse,
    second_strongest_rhythmic_pulse_tempo_standardized,
    harmonicity_of_two_strongest_rhythmic_pulses,
    harmonicity_of_two_strongest_rhythmic_pulses_tempo_standardized,
    strength_of_strongest_rhythmic_pulse,
    strength_of_strongest_rhythmic_pulse_tempo_standardized,
    strength_of_second_strongest_rhythmic_pulse,
    strength_of_second_strongest_rhythmic_pulse_tempo_standardized,
    strength_ratio_of_two_strongest_rhythmic_pulses,
    strength_ratio_of_two_strongest_rhythmic_pulses_tempo_standardized,
    combined_strength_of_two_strongest_rhythmic_pulses,
    combined_strength_of_two_strongest_rhythmic_pulses_tempo_standardized,
    rhythmic_variability,
    rhythmic_variability_tempo_standardized,
    rhythmic_looseness,
    rhythmic_looseness_tempo_standardized,
    polyrhythms,
    polyrhythms_tempo_standardized,
    number_of_strong_rhythmic_pulses,
    number_of_strong_rhythmic_pulses_tempo_standardized,
    number_of_moderate_rhythmic_pulses,
    number_of_moderate_rhythmic_pulses_tempo_standardized,
    number_of_relatively_strong_rhythmic_pulses,
    number_of_relatively_strong_rhythmic_pulses_tempo_standardized,
    minimum_note_duration,
    maximum_note_duration,
    equal_duration_transitions,
    half_duration_transitions,
    dotted_duration_transitions,
    amount_of_staccato,
    short_note_fraction,
    npvi,
    onset_autocorrelation,
    onset_autocorr_peak,
)
from .feature_definitions.inter_onset_interval import (
    ioi,
    ioi_mean,
    average_time_between_attacks,
    ioi_standard_deviation,
    variability_of_time_between_attacks,
    ioi_ratio,
    ioi_ratio_mean,
    ioi_ratio_standard_deviation,
    ioi_range,
    ioi_contour,
    ioi_contour_mean,
    ioi_contour_standard_deviation,
    ioi_histogram,
)
from .feature_definitions.complexity import (
    _KK_MAJ_PROFILE,
    _KK_MIN_PROFILE,
    _kkcc_from_pcd,
    _keymode_from_pcd,
    _tonality_midi_toolbox,
    _notedensity_seconds,
    _complebm,
    pitch_entropy,
    interval_entropy,
    duration_entropy,
    duration_accent,
    mean_duration_accent,
    duration_accent_std,
    tonal_entropy,
    gradus,
    complebm_pitch,
    complebm_rhythm,
    complebm_optimal,
    bisect_unbalance,
    center_mass_offset,
    event_heterogeneity,
    av_abs_interval,
    mel_abruptness,
    dur_abruptness,
    rhythm_abruptness,
    asym_total,
    asym_index,
    event_density,
    av_local_p1_entropy,
    p1_entropy,
    p2_entropy,
    p3_entropy,
    i1_entropy,
    i2_entropy,
    d1_entropy,
    d2_entropy,
    d3_entropy,
    wp_entropy,
)
from .feature_definitions.lexical_diversity import (
    get_mtype_features,
    get_lexical_diversity_features,
)
from .idyom.config import (
    VALID_VIEWPOINTS,
    IDyOMConfig,
    _DEFAULT_CORPUS,
    _DEFAULT_IDYOM_CONFIGS,
    _default_idyom_configs,
    _IDYOM_MEAN_INFORMATION_CONTENT_EXPORTS,
    _resolve_idyom_corpus,
    _validate_viewpoints,
)
from .idyom.interface import run_idyom
from .idyom.runners import (
    _cleanup_idyom_temp_output,
    _idyom_mean_information_content,
    _melody_idyom_input_directory,
    _run_idyom_analysis,
    create_temp_midi_with_key_signature,
    get_idyom_results,
    to_mido_key_string,
)
from .pipeline.config import (
    Config,
    DEFAULT_MAX_NGRAM_ORDER,
    FantasticConfig,
    _setup_default_config,
    _validate_config,
)
from .pipeline.loading import FeatureInput, _load_melody_data
from .pipeline.output import (
    _get_category_display_name,
    log_timing_statistics,
    rename_feature_columns,
)
from .pipeline.processing import (
    _process_melodies_parallel,
    _setup_parallel_processing,
    process_melody,
)
from .pipeline.timing import TIMING_STAT_CATEGORIES, _init_timing_stats
from .utils.logging import _setup_logger
from .utils.validation import _check_is_monophonic

suppress_common_melody_warnings()

import csv
import inspect
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import glob
import logging
import time
from random import choices
from typing import Dict, List, Optional, Tuple, Union, Literal, Any

import mido
import numpy as np
import pandas as pd
import scipy
from functools import lru_cache
from natsort import natsorted
from tqdm import tqdm

from melody_features.algorithms import (
    arpeggiation_proportion,
    chromatic_motion_proportion,
    circle_of_fifths,
    compute_tonality_vector,
    get_duration_ratios,
    melodic_embellishment_proportion,
    n_percent_significant_values,
    rank_values,
    repeated_notes_proportion,
    stepwise_motion_proportion,
)
from melody_features.corpus import load_corpus_stats, make_corpus_stats
from melody_features.utils.distributional import (
    distribution_proportions,
    histogram_bins,
    kurtosis,
    skew,
)
from melody_features.io.midi import import_midi
from melody_features.melody_tokenizer import FantasticTokenizer
from melody_features.algorithms.narmour import (
    closure,
    intervallic_difference,
    proximity,
    registral_direction,
    registral_return,
)
from melody_features.ngram_counter import NGramCounter
from melody_features.core.representations import Melody
from melody_features.feature_histogram import (
    PitchHistogram,
    PitchClassHistogram,
    DurationHistogram,
    RhythmicValueHistogram,
    create_rhythmic_value_histogram,
    create_beat_histogram,
    create_melodic_interval_histogram,
)
from melody_features.utils.stats import (
    distribution_entropy,
    get_mode,
    midi_toolbox_entropy,
    range_func,
    shannon_entropy,
    standard_deviation,
)
from melody_features.algorithms.meter_estimation import (
    compute_onset_autocorrelation,
    duration_accent as _duration_accent,
    melodic_accent as _melodic_accent,
    metric_hierarchy as _metric_hierarchy,
)
from melody_features.algorithms.pitch_spelling import (
    estimate_spelling_from_melody as _estimate_spelling_from_melody,
)
from melody_features.algorithms.tonal_tension import (
    estimate_tonaltension,
    SCALE_FACTOR,
    DEFAULT_WEIGHTS,
    ALPHA,
    BETA
)

def _get_features_by_type(feature_type: str) -> dict:
    """Get all features of a specific type.

    Parameters
    ----------
    feature_type : str
        The type of features to collect (e.g., 'absolute', 'contour', 'tonality', etc.)

    Returns
    -------
    dict
        Dictionary mapping feature names to functions
    """
    current_module = sys.modules[__name__]
    return _registry_get_features_by_type(current_module, feature_type)

def _get_features_by_domain(domain: str) -> dict:
    """Get all features of a specific domain.

    Parameters
    ----------
    domain : str
        The domain of features to collect ('pitch', 'rhythm', or 'both')

    Returns
    -------
    dict
        Dictionary mapping feature names to functions
    """
    current_module = sys.modules[__name__]
    return _registry_get_features_by_domain(current_module, domain)

def _get_features_by_domain_and_types(domain: str, allowed_types: list[str]) -> dict:
    """Get all features of a specific domain that match any of the allowed types.

    Parameters
    ----------
    domain : str
        The domain of features to collect ('pitch', 'rhythm', or 'both')
    allowed_types : list[str]
        List of allowed feature types (e.g., ['absolute', 'interval'])

    Returns
    -------
    dict
        Dictionary mapping feature names to functions
    """
    current_module = sys.modules[__name__]
    return _registry_get_features_by_domain_and_types(
        current_module,
        domain,
        allowed_types,
    )

def _invoke_feature(func, melody: Melody, **extra):
    """Call a feature function, binding ``melody`` fields and extras by parameter name."""
    return _dispatch_invoke_feature(
        func,
        melody,
        default_max_ngram_order=DEFAULT_MAX_NGRAM_ORDER,
        **extra,
    )

def _collect_feature_values(
    feature_functions: Dict[str, callable],
    melody: Melody,
    *,
    tuple_suffix: Optional[str] = None,
    **extra,
) -> Dict[str, Any]:
    """Compute feature functions with shared melody argument dispatch."""
    return _dispatch_collect_feature_values(
        feature_functions,
        melody,
        default_max_ngram_order=DEFAULT_MAX_NGRAM_ORDER,
        tuple_suffix=tuple_suffix,
        **extra,
    )

def get_pitch_features(melody: Melody) -> Dict:
    """Dynamically collect all pitch features for a melody.

    Collects features decorated with @pitch domain and @absolute type.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    Dict
        Dictionary of pitch feature values
    """
    pitch_functions = _get_features_by_domain_and_types("pitch", ["absolute"])
    return _collect_feature_values(pitch_functions, melody)

def get_pitch_class_features(melody: Melody) -> Dict:
    """Dynamically collect all pitch class features for a melody.

    Collects features decorated with @pitch domain and @pitch_class type.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    Dict
        Dictionary of pitch class feature values
    """
    pitch_class_functions = _get_features_by_domain_and_types("pitch", ["pitch_class"])
    return _collect_feature_values(pitch_class_functions, melody)

# I still think this is cool and I like it a lot, but it's not included in any of the software
# since this feature set is the result of a systematic review of toolboxes, we can't return it right now
# but it's here and it works
@novel
def temperley_likelihood(pitches: list[int]) -> float:
    """
    The likelihood of a melody using Bayesian reasoning,
    according to David Temperley's model
    (http://davidtemperley.com/wp-content/uploads/2015/11/temperley-cs08.pdf).

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Likelihood of the melody using Bayesian reasoning
    """
    # represent all possible notes as int
    notes_ints = np.arange(0, 120, 1)

    # Calculate central pitch profile
    central_pitch_profile = scipy.stats.norm.pdf(notes_ints, loc=68, scale=np.sqrt(5.0))
    central_pitch = choices(notes_ints, central_pitch_profile)
    range_profile = scipy.stats.norm.pdf(
        notes_ints, loc=central_pitch, scale=np.sqrt(23.0)
    )

    # Get key probabilities
    rpk_major = [
        0.184,
        0.001,
        0.155,
        0.003,
        0.191,
        0.109,
        0.005,
        0.214,
        0.001,
        0.078,
        0.004,
        0.055,
    ] * 10
    rpk_minor = [
        0.192,
        0.005,
        0.149,
        0.179,
        0.002,
        0.144,
        0.002,
        0.201,
        0.038,
        0.012,
        0.053,
        0.022,
    ] * 10

    # Calculate total probability
    total_prob = 1.0
    for i in range(1, len(pitches)):
        # Calculate proximity profile centered on previous note
        prox_profile = scipy.stats.norm.pdf(
            notes_ints, loc=pitches[i - 1], scale=np.sqrt(10)
        )
        rp = range_profile * prox_profile

        # Apply key profile based on major/minor
        if "major" in compute_tonality_vector([p % 12 for p in pitches])[0][0]:
            rpk = rp * rpk_major
        else:
            rpk = rp * rpk_minor

        # Normalize probabilities
        rpk_normed = rpk / np.sum(rpk)

        # Get probability of current note
        note_prob = rpk_normed[pitches[i]]
        total_prob *= note_prob

    return total_prob

def get_complexity_features(
    melody: Melody, phrase_gap: float = 1.5, max_ngram_order: int = DEFAULT_MAX_NGRAM_ORDER
) -> Dict:
    """Dynamically collect all complexity features for a melody."""
    complexity_functions = _get_features_by_type(FeatureType.COMPLEXITY)
    skip = frozenset({"InverseEntropyWeighting", "get_mtype_features"})
    filtered = {
        name: func for name, func in complexity_functions.items() if name not in skip
    }
    return _collect_feature_values(
        filtered,
        melody,
        phrase_gap=phrase_gap,
        max_ngram_order=max_ngram_order,
    )

def get_complexity_feature_bundle(
    melody: Melody,
    phrase_gap: float = 1.5,
    max_ngram_order: int = DEFAULT_MAX_NGRAM_ORDER,
) -> Dict:
    """Return complexity and lexical-diversity features for export."""
    return {
        **get_complexity_features(
            melody, phrase_gap=phrase_gap, max_ngram_order=max_ngram_order
        ),
        **get_lexical_diversity_features(
            melody, phrase_gap=phrase_gap, max_ngram_order=max_ngram_order
        ),
    }

def get_interval_features(melody: Melody) -> Dict:
    """Dynamically collect all interval features for a melody.
    Collects features decorated with @pitch domain and @interval type.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    Dict
        Dictionary of interval feature values
    """
    interval_functions = _get_features_by_domain_and_types("pitch", ["interval"])
    return _collect_feature_values(interval_functions, melody, tuple_suffix="sd")

def get_metric_accent_features(melody: Melody) -> Dict:
    """Compute metric hierarchy and meter accent features for a melody.

    Based on MIDI toolbox metric hierarchy and meteraccent analysis.
    Calculates the strength of each note position within the known or estimated meter,
    and computes phenomenal accent synchrony.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    Dict
        Dictionary containing:
        - metric_hierarchy: List of hierarchy values for each note
        - meter_accent: Phenomenal accent synchrony measure (from MIDI toolbox meteraccent.m)
    """
    return {
        "metric_hierarchy": metric_hierarchy(melody),
        "meter_accent": meter_accent(melody),
    }

def _collect_rhythm_domain_features(melody: Melody, allowed_types: list[str]) -> Dict:
    """Collect @rhythm-domain features whose types intersect ``allowed_types``."""
    rhythm_functions = _get_features_by_domain_and_types("rhythm", allowed_types)
    return _collect_feature_values(rhythm_functions, melody, tuple_suffix="std")

def get_timing_features(melody: Melody) -> Dict:
    """Collect @rhythm-domain features decorated with @timing."""
    return _collect_rhythm_domain_features(melody, ["timing"])

def get_inter_onset_interval_features(melody: Melody) -> Dict:
    """Collect @rhythm-domain features decorated with @interval (IOI family)."""
    return _collect_rhythm_domain_features(melody, ["interval"])

def get_rhythm_features(melody: Melody) -> Dict:
    """Dynamically collect all rhythm features for a melody.

    Combines timing, inter-onset interval, and metric-accent features for
    backward-compatible ``rhythm_features`` export.
    """
    features: Dict[str, Any] = {}
    features.update(get_timing_features(melody))
    features.update(get_inter_onset_interval_features(melody))
    features.update(get_metric_accent_features(melody))
    return features


def collect_rhythm_for_pipeline(melody: Melody) -> tuple[Dict[str, Any], Dict[str, float]]:
    """Collect rhythm features and per-subcategory timings for pipeline workers.

    ``timing`` and ``inter_onset_interval`` are timed separately so pipeline
    statistics keep them as discrete taxonomy categories.
    """
    rhythm_timings: Dict[str, float] = {}

    start = time.time()
    timing_features = get_timing_features(melody)
    rhythm_timings["timing"] = time.time() - start

    start = time.time()
    ioi_features = get_inter_onset_interval_features(melody)
    rhythm_timings["inter_onset_interval"] = time.time() - start

    metric_accent_features = get_metric_accent_features(melody)
    rhythm_features = {
        **timing_features,
        **ioi_features,
        **metric_accent_features,
    }
    return rhythm_features, rhythm_timings

def get_expectation_features(melody: Melody) -> Dict:
    """Dynamically collect all expectation features for a melody."""
    expectation_functions = _get_features_by_type(FeatureType.EXPECTATION)
    filtered = {
        name: func
        for name, func in expectation_functions.items()
        if name not in _IDYOM_MEAN_INFORMATION_CONTENT_EXPORTS
    }
    return _collect_feature_values(
        filtered, melody, tuple_suffix="std", numeric_tuple_only=True
    )

def get_metre_features(melody: Melody) -> Dict:
    """Dynamically collect all metre features for a melody."""
    metre_functions = _get_features_by_type(FeatureType.METRE)
    return _collect_feature_values(metre_functions, melody)


def _get_features_by_source(source: str) -> Dict[str, callable]:
    """Get all functions/classes decorated with a specific source."""
    return _registry_get_features_by_source(sys.modules[__name__], source)

def get_fantastic_features(
    melody: Melody,
    corpus_stats: Optional[dict] = None,
    phrase_gap: float = 1.5,
    max_ngram_order: int = DEFAULT_MAX_NGRAM_ORDER,
) -> Dict:
    """Get all FANTASTIC features for a melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : Optional[dict], optional
        Corpus statistics for distributional features (default: None)
    phrase_gap : float, optional
        Gap threshold for phrase segmentation (default: 1.5)
    max_ngram_order : int, optional
        Maximum inclusive n-gram length (default: 5)

    Returns
    -------
    Dict
        Dictionary containing all FANTASTIC features
    """
    return _compute_features_by_source(
        melody,
        "fantastic",
        corpus_stats=corpus_stats,
        phrase_gap=phrase_gap,
        max_ngram_order=max_ngram_order
    )

def get_jsymbolic_features(melody: Melody) -> Dict:
    """Get all jSymbolic features for a melody.

    Parameters
    ----------
    melody : Melody
        The melody to extract features from

    Returns
    -------
    Dict
        Dictionary containing all jSymbolic features
    """
    return _compute_features_by_source(melody, "jsymbolic")

def get_must_features(melody: Melody) -> Dict:
    """Get all MUST features for a melody.

    Parameters
    ----------
    melody : Melody
        The melody to extract features from

    Returns
    -------
    Dict
        Dictionary containing all MUST features
    """
    return _compute_features_by_source(melody, "must")

def get_midi_toolbox_features(melody: Melody) -> Dict:
    """Get all MIDI Toolbox features for a melody.

    Parameters
    ----------
    melody : Melody
        The melody to extract features from

    Returns
    -------
    Dict
        Dictionary containing all MIDI Toolbox features
    """
    return _compute_features_by_source(melody, "midi_toolbox")

def get_idyom_features(melody: Melody) -> Dict:
    """Get all IDyOM features for a melody.

    Parameters
    ----------
    melody : Melody
        The melody to extract features from

    Returns
    -------
    Dict
        Dictionary containing all IDyOM features
    """
    return _compute_features_by_source(melody, "idyom")

def get_simile_features(melody: Melody) -> Dict:
    """Get all SIMILE features for a melody.

    Parameters
    ----------
    melody : Melody
        The melody to extract features from

    Returns
    -------
    Dict
        Dictionary containing all SIMILE features
    """
    return _compute_features_by_source(melody, "simile")

def get_novel_features(melody: Melody) -> Dict:
    """Get all novel/custom features for a melody.

    Parameters
    ----------
    melody : Melody
        The melody to extract features from

    Returns
    -------
    Dict
        Dictionary containing all novel features
    """
    return _compute_features_by_source(melody, "custom")

def _compute_features_by_source(
    melody: Melody,
    source: str,
    corpus_stats: Optional[dict] = None,
    phrase_gap: float = 1.5,
    max_ngram_order: int = DEFAULT_MAX_NGRAM_ORDER,
) -> Dict:
    """Compute all features for a melody that are decorated with a specific source.

    Parameters
    ----------
    melody : Melody
        The melody to extract features from
    source : str
        The source label to filter by
    corpus_stats : Optional[dict], optional
        Corpus statistics for FANTASTIC features (default: None)
    phrase_gap : float, optional
        Gap threshold for phrase segmentation (default: 1.5)
    max_ngram_order : int, optional
        Maximum n-gram order for FANTASTIC features (default: 6)

    Returns
    -------
    Dict
        Dictionary containing all features from the specified source
    """
    import inspect

    source_features = _get_features_by_source(source)
    computed_features = {}

    for name, func in source_features.items():
        try:
            result = _invoke_feature(
                func,
                melody,
                corpus_stats=corpus_stats,
                phrase_gap=phrase_gap,
                max_ngram_order=max_ngram_order,
            )

            if hasattr(result, '__dict__') and not isinstance(result, (str, int, float, list, dict)):
                computed_features.update(result.__dict__)
            else:
                computed_features[name] = result

        except Exception as e:
            logger = logging.getLogger("melody_features")
            logger.warning(f"Could not compute {name}: {e}")
            continue

    return computed_features

def get_all_features(
    input: FeatureInput,
    config: Optional[Config] = None,
    log_level: int = logging.INFO,
    skip_idyom: bool = False,
) -> "pd.DataFrame":
    """Calculate a multitude of features from across the computational melody analysis field.
    This function returns a pandas DataFrame with a row for every melody in the supplied input.

    The input can be:
    - A directory path containing MIDI files
    - A list of MIDI file paths
    - A single MIDI file path
    - A list of :class:`~melody_features.core.representations.Melody` objects

    If a path to a corpus of MIDI files is provided in the Config,
    corpus statistics will be computed following FANTASTIC's n-gram document frequency
    model (Müllensiefen, 2009). If not, this will be skipped.
    This function will also run IDyOM (Pearce, 2005) on the input MIDI files.
    If a corpus of MIDI files is provided in the Config, IDyOM will be run with
    pretraining on the corpus. If not, it will be run without pretraining.

    Parameters
    ----------
    input : FeatureInput
        Path to input MIDI directory, list of MIDI file paths, single MIDI file
        path, or list of in-memory :class:`~melody_features.core.representations.Melody`
        objects
    config : Config
        Configuration object containing corpus path, IDyOM configurations (as a dict), and FANTASTIC configuration.
        If idyom.corpus or fantastic.corpus is set, those take precedence over config.corpus for their respective methods.
        If multiple IDyOM configs are provided, IDyOM will run for each config and features for each
        will be included with an identifier in the output.
    log_level : int
        Logging level (default: logging.INFO)
    skip_idyom : bool
        If True, skip IDyOM feature calculation (default: False)

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with a row for every melody in the input, containing all extracted features.
        You can save this to CSV using df.to_csv('filename.csv') if needed.

    """
    suppress_common_melody_warnings()

    # Set up logger
    logger = _setup_logger(log_level)

    # Clean up any existing IDyOM temporary output directory
    _cleanup_idyom_temp_output()

    config = _setup_default_config(config)
    _validate_config(config)

    logger.info("Starting feature extraction job...")

    # Log configuration parameters
    logger.info("Configuration Parameters:")
    logger.info(f"  Key Estimation Strategy: {config.key_estimation}")
    logger.info(f"  Key Finding Algorithm: {config.key_finding_algorithm}")
    logger.info(f"  Corpus Path: {config.corpus if config.corpus else 'None (corpus features disabled)'}")

    logger.info(f"  IDyOM Configurations: {len(config.idyom)} config(s)")
    for idyom_name, idyom_cfg in config.idyom.items():
        logger.info(f"    [{idyom_name}]:")
        logger.info(f"      Models: {idyom_cfg.models}")
        logger.info(f"      Corpus: {idyom_cfg.corpus if idyom_cfg.corpus else 'Using Corpus Path from Config'}")
        logger.info(f"      Target Viewpoints: {idyom_cfg.target_viewpoints}")
        logger.info(f"      Source Viewpoints: {idyom_cfg.source_viewpoints}")
        logger.info(f"      PPM Order: {idyom_cfg.ppm_order}")

    logger.info(f"  FANTASTIC Configuration:")
    logger.info(f"    Max N-gram Order: {config.fantastic.max_ngram_order}")
    logger.info(f"    Corpus: {config.fantastic.corpus if config.fantastic.corpus else 'Using Corpus Path from Config'}")

    # Keep internal corpus-stat caches out of the caller's working tree.
    corpus_cache_dir = Path(tempfile.gettempdir()) / "melody_features" / "corpus_stats"
    corpus_cache_dir.mkdir(parents=True, exist_ok=True)
    temp_output_file = str(corpus_cache_dir / "temp_corpus_stats.csv")
    corpus_stats = _setup_corpus_statistics(config, temp_output_file)

    melody_data_list = _load_melody_data(input)

    if not melody_data_list:
        logger.warning("No valid monophonic melodies found to process.")
        return

    if skip_idyom:
        logger.info("Skipping IDyOM analysis...")
        idyom_results_dict = {}
    else:
        # Add retry logic for IDyOM to handle database locking issues
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                idyom_results_dict = _run_idyom_analysis(input, config)
                break
            except Exception as e:
                if "database is locked" in str(e).lower() or "sqlite" in str(e).lower():
                    if attempt < max_retries - 1:
                        logger.warning(f"IDyOM database locked (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.error(f"IDyOM failed after {max_retries} attempts due to database locking. Skipping IDyOM analysis.")
                        idyom_results_dict = {}
                else:
                    raise

    start_time = time.time()

    headers, melody_args, timing_stats = _setup_parallel_processing(
        melody_data_list, corpus_stats, idyom_results_dict, config
    )

    all_features = _process_melodies_parallel(
        melody_args,
        headers,
        melody_data_list,
        idyom_results_dict,
        timing_stats,
    )

    if not all_features:
        logger.warning("No features were successfully extracted from any melodies")
        return pd.DataFrame()

    # Create DataFrame from results

    # Sort results by melody_id
    all_features.sort(key=lambda x: x[0])

    # Create DataFrame
    df = pd.DataFrame(all_features, columns=headers)

    # Rename columns to use display names
    df = rename_feature_columns(df)

    # Log timing statistics
    end_time = time.time()
    log_timing_statistics(logger, timing_stats, end_time - start_time)

    logger.info(f"Successfully extracted features for {len(df)} melodies")

    return df
