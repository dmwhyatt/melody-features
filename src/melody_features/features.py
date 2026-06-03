"""
This module contains functions to compute features from melodies.
Features are the product of an input list and at least one algorithm.
"""

__author__ = "David Whyatt"

import warnings
from importlib import resources

from .feature_decorators import (
    fantastic, idyom, midi_toolbox, melsim, jsymbolic, novel, simile, partitura,
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
from .absolute_pitch_features import (
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
from .pitch_class_features import (
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
from .pitch_interval_features import (
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
from .expectation_features import (
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
from .metre_features import (
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
from .corpus_features import (
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
from .contour_features import (
    get_step_contour_features,
    get_interpolation_contour_features,
    comb_contour_matrix,
    get_comb_contour_matrix,
    get_polynomial_contour_features,
    get_huron_contour_features,
    get_contour_features,
)
from .tonality_features import (
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
from .timing_features import (
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
    _is_beat_histogram_function,
    _precompute_beat_histogram_data,
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
from .inter_onset_interval_features import (
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
from .complexity_features import (
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
)
from .lexical_diversity_features import (
    get_mtype_features,
    get_lexical_diversity_features,
)

warnings.filterwarnings("ignore", category=UserWarning, module="pretty_midi")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pretty_midi")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*pkg_resources is deprecated.*"
)

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
from melody_features.distributional import (
    distribution_proportions,
    histogram_bins,
    kurtosis,
    skew,
)
from melody_features.huron_contour import HuronContour
from melody_features.idyom_interface import run_idyom
from melody_features.import_mid import import_midi
from melody_features.interpolation_contour import InterpolationContour
from melody_features.melody_tokenizer import FantasticTokenizer
from melody_features.narmour import (
    closure,
    intervallic_difference,
    proximity,
    registral_direction,
    registral_return,
)
from melody_features.ngram_counter import NGramCounter
from melody_features.polynomial_contour import PolynomialContour
from melody_features.representations import Melody
from melody_features.feature_histogram import (
    PitchHistogram,
    PitchClassHistogram,
    DurationHistogram,
    RhythmicValueHistogram,
    create_rhythmic_value_histogram,
    create_beat_histogram,
    create_melodic_interval_histogram,
)
from melody_features.stats import (
    distribution_entropy,
    get_mode,
    midi_toolbox_entropy,
    range_func,
    shannon_entropy,
    standard_deviation,
)
from melody_features.step_contour import StepContour
from melody_features.meter_estimation import (
    compute_onset_autocorrelation,
    duration_accent as _duration_accent,
    melodic_accent as _melodic_accent,
    metric_hierarchy as _metric_hierarchy,
)
from melody_features.pitch_spelling import (
    estimate_spelling_from_melody as _estimate_spelling_from_melody,
)
from melody_features.tonal_tension import (
    estimate_tonaltension,
    SCALE_FACTOR,
    DEFAULT_WEIGHTS,
    ALPHA,
    BETA
)

VALID_VIEWPOINTS = {
    "onset",
    "cpitch",
    "dur",
    "keysig",
    "mode",
    "tempo",
    "pulses",
    "barlength",
    "deltast",
    "bioi",
    "phrase",
    "mpitch",
    "accidental",
    "dyn",
    "voice",
    "ornament",
    "comma",
    "articulation",
    "ioi",
    "posinbar",
    "dur-ratio",
    "referent",
    "cpint",
    "contour",
    "cpitch-class",
    "cpcint",
    "cpintfref",
    "cpintfip",
    "cpintfiph",
    "cpintfib",
    "inscale",
    "ioi-ratio",
    "ioi-contour",
    "metaccent",
    "bioi-ratio",
    "bioi-contour",
    "lphrase",
    "cpint-size",
    "newcontour",
    "cpcint-size",
    "cpcint-2",
    "cpcint-3",
    "cpcint-4",
    "cpcint-5",
    "cpcint-6",
    "octave",
    "tessitura",
    "mpitch-class",
    "registral-direction",
    "intervallic-difference",
    "registral-return",
    "proximity",
    "closure",
    "fib",
    "crotchet",
    "tactus",
    "fiph",
    "liph",
    "thr-cpint-fib",
    "thr-cpint-fiph",
    "thr-cpint-liph",
    "thr-cpint-crotchet",
    "thr-cpint-tactus",
    "thr-cpintfref-liph",
    "thr-cpintfref-fib",
    "thr-cpint-cpintfref-liph",
    "thr-cpint-cpintfref-fib",
}


def _setup_logger(level: int = logging.INFO) -> logging.Logger:
    """Set up and configure the logger for the melodic feature set.

    Parameters
    ----------
    level : int
        Logging level (default: logging.INFO)

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger("melody_features")
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger


def _validate_viewpoints(viewpoints: list[str], name: str) -> None:
    """Validate that all viewpoints are valid.

    Parameters
    ----------
    viewpoints : list[str]
        List of viewpoints to validate
    name : str
        Name of the parameter for error messages

    Raises
    ------
    ValueError
        If any viewpoint is invalid
    """
    if not isinstance(viewpoints, list):
        raise ValueError(f"{name} must be a list, got {type(viewpoints)}")

    all_viewpoints = set()
    for viewpoint in viewpoints:
        if isinstance(viewpoint, (list, tuple)):
            if len(viewpoint) < 2:
                raise ValueError(
                    f"Linked viewpoints must have at least 2 elements, got {len(viewpoint)} elements: {viewpoint}"
                )
            all_viewpoints.update(viewpoint)
        else:
            all_viewpoints.add(viewpoint)

    invalid_viewpoints = all_viewpoints - VALID_VIEWPOINTS
    if invalid_viewpoints:
        raise ValueError(
            f"Invalid viewpoint(s) in {name}: {', '.join(invalid_viewpoints)}.\n"
            f"Valid viewpoints are: {', '.join(sorted(list(VALID_VIEWPOINTS)))}"
        )

def _check_is_monophonic(melody: Melody) -> bool:
    """Check if the melody is monophonic.

    This function determines if a melody is monophonic by ensuring that no
    notes overlap in time. It assumes the notes within the Melody object are
    sorted by their start times. A melody is considered polyphonic if any
    note starts before the previous note has ended.
    Parameters
    ----------
    melody : Melody
        The melody to analyze as a Melody object.

    Returns
    -------
    bool
        True if the melody is monophonic, False otherwise.
    """
    starts = melody.starts
    ends = melody.ends

    # A melody with 0 or 1 notes can only be monophonic.
    if len(starts) < 2:
        return True

    # otherwise, if start time of current note is less than end time of previous note,
    # the melody cannot be monophonic.
    for i in range(1, len(starts)):
        if starts[i] < ends[i - 1]:
            return False

    return True



# Setup config classes for the different feature sets
@dataclass
class IDyOMConfig:
    """Configuration class for IDyOM analysis.
    Parameters
    ----------
    target_viewpoints : list[str]
        List of target viewpoints to use for IDyOM analysis.
    source_viewpoints : list[str]
        List of source viewpoints to use for IDyOM analysis.
    ppm_order : int
        Order of the PPM model. Set to None for unbounded order.
    models : str
        Models to use for IDyOM analysis. Can be ":stm", ":ltm" or ":both".
    corpus : Optional[os.PathLike]
        Path to the corpus to use for IDyOM analysis. If not provided, the corpus will be the one specified in the Config class.
        This will override the corpus specified in the Config class if both are provided.
        This should be set to None if using :stm model, as the short term model does not use pretraining.
        You can use the bundled corpora (essen_folksong_collection and pearce_default_idyom) or provide a path to a directory containing MIDI files
        for a different corpus.
    """

    target_viewpoints: list[str]
    source_viewpoints: list[str]
    ppm_order: int
    models: str
    corpus: Optional[os.PathLike] = None

    def __post_init__(self):
        """Validate the configuration after initialization."""
        _validate_viewpoints(self.target_viewpoints, "target_viewpoints")
        _validate_viewpoints(self.source_viewpoints, "source_viewpoints")

        valid_models = {":stm", ":ltm", ":both"}
        if not isinstance(self.models, str):
            raise ValueError(f"models must be a string, got {type(self.models)}")
        if self.models not in valid_models:
            raise ValueError(f"models must be one of {valid_models}, got {self.models}")

        if self.corpus is not None:
            if self.models == ":stm":
                raise ValueError(
                    "IDyOM short-term models (:stm) do not use a corpus. "
                    "Set corpus=None for :stm configurations."
                )
            if not isinstance(self.corpus, (str, os.PathLike)):
                raise ValueError(
                    f"corpus must be a string or PathLike, got {type(self.corpus)}"
                )
            if not Path(self.corpus).exists():
                raise ValueError(f"corpus path does not exist: {self.corpus}")


_IDYOM_MEAN_INFORMATION_CONTENT_EXPORTS = frozenset({
    "pitch_stm_mean_information_content",
    "pitch_ltm_mean_information_content",
    "rhythm_stm_mean_information_content",
    "rhythm_ltm_mean_information_content",
})

_DEFAULT_CORPUS = resources.files("melody_features") / "corpora/pearce_default_idyom"


def _resolve_idyom_corpus(
    idyom_config: IDyOMConfig,
    config_corpus: Optional[os.PathLike] = None,
    override: Optional[os.PathLike] = None,
) -> Optional[os.PathLike]:
    """Helper function to resolve the pretraining corpus for an IDyOM configuration.
    Mostly used for individual calls, rather than `get_all_features` which uses the Config class
    to setup all IDyOM configurations at once.

    Resolution order:
    1. Explicit override (used in batch mode)
    2. Per-config corpus (IDyOMConfig.corpus)
    3. Config.corpus (from get_all_features)
    4. Package default (_DEFAULT_CORPUS)

    Short-term models never use a corpus.
    """
    if idyom_config.models == ":stm":
        return None
    if override is not None:
        return override
    if idyom_config.corpus is not None:
        return idyom_config.corpus
    if config_corpus is not None:
        return config_corpus
    return _DEFAULT_CORPUS


def _default_idyom_configs(corpus: Optional[os.PathLike] = None) -> dict[str, IDyOMConfig]:
    """Build the standard four IDyOM configurations used by ``get_all_features``.

    Parameters
    ----------
    corpus : Optional[os.PathLike]
        Pretraining corpus for long-term (LTM) models. For the default
        ``Config`` from ``_setup_default_config``, pass ``_DEFAULT_CORPUS`` so
        LTM entries match ``Config.corpus`` explicitly. Pass ``None`` only when
        LTM configs should inherit ``Config.corpus`` at runtime instead.
    """
    return {
        "pitch_stm": IDyOMConfig(
            target_viewpoints=["cpitch"],
            source_viewpoints=[("cpitch", "cpint", "cpintfref")],
            ppm_order=None,
            models=":stm",
            corpus=None,
        ),
        "pitch_ltm": IDyOMConfig(
            target_viewpoints=["cpitch"],
            source_viewpoints=[("cpitch", "cpint", "cpintfref")],
            ppm_order=None,
            models=":ltm",
            corpus=corpus,
        ),
        "rhythm_stm": IDyOMConfig(
            target_viewpoints=["onset"],
            source_viewpoints=["ioi", "ioi-ratio"],
            ppm_order=None,
            models=":stm",
            corpus=None,
        ),
        "rhythm_ltm": IDyOMConfig(
            target_viewpoints=["onset"],
            source_viewpoints=["ioi", "ioi-ratio"],
            ppm_order=None,
            models=":ltm",
            corpus=corpus,
        ),
    }


_DEFAULT_IDYOM_CONFIGS = _default_idyom_configs(_DEFAULT_CORPUS)

# Inclusive maximum m-type length (FANTASTIC n.limits default: 1–5).
DEFAULT_MAX_NGRAM_ORDER = 5


@dataclass
class FantasticConfig:
    """Configuration class for FANTASTIC analysis.
    Parameters
    ----------
    max_ngram_order : int
        Maximum order of n-grams to use for FANTASTIC analysis.
    phrase_gap : float
        Phrase gap to use for FANTASTIC analysis.
    corpus : Optional[os.PathLike]
        Path to the corpus to use for FANTASTIC analysis. If not provided, the corpus will be the one specified in the Config class.
        This will override the corpus specified in the Config class if both are provided.
        You can use the bundled corpora (essen_folksong_collection and pearce_default_idyom) or provide a path to a directory containing MIDI files
        for a different corpus.
    """

    max_ngram_order: int
    phrase_gap: float
    corpus: Optional[os.PathLike] = None

    def __post_init__(self):
        """Validate the configuration after initialization."""
        if not isinstance(self.max_ngram_order, int):
            raise ValueError(
                f"max_ngram_order must be an integer, got {type(self.max_ngram_order)}"
            )
        if self.max_ngram_order < 1:
            raise ValueError(
                f"max_ngram_order must be at least 1, got {self.max_ngram_order}"
            )

        if not isinstance(self.phrase_gap, (int, float)):
            raise ValueError(
                f"phrase_gap must be a number, got {type(self.phrase_gap)}"
            )
        if self.phrase_gap <= 0:
            raise ValueError(f"phrase_gap must be positive, got {self.phrase_gap}")

        if self.corpus is not None:
            if not isinstance(self.corpus, (str, os.PathLike)):
                raise ValueError(
                    f"corpus must be a string or PathLike, got {type(self.corpus)}"
                )
            if not Path(self.corpus).exists():
                raise ValueError(f"corpus path does not exist: {self.corpus}")


@dataclass
class Config:
    """Configuration class for the feature set.
    Parameters
    ----------
    idyom : dict[str, IDyOMConfig]
        Dictionary of IDyOM configurations, with the key being the name of the IDyOM configuration.
    fantastic : FantasticConfig
        Configuration object for FANTASTIC analysis.
    corpus : Optional[os.PathLike]
        Path to the corpus to use for the feature set. This can be overridden by the corpus parameter in the IDyOMConfig and FantasticConfig classes.
        If None, no corpus-dependent features will be computed unless specified in individual configs.
        You can use the bundled corpora (essen_folksong_collection and pearce_default_idyom) or provide a path to a directory containing MIDI files
        for a different corpus.
    key_estimation: str
        The key estimation method to use. Can be
        `always_read_from_file`, `infer_if_necessary` or `always_infer`:
        - When set to `always_read_from_file`, the key will be read from the MIDI file,
        and if key signature information is not present, an error will be raised.
        - When set to `infer_if_necessary`, the key will be inferred from the melody if key signature information is not present in the file.
        - When set to `always_infer`, the key will be inferred from the melody regardless of whether key signature information is present.
    key_finding_algorithm: str
        The algorithm that will be used to infer the key of the melody, where required. Currently,
        can only be `krumhansl_schmuckler`, and this is the default value.
        Support for additional algorithms may be added in the future.
    """

    idyom: dict[str, IDyOMConfig]
    fantastic: FantasticConfig
    corpus: Optional[os.PathLike] = None
    key_estimation: Literal["always_read_from_file", "infer_if_necessary", "always_infer"] = "infer_if_necessary"
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler"

    def __post_init__(self):
        """Validate the configuration after initialization."""
        if self.corpus is not None:
            if not isinstance(self.corpus, (str, os.PathLike)):
                raise ValueError(
                    f"corpus must be a string or PathLike, got {type(self.corpus)}"
                )
            if not Path(self.corpus).exists():
                raise ValueError(f"corpus path does not exist: {self.corpus}")

        if not isinstance(self.idyom, dict):
            raise ValueError(f"idyom must be a dictionary, got {type(self.idyom)}")
        if not self.idyom:
            raise ValueError("idyom dictionary cannot be empty")

        for name, config in self.idyom.items():
            if not isinstance(name, str):
                raise ValueError(
                    f"idyom dictionary keys must be strings, got {type(name)}"
                )
            if not isinstance(config, IDyOMConfig):
                raise ValueError(
                    f"idyom dictionary values must be IDyOMConfig objects, got {type(config)}"
                )

        if not isinstance(self.fantastic, FantasticConfig):
            raise ValueError(
                f"fantastic must be a FantasticConfig object, got {type(self.fantastic)}"
            )

        if self.key_estimation not in ["always_read_from_file", "infer_if_necessary", "always_infer"]:
            raise ValueError(f"key_estimation must be one of ['always_read_from_file', 'infer_if_necessary', 'always_infer'], got {self.key_estimation}")

        if self.key_finding_algorithm != "krumhansl_schmuckler":
            raise NotImplementedError(
                f"key_finding_algorithm '{self.key_finding_algorithm}' is not supported. "
                f"Currently only 'krumhansl_schmuckler' is implemented. More algorithms may be added in the future."
            )











# refstat('kkmaj') / refstat('kkmin') — used by tonality.m / keymode.m

































# backwards-compatible alias (typo preserved for semantic versioning).


































































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









# Undecorated helper for internal use only





























































































































# readable alias








# Tonality Features


























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









# Melodic Movement Features
































def _melody_idyom_input_directory(melody: Melody) -> tuple[str, list[str]]:
    """Build a one-melody MIDI directory for IDyOM. Returns (directory, paths to remove)."""
    import pretty_midi

    cleanup_paths: list[str] = []
    melody_path = melody.id
    if (
        melody_path
        and str(melody_path).lower().endswith((".mid", ".midi"))
        and os.path.isfile(str(melody_path))
    ):
        temp_dir = tempfile.mkdtemp(prefix="idyom_melody_")
        cleanup_paths.append(temp_dir)
        dest = os.path.join(temp_dir, os.path.basename(str(melody_path)))
        shutil.copy2(str(melody_path), dest)
        return temp_dir, cleanup_paths

    temp_dir = tempfile.mkdtemp(prefix="idyom_melody_")
    cleanup_paths.append(temp_dir)
    temp_midi = os.path.join(temp_dir, "melody.mid")
    pm = pretty_midi.PrettyMIDI(initial_tempo=melody.tempo)
    instrument = pretty_midi.Instrument(program=0)
    for pitch, start, end in zip(melody.pitches, melody.starts, melody.ends):
        instrument.notes.append(
            pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
        )
    pm.instruments.append(instrument)
    pm.write(temp_midi)
    return temp_dir, cleanup_paths


def _idyom_mean_information_content(
    melody: Melody,
    config_key: str,
    corpus: Optional[os.PathLike] = None,
    key_estimation: str = "infer_if_necessary",
) -> float:
    """Run IDyOM for one melody and return mean information content."""
    if config_key not in _DEFAULT_IDYOM_CONFIGS:
        raise ValueError(
            f"Unknown IDyOM configuration {config_key!r}; "
            f"expected one of {sorted(_DEFAULT_IDYOM_CONFIGS)}"
        )

    idyom_config = _DEFAULT_IDYOM_CONFIGS[config_key]
    idyom_corpus = _resolve_idyom_corpus(idyom_config, override=corpus)

    input_directory, cleanup_paths = _melody_idyom_input_directory(melody)
    try:
        idyom_results = get_idyom_results(
            input_directory,
            idyom_config.target_viewpoints,
            idyom_config.source_viewpoints,
            idyom_config.models,
            idyom_config.ppm_order,
            idyom_corpus,
            f"IDyOM_{config_key}_Results",
            key_estimation,
        )
    finally:
        for path in cleanup_paths:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

    melody_features = idyom_results.get("1")
    if not melody_features or "mean_information_content" not in melody_features:
        raise RuntimeError(
            f"IDyOM did not return mean information content for configuration {config_key!r}"
        )
    return float(melody_features["mean_information_content"])
















def get_complexity_features(
    melody: Melody, phrase_gap: float = 1.5, max_ngram_order: int = DEFAULT_MAX_NGRAM_ORDER
) -> Dict:
    """Dynamically collect all complexity features for a melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze
    phrase_gap : float, optional
        Phrase gap for mtype features (default: 1.5)
    max_ngram_order : int, optional
        Maximum inclusive n-gram length for m-type features (default: 5)

    Returns
    -------
    Dict
        Dictionary of complexity feature values
    """
    features = {}
    complexity_functions = _get_features_by_type(FeatureType.COMPLEXITY)

    for name, func in complexity_functions.items():
        try:
            if name in ('InverseEntropyWeighting', 'get_mtype_features'):
                continue

            result = _invoke_feature(
                func,
                melody,
                phrase_gap=phrase_gap,
                max_ngram_order=max_ngram_order,
            )
            features[name] = result

        except Exception as e:
            print(f"Warning: Could not compute {name}: {e}")
            features[name] = None

    return features




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
    features: Dict[str, Any] = {}
    rhythm_functions = _get_features_by_domain_and_types("rhythm", allowed_types)

    beat_histogram_functions = []
    regular_functions = []

    for name, func in rhythm_functions.items():
        if _is_beat_histogram_function(func):
            beat_histogram_functions.append((name, func))
        else:
            regular_functions.append((name, func))

    for name, func in regular_functions:
        try:
            result = _invoke_feature(func, melody)
            if isinstance(result, tuple) and len(result) == 2:
                features[f"{name}_mean"] = result[0]
                features[f"{name}_std"] = result[1]
            else:
                features[name] = result
        except Exception as e:
            print(f"Warning: Could not compute {name}: {e}")
            features[name] = None

    if beat_histogram_functions:
        beat_histogram_data = _precompute_beat_histogram_data(melody)
        for name, func in beat_histogram_functions:
            try:
                result = _invoke_feature(func, melody)
                if isinstance(result, tuple) and len(result) == 2:
                    features[f"{name}_mean"] = result[0]
                    features[f"{name}_std"] = result[1]
                else:
                    features[name] = result
            except Exception as e:
                print(f"Warning: Could not compute {name}: {e}")
                features[name] = None

    return features


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


def get_expectation_features(melody: Melody) -> Dict:
    """Dynamically collect all expectation features for a melody.

    Collects features decorated with FeatureType.EXPECTATION regardless of domain.
    """
    features: Dict[str, Any] = {}
    expectation_functions = _get_features_by_type(FeatureType.EXPECTATION)

    for name, func in expectation_functions.items():
        # IDyOM mean-IC features are computed in batch via get_idyom_results / _run_idyom_analysis
        if name in _IDYOM_MEAN_INFORMATION_CONTENT_EXPORTS:
            continue
        try:
            result = _invoke_feature(func, melody)

            # Allow tuple returns to be expanded into mean/std when applicable
            if isinstance(result, tuple) and len(result) == 2 and all(isinstance(x, (int, float)) for x in result):
                features[f"{name}_mean"] = result[0]
                features[f"{name}_std"] = result[1]
            else:
                features[name] = result

        except Exception as e:
            print(f"Warning: Could not compute {name}: {e}")
            features[name] = None

    return features


def get_metre_features(melody: Melody) -> Dict:
    """Dynamically collect all metre features for a melody.

    Collects features decorated with @metre type.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    Dict
        Dictionary of metre feature values
    """
    features = {}
    metre_functions = _get_features_by_type(FeatureType.METRE)

    for name, func in metre_functions.items():
        try:
            result = _invoke_feature(func, melody)
            features[name] = result

        except Exception as e:
            print(f"Warning: Could not compute {name}: {e}")
            features[name] = None

    return features




















# Per-melody timing keys aligned with FeatureType taxonomy (see feature_decorators.py).
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


def process_melody(args):
    """Process a single melody and return its features.

    Parameters
    ----------
    args : tuple
        Tuple containing (melody_data, corpus_stats, idyom_features, phrase_gap, max_ngram_order, key_estimation)

    Returns
    -------
    tuple
        Tuple containing (melody_id, feature_dict, timings)
    """
    # Suppress warnings in worker processes
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="pretty_midi")
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="pkg_resources"
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*pkg_resources is deprecated.*"
    )

    start_total = time.time()

    melody_data, corpus_stats, idyom_results_dict, phrase_gap, max_ngram_order, key_estimation = args
    mel = Melody(melody_data)

    # Time each taxonomy category separately for logging
    timings: Dict[str, float] = {category: 0.0 for category in TIMING_STAT_CATEGORIES}

    start = time.time()
    pitch_features = get_pitch_features(mel)
    timings["absolute_pitch"] = time.time() - start

    start = time.time()
    pitch_class_features = get_pitch_class_features(mel)
    timings["pitch_class"] = time.time() - start

    start = time.time()
    interval_features = get_interval_features(mel)
    timings["pitch_interval"] = time.time() - start

    start = time.time()
    contour_features = get_contour_features(mel)
    timings["contour"] = time.time() - start

    start = time.time()
    timing_features = get_timing_features(mel)
    timings["timing"] = time.time() - start

    start = time.time()
    ioi_features = get_inter_onset_interval_features(mel)
    timings["inter_onset_interval"] = time.time() - start

    rhythm_features = {
        **timing_features,
        **ioi_features,
        **get_metric_accent_features(mel),
    }

    start = time.time()
    tonality_features = get_tonality_features(mel, key_estimation=key_estimation)
    timings["tonality"] = time.time() - start

    start = time.time()
    metre_features = get_metre_features(mel)
    timings["metre"] = time.time() - start

    start = time.time()
    expectation_features = get_expectation_features(mel)
    timings["expectation"] = time.time() - start

    start = time.time()
    complexity_features = get_complexity_features(
        mel, phrase_gap=phrase_gap, max_ngram_order=max_ngram_order
    )
    timings["complexity"] = time.time() - start

    start = time.time()
    lexical_diversity_features = get_lexical_diversity_features(
        mel, phrase_gap=phrase_gap, max_ngram_order=max_ngram_order
    )
    timings["lexical_diversity"] = time.time() - start

    melody_features = {
        "pitch_features": pitch_features,
        "pitch_class_features": pitch_class_features,
        "interval_features": interval_features,
        "contour_features": contour_features,
        "rhythm_features": rhythm_features,
        "tonality_features": tonality_features,
        "metre_features": metre_features,
        "expectation_features": expectation_features,
        "complexity_features": {**complexity_features, **lexical_diversity_features},
    }

    # Add corpus features only if corpus stats are available
    if corpus_stats:
        start = time.time()
        melody_features["corpus_features"] = get_corpus_features(
            mel, corpus_stats, phrase_gap=phrase_gap, max_ngram_order=max_ngram_order
        )
        timings["corpus"] = time.time() - start

    # Add pre-computed IDyOM features if available for this melody's ID
    melody_id_str = str(melody_data["melody_num"])

    # Handle IDyOM results dictionary (multiple configurations)
    idyom_features = {}
    if idyom_results_dict:
        for idyom_name, idyom_results in idyom_results_dict.items():
            if idyom_results and melody_id_str in idyom_results:
                for feature_key, feature_value in idyom_results[melody_id_str].items():
                    # Match the header format: idyom_{idyom_name}_features.{feature_key}
                    idyom_features[f"idyom_{idyom_name}_features.{feature_key}"] = feature_value
            else:
                # Add fallback value for this config if results not found
                idyom_features[f"idyom_{idyom_name}_features.mean_information_content"] = -1

    if idyom_features:
        melody_features["idyom_features"] = idyom_features

    timings["total"] = time.time() - start_total

    return melody_data["ID"], melody_features, timings


def get_idyom_results(
    input_directory,
    idyom_target_viewpoints,
    idyom_source_viewpoints,
    models,
    ppm_order,
    corpus_path,
    experiment_name="IDyOM_Feature_Set_Results",
    key_estimation="infer_if_necessary",
) -> dict:
    logger = logging.getLogger("melody_features")
    """Run IDyOM on the input MIDI directory and return mean information content for each melody.
    Uses the parameters supplied from Config dataclass to control IDyOM behaviour.

    Parameters
    ----------
    key_estimation : str, optional
        Key estimation strategy: "always_read_from_file", "infer_if_necessary", or "always_infer"

    Returns
    -------
    dict
        A dictionary mapping melody IDs to their mean information content.
    """
    logger = logging.getLogger("melody_features")

    # Set default IDyOM viewpoints if not provided.
    if idyom_target_viewpoints is None:
        idyom_target_viewpoints = ["cpitch"]

    if idyom_source_viewpoints is None:
        idyom_source_viewpoints = [("cpint", "cpintfref")]

    logger.info(
        f"Creating temporary MIDI files with key_estimation='{key_estimation}' for IDyOM processing..."
    )
    temp_dir = tempfile.mkdtemp(prefix="idyom_key_")
    original_input_dir = input_directory
    input_directory = create_temp_midi_with_key_signature(input_directory, temp_dir, key_estimation)

    temp_files = glob.glob(os.path.join(input_directory, "*.mid"))
    temp_files.extend(glob.glob(os.path.join(input_directory, "*.midi")))

    try:
        # Try without pretraining first to see if that's the issue
        dat_file_path = run_idyom(
            input_directory,
            pretraining_path=corpus_path,  # Use the actual corpus path
            output_dir=".",
            experiment_name=experiment_name,
            target_viewpoints=idyom_target_viewpoints,
            source_viewpoints=idyom_source_viewpoints,
            models=models,
            detail=2,
            ppm_order=ppm_order,
        )

        if not dat_file_path:
            logger.warning(
                "run_idyom did not produce an output file. Skipping IDyOM features."
            )
            return {}

        # Get a naturally sorted list of MIDI files to match IDyOM's processing order.
        # Since we created temp files, IDyOM processed the temp directory files
        if temp_dir:
            # Use the temp directory files since that's what IDyOM actually processed
            midi_files = natsorted(glob.glob(os.path.join(input_directory, "*.mid")))
        else:
            # Use original input directory for file mapping if no temp files were created
            midi_files = natsorted(glob.glob(os.path.join(original_input_dir, "*.mid")))
            midi_files.extend(
                natsorted(glob.glob(os.path.join(original_input_dir, "*.midi")))
            )

        idyom_results = {}
        try:
            with open(dat_file_path, "r", encoding="utf-8") as f:
                # Read header to determine column names
                header_line = next(f).strip()
                header_parts = header_line.split()

                logger.debug(f"IDyOM header: {header_line}")
                logger.debug(f"IDyOM header parts: {header_parts}")

                # Find the column index for information content
                # The dat file typically has: melody.id melody.name information.content
                # We want to extract the last column (information content)
                if len(header_parts) < 2:
                    logger.error(f"Invalid header format: {header_line}")
                    return {}

                # The last column should be the information content value
                info_content_col_idx = len(header_parts) - 1

                logger.debug(f"Will extract information content from column index {info_content_col_idx} (header has {len(header_parts)} columns)")

                line_count = 0
                for line in f:
                    line_count += 1
                    parts = line.strip().split()

                    if len(parts) < 2:
                        logger.warning(f"Skipping malformed line: {line.strip()}")
                        continue  # Skip malformed lines

                    try:
                        # IDyOM's melody ID is a 1-based index.
                        melody_idx = int(parts[0]) - 1

                        if 0 <= melody_idx < len(midi_files):
                            # Map the index to the melody number (1-based index)
                            melody_id = str(melody_idx + 1)

                            logger.debug(f"Processing melody {melody_id}: parts={parts}, len={len(parts)}")

                            # Extract the information content value from the correct column
                            if len(parts) <= info_content_col_idx:
                                logger.warning(
                                    f"Not enough columns in line for melody {melody_id}. Expected at least {info_content_col_idx + 1}, got {len(parts)}. Parts: {parts}"
                                )
                                continue

                            try:
                                feature_value = float(parts[info_content_col_idx])
                                features = {"mean_information_content": feature_value}
                                idyom_results[melody_id] = features
                                logger.debug(f"Extracted mean_information_content={feature_value} for melody {melody_id}")
                            except (ValueError, IndexError) as e:
                                logger.warning(
                                    f"Could not parse information content at index {info_content_col_idx} for melody {melody_id}: {e}, parts={parts}"
                                )
                        else:
                            logger.warning(
                                f"IDyOM returned an out-of-bounds index: {parts[0]} (max: {len(midi_files)-1})"
                            )
                    except (ValueError, IndexError) as e:
                        logger.warning(
                            f"Could not parse line in IDyOM output: '{line.strip()}'. Error: {e}"
                        )

            os.remove(dat_file_path)

        except FileNotFoundError:
            logger.warning(
                f"IDyOM output file not found at {dat_file_path}. Skipping IDyOM features."
            )
            return {}
        except Exception as e:
            logger.error(
                f"Error parsing IDyOM output file: {e}. Skipping IDyOM features."
            )
            if os.path.exists(dat_file_path):
                os.remove(dat_file_path)
            return {}

        return idyom_results

    finally:
        # Clean up temporary directory if it was created
        if temp_dir and os.path.exists(temp_dir):
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)


def to_mido_key_string(key_name):
    """Convert key name to mido key string format."""
    key_name = key_name.strip().lower()

    # mido only allows certain key names, so we need to catch enharmonics
    # see https://mido.readthedocs.io/en/stable/meta_message_types.html
    enharmonic_map = {
        "a#": "bb",
        "a# major": "bb major",
        "d#": "eb",
        "d# major": "eb major",
        "g#": "ab",
        "g# major": "ab major",
        "c#": "db",
        "c# major": "db major",
        "f#": "gb",
        "f# major": "gb major",
    }

    # Apply enharmonic mapping
    if key_name in enharmonic_map:
        key_name = enharmonic_map[key_name]

    if "minor" in key_name:
        root = key_name.replace(" minor", "").replace("min", "").strip().capitalize()
        return f"{root}m"
    else:
        root = key_name.replace(" major", "").strip().capitalize()
        return root


def create_temp_midi_with_key_signature(
    input_directory: str,
    temp_dir: str,
    key_estimation: str = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler"
) -> str:
    """
    Create temporary MIDI files with key signatures for IDyOM processing.

    Parameters
    ----------
    input_directory : str
        Path to the input directory containing MIDI files
    temp_dir : str
        Path to the temporary directory to create the modified MIDI files
    key_estimation : str, optional
        Key estimation strategy: "always_read_from_file", "infer_if_necessary", or "always_infer"
    key_finding_algorithm : Literal["krumhansl_schmuckler"], optional
        Key-finding algorithm to use when inferring key. Default is "krumhansl_schmuckler".

    Returns
    -------
    str
        Path to the temporary directory containing MIDI files with key signatures
    """
    logger = logging.getLogger("melody_features")
    from mido import MetaMessage, MidiFile

    # Create temporary directory
    os.makedirs(temp_dir, exist_ok=True)

    # Get all MIDI files from input directory
    midi_files = glob.glob(os.path.join(input_directory, "*.mid"))
    midi_files.extend(glob.glob(os.path.join(input_directory, "*.midi")))

    logger.info(
        f"Processing {len(midi_files)} MIDI files with key_estimation='{key_estimation}'..."
    )

    successful_copies = 0
    for midi_file in midi_files:
        try:
            # First, try to copy the original file as a fallback
            # Always save with .mid extension for IDyOM compatibility
            base_filename = os.path.splitext(os.path.basename(midi_file))[0]
            output_filename = base_filename + ".mid"
            output_path = os.path.join(temp_dir, output_filename)
            shutil.copy2(midi_file, output_path)
            successful_copies += 1

            # Handle key signature based on key_estimation strategy
            try:
                mid = MidiFile(midi_file)

                has_key_signature = False
                for track in mid.tracks:
                    for msg in track:
                        if msg.type == "key_signature":
                            has_key_signature = True
                            break
                    if has_key_signature:
                        break

                should_apply_estimated_key = False

                if key_estimation == "always_read_from_file":
                    if not has_key_signature:
                        raise ValueError(f"No key signature found in MIDI file: {midi_file}")
                    continue
                elif key_estimation == "infer_if_necessary":
                    should_apply_estimated_key = not has_key_signature
                elif key_estimation == "always_infer":
                    should_apply_estimated_key = True
                else:
                    raise ValueError(f"Invalid key_estimation value: {key_estimation}")

                if should_apply_estimated_key:
                    midi_dict = import_midi(midi_file)
                    if midi_dict is None:
                        logger.warning(f"Could not import {midi_file}, using original file")
                        continue

                    melody = Melody(midi_dict)
                    if not melody.pitches:
                        logger.warning(
                            f"No pitches found in {midi_file}, using original file"
                        )
                        continue

                    key_name, mode = infer_key_from_pitches(
                        melody.pitches,
                        algorithm=key_finding_algorithm
                    )
                    if key_name and mode:
                        detected_key = f"{key_name} {mode}"
                    else:
                        logger.warning(
                            f"Could not infer key for {midi_file}, using original file"
                        )
                        continue

                    mido_key = to_mido_key_string(detected_key)

                    for track in mid.tracks:
                        track[:] = [
                            msg for msg in track if not (msg.type == "key_signature")
                        ]

                    key_msg = MetaMessage("key_signature", key=mido_key, time=0)
                    mid.tracks[0].insert(0, key_msg)

                    mid.save(output_path)

            except Exception as e:
                logger.warning(
                    f"Could not add key signature to {midi_file}: {str(e)}, using original file"
                )

        except Exception as e:
            logger.error(f"Could not copy {midi_file}: {str(e)}")
            continue

    created_files = glob.glob(os.path.join(temp_dir, "*.mid"))

    logger.info(
        f"Successfully created {len(created_files)} files in temporary directory"
    )

    return temp_dir

def _setup_default_config(config: Optional[Config]) -> Config:
    """Set up default configuration if none provided.

    Parameters
    ----------
    config : Optional[Config]
        Configuration object or None

    Returns
    -------
    Config
        Valid configuration object
    """
    if config is None:
        config = Config(
            corpus=_DEFAULT_CORPUS,
            idyom={
                "pitch_stm": IDyOMConfig(
                    target_viewpoints=["cpitch"],
                    source_viewpoints=[("cpitch", "cpint", "cpintfref")],
                    ppm_order=None,
                    models=":stm",
                    corpus=None,
                ),
                "pitch_ltm": IDyOMConfig(
                    target_viewpoints=["cpitch"],
                    source_viewpoints=[("cpitch", "cpint", "cpintfref")],
                    ppm_order=None,
                    models=":ltm",
                    corpus=_DEFAULT_CORPUS,
                ),
                "rhythm_stm": IDyOMConfig(
                    target_viewpoints=["onset"],
                    source_viewpoints=["ioi", "ioi-ratio"],
                    ppm_order=None,
                    models=":stm",
                    corpus=None,
                ),
                "rhythm_ltm": IDyOMConfig(
                    target_viewpoints=["onset"],
                    source_viewpoints=["ioi", "ioi-ratio"],
                    ppm_order=None,
                    models=":ltm",
                    corpus=_DEFAULT_CORPUS,
                ),
            },
            fantastic=FantasticConfig(
                max_ngram_order=DEFAULT_MAX_NGRAM_ORDER,
                phrase_gap=1.5,
                corpus=None,
            ),
            key_estimation="infer_if_necessary",
        )
    return config


def _validate_config(config: Config) -> None:
    """Validate the configuration object.

    Parameters
    ----------
    config : Config
        Configuration object to validate

    Raises
    ------
    ValueError
        If configuration is invalid
    """
    if not hasattr(config, "idyom") or not config.idyom:
        raise ValueError("Config must have at least one IDyOM configuration")

    if not hasattr(config, "fantastic"):
        raise ValueError("Config must have FANTASTIC configuration")




def _load_melody_data(input: Union[os.PathLike, List[os.PathLike]]) -> List[dict]:
    """Load and validate melody data from MIDI files or JSON.

    Parameters
    ----------
    input : Union[os.PathLike, List[os.PathLike]]
        Path to input directory, JSON file, list of MIDI file paths, or single MIDI file path

    Returns
    -------
    List[dict]
        List of valid monophonic melody data dictionaries

    Raises
    ------
    FileNotFoundError
        If no MIDI files found in directory or list
    ValueError
        If input is not a valid type
    """
    logger = logging.getLogger("melody_features")
    from multiprocessing import Pool, cpu_count

    melody_data_list = []

    if isinstance(input, list):
        midi_files = []
        for file_path in input:
            if isinstance(file_path, (str, os.PathLike)):
                file_path = str(file_path)
                if file_path.lower().endswith(('.mid', '.midi')):
                    midi_files.append(file_path)
                else:
                    logger.warning(f"Skipping non-MIDI file: {file_path}")
            else:
                logger.warning(f"Skipping invalid file path: {file_path}")

        if not midi_files:
            raise FileNotFoundError("No valid MIDI files found in the provided list")

        midi_files = natsorted(midi_files)

    elif os.path.isdir(input):
        midi_files = glob.glob(os.path.join(input, "*.mid"))
        midi_files.extend(glob.glob(os.path.join(input, "*.midi")))

        if not midi_files:
            raise FileNotFoundError(
                f"No MIDI files found in the specified directory: {input}"
            )

        # Sort MIDI files in natural order
        midi_files = natsorted(midi_files)

    elif isinstance(input, (str, os.PathLike)) and str(input).lower().endswith(('.mid', '.midi')):
        # Handle single MIDI file
        midi_files = [str(input)]

    elif isinstance(input, (str, os.PathLike)) and str(input).endswith(".json"):
        with open(input, encoding="utf-8") as f:
            all_data = json.load(f)

        # Filter for monophonic melodies from the JSON data.
        for melody_data in all_data:
            if melody_data:
                temp_mel = Melody(melody_data)
                if _check_is_monophonic(temp_mel):
                    melody_data_list.append(melody_data)
                else:
                    logger.warning(
                        f"Skipping polyphonic melody from JSON: {melody_data.get('ID', 'Unknown ID')}"
                    )

        melody_data_list = [m for m in melody_data_list if m is not None]
        logger.info(f"Processing {len(melody_data_list)} melodies from JSON")

        if not melody_data_list:
            return []

        for idx, melody_data in enumerate(melody_data_list, 1):
            melody_data["melody_num"] = idx

        return melody_data_list

    else:
        raise ValueError(
            f"Input must be a directory containing MIDI files, a JSON file, a list of MIDI file paths, or a single MIDI file path. Got: {input}"
        )

    for midi_file in midi_files:
        try:
            midi_data = import_midi(midi_file)
            if midi_data:
                temp_mel = Melody(midi_data)
                if _check_is_monophonic(temp_mel):
                    melody_data_list.append(midi_data)
                else:
                    logger.warning(f"Skipping polyphonic file: {midi_file}")
        except Exception as e:
            logger.error(f"Error importing {midi_file}: {str(e)}")
            continue

    melody_data_list = [m for m in melody_data_list if m is not None]
    logger.info(f"Processing {len(melody_data_list)} melodies")

    if not melody_data_list:
        return []

    # Assign unique melody_num to each melody (in sorted order)
    for idx, melody_data in enumerate(melody_data_list, 1):
        melody_data["melody_num"] = idx

    return melody_data_list


def _run_idyom_analysis(
    input: Union[os.PathLike, List[os.PathLike]], config: Config
) -> Dict[str, dict]:
    """Run IDyOM analysis for all configurations.

    Parameters
    ----------
    input : Union[os.PathLike, List[os.PathLike]]
        Path to input directory, list of MIDI file paths, or single MIDI file path
    config : Config
        Configuration object containing IDyOM settings

    Returns
    -------
    Dict[str, dict]
        Dictionary mapping IDyOM configuration names to their results
    """
    logger = logging.getLogger("melody_features")
    idyom_results_dict = {}

    if isinstance(input, list):
        temp_dir = tempfile.mkdtemp(prefix="idyom_input_")
        try:
            for i, file_path in enumerate(input):
                if isinstance(file_path, (str, os.PathLike)) and str(file_path).lower().endswith(('.mid', '.midi')):
                    import shutil
                    file_ext = os.path.splitext(str(file_path))[1]
                    temp_file_path = os.path.join(temp_dir, f"file_{i+1:04d}{file_ext}")
                    shutil.copy2(str(file_path), temp_file_path)

            idyom_input_path = temp_dir
        except Exception as e:
            logger.error(f"Error creating temporary directory for IDyOM: {e}")
            return {}
    elif os.path.isdir(input):
        idyom_input_path = input
    elif isinstance(input, (str, os.PathLike)) and str(input).lower().endswith(('.mid', '.midi')):
        temp_dir = tempfile.mkdtemp(prefix="idyom_input_")
        try:
            import shutil
            file_ext = os.path.splitext(str(input))[1]
            temp_file_path = os.path.join(temp_dir, f"file_0001{file_ext}")
            shutil.copy2(str(input), temp_file_path)
            idyom_input_path = temp_dir
        except Exception as e:
            logger.error(f"Error creating temporary directory for IDyOM: {e}")
            return {}
    else:
        logger.error(f"Unsupported input type for IDyOM: {type(input)}")
        return {}

    for idyom_name, idyom_config in config.idyom.items():
        idyom_corpus = _resolve_idyom_corpus(idyom_config, config_corpus=config.corpus)
        logger.info(
            f"Running IDyOM analysis for '{idyom_name}' with corpus: {idyom_corpus}"
        )

        try:
            idyom_results = get_idyom_results(
                idyom_input_path,
                idyom_config.target_viewpoints,
                idyom_config.source_viewpoints,
                idyom_config.models,
                idyom_config.ppm_order,
                idyom_corpus,
                f"IDyOM_{idyom_name}_Results",
                config.key_estimation,
            )
            idyom_results_dict[idyom_name] = idyom_results
        except Exception as e:
            logger.error(f"Failed to run IDyOM for '{idyom_name}': {e}")
            idyom_results_dict[idyom_name] = {}

    # Clean up temporary directory if it was created
    if isinstance(input, list) or (isinstance(input, (str, os.PathLike)) and str(input).lower().endswith(('.mid', '.midi'))):
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Could not clean up temporary IDyOM directory: {e}")

    return idyom_results_dict


def _setup_parallel_processing(
    melody_data_list: List[dict],
    corpus_stats: Optional[dict],
    idyom_results_dict: Dict[str, dict],
    config: Config,
) -> Tuple[List[str], List, Dict[str, List[float]]]:
    """Set up parallel processing arguments and headers.

    Parameters
    ----------
    melody_data_list : List[dict]
        List of melody data dictionaries
    corpus_stats : Optional[dict]
        Corpus statistics dictionary
    idyom_results_dict : Dict[str, dict]
        Dictionary of IDyOM results
    config : Config
        Configuration object

    Returns
    -------
    Tuple[List[str], List, Dict[str, List[float]]]
        Headers, melody arguments, and timing statistics dictionary
    """
    # Suppress warnings at the system level
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="pretty_midi")
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="pkg_resources"
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*pkg_resources is deprecated.*"
    )

    from multiprocessing import cpu_count

    # Process first melody to get header structure
    mel = Melody(melody_data_list[0])
    first_features = {
        "pitch_features": get_pitch_features(mel),
        "pitch_class_features": get_pitch_class_features(mel),
        "interval_features": get_interval_features(mel),
        "contour_features": get_contour_features(mel),
        "rhythm_features": get_rhythm_features(mel),
        "tonality_features": get_tonality_features(mel, key_estimation=config.key_estimation),
        "metre_features": get_metre_features(mel),
        "expectation_features": get_expectation_features(mel),
        "complexity_features": get_complexity_feature_bundle(
            mel,
            phrase_gap=config.fantastic.phrase_gap,
            max_ngram_order=config.fantastic.max_ngram_order,
        ),
    }

    if corpus_stats:
        first_features["corpus_features"] = get_corpus_features(
            mel,
            corpus_stats,
            phrase_gap=config.fantastic.phrase_gap,
            max_ngram_order=config.fantastic.max_ngram_order,
        )

    # Add IDyOM features for each config to the header
    for idyom_name, idyom_results in idyom_results_dict.items():
        if idyom_results:
            sample_id = next(iter(idyom_results))
            for feature in idyom_results[sample_id].keys():
                first_features[f"idyom_{idyom_name}_features.{feature}"] = None
        else:
            # Add header for fallback value even if no results
            first_features[f"idyom_{idyom_name}_features.mean_information_content"] = None

    # Create header by flattening feature names
    headers = ["melody_num", "melody_id"]
    for category, features in first_features.items():
        if isinstance(features, dict):
            headers.extend(f"{category}.{feature}" for feature in features.keys())
        elif features is None:
            # Already prefixed for IDyOM
            headers.append(category)

    logger = logging.getLogger("melody_features")
    logger.info("Starting parallel processing...")
    # Create pool of workers
    n_cores = cpu_count()
    logger.info(f"Using {n_cores} CPU cores")

    # Prepare arguments for parallel processing
    melody_args = [
        (
            melody_data,
            corpus_stats,
            idyom_results_dict,
            config.fantastic.phrase_gap,
            config.fantastic.max_ngram_order,
            config.key_estimation,
        )
        for melody_data in melody_data_list
    ]

    timing_stats = _init_timing_stats()

    return headers, melody_args, timing_stats


def _process_melodies_parallel(
    melody_args: List,
    headers: List[str],
    melody_data_list: List[dict],
    idyom_results_dict: Dict[str, dict],
    timing_stats: Dict[str, List[float]],
) -> List[List]:
    """Process melodies in parallel and collect results.

    Parameters
    ----------
    melody_args : List
        Arguments for parallel processing
    headers : List[str]
        CSV headers
    melody_data_list : List[dict]
        List of melody data dictionaries
    idyom_results_dict : Dict[str, dict]
        Dictionary of IDyOM results
    timing_stats : Dict[str, List[float]]
        Timing statistics dictionary

    Returns
    -------
    List[List]
        List of feature rows
    """
    all_features = []

    try:
        # Try to use multiprocessing
        from multiprocessing import Pool, cpu_count
        import multiprocessing as mp

        # Set start method to 'fork' for better compatibility
        try:
            mp.set_start_method('fork', force=True)
        except RuntimeError:
            pass  # Start method already set

        logger = logging.getLogger("melody_features")
        logger.info("Parallel processing initiated")

        n_cores = cpu_count()
        chunk_size = max(1, len(melody_args) // (n_cores * 4))

        with Pool(n_cores) as pool:
            # Use tqdm to show progress as melodies are processed
            with tqdm(
                total=len(melody_args),
                desc="Processing melodies",
                unit="melody",
                ncols=80,
                mininterval=0.5,
                maxinterval=2.0,
                miniters=1,
                smoothing=0.1,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ) as pbar:
                for result in pool.imap(process_melody, melody_args, chunksize=chunk_size):
                    try:
                        melody_id, melody_features, timings = result
                        melody_num = None
                        for m in melody_data_list:
                            if str(m["ID"]) == str(melody_id):
                                melody_num = m.get("melody_num", None)
                                break
                        row = [melody_num, melody_id]
                        for header in headers[2:]:  # Skip melody_num and melody_id headers
                            if header.startswith("idyom_"):
                                prefix, feature_name = header.split(".", 1)
                                idyom_name = prefix[len("idyom_") : -len("_features")]
                                # Use melody_num for IDyOM lookup since IDyOM results are indexed by melody number
                                value = (
                                    idyom_results_dict.get(idyom_name, {})
                                    .get(str(melody_num), {})
                                    .get(feature_name, 0.0)
                                )
                                row.append(value)
                            else:
                                category, feature_name = header.split(".", 1)
                                value = melody_features.get(category, {}).get(
                                    feature_name, 0.0
                                )
                                row.append(value)
                        all_features.append(row)

                        for category, duration in timings.items():
                            timing_stats[category].append(duration)

                        # Update progress bar
                        pbar.update(1)

                    except Exception as e:
                        logger = logging.getLogger("melody_features")
                        logger.error(f"Error processing melody: {str(e)}")
                        pbar.update(1)  # Still update progress even on error
                        continue

    except Exception as e:
        # Fall back to sequential processing if multiprocessing fails
        logger = logging.getLogger("melody_features")
        logger.warning(f"Parallel processing failed ({str(e)}), falling back to sequential processing")

        with tqdm(
            total=len(melody_args),
            desc="Processing melodies (sequential)",
            unit="melody",
            ncols=80,
            mininterval=0.5,
            maxinterval=2.0,
            miniters=1,
            smoothing=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) as pbar:
            for i, args in enumerate(melody_args):
                try:
                    result = process_melody(args)
                    melody_id, melody_features, timings = result

                    melody_num = None
                    for m in melody_data_list:
                        if str(m["ID"]) == str(melody_id):
                            melody_num = m.get("melody_num", None)
                            break
                    row = [melody_num, melody_id]

                    for header in headers[2:]:  # Skip melody_num and melody_id headers
                        if header.startswith("idyom_"):
                            prefix, feature_name = header.split(".", 1)
                            idyom_name = prefix[len("idyom_") : -len("_features")]
                            # Use melody_num for IDyOM lookup since IDyOM results are indexed by melody number
                            value = (
                                idyom_results_dict.get(idyom_name, {})
                                .get(str(melody_num), {})
                                .get(feature_name, 0.0)
                            )
                            row.append(value)
                        else:
                            category, feature_name = header.split(".", 1)
                            value = melody_features.get(category, {}).get(
                                feature_name, 0.0
                            )
                            row.append(value)
                    all_features.append(row)

                    # Update timing statistics
                    for category, duration in timings.items():
                        timing_stats[category].append(duration)

                    # Update progress bar
                    pbar.update(1)

                except Exception as e:
                    logger.error(f"Error processing melody {i}: {str(e)}")
                    pbar.update(1)  # Still update progress even on error
                    continue

    return all_features



def _cleanup_idyom_temp_output():
    """Clean up any existing IDyOM temporary output directory to prevent conflicts."""
    import shutil
    from pathlib import Path

    idyom_temp_dir = Path("idyom_temp_output")
    if idyom_temp_dir.exists():
        logger = logging.getLogger("melody_features")
        logger.info(f"Cleaning up existing IDyOM temporary directory: {idyom_temp_dir}")
        try:
            shutil.rmtree(idyom_temp_dir)
            logger.info("Successfully cleaned up IDyOM temporary directory")
        except Exception as e:
            logger.warning(f"Could not clean up IDyOM temporary directory: {e}")


def _get_features_by_source(source: str) -> Dict[str, callable]:
    """Get all functions/classes decorated with a specific source.

    Parameters
    ----------
    source : str
        The source label to filter by (e.g., 'fantastic', 'jsymbolic')

    Returns
    -------
    Dict[str, callable]
        Dictionary mapping function names to their callable objects
    """
    import inspect
    import melody_features.features as features_module

    source_features = {}

    for name, obj in inspect.getmembers(features_module):
        # Check if it's a function or class with the specified source
        if (inspect.isfunction(obj) or
            (inspect.isclass(obj) or (hasattr(obj, "__call__") and hasattr(obj, "__name__")))):

            # Check for multiple sources (new approach)
            if hasattr(obj, "_feature_sources") and source in obj._feature_sources:
                source_features[name] = obj
            # Fallback to single source (backward compatibility)
            elif hasattr(obj, "_feature_source") and obj._feature_source == source:
                source_features[name] = obj

    return source_features


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


def get_all_features(
    input: Union[os.PathLike, List[os.PathLike]],
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

    If a path to a corpus of MIDI files is provided in the Config,
    corpus statistics will be computed following FANTASTIC's n-gram document frequency
    model (Müllensiefen, 2009). If not, this will be skipped.
    This function will also run IDyOM (Pearce, 2005) on the input MIDI files.
    If a corpus of MIDI files is provided in the Config, IDyOM will be run with
    pretraining on the corpus. If not, it will be run without pretraining.

    Parameters
    ----------
    input : Union[os.PathLike, List[os.PathLike]]
        Path to input MIDI directory, list of MIDI file paths, or single MIDI file path
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
    # Suppress warnings at the system level
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="pretty_midi")
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="pkg_resources"
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*pkg_resources is deprecated.*"
    )

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
    column_rename_map = {}
    categories = set()
    for col in df.columns:
        # skip non-feature columns
        if col in ["melody_num", "melody_id"]:
            continue
        # Check for IDyOM columns first (before generic "." check)
        if col.startswith("idyom_"):
            if "_features" in col:
                category = col.rsplit("_features", 1)[0]
            else:
                category = col
            display_name = _get_category_display_name(category)
            display_name_lower = display_name.lower().replace(" ", "_").replace("-", "_")
            if "." in col:
                _, feature_name = col.split(".", 1)
                # Extract the config name from the category (e.g., "idyom_pitch_stm" -> "pitch_stm")
                if category.startswith("idyom_"):
                    config_name = category[6:]  # Remove "idyom_" prefix
                    new_col_name = f"{display_name_lower}.{config_name}_{feature_name}"
                else:
                    new_col_name = f"{display_name_lower}.{feature_name}"
            else:
                new_col_name = col  # keep as is if no feature name
            column_rename_map[col] = new_col_name
            categories.add(display_name_lower)
        elif "." in col:
            category, feature_name = col.split(".", 1)
            display_name = _get_category_display_name(category, feature_name)
            display_name_lower = display_name.lower().replace(" ", "_").replace("-", "_")
            new_col_name = f"{display_name_lower}.{feature_name}"
            column_rename_map[col] = new_col_name
            categories.add(display_name_lower)

    # rename columns in DataFrame
    df = df.rename(columns=column_rename_map)

    # Log timing statistics
    end_time = time.time()
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    logger.info("Timing Statistics (average milliseconds per melody):")
    for category in TIMING_STAT_CATEGORIES:
        times = timing_stats.get(category, [])
        if times:
            avg_time = sum(times) / len(times) * 1000  # Convert to milliseconds
            display_name = _get_category_display_name(category)
            logger.info(f"{display_name:22s}: {avg_time:8.2f}ms")

    logger.info(f"Successfully extracted features for {len(df)} melodies")

    return df
