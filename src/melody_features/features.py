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









@fantastic
@complexity
@pitch
def pitch_entropy(pitches: list[int]) -> float:
    """The zeroth-order base-2 entropy of the pitch distribution.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Shannon entropy of pitch distribution
    """
    return float(shannon_entropy(pitches))


# refstat('kkmaj') / refstat('kkmin') — used by tonality.m / keymode.m
_KK_MAJ_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=float,
)
_KK_MIN_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=float,
)






def _durdist1_vector(
    starts: list[float], ends: list[float], tempo: float = 120.0
) -> np.ndarray:
    if not starts or not ends:
        return np.zeros(9, dtype=float)
    beat_durs = [
        (float(end) - float(start)) * (tempo / 60.0)
        for start, end in zip(starts, ends)
        if float(end) > float(start)
    ]
    if not beat_durs:
        return np.zeros(9, dtype=float)
    du = np.round(2 * np.log2(np.asarray(beat_durs, dtype=float))).astype(int)
    du = du[np.abs(du) <= 4] + 5
    hist = np.zeros(9, dtype=float)
    for b in du:
        if 1 <= b <= 9:
            hist[b - 1] += 1.0
    return hist / (hist.sum() + 1e-12)


def _kkcc_from_pcd(pcd: np.ndarray) -> np.ndarray:
    majors = np.array([np.roll(_KK_MAJ_PROFILE, k) for k in range(12)])
    minors = np.array([np.roll(_KK_MIN_PROFILE, k) for k in range(12)])
    profiles = np.vstack([majors, minors])
    pcd = np.asarray(pcd, dtype=float).ravel()
    corrs = []
    for profile in profiles:
        if np.std(pcd) == 0 or np.std(profile) == 0:
            corrs.append(0.0)
        else:
            c = np.corrcoef(pcd, profile)[0, 1]
            corrs.append(0.0 if np.isnan(c) else float(c))
    return np.array(corrs)


def _keymode_from_pcd(pcd: np.ndarray) -> int:
    u = _kkcc_from_pcd(pcd)
    if u[0] > u[12]:
        return 1
    if u[0] < u[12]:
        return 2
    return 0


def _tonality_midi_toolbox(
    pitches: list[int], starts: list[float], ends: list[float]
) -> list[float]:
    if not pitches:
        return []
    pcd = _pcdist1_vector(pitches, starts, ends)
    kk = _KK_MIN_PROFILE if _keymode_from_pcd(pcd) == 2 else _KK_MAJ_PROFILE
    return [float(kk[int(p) % 12]) for p in pitches]


def _notedensity_seconds(starts: list[float]) -> float:
    if not starts or len(starts) < 2:
        return 0.0
    span = float(starts[-1]) - float(starts[0])
    if span == 0:
        return 0.0
    return (len(starts) - 1) / span



















# backwards-compatible alias (typo preserved for semantic versioning).


































@fantastic
@complexity
@pitch
def interval_entropy(pitches: list[int]) -> float:
    """The zeroth-order base-2 entropy of the interval distribution.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Shannon entropy of signed melodic intervals (in semitones).

    Note
    -----
    Uses signed intervals rather than absolute interval sizes,
    consistent with FANTASTIC implementation.
    """
    return float(shannon_entropy(pitch_interval(pitches)))


def _get_durations(starts: list[float], ends: list[float], tempo: float = 120.0) -> list[float]:
    """Safely calculate durations from start and end times, converted to quarter notes.

    Parameters
    ----------
    starts : list[float]
        List of note start times (in seconds)
    ends : list[float]
        List of note end times (in seconds)
    tempo : float
        Tempo in BPM (beats per minute), default 120.0

    Returns
    -------
    list[float]
        List of durations in quarter notes, or empty list if calculation fails
    """
    if not starts or not ends or len(starts) != len(ends):
        return []
    try:
        durations_seconds = [float(end - start) for start, end in zip(starts, ends)]
        durations_quarter_notes = [duration * (tempo / 60.0) for duration in durations_seconds]
        return durations_quarter_notes
    except (TypeError, ValueError):
        return []




@midi_toolbox
@rhythm
@timing
def durdist1(
    starts: list[float], ends: list[float], tempo: float = 120.0
) -> dict[int, float]:
    """Note duration distribution in nine log-spaced beat bins.

    Bin centers (in beats): 1/4, √2/4, 1/2, √2/2, 1, √2, 2, 2√2, 4.

    Parameters
    ----------
    starts : list[float]
        Note onset times in seconds
    ends : list[float]
        Note offset times in seconds
    tempo : float
        Tempo in BPM (default 120)

    Returns
    -------
    dict[int, float]
        Map from bin index (1–9) to proportion
    """
    if not starts or not ends:
        return {}
    vec = _durdist1_vector(starts, ends, tempo)
    return {i + 1: float(vec[i]) for i in range(9) if vec[i] > 0}


























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


@fantastic
@contour
@pitch
def get_step_contour_features(
    pitches: list[int],
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    method: str = "amads",
) -> Tuple[float, float, float]:
    """Calculate summary features from a duration-weighted step contour.

    A step contour represents a melody as a fixed-length sequence of pitch samples:
    each note's MIDI pitch is repeated in proportion to its duration relative to the
    whole melody. The implementation follows the FANTASTIC convention of resampling
    the melody to 64 steps by default. The returned features summarize the resulting
    pitch sequence as global variation, global direction, and local variation.

    Parameters
    ----------
    pitches : list[int]
        MIDI pitch values for the melody notes.
    starts : list[float]
        Note onset times in seconds.
    ends : list[float]
        Note offset times in seconds.
    tempo : float, optional
        Tempo in beats per minute, used to convert note durations to quarter-note
        units.
    method : str, optional
        Contour statistic method, either ``"amads"`` or ``"fantastic"``. Defaults
        to ``"amads"``.

    Returns
    -------
    Tuple[float, float, float]
        ``(global_variation, global_direction, local_variation)``, where global
        variation is the standard deviation of the step-contour vector, global
        direction is its correlation with an ascending linear ramp, and local
        variation is the mean absolute difference between adjacent contour samples.
    """
    if not pitches or not starts or not ends or len(pitches) < 2:
        return 0.0, 0.0, 0.0

    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return 0.0, 0.0, 0.0

    sc = StepContour(pitches, durations, method=method)
    return sc.global_variation, sc.global_direction, sc.local_variation

@fantastic
@contour
@pitch
def get_interpolation_contour_features(
    pitches: list[int], starts: list[float]
) -> Tuple[int, float, float, float, str]:
    """Calculate features from an interpolation contour.

    An interpolation contour approximates melodic shape by identifying contour
    turning points and replacing the pitch trajectory between successive turning
    points with linear gradients. This function uses the AMADS turning-point method
    by default, then summarizes the gradient sequence by overall direction, mean
    absolute gradient, gradient variability, direction-change rate, and a four-letter
    contour class.

    Parameters
    ----------
    pitches : list[int]
        MIDI pitch values for the melody notes.
    starts : list[float]
        Note onset times in seconds.

    Returns
    -------
    Tuple[int, float, float, float, str]
        ``(global_direction, mean_gradient, gradient_std, direction_changes,
        class_label)``. The class label encodes four sampled gradient categories from
        strong downward to strong upward.
    """
    ic = InterpolationContour(pitches, starts)
    return (
        ic.global_direction,
        ic.mean_gradient,
        ic.gradient_std,
        ic.direction_changes,
        ic.class_label,
    )

@midi_toolbox
@contour
@pitch
def comb_contour_matrix(pitches: list[int]) -> list[list[int]]:
    """The Marvin and Laprade comb contour matrix.

    For a melody with ``n`` notes, this feature returns an ``n x n`` lower-triangular
    binary matrix. Entry ``C[i][j]`` is ``1`` when note ``j`` is higher than note
    ``i`` and ``i >= j``; otherwise it is ``0``. The matrix encodes pairwise
    pitch-height relations in the melodic contour.

    Parameters
    ----------
    pitches : list[int]
        Sequence of MIDI pitches.

    Returns
    -------
    list[list[int]]
        Lower-triangular binary contour matrix.
    """
    num_notes = len(pitches)
    if num_notes == 0:
        return []

    matrix: list[list[int]] = [[0 for _ in range(num_notes)] for _ in range(num_notes)]
    for col_index in range(num_notes):
        pitch_at_col = pitches[col_index]
        for row_index in range(col_index, num_notes):
            matrix[row_index][col_index] = 1 if pitch_at_col > pitches[row_index] else 0

    return matrix

get_comb_contour_matrix = comb_contour_matrix

@fantastic
@contour
@pitch
def get_polynomial_contour_features(
    melody: Melody
) -> List[float]:
    """Calculate polynomial contour features.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    List[float]
        List of first 3 polynomial contour coefficients for the melody
    """
    pc = PolynomialContour(melody)
    return pc.coefficients

@fantastic
@contour
@pitch
def get_huron_contour_features(melody: Melody) -> str:
    """Classify a melody using Huron's three-point contour scheme.

    The Huron contour reduces a melody to three pitch points: the first pitch, a
    rounded duration-weighted mean pitch, and the last pitch. Their relative ordering
    is mapped to a categorical contour label such as ``"ascending"``,
    ``"descending"``, ``"convex"``, ``"concave"``, or ``"horizontal"``.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    str
        Huron contour classification.
    """
    hc = HuronContour(melody)
    return hc.class_label

@fantastic
@jsymbolic
@rhythm
@timing
def initial_tempo(melody: Melody) -> float:
    """The first tempo of the melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Tempo of melody in bpm

    """
    return melody.tempo

# Undecorated helper for internal use only
def _get_tempo(melody: Melody) -> float:
    return initial_tempo(melody)

@jsymbolic
@rhythm
@timing
def mean_tempo(melody: Melody) -> float:
    """The mean tempo of the melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Mean tempo of melody in bpm
    """
    if not melody.tempo_changes:
        return melody.tempo

    total_duration = max(melody.ends) if melody.ends else 0
    if total_duration == 0:
        return melody.tempo

    weighted_sum = 0.0
    last_time = 0.0
    last_tempo = melody.tempo

    for time, tempo in melody.tempo_changes:
        duration = time - last_time
        weighted_sum += last_tempo * duration
        last_time = time
        last_tempo = tempo

    final_duration = total_duration - last_time
    weighted_sum += last_tempo * final_duration
    
    return float(weighted_sum / total_duration)

@jsymbolic
@rhythm
@timing
def tempo_variability(melody: Melody) -> float:
    """The duration-weighted variability of tempo across the melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Weighted population standard deviation (BPM) of tempo segments.
    """
    if not melody.tempo_changes:
        return 0.0
    if not melody.ends:
        return 0.0

    total_duration = max(melody.ends)
    if total_duration <= 0.0:
        return 0.0

    segments: list[tuple[float, float]] = []
    current_time = 0.0
    current_tempo = float(melody.tempo)

    for change_time, new_tempo in melody.tempo_changes:
        change_time = float(change_time)
        if change_time <= current_time:
            current_tempo = float(new_tempo)
            continue
        segment_end = min(change_time, total_duration)
        segment_duration = segment_end - current_time
        if segment_duration > 0.0:
            segments.append((segment_duration, current_tempo))
        current_time = segment_end
        current_tempo = float(new_tempo)
        if current_time >= total_duration:
            break

    if current_time < total_duration:
        segments.append((total_duration - current_time, current_tempo))

    if not segments:
        return 0.0

    weighted_mean = sum(duration * tempo for duration, tempo in segments) / total_duration
    weighted_variance = (
        sum(duration * ((tempo - weighted_mean) ** 2) for duration, tempo in segments)
        / total_duration
    )
    return float(np.sqrt(weighted_variance))

@fantastic
@rhythm
@timing
def duration_range(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The range between the longest and shortest note duration in quarter notes.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Range between longest and shortest duration
    """
    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return 0.0
    return float(range_func(durations))

@novel
@rhythm
@timing
def mean_duration(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The mean note duration in quarter notes, computed from the raw durations.

    Parameters
    ----------
    starts : list[float]
        List of note start times (in seconds)
    ends : list[float]
        List of note end times (in seconds)
    tempo : float
        Tempo in BPM (beats per minute), default 120.0

    Returns
    -------
    float
        Mean raw note duration in quarter notes

    Note
    ----
    We use raw durations here (in the style of FANTASTIC), rather than jSymbolic's quantized rhythmic-value bins.
    """
    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return 0.0
    
    return float(np.mean(durations))

@jsymbolic
@rhythm
@timing
def average_note_duration(starts: list[float], ends: list[float]) -> float:
    """
    The average note duration in seconds.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
    
    Returns
    -------
    float
        Average note duration in seconds
    
    Note
    ----
    This feature reports duration in seconds, unlike quarter-note duration means
    such as `mean_duration` and `mean_rhythmic_value`.
    """
    durations = [end - start for start, end in zip(starts, ends)]
    if not durations:
        return 0.0
    return float(np.mean(durations))

@novel
@rhythm
@timing
def duration_standard_deviation(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The standard deviation of note durations in quarter notes.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Standard deviation of note durations
    """
    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return 0.0
    return float(np.std(durations, ddof=1))

@jsymbolic
@rhythm
@timing
def variability_of_note_durations(starts: list[float], ends: list[float]) -> float:
    """The standard deviation of note durations in seconds.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Standard deviation of note durations
    """
    durations = [end - start for start, end in zip(starts, ends)]
    if not durations:
        return 0.0
    return float(np.std(durations, ddof=1))

@fantastic
@jsymbolic
@rhythm
@timing
def modal_duration(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """
    The modal raw note duration in quarter notes.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times (in seconds)
    ends : list[float]
        List of note end times (in seconds)
    tempo : float
        Tempo in BPM (beats per minute), default 120.0
    
    Returns
    -------
    float
        Most frequent raw note duration in quarter notes
    
    Note
    ----
    This computes the mode of raw quarter-note durations, so differs
    from jSymbolic `most_common_rhythmic_value`, which uses the modal
    bin in a 12-bin rhythmic-value histogram.
    """
    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return 0.0
    
    return float(get_mode(durations))

@fantastic
@rhythm
@complexity
def duration_entropy(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The zeroth-order base-2 entropy of the duration distribution in quarter notes.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Shannon entropy of note durations
    """
    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return 0.0
    return float(shannon_entropy(durations))

@fantastic
@jsymbolic
@rhythm
@timing
def length(starts: list[float]) -> int:
    """The total number of notes.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    int
        Total number of notes

    """
    return len(starts)

total_number_of_notes = length

@novel
@rhythm
@timing
def number_of_unique_durations(starts: list[float], ends: list[float], tempo: float = 120.0) -> int:
    """The number of unique note durations, measured in quarter notes.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    int
        Number of unique note durations
    """
    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return 0
    return int(len(set(durations)))

@fantastic
@jsymbolic
@rhythm
@timing
def global_duration(melody: Melody) -> float:
    """The total duration in seconds of the melody.

    Parameters
    ----------
    melody : Melody
        Melody object containing MIDI data

    Returns
    -------
    float
        Total duration of the MIDI sequence in seconds

    """
    return melody.total_duration

duration_in_seconds = global_duration

@fantastic
@jsymbolic
@rhythm
@timing
def note_density(melody: Melody) -> float:
    """The average number of notes per second.

    Parameters
    ----------
    melody : Melody
        Melody object containing MIDI data

    Returns
    -------
    float
        Note density (notes per unit time)
    """
    if not melody.starts or not melody.ends or len(melody.starts) == 0 or len(melody.ends) == 0:
        return 0.0
    total_duration = melody.total_duration
    if total_duration == 0:
        return 0.0
    return float(len(melody.starts) / total_duration)

@jsymbolic
@rhythm
@timing
def note_density_variability(melody: Melody) -> float:
    """The standard deviation of note density across 5-second windows.

    Parameters
    ----------
    melody : Melody
        Melody object containing MIDI data
        
    Returns
    -------
    float
        Standard deviation of note density using 5-second windows

    Note
    ----

    Our tests indicate a certain discrepancy between our outputs and JSymbolic's outputs,
    which may be a consequence of JSymbolic's tick-based approach, or perhaps its
    idiosyncratic windowing approach.
    
    """
    if not melody.starts or not melody.ends or len(melody.starts) < 2:
        return 0.0

    # Create 5-second windows and calculate note density for each
    window_duration = 5.0
    window_densities = []
    
    # Start from 0 and create non-overlapping 5-second windows
    start_time = 0.0
    while start_time < melody.total_duration:
        end_time = min(start_time + window_duration, melody.total_duration)
        
        # Count notes that start within this window
        notes_in_window = sum(1.0 for start in melody.starts if start_time <= start < end_time)
        
        # we tried this too, but it just exacerbatated the discrepancy
        # last_onset_in_window = max(start for start in melody.starts if start_time <= start < end_time)
        # last_offset_in_window = max(end for end in melody.ends if start_time <= end < end_time)
        # last_event_in_window = max(last_onset_in_window, last_offset_in_window)

        # window_duration_actual = last_event_in_window - start_time
        window_duration_actual = end_time - start_time


        if window_duration_actual > 0:
            density = notes_in_window / window_duration_actual
            window_densities.append(density)
        
        start_time += window_duration
    
    if len(window_densities) < 2:
        return 0.0
    
    return np.std(window_densities, ddof=1)

@jsymbolic
@rhythm
@timing
def note_density_per_quarter_note(melody: Melody) -> float:
    """The average number of note onsets per unit of time corresponding to an
    idealized quarter note duration based on the tempo.
    
    Parameters
    ----------
    melody : Melody
        Melody object containing MIDI data
        
    Returns
    -------
    float
        Average number of notes per quarter note duration
    """
    if not melody.starts:
        return 0.0

    quarter_note_duration = 60.0 / melody.tempo
    total_duration_seconds = melody.total_duration
    total_duration_quarter_notes = total_duration_seconds / quarter_note_duration

    if total_duration_quarter_notes == 0:
        return 0.0

    return float(len(melody.starts) / total_duration_quarter_notes)

@jsymbolic
@rhythm
@timing
def note_density_per_quarter_note_variability(melody: Melody) -> float:
    """The standard deviation of note density per quarter note.
    
    Divides the melody into 8-quarter-note windows and calculates the standard deviation
    of note density across these windows.
    
    Parameters
    ----------
    melody : Melody
        Melody object containing MIDI data
        
    Returns
    -------
    float
        Standard deviation of note density across windows

    Note
    ----

    Our tests indicate a certain discrepancy between our outputs and JSymbolic's outputs,
    which may be a consequence of JSymbolic's tick-based approach, or perhaps its
    idiosyncratic windowing approach.
    """
    if not melody.starts or not melody.ends or len(melody.starts) < 2:
        return 0.0

    # Use 8-quarter-note windows (matching jSymbolic)
    window_size_quarter_notes = 8.0
    quarter_note_duration = 60.0 / melody.tempo
    window_size_seconds = window_size_quarter_notes * quarter_note_duration
    window_densities = []
    
    # Start from 0 and create non-overlapping 8-quarter-note windows
    start_time = 0.0
    while start_time < melody.total_duration:
        end_time = min(start_time + window_size_seconds, melody.total_duration)
        
        # Count notes that start within this window
        notes_in_window = sum(1 for start in melody.starts if start_time <= start < end_time)
        
        window_duration_seconds = end_time - start_time
        window_duration_quarter_notes = window_duration_seconds / quarter_note_duration
        
        if window_duration_quarter_notes > 0:
            # Calculate note density per quarter note for this window
            density_per_quarter_note = float(notes_in_window) / window_duration_quarter_notes
            window_densities.append(density_per_quarter_note)
        
        start_time += window_size_seconds
    
    if len(window_densities) < 2:
        return 0.0
    
    return np.std(window_densities, ddof=1)

@idyom
@rhythm
@interval
def ioi(starts: list[float]) -> list[float]:
    """The sequence of inter-onset intervals.

    An inter-onset interval (IOI) is the elapsed time from one note onset to the
    next note onset. Unlike note duration, it includes any overlap or silence
    between consecutive notes because it depends only on onset times.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    list[float]
        List of time intervals between consecutive onsets
    """
    intervals = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
    if not intervals:
        return []
    return intervals

@idyom
@jsymbolic
@rhythm
@interval
def ioi_mean(starts: list[float]) -> float:
    """
    The arithmetic mean of inter-onset intervals.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    
    Returns
    -------
    float
        Mean of inter-onset intervals
    
    Note
    ----
    This is called `average_time_between_attacks` in jSymbolic.
    """
    intervals = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
    if not intervals:
        return 0.0
    return float(np.mean(intervals))

average_time_between_attacks = ioi_mean

@idyom
@jsymbolic
@rhythm
@interval
def ioi_standard_deviation(starts: list[float]) -> float:
    """
    The standard deviation of inter-onset intervals.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    
    Returns
    -------
    float
        Standard deviation of inter-onset intervals
    
    Note
    ----
    This is called `variability_of_time_between_attacks` in jSymbolic.
    """
    intervals = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
    if not intervals:
        return 0.0
    return float(np.std(intervals, ddof=1))

variability_of_time_between_attacks = ioi_standard_deviation

@idyom
@rhythm
@interval
def ioi_ratio(starts: list[float]) -> list[float]:
    """The sequence of ratios between successive inter-onset intervals.

    First, consecutive onset times are converted to inter-onset intervals (IOIs).
    Each output value is then ``IOI[i] / IOI[i - 1]``. Values greater than ``1``
    indicate that the current onset gap is longer than the previous one, values
    less than ``1`` indicate a shorter gap, and ``1`` indicates no change.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    list[float]
        Sequence of IOI ratios
    """
    if len(starts) < 3:
        return []

    intervals = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
    if len(intervals) < 2:
        return []

    ratios = [intervals[i] / intervals[i - 1] for i in range(1, len(intervals))]
    return [float(r) for r in ratios]

@novel
@rhythm
@interval
def ioi_ratio_mean(starts: list[float]) -> float:
    """The arithmetic mean of successive inter-onset interval ratios.

    The ratio sequence is computed as ``IOI[i] / IOI[i - 1]`` for each pair of
    adjacent inter-onset intervals. This summary is above ``1`` when IOIs tend to
    lengthen, below ``1`` when they tend to shorten, and close to ``1`` when
    adjacent IOIs tend to have similar lengths.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Mean of IOI ratios
    """
    ratios = ioi_ratio(starts)
    if not ratios:
        return 0.0
    return float(np.mean(ratios))

@novel
@rhythm
@interval
def ioi_ratio_standard_deviation(starts: list[float]) -> float:
    """The sample standard deviation of successive inter-onset interval ratios.

    This feature measures the variability of ``IOI[i] / IOI[i - 1]`` across the
    melody. Larger values indicate less regular proportional change between
    neighboring onset gaps.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Standard deviation of IOI ratios
    """
    ratios = ioi_ratio(starts)
    if not ratios:
        return 0.0
    return float(np.std(ratios, ddof=1))

@novel
@rhythm
@interval
def ioi_range(starts: list[float]) -> float:
    """The range of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Range of inter-onset intervals (0.0 if fewer than two onsets)
    """
    if len(starts) < 2:
        return 0.0
    intervals = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
    return max(intervals) - min(intervals)

@novel
@rhythm
@interval
def ioi_contour(starts: list[float]) -> list[int]:
    """The sequence of IOI-ratio contour values (-1: shorter, 0: same, 1: longer).

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    list[int]
        Sequence of contour values

    Note
    ----
    This contour is computed from ratios of consecutive IOIs, so it requires at
    least three onsets.
    """
    if len(starts) < 3:
        return []

    intervals = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
    if len(intervals) < 2:
        return []

    ratios = [intervals[i] / intervals[i - 1] for i in range(1, len(intervals))]
    contour = [int(np.sign(ratio - 1)) for ratio in ratios]
    return [int(c) for c in contour]

@novel
@rhythm
@interval
def ioi_contour_mean(starts: list[float]) -> float:
    """The arithmetic mean of ordinal IOI contour values.

    IOI contour values are ``-1`` for shorter, ``0`` for unchanged, and ``1`` for
    longer successive inter-onset intervals. The mean summarizes the balance of
    lengthening versus shortening onset gaps.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Mean of contour values
    """
    contour = ioi_contour(starts)
    if not contour:
        return 0.0
    return float(np.mean(contour))

@novel
@rhythm
@interval
def ioi_contour_standard_deviation(starts: list[float]) -> float:
    """The sample standard deviation of ordinal IOI contour values.

    IOI contour values encode whether successive inter-onset intervals shorten,
    stay the same, or lengthen. This feature measures how variable those ordinal
    changes are across the melody.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Standard deviation of contour values
    """
    contour = ioi_contour(starts)
    if not contour:
        return 0.0
    return float(np.std(contour, ddof=1))

@jsymbolic
@rhythm
@timing
def duration_histogram(starts: list[float], ends: list[float], tempo: float = 120.0) -> dict:
    """A histogram of note durations in quarter notes.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    dict
        Histogram of note durations
    """
    # we use the simplified output once more
    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return {}
    num_durations = max(1, len(set(durations)))
    return histogram_bins(durations, num_durations)


@jsymbolic
@rhythm
@timing
def range_of_rhythmic_values(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The range of rhythmic values located within the 12-bin PPQN-based histogram. Durations are 
    converted to quarter notes and mapped to 12 fixed rhythmic bins using midpoints. The
    returned value is the difference between the highest and lowest non-empty bins.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Range in bins (int cast to float), 0 if no durations present
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)

    hist = rhythmic_value_histogram_object.histogram
    lowest = None
    highest = None
    for i in range(12):
        if hist.get(i, 0.0) > 0.0:
            lowest = i
            break
    for i in range(11, -1, -1):
        if hist.get(i, 0.0) > 0.0:
            highest = i
            break

    if lowest is None or highest is None:
        return 0.0

    return float(highest - lowest)


@jsymbolic
@rhythm
@timing
def number_of_different_rhythmic_values_present(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The number of distinct rhythmic value bins that are present in the melody (non-zero).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Count of non-zero bins as a float (0.0 if no durations)
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    count = 0
    for i in range(12):
        if hist.get(i, 0.0) > 0.0:
            count += 1

    return float(count)


@jsymbolic
@rhythm
@timing
def number_of_common_rhythmic_values_present(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The number of rhythmic value bins with normalized proportion >= 0.15.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Count of bins with mass >= 0.15 as a float (0.0 if no durations)
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    count = 0
    for i in range(12):
        if hist.get(i, 0.0) >= 0.15:
            count += 1

    return float(count)


@jsymbolic
@rhythm
@timing
def prevalence_of_very_short_rhythmic_values(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The sum of the two shortest rhythmic bins (indexes 0 and 1).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Proportion in [0, 1] for bins 0 and 1 combined (0.0 if no durations)
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    return float(hist.get(0, 0.0) + hist.get(1, 0.0))


@jsymbolic
@rhythm
@timing
def prevalence_of_short_rhythmic_values(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The sum of the three shortest rhythmic bins (indexes 0, 1, and 2).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Proportion in [0, 1] for bins 0, 1 and 2 combined (0.0 if no durations)

    Note
    ----
    Rhythmic-bin families overlap by construction in jSymbolic 
    (e.g., short/medium/long), so these prevalence values are not mutually
    exclusive and can sum to more than 1.0.
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    return float(hist.get(0, 0.0) + hist.get(1, 0.0) + hist.get(2, 0.0))


@jsymbolic
@rhythm
@timing
def prevalence_of_medium_rhythmic_values(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The sum of rhythmic bins 2 to 6 (8th notes to half notes).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Proportion in [0, 1] for bins 2..6 combined (0.0 if no durations)

    Note
    ----
    Rhythmic-bin families overlap by construction in jSymbolic 
    (e.g., short/medium/long), so these prevalence values are not mutually
    exclusive and can sum to more than 1.0.
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    return float(
        hist.get(2, 0.0)
        + hist.get(3, 0.0)
        + hist.get(4, 0.0)
        + hist.get(5, 0.0)
        + hist.get(6, 0.0)
    )


@jsymbolic
@rhythm
@timing
def prevalence_of_long_rhythmic_values(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The sum of rhythmic bins 6 to 11 (half notes to dotted double whole notes or more).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Proportion in [0, 1] for bins 6 to 11 combined (0.0 if no durations)

    Note
    ----
    Rhythmic-bin families overlap by construction in jSymbolic 
    (e.g., short/medium/long), so these prevalence values are not mutually
    exclusive and can sum to more than 1.0.
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    return float(
        hist.get(6, 0.0)
        + hist.get(7, 0.0)
        + hist.get(8, 0.0)
        + hist.get(9, 0.0)
        + hist.get(10, 0.0)
        + hist.get(11, 0.0)
    )


@jsymbolic
@rhythm
@timing
def prevalence_of_very_long_rhythmic_values(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The sum of rhythmic bins 9 to 11 (dotted whole notes to dotted double whole notes or more).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Proportion in [0, 1] for bins 9..11 combined (0.0 if no durations)
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    return float(
        hist.get(9, 0.0)
        + hist.get(10, 0.0)
        + hist.get(11, 0.0)
    )


@jsymbolic
@rhythm
@timing
def prevalence_of_dotted_notes(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The sum of dotted rhythmic bins: 3, 5, 7, 9, 11.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Proportion in [0, 1] for dotted bins combined (0.0 if no durations)
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    return float(
        hist.get(3, 0.0)
        + hist.get(5, 0.0)
        + hist.get(7, 0.0)
        + hist.get(9, 0.0)
        + hist.get(11, 0.0)
    )


@jsymbolic
@rhythm
@timing
def shortest_rhythmic_value(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The shortest quantized (non-zero) rhythmic-bin value (in quarter notes).
    
    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Shortest quantized rhythmic-bin ideal value in quarter notes (0.0 if empty)

    Note
    ----
    This returns the ideal value of the shortest occupied histogram bin, not
    the raw minimum note duration.
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rvh.histogram
    ideals = rvh.bin_values_quarter_notes()
    for i in range(12):
        if hist.get(i, 0.0) > 0.0:
            return float(ideals[i])
    return 0.0


@jsymbolic
@rhythm
@timing
def longest_rhythmic_value(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The longest quantized rhythmic-bin value (in quarter notes).
    
    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Longest quantized rhythmic-bin ideal value in quarter notes (0.0 if empty)

    Note
    ----
    This returns the ideal value of the longest occupied histogram bin, not the
    raw maximum note duration.
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rvh.histogram
    ideals = rvh.bin_values_quarter_notes()
    for i in range(11, -1, -1):
        if hist.get(i, 0.0) > 0.0:
            return float(ideals[i])
    return 0.0


@jsymbolic
@rhythm
@timing
def mean_rhythmic_value(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The mean quantized rhythmic value in quarter notes.
    
    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Weighted mean rhythmic-bin ideal value in quarter notes (0.0 if empty)

    Note
    ----
    Uses histogram-bin ideal values rather than raw note durations.
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rvh.histogram
    ideals = rvh.bin_values_quarter_notes()
    weights = [hist.get(i, 0.0) for i in range(12)]
    total = sum(weights)
    if total == 0.0:
        return 0.0
    mean_val = sum(ideals[i] * w for i, w in enumerate(weights)) / total
    return float(mean_val)


@jsymbolic
@rhythm
@timing
def most_common_rhythmic_value(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """
    The modal quantized rhythmic value (in quarter notes).
    
    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    
    Returns
    -------
    float
        Modal rhythmic-bin ideal value in quarter notes (0.0 if empty or all-zero)
    
    Note
    ----
    Uses rhythmic-value mode from a 12-bin histogram. Differs from `modal_duration`, which computes a raw-duration mode in quarter-note units.
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rvh.histogram
    ideals = rvh.bin_values_quarter_notes()
    # Choose the smallest index in case of ties
    max_val = -1.0
    max_idx = 0
    for i in range(12):
        val = hist.get(i, 0.0)
        if val > max_val:
            max_val = val
            max_idx = i
    return float(ideals[max_idx]) if max_val > 0.0 else 0.0


@jsymbolic
@rhythm
@timing
def prevalence_of_most_common_rhythmic_value(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The proportion (0.0 - 1.0) of the modal rhythmic bin.
    
    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Proportion (0.0 - 1.0) of the modal rhythmic bin (0.0 if empty)
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rvh.histogram
    max_val = 0.0
    for i in range(12):
        max_val = max(max_val, hist.get(i, 0.0))
    return float(max_val)


@jsymbolic
@rhythm
@timing
def relative_prevalence_of_most_common_rhythmic_values(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The ratio of the second-most-common rhythmic bin to the most common bin.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Ratio of the second-most-common rhythmic bin to the most common bin (0.0 if empty)
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rvh.histogram

    # Convert to ordered list for deterministic tie-breaking by smaller index
    values = [hist.get(i, 0.0) for i in range(12)]
    if not values:
        return 0.0

    most_idx = 0
    for i in range(1, 12):
        if values[i] > values[most_idx]:
            most_idx = i
    second_idx = None
    for i in range(12):
        if i == most_idx:
            continue
        if second_idx is None or values[i] > values[second_idx]:
            second_idx = i

    most_val = values[most_idx]
    second_val = 0.0 if second_idx is None else values[second_idx]

    if most_val == 0.0:
        return 0.0
    return float(second_val / most_val)


@jsymbolic
@rhythm
@timing
def difference_between_most_common_rhythmic_values(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The absolute difference in bins between most and second most common rhythmic values.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Absolute difference in bins between most and second most common rhythmic values (0.0 if empty)
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rvh.histogram
    values = [hist.get(i, 0.0) for i in range(12)]

    most_idx = 0
    for i in range(1, 12):
        if values[i] > values[most_idx]:
            most_idx = i

    second_idx = None
    for i in range(12):
        if i == most_idx:
            continue
        if second_idx is None or values[i] > values[second_idx]:
            second_idx = i

    if values[most_idx] == 0.0 or second_idx is None:
        return 0.0

    return float(abs(most_idx - second_idx))

def _rhythmic_run_lengths(starts: list[float], ends: list[float], tempo: float = 120.0) -> List[int]:
    """Helper function to compute run lengths of identical rhythmic bins for a melody."""
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return []
    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    bin_sequence: List[int] = [rvh.map_quarter_notes_to_bin_index(d) for d in durations_qn]
    if not bin_sequence:
        return []
    run_lengths: List[int] = []
    current_run = 1
    for i in range(1, len(bin_sequence)):
        if bin_sequence[i] == bin_sequence[i - 1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    return run_lengths

@jsymbolic
@rhythm
@timing
def mean_rhythmic_value_run_length(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The mean run length of identical rhythmic values across the melody. Run length is the number of consecutive 
    notes with the same rhythmic value.

    Returns 0.0 if there are fewer than 1 notes.
    """
    runs = _rhythmic_run_lengths(starts, ends, tempo)
    if not runs:
        return 0.0
    return float(np.mean(runs))

@jsymbolic
@rhythm
@timing
def median_rhythmic_value_run_length(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The median run length of identical rhythmic values across the melody. Run length is the number of consecutive 
    notes with the same rhythmic value."""
    runs = _rhythmic_run_lengths(starts, ends, tempo)
    if not runs:
        return 0.0
    return float(np.median(runs))


@jsymbolic
@rhythm
@timing
def variability_in_rhythmic_value_run_lengths(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The standard deviation of rhythmic value run lengths. Run length is the number of consecutive 
    notes with the same rhythmic value."""
    runs = _rhythmic_run_lengths(starts, ends, tempo)
    if not runs or len(runs) == 1:
        return 0.0
    return float(np.std(runs, ddof=1))


def _rhythmic_value_offsets(starts: list[float], ends: list[float], tempo: float = 120.0) -> List[float]:
    """Helper function to compute absolute offsets (in quarter notes) from nearest ideal value for each note."""
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return []
    rvh = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    ideals = rvh.bin_values_quarter_notes()
    offsets: List[float] = []
    for d in durations_qn:
        bin_idx = rvh.map_quarter_notes_to_bin_index(d)
        ideal = ideals[bin_idx]
        offsets.append(abs(float(d) - float(ideal)))
    return offsets


@jsymbolic
@rhythm
@timing
def mean_rhythmic_value_offset(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The mean quantized offset from the nearest ideal rhythmic value (in quarter notes).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Mean quantized offset from the nearest ideal rhythmic value (in quarter notes) (0.0 if no durations)
    """
    offsets = _rhythmic_value_offsets(starts, ends, tempo)
    if not offsets:
        return 0.0
    return float(np.mean(offsets))


@jsymbolic
@rhythm
@timing
def median_rhythmic_value_offset(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The median quantized offset from the nearest ideal rhythmic value (in quarter notes).
    
    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Median quantized offset from the nearest ideal rhythmic value (in quarter notes) (0.0 if no durations)
    """
    offsets = _rhythmic_value_offsets(starts, ends, tempo)
    if not offsets:
        return 0.0
    return float(np.median(offsets))


@jsymbolic
@rhythm
@timing
def variability_of_rhythmic_value_offsets(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The standard deviation of rhythmic value offsets (in quarter notes).
    
    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)

    Returns
    -------
    float
        Standard deviation of rhythmic value offsets (in quarter notes) (0.0 if no durations)
    """
    offsets = _rhythmic_value_offsets(starts, ends, tempo)
    if not offsets or len(offsets) == 1:
        return 0.0
    return float(np.std(offsets, ddof=1))


def _silent_run_lengths_qn(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
    min_qn_threshold: float = 0.1,
) -> list[float]:
    """Return list of complete rest lengths in quarter notes, filtered by threshold. By complete rest,
    we mean that there is nothing sounding at all at any time during the rest.

    Discretizes time to ticks (constant tempo), builds a per-tick pitched-activity mask,
    adds a 1-quarter-note silent tail, collects silent run lengths, converts to quarter
    notes, and filters out runs shorter than min_qn_threshold.
    """
    if not starts or not ends or len(starts) != len(ends):
        return []

    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))

    start_ticks = [to_ticks(s) for s in starts]
    end_ticks = [to_ticks(e) for e in ends]
    if not end_ticks:
        return []

    duration_in_ticks = max(0, max(end_ticks))
    if duration_in_ticks <= 0:
        return []

    total_ticks = duration_in_ticks + int(ppqn)

    active = [False] * total_ticks
    for s_tick, e_tick in zip(start_ticks, end_ticks):
        if e_tick <= s_tick:
            continue
        a = max(0, min(total_ticks - 1, s_tick))
        b = max(0, min(total_ticks, e_tick))
        for t in range(a, b):
            active[t] = True

    runs_ticks: list[int] = []
    current = 0
    for t in range(total_ticks):
        if not active[t]:
            current += 1
        else:
            if current > 0:
                runs_ticks.append(current)
                current = 0
    if current > 0:
        runs_ticks.append(current)

    if not runs_ticks:
        return []

    qn_per_tick = seconds_per_tick / (60.0 / float(tempo))
    runs_qn = [(rl * qn_per_tick) for rl in runs_ticks]
    return [rl for rl in runs_qn if rl >= float(min_qn_threshold)]


@jsymbolic
@rhythm
@timing
def complete_rests_fraction(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The fraction of the total duration during which no pitched notes are sounding.
    
    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Fraction of total duration during which no pitched notes are sounding (0.0 if no durations)

    Note
    ----
    This feature includes all complete silent runs (including those shorter than
    0.1 quarter notes), whereas other complete-rest summary statistics apply a
    minimum 0.1 quarter-note threshold.
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    runs_qn = _silent_run_lengths_qn(starts, ends, tempo=tempo, ppqn=ppqn, min_qn_threshold=0.0)

    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    end_ticks = [to_ticks(e) for e in ends]
    if not end_ticks:
        return 0.0
    duration_in_ticks = max(0, max(end_ticks))
    if duration_in_ticks <= 0:
        return 0.0
    total_ticks = duration_in_ticks + int(ppqn)
    qn_per_tick = seconds_per_tick / (60.0 / float(tempo))
    total_qn = total_ticks * qn_per_tick

    rest_qn = sum(runs_qn) if runs_qn else 0.0
    if total_qn <= 0.0:
        return 0.0
    return float(rest_qn / total_qn)


@jsymbolic
@rhythm
@timing
def longest_complete_rest(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The longest uninterrupted complete rest in quarter-note units (ignoring rests shorter than 0.1 QN).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Longest uninterrupted complete rest in quarter-note units (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    runs_qn = _silent_run_lengths_qn(starts, ends, tempo=tempo, ppqn=ppqn, min_qn_threshold=0.1)
    if not runs_qn:
        return 0.0
    return float(max(runs_qn))

@jsymbolic
@rhythm
@timing
def mean_complete_rest_duration(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The mean duration of complete rests in quarter-note units (ignoring rests shorter than 0.1 QN).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Mean duration of complete rests in quarter-note units (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    runs_qn = _silent_run_lengths_qn(starts, ends, tempo=tempo, ppqn=ppqn, min_qn_threshold=0.1)
    if not runs_qn:
        return 0.0
    return float(np.mean(runs_qn))

@jsymbolic
@rhythm
@timing
def median_complete_rest_duration(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The median duration of complete rests in quarter-note units (ignoring rests shorter than 0.1 QN).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Median duration of complete rests in quarter-note units (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    runs_qn = _silent_run_lengths_qn(starts, ends, tempo=tempo, ppqn=ppqn, min_qn_threshold=0.1)
    if not runs_qn:
        return 0.0
    return float(np.median(runs_qn))

@jsymbolic
@rhythm
@timing
def variability_of_complete_rest_durations(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The standard deviation of complete rest durations in quarter notes (ignoring rests shorter than 0.1 QN).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Standard deviation of complete rest durations in quarter notes (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    runs_qn = _silent_run_lengths_qn(starts, ends, tempo=tempo, ppqn=ppqn, min_qn_threshold=0.1)
    if len(runs_qn) < 2:
        return 0.0
    return float(np.std(runs_qn, ddof=1))

def _calculate_thresholded_peak_table(values: list[float]) -> list[list[float]]:
    """Build jSymbolic-style thresholded peak table (n x 3) from a histogram array.

    Columns thresholds:
    - col 0: > 0.1
    - col 1: > 0.01
    - col 2: > 0.3 * max(hist)
    Then suppress adjacent peaks keeping only the larger in any adjacent pair per column.
    """
    n = len(values)
    if n == 0:
        return []
    table = [[0.0, 0.0, 0.0] for _ in range(n)]
    highest = values[int(np.argmax(values))] if n > 0 else 0.0
    for i in range(n):
        v = float(values[i])
        if v > 0.1:
            table[i][0] = v
        if v > 0.01:
            table[i][1] = v
        if highest > 0.0 and v > 0.3 * highest:
            table[i][2] = v
    for i in range(1, n):
        for j in range(3):
            if table[i][j] > 0.0 and table[i - 1][j] > 0.0:
                if table[i][j] > table[i - 1][j]:
                    table[i - 1][j] = 0.0
                else:
                    table[i][j] = 0.0
    return table

@lru_cache(maxsize=256)
def _get_beat_histogram_values_from_ticks(
    start_ticks: tuple[int, ...],
    end_ticks: tuple[int, ...],
    tempo: float,
    ppqn: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """LRU-cached beat histogram arrays (normal, standardized) from tick inputs.
    This is cached to avoid recomputing the beat histogram for the same start and end ticks, or worse, the autocorrelation.
    We've optimised the beat histogram computation to be more efficient, but caching still makes sense to me at this time."""
    if not end_ticks:
        return tuple(), tuple()
    duration_in_ticks = max(0, max(end_ticks))
    if duration_in_ticks <= 0:
        return tuple(), tuple()
    total_ticks = duration_in_ticks + int(ppqn)

    rhythm_score: list[int] = [0] * (total_ticks + 1)
    for tick in start_ticks:
        if 0 <= tick < len(rhythm_score):
            rhythm_score[tick] += 1

    mean_ticks_per_second = float(ppqn) * (float(tempo) / 60.0)
    bh = create_beat_histogram(
        rhythm_score=rhythm_score,
        mean_ticks_per_second=mean_ticks_per_second,
        ppqn=ppqn,
    )
    return tuple(bh.beat_histogram), tuple(bh.beat_histogram_120_bpm_standardized)

@lru_cache(maxsize=256)
def _compute_beat_histogram_tables(
    starts: tuple[float, ...], ends: tuple[float, ...], tempo: float, ppqn: int
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Compute thresholded peak tables for normal and 120-BPM-standardized beat histograms."""
    if not starts or not ends or len(starts) != len(ends):
        return (), ()

    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))

    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)

    normal_vals, std_vals = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    normal_table = _calculate_thresholded_peak_table(list(normal_vals))
    std_table = _calculate_thresholded_peak_table(list(std_vals))
    return tuple(tuple(row) for row in normal_table), tuple(tuple(row) for row in std_table)

def _count_strong_pulses(table: list[list[float]], column_index: int = 0) -> float:
    """Count peaks in BPM bins 40-200 whose thresholded value in the given column is non-zero."""
    if not table:
        return 0.0
    n = len(table)
    min_bpm = 40
    max_bpm = min(200, n - 1)
    count = 0
    for b in range(min_bpm, max_bpm + 1):
        if table[b][column_index] > 0.0:
            count += 1
    return float(count)

@jsymbolic
@rhythm
@timing
def strongest_rhythmic_pulse(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The bin index (BPM) of the maximum beat histogram magnitude.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Bin index (BPM) of the maximum beat histogram magnitude (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    values, _ = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values:
        return 0.0

    return float(int(np.argmax(values)))

@jsymbolic
@rhythm
@timing
def strongest_rhythmic_pulse_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The bin index (BPM) of the maximum in the 120-BPM standardized beat histogram.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Bin index (BPM) of the maximum beat histogram magnitude (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    _, values = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values:
        return 0.0

    return float(int(np.argmax(values)))

@jsymbolic
@rhythm
@timing
def second_strongest_rhythmic_pulse(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The bin index (BPM) of the second-highest magnitude in the beat histogram.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Bin index (BPM) of the second-highest magnitude in the beat histogram (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    values, _ = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values or len(values) < 2:
        return 0.0

    max_idx = int(np.argmax(values))
    max_val = values[max_idx]

    values_list = list(values)
    values_list[max_idx] = 0.0
    second_max_idx = int(np.argmax(values_list))

    return float(second_max_idx)

@jsymbolic
@rhythm
@timing
def second_strongest_rhythmic_pulse_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The bin index (BPM) of the second-highest magnitude in the 120-BPM standardized beat histogram.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Bin index (BPM) of the second-highest magnitude in the 120-BPM standardized beat histogram (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    _, values = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values or len(values) < 2:
        return 0.0

    max_idx = int(np.argmax(values))
    max_val = values[max_idx]

    values_list = list(values)
    values_list[max_idx] = 0.0
    second_max_idx = int(np.argmax(values_list))

    return float(second_max_idx)

@jsymbolic
@rhythm
@timing
def harmonicity_of_two_strongest_rhythmic_pulses(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The ratio of higher to lower bin index of the two strongest rhythmic pulses.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Ratio of higher to lower bin index of the two strongest rhythmic pulses
        (0.0 if no durations).

    Note
    ----
    The first peak is selected from the raw beat histogram 
    and the second peak is selected from the thresholded
    peak table (column 1), excluding the first peak bin.
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0

    values, _ = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values:
        return 0.0

    normal_table, _ = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    if not normal_table:
        return 0.0

    # Find the bin with the highest magnitude from regular beat histogram
    max_value = 0.0
    max_idx = 1
    for bin in range(len(values)):
        if values[bin] > max_value:
            max_value = values[bin]
            max_idx = bin

    # Find the bin with the second highest magnitude from thresholded table column 1
    second_highest_bin_magnitude = 0.0
    second_max_idx = 1
    for bin in range(len(normal_table)):
        if (len(normal_table[bin]) > 1 and 
            normal_table[bin][1] > second_highest_bin_magnitude and 
            bin != max_idx):
            second_highest_bin_magnitude = normal_table[bin][1]
            second_max_idx = bin

    # Calculate the feature value
    if second_max_idx == 0 or max_idx == 0:
        value = 0.0
    elif max_idx > second_max_idx:
        value = float(max_idx) / float(second_max_idx)
    else:
        value = float(second_max_idx) / float(max_idx)
    
    return value

@jsymbolic
@rhythm
@timing
def harmonicity_of_two_strongest_rhythmic_pulses_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The ratio of higher to lower bin index of the two strongest rhythmic pulses (120-BPM standardized histogram).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Ratio of higher to lower bin index of the two strongest rhythmic pulses
        (120-BPM standardized histogram) (0.0 if no durations).

    Note
    ----
    The first peak is selected from the standardized histogram values 
    from the standardized histogram values and the second peak is selected from the
    standardized thresholded peak table (column 1), excluding the first peak bin.
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    
    _, values = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values:
        return 0.0

    _, std_table = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    if not std_table:
        return 0.0
    
    # Find the bin with the highest magnitude from tempo standardized beat histogram
    max_value = 0.0
    max_idx = 1
    for bin in range(len(values)):
        if values[bin] > max_value:
            max_value = values[bin]
            max_idx = bin
    
    # Find the bin with the second highest magnitude from thresholded table column 1
    second_highest_bin_magnitude = 0.0
    second_max_idx = 1
    for bin in range(len(std_table)):
        if (len(std_table[bin]) > 1 and 
            std_table[bin][1] > second_highest_bin_magnitude and 
            bin != max_idx):
            second_highest_bin_magnitude = std_table[bin][1]
            second_max_idx = bin
    
    # Calculate the feature value
    if second_max_idx == 0 or max_idx == 0:
        value = 0.0
    elif max_idx > second_max_idx:
        value = float(max_idx) / float(second_max_idx)
    else:
        value = float(second_max_idx) / float(max_idx)
    
    return value

@jsymbolic
@rhythm
@timing
def strength_of_strongest_rhythmic_pulse(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The magnitude of the beat histogram bin with the highest magnitude.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Magnitude of the beat histogram bin with the highest magnitude (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    values, _ = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values:
        return 0.0
    return float(max(values))

@jsymbolic
@rhythm
@timing
def strength_of_strongest_rhythmic_pulse_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The magnitude of the tempo-standardized beat histogram bin with the highest magnitude.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Magnitude of the tempo-standardized beat histogram bin with the highest magnitude (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    _, values = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values:
        return 0.0
    return float(max(values))

@jsymbolic
@rhythm
@timing
def strength_of_second_strongest_rhythmic_pulse(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The magnitude of the beat histogram bin with the second-highest magnitude.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Magnitude of the beat histogram bin with the second-highest magnitude (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    values, _ = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values or len(values) < 2:
        return 0.0
    
    # Find the two highest values
    max_val = max(values)
    values_list = list(values)
    values_list[values_list.index(max_val)] = 0.0
    second_max_val = max(values_list)
    
    return float(second_max_val)

@jsymbolic
@rhythm
@timing
def strength_of_second_strongest_rhythmic_pulse_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The magnitude of the tempo-standardized beat histogram bin with the second-highest magnitude.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Magnitude of the tempo-standardized beat histogram bin with the second-highest magnitude (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    _, values = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values or len(values) < 2:
        return 0.0
    
    # Find the two highest values
    max_val = max(values)
    values_list = list(values)
    values_list[values_list.index(max_val)] = 0.0
    second_max_val = max(values_list)
    
    return float(second_max_val)

@jsymbolic
@rhythm
@timing
def strength_ratio_of_two_strongest_rhythmic_pulses(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """Ratio of the magnitude of the strongest to second-strongest rhythmic pulse.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Ratio of the magnitude of the strongest to second-strongest rhythmic pulse (0.0 if no durations)
    """
    strongest_strength = strength_of_strongest_rhythmic_pulse(starts, ends, tempo, ppqn)
    second_strongest_strength = strength_of_second_strongest_rhythmic_pulse(starts, ends, tempo, ppqn)
    
    if second_strongest_strength == 0:
        return 0.0
    return float(strongest_strength) / float(second_strongest_strength)

@jsymbolic
@rhythm
@timing
def strength_ratio_of_two_strongest_rhythmic_pulses_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """Ratio of the magnitude of the strongest to second-strongest rhythmic pulse (120-BPM standardized histogram).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Ratio of the magnitude of the strongest to second-strongest rhythmic pulse (120-BPM standardized histogram) (0.0 if no durations)
    """
    strongest_strength = strength_of_strongest_rhythmic_pulse_tempo_standardized(starts, ends, tempo, ppqn)
    second_strongest_strength = strength_of_second_strongest_rhythmic_pulse_tempo_standardized(starts, ends, tempo, ppqn)
    
    if second_strongest_strength == 0:
        return 0.0
    return float(strongest_strength) / float(second_strongest_strength)

@jsymbolic
@rhythm
@timing
def combined_strength_of_two_strongest_rhythmic_pulses(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """Sum of the magnitudes of the two strongest rhythmic pulses.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Sum of the magnitudes of the two strongest rhythmic pulses (0.0 if no durations)
    """
    strongest_strength = strength_of_strongest_rhythmic_pulse(starts, ends, tempo, ppqn)
    second_strongest_strength = strength_of_second_strongest_rhythmic_pulse(starts, ends, tempo, ppqn)
    
    return float(strongest_strength) + float(second_strongest_strength)

@jsymbolic
@rhythm
@timing
def combined_strength_of_two_strongest_rhythmic_pulses_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """Sum of the magnitudes of the two strongest rhythmic pulses using tempo-standardized histogram.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Sum of the magnitudes of the two strongest rhythmic pulses using tempo-standardized histogram (0.0 if no durations)
    """
    strongest_strength = strength_of_strongest_rhythmic_pulse_tempo_standardized(starts, ends, tempo, ppqn)
    second_strongest_strength = strength_of_second_strongest_rhythmic_pulse_tempo_standardized(starts, ends, tempo, ppqn)
    
    return float(strongest_strength) + float(second_strongest_strength)

@jsymbolic
@rhythm
@timing
def rhythmic_variability(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The standard deviation of the beat histogram bin magnitudes, excluding the first 40 bins.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Standard deviation of the beat histogram bin magnitudes, excluding the first 40 bins (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    values, _ = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values or len(values) <= 40:
        return 0.0
    
    # Exclude the first 40 bins (BPM 0-39)
    reduced_values = values[40:]
    if len(reduced_values) < 2:
        return 0.0
    
    return float(np.std(reduced_values, ddof=1))

@jsymbolic
@rhythm
@timing
def rhythmic_variability_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The standard deviation of the tempo-standardized beat histogram bin magnitudes, excluding the first 40 bins.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Standard deviation of the tempo-standardized beat histogram bin magnitudes, excluding the first 40 bins (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    _, values = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values or len(values) <= 40:
        return 0.0
    
    # Exclude the first 40 bins (BPM 0-39)
    reduced_values = values[40:]
    if len(reduced_values) < 2:
        return 0.0
    
    return float(np.std(reduced_values, ddof=1))

@jsymbolic
@rhythm
@timing
def rhythmic_looseness(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The average width of beat histogram peaks. Width is defined as the distance between points at 30% of the peak height.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Average width of beat histogram peaks (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    table, _ = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    if not table:
        return 0.0

    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    values, _ = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values:
        return 0.0
    
    # Find peaks with magnitude >= 30% of highest peak (column 2 in thresholded table)
    peak_bins = []
    for bin_idx in range(len(table)):
        if table[bin_idx][2] > 0.001:
            peak_bins.append(bin_idx)
    
    if not peak_bins:
        return 0.0

    widths = []
    for peak_bin in peak_bins:
        if peak_bin >= len(values):
            continue

        # 30% of this peak's height
        limit_value = 0.3 * values[peak_bin]
        
        # Find left limit
        left_index = 0
        i = peak_bin
        while i >= 0:
            if values[i] < limit_value:
                break
            left_index = i
            i -= 1
        
        # Find right limit
        right_index = len(values) - 1
        i = peak_bin
        while i < len(values):
            if values[i] < limit_value:
                break
            right_index = i
            i += 1
        
        # Calculate width (in BPM bins)
        width = float(right_index - left_index)
        widths.append(width)

    if not widths:
        return 0.0

    return float(np.mean(widths))

@jsymbolic
@rhythm
@timing
def rhythmic_looseness_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The average width of beat histogram peaks using tempo-standardized histogram. Width is defined as the distance between points at 30% of the peak height.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Average width of beat histogram peaks using tempo-standardized histogram (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    _, table = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    if not table:
        return 0.0
    
    seconds_per_tick = (60.0 / float(tempo)) / float(ppqn)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in starts)
    end_ticks = tuple(to_ticks(e) for e in ends)
    if not end_ticks:
        return 0.0
    _, values = _get_beat_histogram_values_from_ticks(start_ticks, end_ticks, float(tempo), int(ppqn))
    if not values:
        return 0.0
    
    # Find peaks with magnitude >= 30% of highest peak (column 2 in thresholded table)
    peak_bins = []
    for bin_idx in range(len(table)):
        if table[bin_idx][2] > 0.001:
            peak_bins.append(bin_idx)
    
    if not peak_bins:
        return 0.0
    
    widths = []
    for peak_bin in peak_bins:
        if peak_bin >= len(values):
            continue
            
        # 30% of this peak's height
        limit_value = 0.3 * values[peak_bin]
        
        # Find left limit
        left_index = 0
        i = peak_bin
        while i >= 0:
            if values[i] < limit_value:
                break
            left_index = i
            i -= 1
        
        # Find right limit
        right_index = len(values) - 1
        i = peak_bin
        while i < len(values):
            if values[i] < limit_value:
                break
            right_index = i
            i += 1
        
        # Calculate width (in BPM bins)
        width = float(right_index - left_index)
        widths.append(width)
    
    if not widths:
        return 0.0
    
    return float(np.mean(widths))

def _is_factor_or_multiple(bin_idx: int, highest_bin: int, multipliers: list[int]) -> bool:
    """Check if bin_idx is a factor or multiple of highest_bin using given multipliers with +/-3 tolerance."""
    for mult in multipliers:
        # Check if bin_idx is a multiple of highest_bin * mult (within tolerance)
        expected = highest_bin * mult
        if abs(bin_idx - expected) <= 3:
            return True
        # Check if bin_idx is a factor of highest_bin (within tolerance)
        if highest_bin % mult == 0:
            expected = highest_bin // mult
            if abs(bin_idx - expected) <= 3:
                return True
        # Also check if highest_bin is a multiple of bin_idx * mult (within tolerance)
        expected = bin_idx * mult
        if abs(highest_bin - expected) <= 3:
            return True
        # And if highest_bin is a factor of bin_idx (within tolerance)
        if bin_idx % mult == 0:
            expected = bin_idx // mult
            if abs(highest_bin - expected) <= 3:
                return True
    return False

@jsymbolic
@rhythm
@timing
def polyrhythms(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The fraction of strong beat-histogram peaks related to the strongest peak.

    Among peaks at least 30% as tall as the maximum, returns the proportion whose bin is
    an integer multiple/factor of the strongest (multipliers 1, 2, 3, 4, 6, 8; ±3 bins).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        ``hits / n_peaks``, or ``0.0`` if there are no qualifying peaks.
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    
    # Get thresholded peak table
    table, _ = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    if not table:
        return 0.0
    
    # Find peaks with magnitude >= 30% of highest peak (column 2 in thresholded table)
    peak_bins = []
    for bin_idx in range(len(table)):
        if table[bin_idx][2] > 0.001:
            peak_bins.append(bin_idx)
    
    if not peak_bins:
        return 0.0
    
    # Find the highest peak
    highest_index = 0
    max_magnitude = 0.0
    for peak_bin in peak_bins:
        if table[peak_bin][2] > max_magnitude:
            max_magnitude = table[peak_bin][2]
            highest_index = peak_bin
    
    # Count peaks that are multiples/factors of the highest peak
    multipliers = [1, 2, 3, 4, 6, 8]
    hits = 0
    
    for peak_bin in peak_bins:
        if _is_factor_or_multiple(peak_bin, highest_index, multipliers):
            hits += 1

    return float(hits) / float(len(peak_bins))

@jsymbolic
@rhythm
@timing
def polyrhythms_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The fraction of strong beat-histogram peaks related to the strongest peak using the tempo-standardized beat histogram.

    Among peaks at least 30% as tall as the maximum, returns the proportion whose bin is
    an integer multiple/factor of the strongest (multipliers 1, 2, 3, 4, 6, 8; ±3 bins).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        ``hits / n_peaks``, or ``0.0`` if there are no qualifying peaks.
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    _, table = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    if not table:
        return 0.0

    # Find peaks with magnitude >= 30% of highest peak (column 2 in thresholded table)
    peak_bins = []
    for bin_idx in range(len(table)):
        if table[bin_idx][2] > 0.001:
            peak_bins.append(bin_idx)
    
    if not peak_bins:
        return 0.0
    
    # Find the highest peak
    highest_index = 0
    max_magnitude = 0.0
    for peak_bin in peak_bins:
        if table[peak_bin][2] > max_magnitude:
            max_magnitude = table[peak_bin][2]
            highest_index = peak_bin
    
    # Count peaks that are multiples/factors of the highest peak
    multipliers = [1, 2, 3, 4, 6, 8]
    hits = 0
    
    for peak_bin in peak_bins:
        if _is_factor_or_multiple(peak_bin, highest_index, multipliers):
            hits += 1

    return float(hits) / float(len(peak_bins))

@jsymbolic
@rhythm
@timing
def number_of_strong_rhythmic_pulses(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The count of BPM bins with strong rhythmic pulses (> 0.1 in the underlying beat histogram).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Count of BPM bins 40-200 with a thresholded strong-pulse value (> 0.1 in the
        underlying beat histogram), or 0.0 if no durations.
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    # Use cached beat histogram tables
    table, _ = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    return _count_strong_pulses(list(table), column_index=0)

@jsymbolic
@rhythm
@timing
def number_of_strong_rhythmic_pulses_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The count of BPM bins with strong pulses (> 0.1 in the underlying standardized beat histogram).

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Count of BPM bins 40-200 with a thresholded strong-pulse value (> 0.1 in the
        underlying standardized histogram), or 0.0 if no durations.
    """
    _, std_table = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    return _count_strong_pulses(list(std_table), column_index=0)

@jsymbolic
@rhythm
@timing
def number_of_moderate_rhythmic_pulses(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The number of beat histogram peaks with normalized magnitudes over 0.01.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Number of beat histogram peaks with normalized magnitudes over 0.01 (0.0 if no durations)
    """
    table, _ = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    return _count_strong_pulses(list(table), column_index=1)

@jsymbolic
@rhythm
@timing
def number_of_moderate_rhythmic_pulses_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The number of tempo-standardized beat histogram peaks with normalized magnitudes over 0.01.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Number of tempo-standardized beat histogram peaks with normalized magnitudes over 0.01 (0.0 if no durations)
    """
    _, std_table = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    return _count_strong_pulses(list(std_table), column_index=1)

@jsymbolic
@rhythm
@timing
def number_of_relatively_strong_rhythmic_pulses(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The number of peaks at least 30% of the max magnitude.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Number of peaks at least 30% of the max magnitude (0.0 if no durations)
    """
    table, _ = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    return _count_strong_pulses(list(table), column_index=2)

@jsymbolic
@rhythm
@timing
def number_of_relatively_strong_rhythmic_pulses_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The number of tempo-standardized peaks at least 30% of the max magnitude.

    Parameters
    ----------
    starts : list[float]
        Note start times (seconds)
    ends : list[float]
        Note end times (seconds)
    tempo : float, optional
        Tempo in BPM (only used to convert seconds to quarter notes)
    ppqn : int, optional
        Pulses per quarter note (MIDI resolution), default 480

    Returns
    -------
    float
        Number of tempo-standardized peaks at least 30% of the max magnitude (0.0 if no durations)
    """
    _, std_table = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    return _count_strong_pulses(list(std_table), column_index=2)

@novel
@rhythm
@interval
def ioi_histogram(starts: list[float]) -> dict:
    """A histogram of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    dict
        Histogram of inter-onset intervals
    """
    intervals = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
    num_intervals = len(set(intervals))
    return histogram_bins(intervals, num_intervals)

@jsymbolic
@rhythm
@timing
def minimum_note_duration(starts: list[float], ends: list[float]) -> float:
    """The minimum note duration in seconds.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Minimum note duration in seconds (0.0 if there are no notes)
    """
    if not starts or not ends:
        return 0.0
    durations = [end - start for start, end in zip(starts, ends)]
    return float(min(durations)) if durations else 0.0

@jsymbolic
@rhythm
@timing
def maximum_note_duration(starts: list[float], ends: list[float]) -> float:
    """The maximum note duration in seconds.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Maximum note duration in seconds
    """
    return max([end - start for start, end in zip(starts, ends)])

@fantastic
@rhythm
@timing
def equal_duration_transitions(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The proportion of duration transitions that are equal in length.
    
    
    Parameters
    ----------
    starts : list[float]
        List of note start times  
    ends : list[float]
        List of note end times
    tempo : float, optional
        Included for API consistency; not used in this calculation.
        
    Returns
    -------
    float
        Proportion of equal duration transitions (0.0 to 1.0)

    Citation
    --------
    Steinbeck (1982)
    """
    ratios = get_duration_ratios(starts, ends)
    if not ratios:
        return 0.0
    
    # Count ratios that equal 1.0 (equal durations)
    equal_count = sum(1 for ratio in ratios if ratio == 1.0)
    
    return equal_count / len(ratios)

@fantastic
@rhythm
@timing
def half_duration_transitions(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The proportion of duration transitions that are halved or doubled.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
    tempo : float, optional
        Included for API consistency; not used in this calculation.

    Returns
    -------
    float
        Proportion of half/double duration transitions (0.0 to 1.0)

    Citation
    --------
    Steinbeck (1982)
    """
    ratios = get_duration_ratios(starts, ends)
    if not ratios:
        return 0.0
    
    # Count ratios that equal 0.5 or round to 2
    half_count = sum(1 for ratio in ratios if ratio == 0.5)
    double_count = sum(1 for ratio in ratios if round(ratio) == 2)
    
    return (half_count + double_count) / len(ratios)

@fantastic
@rhythm
@timing
def dotted_duration_transitions(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The proportion of duration transitions that are dotted.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
    tempo : float, optional
        Included for API consistency; not used in this calculation.
        
    Returns
    -------
    float
        Proportion of dotted duration transitions (0.0 to 1.0)

    Citation
    --------
    Steinbeck (1982)
    """
    ratios = get_duration_ratios(starts, ends)
    if not ratios:
        return 0.0
    
    # Count ratios that equal 1/3 or round to 3
    one_third_count = sum(1 for ratio in ratios if abs(ratio - (1/3)) < 1e-10)
    triple_count = sum(1 for ratio in ratios if round(ratio) == 3)
    
    return (one_third_count + triple_count) / len(ratios)

@jsymbolic
@rhythm
@timing
def amount_of_staccato(starts: list[float], ends: list[float]) -> float:
    """The proportion of notes with a duration shorter than 0.1 seconds.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
        
    Returns
    -------
    float
        Fraction of notes shorter than 0.1 seconds

    Note
    ----
    Though this feature is named `Amount Of Staccato`, it is a
    fixed-duration cutoff statistic rather than symbolic articulation parsing.
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    durations_seconds = [float(end - start) for start, end in zip(starts, ends)]
    if not durations_seconds:
        return 0.0
    short_count = sum(1 for d in durations_seconds if d < 0.1)
    return float(short_count / len(durations_seconds))


# readable alias
short_note_fraction = amount_of_staccato

@midi_toolbox
@rhythm
@complexity
def duration_accent(starts: list[float], ends: list[float], tau: float = 0.5, accent_index: float = 2.0) -> list[float]:
    """Calculate duration accent for each note based on Parncutt (1994).
    Duration accent represents the perceptual salience of notes based on their duration.
    
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
    tau : float, optional
        Saturation duration in seconds, by default 0.5
    accent_index : float, optional
        Minimum discriminable duration parameter, by default 2.0
        
    Citation
    --------
    Parncutt (1994)

    Returns
    -------
    list[float]
        List of duration accent values for each note

    Note
    -----
    The MIDI toolbox implementation uses defaults of 0.5 for tau (saturation duration) 
    and 2.0 for accent_index (minimum discriminable duration).
    """
    return _duration_accent(starts, ends, tau, accent_index)

@midi_toolbox
@rhythm
@complexity
def mean_duration_accent(starts: list[float], ends: list[float], tau: float = 0.5, accent_index: float = 2.0) -> float:
    """The mean duration accent across all notes. Duration accent represents the perceptual salience of notes based on their duration,
    as defined by Parncutt (1994).
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
    tau : float, optional
        Saturation duration in seconds, by default 0.5
    accent_index : float, optional
        Minimum discriminable duration parameter, by default 2.0
        
    Returns
    -------
    float
        Mean duration accent value
    
    Citation
    --------
    Parncutt (1994)
    """
    accents = duration_accent(starts, ends, tau, accent_index)
    if not accents:
        return 0.0
    return float(np.mean(accents))

@midi_toolbox
@rhythm
@complexity
def duration_accent_std(starts: list[float], ends: list[float], tau: float = 0.5, accent_index: float = 2.0) -> float:
    """The standard deviation of duration accents. Duration accent represents the perceptual salience of notes based on their duration,
    as defined by Parncutt (1994).
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
    tau : float, optional
        Saturation duration in seconds, by default 0.5
    accent_index : float, optional
        Minimum discriminable duration parameter, by default 2.0
        
    Returns
    -------
    float
        Standard deviation of duration accent values

    Citation
    --------
    Parncutt (1994)
    """
    accents = duration_accent(starts, ends, tau, accent_index)
    if not accents:
        return 0.0
    return float(np.std(accents, ddof=1))

@midi_toolbox
@rhythm
@timing
def npvi(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The normalized Pairwise Variability Index (nPVI) of note durations in quarter notes.
    The nPVI measures the durational variability of events, originally developed for 
    language research to distinguish stress-timed vs. syllable-timed languages.
    Applied to music by Patel & Daniele (2003) to study the prosodic
    influences on musical rhythm.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
        
    Returns
    -------
    float
        nPVI index value (higher values indicate greater durational variability)

    Citation
    --------
    Patel & Daniele (2003)
    """
    durations = _get_durations(starts, ends, tempo)
    if len(durations) < 2:
        return 0.0
    
    normalized_diffs = []
    for i in range(1, len(durations)):
        prev_dur = durations[i-1]
        curr_dur = durations[i]
        
        if prev_dur + curr_dur == 0:
            normalized_diffs.append(0.0)
        else:
            # Normalized difference: (d1 - d2) / ((d1 + d2) / 2)
            mean_duration = (prev_dur + curr_dur) / 2
            normalized_diff = (prev_dur - curr_dur) / mean_duration
            normalized_diffs.append(abs(normalized_diff))

    if not normalized_diffs:
        return 0.0
    
    npvi_value = (100 / len(normalized_diffs)) * sum(normalized_diffs)
    return float(npvi_value)

@midi_toolbox
@rhythm
@timing
def onset_autocorrelation(
    starts: list[float],
    ends: list[float],
    divisions_per_quarter: int = 4,
    max_lag_quarters: int = 8,
    tempo: float = 120.0,
) -> list[float]:
    """The autocorrelation function of onset times weighted by duration accents.
    This is calculated by weighting the onset times by the duration accents,
    as defined by Parncutt (1994).

    Onsets are converted to quarter-note beats using
    tempo before grid quantization.

    Parameters
    ----------
    starts : list[float]
        Note onset times in seconds
    ends : list[float]
        Note offset times in seconds
    divisions_per_quarter : int, optional
        Grid divisions per quarter note (default 4)
    max_lag_quarters : int, optional
        Maximum lag in quarter notes (default 8)
    tempo : float, optional
        Tempo in BPM (default 120)

    Returns
    -------
    list[float]
        Normalized autocorrelation from lag 0 through ``max_lag_quarters`` quarters

    Citation
    --------
    Parncutt (1994)
    """
    return compute_onset_autocorrelation(
        starts,
        ends,
        divisions_per_quarter=divisions_per_quarter,
        max_lag_quarters=max_lag_quarters,
        tempo=tempo,
    )


@midi_toolbox
@rhythm
@timing
def onset_autocorr_peak(
    starts: list[float],
    ends: list[float],
    divisions_per_quarter: int = 4,
    max_lag_quarters: int = 8,
    tempo: float = 120.0,
) -> float:
    """Maximum onset autocorrelation excluding lag 0."""
    autocorr_values = onset_autocorrelation(
        starts, ends, divisions_per_quarter, max_lag_quarters, tempo
    )
    if len(autocorr_values) <= 1:
        return 0.0
    return float(max(autocorr_values[1:]))

# Tonality Features
def infer_key_from_pitches(
    pitches: list[int],
    algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler"
) -> tuple[Optional[str], Optional[str]]:
    """
    Infer the key of a melody using the specified algorithm.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    algorithm : Literal["krumhansl_schmuckler"], optional
        Key-finding algorithm to use. Default is "krumhansl_schmuckler".
        
    Returns
    -------
    tuple[Optional[str], Optional[str]]
        (key_name, mode) e.g., ("C", "major") or (None, None) if cannot determine
        
    Raises
    ------
    NotImplementedError
        If algorithm is not supported
        
    Citations
    --------
    Krumhansl (1990)
    """
    if algorithm != "krumhansl_schmuckler":
        raise NotImplementedError(
            f"Key-finding algorithm '{algorithm}' is not implemented. "
            f"Currently only 'krumhansl_schmuckler' is supported."
        )
    
    pitch_classes = [p % 12 for p in pitches]
    correlations = compute_tonality_vector(pitch_classes)
    
    if not correlations:
        return None, None
        
    key_string = correlations[0][0]  # e.g., "C major"
    parts = key_string.split()
    key_name = parts[0]
    mode = parts[1] if len(parts) > 1 else "major"
    
    return key_name, mode


def _normalize_key_root(root: str) -> str:
    """Normalize key root names to a canonical display form."""
    if not root:
        return root
    root = root.strip()
    if not root:
        return root
    return root[0].upper() + root[1:]


def _canonical_key_string(key_name: Optional[str], mode: Optional[str]) -> Optional[str]:
    """Return canonical key strings like 'C major' or 'F# minor'."""
    if not key_name or not mode:
        return None
    mode_normalized = mode.strip().lower()
    if mode_normalized not in {"major", "minor"}:
        return None
    root = _normalize_key_root(key_name)
    return f"{root} {mode_normalized}"


def _resolve_key_for_melody(
    melody: Melody,
    key_estimation: Literal["always_read_from_file", "infer_if_necessary", "always_infer"] = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler",
) -> Optional[str]:
    """Resolve a melody key using the requested key-estimation policy."""
    inferred_key = None
    inferred_key_name, inferred_mode = infer_key_from_pitches(
        melody.pitches, algorithm=key_finding_algorithm
    )
    inferred_key = _canonical_key_string(inferred_key_name, inferred_mode)

    key_from_melody = None
    if melody.has_key_signature:
        key_sig = melody.key_signature
        if key_sig and len(key_sig) >= 2:
            key_root = key_sig[0]
            mode = key_sig[1]
            if isinstance(key_root, str) and key_root.endswith("m") and mode == "minor":
                key_root = key_root[:-1]
            key_from_melody = _canonical_key_string(key_root, mode)

    if key_estimation == "always_infer":
        return inferred_key

    if key_estimation == "always_read_from_file":
        if key_from_melody is None:
            raise ValueError(f"No key signature found in MIDI file: {melody.id}")
        return key_from_melody

    # key_estimation == "infer_if_necessary"
    return key_from_melody if key_from_melody is not None else inferred_key


def _tonality_correlations_for_key(correlations: list[tuple[str, float]], key_string: str) -> list[tuple[str, float]]:
    """Return correlations reordered so the requested key is first."""
    normalized_target = key_string.lower()
    for index, (candidate_key, _) in enumerate(correlations):
        if candidate_key.lower() == normalized_target:
            if index == 0:
                return correlations
            return [correlations[index], *correlations[:index], *correlations[index + 1 :]]
    # Fallback keeps downstream scalar helpers deterministic for chosen key.
    return [(key_string, 1.0), *correlations]

@midi_toolbox
@tonality
@pitch
def key(
    melody: Melody,
    key_estimation: Literal["always_read_from_file", "infer_if_necessary", "always_infer"] = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler"
) -> str:
    """
    The key of the melody, either read from the MIDI file or estimated using
    the specified key finding algorithm, depending on the key estimation strategy.
    
    Parameters
    ----------
    melody : Melody
        A melody-features Melody object
    key_estimation : str, optional
        Key estimation strategy, default "infer_if_necessary"
        Can be "always_read_from_file", "infer_if_necessary", or "always_infer"
    key_finding_algorithm : Literal["krumhansl_schmuckler"], optional
        Key-finding algorithm to use when inferring key. Default is "krumhansl_schmuckler".
        
    Returns
    -------
    str
        The key of the melody, in the format "key name major/minor"
    
    Citation
    ----------
    Krumhansl (1990)
    
    Note
    -----
    This feature is named `keyname` in MIDI Toolbox.
    """
    resolved_key = _resolve_key_for_melody(
        melody,
        key_estimation=key_estimation,
        key_finding_algorithm=key_finding_algorithm,
    )
    return resolved_key if resolved_key else "unknown"
 
keyname = key

@fantastic
@tonality
@pitch
def tonalness(pitches: list[int]) -> float:
    """The magnitude of the highest correlation with a precomputed key profile.
    This key profile is established and elaborated on in Krumhansl (1990).

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Magnitude of highest key correlation value

    Citation
    --------
    Krumhansl (1990)
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlation = compute_tonality_vector(pitch_classes)
    if not correlation:
        return 0.0
    return abs(correlation[0][1])

@fantastic
@tonality
@pitch
def tonal_clarity(pitches: list[int]) -> float:
    """The ratio between the top two key correlation values.


    Citation
    ------------------
    Temperley (2007)

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Ratio between highest and second highest key correlation values.
        Returns ``0.0`` when top/second correlations are unavailable or near-zero.
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)

    if len(correlations) < 2:
        return 0.0

    # Get top 2 correlation values
    top_corr = abs(correlations[0][1])
    second_corr = abs(correlations[1][1])

    # Avoid division by zero
    if top_corr <= 0 or second_corr <= 0:
        return 0.0

    return top_corr / second_corr

@fantastic
@tonality
@pitch
def tonal_spike(pitches: list[int]) -> float:
    """The ratio between the highest key correlation and the sum of all other correlations.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Ratio between highest correlation value and sum of all others.
        Returns ``0.0`` when top/other correlations are unavailable or near-zero.
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)

    if len(correlations) < 2:
        return 0.0

    # Get highest correlation and sum of rest
    top_corr = abs(correlations[0][1])
    other_sum = sum(abs(corr[1]) for corr in correlations[1:])

    # Avoid division by zero
    if top_corr <= 0 or other_sum <= 0:
        return 0.0

    return top_corr / other_sum

@novel
@complexity
@pitch
def tonal_entropy(pitches: list[int]) -> float:
    """Zeroth-order base-2 entropy of the 24-key Krumhansl-Schmuckler key correlation distribution. 
    Normalizes the correlation values to a probability mass over all 24 major/minor keys, then computes Shannon entropy.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Entropy in bits; ``0.0`` if all correlations are zero.
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)
    weights = [abs(value) for _, value in correlations]
    return float(distribution_entropy(weights))




@idyom
@tonality
@pitch
def referent(
    melody: Melody,
    key_estimation: Literal["always_read_from_file", "infer_if_necessary", "always_infer"] = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler",
) -> int:
    """
    Calculate the `referent` (pitch-class root) of a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    int
        Pitch class in semitones above C for the resolved key root;
        returns -1 when no key can be resolved.
    """
    resolved_key = _resolve_key_for_melody(
        melody,
        key_estimation=key_estimation,
        key_finding_algorithm=key_finding_algorithm,
    )
    if not resolved_key:
        return -1
    key_name = resolved_key.split()[0]
    key_distances = _get_key_distances()
    return key_distances.get(key_name, -1)

@partitura
@tonality
@pitch
def tonal_tension(
    melody: Melody,
    ws: float = 1.0,
    ss: str = "onset",
    scale_factor: float = SCALE_FACTOR,
    w: np.ndarray = DEFAULT_WEIGHTS,
    alpha: float = ALPHA,
    beta: float = BETA,
    tonality_vector: Optional[list] = None,
    key_estimation: Literal["always_read_from_file", "infer_if_necessary", "always_infer"] = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler"
) -> dict:
    """Computes tension ribbons using the tonal tension algorithm. 
    Provides a means of comparing Chew's spiral array and the tonal tension 
    profiles produced from Herremans and Chew's tension ribbons. This returns a dictionary 
    containing the cloud diameter, cloud momentum, tensile strain, ordered by onset.

    Parameters
    ----------
    melody : Melody
        A melody-features Melody object.
    ws : float, optional
        Window size in beats. Default is 1.0 beat.
    ss : str, optional
        Step size in beats or score position for computing the tonal tension features.
        Default is "onset" (compute at each unique score position).
    scale_factor : float, optional
        Multiplicative scaling factor. Default uses the distance between C and B#.
    w : np.ndarray, optional
        Weights for the chords. Default is [0.516, 0.315, 0.168]
    alpha : float, optional
        Preference for V vs v chord in minor key (0-1). Default is 0.75.
    beta : float, optional
        Preference for iv vs IV in minor key (0-1). Default is 0.75.
    tonality_vector : list, optional
        Pre-computed tonality vector (list of (key_name, correlation) tuples). Default is None.
    key_estimation : Literal["always_read_from_file", "infer_if_necessary", "always_infer"], optional
        Key estimation strategy: "always_read_from_file", "infer_if_necessary", or "always_infer".
        Default is "infer_if_necessary".
    key_finding_algorithm : Literal["krumhansl_schmuckler"], optional
        Key-finding algorithm to use when inferring key. Default is "krumhansl_schmuckler".

    Returns
    -------
    dict
        Dictionary containing tonal tension values keyed by
        `cloud_diameter`, `cloud_momentum`, and `tensile_strain`,
        along with onset-aligned index fields from Partitura.

    Citation
    --------
    Herremans & Chew (2016)
    """
    return estimate_tonaltension(
        melody,
        ws=ws,
        ss=ss,
        scale_factor=scale_factor,
        w=w,
        alpha=alpha,
        beta=beta,
        tonality_vector=tonality_vector,
        key_estimation=key_estimation,
        key_finding_algorithm=key_finding_algorithm
    )

@partitura
@tonality
@pitch
def mean_cloud_diameter(
    melody: Melody,
    ws: float = 1.0,
    ss: str = "onset",
    scale_factor: float = SCALE_FACTOR,
    w: np.ndarray = DEFAULT_WEIGHTS,
    alpha: float = ALPHA,
    beta: float = BETA,
    tonality_vector: Optional[list] = None,
    key_estimation: str = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler"
) -> float:
    """Mean cloud diameter from the tonal tension model. Cloud Diameter provides a
    measure of the maximal tonal distance of the notes in a chord, 
    following the definition in Partitura.
    
    Parameters
    ----------
    melody : Melody
        A melody-features Melody object
    ws : float, optional
        Window size in beats. Default is 1.0 beat.
    ss : str, optional
        Step size or score position for computing the tonal tension features.
        Default is "onset" (compute at each unique score position).
    scale_factor : float, optional
        Multiplicative scaling factor. Default uses the distance between C and B#.
    w : np.ndarray, optional
        Weights for the chords. Default is [0.516, 0.315, 0.168].
    alpha : float, optional
        Preference for V vs v chord in minor key (0-1). Default is 0.75.
    beta : float, optional
        Preference for iv vs IV in minor key (0-1). Default is 0.75.
    tonality_vector : list, optional
        Pre-computed tonality vector (list of (key_name, correlation) tuples). Default is None.
    key_estimation : str, optional
        Key estimation strategy, default "infer_if_necessary"
    key_finding_algorithm : Literal["krumhansl_schmuckler"], optional
        Key-finding algorithm to use when inferring key. Default is "krumhansl_schmuckler".
        
    Returns
    -------
    float
        Mean cloud diameter value
        
    Citation
    --------
    Herremans & Chew (2016)
    """
    tension_dict = estimate_tonaltension(
        melody, ws=ws, ss=ss, scale_factor=scale_factor, w=w,
        alpha=alpha, beta=beta, tonality_vector=tonality_vector,
        key_estimation=key_estimation, key_finding_algorithm=key_finding_algorithm
    )
    cloud_diameter = tension_dict.get("cloud_diameter", [])
    if not cloud_diameter:
        return 0.0
    return float(np.mean(cloud_diameter))

@partitura
@tonality
@pitch
def std_cloud_diameter(
    melody: Melody,
    ws: float = 1.0,
    ss: str = "onset",
    scale_factor: float = SCALE_FACTOR,
    w: np.ndarray = DEFAULT_WEIGHTS,
    alpha: float = ALPHA,
    beta: float = BETA,
    tonality_vector: Optional[list] = None,
    key_estimation: str = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler"
) -> float:
    """Standard deviation of cloud diameter from the tonal tension model. Cloud Diameter provides a
    measure of the maximal tonal distance of the notes in a chord, 
    following the definition in Partitura.
    
    Parameters
    ----------
    melody : Melody
        A melody-features Melody object
    ws : float, optional
        Window size in beats. Default is 1.0 beat.
    ss : str, optional
        Step size or score position for computing the tonal tension features.
        Default is "onset" (compute at each unique score position).
    scale_factor : float, optional
        Multiplicative scaling factor. Default uses the distance between C and B#.
    w : np.ndarray, optional
        Weights for the chords. Default is [0.516, 0.315, 0.168].
    alpha : float, optional
        Preference for V vs v chord in minor key (0-1). Default is 0.75.
    beta : float, optional
        Preference for iv vs IV in minor key (0-1). Default is 0.75.
    tonality_vector : list, optional
        Pre-computed tonality vector (list of (key_name, correlation) tuples). Default is None.
    key_estimation : str, optional
        Key estimation strategy, default "infer_if_necessary"
    key_finding_algorithm : Literal["krumhansl_schmuckler"], optional
        Key-finding algorithm to use when inferring key. Default is "krumhansl_schmuckler".
        
    Returns
    -------
    float
        Standard deviation of cloud diameter values
        
    Citation
    --------
    Herremans & Chew (2016)
    """
    tension_dict = estimate_tonaltension(
        melody, ws=ws, ss=ss, scale_factor=scale_factor, w=w,
        alpha=alpha, beta=beta, tonality_vector=tonality_vector,
        key_estimation=key_estimation, key_finding_algorithm=key_finding_algorithm
    )
    cloud_diameter = tension_dict.get("cloud_diameter", [])
    if len(cloud_diameter) < 2:
        return 0.0
    return float(np.std(cloud_diameter, ddof=1))

@partitura
@tonality
@pitch
def mean_cloud_momentum(
    melody: Melody,
    ws: float = 1.0,
    ss: str = "onset",
    scale_factor: float = SCALE_FACTOR,
    w: np.ndarray = DEFAULT_WEIGHTS,
    alpha: float = ALPHA,
    beta: float = BETA,
    tonality_vector: Optional[list] = None,
    key_estimation: str = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler"
) -> float:
    """Mean cloud momentum from the tonal tension model.
    
    Cloud momentum captures movement of pitch sets in the spiral array 
    space, weighted by note durations, following the definition in Partitura.
    
    Parameters
    ----------
    melody : Melody
        A melody-features Melody object
    ws : float, optional
        Window size in beats. Default is 1.0 beat.
    ss : str, optional
        Step size or score position for computing the tonal tension features.
        Default is "onset" (compute at each unique score position).
    scale_factor : float, optional
        Multiplicative scaling factor. Default uses the distance between C and B#.
    w : np.ndarray, optional
        Weights for the chords. Default is [0.516, 0.315, 0.168].
    alpha : float, optional
        Preference for V vs v chord in minor key (0-1). Default is 0.75.
    beta : float, optional
        Preference for iv vs IV in minor key (0-1). Default is 0.75.
    tonality_vector : list, optional
        Pre-computed tonality vector (list of (key_name, correlation) tuples). Default is None.
    key_estimation : str, optional
        Key estimation strategy, default "infer_if_necessary"
    key_finding_algorithm : Literal["krumhansl_schmuckler"], optional
        Key-finding algorithm to use when inferring key. Default is "krumhansl_schmuckler".
        
    Returns
    -------
    float
        Mean cloud momentum value
        
    Citation
    --------
    Herremans & Chew (2016)
    """
    tension_dict = estimate_tonaltension(
        melody, ws=ws, ss=ss, scale_factor=scale_factor, w=w,
        alpha=alpha, beta=beta, tonality_vector=tonality_vector,
        key_estimation=key_estimation, key_finding_algorithm=key_finding_algorithm
    )
    cloud_momentum = tension_dict.get("cloud_momentum", [])
    if not cloud_momentum:
        return 0.0
    return float(np.mean(cloud_momentum))

@partitura
@tonality
@pitch
def std_cloud_momentum(
    melody: Melody,
    ws: float = 1.0,
    ss: str = "onset",
    scale_factor: float = SCALE_FACTOR,
    w: np.ndarray = DEFAULT_WEIGHTS,
    alpha: float = ALPHA,
    beta: float = BETA,
    tonality_vector: Optional[list] = None,
    key_estimation: str = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler"
) -> float:
    """Standard deviation of cloud momentum from the tonal tension model. Cloud Momentum provides a
    measure of movement of pitch sets in the spiral array space, weighted by note durations, 
    following the definition in Partitura.
    
    Parameters
    ----------
    melody : Melody
        A melody-features Melody object
    ws : float, optional
        Window size in beats. Default is 1.0 beat.
    ss : str, optional
        Step size or score position for computing the tonal tension features.
        Default is "onset" (compute at each unique score position).
    scale_factor : float, optional
        Multiplicative scaling factor. Default uses the distance between C and B#.
    w : np.ndarray, optional
        Weights for the chords. Default is [0.516, 0.315, 0.168].
    alpha : float, optional
        Preference for V vs v chord in minor key (0-1). Default is 0.75.
    beta : float, optional
        Preference for iv vs IV in minor key (0-1). Default is 0.75.
    tonality_vector : list, optional
        Pre-computed tonality vector (list of (key_name, correlation) tuples). Default is None.
    key_estimation : str, optional
        Key estimation strategy, default "infer_if_necessary"
    key_finding_algorithm : Literal["krumhansl_schmuckler"], optional
        Key-finding algorithm to use when inferring key. Default is "krumhansl_schmuckler".
        
    Returns
    -------
    float
        Standard deviation of cloud momentum values
        
    Citation
    --------
    Herremans & Chew (2016)
    """
    tension_dict = estimate_tonaltension(
        melody, ws=ws, ss=ss, scale_factor=scale_factor, w=w,
        alpha=alpha, beta=beta, tonality_vector=tonality_vector,
        key_estimation=key_estimation, key_finding_algorithm=key_finding_algorithm
    )
    cloud_momentum = tension_dict.get("cloud_momentum", [])
    if len(cloud_momentum) < 2:
        return 0.0
    return float(np.std(cloud_momentum, ddof=1))

@partitura
@tonality
@pitch
def mean_tensile_strain(
    melody: Melody,
    ws: float = 1.0,
    ss: str = "onset",
    scale_factor: float = SCALE_FACTOR,
    w: np.ndarray = DEFAULT_WEIGHTS,
    alpha: float = ALPHA,
    beta: float = BETA,
    tonality_vector: Optional[list] = None,
    key_estimation: str = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler"
) -> float:
    """Mean tensile strain from the tonal tension model. Tensile strain provides a 
    measure of the distance between the local and global tonal context, 
    following the definition in Partitura.
    
    Parameters
    ----------
    melody : Melody
        A melody-features Melody object
    ws : float, optional
        Window size in beats. Default is 1.0 beat.
    ss : str, optional
        Step size or score position for computing the tonal tension features.
        Default is "onset" (compute at each unique score position).
    scale_factor : float, optional
        Multiplicative scaling factor. Default uses the distance between C and B#.
    w : np.ndarray, optional
        Weights for the chords. Default is [0.516, 0.315, 0.168].
    alpha : float, optional
        Preference for V vs v chord in minor key (0-1). Default is 0.75.
    beta : float, optional
        Preference for iv vs IV in minor key (0-1). Default is 0.75.
    tonality_vector : list, optional
        Pre-computed tonality vector (list of (key_name, correlation) tuples). Default is None.
    key_estimation : str, optional
        Key estimation strategy, default "infer_if_necessary"
    key_finding_algorithm : Literal["krumhansl_schmuckler"], optional
        Key-finding algorithm to use when inferring key. Default is "krumhansl_schmuckler".
        
    Returns
    -------
    float
        Mean tensile strain value
        
    Citation
    --------
    Herremans & Chew (2016)
    """
    tension_dict = estimate_tonaltension(
        melody, ws=ws, ss=ss, scale_factor=scale_factor, w=w,
        alpha=alpha, beta=beta, tonality_vector=tonality_vector,
        key_estimation=key_estimation, key_finding_algorithm=key_finding_algorithm
    )
    tensile_strain = tension_dict.get("tensile_strain", [])
    if not tensile_strain:
        return 0.0
    return float(np.mean(tensile_strain))

@partitura
@tonality
@pitch
def std_tensile_strain(
    melody: Melody,
    ws: float = 1.0,
    ss: str = "onset",
    scale_factor: float = SCALE_FACTOR,
    w: np.ndarray = DEFAULT_WEIGHTS,
    alpha: float = ALPHA,
    beta: float = BETA,
    tonality_vector: Optional[list] = None,
    key_estimation: str = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler"
) -> float:
    """Standard deviation of tensile strain from the tonal tension model. Tensile strain provides a 
    measure of the distance between the local and global tonal context, 
    following the definition in Partitura.
    
    Parameters
    ----------
    melody : Melody
        A melody-features Melody object
    ws : float, optional
        Window size in beats. Default is 1.0 beat.
    ss : str, optional
        Step size or score position for computing the tonal tension features.
        Default is "onset" (compute at each unique score position).
    scale_factor : float, optional
        Multiplicative scaling factor. Default uses the distance between C and B#.
    w : np.ndarray, optional
        Weights for the chords. Default is [0.516, 0.315, 0.168].
    alpha : float, optional
        Preference for V vs v chord in minor key (0-1). Default is 0.75.
    beta : float, optional
        Preference for iv vs IV in minor key (0-1). Default is 0.75.
    tonality_vector : list, optional
        Pre-computed tonality vector (list of (key_name, correlation) tuples). Default is None.
    key_estimation : str, optional
        Key estimation strategy, default "infer_if_necessary"
    key_finding_algorithm : Literal["krumhansl_schmuckler"], optional
        Key-finding algorithm to use when inferring key. Default is "krumhansl_schmuckler".
        
    Returns
    -------
    float
        Standard deviation of tensile strain values
        
    Citation
    --------
    Herremans & Chew (2016)
    """
    tension_dict = estimate_tonaltension(
        melody, ws=ws, ss=ss, scale_factor=scale_factor, w=w,
        alpha=alpha, beta=beta, tonality_vector=tonality_vector,
        key_estimation=key_estimation, key_finding_algorithm=key_finding_algorithm
    )
    tensile_strain = tension_dict.get("tensile_strain", [])
    if len(tensile_strain) < 2:
        return 0.0
    return float(np.std(tensile_strain, ddof=1))

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

@novel
@tonality
@pitch
def tonalness_histogram(pitches: list[int]) -> dict:
    """Equal-width histogram of all 24 Krumhansl-Schmuckler key correlations.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    dict
        Bin-range strings mapped to counts (24 correlations, 24 bins).

    Citation
    --------
    Krumhansl (1990)
    """
    pitch_classes = [p % 12 for p in pitches]
    correlation_values = [value for _, value in compute_tonality_vector(pitch_classes)]
    return histogram_bins(correlation_values, 24)








# Melodic Movement Features






@midi_toolbox
@pitch
@complexity
def gradus(pitches: list[int]) -> float:
    """
    The degree of melodiousness based on Euler's `gradus` suavitatis.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    
    Citation
    --------
    Euler (1739)
    
    Returns
    -------
    float
        Mean gradus suavitatis value across all intervals, where lower values 
        indicate higher melodiousness.
    """
    if len(pitches) < 2:
        return 0.0
    
    # Calculate intervals and collapse to within one octave (interval classes)
    intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches) - 1)]
    intervals = [(interval % 12) for interval in intervals]
    
    # Frequency ratios for intervals (0-11 semitones)
    numerators = [1, 16, 9, 6, 5, 4, 45, 3, 8, 5, 16, 15]
    denominators = [1, 15, 8, 5, 4, 3, 32, 2, 5, 3, 9, 8]
    
    gradus_values = []
    
    for interval in intervals:
        if interval == 0:  # Unison
            gradus_values.append(1.0)
            continue
            
        # Get frequency ratio for this interval
        n = numerators[interval]
        d = denominators[interval]
        
        # Calculate gradus suavitatis using prime factorization
        product = n * d
        
        # Get prime factors
        factors = []
        temp = product
        divisor = 2
        while divisor * divisor <= temp:
            while temp % divisor == 0:
                factors.append(divisor)
                temp //= divisor
            divisor += 1
        if temp > 1:
            factors.append(temp)
        
        # gradus = sum of (prime - 1) + 1
        if factors:
            gradus = sum(factor - 1 for factor in factors) + 1
        else:
            gradus = 1
            
        gradus_values.append(float(gradus))
    
    return float(np.mean(gradus_values)) if gradus_values else 0.0













@fantastic
@both
@complexity
def get_mtype_features(melody: Melody, phrase_gap: float, max_ngram_order: int) -> dict:
    """Various n-gram statistics for the melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze as a Melody object

    Returns
    -------
    dict
        Dictionary containing complexity measures averaged across n-gram lengths
    """
    # Initialize tokenizer and get M-type tokens
    tokenizer = FantasticTokenizer()

    # Segment the melody first, using quarters as the time unit
    segments = tokenizer.segment_melody(melody, phrase_gap=phrase_gap, units="quarters")

    # Get tokens for each segment
    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, segment.starts, segment.ends
        )
        all_tokens.extend(segment_tokens)

    # Create a fresh counter for this melody
    ngram_counter = NGramCounter()
    ngram_counter.ngram_counts = {}  # Explicitly reset the counter

    ngram_counter.count_ngrams(all_tokens, max_order=max_ngram_order)

    # Calculate complexity measures for each n-gram length
    mtype_features = {}

    # Initialize all features to NaN
    mtype_features["yules_k"] = float("nan")
    mtype_features["simpsons_d"] = float("nan")
    mtype_features["sichels_s"] = float("nan")
    mtype_features["honores_h"] = float("nan")
    mtype_features["mean_entropy"] = float("nan")
    mtype_features["mean_productivity"] = float("nan")

    # Try to calculate each feature individually
    if ngram_counter.ngram_counts:
        try:
            mtype_features["yules_k"] = ngram_counter.yules_k
        except Exception as e:
            warnings.warn(f"Error calculating Yule's K: {str(e)}")
        try:
            mtype_features["simpsons_d"] = ngram_counter.simpsons_d
        except Exception as e:
            warnings.warn(f"Error calculating Simpson's D: {str(e)}")

        try:
            mtype_features["sichels_s"] = ngram_counter.sichels_s
        except Exception as e:
            warnings.warn(f"Error calculating Sichel's S: {str(e)}")

        try:
            mtype_features["honores_h"] = ngram_counter.honores_h
        except Exception as e:
            warnings.warn(f"Error calculating Honoré's H: {str(e)}")

        try:
            mtype_features["mean_entropy"] = ngram_counter.mean_entropy
        except Exception as e:
            warnings.warn(f"Error calculating mean entropy: {str(e)}")

        try:
            mtype_features["mean_productivity"] = ngram_counter.mean_productivity
        except Exception as e:
            warnings.warn(f"Error calculating mean productivity: {str(e)}")

    return mtype_features





def _complebm(melody: Melody, method: str = 'o') -> float:
    """Expectancy-based melodic complexity (MIDI Toolbox ``complebm.m``).

    Essen-calibrated: mean 5, SD 1. Methods ``p`` (pitch), ``r`` (rhythm), ``o`` (optimal).

    Citation
    --------
    Eerola & North (2000)
    """
    if not melody.pitches or len(melody.pitches) < 2:
        return 5.0

    method = method.lower()
    if method not in ('p', 'r', 'o'):
        raise ValueError("Method must be 'p' (pitch), 'r' (rhythm), or 'o' (optimal)")

    pitches = melody.pitches
    starts = melody.starts
    ends = melody.ends
    tempo = melody.tempo

    pcd = _pcdist1_vector(pitches, starts, ends)
    ivd = _ivdist1_vector(pitches, starts, ends)
    ton = _tonality_midi_toolbox(pitches, starts, ends)
    dur_acc = duration_accent(starts, ends)
    n = min(len(ton), len(dur_acc))
    ton_component = float(np.mean([ton[i] * dur_acc[i] for i in range(n)])) * -1.0 if n else 0.0

    intervals = np.diff(np.asarray(pitches, dtype=float))
    aveint = float(np.mean(intervals)) if len(intervals) else 0.0

    if method == 'p':
        constant = -0.2407
        y = (
            constant
            + aveint * 0.3
            + midi_toolbox_entropy(pcd) * 1.0
            + midi_toolbox_entropy(ivd) * 0.8
            + ton_component
        ) / 0.9040 + 5.0
        return float(y)

    dud = midi_toolbox_entropy(_durdist1_vector(starts, ends, tempo))
    noteden = _notedensity_seconds(starts)
    du_sec = [float(e) - float(s) for s, e in zip(starts, ends) if float(e) > float(s)]
    if len(du_sec) > 1:
        rhyvar = float(np.std(np.log(du_sec), ddof=1))
    elif len(du_sec) == 1:
        rhyvar = 0.0
    else:
        rhyvar = 0.0

    metach = _meter_accent_mean(melody)

    if method == 'r':
        constant = -0.7841
        y = (constant + dud * 0.7 + noteden * 0.2 + rhyvar * 0.5 + metach * 0.5) / 0.3637 + 5.0
        return float(y)

    constant = -1.9025
    y = (
        constant
        + aveint * 0.2
        + midi_toolbox_entropy(ivd) * 1.5
        + midi_toolbox_entropy(ivd) * 1.3
        + ton_component
        + dud * 0.5
        + noteden * 0.4
        + rhyvar * 0.9
        + metach * 0.8
    ) / 1.5034 + 5.0
    return float(y)


@midi_toolbox
@pitch
@complexity
def complebm_pitch(melody: Melody) -> float:
    """Expectancy-based melodic complexity calculated using pitch patterns only,
    according to Eerola & North (2000). The complexity score is normalized against
    the Essen folksong collection, where a score of 5 represents average complexity.

    Citation
    --------
    Eerola & North (2000)
    """
    return _complebm(melody, "p")


@midi_toolbox
@rhythm
@complexity
def complebm_rhythm(melody: Melody) -> float:
    """Expectancy-based melodic complexity calculated using rhythmic features only,
    according to Eerola & North (2000). The complexity score is normalized against
    the Essen folksong collection, where a score of 5 represents average complexity.

    Citation
    --------
    Eerola & North (2000)
    """
    return _complebm(melody, "r")


@midi_toolbox
@both
@complexity
def complebm_optimal(melody: Melody) -> float:
    """Expectancy-based melodic complexity calculated using an optimal combination
    of pitch patterns and rhythmic features, according to Eerola & North (2000).
    The complexity score is normalized against the Essen folksong collection,
    where a score of 5 represents average complexity.

    Citation
    --------
    Eerola & North (2000)
    """
    return _complebm(melody, "o")


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


def get_lexical_diversity_features(
    melody: Melody,
    phrase_gap: float = 1.5,
    max_ngram_order: int = DEFAULT_MAX_NGRAM_ORDER,
) -> Dict:
    """Collect lexical-diversity (m-type) features for a melody.

    Corpus-dependent lexical-diversity features (e.g. TF/DF correlations) are
    computed in :func:`get_corpus_features` instead.
    """
    return get_mtype_features(
        melody, phrase_gap=phrase_gap, max_ngram_order=max_ngram_order
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


def get_contour_features(melody: Melody) -> Dict:
    """Compute all contour-based features for a melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    Dict
        Dictionary of contour-based feature values

    """
    contour_features = {}

    # Calculate step contour features
    step_contour = get_step_contour_features(melody.pitches, melody.starts, melody.ends, melody.tempo)
    contour_features["step_contour_global_variation"] = step_contour[0]
    contour_features["step_contour_global_direction"] = step_contour[1]
    contour_features["step_contour_local_variation"] = step_contour[2]

    # Calculate interpolation contour features
    interpolation_contour = get_interpolation_contour_features(
        melody.pitches, melody.starts
    )
    contour_features["interpolation_contour_global_direction"] = interpolation_contour[
        0
    ]
    contour_features["interpolation_contour_mean_gradient"] = interpolation_contour[1]
    contour_features["interpolation_contour_gradient_std"] = interpolation_contour[2]
    contour_features["interpolation_contour_direction_changes"] = interpolation_contour[
        3
    ]
    contour_features["interpolation_contour_class_label"] = interpolation_contour[4]
    contour_features["polynomial_contour_coefficients"] = get_polynomial_contour_features(melody)
    contour_features["huron_contour"] = get_huron_contour_features(melody)
    contour_features["comb_contour_matrix"] = comb_contour_matrix(melody.pitches)
    return contour_features


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


def _is_beat_histogram_function(func) -> bool:
    """Check if a function uses beat histogram computations."""
    import inspect
    try:
        source = inspect.getsource(func)
        return '_get_beat_histogram_values_from_ticks' in source or 'create_beat_histogram' in source
    except:
        return False


def _precompute_beat_histogram_data(melody: Melody) -> tuple:
    """Pre-compute beat histogram data for reuse across multiple functions.
    
    Returns
    -------
    tuple
        (normal_values, standardized_values, start_ticks, end_ticks, tempo, ppqn)
    """
    seconds_per_tick = (60.0 / float(melody.tempo)) / float(480)
    to_ticks = lambda t: int(round(float(t) / seconds_per_tick))
    start_ticks = tuple(to_ticks(s) for s in melody.starts)
    end_ticks = tuple(to_ticks(e) for e in melody.ends)
    
    if not end_ticks:
        return tuple(), tuple(), start_ticks, end_ticks, melody.tempo, 480
    
    normal_values, standardized_values = _get_beat_histogram_values_from_ticks(
        start_ticks, end_ticks, float(melody.tempo), 480
    )
    
    return normal_values, standardized_values, start_ticks, end_ticks, melody.tempo, 480


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











@idyom
@pitch
@tonality
def inscale(
    melody: Melody,
    key_estimation: Literal["always_read_from_file", "infer_if_necessary", "always_infer"] = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler",
) -> list[int]:
    """For each pitch in the melody, returns 1 if the pitch is in the estimated key's scale,
    or 0 if it deviates from the scale.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    list[int]
        List of 0/1 values indicating if each pitch is in the estimated key's scale
    """
    """For each pitch, returns 1 if pitch class is in the resolved key scale, else 0.

    Key resolution follows ``key_estimation``:
    - ``always_read_from_file``: require key signature in MIDI metadata
    - ``infer_if_necessary``: use MIDI key signature when present, else infer
    - ``always_infer``: always infer from note content
    """
    pitches = melody.pitches
    pitch_classes = [pitch % 12 for pitch in pitches]
    resolved_key = _resolve_key_for_melody(
        melody,
        key_estimation=key_estimation,
        key_finding_algorithm=key_finding_algorithm,
    )

    if not resolved_key:
        return []

    key_name, mode = resolved_key.split()
    key_distances = _get_key_distances()
    root = key_distances.get(key_name)
    if root is None:
        return []
    is_major = mode == "major"
    scale = [0, 2, 4, 5, 7, 9, 11] if is_major else [0, 2, 3, 5, 7, 8, 10]
    scale = [(note + root) % 12 for note in scale]
    return [1 if pc in scale else 0 for pc in pitch_classes]

@novel
@tonality
@pitch
def proportion_inscale(
    melody: Melody,
    key_estimation: Literal["always_read_from_file", "infer_if_necessary", "always_infer"] = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler",
) -> float:
    """The proportion of notes in the melody that are in the scale of the
    estimated key.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    float
        Proportion of notes in the scale
    """
    inscale_vals = inscale(
        melody,
        key_estimation=key_estimation,
        key_finding_algorithm=key_finding_algorithm,
    )
    if not inscale_vals:
        return -1.0
    return sum(inscale_vals) / len(inscale_vals)

@novel
@tonality
@pitch
def longest_monotonic_conjunct_scalar_passage(
    melody: Melody,
    key_estimation: Literal["always_read_from_file", "infer_if_necessary", "always_infer"] = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler",
) -> int:
    """The longest sequence of consecutive notes that fit within the estimated key's scale
    that move in the same direction. 
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    int
        Length of the longest monotonic conjunct scalar passage
    """
    from .algorithms import longest_monotonic_conjunct_scalar_passage as _longest_monotonic_conjunct_scalar_passage
    pitches = melody.pitches
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)
    resolved_key = _resolve_key_for_melody(
        melody,
        key_estimation=key_estimation,
        key_finding_algorithm=key_finding_algorithm,
    )
    if resolved_key:
        correlations = _tonality_correlations_for_key(correlations, resolved_key)
    return _longest_monotonic_conjunct_scalar_passage(pitches, correlations)

@novel
@tonality
@pitch
def longest_conjunct_scalar_passage(
    melody: Melody,
    key_estimation: Literal["always_read_from_file", "infer_if_necessary", "always_infer"] = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler",
) -> int:
    """The longest sequence of consecutive notes that fit within the estimated key's scale.
    For example, a melody estimated to be in C major with notes C, D, E, F, G would have a 
    longest conjunct scalar passage of 5.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    int
        Length of the longest conjunct scalar passage
    """
    from .algorithms import longest_conjunct_scalar_passage as _longest_conjunct_scalar_passage
    pitches = melody.pitches
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)
    resolved_key = _resolve_key_for_melody(
        melody,
        key_estimation=key_estimation,
        key_finding_algorithm=key_finding_algorithm,
    )
    if resolved_key:
        correlations = _tonality_correlations_for_key(correlations, resolved_key)
    return _longest_conjunct_scalar_passage(pitches, correlations)

@novel
@tonality
@pitch
def proportion_conjunct_scalar(
    melody: Melody,
    key_estimation: Literal["always_read_from_file", "infer_if_necessary", "always_infer"] = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler",
) -> float:
    """Longest conjunct scalar passage length divided by total note count.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    float
        Proportion of conjunct scalar motion
    """
    from .algorithms import proportion_conjunct_scalar as _proportion_conjunct_scalar
    pitches = melody.pitches
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)
    resolved_key = _resolve_key_for_melody(
        melody,
        key_estimation=key_estimation,
        key_finding_algorithm=key_finding_algorithm,
    )
    if resolved_key:
        correlations = _tonality_correlations_for_key(correlations, resolved_key)
    return _proportion_conjunct_scalar(pitches, correlations)

@novel
@tonality
@pitch
def proportion_scalar(
    melody: Melody,
    key_estimation: Literal["always_read_from_file", "infer_if_necessary", "always_infer"] = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler",
) -> float:
    """Longest monotonic conjunct scalar passage length divided by total notes.

    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    float
        Proportion of scalar motion
    """
    from .algorithms import proportion_scalar as _proportion_scalar
    pitches = melody.pitches
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)
    resolved_key = _resolve_key_for_melody(
        melody,
        key_estimation=key_estimation,
        key_finding_algorithm=key_finding_algorithm,
    )
    if resolved_key:
        correlations = _tonality_correlations_for_key(correlations, resolved_key)
    return _proportion_scalar(pitches, correlations)

@fantastic
@tonality
@pitch
def mode(
    melody: Melody,
    key_estimation: Literal["always_read_from_file", "infer_if_necessary", "always_infer"] = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler"
) -> str:
    """Calculate the mode (major/minor) of a melody, either read from the MIDI file or
    estimated using the specified key finding algorithm.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    key_estimation : str, optional
        Key estimation strategy, default "infer_if_necessary"
        Can be "always_read_from_file", "infer_if_necessary", or "always_infer"
    key_finding_algorithm : Literal["krumhansl_schmuckler"], optional
        Key-finding algorithm to use when inferring mode. Default is "krumhansl_schmuckler".
        
    Returns
    -------
    str
        The mode: "major" or "minor"
    """
    resolved_key = _resolve_key_for_melody(
        melody,
        key_estimation=key_estimation,
        key_finding_algorithm=key_finding_algorithm,
    )
    if not resolved_key:
        return "unknown"
    return resolved_key.split()[1]

def get_tonality_features(
    melody: Melody,
    key_estimation: Literal["always_read_from_file", "infer_if_necessary", "always_infer"] = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler"
) -> Dict:
    """Compute all tonality-based features for a melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze
    key_estimation : Literal["always_read_from_file", "infer_if_necessary", "always_infer"], optional
        Key estimation strategy. Default is "infer_if_necessary".
    key_finding_algorithm : Literal["krumhansl_schmuckler"], optional
        Key-finding algorithm to use when inferring key. Default is "krumhansl_schmuckler".

    Returns
    -------
    Dict
        Dictionary of tonality-based feature values

    """
    tonality_features = {}

    pcs = [pitch % 12 for pitch in melody.pitches]
    correlations = compute_tonality_vector(pcs)

    # Keep batch and standalone behavior consistent.
    tonality_features["tonalness"] = tonalness(melody.pitches)
    tonality_features["tonal_clarity"] = tonal_clarity(melody.pitches)
    tonality_features["tonal_spike"] = tonal_spike(melody.pitches)
    tonality_features["tonalness_histogram"] = tonalness_histogram(melody.pitches)

    key_for_features = _resolve_key_for_melody(
        melody,
        key_estimation=key_estimation,
        key_finding_algorithm=key_finding_algorithm,
    )

    if key_for_features:
        tonality_features["referent"] = referent(
            melody,
            key_estimation=key_estimation,
            key_finding_algorithm=key_finding_algorithm,
        )
        tonality_features["inscale"] = inscale(
            melody,
            key_estimation=key_estimation,
            key_finding_algorithm=key_finding_algorithm,
        )
        tonality_features["key"] = key_for_features
        tonality_features["mode"] = key_for_features.split()[1]
    else:
        tonality_features["referent"] = -1
        tonality_features["inscale"] = []
        tonality_features["key"] = "unknown"
        tonality_features["mode"] = "unknown"


    # Scalar passage features
    from .algorithms import longest_monotonic_conjunct_scalar_passage as _longest_monotonic_conjunct_scalar_passage
    from .algorithms import longest_conjunct_scalar_passage as _longest_conjunct_scalar_passage
    scalar_correlations = correlations
    if key_for_features:
        scalar_correlations = _tonality_correlations_for_key(correlations, key_for_features)

    tonality_features["longest_monotonic_conjunct_scalar_passage"] = (
        _longest_monotonic_conjunct_scalar_passage(melody.pitches, scalar_correlations)
    )
    tonality_features["longest_conjunct_scalar_passage"] = (
        _longest_conjunct_scalar_passage(melody.pitches, scalar_correlations)
    )
    from .algorithms import proportion_conjunct_scalar as _proportion_conjunct_scalar
    from .algorithms import proportion_scalar as _proportion_scalar
    tonality_features["proportion_conjunct_scalar"] = _proportion_conjunct_scalar(
        melody.pitches, scalar_correlations
    )
    tonality_features["proportion_scalar"] = _proportion_scalar(melody.pitches, scalar_correlations)
    tonality_features["proportion_inscale"] = proportion_inscale(
        melody,
        key_estimation=key_estimation,
        key_finding_algorithm=key_finding_algorithm,
    )
    
    tension_dict = estimate_tonaltension(
        melody,
        key_estimation=key_estimation,
        key_finding_algorithm=key_finding_algorithm
    )
    
    # Extract individual statistics
    cloud_diameter = tension_dict.get("cloud_diameter", [])
    if cloud_diameter:
        tonality_features["mean_cloud_diameter"] = float(np.mean(cloud_diameter))
        tonality_features["std_cloud_diameter"] = float(np.std(cloud_diameter, ddof=1)) if len(cloud_diameter) > 1 else 0.0
    else:
        tonality_features["mean_cloud_diameter"] = 0.0
        tonality_features["std_cloud_diameter"] = 0.0
    
    cloud_momentum = tension_dict.get("cloud_momentum", [])
    if cloud_momentum:
        tonality_features["mean_cloud_momentum"] = float(np.mean(cloud_momentum))
        tonality_features["std_cloud_momentum"] = float(np.std(cloud_momentum, ddof=1)) if len(cloud_momentum) > 1 else 0.0
    else:
        tonality_features["mean_cloud_momentum"] = 0.0
        tonality_features["std_cloud_momentum"] = 0.0
    
    tensile_strain = tension_dict.get("tensile_strain", [])
    if tensile_strain:
        tonality_features["mean_tensile_strain"] = float(np.mean(tensile_strain))
        tonality_features["std_tensile_strain"] = float(np.std(tensile_strain, ddof=1)) if len(tensile_strain) > 1 else 0.0
    else:
        tonality_features["mean_tensile_strain"] = 0.0
        tonality_features["std_tensile_strain"] = 0.0
    
    # Keep the full dict for backward compatibility
    tonality_features["tonal_tension"] = tension_dict

    return tonality_features


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
