"""
This module contains functions to compute features from melodies.
Features are the product of an input list and at least one algorithm.
"""

__author__ = "David Whyatt"

import warnings
from importlib import resources

from .feature_decorators import (
    fantastic, idyom, midi_toolbox, melsim, jsymbolic, novel, simile, partitura,
    FeatureType, feature_type, interval, class_based, contour, tonality, metre, descriptives, 
    corpus_prevalence, expectation, complexity,
    pitch, rhythm, both
)

warnings.filterwarnings("ignore", category=UserWarning, module="pretty_midi")
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
    get_mode,
    range_func,
    shannon_entropy,
    standard_deviation,
)
from melody_features.step_contour import StepContour
from melody_features.meter_estimation import (
    duration_accent,
    melodic_accent,
    metric_hierarchy
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
@jsymbolic
@descriptives
@pitch
def pitch_range(pitches: list[int]) -> int:
    """Subtract the lowest pitch number in the melody from the highest.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Range between highest and lowest pitch in semitones
    """
    return int(range_func(pitches))


@fantastic
@jsymbolic
@descriptives
@pitch
def pitch_standard_deviation(pitches: list[int]) -> float:
    """Standard deviation of all pitch numbers in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of pitches

    Notes
    -----
    This feature is named 'pitch_variability' in JSymbolic.
    """
    if not pitches or len(pitches) < 2:
        return 0.0
    return float(np.std(pitches, ddof=1))

@jsymbolic
@class_based
@descriptives
@pitch
def pitch_class_variability(pitches: list[int]) -> float:
    """Standard deviation of all pitch classes in the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of pitch class values
    """
    if not pitches or len(pitches) < 2:
        return 0.0
    pcs = [int(p % 12) for p in pitches]
    return float(np.std(pcs, ddof=1))

@jsymbolic
@descriptives
@class_based
@pitch
def pitch_class_variability_after_folding(pitches: list[int]) -> float:
    """Standard deviation of all pitch classes after arranging the pitch classes by perfect fifths.
    Provides a measure of how close the pitch classes are as a whole from the mean pitch class from a 
    dominant-tonic perspective.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of folded pitch class values
    """
    if not pitches:
        return 0.0
    
    if not pitches or len(pitches) < 2:
        return 0.0
    folded_pcs = [int((7 * (p % 12)) % 12) for p in pitches]
    return float(np.std(folded_pcs, ddof=1))


@fantastic
@descriptives
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

@midi_toolbox
@descriptives
@class_based
@pitch
def pcdist1(pitches: list[int], starts: list[float], ends: list[float]) -> dict:
    """The distribution of pitch classes in the melody, weighted by the duration of the notes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    dict
        Duration-weighted distribution proportion of pitch classes
    """
    if not pitches or not starts or not ends:
        return 0.0

    durations = [ends[i] - starts[i] for i in range(len(starts))]
    # Create weighted list by repeating each pitch class according to its duration
    weighted_pitch_classes = []
    for pitch, duration in zip(pitches, durations):
        pitch_class = pitch % 12
        # Convert duration to integer number of repetitions (e.g. duration 2.5 -> 25 repetitions)
        repetitions = max(1, int(duration * 10))  # Ensures at least 1 repetition
        weighted_pitch_classes.extend([pitch_class] * repetitions)

    if not weighted_pitch_classes:
        return 0.0

    return distribution_proportions(weighted_pitch_classes)

@jsymbolic
@descriptives
@pitch
def first_pitch(pitches: list[int]) -> int:
    """The first pitch number in the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    
    Returns
    -------
    int
        First pitch in the melody
    """
    if not pitches:
        return 0
    return int(pitches[0])

@jsymbolic
@descriptives
@class_based
@pitch
def first_pitch_class(pitches: list[int]) -> int:
    """The first pitch class in the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    
    Returns
    -------
    int - between 0 and 11
        First pitch class in the melody
    """
    if not pitches:
        return 0
    return int(pitches[0] % 12)

@jsymbolic
@descriptives
@pitch
def last_pitch(pitches: list[int]) -> int:
    """The last pitch number in the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    
    Returns
    -------
    int
        Last pitch in the melody
    """
    if not pitches:
        return 0
    return int(pitches[-1])

@jsymbolic
@descriptives
@class_based
@pitch
def last_pitch_class(pitches: list[int]) -> int:
    """The last pitch class in the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    
    Returns
    -------
    int - between 0 and 11
    """
    if not pitches:
        return 0
    return int(pitches[-1] % 12)

@jsymbolic
@descriptives
@pitch
def basic_pitch_histogram(pitches: list[int]) -> dict:
    """A histogram of pitch values within the range of input pitches.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    dict
        Dictionary mapping pitch values to counts

    Note
    ----
    We use the histogram in the range of input pitches to reduce the output size. An implementation
    that is truer to the original jSymbolic implementation would return 128 bins (0-127) regardless of how any different pitches are present.
    However, we believe our approach is more concise and easier to understand for many purposes.
    """
    if not pitches:
        return {}

    # Use number of unique pitches as number of bins, with minimum of 1
    # we return this instead of the full PitchHistogram object to reduce simplify the output
    # as the PitchHistogram object would return 128 bins (0-127) regardless of how any different pitches are present
    num_midi_notes = max(1, len(set(pitches)))
    return histogram_bins(pitches, num_midi_notes)

@jsymbolic
@descriptives
@pitch
def melodic_pitch_variety(pitches: list[int], starts: list[float], tempo: float = 120.0, ppqn: int = 480) -> float:
    """The average number of notes that pass before a pitch is repeated.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times
    tempo : float, default=120.0
        Tempo in beats per minute
    ppqn : int, default=480
        Pulses per quarter note (MIDI resolution)

    Returns
    -------
    float
        Average number of notes before pitch repetition
    """
    if not pitches or len(pitches) < 2:
        return 0.0

    from .stats import time_to_ticks
    
    note_sequence = sorted(zip(starts, pitches))
    starts_ordered, pitches_ordered = zip(*note_sequence)
    
    # Convert to ticks
    tick_pitch_map = {}
    for start, pitch in zip(starts_ordered, pitches_ordered):
        tick = time_to_ticks(start, tempo, ppqn)
        if tick not in tick_pitch_map:
            tick_pitch_map[tick] = []
        tick_pitch_map[tick].append(pitch)

    sorted_ticks = sorted(tick_pitch_map.keys())
    
    repeated_notes_count = 0
    total_notes_before_repetition = 0
    max_notes_that_can_go_by = 16

    for tick_idx, tick in enumerate(sorted_ticks):
        notes_at_tick = tick_pitch_map[tick]

        for pitch in notes_at_tick:
            found_repeated_pitch = False
            notes_gone_by_with_different_pitch = 0
            last_tick_examined = tick

            for future_tick_idx in range(tick_idx + 1, len(sorted_ticks)):
                if found_repeated_pitch or notes_gone_by_with_different_pitch > max_notes_that_can_go_by:
                    break

                future_tick = sorted_ticks[future_tick_idx]

                if future_tick != last_tick_examined:
                    notes_gone_by_with_different_pitch += 1
                    last_tick_examined = future_tick

                future_notes = tick_pitch_map[future_tick]

                for future_pitch in future_notes:
                    if future_pitch == pitch and not found_repeated_pitch and notes_gone_by_with_different_pitch <= max_notes_that_can_go_by:
                        found_repeated_pitch = True
                        repeated_notes_count += 1
                        total_notes_before_repetition += notes_gone_by_with_different_pitch
                        break

    if repeated_notes_count == 0:
        return 0.0

    return float(total_notes_before_repetition / repeated_notes_count)


def _consecutive_fifths(pitch_classes: list[int]) -> list[int]:
    """Find longest sequence of pitch classes separated by perfect fifths.

    Parameters
    ----------
    pitch_classes : list[int]
        List of pitch classes (0-11)

    Returns
    -------
    list[int]
        Longest sequence of consecutive pitch classes separated by perfect fifths
    """
    if not pitch_classes:
        return []

    circle_of_fifths_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    
    longest_sequence = [pitch_classes[0]]
    current_sequence = [pitch_classes[0]]
    
    for i in range(1, len(pitch_classes)):
        pc = pitch_classes[i]
        last_pc = current_sequence[-1]
        
        # Check if current PC is a fifth away from the last PC with wraparound
        if (circle_of_fifths_order.index(pc) - circle_of_fifths_order.index(last_pc)) % 12 == 1:
            current_sequence.append(pc)
        else:
            if len(current_sequence) > len(longest_sequence):
                longest_sequence = current_sequence[:]
            current_sequence = [pc]

    if len(current_sequence) > len(longest_sequence):
        longest_sequence = current_sequence[:]
    
    return longest_sequence

@jsymbolic
@descriptives
@class_based
@pitch
def dominant_spread(pitches: list[int]) -> int:
    """The longest sequence of pitch classes separated by perfect 5ths that each appear >9% of the time.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Length of longest sequence of significant pitch classes separated by perfect 5ths
    """
    pcs = [pitch % 12 for pitch in pitches]
    pc_counts = {}
    for pc in pcs:
        pc_counts[pc] = pc_counts.get(pc, 0) + 1

    total_notes = len(pcs)
    threshold = 0.09

    significant_pcs = []
    for pc, count in pc_counts.items():
        if count / total_notes >= threshold:
            significant_pcs.append(pc)

    if not significant_pcs:
        return 0

    circle_of_fifths_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

    test_sequence = []
    for pc in circle_of_fifths_order:
        if pc in significant_pcs:
            test_sequence.append(pc)

    if test_sequence:
        test_sequence = test_sequence * 2

    longest_sequence = _consecutive_fifths(test_sequence)

    return len(longest_sequence)

@jsymbolic
@descriptives
@pitch
def mean_pitch(pitches: list[int]) -> int:
    """The arithmetic mean of the pitch numbers in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Mean pitch value
    """
    return int(np.mean(pitches))

@jsymbolic
@descriptives
@class_based
@pitch
def mean_pitch_class(pitches: list[int]) -> float:
    """The arithmetic mean of the pitch classes in the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Mean pitch class value, so between 0 and 11
    """
    return float(np.mean([pitch % 12 for pitch in pitches]))

@jsymbolic
@descriptives
@pitch
def most_common_pitch(pitches: list[int]) -> int:
    """The most frequently occurring pitch number in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Most common pitch value
    """
    return int(get_mode(pitches))

@jsymbolic
@descriptives
@class_based
@pitch
def most_common_pitch_class(pitches: list[int]) -> int:
    """The most frequently occurring pitch class in the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Most common pitch class value
    """
    if not pitches:
        return 0
    return int(get_mode([pitch % 12 for pitch in pitches]))

@jsymbolic
@descriptives
@class_based
@pitch
def number_of_unique_pitch_classes(pitches: list[int]) -> int:
    """The number of unique pitch classes in the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of unique pitch classes
    """
    return int(len(set([pitch % 12 for pitch in pitches])))

@jsymbolic
@descriptives
@class_based
@pitch
def number_of_common_pitches_classes(pitches: list[int]) -> int:
    """The number of pitch classes that appear in at least 20% of total notes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of significant pitch classes
    """
    pcs = [pitch % 12 for pitch in pitches]
    significant_pcs = n_percent_significant_values(pcs, threshold=0.2)
    return int(len(significant_pcs))

@jsymbolic
@descriptives
@pitch
def number_of_unique_pitches(pitches: list[int]) -> int:
    """The number of unique pitch numbers in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of unique pitches
    """
    return int(len(set(pitches)))

@jsymbolic
@descriptives
@pitch
def number_of_common_pitches(pitches: list[int]) -> int:
    """The number of unique pitch numbers that appear in at least 9% of total notes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of unique pitches that appear in at least 9% of notes
    """

    significant_pitches = n_percent_significant_values(pitches, threshold=0.09)
    return int(len(set(significant_pitches)))

@midi_toolbox
@descriptives
@pitch
def tessitura(pitches: list[int]) -> list[float]:
    """Tessitura is based on standard deviation from median pitch height. The median range 
    of the melody tends to be favoured and thus more expected. Tessitura predicts 
    whether listeners expect tones close to median pitch height. Higher tessitura values
    correspond to melodies that have a wider range of pitches.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    list[float]
        Absolute tessitura value for each note in the sequence

    Citation
    ---------
    von Hippel (2000).
    
    """
    if len(pitches) < 2:
        return [0.0] if len(pitches) == 1 else []
    
    tessitura_values = [0.0]
    
    for i in range(2, len(pitches) + 1):
        median_prev = np.median(pitches[:i-1])
        
        if i == 2:
            tessitura_values.append(0.0)
            continue
            
        std_prev = np.std(pitches[:i-1], ddof=1)
        
        if std_prev == 0:
            tessitura_values.append(0.0)
        else:
            current_pitch = pitches[i-1]
            tessitura_val = (current_pitch - median_prev) / std_prev
            tessitura_values.append(abs(tessitura_val))
    
    tessitura_values = [float(val) for val in tessitura_values]
    return tessitura_values

@novel
@descriptives
@pitch
def mean_tessitura(pitches: list[int]) -> float:
    """The arithmetic mean of the sequence of tessitura values.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Mean tessitura value
    """
    tess_values = tessitura(pitches)
    if not tess_values:
        return 0.0
    return float(np.mean(tess_values))

@novel
@descriptives
@pitch
def tessitura_std(pitches: list[int]) -> float:
    """The standard deviation of the sequence of tessitura values.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Standard deviation of tessitura values
    """
    tess_values = tessitura(pitches)
    if len(tess_values) < 2:
        return 0.0
    return float(np.std(tess_values, ddof=1))

@jsymbolic
@descriptives
@pitch
def prevalence_of_most_common_pitch(pitches: list[int]) -> float:
    """The proportion of pitches that are the most common pitch with regards to the
    total number of pitches in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of most common pitch
    """
    return float(pitches.count(most_common_pitch(pitches)) / len(pitches))

@jsymbolic
@descriptives
@class_based
@pitch
def prevalence_of_most_common_pitch_class(pitches: list[int]) -> float:
    """The proportion of pitch classes that are the most common pitch class with regards to the
    total number of pitch classes in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of most common pitch class
    """
    if not pitches:
        return 0.0
    pcs = [pitch % 12 for pitch in pitches]
    return float(pcs.count(most_common_pitch_class(pcs)) / len(pcs))

@jsymbolic
@descriptives
@pitch
def relative_prevalence_of_top_pitches(pitches: list[int]) -> float:
    """The ratio of the frequency of the second most common pitch to the frequency of the most common pitch.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Ratio of second most common pitch frequency to most common pitch frequency
    """
    if len(pitches) < 2:
        return 0.0

    pitch_counts = {}
    for pitch in pitches:
        pitch_counts[pitch] = pitch_counts.get(pitch, 0) + 1

    if len(pitch_counts) < 2:
        return 0.0

    sorted_pitches = sorted(pitch_counts.items(), key=lambda x: x[1], reverse=True)
    most_common_freq = sorted_pitches[0][1] / len(pitches)
    second_most_freq = sorted_pitches[1][1] / len(pitches)

    return float(second_most_freq / most_common_freq)

@jsymbolic
@descriptives
@class_based
@pitch
def relative_prevalence_of_top_pitch_classes(pitches: list[int]) -> float:
    """The ratio of the frequency of the second most common pitch class to the frequency of the most common pitch class.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Ratio of second most common pitch class frequency to most common pitch class frequency
    """
    if len(pitches) < 2:
        return 0.0

    pcs = [pitch % 12 for pitch in pitches]
    if len(pcs) < 2:
        return 0.0

    pc_counts = {}
    for pc in pcs:
        pc_counts[pc] = pc_counts.get(pc, 0) + 1

    if len(pc_counts) < 2:
        return 0.0

    sorted_pcs = sorted(pc_counts.items(), key=lambda x: x[1], reverse=True)
    most_common_freq = sorted_pcs[0][1] / len(pcs)
    second_most_freq = sorted_pcs[1][1] / len(pcs)

    return float(second_most_freq / most_common_freq)

@jsymbolic
@descriptives
@pitch
def interval_between_most_prevalent_pitches(pitches: list[int]) -> int:
    """The number of semitones between the two most prevalent pitches.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of semitones between the most prevalent pitches
    """
    if not pitches:
        return 0

    pitch_hist = PitchHistogram(pitches)
    histogram = pitch_hist.histogram
    if not histogram or sum(1 for v in histogram.values() if v > 0) < 2:
        return 0

    max_index = max(histogram, key=lambda k: histogram[k])
    tmp = dict(histogram)
    tmp.pop(max_index, None)
    if not tmp:
        return 0
    second_max_index = max(tmp, key=lambda k: tmp[k])

    return int(abs(int(max_index) - int(second_max_index)))

@jsymbolic
@descriptives
@class_based
@pitch
def interval_between_most_prevalent_pitch_classes(pitches: list[int]) -> int:
    """The number of semitones between the two most prevalent pitch classes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of semitones between the most prevalent pitch classes
    """
    if not pitches:
        return 0

    pch = PitchClassHistogram(pitches)
    histogram = pch.histogram
    if not histogram or sum(1 for v in histogram.values() if v > 0) < 2:
        return 0

    max_index = max(histogram, key=lambda k: histogram[k])
    tmp = dict(histogram)
    tmp.pop(max_index, None)
    if not tmp:
        return 0
    second_max_index = max(tmp, key=lambda k: tmp[k])

    diff = abs(int(max_index) - int(second_max_index))
    return int(diff)

@jsymbolic
@descriptives
@class_based
@pitch
def folded_fifths_pitch_class_histogram(pitches: list[int]) -> dict:
    """A histogram of pitch classes arranged according to the circle of fifths.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    dict
        Dictionary mapping pitch classes to counts, arranged according to the circle of fifths
    """
    # again, we don't use the histogram object for this one to simplify the output
    pcs = [pitch % 12 for pitch in pitches]
    unique = []
    counts = []
    for pc in set(pcs):
        unique.append(pc)
        counts.append(pcs.count(pc))
    return circle_of_fifths(unique, counts)


@jsymbolic
@descriptives
@class_based
@pitch
def pitch_class_skewness(pitches: list[int]) -> float:
    """The skewness of the pitch class histogram, using Pearson's median skewness formula.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Skewness of pitch class histogram values, or 0 for empty input
    """
    if not pitches:
        return 0.0
    
    histogram = PitchClassHistogram(pitches, folded=False)
    return histogram.skewness

@jsymbolic
@descriptives
@class_based
@pitch
def pitch_class_kurtosis(pitches: list[int]) -> float:
    """The sample excess kurtosis of the pitch class histogram.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Kurtosis of pitch class histogram values, or 0 for empty input
    """
    if not pitches:
        return 0.0

    histogram = PitchClassHistogram(pitches, folded=False)
    return histogram.kurtosis

@jsymbolic
@descriptives
@class_based
@pitch
def pitch_class_skewness_after_folding(pitches: list[int]) -> float:
    """The skewness of the pitch class histogram, using Pearson's median skewness formula, 
    after arranging the pitch classes according to the circle of fifths.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Skewness of folded fifths histogram values, or 0 for empty input
    """
    if not pitches:
        return 0.0
    
    histogram = PitchClassHistogram(pitches, folded=True)
    return histogram.skewness

@jsymbolic
@descriptives
@class_based
@pitch
def pitch_class_kurtosis_after_folding(pitches: list[int]) -> float:
    """The sample excess kurtosis of the pitch class histogram, after arranging 
    the pitch classes according to the circle of fifths.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Kurtosis of folded fifths histogram values, or 0 for empty input
    """
    if not pitches:
        return 0.0
    
    histogram = PitchClassHistogram(pitches, folded=True)
    return histogram.kurtosis

@jsymbolic
@descriptives
@class_based
@pitch
def strong_tonal_centres(pitches: list[int]) -> float:
    """The number of isolated peaks in the pitch class histogram that each account for at least 9% of notes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Number of strong tonal centres (peaks >= 9% in fifths histogram)
    """
    if not pitches:
        return 0.0

    fifths_histogram = PitchClassHistogram(pitches, folded=True)
    fifths_hist = fifths_histogram.histogram

    total_notes = sum(fifths_hist.values())
    if total_notes == 0:
        return 0.0

    normalized_fifths = [fifths_hist[i] / total_notes for i in range(12)]

    peaks = 0
    for bin in range(12):
        if normalized_fifths[bin] >= 0.09:
            left = (bin - 1) % 12
            right = (bin + 1) % 12

            if (normalized_fifths[bin] > normalized_fifths[left] and 
                normalized_fifths[bin] > normalized_fifths[right]):
                peaks += 1

    return float(peaks)


@jsymbolic
@descriptives
@pitch
def pitch_skewness(pitches: list[int]) -> float:
    """The skewness of the pitch histogram, using Pearson's median skewness formula.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Median skewness of pitch values, or 0 for empty input or when std dev is 0
    """
    if not pitches:
        return 0.0
    
    histogram = PitchHistogram(pitches)
    return histogram.skewness

@jsymbolic
@descriptives
@pitch
def pitch_kurtosis(pitches: list[int]) -> float:
    """The sample excess kurtosis of the pitch histogram.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Kurtosis of regular pitch histogram values, or 0 for empty input
    """
    if not pitches:
        return 0.0
    histogram = PitchHistogram(pitches)
    return histogram.kurtosis

@jsymbolic
@descriptives
@pitch
def importance_of_bass_register(pitches: list[int]) -> float:
    """The proportion of pitch numbers in the melody that are between 0 and 54. 
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of MIDI pitch numbers that are between 0 and 54
    """
    return float(sum(1 for pitch in pitches if 0 <= pitch <= 54) / len(pitches))

@jsymbolic
@descriptives
@pitch
def importance_of_middle_register(pitches: list[int]) -> float:
    """The proportion of pitch numbers in the melody that are between 55 and 72. 

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of MIDI pitch numbers that are between 55 and 72
    """
    return float(sum(1 for pitch in pitches if 55 <= pitch <= 72) / len(pitches))

@jsymbolic
@descriptives
@pitch
def importance_of_high_register(pitches: list[int]) -> float:
    """The proportion of pitch numbers in the melody that are between 73 and 127. 
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of MIDI pitch numbers that are between 73 and 127
    """
    return float(sum(1 for pitch in pitches if 73 <= pitch <= 127) / len(pitches))


@partitura
@descriptives
@pitch
def pitch_spelling(melody: Melody) -> list[str]:
    """Pitch spelling using the ps13s1 algorithm.
    
    Parameters
    ----------
    melody : Melody
        A melody-features Melody object.

    Returns
    -------
    list[str]
        List of pitch spellings.

    Citation
    ----------
    Meredith (2006)
    """
    return _estimate_spelling_from_melody(melody)

@simile
@interval
@pitch
def pitch_interval(pitches: list[int]) -> list[int]:
    """The intervals (in semitones) between consecutive pitches in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    list[int]
        List of intervals between consecutive pitches in semitones
    """
    return [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]

@fantastic
@interval
@pitch
def absolute_interval_range(pitches: list[int]) -> int:
    """The range between the largest and smallest absolute interval size.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Range between largest and smallest absolute interval in semitones
    """
    return int(range_func([abs(x) for x in pitch_interval(pitches)]))

@fantastic
@jsymbolic
@interval
@pitch
def mean_absolute_interval(pitches: list[int]) -> float:
    """The arithmetic mean of the absolute intervals in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Mean absolute interval size in semitones

    Notes
    -----
    This feature is named 'mean_melodic_interval' in JSymbolic.
    """
    return float(np.mean([abs(x) for x in pitch_interval(pitches)]))

@fantastic
@interval
@pitch
def standard_deviation_absolute_interval(pitches: list[int]) -> float:
    """The standard deviation of the absolute intervals in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of absolute interval sizes in semitones
    """
    return float(np.std([abs(x) for x in pitch_interval(pitches)], ddof=1))

@fantastic
@jsymbolic
@interval
@pitch
def modal_interval(pitches: list[int]) -> int:
    """The most common interval size in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Most frequent interval size in semitones

    Notes
    -----
    This feature is named 'most_common_interval' in JSymbolic.
    """

    intervals_abs = [abs(x) for x in pitch_interval(pitches)]
    if not intervals_abs:
        return 0
    return int(get_mode(intervals_abs))

@fantastic
@interval
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
        Shannon entropy of interval sizes
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
@interval
@pitch
def ivdist1(pitches: list[int], starts: list[float], ends: list[float], tempo: float = 120.0) -> dict:
    """The distribution of intervals in the melody, weighted by their durations.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
    tempo : float
        Tempo in BPM (beats per minute), default 120.0

    Returns
    -------
    dict
        Duration-weighted distribution proportion of intervals
    """
    if not pitches or not starts or not ends or len(pitches) < 2:
        return {}

    intervals = pitch_interval(pitches)
    durations = _get_durations(starts, ends, tempo)

    if not intervals or not durations:
        return {}

    weighted_intervals = []
    for interval, duration in zip(intervals, durations[:-1]):
        repetitions = max(1, int(duration * 10))
        weighted_intervals.extend([interval] * repetitions)

    if not weighted_intervals:
        return {}

    return distribution_proportions(weighted_intervals)

@midi_toolbox
@interval
@pitch
def ivdirdist1(pitches: list[int]) -> dict[int, float]:
    """The proportion of upward intervals for each interval size (1-12 semitones).
    Returns the proportion of upward intervals for each interval size in the melody
    as a dictionary mapping interval sizes to their directional bias values.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    dict[int, float]
        Dictionary mapping interval sizes (1-12 semitones) to directional bias values.
        Each value ranges from -1.0 (all downward) to 1.0 (all upward), with 0.0 being equal.
        Keys: 1=minor second, 2=major second, ..., 12=octave
    """
    if not pitches or len(pitches) < 2:
        return {interval_size: 0.0 for interval_size in range(1, 13)}
    
    intervals = pitch_interval(pitches)
    if not intervals:
        return {interval_size: 0.0 for interval_size in range(1, 13)}
    
    interval_distribution = distribution_proportions(intervals)
    
    interval_direction_distribution = {}
    
    for interval_size in range(1, 13):
        upward_proportion = interval_distribution.get(float(interval_size), 0.0)
        downward_proportion = interval_distribution.get(float(-interval_size), 0.0)
        
        total_proportion = upward_proportion + downward_proportion
        
        if total_proportion > 0:
            directional_bias = (upward_proportion - downward_proportion) / total_proportion
            interval_direction_distribution[interval_size] = directional_bias
        else:
            interval_direction_distribution[interval_size] = 0.0
    
    return interval_direction_distribution

@midi_toolbox
@interval
@pitch
def ivsizedist1(pitches: list[int]) -> dict[int, float]:
    """The distribution of interval sizes (0-12 semitones). Returns the distribution 
    of interval sizes by combining upward and downward intervals of the 
    same absolute size. The first component represents a unison (0)
    and the last component represents an octave (12).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    dict[int, float]
        Dictionary mapping interval sizes (0-12 semitones) to their proportions.
        Keys: 0=unison, 1=minor second, 2=major second, ..., 12=octave
    """
    if not pitches or len(pitches) < 2:
        return {interval_size: 0.0 for interval_size in range(13)}
    
    intervals = pitch_interval(pitches)
    if not intervals:
        return {interval_size: 0.0 for interval_size in range(13)}
    
    interval_distribution = distribution_proportions(intervals)
    
    interval_size_distribution = {}
    
    for interval_size in range(13):
        if interval_size == 0:
            size_proportion = interval_distribution.get(0.0, 0.0)
        else:
            # Combine upward and downward intervals of same absolute size
            upward_proportion = interval_distribution.get(float(interval_size), 0.0)
            downward_proportion = interval_distribution.get(float(-interval_size), 0.0)
            size_proportion = upward_proportion + downward_proportion
        
        interval_size_distribution[interval_size] = size_proportion
    
    return interval_size_distribution

@simile
@interval
@pitch
def interval_direction(pitches: list[int]) -> list[int]:
    """The sequence of interval directions in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    list[int]
        Sequence of interval directions, where:
        1 represents upward motion
        0 represents same pitch
        -1 represents downward motion
    """
    return [
        1 if pitches[i + 1] > pitches[i] else 0 if pitches[i + 1] == pitches[i] else -1
        for i in range(len(pitches) - 1)
    ]

@novel
@interval
@pitch
def interval_direction_mean(pitches: list[int]) -> float:
    """The mean of the direction of each interval in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Mean of interval directions
    """
    directions = interval_direction(pitches)
    
    if not directions:
        return 0.0
    
    return float(sum(directions) / len(directions))

@novel
@interval
@pitch
def interval_direction_std(pitches: list[int]) -> float:
    """The standard deviation of the direction of each interval in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of interval directions
    """
    directions = interval_direction(pitches)
    
    if not directions:
        return 0.0
    
    mean = sum(directions) / len(directions)
    variance = sum((x - mean) ** 2 for x in directions) / len(directions)
    std_dev = math.sqrt(variance)
    
    return float(std_dev)

@jsymbolic
@interval
@pitch
def average_length_of_melodic_arcs(pitches: list[int]) -> float:
    """The average number of notes that separate peaks and troughs in melodic arcs.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Average number of notes that separate peaks and troughs in melodic arcs
    """
    if not pitches:
        return 0.0

    intervals = pitch_interval(pitches)

    total_intervening_intervals = 0
    number_arcs = 0
    direction = 0

    for interval in intervals:
        if direction == -1:
            if interval < 0:
                total_intervening_intervals += 1
            elif interval > 0:
                total_intervening_intervals += 1
                number_arcs += 1
                direction = 1

        elif direction == 1:
            if interval > 0:
                total_intervening_intervals += 1
            elif interval < 0:
                total_intervening_intervals += 1
                number_arcs += 1
                direction = -1

        else:
            if interval > 0:
                direction = 1
                total_intervening_intervals += 1
            elif interval < 0:
                direction = -1
                total_intervening_intervals += 1

    if number_arcs == 0:
        return 0.0

    return float(total_intervening_intervals) / float(number_arcs)

@jsymbolic
@interval
@pitch
def average_interval_span_by_melodic_arcs(pitches: list[int]) -> float:
    """The average interval span of melodic arcs.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Average interval span of melodic arcs, or 0.0 if no arcs found
    """
    total_intervals = 0
    number_intervals = 0

    intervals = pitch_interval(pitches)
    direction = 0
    interval_so_far = 0

    for interval in intervals:
        if direction == -1:
            if interval < 0:
                interval_so_far += abs(interval)
            elif interval > 0:
                total_intervals += interval_so_far
                number_intervals += 1
                interval_so_far = abs(interval)
                direction = 1

        elif direction == 1:
            if interval > 0:
                interval_so_far += abs(interval)
            elif interval < 0:
                total_intervals += interval_so_far
                number_intervals += 1
                interval_so_far = abs(interval)
                direction = -1

        elif direction == 0:
            if interval > 0:
                direction = 1
                interval_so_far += abs(interval)
            elif interval < 0:
                direction = -1
                interval_so_far += abs(interval)

    if number_intervals == 0:
        value = 0.0
    else:
        value = total_intervals / number_intervals

    return float(value)

@jsymbolic
@interval
@pitch
def distance_between_most_prevalent_melodic_intervals(pitches: list[int]) -> float:
    """The absolute difference between the two most common interval sizes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Absolute difference between two most common intervals, or 0.0 if fewer than 2 intervals
    """
    if len(pitches) < 2:
        return 0.0

    intervals = pitch_interval(pitches)
    
    interval_hist = create_melodic_interval_histogram(intervals, use_absolute=True)
    
    histogram = interval_hist.histogram
    
    max_value = 0.0
    max_index = 0
    for interval, count in histogram.items():
        if count > max_value:
            max_value = count
            max_index = interval
    
    second_max_value = 0.0
    second_max_index = 0
    for interval, count in histogram.items():
        if count > second_max_value and interval != max_index:
            second_max_value = count
            second_max_index = interval
    
    if second_max_value == 0.0:
        return 0.0
    
    return float(abs(max_index - second_max_index))

@jsymbolic
@interval
@pitch
def melodic_interval_histogram(pitches: list[int]) -> dict:
    """A histogram of interval sizes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    dict
        Dictionary mapping interval sizes to counts
    """
    intervals = pitch_interval(pitches)
    num_intervals = max(1, int(range_func(intervals)))
    return histogram_bins(intervals, num_intervals)

@jsymbolic
@interval
@pitch
def melodic_large_intervals(pitches: list[int]) -> float:
    """The proportion of intervals >= 13 semitones.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of large intervals, or -1.0 if no intervals
    """
    intervals = pitch_interval(pitches)
    if not intervals:
        return -1.0
    large_intervals = sum(1 for interval in intervals if abs(interval) >= 13)
    return float(large_intervals / len(intervals) if intervals else 0.0)


def variable_melodic_intervals(pitches: list[int], interval_level: Union[int, list[int]]) -> float:
    """The proportion of intervals >= specified size.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    interval_level : int | list[int]
        Minimum interval size in semitones

    Returns
    -------
    float
        Proportion of intervals == interval_level, or -1.0 if no intervals
    """
    intervals = pitch_interval(pitches)
    if not intervals:
        return -1.0
    if isinstance(interval_level, int):
        target_intervals = sum(
            1 for interval in intervals if abs(interval) == interval_level
        )
        return float(target_intervals / len(intervals) if intervals else 0.0)
    else:
        target_intervals = sum(
            1 for interval in intervals if abs(interval) in interval_level
        )
        return float(target_intervals / len(intervals) if intervals else 0.0)

@jsymbolic
@interval
@pitch
def melodic_thirds(pitches: list[int]) -> float:
    """The proportion of intervals that are thirds (3 or 4 semitones).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Proportion of intervals that are thirds (3 or 4 semitones)
    """
    
    return variable_melodic_intervals(pitches, [3, 4])

@jsymbolic
@interval
@pitch
def melodic_perfect_fourths(pitches: list[int]) -> float:
    """The proportion of intervals that are perfect fourths (5 semitones).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Proportion of intervals that are perfect fourths (5 semitones)
    """
    return variable_melodic_intervals(pitches, 5)

@jsymbolic
@interval
@pitch
def melodic_tritones(pitches: list[int]) -> float:
    """The proportion of intervals that are tritones (6 semitones).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Proportion of intervals that are tritones (6 semitones)
    """
    return variable_melodic_intervals(pitches, 6)

@jsymbolic
@interval
@pitch
def melodic_perfect_fifths(pitches: list[int]) -> float:
    """The proportion of intervals that are perfect fifths (7 semitones).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Proportion of intervals that are perfect fifths (7 semitones)
    """
    return variable_melodic_intervals(pitches, 7)

@jsymbolic
@interval
@pitch
def melodic_sixths(pitches: list[int]) -> float:
    """The proportion of intervals that are sixths (8 or 9 semitones).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of intervals that are sixths (8 or 9 semitones)
    """
    return variable_melodic_intervals(pitches, [8, 9])

@jsymbolic
@interval
@pitch
def melodic_sevenths(pitches: list[int]) -> float:
    """The proportion of intervals that are sevenths (10 or 11 semitones).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Proportion of intervals that are sevenths (10 or 11 semitones)
    """
    return variable_melodic_intervals(pitches, [10, 11])

@jsymbolic
@interval
@pitch
def melodic_octaves(pitches: list[int]) -> int:
    """The proportion of intervals that are octaves (12 semitones).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Proportion of intervals that are octaves (12 semitones)
    """
    return variable_melodic_intervals(pitches, 12)

@jsymbolic
@interval
@pitch
def minor_major_third_ratio(pitches: list[int]) -> float:
    """The ratio of minor thirds to major thirds.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Ratio of minor thirds to major thirds, or 0.0 if no major thirds exist
    """
    minor_thirds = variable_melodic_intervals(pitches, 3)
    major_thirds = variable_melodic_intervals(pitches, 4)

    if major_thirds == 0:
        return 0.0

    return minor_thirds / major_thirds

@jsymbolic
@interval
@pitch
def direction_of_melodic_motion(pitches: list[int]) -> float:
    """The proportion of upward melodic motions with regards to the total number of melodic motions.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Proportion of upward melodic motion (0.0 to 1.0), or -1.0 if no intervals
    """
    intervals = pitch_interval(pitches)
    if not intervals:
        return -1.0

    ups = 0
    downs = 0

    for interval in intervals:
        if interval > 0:
            ups += 1
        elif interval < 0:
            downs += 1

    if (ups + downs) == 0:
        return 0.0

    return float(ups) / float(ups + downs)

@jsymbolic
@interval
@pitch
def number_of_common_melodic_intervals(pitches: list[int]) -> int:
    """The number of intervals that appear in at least 9% of melodic transitions.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of significant intervals
    """
    if len(pitches) < 2:
        return 0

    intervals = pitch_interval(pitches)
    absolute_intervals = [abs(iv) for iv in intervals]
    significant_intervals = n_percent_significant_values(absolute_intervals, threshold=0.09)

    return int(len(significant_intervals))

@jsymbolic
@interval
@pitch
def prevalence_of_most_common_melodic_interval(pitches: list[int]) -> float:
    """The proportion of intervals that are the most common interval.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of intervals that are the most common interval, or 0 if no intervals
    """
    intervals = pitch_interval(pitches)
    absolute_intervals = [abs(iv) for iv in intervals]
    if not absolute_intervals:
        return 0

    interval_counts = {}
    for interval in absolute_intervals:
        interval_counts[interval] = interval_counts.get(interval, 0) + 1

    return float(max(interval_counts.values()) / len(absolute_intervals))

@jsymbolic
@interval
@pitch
def relative_prevalence_of_most_common_melodic_intervals(pitches: list[int]) -> float:
    """The ratio of the frequency of the second most common interval to the frequency of the most common interval.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Ratio of second most common interval frequency to most common interval frequency.
        Returns 0.0 if fewer than 2 intervals or only one unique interval.
    """
    intervals = pitch_interval(pitches)
    absolute_intervals = [abs(iv) for iv in intervals]
    
    if len(absolute_intervals) < 2:
        return 0.0
        
    interval_counts = {}
    for interval in absolute_intervals:
        interval_counts[interval] = interval_counts.get(interval, 0) + 1
        
    if len(interval_counts) < 2:
        return 0.0
        
    sorted_intervals = sorted(interval_counts.items(), key=lambda x: x[1], reverse=True)
    most_common_freq = sorted_intervals[0][1] / len(absolute_intervals)
    second_most_freq = sorted_intervals[1][1] / len(absolute_intervals)
    
    return float(second_most_freq / most_common_freq)

def _get_features_by_type(feature_type: str) -> dict:
    """Get all features of a specific type.
    
    Parameters
    ----------
    feature_type : str
        The type of features to collect (e.g., 'descriptives', 'contour', 'tonality', etc.)
        
    Returns
    -------
    dict
        Dictionary mapping feature names to functions
    """
    import inspect
    import sys

    current_module = sys.modules[__name__]
    
    features = {}
    seen_function_ids = set()
    for name, obj in inspect.getmembers(current_module):
        if (inspect.isfunction(obj) and
            hasattr(obj, '_feature_types') and
            feature_type in obj._feature_types):
            # skip aliased functions to avoid repetition
            if obj.__name__ != name:
                continue
            # safeguard using function id
            func_id = id(obj)
            if func_id in seen_function_ids:
                continue
            seen_function_ids.add(func_id)
            features[name] = obj
    
    return features


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
    import inspect
    import sys

    current_module = sys.modules[__name__]
    
    features = {}
    seen_function_ids = set()
    for name, obj in inspect.getmembers(current_module):
        if (inspect.isfunction(obj) and 
            hasattr(obj, '_feature_domain') and 
            obj._feature_domain == domain):
            # skip aliased functions to avoid repetition
            if obj.__name__ != name:
                continue
            # safeguard using function id
            func_id = id(obj)
            if func_id in seen_function_ids:
                continue
            seen_function_ids.add(func_id)
            features[name] = obj
    
    return features


def _get_features_by_domain_and_types(domain: str, allowed_types: list[str]) -> dict:
    """Get all features of a specific domain that match any of the allowed types.
    
    Parameters
    ----------
    domain : str
        The domain of features to collect ('pitch', 'rhythm', or 'both')
    allowed_types : list[str]
        List of allowed feature types (e.g., ['descriptives', 'interval'])
        
    Returns
    -------
    dict
        Dictionary mapping feature names to functions
    """
    import inspect
    import sys

    current_module = sys.modules[__name__]
    
    features = {}
    seen_function_ids = set()
    for name, obj in inspect.getmembers(current_module):
        if (inspect.isfunction(obj) and 
            hasattr(obj, '_feature_domain') and 
            obj._feature_domain == domain and
            hasattr(obj, '_feature_types')):
            # skip aliased functions to avoid repetition
            if obj.__name__ != name:
                continue
            # safeguard using function id
            func_id = id(obj)
            if func_id in seen_function_ids:
                continue
            seen_function_ids.add(func_id)
            # Check if any of the function's types are in allowed_types
            if any(ftype in allowed_types for ftype in obj._feature_types):
                features[name] = obj
    
    return features


def get_pitch_features(melody: Melody) -> Dict:
    """Dynamically collect all pitch features for a melody.
    
    Collects features decorated with @pitch domain and @descriptives type.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of pitch feature values
    """
    features = {}
    pitch_functions = _get_features_by_domain_and_types("pitch", ["descriptives"])
    
    for name, func in pitch_functions.items():
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Call function with appropriate parameters based on signature
            if 'pitches' in params and 'starts' in params and 'ends' in params and 'tempo' in params:
                result = func(melody.pitches, melody.starts, melody.ends, melody.tempo)
            elif 'pitches' in params and 'starts' in params and 'ends' in params:
                result = func(melody.pitches, melody.starts, melody.ends)
            elif 'pitches' in params and 'starts' in params:
                if 'tempo' in params and 'ppqn' in params:
                    result = func(melody.pitches, melody.starts, melody.tempo, 480)
                elif 'tempo' in params:
                    result = func(melody.pitches, melody.starts, melody.tempo)
                else:
                    result = func(melody.pitches, melody.starts)
            elif 'pitches' in params:
                result = func(melody.pitches)
            elif 'starts' in params and 'ends' in params:
                result = func(melody.starts, melody.ends)
            elif 'starts' in params:
                result = func(melody.starts)
            elif 'ends' in params:
                result = func(melody.ends)
            elif 'melody' in params:
                result = func(melody)
            else:
                result = func(melody)

            features[name] = result
        except Exception as e:
            print(f"Warning: Could not compute {name}: {e}")
            features[name] = None

    return features


@fantastic
@contour
@pitch
def get_step_contour_features(
    pitches: list[int], starts: list[float], ends: list[float], tempo: float = 120.0
) -> StepContour:
    """Calculate step contour features.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    StepContour
        StepContour object with global variation, direction and local variation
    """
    if not pitches or not starts or not ends or len(pitches) < 2:
        return 0.0, 0.0, 0.0

    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return 0.0, 0.0, 0.0

    sc = StepContour(pitches, durations)
    return sc.global_variation, sc.global_direction, sc.local_variation

@fantastic
@contour
@pitch
def get_interpolation_contour_features(
    pitches: list[int], starts: list[float]
) -> InterpolationContour:
    """Calculate interpolation contour features.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times

    Returns
    -------
    InterpolationContour
        InterpolationContour object with direction, gradient and class features
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
def get_comb_contour_matrix(pitches: list[int]) -> list[list[int]]:
    """The Marvin & Laprade (1987) comb contour matrix.
    For a melody with n notes, returns an n x n binary matrix C where
    C[i][j] = 1 if pitch of note j is higher than pitch of note i (p[j] > p[i])
    for i >= j (lower triangle including diagonal), and 0 otherwise.

    Parameters
    ----------
    pitches : List[int]
        Sequence of MIDI pitches

    Returns
    -------
    List[List[int]]
        n x n binary matrix (as a list of lists)
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

@fantastic
@contour
@pitch
def get_polynomial_contour_features(
    melody: Melody
) -> PolynomialContour:
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
    """Calculate Huron contour features.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    str
        Huron contour classification
    """
    hc = HuronContour(melody)
    return hc.class_label

@fantastic
@jsymbolic
@rhythm
@descriptives
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
@descriptives
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
@descriptives
def tempo_variability(melody: Melody) -> float:
    """The variability of tempo of the melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Tempo variability of melody
    """
    if not melody.tempo_changes or len(melody.tempo_changes) < 2:
        return 0.0
    return float(np.std([tempo for time, tempo in melody.tempo_changes], ddof=1))

@fantastic
@rhythm
@descriptives
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
@descriptives
def mean_duration(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The mean note duration in quarter notes.

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
        Mean note duration in quarter notes
    """
    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return 0.0
    
    return float(np.mean(durations))

@jsymbolic
@rhythm
@descriptives
def average_note_duration(starts: list[float], ends: list[float]) -> float:
    """The average note duration in seconds.

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
    """
    durations = [end - start for start, end in zip(starts, ends)]
    if not durations:
        return 0.0
    return float(np.mean(durations))

@novel
@rhythm
@descriptives
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
@descriptives
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
@descriptives
def modal_duration(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The most common note duration in quarter notes.

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
        Most frequent note duration in quarter notes
    """
    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return 0.0
    
    return float(get_mode(durations))

@fantastic
@rhythm
@descriptives
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
@rhythm
@descriptives
def length(starts: list[float]) -> float:
    """The total number of notes.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Total number of notes
    """
    return len(starts)

@novel
@rhythm
@descriptives
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
@descriptives
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

    Notes
    -----
    This feature is named 'duration_in_seconds' in JSymbolic.
    """
    return melody.total_duration

@fantastic
@jsymbolic
@rhythm
@descriptives
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
@descriptives
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
@descriptives
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
    if not melody.starts or len(melody.starts) < 2:
        return 0.0

    quarter_note_duration = 60.0 / melody.tempo
    total_duration_seconds = melody.total_duration
    total_duration_quarter_notes = total_duration_seconds / quarter_note_duration

    if total_duration_quarter_notes == 0:
        return 0.0

    return float(len(melody.starts) / total_duration_quarter_notes)

@jsymbolic
@rhythm
@descriptives
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
    """The time between consecutive onsets (inter-onset interval).
    
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
    """The arithmetic mean of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Mean of inter-onset intervals
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
    """The standard deviation of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Standard deviation of inter-onset intervals
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
    """The sequence of inter-onset interval ratios.

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
    """The arithmetic mean of inter-onset interval ratios.

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
    """The standard deviation of inter-onset interval ratios.

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
        Range of inter-onset intervals
    """
    intervals = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
    return max(intervals) - min(intervals)

@novel
@rhythm
@interval
def ioi_contour(starts: list[float]) -> list[int]:
    """The sequence of IOI contour values (-1: shorter, 0: same, 1: longer).

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    list[int]
        Sequence of contour values
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
    """The arithmetic mean of IOI contour values.

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
    """The standard deviation of IOI contour values.

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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
    """
    durations_qn = _get_durations(starts, ends, tempo)
    if not durations_qn:
        return 0.0

    rhythmic_value_histogram_object = create_rhythmic_value_histogram(durations_qn, ppqn=1)
    hist = rhythmic_value_histogram_object.histogram

    return float(hist.get(0, 0.0) + hist.get(1, 0.0) + hist.get(2, 0.0))


@jsymbolic
@rhythm
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
def shortest_rhythmic_value(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The shortest rhythmic value (in quarter notes) among non-empty bins.
    
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
        Shortest rhythmic value (in quarter notes) among non-empty bins (0.0 if empty)
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
@descriptives
def longest_rhythmic_value(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The longest rhythmic value (in quarter notes) among non-empty bins.
    
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
        Longest rhythmic value (in quarter notes) among non-empty bins (0.0 if empty)
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
@descriptives
def mean_rhythmic_value(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The mean rhythmic value (in quarter notes) using the normalized histogram, weighted by the frequency of the rhythmic value.
    
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
        Weighted mean rhythmic value (in quarter notes) using normalized histogram (0.0 if empty)
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
@descriptives
def most_common_rhythmic_value(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The modal rhythmic value (in quarter notes).
    
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
        Modal rhythmic value (in quarter notes) (0.0 if empty or all-zero)
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
def median_rhythmic_value_run_length(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The median run length of identical rhythmic values across the melody. Run length is the number of consecutive 
    notes with the same rhythmic value."""
    runs = _rhythmic_run_lengths(starts, ends, tempo)
    if not runs:
        return 0.0
    return float(np.median(runs))


@jsymbolic
@rhythm
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
    """Count peaks in BPM bins 40..200 whose thresholded value in the given column > 0.001."""
    if not table:
        return 0.0
    n = len(table)
    min_bpm = 40
    max_bpm = min(200, n - 1)
    count = 0
    for b in range(min_bpm, max_bpm + 1):
        if table[b][column_index] > 0.001:
            count += 1
    return float(count)

@jsymbolic
@rhythm
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
        Ratio of higher to lower bin index of the two strongest rhythmic pulses (0.0 if no durations)
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
@descriptives
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
        Ratio of higher to lower bin index of the two strongest rhythmic pulses (120-BPM standardized histogram) (0.0 if no durations)
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
def polyrhythms(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The fraction of beat histogram peaks that are not integer multiples/factors of the highest peak.

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
        Fraction of beat histogram peaks that are not integer multiples/factors of the highest peak (0.0 if no durations)
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
@descriptives
def polyrhythms_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The fraction of beat histogram peaks that are not integer multiples/factors of the highest peak using tempo-standardized histogram.

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
        Fraction of beat histogram peaks that are not integer multiples/factors of the highest peak using tempo-standardized histogram (0.0 if no durations)
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
@descriptives
def number_of_strong_rhythmic_pulses(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The count of BPM bins with pulses greater than 0.001.

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
        Count of BPM bins with sufficiently strong pulses (> 0.001) (0.0 if no durations)
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0

    # Use cached beat histogram tables
    table, _ = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    return _count_strong_pulses(list(table), column_index=0)

@jsymbolic
@rhythm
@descriptives
def number_of_strong_rhythmic_pulses_tempo_standardized(
    starts: list[float],
    ends: list[float],
    tempo: float = 120.0,
    ppqn: int = 480,
) -> float:
    """The count of BPM bins with pulses greater than 0.001 using the tempo-standardized beat histogram.

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
        Count of BPM bins with sufficiently strong pulses (> 0.001) (0.0 if no durations)
    """
    _, std_table = _compute_beat_histogram_tables(tuple(starts), tuple(ends), tempo, ppqn)
    return _count_strong_pulses(list(std_table), column_index=0)

@jsymbolic
@rhythm
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
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
@descriptives
def minimum_note_duration(starts: list[float], ends: list[float]) -> float:
    """The minimum note duration in seconds.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Minimum note duration in seconds
    """
    return min([end - start for start, end in zip(starts, ends)])

@jsymbolic
@rhythm
@descriptives
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
@descriptives
def equal_duration_transitions(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The proportion of duration transitions that are equal in length.
    
    
    Parameters
    ----------
    starts : list[float]
        List of note start times  
    ends : list[float]
        List of note end times
        
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
@descriptives
def half_duration_transitions(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The proportion of duration transitions that are halved or doubled.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

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
@descriptives
def dotted_duration_transitions(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """The proportion of duration transitions that are dotted.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
        
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
@descriptives
def total_number_of_notes(starts: list[float]) -> int:
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

@jsymbolic
@rhythm
@descriptives
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
        Amount of staccato
    """
    if not starts or not ends or len(starts) != len(ends):
        return 0.0
    durations_seconds = [float(end - start) for start, end in zip(starts, ends)]
    if not durations_seconds:
        return 0.0
    short_count = sum(1 for d in durations_seconds if d < 0.1)
    return float(short_count / len(durations_seconds))

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

@novel
@rhythm
@descriptives
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
@descriptives
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
@descriptives
def onset_autocorrelation(starts: list[float], ends: list[float], divisions_per_quarter: int = 4, max_lag_quarters: int = 8) -> list[float]:
    """The autocorrelation function of onset times weighted by duration accents.
    This is calculated by weighting the onset times by the duration accents,
    as defined by Parncutt (1994).
    
    Parameters
    ----------
    starts : list[float]
        List of note start times in seconds
    ends : list[float]
        List of note end times in seconds
    divisions_per_quarter : int, optional
        Divisions per quarter note, by default 4
    max_lag_quarters : int, optional
        Maximum lag in quarter notes, by default 8
        
    Returns
    -------
    list[float]
        Autocorrelation values from lag 0 to max_lag_quarters quarter notes
    """
    expected_length = max_lag_quarters * divisions_per_quarter + 1
    
    if not starts or not ends or len(starts) != len(ends):
        return [0.0] * expected_length
    
    if len(starts) == 0:
        return [0.0] * expected_length
    
    # Get duration accents using Parncutt's model
    duration_accents = duration_accent(starts, ends)
    if not duration_accents:
        return [0.0] * expected_length
    
    # Create onset time grid
    max_onset_time = max(starts) if starts else 0
    grid_length = divisions_per_quarter * max(2 * max_lag_quarters, int(np.ceil(max_onset_time)) + 1)
    onset_grid = np.zeros(grid_length)
    
    # Place accents at quantized onset positions
    for note_idx, onset_time in enumerate(starts):
        if note_idx < len(duration_accents):
            # Quantize onset time to grid divisions
            grid_index = int(np.round(onset_time * divisions_per_quarter)) % len(onset_grid)
            onset_grid[grid_index] += duration_accents[note_idx]
    
    # autocorrelation using scipy's cross-correlation function
    from scipy.signal import correlate
    
    # Compute autocorrelation
    full_autocorr = correlate(onset_grid, onset_grid, mode='full')
    
    # Extract the positive lags up to max_lag_quarters
    center_index = len(full_autocorr) // 2
    autocorr_result = full_autocorr[center_index:center_index + expected_length]
    
    # Normalize by the zero-lag value
    if autocorr_result[0] != 0:
        autocorr_result = autocorr_result / autocorr_result[0]
    else:
        autocorr_result = np.zeros_like(autocorr_result)
    
    return autocorr_result.tolist()

@novel
@rhythm
@descriptives
def onset_autocorr_peak(starts: list[float], ends: list[float], divisions_per_quarter: int = 4, max_lag_quarters: int = 8) -> float:
    """The maximum onset autocorrelation value (excluding lag 0).
    
    Parameters
    ----------
    starts : list[float]
        List of note start times in seconds
    ends : list[float]
        List of note end times in seconds
    divisions_per_quarter : int, optional
        Divisions per quarter note, by default 4
    max_lag_quarters : int, optional
        Maximum lag in quarter notes, by default 8
        
    Returns
    -------
    float
        Maximum autocorrelation value excluding lag 0
    """
    autocorr_values = onset_autocorrelation(starts, ends, divisions_per_quarter, max_lag_quarters)
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

@novel
@tonality
@pitch
def key(
    melody: Melody,
    key_estimation: str = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler"
) -> str:
    """The key of the melody, either read from the MIDI file or estimated using
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
    """
    # Infer key using specified algorithm
    key_name, mode = infer_key_from_pitches(melody.pitches, algorithm=key_finding_algorithm)
    inferred_key = f"{key_name} {mode}" if key_name and mode else None
    
    # Determine which key to use based on strategy
    if key_estimation == "always_infer":
        # Always use inferred key
        return inferred_key if inferred_key else "unknown"
    else:
        # Try to read from MIDI file
        key_from_melody = None
        if melody.has_key_signature:
            key_sig = melody.key_signature
            if key_sig:
                key_from_melody = f"{key_sig[0]} {key_sig[1]}"
        
        if key_estimation == "always_read_from_file":
            if key_from_melody is None:
                raise ValueError(f"No key signature found in MIDI file: {melody.id}")
            return key_from_melody
        else:
            # key_estimation == "infer_if_necessary"
            if key_from_melody is not None:
                # Use key from MIDI
                return key_from_melody
            else:
                # Infer if no MIDI key available
                return inferred_key if inferred_key else "unknown"


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
    return correlation[0][1]

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
        Returns 1.0 if fewer than 2 correlation values.
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)

    if len(correlations) < 2:
        return -1.0

    # Get top 2 correlation values
    top_corr = abs(correlations[0][1])
    second_corr = abs(correlations[1][1])

    # Avoid division by zero
    if second_corr == 0:
        return 1.0

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
        Returns 1.0 if fewer than 2 correlation values or sum is zero.
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)

    if len(correlations) < 2:
        return -1.0

    # Get highest correlation and sum of rest
    top_corr = abs(correlations[0][1])
    other_sum = sum(abs(corr[1]) for corr in correlations[1:])

    # Avoid division by zero
    if other_sum == 0:
        return 1.0

    return top_corr / other_sum

@novel
@tonality
@pitch
def tonal_entropy(pitches: list[int]) -> float:
    """The zeroth-order base-2 entropy of all key correlations.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Entropy of the tonality vector correlation distribution
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)
    if not correlations:
        return -1.0

    # Calculate entropy of correlation distribution
    # Extract just the correlation values and normalize them to positive values
    corr_values = [abs(corr[1]) for corr in correlations]

    # Calculate entropy of the correlation distribution
    return shannon_entropy(corr_values)


def _get_key_distances() -> dict[str, int]:
    """Returns a dictionary mapping key names to their semitone distances from C.
    
    Includes both sharp and flat enharmonic equivalents.

    Returns
    -------
    dict[str, int]
        Dictionary mapping key names (both major and minor) to semitone distances from C.
    """
    return {
        "C": 0,
        "C#": 1, "Db": 1,
        "D": 2,
        "D#": 3, "Eb": 3,
        "E": 4, "Fb": 4,
        "F": 5, "E#": 5,
        "F#": 6, "Gb": 6,
        "G": 7,
        "G#": 8, "Ab": 8,
        "A": 9,
        "A#": 10, "Bb": 10,
        "B": 11, "Cb": 11,
        # Minor keys (lowercase)
        "c": 0,
        "c#": 1, "db": 1,
        "d": 2,
        "d#": 3, "eb": 3,
        "e": 4, "fb": 4,
        "f": 5, "e#": 5,
        "f#": 6, "gb": 6,
        "g": 7,
        "g#": 8, "ab": 8,
        "a": 9,
        "a#": 10, "bb": 10,
        "b": 11, "cb": 11,
    }


@idyom
@tonality
@pitch
def referent(melody: Melody) -> int:
    """Calculate the referent (root note) of a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    int
        The referent (root note) of the strongest key
    """
    pitches = melody.pitches
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)
    
    if correlations:
        key_name = correlations[0][0].split()[0]
        key_distances = _get_key_distances()
        return key_distances[key_name]
    else:
        return -1

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
    key_estimation: str = "infer_if_necessary",
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
    key_estimation : str, optional
        Key estimation strategy: "always_read_from_file", "infer_if_necessary", or "always_infer".
        Default is "infer_if_necessary".
    key_finding_algorithm : Literal["krumhansl_schmuckler"], optional
        Key-finding algorithm to use when inferring key. Default is "krumhansl_schmuckler".

    Returns
    -------
    dict
        Dictionary containing the tonal tension features: `cloud_diameter`, 
        `cloud_momentum`, `tensile_strain`, ordered by `onset`.

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
@novel
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
@novel
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
@novel
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
@novel
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
@novel
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
@novel
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
    """
    A histogram of Krumhansl-Schmuckler correlation values.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    dict
        Histogram of KS correlation values

    Citation
    --------
    Krumhansl (1990)
    """
    p = [p % 12 for p in pitches]
    return histogram_bins(compute_tonality_vector(p)[0][1], 24)

@idyom
@midi_toolbox
@pitch
@expectation
def narmour_registral_direction(pitches: list[int]) -> int:
    """The score is set to zero. If an interval greater than a perfect fifth is followed by a direction change, a score
    of 1 is given. If an interval smaller than a perfect fourth continues in the same direction, 
    a score of 1 is given. This feature returns either 0 or 1 accordingly.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Narmour registral direction score (0 or 1)

    Citation
    --------
    Narmour (1990)
    """
    return int(registral_direction(pitches))

@idyom
@midi_toolbox
@pitch
@expectation
def narmour_proximity(pitches: list[int]) -> int:
    """Proximity is defined as 6 minus the absolute interval between the last two notes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Narmour proximity score (0 to 6)

    Citation
    --------
    Narmour (1990)
    """
    return int(proximity(pitches))

@idyom
@midi_toolbox
@pitch
@expectation
def narmour_closure(pitches: list[int]) -> int:
    """A score of 1 is given if the last three notes in a melody constitute a change in
    direction. Another score of 1 is given if the final interval is more than one tone
    smaller than the penultimate. As such, this returns integer values between 0 and 2.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Narmour closure score (0 to 2)

    Citation
    --------
    Narmour (1990)
    """
    return int(closure(pitches))

@idyom
@midi_toolbox
@pitch
@expectation
def narmour_registral_return(pitches: list[int]) -> int:
    """If the last three notes move away from and then back to the same pitch, a score
    of 3 is returned. If the pitch returned to is 1 semitone away from the initial,
    returns 2. If the pitch returned to is 2 semitones away from the initial, returns 1. 
    Otherwise, a score of 0 is returned.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Narmour registral return score (0 to 3)

    Citation
    --------
    Narmour (1990)
    """
    return int(registral_return(pitches))

@idyom
@midi_toolbox
@pitch
@expectation
def narmour_intervallic_difference(pitches: list[int]) -> int:
    """If a large interval is followed by a smaller interval, returns 1 if either:
    - The smaller interval continues in the same direction and is at least 3 semitones smaller
    - The smaller interval changes direction and is at least 2 semitones smaller
    Additionally, returns 1 if a small interval is followed by another interval of the same size.
    Otherwise returns 0.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Narmour intervallic difference score (0 or 1)

    Citation
    --------
    Narmour (1990)
    """
    return int(intervallic_difference(pitches))

def get_narmour_features(melody: Melody) -> Dict:
    """Calculate Narmour's implication-realization features.

    Parameters
    ----------
    melody : Melody
        The melody to analyze as a Melody object

    Returns
    -------
    Dict
        Dictionary containing scores for:
        - Registral direction (0 or 1)
        - Proximity (0-6)
        - Closure (0-2)
        - Registral return (0-3)
        - Intervallic difference (0 or 1)

    Notes
    -----
    Features represent:
    - Registral direction: Large intervals followed by direction change
    - Proximity: Closeness of consecutive pitches
    - Closure: Direction changes and interval size changes
    - Registral return: Return to previous pitch level
    - Intervallic difference: Relationship between consecutive intervals
    """
    pitches = melody.pitches
    return {
        "registral_direction": narmour_registral_direction(pitches),
        "proximity": narmour_proximity(pitches),
        "closure": narmour_closure(pitches),
        "registral_return": narmour_registral_return(pitches),
        "intervallic_difference": narmour_intervallic_difference(pitches),
    }


# Melodic Movement Features
@jsymbolic
@pitch
@descriptives
def amount_of_arpeggiation(pitches: list[int]) -> float:
    """The proportion of pitch intervals in the melody that constitute triadic movements.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of intervals that match arpeggio patterns (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.
    """
    return arpeggiation_proportion(pitches)


@jsymbolic
@pitch
@descriptives
def chromatic_motion(pitches: list[int]) -> float:
    """The proportion of chromatic motion in the melody. Chromatic motion is defined as a melodic interval of 1 semitone.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of intervals that are chromatic (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.
    """
    return chromatic_motion_proportion(pitches)

@jsymbolic
@pitch
@descriptives
def melodic_embellishment(
    pitches: list[int], starts: list[float], ends: list[float]
) -> float:
    """The proportion of melodic embellishments in the melody. Melodic embellishments are identified by notes 
    that are surrounded on both sides by notes with durations at least 3 times longer than the central 
    note.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Proportion of notes that are embellishments (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty.
    """
    if not pitches or not starts or not ends:
        return -1.0
    if len(pitches) != len(starts) or len(starts) != len(ends):
        return -1.0
    if len(pitches) == 0:
        return 0.0

    durations = [end - start for start, end in zip(starts, ends)]

    embellishment_count = 0
    for i in range(1, len(pitches) - 1):
        # Check if surrounded by notes with duration >= 3x this note
        if (durations[i-1] >= 3 * durations[i] and 
            durations[i+1] >= 3 * durations[i]):
            embellishment_count += 1

    return float(embellishment_count) / len(pitches)

@jsymbolic
@pitch
@descriptives
def repeated_notes(pitches: list[int]) -> float:
    """The proportion of repeated notes in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of intervals that are repeated notes (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.
    """
    return repeated_notes_proportion(pitches)

@jsymbolic
@pitch
@descriptives
def stepwise_motion(pitches: list[int]) -> float:
    """The proportion of stepwise motion in the melody. Stepwise motion is defined as a melodic interval of 1 or 2 semitones.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of intervals that are stepwise (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.
    """
    return stepwise_motion_proportion(pitches)

@midi_toolbox
@pitch
@complexity
def gradus(pitches: list[int]) -> int:
    """The degree of melodiousness based on Euler's gradus suavitatis.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    
    Citation
    --------
    Euler (1739)

    Returns
    -------
    int
        Mean gradus suavitatis value across all intervals, where lower values 
        indicate higher melodiousness.
    """
    if len(pitches) < 2:
        return 0
    
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
    
    return int(np.mean(gradus_values)) if gradus_values else 0

@midi_toolbox
@pitch
@expectation
def mobility(pitches: list[int]) -> list[float]:
    """The melodic mobility for each note based on von Hippel (2000).
    Mobility describes why melodies change direction after large skips by 
    observing that they would otherwise run out of the comfortable melodic range.
    It uses lag-one autocorrelation between successive pitch heights.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    list[float]
        Absolute mobility value for each note in the sequence

    Citation
    --------
    von Hippel (2000)
    """
    if len(pitches) < 2:
        return [0.0] if len(pitches) == 1 else []
    
    mobility_values = [0.0]  # First note gets 0
    
    for i in range(2, len(pitches) + 1):  # Start from note 2 (index 1)
        if i == 2:
            mobility_values.append(0.0)  # Second note gets 0
            continue
            
        # Calculate mean of previous pitches (notes 1 to i-1)
        mean_prev = np.mean(pitches[:i-1])
        
        # Calculate deviations from mean for correlation
        p = [pitches[j] - mean_prev for j in range(i-1)]
        
        if len(p) < 2:
            mobility_values.append(0.0)
            continue
            
        # Create lagged series for correlation
        p_current = p[:-1]  # p[0] to p[i-3]
        p_lagged = p[1:]    # p[1] to p[i-2]
        
        if len(p_current) < 2 or len(p_lagged) < 2:
            mobility_values.append(0.0)
            continue
            
        # Calculate correlation coefficient
        try:
            # Check for variance before computing correlation to avoid zero division errors
            if np.var(p_current) == 0 or np.var(p_lagged) == 0:
                correlation = 0.0
            else:
                correlation_matrix = np.corrcoef(p_current, p_lagged)
                correlation = correlation_matrix[0, 1]
                
                # Handle NaN correlation (when no variance)
                if np.isnan(correlation):
                    correlation = 0.0
                
        except (ValueError, np.linalg.LinAlgError):
            correlation = 0.0
        
        # Calculate mobility for current note
        # mob(i) * (pitch(i) - mean_prev)
        current_deviation = pitches[i-2] - mean_prev  # Previous note deviation
        mob_value = correlation * current_deviation
        mobility_values.append(abs(mob_value))
    
    return mobility_values

@novel
@pitch
@expectation
def mean_mobility(pitches: list[int]) -> float:
    """The arithmetic mean of the mobility values across all notes.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Mean mobility value
    """
    mob_values = mobility(pitches)
    if not mob_values:
        return 0.0
    return float(np.mean(mob_values))


@novel
@pitch
@expectation
def mobility_std(pitches: list[int]) -> float:
    """The standard deviation of the mobility values across all notes.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Standard deviation of mobility values
    """
    mob_values = mobility(pitches)
    if len(mob_values) < 2:
        return 0.0
    return float(np.std(mob_values, ddof=1))


def _stability_distance(weight1: float, weight2: float, proximity: float) -> float:
    """Calculate stability distance for melodic attraction.
    
    Helper function implementing the stabilitydistance subfunction from melattraction.m
    
    Parameters
    ----------
    weight1 : float
        Anchoring weight of first note
    weight2 : float  
        Anchoring weight of second note
    proximity : float
        Distance in semitones between notes
        
    Returns
    -------
    float
        Stability distance value
    """
    if weight1 == 0 or proximity == 0:
        return 0.0

    return (weight2 / weight1) * (1.0 / (proximity ** 2))

@midi_toolbox
@pitch
@expectation
def melodic_attraction(pitches: list[int]) -> list[float]:
    """The melodic attraction according to Lerdahl (1996).
    Each tone in a key has certain anchoring strength ("weight") in tonal pitch space.
    Melodic attraction strength is affected by the distance between tones and 
    directed motion patterns.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Citation
    --------
    Lerdahl (1996)
        
    Returns
    -------
    list[float]
        Melodic attraction values for each note (0-1 scale, higher = more attraction)
    """
    if len(pitches) < 2:
        return [0.0] if len(pitches) == 1 else []

    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)

    if not correlations:
        return [0.0] * len(pitches)

    key_name = correlations[0][0].split()[0]
    is_major = "major" in correlations[0][0]

    # Get tonic pitch class for transposition to C
    key_distances = _get_key_distances()
    tonic_pc = key_distances[key_name]

    transposed_pcs = [(pc - tonic_pc) % 12 for pc in pitch_classes]
    
    # Anchoring weights for each pitch class (C=0, C#=1, ..., B=11)
    if is_major:
        anchor_weights = [4, 1, 2, 1, 3, 2, 1, 3, 1, 2, 1, 2]  # MAJOR
    else:
        anchor_weights = [4, 1, 2, 3, 1, 2, 1, 3, 2, 2, 1, 2]  # MINOR
    
    pc_weights = [anchor_weights[pc] for pc in transposed_pcs]
    
    # Calculate directed motion index
    # (change of direction = -1, repetition = 0, continuation = 1)
    pitch_diffs = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
    directions = [1 if diff > 0 else -1 if diff < 0 else 0 for diff in pitch_diffs]
    
    motion = [0]
    for i in range(1, len(directions)):
        if directions[i] == 0:
            motion.append(0)
        elif i == 0 or directions[i-1] == 0:  # First direction or after repetition
            motion.append(1)
        elif directions[i] == directions[i-1]:  # Continuation
            motion.append(1)
        else:  # Direction change
            motion.append(-1)
    
    attraction_values = [0.0]
    
    for i in range(len(pitches) - 1):
        current_weight = pc_weights[i]
        next_weight = pc_weights[i + 1]
        proximity = abs(pitches[i + 1] - pitches[i])
        
        # Primary attraction (sd1)
        if current_weight >= next_weight:
            sd1 = 0.0
        else:
            sd1 = _stability_distance(current_weight, next_weight, proximity)
        
        # Alternative attraction (sd2) - attraction to other stable tones
        current_pc = transposed_pcs[i]
        
        # Check other pitch classes for stronger alternatives
        sd2_values = []
        for candidate_pc in range(12):
            candidate_weight = anchor_weights[candidate_pc]
            
            # Only consider stable candidates
            if candidate_weight > current_weight and candidate_pc != transposed_pcs[i + 1]:
                candidate_distance = min(abs(candidate_pc - current_pc), 12 - abs(candidate_pc - current_pc))
                sd2_candidate = _stability_distance(current_weight, candidate_weight, candidate_distance)
                sd2_values.append(sd2_candidate)
        
        # Calculate total alternative attraction
        if len(sd2_values) > 1:
            # Take max + half of others
            max_sd2 = max(sd2_values)
            other_sd2 = sum(val * 0.5 for val in sd2_values if val != max_sd2)
            sd2 = max_sd2 + other_sd2
        elif len(sd2_values) == 1:
            sd2 = sd2_values[0]
        else:
            sd2 = 0.0
        
        # Combine with directed motion
        anchoring = sd1 - sd2
        attraction = motion[i] + anchoring
        
        attraction_values.append(attraction)

    # Scale results between 0 and 1
    scaled_attraction = [(val + 1) / 5 for val in attraction_values]

    # Clamp to [0, 1]
    scaled_attraction = [max(0.0, min(1.0, val)) for val in scaled_attraction]

    return scaled_attraction

@novel
@pitch
@expectation
def mean_melodic_attraction(pitches: list[int]) -> float:
    """The arithmetic mean of the melodic attraction values across all notes.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Mean melodic attraction value
    """
    attraction_values = melodic_attraction(pitches)
    if not attraction_values:
        return 0.0
    return float(np.mean(attraction_values))

@novel
@pitch
@expectation
def melodic_attraction_std(pitches: list[int]) -> float:
    """The standard deviation of the melodic attraction values across all notes.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Standard deviation of melodic attraction values
    """
    attraction_values = melodic_attraction(pitches)
    if len(attraction_values) < 2:
        return 0.0
    return float(np.std(attraction_values, ddof=1))


@novel
@pitch
@expectation
def mean_melodic_accent(pitches: list[int]) -> float:
    """The arithmetic mean of the melodic accent values across all notes.
    Melodic accent is defined by Thomassen's model (1982) according to the 
    possible melodic contours arising in 3-pitch windows.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Mean melodic accent value
    """
    accents = melodic_accent(pitches)
    if not accents:
        return 0.0
    return float(np.mean(accents))

@novel
@pitch
@expectation
def melodic_accent_std(pitches: list[int]) -> float:
    """The standard deviation of the melodic accent values across all notes.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Standard deviation of melodic accent values
    """
    accents = melodic_accent(pitches)
    if not accents:
        return 0.0
    return float(np.std(accents, ddof=1))

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

@fantastic
def get_ngram_document_frequency(ngram: tuple, corpus_stats: dict) -> int:
    """Retrieve the document frequency for a given n-gram from the corpus statistics.

    Parameters
    ----------
    ngram : tuple
        The n-gram to look up
    corpus_stats : dict
        Dictionary containing corpus statistics

    Returns
    -------
    int
        Document frequency count for the n-gram
    """
    # Get document frequencies dictionary once
    doc_freqs = corpus_stats.get("document_frequencies", {})

    # Convert ngram to string only once
    ngram_str = str(ngram)

    # Look up the count directly
    return doc_freqs.get(ngram_str, {}).get("count", 0)

@fantastic
@both
@complexity
class InverseEntropyWeighting:
    """Calculate local weights for n-grams using an inverse-entropy measure. 

    Inverse-entropy weighting is implemented following the specification in 
    FANTASTIC and the Handbook of Latent Semantic Analysis (Landauer et al., 2007).
    It provides several quantifiers of the importance of an n-gram (here: m-type)
    based on its relative frequency in a given passage (here: melody)
    and its relative frequency in that passage as compared to the reference corpus.

    This class contains functions to compute the local weight of an m-type,
    the global weight of an m-type, and the combined weight of an m-type.
    """
    def __init__(self, ngram_counts: dict, corpus_stats: dict):
        self.ngram_counts = ngram_counts
        self.corpus_stats = corpus_stats

    @property
    def local_weights(self) -> list[float]:
        """Calculate local weights for n-grams using an inverse-entropy measure.
        The local weight of an m-type is defined as 
        `loc.w(τ) = log2(f(τ) + 1)` where `f(τ)` is the frequency of a 
        given m-type in the melody. As such, the local weight can take any real value 
        greater than zero. High values mean that the m-type provides a lot of information
        about the melody, while low values mean that the m-type provides little information.

        Parameters
        ----------
        ngram_counts : dict
            Dictionary containing n-gram counts

        Returns
        -------
        list[float]
            List of local weights, x >= 0 for all x in list
        """
        if not self.ngram_counts:
            return []

        local_weights = []
        for tf in self.ngram_counts.values():
            local_weight = np.log2(tf + 1)
            local_weights.append(local_weight)

        return local_weights
    
    @property
    def global_weights(self) -> list[float]:
        """Calculate global weights for n-grams using an inverse-entropy measure.
        First, a ratio between the frequency of an m-type in the melody and the frequency
        of the same m-type in the corpus is calculated:
        `Pc(τ) = fc(τ)/fC(τ)` where `fc(τ)` is the frequency of a given m-type in the melody,
        and `fC(τ)` is the frequency of the same m-type in the corpus.
        This ratio is then used to calculate the global weight of an m-type: 
        `glob.w = 1 + Σ Pc(τ) · log2(Pc(τ)) / log2(|C|)` where `|C|` is the number of 
        documents in the corpus.
        Global weights take a value from 0 to 1. A high value corresponds to a less informative m-type,
        while a low value corresponds to a more informative m-type, with regard to its position in the melody.

        Parameters
        ----------
        ngram_counts : dict
            Dictionary containing n-gram counts
        corpus_stats : dict
            Dictionary containing corpus statistics

        Returns
        -------
        list[float]
            List of global weights, 0 <= x <= 1 for all x in list
        """
        if not self.ngram_counts or not self.corpus_stats:
            return []

        doc_freqs = self.corpus_stats.get("document_frequencies", {})
        total_docs = len(doc_freqs) if doc_freqs else 1

        global_weights = []
        for ngram, tf in self.ngram_counts.items():
            ngram_str = str(ngram)
            df = doc_freqs.get(ngram_str, {}).get("count", 0)

            if df > 0 and total_docs > 0:
                pc_ratio = tf / df if df > 0 else 0.0

                if pc_ratio > 0:
                    entropy_term = pc_ratio * np.log2(pc_ratio)
                    global_weight = 1 + entropy_term / np.log2(total_docs)
                else:
                    global_weight = 1.0
            else:
                global_weight = 1.0

            global_weights.append(global_weight)

        return global_weights

    @property
    def combined_weights(self) -> list[float]:
        """Calculate combined local-global weights for n-grams.
        The combined weight of an m-type is the product of the local and global weights.
        It summarises the relationship between distinctiveness of an m-type compared to the corpus
        and its frequency in the melody. A high combined weight indicates that the m-type is both
        distinctive and frequent in the melody, while a low combined weight indicates that the m-type
        is either not distinctive or not frequent in the melody.

        Parameters
        ----------
        ngram_counts : dict
            Dictionary containing n-gram counts
        corpus_stats : dict
            Dictionary containing corpus statistics

        Returns
        -------
        list[float]
            List of combined weights, x >= 0 for all x in list
        """
        if not self.ngram_counts or not self.corpus_stats:
            return []
    
        if len(self.local_weights) != len(self.global_weights):
            return []

        return [l * g for l, g in zip(self.local_weights, self.global_weights)]

def _get_simonton_transition_matrix() -> np.ndarray:
    """Get Simonton's pitch class transition probabilities from 15,618 classical themes.
    
    This is basically just refstat('pcdist2classical1') from MIDI toolbox.
    Matrix indices correspond to an enumeration of the 12 pitch classes.
    
    Returns
    -------
    np.ndarray
        12x12 matrix of transition probabilities
    """
    transition_matrix = np.zeros((12, 12))
    
    transition_matrix[4, :] = 0.005  
    transition_matrix[9, :] = 0.005  
    transition_matrix[11, :] = 0.005  
    transition_matrix[:, 4] = 0.005  
    transition_matrix[:, 9] = 0.005  
    transition_matrix[:, 11] = 0.005  
    transition_matrix[7, 8] = 0.005  
    transition_matrix[8, 7] = 0.005  
    
    common_transitions = [
        (8, 8, 0.067),  
        (1, 1, 0.053),  
        (8, 1, 0.049),  
        (1, 3, 0.044),  
        (1, 12, 0.032), 
        (1, 8, 0.032),  
        (8, 6, 0.031),  
        (5, 5, 0.030),  
        (5, 3, 0.030),  
        (3, 1, 0.030),  
        (8, 5, 0.029),  
        (8, 10, 0.029), 
        (5, 6, 0.028),  
        (5, 8, 0.026),  
        (3, 5, 0.024),  
        (12, 1, 0.023), 
        (1, 5, 0.022),  
        (6, 8, 0.021),  
        (6, 5, 0.021),  
        (10, 8, 0.020), 
        (4, 3, 0.018),  
        (5, 1, 0.016),  
        (3, 4, 0.014),  
        (10, 12, 0.012),
        (12, 10, 0.011),
        (3, 3, 0.011),  
        (9, 8, 0.011),  
    ]
    
    # convert from 1-indexed MATLAB to 0-indexed Python and set probabilities
    for from_pc_matlab, to_pc_matlab, prob in common_transitions:
        from_pc = (from_pc_matlab - 1) % 12
        to_pc = (to_pc_matlab - 1) % 12
        transition_matrix[from_pc, to_pc] = prob
    
    return transition_matrix

@midi_toolbox
@pitch
@expectation
def compltrans(melody: Melody) -> float:
    """The melodic originality measure, according to Simonton (1984).
    Calculated based on 2nd order pitch-class distribution derived from 15,618 classical music themes.
    Higher values indicate higher melodic originality (less predictable transitions).
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    float
        Originality score scaled 0-10 (higher = more original/unexpected)

    Citation
    --------
    Simonton (1984)
    """
    if not melody.pitches or len(melody.pitches) < 2:
        return 5.0  # Return neutral originality for edge cases
    
    melody_pitch_classes = [pitch % 12 for pitch in melody.pitches]
    
    melody_transition_matrix = np.zeros((12, 12))
    for i in range(len(melody_pitch_classes) - 1):
        from_pitch_class = melody_pitch_classes[i]
        to_pitch_class = melody_pitch_classes[i + 1]
        melody_transition_matrix[from_pitch_class, to_pitch_class] += 1

    classical_transition_probabilities = _get_simonton_transition_matrix()

    transition_probability_products = melody_transition_matrix * classical_transition_probabilities
    total_weighted_probability = np.sum(transition_probability_products)
    total_melody_transitions = len(melody_pitch_classes) - 1
    
    if total_melody_transitions == 0:
        return 5.0
    
    average_transition_probability = total_weighted_probability / total_melody_transitions
    inverted_probability = average_transition_probability * -1.0
    
    # Apply Simonton's scaling formula (0-10 scale, 10 = most original)
    simonton_originality_score = (inverted_probability + 0.0530) * 188.68
    
    return float(simonton_originality_score)

@midi_toolbox
@pitch
@rhythm
@both
@complexity
def complebm(melody: Melody, method: str = 'o') -> float:
    """Expectancy-based melodic complexity, according to Eerola & North (2000).
    Calculated using an expectancy-based model that considers pitch patterns,
    rhythmic features, or both. The complexity score is normalized against the Essen folksong
    collection, where a score of 5 represents average complexity (standard deviation = 1).
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    method : str, optional
        Complexity method: 'p' = pitch only, 'r' = rhythm only, 'o' = optimal combination
        
    Returns
    -------
    float
        Complexity value calibrated to Essen collection (higher = more complex)

    Citation
    --------
    Eerola & North (2000)
    """
    if not melody.pitches or len(melody.pitches) < 2:
        return 5.0  # Return neutral complexity for edge cases

    method = method.lower()

    if method == 'p':
        constant = -0.2407

        melodic_intervals = pitch_interval(melody.pitches)
        average_interval_component = float(np.mean(melodic_intervals)) * 0.3 if melodic_intervals else 0.0

        pitch_class_distribution = pcdist1(melody.pitches, melody.starts, melody.ends)
        pitch_class_entropy_component = shannon_entropy(list(pitch_class_distribution.values())) * 1.0 if pitch_class_distribution else 0.0

        interval_distribution = ivdist1(melody.pitches, melody.starts, melody.ends, melody.tempo)
        interval_entropy_component = shannon_entropy(list(interval_distribution.values())) * 0.8 if interval_distribution else 0.0

        melodic_attraction_values = melodic_attraction(melody.pitches)
        duration_accent_values = duration_accent(melody.starts, melody.ends)

        # Align arrays to same length
        min_length = min(len(melodic_attraction_values), len(duration_accent_values))
        if min_length > 0:
            tonality_duration_products = [a * d for a, d in zip(melodic_attraction_values[:min_length], duration_accent_values[:min_length])]
            tonality_component = float(np.mean(tonality_duration_products)) * -1.0
        else:
            tonality_component = 0.0

        # Combine components using Essen-calibrated formula
        pitch_complexity = (constant + average_interval_component + pitch_class_entropy_component + interval_entropy_component + tonality_component) / 0.9040
        pitch_complexity = pitch_complexity + 5

    elif method == 'r':
        constant = -0.7841

        note_durations = _get_durations(melody.starts, melody.ends)
        duration_entropy_component = shannon_entropy(note_durations) * 0.7 if note_durations else 0.0

        note_density_component = note_density(melody) * 0.2

        positive_durations = [d for d in note_durations if d > 0]
        if positive_durations:
            log_durations = [math.log(d) for d in positive_durations]
            rhythmic_variability_component = float(np.std(log_durations, ddof=1)) * 0.5
        else:
            rhythmic_variability_component = 0.0

        metric_accent_features = get_metric_accent_features(melody)
        meter_accent_component = float(metric_accent_features.get("meter_accent", 0)) * 0.5

        # Combine components using Essen-calibrated formula
        rhythm_complexity = (constant + duration_entropy_component + note_density_component + rhythmic_variability_component + meter_accent_component) / 0.3637
        rhythm_complexity = rhythm_complexity + 5

    elif method == 'o':
        constant = -1.9025

        melodic_intervals = pitch_interval(melody.pitches)
        average_interval_component = float(np.mean(melodic_intervals)) * 0.2 if melodic_intervals else 0.0

        pitch_class_distribution = pcdist1(melody.pitches, melody.starts, melody.ends)
        pitch_class_entropy_component = shannon_entropy(list(pitch_class_distribution.values())) * 1.5 if pitch_class_distribution else 0.0

        interval_distribution = ivdist1(melody.pitches, melody.starts, melody.ends, melody.tempo)
        interval_entropy_component = shannon_entropy(list(interval_distribution.values())) * 1.3 if interval_distribution else 0.0

        melodic_attraction_values = melodic_attraction(melody.pitches)
        duration_accent_values = duration_accent(melody.starts, melody.ends)

        min_length = min(len(melodic_attraction_values), len(duration_accent_values))
        if min_length > 0:
            tonality_duration_products = [a * d for a, d in zip(melodic_attraction_values[:min_length], duration_accent_values[:min_length])]
            tonality_component = float(np.mean(tonality_duration_products)) * -1.0
        else:
            tonality_component = 0.0

        note_durations = _get_durations(melody.starts, melody.ends)
        duration_entropy_component = shannon_entropy(note_durations) * 0.5 if note_durations else 0.0

        note_density_component = note_density(melody) * 0.4

        positive_durations = [d for d in note_durations if d > 0]
        if positive_durations:
            log_durations = [math.log(d) for d in positive_durations]
            rhythmic_variability_component = float(np.std(log_durations, ddof=1)) * 0.9
        else:
            rhythmic_variability_component = 0.0

        metric_accent_features = get_metric_accent_features(melody)
        meter_accent_component = float(metric_accent_features.get("meter_accent", 0)) * 0.8

        # Combine all components using Essen-calibrated formula
        optimal_complexity = (constant + average_interval_component + pitch_class_entropy_component + interval_entropy_component + tonality_component + duration_entropy_component + note_density_component + rhythmic_variability_component + meter_accent_component) / 1.5034
        optimal_complexity = optimal_complexity + 5

    else:
        raise ValueError("Method must be 'p' (pitch), 'r' (rhythm), or 'o' (optimal)")
    
    if method == 'p':
        return float(pitch_complexity)
    elif method == 'r':
        return float(rhythm_complexity)
    else:
        return float(optimal_complexity)


def get_complexity_features(melody: Melody, phrase_gap: float = 1.5, max_ngram_order: int = 6) -> Dict:
    """Dynamically collect all complexity features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    phrase_gap : float, optional
        Phrase gap for mtype features (default: 1.5)
    max_ngram_order : int, optional
        Maximum n-gram order for mtype features (default: 6)
        
    Returns
    -------
    Dict
        Dictionary of complexity feature values
    """
    features = {}
    complexity_functions = _get_features_by_type(FeatureType.COMPLEXITY)
    
    for name, func in complexity_functions.items():
        try:
            # Skip classes/functions that require special parameters
            if name in ('InverseEntropyWeighting', 'get_mtype_features'):
                continue
                
            # Get function signature to determine parameters
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Call function with appropriate parameters
            if 'melody' in params and 'method' in params:
                # Special case for complebm with different methods
                if name == 'complebm':
                    features[f"{name}_pitch"] = func(melody, 'p')
                    features[f"{name}_rhythm"] = func(melody, 'r')
                    features[f"{name}_optimal"] = func(melody, 'o')
                    continue
                else:
                    result = func(melody, 'o')  # Default method
            elif 'melody' in params:
                result = func(melody)
            elif 'starts' in params and 'ends' in params and 'tau' in params:
                # edge case for mean_duration_accent
                result = func(melody.starts, melody.ends, 0.5, 2.0)
            elif 'pitches' in params and 'starts' in params and 'ends' in params:
                result = func(melody.pitches, melody.starts, melody.ends)
            elif 'pitches' in params:
                result = func(melody.pitches)
            else:
                # Try with melody object
                result = func(melody)
            
            # Handle functions that return lists (like mobility)
            if isinstance(result, list) and len(result) > 1:
                features[f"{name}_mean"] = np.mean(result)
                features[f"{name}_std"] = np.std(result, ddof=1) if len(result) > 1 else 0.0
            else:
                features[name] = result
                
        except Exception as e:
            print(f"Warning: Could not compute {name}: {e}")
            features[name] = None

    mtype_features = get_mtype_features(melody, phrase_gap=phrase_gap, max_ngram_order=max_ngram_order)
    features.update(mtype_features)
    
    return features

@fantastic
@corpus_prevalence
@both
def tfdf_spearman(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Calculate Spearman correlation between term frequency and document frequency.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider
        
    Returns
    -------
    float
        Spearman correlation coefficient between TF and DF
    """
    tokenizer = FantasticTokenizer()
    segments = tokenizer.segment_melody(melody, phrase_gap=phrase_gap, units="quarters")

    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, segment.starts, segment.ends
        )
        all_tokens.extend(segment_tokens)

    doc_freqs = corpus_stats.get("document_frequencies", {})
    
    all_tf = []
    all_df = []
    
    for n in range(1, max_ngram_order):
        ngram_counts = {}
        for i in range(len(all_tokens) - n + 1):
            ngram = tuple(all_tokens[i : i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        if not ngram_counts:
            continue

        for ngram, tf in ngram_counts.items():
            ngram_str = str(ngram)
            df = doc_freqs.get(ngram_str, {}).get("count", 0)
            if df > 0:
                all_tf.append(tf)
                all_df.append(df)

    if len(all_tf) >= 2:
        try:
            tf_variance = np.var(all_tf)
            df_variance = np.var(all_df)
            
            if tf_variance == 0 or df_variance == 0:
                return 0.0
            else:
                spearman = scipy.stats.spearmanr(all_tf, all_df)[0]
                return float(spearman if not np.isnan(spearman) else 0.0)
        except:
            return 0.0
    else:
        return 0.0

@fantastic
@corpus_prevalence
@both
def tfdf_kendall(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Calculate Kendall's tau correlation between term frequency and document frequency.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider
        
    Returns
    -------
    float
        Kendall's tau correlation coefficient between TF and DF
    """
    tokenizer = FantasticTokenizer()
    segments = tokenizer.segment_melody(melody, phrase_gap=phrase_gap, units="quarters")

    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, segment.starts, segment.ends
        )
        all_tokens.extend(segment_tokens)

    doc_freqs = corpus_stats.get("document_frequencies", {})
    
    all_tf = []
    all_df = []
    
    for n in range(1, max_ngram_order):
        ngram_counts = {}
        for i in range(len(all_tokens) - n + 1):
            ngram = tuple(all_tokens[i : i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        if not ngram_counts:
            continue

        for ngram, tf in ngram_counts.items():
            ngram_str = str(ngram)
            df = doc_freqs.get(ngram_str, {}).get("count", 0)
            if df > 0:
                all_tf.append(tf)
                all_df.append(df)

    if len(all_tf) >= 2:
        try:
            tf_variance = np.var(all_tf)
            df_variance = np.var(all_df)
            
            if tf_variance == 0 or df_variance == 0:
                return 0.0
            else:
                kendall = scipy.stats.kendalltau(all_tf, all_df)[0]
                return float(kendall if not np.isnan(kendall) else 0.0)
        except:
            return 0.0
    else:
        return 0.0

@fantastic
@corpus_prevalence
@both
def mean_log_tfdf(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Calculate mean log TF-DF score across all n-grams.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider
        
    Returns
    -------
    float
        Mean log TF-DF score
    """
    tokenizer = FantasticTokenizer()
    segments = tokenizer.segment_melody(melody, phrase_gap=phrase_gap, units="quarters")

    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, segment.starts, segment.ends
        )
        all_tokens.extend(segment_tokens)

    doc_freqs = corpus_stats.get("document_frequencies", {})
    total_docs = len(doc_freqs)
    
    tfdf_values = []
    
    for n in range(1, max_ngram_order):
        ngram_counts = {}
        for i in range(len(all_tokens) - n + 1):
            ngram = tuple(all_tokens[i : i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        if not ngram_counts:
            continue

        tf_values = []
        df_values = []
        total_tf = sum(ngram_counts.values())
        
        for ngram, tf in ngram_counts.items():
            ngram_str = str(ngram)
            df = doc_freqs.get(ngram_str, {}).get("count", 0)
            if df > 0:
                tf_values.append(tf)
                df_values.append(df)

        if tf_values and total_docs > 0:
            tf_array = np.array(tf_values)
            df_array = np.array(df_values)
            tf_norm = tf_array / total_tf
            df_norm = df_array / total_docs
            tfdf = np.dot(tf_norm, df_norm)
            tfdf_values.append(tfdf)

    return float(np.mean(tfdf_values) if tfdf_values else 0.0)

@fantastic
@corpus_prevalence
@both
def norm_log_dist(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Calculate normalized log distance between TF and DF distributions.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider
        
    Returns
    -------
    float
        Normalized log distance
    """
    tokenizer = FantasticTokenizer()
    segments = tokenizer.segment_melody(melody, phrase_gap=phrase_gap, units="quarters")

    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, segment.starts, segment.ends
        )
        all_tokens.extend(segment_tokens)

    doc_freqs = corpus_stats.get("document_frequencies", {})
    total_docs = len(doc_freqs)
    
    distances = []
    
    for n in range(1, max_ngram_order):
        ngram_counts = {}
        for i in range(len(all_tokens) - n + 1):
            ngram = tuple(all_tokens[i : i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        if not ngram_counts:
            continue

        tf_values = []
        df_values = []
        total_tf = sum(ngram_counts.values())
        
        for ngram, tf in ngram_counts.items():
            ngram_str = str(ngram)
            df = doc_freqs.get(ngram_str, {}).get("count", 0)
            if df > 0:
                tf_values.append(tf)
                df_values.append(df)

        if tf_values and total_docs > 0:
            tf_array = np.array(tf_values)
            df_array = np.array(df_values)
            tf_norm = tf_array / total_tf
            df_norm = df_array / total_docs
            distances.extend(np.abs(tf_norm - df_norm))

    return float(np.mean(distances) if distances else 0.0)

@fantastic
@corpus_prevalence
@both
def max_log_df(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Calculate maximum log document frequency across all n-grams.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider
        
    Returns
    -------
    float
        Maximum log document frequency
    """
    tokenizer = FantasticTokenizer()
    segments = tokenizer.segment_melody(melody, phrase_gap=phrase_gap, units="quarters")

    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, segment.starts, segment.ends
        )
        all_tokens.extend(segment_tokens)

    doc_freqs = corpus_stats.get("document_frequencies", {})
    
    max_df = 0
    
    for n in range(1, max_ngram_order):
        ngram_counts = {}
        for i in range(len(all_tokens) - n + 1):
            ngram = tuple(all_tokens[i : i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        if not ngram_counts:
            continue

        for ngram, tf in ngram_counts.items():
            ngram_str = str(ngram)
            df = doc_freqs.get(ngram_str, {}).get("count", 0)
            if df > 0:
                max_df = max(max_df, df)

    return float(np.log1p(max_df) if max_df > 0 else 0.0)

@fantastic
@corpus_prevalence
@both
def min_log_df(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Calculate minimum log document frequency across all n-grams.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider
        
    Returns
    -------
    float
        Minimum log document frequency
    """
    tokenizer = FantasticTokenizer()
    segments = tokenizer.segment_melody(melody, phrase_gap=phrase_gap, units="quarters")

    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, segment.starts, segment.ends
        )
        all_tokens.extend(segment_tokens)

    doc_freqs = corpus_stats.get("document_frequencies", {})
    
    min_df = float("inf")
    
    for n in range(1, max_ngram_order):
        ngram_counts = {}
        for i in range(len(all_tokens) - n + 1):
            ngram = tuple(all_tokens[i : i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        if not ngram_counts:
            continue

        for ngram, tf in ngram_counts.items():
            ngram_str = str(ngram)
            df = doc_freqs.get(ngram_str, {}).get("count", 0)
            if df > 0:
                min_df = min(min_df, df)

    return float(np.log1p(min_df) if min_df < float("inf") else 0.0)

@fantastic
@corpus_prevalence
@both
def mean_log_df(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Calculate mean log document frequency across all n-grams.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider
        
    Returns
    -------
    float
        Mean log document frequency
    """
    tokenizer = FantasticTokenizer()
    segments = tokenizer.segment_melody(melody, phrase_gap=phrase_gap, units="quarters")

    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, segment.starts, segment.ends
        )
        all_tokens.extend(segment_tokens)

    doc_freqs = corpus_stats.get("document_frequencies", {})
    
    total_log_df = 0.0
    df_count = 0
    
    for n in range(1, max_ngram_order):
        ngram_counts = {}
        for i in range(len(all_tokens) - n + 1):
            ngram = tuple(all_tokens[i : i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        if not ngram_counts:
            continue

        df_values = []
        for ngram, tf in ngram_counts.items():
            ngram_str = str(ngram)
            df = doc_freqs.get(ngram_str, {}).get("count", 0)
            if df > 0:
                df_values.append(df)

        if df_values:
            df_array = np.array(df_values)
            total_log_df += np.sum(np.log1p(df_array))
            df_count += len(df_array)

    return float(total_log_df / df_count if df_count > 0 else 0.0)

@fantastic
@corpus_prevalence
@both
def mean_global_local_weight(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Calculate mean global-local weight using inverse entropy weighting.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider
        
    Returns
    -------
    float
        Mean global-local weight
    """
    tokenizer = FantasticTokenizer()
    segments = tokenizer.segment_melody(melody, phrase_gap=phrase_gap, units="quarters")

    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, segment.starts, segment.ends
        )
        all_tokens.extend(segment_tokens)

    all_ngram_counts = {}
    for n in range(1, max_ngram_order):
        ngram_counts = {}
        for i in range(len(all_tokens) - n + 1):
            ngram = tuple(all_tokens[i : i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        for ngram, tf in ngram_counts.items():
            all_ngram_counts[ngram] = all_ngram_counts.get(ngram, 0) + tf

    if all_ngram_counts and len(corpus_stats.get("document_frequencies", {})) > 0:
        weights = InverseEntropyWeighting(all_ngram_counts, corpus_stats)
        combined_weights = weights.combined_weights
        return float(np.mean(combined_weights) if combined_weights else 0.0)
    else:
        return 0.0

@fantastic
@corpus_prevalence
@both
def std_global_local_weight(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Calculate standard deviation of global-local weight using inverse entropy weighting.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider
        
    Returns
    -------
    float
        Standard deviation of global-local weight
    """
    tokenizer = FantasticTokenizer()
    segments = tokenizer.segment_melody(melody, phrase_gap=phrase_gap, units="quarters")

    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, segment.starts, segment.ends
        )
        all_tokens.extend(segment_tokens)

    all_ngram_counts = {}
    for n in range(1, max_ngram_order):
        ngram_counts = {}
        for i in range(len(all_tokens) - n + 1):
            ngram = tuple(all_tokens[i : i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        for ngram, tf in ngram_counts.items():
            all_ngram_counts[ngram] = all_ngram_counts.get(ngram, 0) + tf

    if all_ngram_counts and len(corpus_stats.get("document_frequencies", {})) > 0:
        weights = InverseEntropyWeighting(all_ngram_counts, corpus_stats)
        combined_weights = weights.combined_weights
        return float(np.std(combined_weights, ddof=1) if len(combined_weights) > 1 else 0.0)
    else:
        return 0.0

@fantastic
@corpus_prevalence
@both
def mean_global_weight(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Calculate mean global weight using inverse entropy weighting.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider
        
    Returns
    -------
    float
        Mean global weight
    """
    tokenizer = FantasticTokenizer()
    segments = tokenizer.segment_melody(melody, phrase_gap=phrase_gap, units="quarters")

    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, segment.starts, segment.ends
        )
        all_tokens.extend(segment_tokens)

    all_ngram_counts = {}
    for n in range(1, max_ngram_order):
        ngram_counts = {}
        for i in range(len(all_tokens) - n + 1):
            ngram = tuple(all_tokens[i : i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        for ngram, tf in ngram_counts.items():
            all_ngram_counts[ngram] = all_ngram_counts.get(ngram, 0) + tf

    if all_ngram_counts and len(corpus_stats.get("document_frequencies", {})) > 0:
        weights = InverseEntropyWeighting(all_ngram_counts, corpus_stats)
        global_weights = weights.global_weights
        return float(np.mean(global_weights) if global_weights else 0.0)
    else:
        return 0.0

@fantastic
@corpus_prevalence
@both
def std_global_weight(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Calculate standard deviation of global weight using inverse entropy weighting.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider
        
    Returns
    -------
    float
        Standard deviation of global weight
    """
    tokenizer = FantasticTokenizer()
    segments = tokenizer.segment_melody(melody, phrase_gap=phrase_gap, units="quarters")

    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, segment.starts, segment.ends
        )
        all_tokens.extend(segment_tokens)

    all_ngram_counts = {}
    for n in range(1, max_ngram_order):
        ngram_counts = {}
        for i in range(len(all_tokens) - n + 1):
            ngram = tuple(all_tokens[i : i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        for ngram, tf in ngram_counts.items():
            all_ngram_counts[ngram] = all_ngram_counts.get(ngram, 0) + tf

    if all_ngram_counts and len(corpus_stats.get("document_frequencies", {})) > 0:
        weights = InverseEntropyWeighting(all_ngram_counts, corpus_stats)
        global_weights = weights.global_weights
        return float(np.std(global_weights, ddof=1) if len(global_weights) > 1 else 0.0)
    else:
        return 0.0

def get_corpus_features(
    melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int
) -> Dict:
    """Compute all corpus-based features for a melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics

    Returns
    -------
    Dict
        Dictionary of corpus-based feature values
    """
    tokenizer = FantasticTokenizer()
    segments = tokenizer.segment_melody(melody, phrase_gap=phrase_gap, units="quarters")

    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, segment.starts, segment.ends
        )
        all_tokens.extend(segment_tokens)

    doc_freqs = corpus_stats.get("document_frequencies", {})
    total_docs = len(doc_freqs)

    ngram_data = []
    for n in range(1, max_ngram_order):
        ngram_counts = {}
        for i in range(len(all_tokens) - n + 1):
            ngram = tuple(all_tokens[i : i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        if not ngram_counts:
            continue

        # Get document frequencies for all n-grams at once
        ngram_df_data = {
            "counts": ngram_counts,
            "total_tf": sum(ngram_counts.values()),
            "df_values": [],
            "tf_values": [],
            "ngrams": [],
        }

        # Batch lookup document frequencies
        for ngram, tf in ngram_counts.items():
            ngram_str = str(ngram)
            df = doc_freqs.get(ngram_str, {}).get("count", 0)
            if df > 0:
                ngram_df_data["df_values"].append(df)
                ngram_df_data["tf_values"].append(tf)
                ngram_df_data["ngrams"].append(ngram)

        if ngram_df_data["df_values"]:
            ngram_data.append(ngram_df_data)

    features = {}

    # Compute correlation features using pre-computed values
    if ngram_data:
        all_tf = []
        all_df = []
        for data in ngram_data:
            all_tf.extend(data["tf_values"])
            all_df.extend(data["df_values"])

        if len(all_tf) >= 2:
            try:
                # Check for no variance to avoid correlation problems
                tf_variance = np.var(all_tf)
                df_variance = np.var(all_df)
                
                if tf_variance == 0 or df_variance == 0:
                    # If either array is constant, correlation is undefined
                    features["tfdf_spearman"] = 0.0
                    features["tfdf_kendall"] = 0.0
                else:
                    spearman = scipy.stats.spearmanr(all_tf, all_df)[0]
                    kendall = scipy.stats.kendalltau(all_tf, all_df)[0]
                    features["tfdf_spearman"] = float(
                        spearman if not np.isnan(spearman) else 0.0
                    )
                    features["tfdf_kendall"] = float(
                        kendall if not np.isnan(kendall) else 0.0
                    )
            except:
                features["tfdf_spearman"] = 0.0
                features["tfdf_kendall"] = 0.0
        else:
            features["tfdf_spearman"] = 0.0
            features["tfdf_kendall"] = 0.0
    else:
        features["tfdf_spearman"] = 0.0
        features["tfdf_kendall"] = 0.0

    # Compute TFDF and distance features
    tfdf_values = []
    distances = []
    max_df = 0
    min_df = float("inf")
    total_log_df = 0.0
    df_count = 0

    for data in ngram_data:
        # TFDF calculation
        tf_array = np.array(data["tf_values"])
        df_array = np.array(data["df_values"])
        if len(tf_array) > 0:
            # Normalize vectors
            tf_norm = tf_array / data["total_tf"]
            df_norm = df_array / total_docs
            tfdf = np.dot(tf_norm, df_norm)
            tfdf_values.append(tfdf)

            # Distance calculation
            distances.extend(np.abs(tf_norm - df_norm))

            # Track max/min/total log DF
            max_df = max(max_df, max(data["df_values"]))
            min_df = min(min_df, min(x for x in data["df_values"] if x > 0))
            total_log_df += np.sum(np.log1p(df_array))
            df_count += len(df_array)

    features["mean_log_tfdf"] = float(np.mean(tfdf_values) if tfdf_values else 0.0)
    features["norm_log_dist"] = float(np.mean(distances) if distances else 0.0)
    features["max_log_df"] = float(np.log1p(max_df) if max_df > 0 else 0.0)
    features["min_log_df"] = float(np.log1p(min_df) if min_df < float("inf") else 0.0)
    features["mean_log_df"] = float(total_log_df / df_count if df_count > 0 else 0.0)

    # Entropy-based weighting features
    if ngram_data and total_docs > 0:
        all_ngram_counts = {}
        for data in ngram_data:
            for ngram, tf in zip(data["ngrams"], data["tf_values"]):
                all_ngram_counts[ngram] = all_ngram_counts.get(ngram, 0) + tf

        weights = InverseEntropyWeighting(all_ngram_counts, corpus_stats)
        all_combined_weights = weights.combined_weights
        all_global_weights = weights.global_weights
    else:
        all_combined_weights = []
        all_global_weights = []

    # Calculate statistics
    if all_combined_weights:
        features["mean_global_local_weight"] = float(np.mean(all_combined_weights))
        features["std_global_local_weight"] = float(np.std(all_combined_weights, ddof=1) if len(all_combined_weights) > 1 else 0.0)
    else:
        features["mean_global_local_weight"] = 0.0
        features["std_global_local_weight"] = 0.0

    if all_global_weights:
        features["mean_global_weight"] = float(np.mean(all_global_weights))
        features["std_global_weight"] = float(np.std(all_global_weights, ddof=1) if len(all_global_weights) > 1 else 0.0)
    else:
        features["mean_global_weight"] = 0.0
        features["std_global_weight"] = 0.0

    return features


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
    features = {}
    interval_functions = _get_features_by_domain_and_types("pitch", ["interval"])
    
    for name, func in interval_functions.items():
        try:
            # Get function signature to determine parameters
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Call function with appropriate parameters based on signature
            if 'pitches' in params and 'starts' in params and 'ends' in params and 'tempo' in params:
                result = func(melody.pitches, melody.starts, melody.ends, melody.tempo)
            elif 'pitches' in params and 'starts' in params and 'ends' in params:
                result = func(melody.pitches, melody.starts, melody.ends)
            elif 'pitches' in params and 'interval_level' in params:
                # Special case for variable_melodic_intervals
                result = func(melody.pitches, 7)  # Default interval level
            elif 'pitches' in params:
                result = func(melody.pitches)
            elif 'starts' in params and 'ends' in params and 'tau' in params:
                # edge case for mean_duration_accent
                result = func(melody.starts, melody.ends, 0.5, 2.0)
            elif 'starts' in params and 'ends' in params:
                result = func(melody.starts, melody.ends)
            elif 'starts' in params:
                result = func(melody.starts)
            elif 'melody' in params:
                result = func(melody)
            else:
                result = func(melody)
            
            # Handle functions that return tuples (like interval_direction)
            if isinstance(result, tuple) and len(result) == 2:
                features[f"{name}_mean"] = result[0]
                features[f"{name}_sd"] = result[1]
            else:
                features[name] = result
                
        except Exception as e:
            print(f"Warning: Could not compute {name}: {e}")
            features[name] = None
    
    return features


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
    contour_features["comb_contour_matrix"] = get_comb_contour_matrix(melody.pitches)
    contour_features["mean_melodic_accent"] = mean_melodic_accent(melody.pitches)
    contour_features["melodic_accent_std"] = melodic_accent_std(melody.pitches)
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
    metric_features = {}

    hierarchy_values = metric_hierarchy(
        melody.starts, melody.ends, 
        time_signature=melody.meter, tempo=melody.tempo, pitches=melody.pitches
    )
    metric_features["metric_hierarchy"] = hierarchy_values

    if hierarchy_values:
        melodic_accents = melodic_accent(melody.pitches)
        durational_accents = duration_accent(melody.starts, melody.ends)

        min_length = min(len(hierarchy_values), len(melodic_accents), len(durational_accents))
        if min_length > 0:
            accent_products = [
                h * m * d for h, m, d in zip(
                    hierarchy_values[:min_length],
                    melodic_accents[:min_length], 
                    durational_accents[:min_length]
                )
            ]
            metric_features["meter_accent"] = int(round(-1.0 * float(np.mean(accent_products))))
        else:
            metric_features["meter_accent"] = 0
    else:
        metric_features["meter_accent"] = 0
    
    return metric_features


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


def get_rhythm_features(melody: Melody) -> Dict:
    """Dynamically collect all rhythm features for a melody.
    
    Collects features decorated with @rhythm domain and @descriptives or @interval type.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of rhythm feature values
    """
    features = {}
    rhythm_functions = _get_features_by_domain_and_types("rhythm", ["descriptives", "interval"])
    
    # Pre-compute beat histogram data once for all beat histogram functions
    beat_histogram_data = None
    beat_histogram_functions = []
    regular_functions = []
    
    # Separate beat histogram functions from regular functions
    for name, func in rhythm_functions.items():
        if _is_beat_histogram_function(func):
            beat_histogram_functions.append((name, func))
        else:
            regular_functions.append((name, func))
    
    # Process regular functions first
    for name, func in regular_functions:
        try:
            # Get function signature to determine parameters
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Call function with appropriate parameters
            if 'melody' in params:
                result = func(melody)
            elif 'starts' in params and 'ends' in params and 'tempo' in params:
                result = func(melody.starts, melody.ends, melody.tempo)
            elif 'starts' in params and 'ends' in params and 'divisions_per_quarter' in params:
                result = func(melody.starts, melody.ends, 4, 8)  # Default values
            elif 'starts' in params and 'ends' in params and 'tau' in params:
                result = func(melody.starts, melody.ends, 0.5, 2.0)  # Default values
            elif 'starts' in params and 'ends' in params:
                result = func(melody.starts, melody.ends)
            elif 'starts' in params:
                result = func(melody.starts)
            else:
                # Try with melody object
                result = func(melody)
            
            # Handle functions that return tuples (like ioi_ratio, ioi_contour)
            if isinstance(result, tuple) and len(result) == 2:
                features[f"{name}_mean"] = result[0]
                features[f"{name}_std"] = result[1]
            else:
                features[name] = result
                
        except Exception as e:
            print(f"Warning: Could not compute {name}: {e}")
            features[name] = None
    
    # Process beat histogram functions with pre-computed data
    if beat_histogram_functions:
        beat_histogram_data = _precompute_beat_histogram_data(melody)
        normal_values, standardized_values, start_ticks, end_ticks, tempo, ppqn = beat_histogram_data
        
        for name, func in beat_histogram_functions:
            try:
                # Get function signature to determine parameters
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                
                # Call function with appropriate parameters
                if 'melody' in params:
                    result = func(melody)
                elif 'starts' in params and 'ends' in params and 'tempo' in params:
                    result = func(melody.starts, melody.ends, melody.tempo)
                elif 'starts' in params and 'ends' in params:
                    result = func(melody.starts, melody.ends)
                elif 'starts' in params:
                    result = func(melody.starts)
                else:
                    # Try with melody object
                    result = func(melody)
                
                # Handle functions that return tuples (like ioi_ratio, ioi_contour)
                if isinstance(result, tuple) and len(result) == 2:
                    features[f"{name}_mean"] = result[0]
                    features[f"{name}_std"] = result[1]
                else:
                    features[name] = result
                    
            except Exception as e:
                print(f"Warning: Could not compute {name}: {e}")
                features[name] = None
    
    # Add metric accent features
    features.update(get_metric_accent_features(melody))
    
    return features


def get_expectation_features(melody: Melody) -> Dict:
    """Dynamically collect all expectation features for a melody.
    
    Collects features decorated with FeatureType.EXPECTATION regardless of domain.
    """
    features: Dict[str, Any] = {}
    expectation_functions = _get_features_by_type(FeatureType.EXPECTATION)

    for name, func in expectation_functions.items():
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            if 'melody' in params:
                result = func(melody)
            elif 'pitches' in params and 'starts' in params and 'ends' in params and 'tempo' in params:
                result = func(melody.pitches, melody.starts, melody.ends, melody.tempo)
            elif 'pitches' in params and 'starts' in params and 'ends' in params:
                result = func(melody.pitches, melody.starts, melody.ends)
            elif 'pitches' in params:
                result = func(melody.pitches)
            elif 'starts' in params and 'ends' in params and 'tempo' in params:
                result = func(melody.starts, melody.ends, melody.tempo)
            elif 'starts' in params and 'ends' in params:
                result = func(melody.starts, melody.ends)
            elif 'starts' in params:
                result = func(melody.starts)
            else:
                result = func(melody)

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
            # Get function signature to determine parameters
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Call function with appropriate parameters
            if 'melody' in params:
                result = func(melody)
            elif 'starts' in params and 'ends' in params and 'tempo' in params:
                result = func(melody.starts, melody.ends, melody.tempo)
            elif 'starts' in params and 'ends' in params:
                result = func(melody.starts, melody.ends)
            elif 'starts' in params:
                result = func(melody.starts)
            else:
                # Try with melody object
                result = func(melody)
            
            features[name] = result
                
        except Exception as e:
            print(f"Warning: Could not compute {name}: {e}")
            features[name] = None
    
    return features


@jsymbolic
@rhythm
@metre
def meter_numerator(melody: Melody) -> int:
    """Time signature numerator for the melody.

    Returns
    -------
    int
        The numerator of the notated meter.
    """
    return melody.meter[0]


@jsymbolic
@rhythm
@metre
def meter_denominator(melody: Melody) -> int:
    """Time signature denominator for the melody.

    Returns
    -------
    int
        The denominator of the notated meter.
    """
    return melody.meter[1]


@novel
@rhythm
@metre
def proportion_of_time_in_first_meter(melody: Melody) -> float:
    """The proportion of time spent in the first time signature.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        The proportion of time spent in the first time signature.
    """
    return melody.proportion_of_time_in_first_meter

@jsymbolic
@rhythm
@metre
def number_of_unique_time_signatures(melody: Melody) -> int:
    """The number of unique time signatures in the melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    int
        The number of unique time signatures in the melody.

    Note
    -----
    This feature is named "Metrical Diversity" in jSymbolic.
    """
    return len(set(melody.time_signatures))

@novel
@rhythm
@metre
def syncopation(melody: Melody) -> float:
    """Calculate the mean syncopation value based on the Longuet-Higgins and Lee (1984) model.
    This syncopation model assigns metrical weights to each
    note position based on its position in the metric hierarchy. Syncopation occurs when
    a rest or tied note is preceded by a sounded note of lower metrical weight. The 
    syncopation value is the difference between the rest weight and the preceding note weight.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    float
        The mean syncopation value across all syncopation events (0.0 if no syncopation)
        
    Citation
    --------
    Longuet-Higgins & Lee (1984)
    """
    if not melody.starts or len(melody.starts) < 2:
        return 0.0

    hierarchy_values = metric_hierarchy(
        melody.starts, 
        melody.ends, 
        time_signature=melody.meter, 
        tempo=melody.tempo, 
        pitches=melody.pitches
    )

    if not hierarchy_values or len(hierarchy_values) != len(melody.starts):
        return 0.0

    # Hierarchy 5 (downbeat/measure start) -> weight 0 (strongest)
    # Hierarchy 4 (beat) -> weight -1
    # Hierarchy 3 (half-beat) -> weight -2
    # Hierarchy 2 (quarter-beat) -> weight -3
    # Hierarchy 1 (weakest offbeat) -> weight -4
    weights = [5 - h for h in hierarchy_values]

    syncopation_values = []

    for i in range(len(melody.starts) - 1):
        current_note_end = melody.ends[i]
        next_note_start = melody.starts[i + 1]

        gap_duration = next_note_start - current_note_end

        if gap_duration > 0.001:
            rest_weight = weights[i + 1]
            preceding_note_weight = weights[i]

            syncopation_value = rest_weight - preceding_note_weight

            if syncopation_value > 0:
                syncopation_values.append(syncopation_value)

    if not syncopation_values:
        return 0.0
    
    return float(np.mean(syncopation_values))

@simile
@rhythm
@metre
def syncopicity(melody: Melody) -> float:
    """Calculates the sum syncopicity of a melody across metric levels.
    Syncopicity measures the degree to which notes occur off the main metrical grid
    but are long enough to span across metric boundaries. This calculates syncopations at 
    four metric levels:
    1) Half bar level
    2) Beat level  
    3) First subdivision (half-beat)
    4) Second subdivision (quarter-beat)
    
    An event is considered syncopated at a given level if:
    1) It does not fall on a grid point of this level
    2) It falls on a grid point of the next lower level
    3) Its IOI extends beyond the lower level time unit (or it's the last note)
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    float
        The sum of syncopicity values across all metric levels
    """
    if not melody.starts or len(melody.starts) < 2:
        return 0.0

    numerator, denominator = melody.meter
    tempo = melody.tempo

    quarter_note_duration = 60.0 / tempo

    beat_duration = (4.0 / denominator) * quarter_note_duration

    measure_duration = numerator * beat_duration

    levels = [
        measure_duration / 2.0,  # Half bar
        beat_duration,           # Beat
        beat_duration / 2.0,     # First subdivision
        beat_duration / 4.0      # Second subdivision
    ]

    n_notes = len(melody.starts)
    total_syncopicity = 0.0

    iois = []
    for i in range(n_notes - 1):
        iois.append(melody.starts[i + 1] - melody.starts[i])
    iois.append(0)

    for level_idx in range(len(levels) - 1):
        level_duration = levels[level_idx]
        next_lower_duration = levels[level_idx + 1]

        syncopation_count = 0
        tolerance = 0.01

        for note_idx, start_time in enumerate(melody.starts):
            position_in_level = start_time % level_duration
            on_current_grid = position_in_level < tolerance or position_in_level > (level_duration - tolerance)

            if on_current_grid:
                continue

            position_in_lower = start_time % next_lower_duration
            on_lower_grid = position_in_lower < tolerance or position_in_lower > (next_lower_duration - tolerance)

            if not on_lower_grid:
                continue

            is_last_note = note_idx == n_notes - 1
            ioi_extends = iois[note_idx] > (next_lower_duration + tolerance)
            
            if is_last_note or ioi_extends:
                syncopation_count += 1

        level_syncopicity = syncopation_count / n_notes if n_notes > 0 else 0.0
        total_syncopicity += level_syncopicity

    return float(total_syncopicity)


@idyom
@pitch
@tonality
def inscale(melody: Melody) -> list[int]:
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
    pitches = melody.pitches
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)
    
    if correlations:
        key_name = correlations[0][0].split()[0]
        key_distances = _get_key_distances()
        root = key_distances[key_name]
        
        # Determine scale type and pattern
        is_major = "major" in correlations[0][0]
        scale = [0, 2, 4, 5, 7, 9, 11] if is_major else [0, 2, 3, 5, 7, 8, 10]
        scale = [(note + root) % 12 for note in scale]
        
        # For each pitch, indicate if it's in the estimated key's scale (1) or not (0)
        return [1 if pc in scale else 0 for pc in pitch_classes]
    else:
        return []

@novel
@tonality
@pitch
def proportion_inscale(melody: Melody) -> float:
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
    inscale_vals = inscale(melody)
    if not inscale_vals:
        return -1.0
    return sum(inscale_vals) / len(inscale_vals)

@novel
@tonality
@pitch
def longest_monotonic_conjunct_scalar_passage(melody: Melody) -> int:
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
    return _longest_monotonic_conjunct_scalar_passage(pitches, correlations)

@novel
@tonality
@pitch
def longest_conjunct_scalar_passage(melody: Melody) -> int:
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
    return _longest_conjunct_scalar_passage(pitches, correlations)

@novel
@tonality
@pitch
def proportion_conjunct_scalar(melody: Melody) -> float:
    """The proportion of notes that form conjunct scalar sequences.
    
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
    return _proportion_conjunct_scalar(pitches, correlations)

@novel
@tonality
@pitch
def proportion_scalar(melody: Melody) -> float:
    """The proportion of all notes that form scalar sequences.

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
    return _proportion_scalar(pitches, correlations)

@fantastic
@tonality
@pitch
def mode(
    melody: Melody,
    key_estimation: str = "infer_if_necessary",
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
    # Infer mode using specified algorithm
    _, inferred_mode = infer_key_from_pitches(melody.pitches, algorithm=key_finding_algorithm)
    
    # Determine which mode to use based on strategy
    if key_estimation == "always_infer":
        # Always use inferred mode
        return inferred_mode if inferred_mode else "unknown"
    else:
        # Try to read from MIDI file
        mode_from_melody = None
        if melody.has_key_signature:
            key_sig = melody.key_signature
            if key_sig and len(key_sig) >= 2:
                mode_from_melody = key_sig[1]
        
        if key_estimation == "always_read_from_file":
            if mode_from_melody is None:
                raise ValueError(f"No key signature found in MIDI file: {melody.id}")
            return mode_from_melody
        else:
            # key_estimation == "infer_if_necessary"
            if mode_from_melody is not None:
                # Use mode from MIDI
                return mode_from_melody
            else:
                # Infer if no MIDI mode available
                return inferred_mode if inferred_mode else "unknown"

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
    
    # Pre-compute absolute correlation values
    abs_correlations = [(key, abs(val)) for key, val in correlations]
    abs_corr_values = [val for _, val in abs_correlations]

    # Basic tonality features using cached correlations
    tonality_features["tonalness"] = abs_corr_values[0]

    if len(correlations) >= 2:
        tonality_features["tonal_clarity"] = (
            abs_corr_values[0] / abs_corr_values[1] if abs_corr_values[1] != 0 else 1.0
        )
        other_sum = sum(abs_corr_values[1:])
        tonality_features["tonal_spike"] = (
            abs_corr_values[0] / other_sum if other_sum != 0 else 1.0
        )
    else:
        tonality_features["tonal_clarity"] = -1.0
        tonality_features["tonal_spike"] = -1.0

    # Entropy using cached values
    tonality_features["tonal_entropy"] = (
        shannon_entropy(abs_corr_values) if correlations else -1.0
    )
    
    # Histogram using cached correlations
    tonality_features["tonalness_histogram"] = histogram_bins(correlations[0][1], 24)

    # Determine the key to use for key-dependent features based on strategy
    key_for_features = None
    
    if key_estimation == "always_infer":
        # Use the inferred key using specified algorithm
        key_name, mode = infer_key_from_pitches(melody.pitches, algorithm=key_finding_algorithm)
        if key_name and mode:
            key_for_features = f"{key_name} {mode}"
    else:
        # Try to read from MIDI file using the already-extracted key signature info
        key_from_melody = None
        if melody.has_key_signature:
            key_sig = melody.key_signature
            if key_sig:
                key_from_melody = f"{key_sig[0]} {key_sig[1]}"
        
        if key_estimation == "always_read_from_file":
            if key_from_melody is None:
                raise ValueError(f"No key signature found in MIDI file: {melody.id}")
            key_for_features = key_from_melody
        else:
            # key_estimation == "infer_if_necessary"
            if key_from_melody is not None:
                # Use key from MIDI
                key_for_features = key_from_melody
            else:
                # Infer using specified algorithm
                key_name, mode = infer_key_from_pitches(melody.pitches, algorithm=key_finding_algorithm)
                if key_name and mode:
                    key_for_features = f"{key_name} {mode}"

    if key_for_features:
        key_name = key_for_features.split()[0]
        # Remove trailing 'm' if present (e.g., "G#m" -> "G#")
        if key_name.endswith('m'):
            key_name = key_name[:-1]
        key_distances = _get_key_distances()
        root = key_distances[key_name]
        tonality_features["referent"] = root

        # Determine scale type and pattern
        is_major = "major" in key_for_features
        scale = [0, 2, 4, 5, 7, 9, 11] if is_major else [0, 2, 3, 5, 7, 8, 10]
        scale = [(note + root) % 12 for note in scale]

        # For each pitch, indicate if it's in the estimated key's scale (1) or not (0)
        tonality_features["inscale"] = [1 if pc in scale else 0 for pc in pcs]
        tonality_features["key"] = key_for_features
        tonality_features["mode"] = "major" if is_major else "minor"
    else:
        tonality_features["referent"] = -1
        tonality_features["inscale"] = []
        tonality_features["key"] = "unknown"
        tonality_features["mode"] = "unknown"


    # Scalar passage features
    from .algorithms import longest_monotonic_conjunct_scalar_passage as _longest_monotonic_conjunct_scalar_passage
    from .algorithms import longest_conjunct_scalar_passage as _longest_conjunct_scalar_passage
    tonality_features["longest_monotonic_conjunct_scalar_passage"] = (
        _longest_monotonic_conjunct_scalar_passage(melody.pitches, correlations)
    )
    tonality_features["longest_conjunct_scalar_passage"] = (
        _longest_conjunct_scalar_passage(melody.pitches, correlations)
    )
    from .algorithms import proportion_conjunct_scalar as _proportion_conjunct_scalar
    from .algorithms import proportion_scalar as _proportion_scalar
    tonality_features["proportion_conjunct_scalar"] = _proportion_conjunct_scalar(
        melody.pitches, correlations
    )
    tonality_features["proportion_scalar"] = _proportion_scalar(melody.pitches, correlations)
    tonality_features["proportion_inscale"] = proportion_inscale(melody)
    
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

    # Time each feature category
    timings = {}

    start = time.time()
    pitch_features = get_pitch_features(mel)
    timings["pitch"] = time.time() - start

    start = time.time()
    interval_features = get_interval_features(mel)
    timings["interval"] = time.time() - start

    start = time.time()
    contour_features = get_contour_features(mel)
    timings["contour"] = time.time() - start

    start = time.time()
    rhythm_features = get_rhythm_features(mel)
    timings["rhythm"] = time.time() - start

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
    complexity_features = get_complexity_features(mel, phrase_gap=phrase_gap, max_ngram_order=max_ngram_order)
    timings["complexity"] = time.time() - start

    melody_features = {
        "pitch_features": pitch_features,
        "interval_features": interval_features,
        "contour_features": contour_features,
        "rhythm_features": rhythm_features,
        "tonality_features": tonality_features,
        "metre_features": metre_features,
        "expectation_features": expectation_features,
        "complexity_features": complexity_features,
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
                    idyom_features.update(idyom_results[melody_id_str])

        if idyom_features:
            melody_features["idyom_features"] = idyom_features
        else:
            # Add obvious flag if IDyOM values if not found
            melody_features["idyom_features"] = {"mean_information_content": -1}

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
                next(f)  # Skip header

                line_count = 0
                for line in f:
                    line_count += 1
                    parts = line.strip().split()

                    if len(parts) < 3:
                        logger.warning(f"Skipping malformed line: {line.strip()}")
                        continue  # Skip malformed lines

                    try:
                        # IDyOM's melody ID is a 1-based index.
                        melody_idx = int(parts[0]) - 1
                        mean_ic = float(parts[2])

                        if 0 <= melody_idx < len(midi_files):
                            # Map the index to the melody number (1-based index)
                            melody_id = str(melody_idx + 1)
                            idyom_results[melody_id] = {
                                "mean_information_content": mean_ic
                            }
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
            corpus=resources.files("melody_features") / "corpora/Essen_Corpus",
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
                    corpus=None,
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
                    corpus=None,
                ),
            },
            fantastic=FantasticConfig(max_ngram_order=6, phrase_gap=1.5, corpus=None),
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


def _setup_corpus_statistics(config: Config, output_file: str) -> Optional[dict]:
    """Set up corpus statistics for FANTASTIC features.

    Parameters
    ----------
    config : Config
        Configuration object containing corpus information
    output_file : str
        Path to output file for determining corpus stats location

    Returns
    -------
    Optional[dict]
        Corpus statistics dictionary or None if no corpus provided

    Raises
    ------
    FileNotFoundError
        If corpus path is not a valid directory
    """
    logger = logging.getLogger("melody_features")

    # Determine which corpus to use for FANTASTIC
    fantastic_corpus = (
        config.fantastic.corpus
        if config.fantastic.corpus is not None
        else config.corpus
    )

    if not fantastic_corpus:
        logger.info(
            "No corpus path provided, corpus-dependent features will not be computed."
        )
        return None

    if not Path(fantastic_corpus).is_dir():
        raise FileNotFoundError(
            f"Corpus path is not a valid directory: {fantastic_corpus}"
        )

    logger.info(f"Generating corpus statistics from: {fantastic_corpus}")

    # Define a persistent path for the corpus stats file.
    corpus_name = Path(fantastic_corpus).name
    corpus_stats_path = Path(output_file).parent / f"{corpus_name}_corpus_stats.json"
    logger.info(f"Corpus statistics file will be at: {corpus_stats_path}")

    # Ensure the directory exists
    corpus_stats_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate and load corpus stats.
    if not corpus_stats_path.exists():
        logger.info("Corpus statistics file not found. Generating a new one...")
        make_corpus_stats(fantastic_corpus, str(corpus_stats_path))
        logger.info("Corpus statistics generated.")
    else:
        logger.info("Existing corpus statistics file found.")

    corpus_stats = load_corpus_stats(str(corpus_stats_path))
    logger.info("Corpus statistics loaded successfully.")

    return corpus_stats


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
        idyom_corpus = (
            idyom_config.corpus if idyom_config.corpus is not None else config.corpus
        )
        if idyom_config.models == ":stm" and idyom_config.corpus is None:
            idyom_corpus = None
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
        "interval_features": get_interval_features(mel),
        "contour_features": get_contour_features(mel),
        "rhythm_features": get_rhythm_features(mel),
        "tonality_features": get_tonality_features(mel, key_estimation=config.key_estimation),
        "metre_features": get_metre_features(mel),
        "expectation_features": get_expectation_features(mel),
        "complexity_features": get_complexity_features(
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

    # Add IDyOM features for each config to the header if they were generated
    for idyom_name, idyom_results in idyom_results_dict.items():
        if idyom_results:
            sample_id = next(iter(idyom_results))
            for feature in idyom_results[sample_id].keys():
                first_features[f"idyom_{idyom_name}_features.{feature}"] = None

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

    # Track timing statistics
    timing_stats = {
        "pitch": [],
        "interval": [],
        "contour": [],
        "rhythm": [],
        "tonality": [],
        "metre": [],
        "expectation": [],
        "complexity": [],
        "corpus": [],
        "total": [],
    }

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
    max_ngram_order: int = 6
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
        Maximum n-gram order (default: 6)
        
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
    max_ngram_order: int = 6
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
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            args = []
            for param in params:
                if param == "melody":
                    args.append(melody)
                elif param == "pitches":
                    args.append(melody.pitches)
                elif param == "starts":
                    args.append(melody.starts)
                elif param == "ends":
                    args.append(melody.ends)
                elif param == "tempo":
                    args.append(melody.tempo)
                elif param == "ppqn":
                    args.append(480)
                elif param == "corpus_stats":
                    args.append(corpus_stats)
                elif param == "phrase_gap":
                    args.append(phrase_gap)
                elif param == "max_ngram_order":
                    args.append(max_ngram_order)
                else:
                    if param in sig.parameters and sig.parameters[param].default != inspect.Parameter.empty:
                        args.append(sig.parameters[param].default)
                    else:
                        raise ValueError(f"Unknown parameter: {param}")
            
            result = func(*args)
            
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

    # Use a temporary output file path for corpus statistics
    temp_output_file = "temp_corpus_stats.csv"
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
    
    # Log timing statistics
    end_time = time.time()
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    logger.info("Timing Statistics (average milliseconds per melody):")
    for category, times in timing_stats.items():
        if times:  # Only print if we have timing data
            avg_time = sum(times) / len(times) * 1000  # Convert to milliseconds
            logger.info(f"{category:15s}: {avg_time:8.2f}ms")
    
    logger.info(f"Successfully extracted features for {len(df)} melodies")
    
    return df
