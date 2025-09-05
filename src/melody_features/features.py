"""
This module contains functions to compute features from melodies.
Features are the product of an input list and at least one algorithm.
"""

__author__ = "David Whyatt"

import warnings

# Suppress warnings from external libraries before any other imports
from importlib import resources

from .feature_decorators import (
    fantastic, idyom, midi_toolbox, melsim, jsymbolic, novel, simile,
    FeatureType, feature_type,
    pitch_feature, interval_feature, contour_feature,
    tonality_feature, duration_feature, complexity_feature, corpus_feature,
    mtype_feature
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
from typing import Dict, List, Optional, Tuple, Union

import mido
import numpy as np
import pandas as pd
import scipy
from natsort import natsorted
from tqdm import tqdm

from melody_features.algorithms import (
    arpeggiation_proportion,
    chromatic_motion_proportion,
    circle_of_fifths,
    compute_tonality_vector,
    get_duration_ratios,
    longest_conjunct_scalar_passage,
    longest_monotonic_conjunct_scalar_passage,
    melodic_embellishment_proportion,
    nine_percent_significant_values,
    proportion_conjunct_scalar,
    proportion_scalar,
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
from melody_features.stats import (
    mode,
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
            if len(viewpoint) != 2:
                raise ValueError(
                    f"Linked viewpoints must be pairs, got {len(viewpoint)} elements: {viewpoint}"
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
        Order of the PPM model.
    models : str
        Models to use for IDyOM analysis. Can be ":stm", ":ltm" or ":both".
    corpus : Optional[os.PathLike]
        Path to the corpus to use for IDyOM analysis. If not provided, the corpus will be the one specified in the Config class.
        This will override the corpus specified in the Config class if both are provided.
    """

    target_viewpoints: list[str]
    source_viewpoints: list[str]
    ppm_order: int
    models: str
    corpus: Optional[os.PathLike] = None

    def __post_init__(self):
        """Validate the configuration after initialization."""
        # Validate viewpoints
        _validate_viewpoints(self.target_viewpoints, "target_viewpoints")
        _validate_viewpoints(self.source_viewpoints, "source_viewpoints")

        # Validate ppm_order
        if not isinstance(self.ppm_order, int):
            raise ValueError(
                f"ppm_order must be an integer, got {type(self.ppm_order)}"
            )
        if self.ppm_order < 0:
            raise ValueError(f"ppm_order must be non-negative, got {self.ppm_order}")

        # Validate models
        valid_models = {":stm", ":ltm", ":both"}
        if not isinstance(self.models, str):
            raise ValueError(f"models must be a string, got {type(self.models)}")
        if self.models not in valid_models:
            raise ValueError(f"models must be one of {valid_models}, got {self.models}")

        # Validate corpus path if provided
        if self.corpus is not None:
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
        # Validate max_ngram_order
        if not isinstance(self.max_ngram_order, int):
            raise ValueError(
                f"max_ngram_order must be an integer, got {type(self.max_ngram_order)}"
            )
        if self.max_ngram_order < 1:
            raise ValueError(
                f"max_ngram_order must be at least 1, got {self.max_ngram_order}"
            )

        # Validate phrase_gap
        if not isinstance(self.phrase_gap, (int, float)):
            raise ValueError(
                f"phrase_gap must be a number, got {type(self.phrase_gap)}"
            )
        if self.phrase_gap <= 0:
            raise ValueError(f"phrase_gap must be positive, got {self.phrase_gap}")

        # Validate corpus path if provided
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
    """

    idyom: dict[str, IDyOMConfig]
    fantastic: FantasticConfig
    corpus: Optional[os.PathLike] = None

    def __post_init__(self):
        """Validate the configuration after initialization."""
        # Validate corpus path if provided
        if self.corpus is not None:
            if not isinstance(self.corpus, (str, os.PathLike)):
                raise ValueError(
                    f"corpus must be a string or PathLike, got {type(self.corpus)}"
                )
            if not Path(self.corpus).exists():
                raise ValueError(f"corpus path does not exist: {self.corpus}")

        # Validate idyom dictionary
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

        # Validate fantastic config
        if not isinstance(self.fantastic, FantasticConfig):
            raise ValueError(
                f"fantastic must be a FantasticConfig object, got {type(self.fantastic)}"
            )


# Pitch Features
@fantastic
@jsymbolic
@pitch_feature
def pitch_range(pitches: list[int]) -> int:
    """Calculate the range between the highest and lowest pitches.

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
@pitch_feature
def pitch_standard_deviation(pitches: list[int]) -> float:
    """Calculate the standard deviation of pitch values.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of pitches
    """
    return float(standard_deviation(pitches))

# Alias
pitch_variability = pitch_standard_deviation

@jsymbolic
@pitch_feature
def pitch_class_variability(pitches: list[int]) -> float:
    """Calculate the standard deviation of pitch class values.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of pitch class values
    """
    return float(standard_deviation([pitch % 12 for pitch in pitches]))

@fantastic
@pitch_feature
def pitch_entropy(pitches: list[int]) -> float:
    """Calculate the Shannon entropy of pitch values.

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
@pitch_feature
def pcdist1(pitches: list[int], starts: list[float], ends: list[float]) -> dict:
    """Calculate duration-weighted distribution of pitch classes.

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
        # Convert pitch to pitch class (0-11)
        pitch_class = pitch % 12
        # Convert duration to integer number of repetitions (e.g. duration 2.5 -> 25 repetitions)
        repetitions = max(1, int(duration * 10))  # Ensure at least 1 repetition
        weighted_pitch_classes.extend([pitch_class] * repetitions)

    if not weighted_pitch_classes:
        return 0.0

    return distribution_proportions(weighted_pitch_classes)

@jsymbolic
@pitch_feature
def first_pitch(pitches: list[int]) -> int:
    """Find the first pitch in the melody.
    
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
@pitch_feature
def first_pitch_class(pitches: list[int]) -> int:
    """Find the first pitch class in the melody.
    
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
@pitch_feature
def last_pitch(pitches: list[int]) -> int:
    """Find the last pitch in the melody.
    
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
@pitch_feature
def last_pitch_class(pitches: list[int]) -> int:
    """Find the last pitch class in the melody.
    
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
@pitch_feature
def basic_pitch_histogram(pitches: list[int]) -> dict:
    """Create histogram of pitch values within range of input pitches.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    dict
        Dictionary mapping pitch values to counts
    """
    if not pitches:
        return {}

    # Use number of unique pitches as number of bins, with minimum of 1
    num_midi_notes = max(1, len(set(pitches)))
    return histogram_bins(pitches, num_midi_notes)

@jsymbolic
@pitch_feature
def melodic_pitch_variety(pitches: list[int], starts: list[float]) -> float:
    """Calculate average number of notes before a pitch is repeated.
    
    This matches jSymbolic's implementation which counts ticks (not individual notes)
    and treats simultaneous notes as one note for counting purposes. We should never
    have any simultaneous notes in our melodies - the monophonic check should catch this.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Average number of notes before pitch repetition
    """
    if not pitches or len(pitches) < 2:
        return 0.0

    tick_pitch_map = {}
    for i, (start, pitch) in enumerate(zip(starts, pitches)):
        tick = int(round(start * 10000))
        if tick not in tick_pitch_map:
            tick_pitch_map[tick] = []
        tick_pitch_map[tick].append((i, pitch))

    sorted_ticks = sorted(tick_pitch_map.keys())

    number_of_repeated_notes_found = 0.0
    summed_notes_before_repetition = 0.0
    max_notes_that_can_go_by = 16

    for tick_idx, tick in enumerate(sorted_ticks):
        notes_at_tick = tick_pitch_map[tick]

        for note_idx, pitch in notes_at_tick:
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

                for future_note_idx, future_pitch in future_notes:
                    if future_pitch == pitch and not found_repeated_pitch and notes_gone_by_with_different_pitch <= max_notes_that_can_go_by:
                        found_repeated_pitch = True
                        number_of_repeated_notes_found += 1
                        summed_notes_before_repetition += notes_gone_by_with_different_pitch
                        break

    if number_of_repeated_notes_found == 0:
        return 0.0
    
    return float(summed_notes_before_repetition / number_of_repeated_notes_found)


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
    
    # Circle of fifths order: C, G, D, A, E, B, F#, C#, G#, D#, A#, F
    circle_of_fifths_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    
    longest_sequence = [pitch_classes[0]]  # Start with first pitch class
    current_sequence = [pitch_classes[0]]
    
    for i in range(1, len(pitch_classes)):
        pc = pitch_classes[i]
        last_pc = current_sequence[-1]
        
        # Check if current PC is a fifth away from the last PC
        if (circle_of_fifths_order.index(pc) - circle_of_fifths_order.index(last_pc)) % 12 == 1:
            current_sequence.append(pc)
        else:
            # Sequence broken, check if it's the longest so far
            if len(current_sequence) > len(longest_sequence):
                longest_sequence = current_sequence[:]
            current_sequence = [pc]  # Start new sequence
    
    # Check final sequence
    if len(current_sequence) > len(longest_sequence):
        longest_sequence = current_sequence[:]
    
    return longest_sequence

@jsymbolic
@pitch_feature
def dominant_spread(pitches: list[int]) -> int:
    """Find longest sequence of pitch classes separated by perfect 5ths that each appear >9% of the time.

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

    # If we have significant pitch classes, repeat the sequence to catch wrap-around
    if test_sequence:
        test_sequence = test_sequence * 2

    longest_sequence = _consecutive_fifths(test_sequence)

    return len(longest_sequence)

@jsymbolic
@pitch_feature
def mean_pitch(pitches: list[int]) -> int:
    """Calculate mean pitch value.

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
@pitch_feature
def mean_pitch_class(pitches: list[int]) -> float:
    """Calculate mean pitch class value.
    
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
def most_common_pitch(pitches: list[int]) -> int:
    """Find most frequently occurring pitch.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Most common pitch value
    """
    return int(mode(pitches))

@jsymbolic
@pitch_feature
def most_common_pitch_class(pitches: list[int]) -> int:
    """Find most frequently occurring pitch class.
    
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
    return int(mode([pitch % 12 for pitch in pitches]))

@jsymbolic
@pitch_feature
def number_of_unique_pitch_classes(pitches: list[int]) -> int:
    """Count number of unique pitch classes.
    
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
@pitch_feature
def number_of_common_pitches_classes(pitches: list[int]) -> int:
    """Count pitch classes that appear in at least 20% of notes.

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
    significant_pcs = nine_percent_significant_values(pcs, threshold=0.2)
    return int(len(significant_pcs))

@jsymbolic
@pitch_feature
def number_of_unique_pitches(pitches: list[int]) -> int:
    """Count number of unique pitches.

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
@pitch_feature
def number_of_common_pitches(pitches: list[int]) -> int:
    """Count unique pitches that appear in at least 9% of notes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of unique pitches that appear in at least 9% of notes
    """

    significant_pitches = nine_percent_significant_values(pitches)
    return int(len(set(significant_pitches)))

@midi_toolbox
@pitch_feature
def tessitura(pitches: list[int]) -> list[float]:
    """Calculate melodic tessitura for each note based on von Hippel (2000).
    Implementation based on MIDI toolbox "tessitura.m"
    
    Tessitura is based on deviation from median pitch height. The median range 
    of the melody tends to be favoured and thus more expected. Tessitura predicts 
    whether listeners expect tones close to median pitch height.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    list[float]
        Absolute tessitura value for each note in the sequence
    """
    if len(pitches) < 2:
        return [0.0] if len(pitches) == 1 else []
    
    tessitura_values = [0.0]
    
    for i in range(2, len(pitches) + 1):
        # Calculate median of previous pitches (notes 1 to i-1)
        median_prev = np.median(pitches[:i-1])
        
        # Calculate standard deviation of previous pitches
        if i == 2:
            # For second note, std of single value is 0, so tessitura is 0
            tessitura_values.append(0.0)
            continue
            
        std_prev = np.std(pitches[:i-1], ddof=1)
        
        # Avoid division by zero
        if std_prev == 0:
            tessitura_values.append(0.0)
        else:
            # Calculate tessitura: (current_pitch - median) / std_deviation
            current_pitch = pitches[i-1]  # Current note
            tessitura_val = (current_pitch - median_prev) / std_prev
            tessitura_values.append(abs(tessitura_val))
    
    return tessitura_values

@novel
@pitch_feature
def mean_tessitura(pitches: list[int]) -> float:
    """Calculate mean tessitura across all notes.
    
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
@pitch_feature
def tessitura_std(pitches: list[int]) -> float:
    """Calculate standard deviation of tessitura values.
    
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
@pitch_feature
def prevalence_of_most_common_pitch(pitches: list[int]) -> float:
    """Calculate proportion of most common pitch with regards to the
    number of notes in the whole melody.

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
@pitch_feature
def prevalence_of_most_common_pitch_class(pitches: list[int]) -> float:
    """Calculate proportion of most common pitch class with regards to the
    number of notes in the whole melody.

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
@pitch_feature
def relative_prevalence_of_top_pitches(pitches: list[int]) -> float:
    """Calculate ratio of the frequency of the second most common pitch to the frequency of the most common pitch.

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
@pitch_feature
def relative_prevalence_of_top_pitch_classes(pitches: list[int]) -> float:
    """Calculate ratio of the frequency of the second most common pitch class to the frequency of the most common pitch class.
    
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
@pitch_feature
def interval_between_most_prevalent_pitches(pitches: list[int]) -> int:
    """Calculate the number of semitones between the most prevalent pitches.

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

    pitch_counts = {}
    for pitch in pitches:
        pitch_counts[pitch] = pitch_counts.get(pitch, 0) + 1

    if len(pitch_counts) < 2:
        return 0

    sorted_pitches = sorted(pitch_counts.items(), key=lambda x: x[1], reverse=True)
    most_common_pitch = sorted_pitches[0][0]
    second_most_common_pitch = sorted_pitches[1][0]

    return int(abs(most_common_pitch - second_most_common_pitch))

@jsymbolic
@pitch_feature
def interval_between_most_prevalent_pitch_classes(pitches: list[int]) -> int:
    """Calculate the number of semitones between the most prevalent pitch classes.

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

    pcs = [pitch % 12 for pitch in pitches]
    if len(pcs) < 2:
        return 0

    pc_counts = {}
    for pc in pcs:
        pc_counts[pc] = pc_counts.get(pc, 0) + 1

    if len(pc_counts) < 2:
        return 0

    sorted_pcs = sorted(pc_counts.items(), key=lambda x: x[1], reverse=True)
    most_common_pc = sorted_pcs[0][0]
    second_most_common_pc = sorted_pcs[1][0]

    return int(abs(most_common_pc - second_most_common_pc))

@jsymbolic
@pitch_feature
def folded_fifths_pitch_class_histogram(pitches: list[int]) -> dict:
    """Create histogram of pitch classes arranged in circle of fifths.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    dict
        Dictionary mapping pitch classes to counts, arranged by circle of fifths
    """
    # Get pitch classes and count occurrences
    pcs = [pitch % 12 for pitch in pitches]
    # Count occurrences of each pitch class
    unique = []
    counts = []
    for pc in set(pcs):
        unique.append(pc)
        counts.append(pcs.count(pc))
    return circle_of_fifths(unique, counts)

@jsymbolic
@pitch_feature
def pitch_class_kurtosis_after_folding(pitches: list[int]) -> float:
    """Calculate kurtosis of folded fifths pitch class histogram.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Kurtosis of folded fifths histogram values, or 0 for empty input
    """
    pitches = [pitch % 12 for pitch in pitches]
    histogram = folded_fifths_pitch_class_histogram(pitches)
    if not histogram:
        return 0.0
    return float(kurtosis(list(histogram.keys())))

@jsymbolic
@pitch_feature
def pitch_class_skewness_after_folding(pitches: list[int]) -> float:
    """Calculate skewness of folded fifths pitch class histogram.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Skewness of folded fifths histogram values, or 0 for empty input
    """
    pitches = [pitch % 12 for pitch in pitches]
    histogram = folded_fifths_pitch_class_histogram(pitches)
    if not histogram:
        return 0.0
    return float(skew(list(histogram.keys())))

@jsymbolic
@pitch_feature
def pitch_class_variability_after_folding(pitches: list[int]) -> float:
    """Calculate standard deviation of folded fifths pitch class histogram.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of folded fifths histogram values, or 0 for empty input
    """
    pitches = [pitch % 12 for pitch in pitches]
    histogram = folded_fifths_pitch_class_histogram(pitches)
    if not histogram:
        return 0.0
    return float(standard_deviation(list(histogram.keys())))

@jsymbolic
@pitch_feature
def importance_of_bass_register(pitches: list[int]) -> float:
    """The proportion of MIDI pitch numbers that are between 0 and 54. 
    
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
@pitch_feature
def importance_of_middle_register(pitches: list[int]) -> float:
    """The proportion of MIDI pitch numbers that are between 55 and 72. 

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
@pitch_feature
def importance_of_high_register(pitches: list[int]) -> float:
    """The proportion of MIDI pitch numbers that are between 73 and 127. 
    
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

# Interval Features

@simile
@interval_feature
def pitch_interval(pitches: list[int]) -> list[int]:
    """Calculate intervals between consecutive pitches.

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
@interval_feature
def absolute_interval_range(pitches: list[int]) -> int:
    """Calculate range between largest and smallest absolute interval size.

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
@interval_feature
def mean_absolute_interval(pitches: list[int]) -> float:
    """Calculate mean absolute interval size.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Mean absolute interval size in semitones
    """
    return float(np.mean([abs(x) for x in pitch_interval(pitches)]))


# Alias for mean_absolute_interval / FANTASTIC vs jSymbolic
mean_melodic_interval = mean_absolute_interval

@fantastic
@interval_feature
def standard_deviation_absolute_interval(pitches: list[int]) -> float:
    """Calculate standard deviation of absolute interval sizes.

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
@interval_feature
def modal_interval(pitches: list[int]) -> int:
    """Find most common interval size.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Most frequent interval size in semitones
    """

    intervals_abs = [abs(x) for x in pitch_interval(pitches)]
    if not intervals_abs:
        return 0
    return int(mode(intervals_abs))


# Alias for modal_interval / FANTASTIC vs jSymbolic
most_common_interval = modal_interval

@fantastic
@interval_feature
def interval_entropy(pitches: list[int]) -> float:
    """Calculate Shannon entropy of interval distribution.

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
        # Calculate durations in seconds, then convert to quarter notes
        durations_seconds = [float(end - start) for start, end in zip(starts, ends)]
        # Convert seconds to quarter notes: seconds * (tempo/60) = quarter notes
        durations_quarter_notes = [duration * (tempo / 60.0) for duration in durations_seconds]
        return durations_quarter_notes
    except (TypeError, ValueError):
        return []


@midi_toolbox
@interval_feature
def ivdist1(pitches: list[int], starts: list[float], ends: list[float], tempo: float = 120.0) -> dict:
    """Calculate duration-weighted distribution of intervals.

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
        repetitions = max(1, int(duration * 10))  # Ensure at least 1 repetition
        weighted_intervals.extend([interval] * repetitions)

    if not weighted_intervals:
        return {}

    return distribution_proportions(weighted_intervals)

@midi_toolbox
@interval_feature
def ivdirdist1(pitches: list[int]) -> dict[int, float]:
    """Calculate proportion of upward intervals for each interval size (1-12 semitones).
    Implementation based on MIDI toolbox "ivdirdist1.m"
    
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
@interval_feature
def ivsizedist1(pitches: list[int]) -> dict[int, float]:
    """Calculate distribution of interval sizes (0-12 semitones).
    Implementation based on MIDI toolbox "ivsizedist1.m"
    
    Returns the distribution of interval sizes by combining upward and downward 
    intervals of the same absolute size. The first component represents unison (0)
    and the last component represents octave (12).
    
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
@interval_feature
def interval_direction(pitches: list[int]) -> tuple[float, float]:
    """Determine direction of each interval and calculate mean and standard deviation.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of interval directions, where:
        1 represents upward motion
        0 represents same pitch
        -1 represents downward motion
    """
    directions = [
        1 if pitches[i + 1] > pitches[i] else 0 if pitches[i + 1] == pitches[i] else -1
        for i in range(len(pitches) - 1)
    ]

    if not directions:
        return 0.0, 0.0

    mean = sum(directions) / len(directions)
    variance = sum((x - mean) ** 2 for x in directions) / len(directions)
    std_dev = math.sqrt(variance)

    return float(mean), float(std_dev)

@jsymbolic
@interval_feature
def average_length_of_melodic_arcs(pitches: list[int]) -> float:
    """Calculate average number of notes that separate peaks and troughs in melodic arcs.

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
@interval_feature
def average_interval_span_by_melodic_arcs(pitches: list[int]) -> float:
    """Calculate average interval span of melodic arcs.

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
        if direction == -1:  # Arc is currently descending
            if interval < 0:
                interval_so_far += abs(interval)
            elif interval > 0:
                total_intervals += interval_so_far
                number_intervals += 1
                interval_so_far = abs(interval)
                direction = 1

        elif direction == 1:  # Arc is currently ascending
            if interval > 0:
                interval_so_far += abs(interval)
            elif interval < 0:
                total_intervals += interval_so_far
                number_intervals += 1
                interval_so_far = abs(interval)
                direction = -1

        elif direction == 0:  # Arc is currently stationary
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
@interval_feature
def distance_between_most_prevalent_melodic_intervals(pitches: list[int]) -> float:
    """Calculate absolute difference between two most common interval sizes.

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

    interval_counts = {}
    for interval in intervals:
        interval_counts[interval] = interval_counts.get(interval, 0) + 1

    if len(interval_counts) < 2:
        return 0.0

    sorted_intervals = sorted(interval_counts.items(), key=lambda x: x[1], reverse=True)
    most_common = sorted_intervals[0][0]
    second_most_common = sorted_intervals[1][0]
    return float(abs(most_common - second_most_common))

@jsymbolic
@interval_feature
def melodic_interval_histogram(pitches: list[int]) -> dict:
    """Create histogram of interval sizes.

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
@interval_feature
def melodic_large_intervals(pitches: list[int]) -> float:
    """Calculate proportion of intervals >= 13 semitones.

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
    """Calculate proportion of intervals >= specified size.

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
@interval_feature
def melodic_thirds(pitches: list[int]) -> float:
    """Calculate proportion of intervals that are thirds (3 or 4 semitones).
    
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
@interval_feature
def melodic_perfect_fourths(pitches: list[int]) -> float:
    """Calculate proportion of intervals that are perfect fourths (5 semitones).
    
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
@interval_feature
def melodic_tritones(pitches: list[int]) -> float:
    """Calculate proportion of intervals that are tritones (6 semitones).
    
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
@interval_feature
def melodic_perfect_fifths(pitches: list[int]) -> float:
    """Calculate proportion of intervals that are perfect fifths (7 semitones).
    
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
@interval_feature
def melodic_sixths(pitches: list[int]) -> float:
    """Calculate proportion of intervals that are sixths (8 or 9 semitones).
    
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
@interval_feature
def melodic_sevenths(pitches: list[int]) -> float:
    """Calculate proportion of intervals that are sevenths (10 or 11 semitones).
    
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
@interval_feature
def melodic_octaves(pitches: list[int]) -> int:
    """Calculate proportion of intervals that are octaves (12 semitones).
    
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
@interval_feature
def minor_major_third_ratio(pitches: list[int]) -> float:
    """Calculate ratio of minor thirds to major thirds.
    
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
@interval_feature
def direction_of_melodic_motion(pitches: list[int]) -> float:
    """Calculate the proportion of upward melodic motion.
    
    This matches jSymbolic's implementation which calculates the fraction
    of melodic intervals that are ascending in pitch.
    
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
@interval_feature
def number_of_common_melodic_intervals(pitches: list[int]) -> int:
    """Count intervals that appear in at least 9% of melodic transitions.

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
    significant_intervals = nine_percent_significant_values(absolute_intervals)

    return int(len(significant_intervals))

@jsymbolic
@interval_feature
def prevalence_of_most_common_melodic_interval(pitches: list[int]) -> float:
    """Calculate proportion of most common interval.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of most common interval, or 0 if no intervals
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
@interval_feature
def relative_prevalence_of_most_common_melodic_intervals(pitches: list[int]) -> float:
    """Calculate ratio of the frequency of the second most common interval to the frequency of the most common interval.

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

# Dynamic Feature Collection Functions
def _get_features_by_type(feature_type: str) -> dict:
    """Get all features of a specific type.
    
    Parameters
    ----------
    feature_type : str
        The type of features to collect (e.g., 'pitch', 'rhythm', etc.)
        
    Returns
    -------
    dict
        Dictionary mapping feature names to functions
    """
    import inspect
    import sys
    
    # Get the current module
    current_module = sys.modules[__name__]
    
    features = {}
    for name, obj in inspect.getmembers(current_module):
        if (inspect.isfunction(obj) and 
            hasattr(obj, '_feature_types') and 
            feature_type in obj._feature_types):
            features[name] = obj
    
    return features


def get_pitch_features(melody: Melody) -> Dict:
    """Dynamically collect all pitch features for a melody.
    
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
    pitch_functions = _get_features_by_type(FeatureType.PITCH)
    
    for name, func in pitch_functions.items():
        try:
            # Get function signature to determine parameters
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Call function with appropriate parameters
            if 'pitches' in params and 'starts' in params and 'ends' in params:
                result = func(melody.pitches, melody.starts, melody.ends)
            elif 'pitches' in params and 'starts' in params:
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
                # Try with melody object as fallback
                result = func(melody)
            
            features[name] = result
        except Exception as e:
            print(f"Warning: Could not compute {name}: {e}")
            features[name] = None
    
    return features


# Contour Features
@fantastic
@contour_feature
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
@contour_feature
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
@contour_feature
def get_comb_contour_matrix(pitches: list[int]) -> list[list[int]]:
    """Calculate the Marvin & Laprade (1987) comb contour matrix.
    Implementation based on MIDI toolbox "combcontour.m"
    For a melody with n notes, returns an n x n binary matrix C where
    C[i][j] = 1 if pitch of note j is higher than pitch of note i (p[j] > p[i])
    for i >= j (lower triangle including diagonal), and 0 otherwise.
    This follows the MIDI Toolbox definition (combcontour.m), which fills the
    lower-triangular part column-wise via c(k:a,k) = p(k) > p(k:a).

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
@contour_feature
def get_polynomial_contour_features(
    melody: Melody
) -> PolynomialContour:
    """Calculate polynomial contour features.

    Parameters

    Returns
    -------
    List[float]
        List of first 3 polynomial contour coefficients for the melody
    """
    pc = PolynomialContour(melody)
    return pc.coefficients

@fantastic
@contour_feature
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
    return hc.huron_contour

# Duration Features
@fantastic
@jsymbolic
@duration_feature
def _get_tempo(melody: Melody) -> float:
    """Access tempo of melody.

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

# Alias - but don't return again with get_all_features()
inital_tempo = _get_tempo

@jsymbolic
@duration_feature
def mean_tempo(melody: Melody) -> float:
    """Calculate mean tempo of melody.

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

    # Calculate weighted average tempo based on duration each tempo is active
    # Get total duration from last note end time
    total_duration = max(melody.ends) if melody.ends else 0
    if total_duration == 0:
        return melody.tempo

    weighted_sum = 0.0
    last_time = 0.0
    last_tempo = melody.tempo

    # Add up (tempo * duration) for each tempo section
    for time, tempo in melody.tempo_changes:
        duration = time - last_time
        weighted_sum += last_tempo * duration
        last_time = time
        last_tempo = tempo
        
    # Add final section
    final_duration = total_duration - last_time
    weighted_sum += last_tempo * final_duration
    
    return float(weighted_sum / total_duration)

@jsymbolic
@duration_feature
def tempo_variability(melody: Melody) -> float:
    """Calculate variability of tempo of melody.

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
@duration_feature
def duration_range(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """Calculate range between longest and shortest note duration.

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
@duration_feature
def mean_duration(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """Calculate mean note duration in quarter notes.

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
@duration_feature
def average_note_duration(starts: list[float], ends: list[float]) -> float:
    """Calculate average note duration in seconds.

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
@duration_feature
def duration_standard_deviation(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """Calculate standard deviation of note durations in quarter notes.

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
@duration_feature
def variability_of_note_durations(starts: list[float], ends: list[float]) -> float:
    """Calculate standard deviation of note durations in seconds.

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
@duration_feature
def modal_duration(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """Find most common note duration in quarter notes.

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
    
    return float(mode(durations))

@fantastic
@duration_feature
def duration_entropy(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """Calculate Shannon entropy of duration distribution in quarter notes.

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
@duration_feature
def length(starts: list[float]) -> float:
    """Count total number of notes.

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
@duration_feature
def number_of_durations(starts: list[float], ends: list[float], tempo: float = 120.0) -> int:
    """Count number of unique note durations.

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
@duration_feature
def global_duration(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """Calculate total duration in secondsfrom first note start to last note end.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Total duration of melody
    """
    if not starts or not ends or len(starts) == 0 or len(ends) == 0:
        return 0.0
    return float(ends[-1] - starts[0])

# Alias - but don't return again with get_all_features()
# our implementation of duration_in_seconds is not exactly the same as jSymbolic's
# due to differences in how we handle the MIDI, but it should be within a 2% tolerance
duration_in_seconds = global_duration

@fantastic
@jsymbolic
@duration_feature
def note_density(starts: list[float], ends: list[float]) -> float:
    """Calculate average number of notes per second.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Note density (notes per unit time)
    """
    if not starts or not ends or len(starts) == 0 or len(ends) == 0:
        return 0.0
    total_duration = global_duration(starts, ends)
    if total_duration == 0:
        return 0.0
    return float(len(starts) / total_duration)

@jsymbolic
@duration_feature
def note_density_variability(starts: list[float], ends: list[float]) -> float:
    """Calculate variability of note density using 5-second windows.
    
    Divides the melody into 5-second windows and calculates the standard deviation
    of note density across these windows.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
        
    Returns
    -------
    float
        Standard deviation of note density using 5-second windows
    """
    if not starts or not ends or len(starts) < 2:
        return 0.0

    window_size = 5.0
    total_duration = ends[-1] - starts[0]

    # If total duration is less than window size, return 0
    if total_duration < window_size:
        return 0.0

    num_windows = int(total_duration / window_size)
    window_densities = []

    for i in range(num_windows):
        window_start = starts[0] + (i * window_size)
        window_end = window_start + window_size

        notes_in_window = sum(1 for start in starts if window_start <= start < window_end)

        window_density = notes_in_window / window_size
        window_densities.append(window_density)

    if len(window_densities) < 2:
        return 0.0
    return float(np.std(window_densities, ddof=1))

@jsymbolic
@duration_feature
def note_density_per_quarter_note(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """Calculate average number of notes per quarter note.
    
    Finds the average number of note onsets per unit of time corresponding to an
    idealized quarter note duration based on the tempo.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
    tempo : float
        Tempo in BPM (beats per minute), default 120.0
        
    Returns
    -------
    float
        Average number of notes per quarter note duration
    """
    if not starts or len(starts) < 2:
        return 0.0
        
    # Calculate quarter note duration in seconds based on tempo
    quarter_note_duration = 60.0 / tempo
    
    # Calculate total duration in quarter notes
    total_duration = (ends[-1] - starts[0]) / quarter_note_duration
    
    if total_duration == 0:
        return 0.0
        
    # Return average number of notes per quarter note duration
    return float(len(starts) / total_duration)

@jsymbolic
@duration_feature
def note_density_per_quarter_note_variability(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """Calculate variability of note density per quarter note.
    
    Divides the melody into 8-quarter-note windows and calculates the standard deviation
    of note density across these windows.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times
    tempo : float
        Tempo in BPM (beats per minute), default 120.0
        
    Returns
    -------
    float
        Standard deviation of note density across windows
    """
    if not starts or not ends:
        return 0.0
        
    # Calculate window size in seconds (8 quarter notes)
    quarter_note_duration = 60.0 / tempo
    window_size = 8.0 * quarter_note_duration
    
    # Get total duration
    total_duration = ends[-1] - starts[0]
    
    # Create windows
    window_densities = []
    window_start = starts[0]
    while window_start < ends[-1]:
        window_end = min(window_start + window_size, ends[-1])
        
        # Count notes in this window
        notes_in_window = sum(1 for i in range(len(starts)) 
                            if starts[i] >= window_start and starts[i] < window_end)
        
        # Calculate density for this window
        if notes_in_window > 0:
            window_density = notes_in_window / 8.0  # Divide by window size in quarter notes
            window_densities.append(window_density)
            
        window_start += window_size
    
    # Calculate standard deviation of window densities
    if len(window_densities) < 2:
        return 0.0
        
    return float(np.std(window_densities, ddof=1))
@idyom
@duration_feature
def ioi(starts: list[float]) -> list[float]:
    """Calculate the time between consecutive onsets (inter-onset interval).
    
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
@duration_feature
def ioi_mean(starts: list[float]) -> float:
    """Calculate mean of inter-onset intervals.

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
@duration_feature
def ioi_standard_deviation(starts: list[float]) -> float:
    """Calculate standard deviation of inter-onset intervals.

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
@duration_feature
def ioi_ratio(starts: list[float]) -> tuple[float, float]:
    """Calculate mean and standard deviation of inter-onset interval ratios.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of IOI ratios
    """
    if len(starts) < 2:
        return 0.0, 0.0

    # Calculate intervals first
    intervals = [starts[i] - starts[i - 1] for i in range(1, len(starts))]

    if len(intervals) < 2:
        return 0.0, 0.0

    # Calculate ratios between consecutive intervals
    ratios = [intervals[i] / intervals[i - 1] for i in range(1, len(intervals))]
    return float(np.mean(ratios)), float(np.std(ratios, ddof=1))

@novel
@duration_feature
def ioi_range(starts: list[float]) -> float:
    """Calculate range of inter-onset intervals.

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
@duration_feature
def ioi_contour(starts: list[float]) -> tuple[float, float]:
    """Calculate mean and standard deviation of IOI contour.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of contour values (-1: shorter, 0: same, 1: longer)
    """
    intervals = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
    if len(intervals) < 2:
        return 0.0, 0.0

    ratios = [intervals[i] / intervals[i - 1] for i in range(1, len(intervals))]
    contour = [int(np.sign(ratio - 1)) for ratio in ratios]
    return float(np.mean(contour)), float(np.std(contour, ddof=1))

@jsymbolic
@duration_feature
def duration_histogram(starts: list[float], ends: list[float], tempo: float = 120.0) -> dict:
    """Calculate histogram of note durations.

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
    durations = _get_durations(starts, ends, tempo)
    if not durations:
        return {}
    num_durations = max(1, len(set(durations)))
    return histogram_bins(durations, num_durations)

@novel
@duration_feature
def ioi_histogram(starts: list[float]) -> dict:
    """Calculate histogram of inter-onset intervals.

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
@duration_feature
def minimum_note_duration(starts: list[float], ends: list[float]) -> float:
    """Calculate minimum note duration in seconds.

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
@duration_feature
def maximum_note_duration(starts: list[float], ends: list[float]) -> float:
    """Calculate maximum note duration in seconds.

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
@duration_feature
def equal_duration_transitions(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """Calculate proportion of equal duration transitions (d.eq.trans).
    
    Based on Steinbeck (1982) as implemented in FANTASTIC toolbox.
    Measures the relative frequency of duration transitions where the ratio equals 1.0.
    
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
    """
    ratios = get_duration_ratios(starts, ends)
    if not ratios:
        return 0.0
    
    # Count ratios that equal 1.0 (equal durations)
    equal_count = sum(1 for ratio in ratios if ratio == 1.0)
    
    return equal_count / len(ratios)

@fantastic
@duration_feature
def half_duration_transitions(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """Calculate proportion of half/double duration transitions (d.half.trans).
    
    Based on Steinbeck (1982) as implemented in FANTASTIC toolbox.
    Measures transitions where duration is halved (ratio = 0.5) or doubled (ratio = 2.0).
    
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
    """
    ratios = get_duration_ratios(starts, ends)
    if not ratios:
        return 0.0
    
    # Count ratios that equal 0.5 or round to 2
    half_count = sum(1 for ratio in ratios if ratio == 0.5)
    double_count = sum(1 for ratio in ratios if round(ratio) == 2)
    
    return (half_count + double_count) / len(ratios)

@fantastic
@duration_feature
def dotted_duration_transitions(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """Calculate proportion of dotted duration transitions (d.dotted.trans).
    
    Based on Steinbeck (1982) as implemented in FANTASTIC toolbox.
    Measures transitions involving dotted note relationships (1/3 or 3/1 ratios).
    
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
    """
    ratios = get_duration_ratios(starts, ends)
    if not ratios:
        return 0.0
    
    # Count ratios that equal 1/3 or round to 3
    one_third_count = sum(1 for ratio in ratios if abs(ratio - (1/3)) < 1e-10)
    triple_count = sum(1 for ratio in ratios if round(ratio) == 3)
    
    return (one_third_count + triple_count) / len(ratios)

@jsymbolic
@duration_feature
def total_number_of_notes(starts: list[float]) -> int:
    """Calculate total number of notes.
    
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
@duration_feature
def amount_of_staccato(starts: list[float], ends: list[float]) -> float:
    """Calculate amount of staccato. Defined as the number of notes with
    a duration shorter than 0.1 seconds, divided by the total number of notes.
    
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
    durations = _get_durations(starts, ends)
    if not durations:
        return 0.0
    return float(sum(1 for duration in durations if duration < 0.1) / len(durations))

@midi_toolbox
@duration_feature
def mean_duration_accent(starts: list[float], ends: list[float], tau: float = 0.5, accent_index: float = 2.0) -> float:
    """Calculate mean duration accent across all notes.
    
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
    """
    accents = duration_accent(starts, ends, tau, accent_index)
    if not accents:
        return 0.0
    return float(np.mean(accents))

@novel
@duration_feature
def duration_accent_std(starts: list[float], ends: list[float], tau: float = 0.5, accent_index: float = 2.0) -> float:
    """Calculate standard deviation of duration accents.
    
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
    """
    accents = duration_accent(starts, ends, tau, accent_index)
    if not accents:
        return 0.0
    return float(np.std(accents, ddof=1))

@midi_toolbox
@duration_feature
def npvi(starts: list[float], ends: list[float], tempo: float = 120.0) -> float:
    """Calculate normalized Pairwise Variability Index (nPVI) for durations.
    Implementation based on MIDI toolbox "nPVI.m"
    The nPVI measures durational variability of events, originally developed for 
    language research to distinguish stress-timed vs. syllable-timed languages.
    It has been applied to music by Patel & Daniele (2003) to study prosodic
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
@duration_feature
def onset_autocorrelation(starts: list[float], ends: list[float], divisions_per_quarter: int = 4, max_lag_quarters: int = 8) -> list[float]:
    """Calculate autocorrelation function of onset times weighted by duration accents.
    Implementation based on MIDI toolbox "onsetacorr.m"
    
    This function calculates the autocorrelation of onset times weighted by onset durations,
    which are in turn weighted by Parncutt's durational accent (1994). This is useful for
    meter induction and rhythmic analysis.
    
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
@duration_feature
def onset_autocorr_peak(starts: list[float], ends: list[float], divisions_per_quarter: int = 4, max_lag_quarters: int = 8) -> float:
    """Calculate the maximum autocorrelation value (excluding lag 0).
    
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
@fantastic
@tonality_feature
def tonalness(pitches: list[int]) -> float:
    """Calculate tonalness as magnitude of highest key correlation.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Magnitude of highest key correlation value
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlation = compute_tonality_vector(pitch_classes)
    return correlation[0][1]

@fantastic
@tonality_feature
def tonal_clarity(pitches: list[int]) -> float:
    """Calculate ratio between top two key correlation values.

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
@tonality_feature
def tonal_spike(pitches: list[int]) -> float:
    """Calculate ratio between highest key correlation and sum of all others.

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
@tonality_feature
def tonal_entropy(pitches: list[int]) -> float:
    """Calculate tonal entropy as the entropy across the key correlations.

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

    Returns
    -------
    dict[str, int]
        Dictionary mapping key names (both major and minor) to semitone distances from C.
    """
    return {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11,
        "c": 0,
        "c#": 1,
        "d": 2,
        "d#": 3,
        "e": 4,
        "f": 5,
        "f#": 6,
        "g": 7,
        "g#": 8,
        "a": 9,
        "a#": 10,
        "b": 11,
    }

@idyom
@tonality_feature
def referent(pitches: list[int]) -> int:
    """
    Feature that describes the chromatic interval of the key centre from C.
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)

    if not correlations:
        return -1

    # Get the key name from the highest correlation
    key_name = correlations[0][0].split()[
        0
    ]  # Take first word (key name without major/minor)

    # Map key names to semitone distances from C
    key_distances = _get_key_distances()

    return key_distances[key_name]

@idyom
@tonality_feature
def inscale(pitches: list[int]) -> int:
    """
    Captures whether the melody contains any notes which deviate from the estimated key.
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)[0]
    key_centre = correlations[0]

    # Get major/minor scales based on key
    if "major" in key_centre:
        # Major scale pattern: W-W-H-W-W-W-H (W=2 semitones, H=1 semitone)
        scale = [0, 2, 4, 5, 7, 9, 11]
    else:
        # Natural minor scale pattern: W-H-W-W-H-W-W
        scale = [0, 2, 3, 5, 7, 8, 10]

    # Get key root pitch class
    key_name = key_centre.split()[0]
    key_distances = _get_key_distances()
    root = key_distances[key_name]

    # Transpose scale to key
    scale = [(note + root) % 12 for note in scale]

    # Check if any pitch classes are outside the scale
    for pc in pitch_classes:
        if pc not in scale:
            return 0

    return 1

@novel
@tonality_feature
def temperley_likelihood(pitches: list[int]) -> float:
    """
    Calculates the likelihood of a melody using Bayesian reasoning,
    according to David Temperley's model
    (http://davidtemperley.com/wp-content/uploads/2015/11/temperley-cs08.pdf).
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
@tonality_feature
def tonalness_histogram(pitches: list[int]) -> dict:
    """
    Calculates the histogram of KS correlation values.
    """
    p = [p % 12 for p in pitches]
    return histogram_bins(compute_tonality_vector(p)[0][1], 24)

@idyom
@midi_toolbox
@tonality_feature
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
        "registral_direction": registral_direction(pitches),
        "proximity": proximity(pitches),
        "closure": closure(pitches),
        "registral_return": registral_return(pitches),
        "intervallic_difference": intervallic_difference(pitches),
    }


# Melodic Movement Features
@jsymbolic
@complexity_feature
def amount_of_arpeggiation(pitches: list[int]) -> float:
    """Calculate the proportion of notes in the melody that constitute triadic movement.

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
@complexity_feature
def chromatic_motion(pitches: list[int]) -> float:
    """Calculate the proportion of chromatic motion in the melody.

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
@complexity_feature
def melodic_embellishment(
    pitches: list[int], starts: list[float], ends: list[float]
) -> float:
    """Calculate proportion of melodic embellishments (e.g. trills, turns, neighbor tones).

    Melodic embellishments are identified by looking for notes with a duration 1/3rd of the
    adjacent note's duration that move away from and return to a pitch level, or oscillate
    between two pitches.


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
        Proportion of intervals that are embellishments (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.
    """
    return melodic_embellishment_proportion(pitches, starts, ends)

@jsymbolic
@complexity_feature
def repeated_notes(pitches: list[int]) -> float:
    """Calculate the proportion of repeated notes in the melody.

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
@complexity_feature
def stepwise_motion(pitches: list[int]) -> float:
    """Calculate the proportion of stepwise motion in the melody.

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
@complexity_feature
def gradus(pitches: list[int]) -> int:
    """Calculate degree of melodiousness based on Euler's gradus suavitatis (1739).
    Implementation based on MIDI toolbox "gradus.m"
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
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
@complexity_feature
def mobility(pitches: list[int]) -> list[float]:
    """Calculate melodic mobility for each note based on von Hippel (2000).
    Implementation based on MIDI toolbox "mobility.m"
    
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
@complexity_feature
def mean_mobility(pitches: list[int]) -> float:
    """Calculate mean mobility across all notes.
    
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
@complexity_feature
def mobility_std(pitches: list[int]) -> float:
    """Calculate standard deviation of mobility values.
    
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
@complexity_feature
def melodic_attraction(pitches: list[int]) -> list[float]:
    """Calculate melodic attraction according to Lerdahl (1996).
    Implementation based on MIDI toolbox "melattraction.m"
    
    Calculates melodic attraction based on tonal pitch space theory.
    Each tone in a key has certain anchoring strength ("weight") in tonal pitch space.
    Melodic attraction strength is affected by the distance between tones and 
    directed motion patterns.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
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
@complexity_feature
def mean_melodic_attraction(pitches: list[int]) -> float:
    """Calculate mean melodic attraction across all notes.
    
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
@complexity_feature
def melodic_attraction_std(pitches: list[int]) -> float:
    """Calculate standard deviation of melodic attraction values.
    
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
@complexity_feature
def mean_melodic_accent(pitches: list[int]) -> float:
    """Calculate mean melodic accent across all notes.
    
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
@complexity_feature
def melodic_accent_std(pitches: list[int]) -> float:
    """Calculate standard deviation of melodic accents.
    
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
def get_mtype_features(melody: Melody, phrase_gap: float, max_ngram_order: int) -> dict:
    """Calculate various n-gram statistics for the melody.

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
@complexity_feature
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
@complexity_feature
def compltrans(melody: Melody) -> float:
    """Calculate melodic originality measure (Simonton, 1984).
    Implementation based on MIDI toolbox "compltrans.m"
    
    Calculates Simonton's melodic originality score based on 2nd order pitch-class
    distribution derived from 15,618 classical music themes. Higher values indicate
    higher melodic originality (less predictable transitions).
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    float
        Originality score scaled 0-10 (higher = more original/unexpected)
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
@complexity_feature
def complebm(melody: Melody, method: str = 'o') -> float:
    """Calculate expectancy-based melodic complexity (Eerola & North, 2000).
    Implementation based on MIDI toolbox "complebm.m"
    
    Calculates melodic complexity using an expectancy-based model that considers pitch patterns,
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

        note_density_component = note_density(melody.starts, melody.ends) * 0.2

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

        note_density_component = note_density(melody.starts, melody.ends) * 0.4

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


def get_complexity_features(melody: Melody) -> Dict:
    """Dynamically collect all complexity features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of complexity feature values
    """
    features = {}
    complexity_functions = _get_features_by_type(FeatureType.COMPLEXITY)
    
    for name, func in complexity_functions.items():
        try:
            # Skip classes that require corpus_stats (like InverseEntropyWeighting)
            if name == 'InverseEntropyWeighting':
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
    
    # Add Narmour features (they have their own collection function)
    features.update(get_narmour_features(melody))
    
    return features

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
    interval_functions = _get_features_by_type(FeatureType.INTERVAL)
    
    for name, func in interval_functions.items():
        try:
            # Get function signature to determine parameters
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Call function with appropriate parameters
            if 'pitches' in params and 'starts' in params and 'ends' in params and 'tempo' in params:
                result = func(melody.pitches, melody.starts, melody.ends, melody.tempo)
            elif 'pitches' in params and 'starts' in params and 'ends' in params:
                result = func(melody.pitches, melody.starts, melody.ends)
            elif 'pitches' in params and 'interval_level' in params:
                # Special case for variable_melodic_intervals
                result = func(melody.pitches, 7)  # Default interval level
            elif 'pitches' in params:
                result = func(melody.pitches)
            elif 'starts' in params and 'ends' in params:
                result = func(melody.starts, melody.ends)
            else:
                # Try with melody object
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


def get_duration_features(melody: Melody) -> Dict:
    """Dynamically collect all duration features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of duration feature values
    """
    features = {}
    duration_functions = _get_features_by_type(FeatureType.DURATION)
    
    for name, func in duration_functions.items():
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
    
    # Add melody-specific features that aren't functions
    features["meter_numerator"] = melody.meter[0]
    features["meter_denominator"] = melody.meter[1]
    features["metric_stability"] = melody.metric_stability
    
    # Add metric accent features
    features.update(get_metric_accent_features(melody))
    
    return features


def get_tonality_features(melody: Melody) -> Dict:
    """Compute all tonality-based features for a melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    Dict
        Dictionary of tonality-based feature values

    """
    tonality_features = {}

    # Pre-compute pitch classes and tonality vector once
    pitches = melody.pitches
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)

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

    # Key-based features using cached correlations
    if correlations:
        key_name = correlations[0][0].split()[0]
        key_distances = _get_key_distances()
        root = key_distances[key_name]
        tonality_features["referent"] = root

        # Determine scale type and pattern
        is_major = "major" in correlations[0][0]
        scale = [0, 2, 4, 5, 7, 9, 11] if is_major else [0, 2, 3, 5, 7, 8, 10]
        scale = [(note + root) % 12 for note in scale]

        # Check if all notes are in scale
        tonality_features["inscale"] = int(all(pc in scale for pc in pitch_classes))
    else:
        tonality_features["referent"] = -1
        tonality_features["inscale"] = 0

    # Optimize temperley_likelihood calculation
    if len(pitches) > 1:
        # Pre-compute constant arrays
        notes_ints = np.arange(0, 120)
        central_pitch_profile = scipy.stats.norm.pdf(
            notes_ints, loc=68, scale=np.sqrt(5.0)
        )
        central_pitch = np.random.choice(
            notes_ints, p=central_pitch_profile / central_pitch_profile.sum()
        )
        range_profile = scipy.stats.norm.pdf(
            notes_ints, loc=central_pitch, scale=np.sqrt(23.0)
        )

        # Pre-compute key profile
        rpk = (
            np.array(
                [
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
                ]
                * 10
            )
            if is_major
            else np.array(
                [
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
                ]
                * 10
            )
        )

        # Vectorize probability calculation
        total_prob = 1.0
        prev_pitches = np.array(pitches[:-1])
        curr_pitches = np.array(pitches[1:])

        # Calculate all proximity profiles at once
        prox_profiles = scipy.stats.norm.pdf(
            notes_ints[:, np.newaxis], loc=prev_pitches, scale=np.sqrt(10)
        )

        # Calculate probabilities for each note transition
        for i in range(len(prev_pitches)):
            rp = range_profile * prox_profiles[:, i]
            rpk_combined = rp * rpk
            rpk_normed = rpk_combined / np.sum(rpk_combined)
            total_prob *= rpk_normed[curr_pitches[i]]

        tonality_features["temperley_likelihood"] = total_prob
    else:
        tonality_features["temperley_likelihood"] = 0.0

    # Scalar passage features
    tonality_features["longest_monotonic_conjunct_scalar_passage"] = (
        longest_monotonic_conjunct_scalar_passage(pitches, correlations)
    )
    tonality_features["longest_conjunct_scalar_passage"] = (
        longest_conjunct_scalar_passage(pitches, correlations)
    )
    tonality_features["proportion_conjunct_scalar"] = proportion_conjunct_scalar(
        pitches, correlations
    )
    tonality_features["proportion_scalar"] = proportion_scalar(pitches, correlations)

    # Histogram using cached correlations
    tonality_features["tonalness_histogram"] = histogram_bins(correlations[0][1], 24)

    tonality_features["mode"] = "major" if is_major else "minor"

    return tonality_features


def get_melodic_movement_features(melody: Melody) -> Dict:
    """Compute all melodic movement-based features for a melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    Dict
        Dictionary of melodic movement-based feature values

    """
    movement_features = {}

    movement_features["amount_of_arpeggiation"] = amount_of_arpeggiation(melody.pitches)
    movement_features["chromatic_motion"] = chromatic_motion(melody.pitches)
    movement_features["melodic_embellishment"] = melodic_embellishment(
        melody.pitches, melody.starts, melody.ends
    )
    movement_features["repeated_notes"] = repeated_notes(melody.pitches)
    movement_features["stepwise_motion"] = stepwise_motion(melody.pitches)

    return movement_features


def process_melody(args):
    """Process a single melody and return its features.

    Parameters
    ----------
    args : tuple
        Tuple containing (melody_data, corpus_stats, idyom_features, phrase_gap, max_ngram_order)

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

    melody_data, corpus_stats, idyom_results_dict, phrase_gap, max_ngram_order = args
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
    duration_features = get_duration_features(mel)
    timings["duration"] = time.time() - start

    start = time.time()
    tonality_features = get_tonality_features(mel)
    timings["tonality"] = time.time() - start

    start = time.time()
    melodic_movement_features = get_melodic_movement_features(mel)
    timings["melodic_movement"] = time.time() - start

    start = time.time()
    mtype_features = get_mtype_features(
        mel, phrase_gap=phrase_gap, max_ngram_order=max_ngram_order
    )
    timings["mtype"] = time.time() - start

    start = time.time()
    complexity_features = get_complexity_features(mel)
    timings["complexity"] = time.time() - start

    melody_features = {
        "pitch_features": pitch_features,
        "interval_features": interval_features,
        "contour_features": contour_features,
        "duration_features": duration_features,
        "tonality_features": tonality_features,
        "melodic_movement_features": melodic_movement_features,
        "mtype_features": mtype_features,
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
) -> dict:
    logger = logging.getLogger("melody_features")
    """Run IDyOM on the input MIDI directory and return mean information content for each melody.
    Uses the parameters supplied from Config dataclass to control IDyOM behaviour.

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
        "Creating temporary MIDI files with detected key signatures for IDyOM processing..."
    )
    temp_dir = tempfile.mkdtemp(prefix="idyom_key_")
    original_input_dir = input_directory
    input_directory = create_temp_midi_with_key_signature(input_directory, temp_dir)

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


def create_temp_midi_with_key_signature(input_directory: str, temp_dir: str) -> str:
    """
    Create temporary MIDI files with key signatures for IDyOM processing.

    Parameters
    ----------
    input_directory : str
        Path to the input directory containing MIDI files
    temp_dir : str
        Path to the temporary directory to create the modified MIDI files

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
        f"Processing {len(midi_files)} MIDI files for key signature detection..."
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

            # Now try to modify it with key signature detection
            try:
                midi_dict = import_midi(midi_file)
                if midi_dict is None:
                    logger.warning(f"Could not import {midi_file}, using original file")
                    continue

                melody = Melody(midi_dict)
                if not melody.pitches:  # Skip if no pitches
                    logger.warning(
                        f"No pitches found in {midi_file}, using original file"
                    )
                    continue

                pitch_classes = [pitch % 12 for pitch in melody.pitches]

                key_correlations = compute_tonality_vector(pitch_classes)
                detected_key = key_correlations[0][0]  # Get the most likely key

                mido_key = to_mido_key_string(detected_key)

                mid = MidiFile(midi_file)

                # Remove existing key signatures and add new one
                for track in mid.tracks:
                    track[:] = [
                        msg for msg in track if not (msg.type == "key_signature")
                    ]

                # Add new key signature at the beginning
                key_msg = MetaMessage("key_signature", key=mido_key, time=0)
                mid.tracks[0].insert(0, key_msg)

                # Save the modified MIDI file with .mid extension
                mid.save(output_path)

            except Exception as e:
                logger.warning(
                    f"Could not add key signature to {midi_file}: {str(e)}, using original file"
                )
                # The original file was already copied, so we're good

        except Exception as e:
            logger.error(f"Could not copy {midi_file}: {str(e)}")
            continue

    # Verify files were created (all should be .mid files now)
    created_files = glob.glob(os.path.join(temp_dir, "*.mid"))

    logger.info(
        f"Successfully created {len(created_files)} files in temporary directory"
    )

    return temp_dir


# e.g.
# idyom_configs = {
#     "pitch": IDyOMConfig(models="both", corpus_path="path/to/corpus1", target_viewpoints=["cpitch"], source_viewpoints=["cpint", "cpintfref"], ppm_order=1),
#     "pitch_stm": IDyOMConfig(models="stm", corpus_path="path/to/corpus1", target_viewpoints=["cpitch"], source_viewpoints=["cpint", "cpintfref"], ppm_order=1),
#     "rhythm : IDyOMConfig(corpus_path="path/to/corpus1", target_viewpoints=["onset"], source_viewpoints=["ioi"], ppm_order=2),
#     "rhythm_stm : IDyOMConfig(models="stm", corpus_path="path/to/corpus1", target_viewpoints=["onset"], source_viewpoints=["ioi"], ppm_order=2),
# }
# config = Config(corpus_path="path/to/corpus", idyom_configs=idyom_configs, fantastic_max_ngram_order=3)





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
            # I hate this so much
            corpus=resources.files("melody_features") / "corpora/Essen_Corpus",
            # corpus=str(Path(__file__).parent.parent.parent / "corpora/Essen_Corpus"),
            idyom={
                "default_pitch": IDyOMConfig(
                    target_viewpoints=["cpitch"],
                    source_viewpoints=[("cpint", "cpintfref")],
                    ppm_order=1,
                    models=":both",
                    corpus=None,
                )
            },
            fantastic=FantasticConfig(max_ngram_order=6, phrase_gap=1.5, corpus=None),
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
        "duration_features": get_duration_features(mel),
        "tonality_features": get_tonality_features(mel),
        "melodic_movement_features": get_melodic_movement_features(mel),
        "mtype_features": get_mtype_features(
            mel,
            phrase_gap=config.fantastic.phrase_gap,
            max_ngram_order=config.fantastic.max_ngram_order,
        ),
        "complexity_features": get_complexity_features(mel),
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
        )
        for melody_data in melody_data_list
    ]

    # Track timing statistics
    timing_stats = {
        "pitch": [],
        "interval": [],
        "contour": [],
        "duration": [],
        "tonality": [],
        "melodic_movement": [],
        "mtype": [],
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


def get_fantastic_features(melody: Melody) -> Dict:
    """Get all FANTASTIC features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to extract features from
        
    Returns
    -------
    Dict
        Dictionary containing all FANTASTIC features
    """
    return _compute_features_by_source(melody, "fantastic")


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


def _compute_features_by_source(melody: Melody, source: str) -> Dict:
    """Compute all features for a melody that are decorated with a specific source.
    
    Parameters
    ----------
    melody : Melody
        The melody to extract features from
    source : str
        The source label to filter by
        
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
                elif param == "phrase_gap":
                    args.append(1.5)
                elif param == "max_ngram_order":
                    args.append(6)
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
