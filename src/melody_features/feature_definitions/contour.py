"""Contour feature definitions."""

from typing import Dict, List, Tuple

from ..feature_decorators import contour, fantastic, midi_toolbox, pitch
from ..feature_utils import _get_durations
from ..contour import (
    HuronContour,
    InterpolationContour,
    PolynomialContour,
    StepContour,
)
from ..core.representations import Melody


__all__ = [
    "get_step_contour_features",
    "get_interpolation_contour_features",
    "comb_contour_matrix",
    "get_comb_contour_matrix",
    "get_polynomial_contour_features",
    "get_huron_contour_features",
    "get_contour_features",
]


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
