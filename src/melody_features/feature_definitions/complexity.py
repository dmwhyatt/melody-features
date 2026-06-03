"""Complexity feature definitions."""

import numpy as np

from ..algorithms import compute_tonality_vector
from ..feature_decorators import both, complexity, fantastic, midi_toolbox, novel, pitch, rhythm
from ..feature_utils import _get_durations
from ..meter_estimation import duration_accent as _duration_accent
from .metre import _meter_accent_mean
from .pitch_class import _pcdist1_vector
from .pitch_interval import _ivdist1_vector, pitch_interval
from ..representations import Melody
from ..stats import distribution_entropy, midi_toolbox_entropy, shannon_entropy
from .timing import _durdist1_vector


__all__ = [
    "pitch_entropy",
    "interval_entropy",
    "duration_entropy",
    "duration_accent",
    "mean_duration_accent",
    "duration_accent_std",
    "tonal_entropy",
    "gradus",
    "complebm_pitch",
    "complebm_rhythm",
    "complebm_optimal",
]


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

_KK_MAJ_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=float,
)

_KK_MIN_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=float,
)

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
