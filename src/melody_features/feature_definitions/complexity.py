"""Complexity feature definitions."""

import numpy as np

from ..algorithms import compute_tonality_vector
from ..algorithms import must as must_algorithms
from ..melody_tokenizer import MustTokenizer
from ..feature_decorators import both, complexity, fantastic, midi_toolbox, must, novel, pitch, rhythm
from ..feature_utils import _get_durations, mean_and_std, population_mean_and_std
from ..algorithms.meter_estimation import duration_accent as _duration_accent
from .metre import _meter_accent_mean
from .pitch_class import _pcdist1_vector
from .pitch_interval import _ivdist1_vector, pitch_interval
from ..core.representations import Melody
from ..utils.stats import distribution_entropy, midi_toolbox_entropy, shannon_entropy
from .timing import _durdist1_vector


_MUST_TOKENIZER = MustTokenizer()

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
    "bisect_unbalance",
    "center_mass_offset",
    "event_heterogeneity",
    "av_abs_interval",
    "mel_abruptness",
    "dur_abruptness",
    "rhythm_abruptness",
    "asym_total",
    "asym_index",
    "event_density",
    "av_local_p1_entropy",
    "p1_entropy",
    "p2_entropy",
    "p3_entropy",
    "i1_entropy",
    "i2_entropy",
    "d1_entropy",
    "d2_entropy",
    "d3_entropy",
    "wp_entropy",
    "pdist1",
    "pdist2",
    "pdist3",
    "idist1",
    "idist2",
    "ddist1",
    "ddist2",
    "ddist3",
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
    mean, _ = mean_and_std(accents)
    return mean

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
    _, std = mean_and_std(accents)
    return std

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
        rhyvar = float(np.std(np.log(du_sec), ddof=0))
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


@must
@complexity
@rhythm
def bisect_unbalance(melody: Melody) -> float:
    """The bisect unbalance of a melody's temporal distribution of note onsets.

    Measures equilibrium between the first and second halves of the stimulus.
    Computed as ``1 - 4 * f1 * f2``, where ``f1`` and ``f2`` are the proportions
    of note onsets falling before and after the temporal midpoint, respectively.
    Values near 1 indicate balanced onset placement; lower values indicate
    concentration of events in one half.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Bisect unbalance score
    """
    return must_algorithms.bisect_unbalance(melody)


@must
@complexity
@rhythm
def center_mass_offset(melody: Melody) -> float:
    """The center of mass offset of a melody's note-onset distribution.

    The absolute distance between the temporal center of mass (mean onset time,
    expressed as a proportion of total stimulus duration) and the geometric
    center (0.5). Values near 0 indicate a centrally concentrated onset
    distribution.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Center of mass offset
    """
    return must_algorithms.center_mass_offset(melody)


@must
@complexity
@rhythm
def event_heterogeneity(melody: Melody) -> float:
    """The event heterogeneity of a melody's temporal distribution of onsets.

    First computes a local unbalance curve using sliding windows sized to
    contain two note events (stepped at half the window length), then returns
    the distance-weighted mean squared deviation of that curve from unity.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Event heterogeneity score
    """
    return must_algorithms.event_heterogeneity(melody)


@must
@complexity
@pitch
def av_abs_interval(melody: Melody) -> float:
    """The mean log-transformed absolute melodic interval size.

    Computed as the mean of ``log(abs(interval) + 1)`` over consecutive pitch
    pairs, where intervals are measured in semitones.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Average log absolute interval
    """
    return must_algorithms.av_abs_interval(melody)


@must
@complexity
@pitch
def mel_abruptness(melody: Melody) -> float:
    """The melodic abruptness of pitch-direction changes in a melody.

    For each interior note where the pitch contour changes direction, accumulates
    the natural logarithm of the mean absolute interval size at that turning
    point, then normalizes by total stimulus duration in seconds.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Melodic abruptness score
    """
    return must_algorithms.mel_abruptness(melody)


@must
@complexity
@rhythm
def dur_abruptness(melody: Melody) -> float:
    """The durational abruptness of pitch-direction changes in a melody.

    The proportion of total note duration (in seconds) accounted for by notes
    at which the pitch contour changes direction.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Durational abruptness score
    """
    return must_algorithms.dur_abruptness(melody)


@must
@complexity
@rhythm
def rhythm_abruptness(melody: Melody) -> float:
    """The rhythmic abruptness of consecutive note durations.

    The mean ratio of consecutive beat durations after applying Parncutt (1994)
    duration accent, taking the larger-over-smaller ratio at each successive
    pair of notes.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Rhythmic abruptness score

    Note
    -----
    Duration accent follows the MIDI Toolbox ``duraccent`` defaults (``tau=0.5``,
    ``accent_index=2.0``).

    Citation
    --------
    Parncutt (1994)
    """
    return must_algorithms.rhythm_abruptness(melody)


@must
@complexity
@pitch
def pdist1(melody: Melody) -> dict:
    """Pitch distribution (MUST ``pdist1.m``).

    Returns normalized weights keyed by MIDI pitch.
    """
    if not melody.pitches:
        return {}
    return _MUST_TOKENIZER.pdist1(melody).as_dict()


@must
@complexity
@pitch
def pdist2(melody: Melody) -> dict:
    """2-tuple pitch distribution (MUST ``pdist2.m``).

    Returns normalized weights keyed by consecutive pitch pairs.
    """
    if not melody.pitches:
        return {}
    return _MUST_TOKENIZER.pdist2(melody).as_dict()


@must
@complexity
@pitch
def pdist3(melody: Melody) -> dict:
    """3-tuple pitch distribution (MUST ``pdist3.m``).

    Returns normalized weights keyed by consecutive pitch triples.
    """
    if not melody.pitches:
        return {}
    return _MUST_TOKENIZER.pdist3(melody).as_dict()


@must
@complexity
@pitch
def idist1(melody: Melody) -> dict:
    """Interval distribution marginalized from ``pdist2`` (MUST ``idist1.m``)."""
    if not melody.pitches:
        return {}
    return _MUST_TOKENIZER.idist1(melody).as_dict()


@must
@complexity
@pitch
def idist2(melody: Melody) -> dict:
    """2-interval distribution marginalized from ``pdist3`` (MUST ``idist2.m``)."""
    if not melody.pitches:
        return {}
    return _MUST_TOKENIZER.idist2(melody).as_dict()


@must
@complexity
@rhythm
def ddist1(melody: Melody) -> dict:
    """Duration distribution in beats (MUST ``ddist1.m``).

    The final note duration is excluded, consistent with the MUST ``ddist*``
    convention. Durations are rounded to two decimal places in beats.
    """
    if not melody.pitches:
        return {}
    return _MUST_TOKENIZER.ddist1(melody).as_dict()


@must
@complexity
@rhythm
def ddist2(melody: Melody) -> dict:
    """2-tuple duration distribution in beats (MUST ``ddist2.m``)."""
    if not melody.pitches:
        return {}
    return _MUST_TOKENIZER.ddist2(melody).as_dict()


@must
@complexity
@rhythm
def ddist3(melody: Melody) -> dict:
    """3-tuple duration distribution in beats (MUST ``ddist3.m``)."""
    if not melody.pitches:
        return {}
    return _MUST_TOKENIZER.ddist3(melody).as_dict()


@must
@complexity
@both
def asym_total(melody: Melody) -> float:
    """The total vertical mirror asymmetry of a melody.

    Mirrors the MUST ``asymTotal`` implementation: build a sustained-pitch time
    series, compare each sample to its temporally reversed counterpart, and
    return the mean absolute pitch difference.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Total asymmetry in semitones (time-averaged)

    Note
    ----
    In the MUST/MIDI Toolbox notematrix, onset (column 1) and duration
    (column 2) are in beats. The reference MATLAB code samples with
    ``for t = 0:0.0001:T`` using that same unit. There is no separate
    millisecond grid.

    This implementation converts ``Melody`` timing to beats and uses the
    same 0.0001 beat step. The real-time length of one step depends on tempo 
    (e.g. 0.1 ms only at 60 BPM, 0.05 ms at 120 BPM).
    """
    return must_algorithms.asym_total(melody)


@must
@complexity
@both
def asym_index(melody: Melody) -> float:
    """The vertical mirror asymmetry index of a melody.

    Mirrors the MUST ``asymIndex`` implementation: the proportion of sampled
    time points at which pitch differs from its temporally mirrored counterpart.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Asymmetry index in the range [0, 1]

    Note
    ----
    In the MUST/MIDI Toolbox notematrix, onset (column 1) and duration
    (column 2) are in beats. The reference MATLAB code samples with
    ``for t = 0:0.0001:T`` using that same unit. There is no separate
    millisecond grid.

    This implementation converts ``Melody`` timing to beats and uses the
    same 0.0001 beat step. The real-time length of one step depends on tempo 
    (e.g. 0.1 ms only at 60 BPM, 0.05 ms at 120 BPM).
    """
    return must_algorithms.asym_index(melody)


@must
@complexity
@rhythm
def event_density(melody: Melody) -> float:
    """The event density of a melody.

    The number of note events divided by total stimulus duration in seconds.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Event density in notes per second
    """
    return must_algorithms.event_density(melody)


@must
@complexity
@pitch
def av_local_p1_entropy(melody: Melody) -> float:
    """The average local zeroth-order pitch entropy across a melody.

    Computes Shannon entropy of the pitch distribution within sliding one-second
    windows advanced in 0.25-second steps, using an inclusive upper onset bound,
    then returns the mean entropy across windows.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Average local pitch entropy

    Note
    -----
    Entropy is computed with the natural logarithm, consistent with the MUST
    Toolbox ``shentropy`` implementation.
    """
    return must_algorithms.av_local_p1_entropy(melody)


@must
@complexity
@pitch
def p1_entropy(melody: Melody) -> float:
    """The zeroth-order pitch entropy of a melody.

    Shannon entropy of the marginal pitch distribution.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Pitch entropy

    Note
    -----
    Entropy is computed with the natural logarithm, consistent with the MUST
    Toolbox ``shentropy`` implementation.
    """
    return must_algorithms.p1_entropy(melody)


@must
@complexity
@pitch
def p2_entropy(melody: Melody) -> float:
    """The first-order (2-tuple) pitch entropy of a melody.

    Shannon entropy of the distribution of consecutive pitch pairs.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        2-tuple pitch entropy

    Note
    -----
    Entropy is computed with the natural logarithm, consistent with the MUST
    Toolbox ``shentropy`` implementation.
    """
    return must_algorithms.p2_entropy(melody)


@must
@complexity
@pitch
def p3_entropy(melody: Melody) -> float:
    """The second-order (3-tuple) pitch entropy of a melody.

    Shannon entropy of the distribution of consecutive pitch triples.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        3-tuple pitch entropy

    Note
    -----
    Entropy is computed with the natural logarithm, consistent with the MUST
    Toolbox ``shentropy`` implementation.
    """
    return must_algorithms.p3_entropy(melody)


@must
@complexity
@pitch
def i1_entropy(melody: Melody) -> float:
    """The zeroth-order interval entropy of a melody.

    Shannon entropy of the distribution of consecutive melodic intervals,
    weighted by the underlying 2-tuple pitch distribution.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Interval entropy

    Note
    -----
    Entropy is computed with the natural logarithm, consistent with the MUST
    Toolbox ``shentropy`` implementation.
    """
    return must_algorithms.i1_entropy(melody)


@must
@complexity
@pitch
def i2_entropy(melody: Melody) -> float:
    """The first-order (2-tuple) interval entropy of a melody.

    Shannon entropy of the distribution of consecutive interval pairs, weighted
    by the underlying 3-tuple pitch distribution.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        2-tuple interval entropy

    Note
    -----
    Entropy is computed with the natural logarithm, consistent with the MUST
    Toolbox ``shentropy`` implementation.
    """
    return must_algorithms.i2_entropy(melody)


@must
@complexity
@rhythm
def d1_entropy(melody: Melody) -> float:
    """The zeroth-order duration entropy of a melody.

    Shannon entropy of the distribution of note durations in quarter-note beats.
    The final note duration is excluded, consistent with the MUST ``ddist1``
    convention.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Duration entropy

    Note
    -----
    Entropy is computed with the natural logarithm, consistent with the MUST
    Toolbox ``shentropy`` implementation. Durations are rounded to two decimal
    places before binning.
    """
    return must_algorithms.d1_entropy(melody)


@must
@complexity
@rhythm
def d2_entropy(melody: Melody) -> float:
    """The first-order (2-tuple) duration entropy of a melody.

    Shannon entropy of the distribution of consecutive duration pairs in
    quarter-note beats. The final note duration is excluded.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        2-tuple duration entropy

    Note
    -----
    Entropy is computed with the natural logarithm, consistent with the MUST
    Toolbox ``shentropy`` implementation. Durations are rounded to two decimal
    places before binning.
    """
    return must_algorithms.d2_entropy(melody)


@must
@complexity
@rhythm
def d3_entropy(melody: Melody) -> float:
    """The second-order (3-tuple) duration entropy of a melody.

    Shannon entropy of the distribution of consecutive duration triples in
    quarter-note beats. The final note duration is excluded.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        3-tuple duration entropy

    Note
    -----
    Entropy is computed with the natural logarithm, consistent with the MUST
    Toolbox ``shentropy`` implementation. Durations are rounded to two decimal
    places before binning.
    """
    return must_algorithms.d3_entropy(melody)


@must
@complexity
@pitch
def wp_entropy(melody: Melody) -> float:
    """The weighted permutation entropy of a melody's pitch sequence.

    Classifies each consecutive 3-note pitch window into one of 13 order
    signatures, weights each class by the standard deviation of the three
    pitches, and computes Shannon entropy over the resulting distribution.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Weighted permutation entropy

    Note
    -----
    Entropy is computed with the natural logarithm, consistent with the MUST
    Toolbox ``shentropy`` implementation.
    """
    return must_algorithms.wp_entropy(melody)
