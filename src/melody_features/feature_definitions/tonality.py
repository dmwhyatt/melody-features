"""Tonality feature definitions."""

from typing import Dict, Literal, Optional

import numpy as np

from ..algorithms import compute_tonality_vector
from ..distributional import histogram_bins
from .expectation import _get_key_distances
from ..feature_decorators import fantastic, idyom, midi_toolbox, novel, partitura, pitch, tonality
from ..representations import Melody
from ..tonal_tension import ALPHA, BETA, DEFAULT_WEIGHTS, SCALE_FACTOR, estimate_tonaltension


__all__ = [
    "infer_key_from_pitches",
    "key",
    "keyname",
    "tonalness",
    "tonal_clarity",
    "tonal_spike",
    "referent",
    "tonal_tension",
    "mean_cloud_diameter",
    "std_cloud_diameter",
    "mean_cloud_momentum",
    "std_cloud_momentum",
    "mean_tensile_strain",
    "std_tensile_strain",
    "tonalness_histogram",
    "inscale",
    "proportion_inscale",
    "longest_monotonic_conjunct_scalar_passage",
    "longest_conjunct_scalar_passage",
    "proportion_conjunct_scalar",
    "proportion_scalar",
    "mode",
    "get_tonality_features",
]


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
    from ..algorithms import longest_monotonic_conjunct_scalar_passage as _longest_monotonic_conjunct_scalar_passage
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
    from ..algorithms import longest_conjunct_scalar_passage as _longest_conjunct_scalar_passage
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
    from ..algorithms import proportion_conjunct_scalar as _proportion_conjunct_scalar
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
    from ..algorithms import proportion_scalar as _proportion_scalar
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
    from ..algorithms import longest_monotonic_conjunct_scalar_passage as _longest_monotonic_conjunct_scalar_passage
    from ..algorithms import longest_conjunct_scalar_passage as _longest_conjunct_scalar_passage
    scalar_correlations = correlations
    if key_for_features:
        scalar_correlations = _tonality_correlations_for_key(correlations, key_for_features)

    tonality_features["longest_monotonic_conjunct_scalar_passage"] = (
        _longest_monotonic_conjunct_scalar_passage(melody.pitches, scalar_correlations)
    )
    tonality_features["longest_conjunct_scalar_passage"] = (
        _longest_conjunct_scalar_passage(melody.pitches, scalar_correlations)
    )
    from ..algorithms import proportion_conjunct_scalar as _proportion_conjunct_scalar
    from ..algorithms import proportion_scalar as _proportion_scalar
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
