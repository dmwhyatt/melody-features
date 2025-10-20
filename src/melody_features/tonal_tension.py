"""
The code below is adapted from the partitura library, and is therefore 
licenced under their Apache-2.0 licence. https://github.com/CPJKU/partitura. 
We make no alteration to the functionality of their code, but adapt it 
to work with our internal Melody object. All credit for this code remains 
with the original authors.

This module contains methods to compute Chew's  spiral array representation
and the tonal tension profiles using Herreman and Chew's tension ribbons

References
----------
D. Herremans and E. Chew (2016) Tension ribbons: Quantifying and
visualising tonal tension. Proceedings of the Second International
Conference on Technologies for Music Notation and Representation
(TENOR), Cambridge, UK.
"""
import warnings
import os

import numpy as np
import scipy.spatial.distance as distance
from scipy.interpolate import interp1d

from .pitch_spelling import (
    ensure_notearray,
    get_time_units_from_note_array,
    estimate_spelling,
)
from .algorithms import compute_tonality_vector

__all__ = ["estimate_tonaltension"]

# Scaling factors
A = np.sqrt(2.0 / 15.0) * np.pi / 2.0
R = 1.0

# From Elaine Chew's thesis
DEFAULT_WEIGHTS = np.array([0.516, 0.315, 0.168])
ALPHA = 0.75
BETA = 0.75

STEPS_BY_FIFTHS = ["F", "C", "G", "D", "A", "E", "B"]

NOTES_BY_FIFTHS = []
for alt in range(-4, 5):
    NOTES_BY_FIFTHS += [(step, alt) for step in STEPS_BY_FIFTHS]

# Index of C
C_IDX = NOTES_BY_FIFTHS.index(("C", 0))
# C lies in the middle of the spiral array
T = (np.arange(len(NOTES_BY_FIFTHS)) - C_IDX) * np.pi / 2.0


def add_field(arr, new_dtype):
    """Return a copy of structured array `arr` with extra fields.

    new_dtype can be a numpy dtype or a list of (name, dtype) field specs.
    """
    if isinstance(new_dtype, list):
        new_fields = new_dtype
    else:
        new_fields = np.dtype(new_dtype).descr

    # Build combined dtype, preserving existing order
    combined_descr = list(arr.dtype.descr) + list(new_fields)
    out = np.empty(arr.shape, dtype=combined_descr)
    # Copy over existing data
    for name in arr.dtype.names:
        out[name] = arr[name]
    return out


def _keyname_to_pc(name: str) -> int:
    name = name.strip()
    # Normalize capitalization of pitch class token
    root = name.split()[0]
    pc_map = {
        "C": 0, "C#": 1, "Db": 1,
        "D": 2, "D#": 3, "Eb": 3,
        "E": 4, "Fb": 4, "E#": 5,
        "F": 5, "F#": 6, "Gb": 6,
        "G": 7, "G#": 8, "Ab": 8,
        "A": 9, "A#": 10, "Bb": 10,
        "B": 11, "Cb": 11, "B#": 0,
        "c": 0, "c#": 1, "db": 1,
        "d": 2, "d#": 3, "eb": 3,
        "e": 4, "fb": 4, "e#": 5,
        "f": 5, "f#": 6, "gb": 6,
        "g": 7, "g#": 8, "ab": 8,
        "a": 9, "a#": 10, "bb": 10,
        "b": 11, "cb": 11, "b#": 0,
    }
    return pc_map.get(root, 0)


def _pc_to_fifths_major(pc: int) -> int:
    pc_to_fifths_major = {
        0: 0,   # C
        7: 1,   # G
        2: 2,   # D
        9: 3,   # A
        4: 4,   # E
        11: 5,  # B
        6: 6,   # F#
        1: 7,   # C#
        5: -1,  # F
        10: -2, # Bb
        3: -3,  # Eb
        8: -4,  # Ab
    }
    return pc_to_fifths_major.get(pc, 0)


def best_key_from_tonality_vector(tonality_vector):
    """Extract (fifths, mode) from a compute_tonality_vector result.

    tonality_vector: list[(key_name, correlation)] sorted by correlation desc.
    mode: 1 for major, -1 for minor
    """
    if not tonality_vector:
        return 0, 1
    key_name = tonality_vector[0][0]
    is_major = "major" in key_name
    pc = _keyname_to_pc(key_name)
    if is_major:
        fifths = _pc_to_fifths_major(pc)
        mode = 1
    else:
        # For minor, use relative major (pc + 3)
        rel_maj_pc = (pc + 3) % 12
        fifths = _pc_to_fifths_major(rel_maj_pc)
        mode = -1
    return int(fifths), int(mode)


def extract_key_signature_from_midi(midi_path):
    """Extract key signature from MIDI file if present.
    
    Parameters
    ----------
    midi_path : str
        Path to MIDI file
    
    Returns
    -------
    tuple or None
        (fifths, mode) where mode is 1 for major, -1 for minor, or None if no key signature found
    """
    import mido
    
    if not os.path.exists(midi_path):
        return None
        
    mid = mido.MidiFile(midi_path)
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'key_signature':
                key_str = msg.key
                
                is_minor = key_str.endswith('m')
                mode = -1 if is_minor else 1
                
                root = key_str[:-1] if is_minor else key_str
                
                pc = _keyname_to_pc(root)
                
                if is_minor:
                    rel_maj_pc = (pc + 3) % 12
                    fifths = _pc_to_fifths_major(rel_maj_pc)
                else:
                    fifths = _pc_to_fifths_major(pc)
                
                return int(fifths), int(mode)

def e_distance(x, y):
    """
    Euclidean distance between two points
    """
    return np.sqrt(((x - y) ** 2).sum())


def helical_to_cartesian(t, r=R, a=A):
    """
    Transform helical coordinates to cartesian
    """
    x = r * np.sin(t)
    y = r * np.cos(t)
    z = a * t

    return x, y, z


def ensure_norm(x):
    """
    Ensure that vectors are normalized
    """
    if not np.isclose(x.sum(), 1):
        return x / x.sum()
    else:
        return x


X, Y, Z = helical_to_cartesian(T)
PITCH_COORDINATES = np.column_stack((X, Y, Z))
MAJOR_IDXS = np.array([0, 1, 4], dtype=int)
MINOR_IDXS = np.array([0, 1, -3], dtype=int)

# The scaling factor is the distance between C and B#, as used
# in Cancino-Chacón and Grachten (2018)
SCALE_FACTOR = 1.0 / e_distance(
    PITCH_COORDINATES[C_IDX], PITCH_COORDINATES[NOTES_BY_FIFTHS.index(("B", 1))]
)


def major_chord(tonic_idx, w=DEFAULT_WEIGHTS):
    """
    Major chord representation in the spiral array space.

    Parameters
    ----------
    tonic_idx : int
        Index of the root of the chord in NOTES_BY_FIFTHS
    w : array
        3D vector containing the tonal weights. Default is DEFAULT_WEIGHTS.

    Returns
    -------
    chord : array
        Vector representation of the chord
    """
    return np.dot(ensure_norm(w), PITCH_COORDINATES[MAJOR_IDXS + tonic_idx])


def minor_chord(tonic_idx, w=DEFAULT_WEIGHTS):
    """
    Minor chord representation in the spiral array space.

    Parameters
    ----------
    tonic_idx : int
        Index of the root of the chord in NOTES_BY_FIFTHS
    w : array
        3D vector containing the tonal weights. Default is DEFAULT_WEIGHTS.

    Returns
    -------
    chord : array
        Vector representation of the chord
    """
    return np.dot(ensure_norm(w), PITCH_COORDINATES[MINOR_IDXS + tonic_idx])


def major_key(tonic_idx, w=DEFAULT_WEIGHTS):
    """
    Major key representation in the spiral array space.

    Parameters
    ----------
    tonic_idx : int
        Index of the tonic of the key in NOTES_BY_FIFTHS
    w : array
        3D vector containing the tonal weights. Default is DEFAULT_WEIGHTS.

    Returns
    -------
    ce : array
        Vector representation of the center of effect of the key
    """

    chords = np.array(
        [
            major_chord(tonic_idx, w),
            major_chord(tonic_idx + 1, w),
            major_chord(tonic_idx - 1, w),
        ]
    )

    return np.dot(ensure_norm(w), chords)


def minor_key(tonic_idx, w=DEFAULT_WEIGHTS, alpha=ALPHA, beta=BETA):
    """
    Minor key representation in the spiral array space

    Parameters
    ----------
    tonic_idx : int
        Index of the tonic of the key in NOTES_BY_FIFTHS
    w : array
        3D vector containing the tonal weights. Default is DEFAULT_WEIGHTS.
    alpha : float
        Preference for V vs v chord in minor key (should lie between 0 and 1)
    beta : float
        Preference for iv vs IV in minor key (should lie between 0 and 1)

    Returns
    -------
    ce : array
        Vector representation of the center of effect of the key
    """

    if alpha > 1.0 or alpha < 0:
        raise ValueError("`alpha` should be between 0 and 1.")

    if beta > 1.0 or beta < 0:
        raise ValueError("`beta` should be between 0 and 1.")

    chords = np.array(
        [
            minor_chord(tonic_idx, w),
            (
                alpha * major_chord(tonic_idx + 1, w)
                + (1 - alpha) * minor_chord(tonic_idx + 1, w)
            ),
            (
                beta * minor_chord(tonic_idx - 1, w)
                + (1 - beta) * major_chord(tonic_idx - 1, w)
            ),
        ]
    )

    return np.dot(ensure_norm(w), chords)


def cloud_diameter(cloud):
    """
    The Cloud Diameter measures the maximal tonal distance of the notes
    in a chord (or cloud of notes).

    Parameters
    ----------
    cloud : 3D array
        Array containing the coordinates in the spiral array
        of the notes in the cloud.


    Returns
    -------
    diameter : float
        Largest distance between any two notes in a cloud
    """
    return distance.pdist(cloud, metric="euclidean").max()


def center_of_effect(cloud, duration):
    """
    The center of effect condenses musical information
    in the spiral array by a single point.

    Parameters
    ----------
    cloud : 3D array
        Array containing the coordinates in the spiral array
        of the notes in the cloud.
    duration : array
        Array containing the duration of each note in the cloud


    Returns
    -------
    ce : array
       Coordinates of the center of effect
    """
    return (duration.reshape(-1, 1) * cloud).sum(0) / duration.sum()


class TonalTension(object):
    """Base class for TonalTension features"""

    def compute_tension(self, cloud, *args, **kwargs):
        raise NotImplementedError


class CloudDiameter(TonalTension):
    """
    Compute cloud diameter
    """

    def compute_tension(self, cloud, *args, **kwargs):
        scale_factor = kwargs.get("scale_factor", SCALE_FACTOR)
        if len(cloud) > 1:
            return cloud_diameter(cloud) * scale_factor
        else:
            return 0.0


class TensileStrain(TonalTension):
    """
    Compute tensile strain
    """

    def __init__(
        self, tonic_idx=0, mode="major", w=DEFAULT_WEIGHTS, alpha=ALPHA, beta=BETA
    ):
        self.update_key(tonic_idx, mode, w, alpha, beta)

    def compute_tension(self, cloud, *args, **kwargs):
        # duration required; take first positional arg
        duration = args[0] if len(args) > 0 else kwargs.get("duration", None)
        if duration is None:
            return 0
        scale_factor = kwargs.get("scale_factor", SCALE_FACTOR)
        if duration.sum() == 0:
            return 0

        cloud_ce = center_of_effect(cloud, duration)

        return e_distance(cloud_ce, self.key_ce) * scale_factor

    def update_key(self, tonic_idx, mode, w=DEFAULT_WEIGHTS, alpha=ALPHA, beta=BETA):
        if mode in ("major", None, 1):
            self.key_ce = major_key(tonic_idx, w=w)
        elif mode in ("minor", -1):
            self.key_ce = minor_key(tonic_idx, w=w, alpha=alpha, beta=beta)


class CloudMomentum(TonalTension):
    """
    Compute cloud momentum
    """

    def __init__(self):
        self.prev_ce = None

    def compute_tension(self, cloud, *args, **kwargs):
        duration = args[0] if len(args) > 0 else kwargs.get("duration", None)
        if duration is None:
            return 0
        reset = kwargs.get("reset", False)
        scale_factor = kwargs.get("scale_factor", SCALE_FACTOR)
        if duration.sum() == 0:
            return 0

        if reset:
            self.prev_ce = None
        cloud_ce = center_of_effect(cloud, duration)

        if self.prev_ce is not None:
            tension = e_distance(cloud_ce, self.prev_ce) * scale_factor

        else:
            tension = 0

        self.prev_ce = cloud_ce

        return tension


def notes_to_idx(note_array):
    """
    Index of the note names in the spiral array
    """
    note_idxs = np.array(
        [NOTES_BY_FIFTHS.index((n["step"], n["alter"])) for n in note_array],
        dtype=int,
    )
    return note_idxs


def prepare_note_array(note_info, tonality_vector=None, key_estimation="infer_if_necessary"):
    """Prepare note array with pitch spelling and key signature information.
    
    Parameters
    ----------
    note_info : Melody or structured array
        Note information
    tonality_vector : list, optional
        Pre-computed tonality vector (list of (key_name, correlation) tuples)
    key_estimation : str, optional
        Key estimation strategy: "always_read_from_file", "infer_if_necessary", or "always_infer"
        Default is "infer_if_necessary"
    
    Returns
    -------
    note_array : structured array
        Note array with pitch spelling and key signature fields
    """
   
    note_array = ensure_notearray(note_info)

    # Ensure pitch spelling fields
    pitch_spelling_fields = ("step", "alter", "octave")
    if len(set(pitch_spelling_fields).difference(note_array.dtype.names)) > 0:
        warnings.warn(
            "No pitch spelling information! Estimating pitch spelling...",
            stacklevel=2,
        )
        spelling = estimate_spelling(note_array)
        note_array = add_field(note_array, spelling.dtype)
        for field in spelling.dtype.names:
            note_array[field] = spelling[field]

    # Ensure key signature fields based on key_estimation strategy
    key_signature_fields = ("ks_fifths", "ks_mode")
    needs_key_signature = len(set(key_signature_fields).difference(note_array.dtype.names)) > 0

    fifths = None
    mode = None
    
    # Determine key signature based on strategy
    if key_estimation == "always_infer":
        # Always estimate from pitch content, ignore MIDI file
        pcs = np.mod(note_array["pitch"], 12).astype(int)
        if tonality_vector is None:
            tv = compute_tonality_vector(pcs)
        else:
            tv = tonality_vector
        fifths, mode = best_key_from_tonality_vector(tv)
    
    elif key_estimation == "always_read_from_file":
        
        if hasattr(note_info, 'id') and note_info.id:
            key_from_midi = extract_key_signature_from_midi(note_info.id)
            if key_from_midi is not None:
                fifths, mode = key_from_midi
            else:
                raise ValueError(f"No key signature found in MIDI file: {note_info.id}")
        else:
            raise ValueError("Cannot read key signature from file: no file path available")
    
    elif key_estimation == "infer_if_necessary":
        
        if hasattr(note_info, 'id') and note_info.id:
            key_from_midi = extract_key_signature_from_midi(note_info.id)
            if key_from_midi is not None:
                fifths, mode = key_from_midi
        
        if fifths is None:
            pcs = np.mod(note_array["pitch"], 12).astype(int)
            if tonality_vector is None:
                tv = compute_tonality_vector(pcs)
            else:
                tv = tonality_vector
            fifths, mode = best_key_from_tonality_vector(tv)
    
    else:
        raise ValueError(f"Invalid key_estimation value: {key_estimation}. Must be 'always_read_from_file', 'infer_if_necessary', or 'always_infer'")

    if needs_key_signature and fifths is not None:
        note_array = add_field(note_array, [("ks_fifths", "i4"), ("ks_mode", "i4")])
        note_array["ks_fifths"] = np.ones(len(note_array), dtype=int) * int(fifths)
        note_array["ks_mode"] = np.ones(len(note_array), dtype=int) * (1 if int(mode) >= 0 else -1)
    elif not needs_key_signature and fifths is not None:
        note_array["ks_fifths"][:] = int(fifths)
        note_array["ks_mode"][:] = int(mode)

    return note_array


def key_map_from_keysignature(notearray, onset_unit="auto"):
    """
    Helper method to get the key map from the key signature information
    in note arrays generated with `prepare_note_array`.

    Parameters
    ----------
    notearray : structured array
        Structured array with score information. Required fields are
        `ks_fifths`, `ks_mode` and `onset`.

    Returns
    -------
    km : function
        Function that maps onset time in beats to the key signature
        in the score.
    """
    if onset_unit == "auto":
        onset_unit, _ = get_time_units_from_note_array(notearray)

    onsets = notearray[onset_unit]

    unique_onsets = np.unique(onsets)
    unique_onset_idxs = [np.where(onsets == u)[0] for u in unique_onsets]

    kss = np.zeros((len(unique_onsets), 2), dtype=int)

    for i, uix in enumerate(unique_onset_idxs):
        # Deal with potential multiple key singatures in the same onset?
        ks = np.unique(
            np.column_stack((notearray["ks_fifths"][uix], notearray["ks_mode"][uix])),
            axis=0,
        )
        if len(ks) > 1:
            warnings.warn(
                "Multiple Key signtures detected at score position. "
                "Taking the first one."
            )
        kss[i] = ks[0]

    return interp1d(
        unique_onsets,
        kss,
        axis=0,
        kind="previous",
        bounds_error=False,
        fill_value="extrapolate",
    )


def estimate_tonaltension(
    note_info,
    ws=1.0,
    ss="onset",
    scale_factor=SCALE_FACTOR,
    w=DEFAULT_WEIGHTS,
    alpha=ALPHA,
    beta=BETA,
    tonality_vector=None,
    key_estimation="infer_if_necessary",
):
    """
    Compute tonal tension ribbons defined in [1]_

    Parameters
    ----------
    note_info : structured array, `Part` or `PerformedPart`
        Note information as a `Part` or `PerformedPart` instances or
        as a structured array. If it is a structured array, it has to
        contain the fields generated by the `note_array` properties
        of `Part` or `PerformedPart` objects. If the array contains
        onset and duration information of both score and performance,
        (e.g., containing both `onset_beat` and `onset_sec`), the score
        information will be preferred. Furthermore, this method requires
        pitch spelling and key signature information. If a structured note
        array is provided as input, this information can be optionally
        provided in fields `step`, `alter`, `ks_fifths` and `ks_mode`.
        If these fields are not found in the input structured array,
        they will be estimated using the key and pitch spelling estimation
        methods from  `partitura.musicanalysis.estimate_key` and
        and `partitura.musicanalysis.estimate_spelling`, respectively.
    ws : {int, float, np.array}, optional
        Window size for computing the tonal tension. If a number, it determines
        the size of the window centered at each specified score position (see
        `ss` below). If a numpy array, a 2D array of shape (`len(ss)`, 2)
        specifying the left and right distance from each score position in
        `ss`. Default is 1 beat.
    ss : {float, int, np.array, 'onset'}, optional.
        Step size or score position for computing the tonal tension features.
        If a number, this parameter determines the size of the step (in beats)
        starting from the first score position. If an array, it specifies the
        score positions at which the tonal tension is estimated. If 'onset',
        it computes the tension at each unique score position (i.e., all notes
        in a chord have the same score position). Default is 'onset'.
    scale_factor : float
        A multiplicative scaling factor.
    w : np.ndarray
        Weights for the chords
    alpha : float
        Alpha.
    beta : float
        Beta.
    key_estimation : str, optional
        Key estimation strategy: "always_read_from_file", "infer_if_necessary", or "always_infer"
        Default is "infer_if_necessary"

    Returns
    -------
    tonal_tension : dict
        Dictionary containing the tonal tension features: 
        keys are `onset`, `cloud_diameter`, `cloud_momentum`, `tensile_strain`
        and values are numpy arrays ordered by `onset`.

    References
    ----------
    .. [1] D. Herremans and E. Chew (2016) Tension ribbons: Quantifying and
           visualising tonal tension. Proceedings of the Second International
           Conference on Technologies for Music Notation and Representation
           (TENOR), Cambridge, UK.
    """

    note_array = prepare_note_array(note_info, tonality_vector=tonality_vector, key_estimation=key_estimation)

    onset_unit, duration_unit = get_time_units_from_note_array(note_array)

    # Open questions:
    # 1. rename score_onsets/offsets to reflect that other units are also
    # possible?
    # 2. In case the input is a performance, perhaps "cluster"/aggregate
    # the onsets into "chord" onsets or something similar?
    score_onset = note_array[onset_unit]
    score_offset = score_onset + note_array[duration_unit]

    # Determine the score position
    if isinstance(ss, (float, int)):
        unique_onsets = np.arange(
            score_onset.min(), score_offset.max() + (ss * 0.5), step=ss
        )
    elif isinstance(ss, np.ndarray):
        unique_onsets = ss
    elif ss == "onset":
        unique_onsets = np.unique(score_onset)
    else:
        raise ValueError("`ss` has to be a float, int, a numpy array or 'onset'")

    # Determine the window sizes for each score position
    if isinstance(ws, (float, int)):
        ws = np.ones((len(unique_onsets), 2)) * 0.5 * ws
    elif isinstance(ws, np.ndarray):
        if len(ws) != len(unique_onsets):
            raise ValueError("`ws` should have the same length as `unique_onsets`")
    else:
        raise ValueError("`ws` has to be a `float`, `int` or a numpy array")

    note_idxs = notes_to_idx(note_array)

    # Get coordinates of the notes in the piece in the spiral array space
    piece_coordinates = PITCH_COORDINATES[note_idxs]

    # Initialize classes for computing tonal tension
    cd = CloudDiameter()
    cm = CloudMomentum()

    # Get key of the piece from key signature information
    # Perhaps add an automatic method in the future (for
    # inferring modulations?)
    km = key_map_from_keysignature(note_array, onset_unit=onset_unit)
    fifths, mode = km(unique_onsets.min()).astype(int)
    ts = TensileStrain(tonic_idx=C_IDX + fifths, mode=mode, w=w, alpha=alpha, beta=beta)
    # Initialize array for holding the tonal tension
    n_windows = len(unique_onsets)

    # Initialize lists to collect values
    onset_list = np.zeros(n_windows, dtype=float)
    cloud_diameter_list = np.zeros(n_windows, dtype=float)
    cloud_momentum_list = np.zeros(n_windows, dtype=float)
    tensile_strain_list = np.zeros(n_windows, dtype=float)

    onset_list[:] = unique_onsets

    # Main loop for computing tension information
    for i, (o, (wlo, whi)) in enumerate(zip(unique_onsets, ws)):
        max_time = o + whi
        min_time = o - wlo

        ema = set(np.where(score_offset >= max_time)[0])
        sma = set(np.where(score_onset <= max_time)[0])
        smi = set(np.where(score_onset >= min_time)[0])
        emi = set(np.where(score_offset <= max_time)[0])

        active_idx = np.array(
            list(smi.intersection(emi).union(ema.intersection(sma))), dtype=int
        )
        active_idx.sort()

        cloud = piece_coordinates[active_idx]
        duration = np.minimum(max_time, score_offset[active_idx]) - np.maximum(
            min_time, score_onset[active_idx]
        )

        # Update key information
        if not np.all([fifths, mode] == km(o)):
            fifths, mode = km(o).astype(int)
            ts.update_key(tonic_idx=C_IDX + fifths, mode=mode)

        cloud_diameter_list[i] = cd.compute_tension(cloud, scale_factor=scale_factor)
        cloud_momentum_list[i] = cm.compute_tension(cloud, duration, scale_factor=scale_factor)
        tensile_strain_list[i] = ts.compute_tension(cloud, duration, scale_factor=scale_factor)
    
    return {
        onset_unit: onset_list.tolist(),
        "cloud_diameter": cloud_diameter_list.tolist(),
        "cloud_momentum": cloud_momentum_list.tolist(),
        "tensile_strain": tensile_strain_list.tolist(),
    }
