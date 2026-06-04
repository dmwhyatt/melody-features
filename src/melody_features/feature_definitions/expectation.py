"""Expectation feature definitions."""

from typing import Dict

import numpy as np

from ..algorithms import compute_tonality_vector
from ..feature_decorators import expectation, idyom, jsymbolic, midi_toolbox, pitch, rhythm
from ..algorithms.meter_estimation import melodic_accent as _melodic_accent
from ..algorithms.narmour import (
    closure,
    intervallic_difference,
    proximity,
    registral_direction,
    registral_return,
)
from ..core.representations import Melody
from ..feature_utils import mean_and_std


__all__ = [
    "narmour_registral_direction",
    "narmour_proximity",
    "narmour_closure",
    "narmour_registral_return",
    "narmour_intervallic_difference",
    "melodic_embellishment",
    "mobility",
    "mean_mobility",
    "mobility_std",
    "melodic_attraction",
    "mean_melodic_attraction",
    "melodic_attraction_std",
    "melodic_accent",
    "mean_melodic_accent",
    "melodic_accent_std",
    "compltrans",
    "pitch_stm_mean_information_content",
    "pitch_ltm_mean_information_content",
    "rhythm_stm_mean_information_content",
    "rhythm_ltm_mean_information_content",
    "get_narmour_features",
]


def _idyom_mean_information_content(melody: Melody, config_key: str) -> float:
    """Delegate to the facade IDyOM infrastructure without an import-time cycle."""
    from melody_features import features as features_module

    return features_module._idyom_mean_information_content(melody, config_key)


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
@midi_toolbox
@pitch
@expectation
def narmour_registral_direction(pitches: list[int]) -> int:
    """Narmour registral-direction score for the final three notes.

    The last three pitches define an implicative interval followed by a realized
    interval. This feature returns ``1`` when a large implicative interval (greater
    than a tritone) is followed by a change of direction, or when a small
    implicative interval (smaller than a tritone) continues in the same direction.
    It returns ``0`` otherwise.

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
    """Narmour proximity score for the final melodic interval.

    Proximity rewards small realized intervals. It is calculated as ``6 - d``,
    where ``d`` is the absolute semitone distance between the final two notes, and
    is clipped at ``0`` for intervals of a tritone or larger. Unisons therefore
    receive ``6``, whole tones receive ``4``, perfect fourths receive ``1``, and
    perfect fifths receive ``0``.

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
    """Narmour closure score for the final three-note pattern.

    The last three pitches define two successive intervals. One point is awarded
    when the second interval changes direction relative to the first. A second
    point is awarded when the second interval is at least two semitones smaller
    than the first in absolute size. The resulting score ranges from ``0`` to ``2``.

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
    """Narmour registral-return score for the final three-note pattern.

    Registral return measures whether the last three notes move away from a pitch
    and then return toward it. The contour must change direction and neither
    interval may be a repeated note. An exact return to the first pitch scores
    ``3``; returning within one semitone scores ``2``; returning within two
    semitones scores ``1``; all other patterns score ``0``.

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
    """Narmour intervallic-difference score for the final three notes.

    The last three pitches define an implicative interval followed by a realized
    interval. If the implicative interval is large (greater than a tritone), this
    feature returns ``1`` when the realized interval is sufficiently smaller: at
    least three semitones smaller in the same direction, or at least two semitones
    smaller after a direction change. If the implicative interval is small (smaller
    than a tritone), it returns ``1`` when the realized interval is similar in size,
    within the same margins. Otherwise it returns ``0``.

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

@jsymbolic
@pitch
@expectation
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

@midi_toolbox
@pitch
@expectation
def mobility(pitches: list[int]) -> list[float]:
    """
    The melodic `mobility` for each note based on von Hippel (2000).
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
        One mobility value per input pitch (length matches ``pitches``).
    
    Citation
    --------
    von Hippel (2000)
    """
    if not pitches:
        return []
    if len(pitches) == 1:
        return [0.0]

    p = np.asarray(pitches, dtype=float)
    n = len(p)

    # 1-based MATLAB arrays; index 0 unused
    p_hist = np.zeros(n + 1)
    p2 = np.zeros(n + 1)
    mob = np.zeros(n + 1)
    y = np.zeros(n - 1)

    for i in range(2, n + 1):
        m_im1 = float(np.mean(p[: i - 1]))
        p_hist[i - 1] = p[i - 2] - m_im1
        p2[i] = p[i - 2] - m_im1

        z = np.concatenate([p_hist[1:i], [p_hist[i - 1]]])
        len_z = len(z)
        p2_row = p2[1 : len_z + 1]
        p3 = np.column_stack([p2_row, z])

        if p3.shape[0] >= 2 and np.std(p3[:, 0]) > 0 and np.std(p3[:, 1]) > 0:
            corr = np.corrcoef(p3, rowvar=False)
            mob[i] = float(corr[0, 1]) if not np.isnan(corr[0, 1]) else 0.0
        else:
            mob[i] = 0.0

        y[i - 2] = mob[i - 1] * (p[i - 1] - m_im1)

    if len(y) >= 2:
        y[1] = 0.0
    elif len(y) == 1:
        y = np.array([y[0], 0.0])

    y = np.abs(np.concatenate([[0.0], y]))
    if len(y) > n:
        y = y[:n]
    return [float(v) for v in y]

@midi_toolbox
@pitch
@expectation
def mean_mobility(pitches: list[int]) -> float:
    """
    The arithmetic mean of the `mobility` values across all notes.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Mean mobility value
    """
    mean, _ = mean_and_std(mobility(pitches))
    return mean

@midi_toolbox
@pitch
@expectation
def mobility_std(pitches: list[int]) -> float:
    """
    The standard deviation of the `mobility` values across all notes.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Standard deviation of mobility values
    """
    _, std = mean_and_std(mobility(pitches))
    return std

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
    """Melodic attraction values following Lerdahl's tonal-attraction model.

    The melody's key is estimated from its pitch-class content, then each pitch
    class is assigned a tonal anchoring weight in that key. For each adjacent
    pitch pair, attraction depends on tonal stability, pitch proximity, and whether
    the melodic motion continues or changes direction. Values are scaled to the
    interval ``[0, 1]``.
    
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

    Notes
    -----
    Uses anchoring weights, directed motion, and scaled output. We
    currently validate it via structural and range tests rather than a full
    one-to-one parity fixture set from MIDI Toolbox example corpora.
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
        elif directions[i-1] == 0:  # After a repetition, treat as continuation onset
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

@midi_toolbox
@pitch
@expectation
def mean_melodic_attraction(pitches: list[int]) -> float:
    """The arithmetic mean of Lerdahl melodic-attraction values.

    Melodic attraction estimates how strongly each note is drawn toward the next
    note, based on tonal anchoring weights, pitch proximity, and directed motion.
    This feature averages those per-note attraction values across the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Mean melodic attraction value

    Citation
    --------
    Lerdahl (1996)
    """
    mean, _ = mean_and_std(melodic_attraction(pitches))
    return mean

@midi_toolbox
@pitch
@expectation
def melodic_attraction_std(pitches: list[int]) -> float:
    """The sample standard deviation of Lerdahl melodic-attraction values.

    Melodic attraction estimates note-to-note tonal pull from anchoring weights,
    pitch proximity, and directed motion. This feature summarizes how much those
    attraction values vary across the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Standard deviation of melodic attraction values

    Citation
    --------
    Lerdahl (1996)
    """
    _, std = mean_and_std(melodic_attraction(pitches))
    return std

@midi_toolbox
@pitch
@expectation
def melodic_accent(pitches: list[int]) -> list[float]:
    """Melodic accent salience for each note using Thomassen's contour model.

    Thomassen's model assigns accent strength from the melodic contour formed by
    three-note pitch windows. Notes at locally salient contour positions receive
    higher values. The implementation follows the MIDI Toolbox ``melaccent.m``
    convention and returns values from ``0`` (no salience) to ``1`` (maximum
    salience).
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    list[float]
        List of melodic accent values for each note

    Citation
    --------
    Thomassen (1982)
    """
    return _melodic_accent(pitches)

@midi_toolbox
@pitch
@expectation
def mean_melodic_accent(pitches: list[int]) -> float:
    """The arithmetic mean of Thomassen melodic-accent values.

    Melodic accent values estimate local contour salience from three-note pitch
    windows. This feature averages those note-level salience values across the
    melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Mean melodic accent value

    Citation
    --------
    Thomassen (1982)
    """
    mean, _ = mean_and_std(melodic_accent(pitches))
    return mean

@midi_toolbox
@pitch
@expectation
def melodic_accent_std(pitches: list[int]) -> float:
    """The sample standard deviation of Thomassen melodic-accent values.

    Melodic accent values estimate local contour salience from three-note pitch
    windows. This feature summarizes how unevenly that salience is distributed
    across the melody.
    
    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
        
    Returns
    -------
    float
        Standard deviation of melodic accent values

    Citation
    --------
    Thomassen (1982)
    """
    _, std = mean_and_std(melodic_accent(pitches))
    return std

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

@idyom
@expectation
@pitch
def pitch_stm_mean_information_content(melody: Melody) -> float:
    """The average information content across all notes in a melody,
    calculated using IDyOM's prediction-by-partial-matching (PPM) algorithm.
    Information content is perceptually related to surprise, and can be calculated
    for pitches or rhythms.

    Citation
    --------
    Pearce, M. (2005)
    """
    return _idyom_mean_information_content(melody, "pitch_stm")

@idyom
@expectation
@pitch
def pitch_ltm_mean_information_content(melody: Melody) -> float:
    """The average information content across all notes in a melody,
    calculated using IDyOM's long-term model (LTM). Information content is
    perceptually related to surprise, and can be calculated for pitches or rhythms.

    Citation
    --------
    Pearce, M. (2005)
    """
    return _idyom_mean_information_content(melody, "pitch_ltm")

@idyom
@expectation
@rhythm
def rhythm_stm_mean_information_content(melody: Melody) -> float:
    """The average rhythmic information content across all notes in a melody,
    calculated using IDyOM's short-term model (STM). Information content is
    perceptually related to surprise, and can be calculated for pitches or rhythms.

    Citation
    --------
    Pearce, M. (2005)
    """
    return _idyom_mean_information_content(melody, "rhythm_stm")

@idyom
@expectation
@rhythm
def rhythm_ltm_mean_information_content(melody: Melody) -> float:
    """The average rhythmic information content across all notes in a melody,
    calculated using IDyOM's long-term model (LTM). Information content is
    perceptually related to surprise, and can be calculated for pitches or rhythms.

    Citation
    --------
    Pearce, M. (2005)
    """
    return _idyom_mean_information_content(melody, "rhythm_ltm")
