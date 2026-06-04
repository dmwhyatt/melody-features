"""Metre feature definitions."""

import numpy as np

from ..feature_decorators import jsymbolic, metre, midi_toolbox, novel, rhythm, simile
from ..algorithms.meter_estimation import (
    duration_accent as _duration_accent,
    melodic_accent as _melodic_accent,
    metric_hierarchy as _metric_hierarchy,
)
from ..core.representations import Melody


__all__ = [
    "metric_hierarchy",
    "meter_accent",
    "meter_numerator",
    "meter_denominator",
    "proportion_of_time_in_first_meter",
    "number_of_unique_time_signatures",
    "syncopation",
    "syncopicity",
]


@rhythm
@metre
@midi_toolbox
def metric_hierarchy(melody: Melody) -> list[int]:
    """Metric hierarchy values for each note, indicating the strength of each note
    position within the known or estimated meter. Higher values indicate stronger
    metric positions (e.g., downbeat = 5, beat = 4, half-beat = 3, etc.).

    Implementation based on MIDI toolbox metrichierarchy.m.
    """
    return _metric_hierarchy(
        melody.starts,
        melody.ends,
        time_signature=melody.meter,
        tempo=melody.tempo,
        pitches=melody.pitches,
    )

def _meter_accent_mean(melody: Melody) -> float:
    """``meteraccent.m`` synchrony (float, unrounded)."""
    hierarchy_values = metric_hierarchy(melody)
    if not hierarchy_values:
        return 0.0
    melodic_accents = _melodic_accent(melody.pitches)
    durational_accents = _duration_accent(melody.starts, melody.ends)
    n = min(len(hierarchy_values), len(melodic_accents), len(durational_accents))
    if n == 0:
        return 0.0
    products = [
        h * m * d
        for h, m, d in zip(
            hierarchy_values[:n], melodic_accents[:n], durational_accents[:n]
        )
    ]
    return float(-1.0 * np.mean(products))

@rhythm
@metre
@midi_toolbox
def meter_accent(melody: Melody) -> int:
    """Phenomenal accent synchrony measure, calculated as the negative mean of
    the product of metric hierarchy, melodic accent, and durational accent
    for each note. Higher values indicate stronger accent synchrony.

    Implementation based on MIDI toolbox meteraccent.m.
    """
    return int(round(_meter_accent_mean(melody)))

@jsymbolic
@rhythm
@metre
def meter_numerator(melody: Melody) -> int:
    """The numerator of the melody's active time signature.

    For a time signature written as ``numerator/denominator``, the numerator is
    the number of beats in each notated bar. If a melody contains meter changes,
    this returns the meter stored on the melody object as its primary meter.

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
    """The denominator of the melody's active time signature.

    For a time signature written as ``numerator/denominator``, the denominator
    gives the note value that represents one beat: for example, ``4`` means a
    quarter-note beat and ``8`` means an eighth-note beat.

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
    """The proportion of the melody's duration spent in its first time signature.

    The numerator and denominator of the first encountered time signature define
    the initial meter. This feature reports the fraction of total melody duration
    before any subsequent meter change. Melodies with no meter change therefore
    return ``1.0``.

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
    """
    The number of unique time signatures in the melody.
    
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
    This feature is named `Metrical Diversity` in jSymbolic.
    """
    return len({(numerator, denominator) for _, numerator, denominator in melody.time_signatures})

@novel
@rhythm
@metre
def syncopation(melody: Melody) -> float:
    """
    Calculate the mean `syncopation` value based on the Longuet-Higgins and Lee (1984) model.
    This `syncopation` model assigns metrical weights to each
    note position based on its position in the metric hierarchy. Syncopation occurs when
    a rest or tied note is preceded by a sounded note of lower metrical weight. The 
    `syncopation` value is the difference between the rest weight and the preceding note weight.
    
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

    hierarchy_values = _metric_hierarchy(
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
    """
    Calculates the sum `syncopicity` of a melody across metric levels.
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
        Sum of per-level syncopation proportions across tested levels (half-bar,
        beat, and first subdivision). Each level contributes
        ``syncopation_count / number_of_notes``, so this is not a raw event count.
    
    Note
    ----
    Grid durations are derived from the melody's initial meter and tempo and
    therefore assume constant meter/tempo over the analyzed passage.
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
