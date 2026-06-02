from types import SimpleNamespace

from melody_features.features import (
    average_time_between_attacks,
    complete_rests_fraction,
    dotted_duration_transitions,
    duration_in_seconds,
    equal_duration_transitions,
    global_duration,
    half_duration_transitions,
    ioi_contour,
    ioi_mean,
    ioi_standard_deviation,
    length,
    note_density_per_quarter_note,
    prevalence_of_long_rhythmic_values,
    prevalence_of_medium_rhythmic_values,
    prevalence_of_short_rhythmic_values,
    total_number_of_notes,
    variability_of_time_between_attacks,
)


def test_note_density_per_quarter_note_handles_single_note():
    melody = SimpleNamespace(starts=[0.0], tempo=120.0, total_duration=0.5)
    assert note_density_per_quarter_note(melody) == 1.0


def test_aliases_point_to_same_implementations():
    assert length is total_number_of_notes
    assert global_duration is duration_in_seconds
    assert ioi_mean is average_time_between_attacks
    assert ioi_standard_deviation is variability_of_time_between_attacks


def test_ioi_contour_requires_three_onsets():
    assert ioi_contour([0.0, 0.5]) == []
    assert ioi_contour([0.0, 0.5, 1.0]) == [0]


def test_rhythmic_prevalence_bins_overlap():
    # Uniform half-note values occupy the shared medium/long boundary bin.
    starts = [0.0, 1.0, 2.0]
    ends = [1.0, 2.0, 3.0]
    total = (
        prevalence_of_short_rhythmic_values(starts, ends)
        + prevalence_of_medium_rhythmic_values(starts, ends)
        + prevalence_of_long_rhythmic_values(starts, ends)
    )
    assert total > 1.0


def test_complete_rests_fraction_uses_all_silent_runs():
    starts = [0.00, 0.30, 0.60]
    ends = [0.05, 0.35, 0.65]
    # Includes both short and long complete silent runs.
    assert complete_rests_fraction(starts, ends, tempo=120.0) > 0.5


def test_duration_transition_features_ignore_tempo_parameter():
    starts = [0.0, 0.5, 1.0, 1.5]
    ends = [0.25, 0.75, 1.25, 1.75]
    assert equal_duration_transitions(starts, ends, tempo=60.0) == equal_duration_transitions(
        starts, ends, tempo=180.0
    )
    assert half_duration_transitions(starts, ends, tempo=60.0) == half_duration_transitions(
        starts, ends, tempo=180.0
    )
    assert dotted_duration_transitions(starts, ends, tempo=60.0) == dotted_duration_transitions(
        starts, ends, tempo=180.0
    )
