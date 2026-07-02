import math

from melody_features.features import (
    direction_of_melodic_motion,
    interval_direction_mean,
    interval_direction_std,
    interval_entropy,
    ivdirdist1,
    mean_absolute_interval,
    mean_melodic_interval,
    melodic_octaves,
    minor_major_third_ratio,
    modal_interval,
    most_common_interval,
    number_of_common_pitch_classes,
    number_of_common_pitches_classes,
    pitch_standard_deviation,
    pitch_variability,
)


def test_melodic_octaves_returns_float_proportion():
    pitches = [60, 72, 74]
    value = melodic_octaves(pitches)
    assert isinstance(value, float)
    assert value == 0.5


def test_ivdirdist1_reports_directional_bias():
    # Intervals: +2, -2, +2 => bias for size 2 is (2/3 - 1/3) / 1 = 1/3
    pitches = [60, 62, 60, 62]
    starts = [0.0, 1.0, 2.0, 3.0]
    ends = [1.0, 2.0, 3.0, 4.0]
    dist = ivdirdist1(pitches, starts, ends)
    assert math.isclose(dist[2], 1.0 / 3.0, rel_tol=1e-9)
    assert dist.get(1, 0.0) == 0.0


def test_interval_entropy_uses_signed_intervals():
    # Signed intervals are [2, -2, 2] => entropy > 0.
    # Absolute intervals would be [2, 2, 2] => entropy == 0.
    pitches = [60, 62, 60, 62]
    value = interval_entropy(pitches)
    assert math.isclose(value, 0.9182958340544896, rel_tol=1e-9)


def test_interval_aliases_share_implementation():
    assert mean_melodic_interval is mean_absolute_interval
    assert most_common_interval is modal_interval
    assert pitch_variability is pitch_standard_deviation
    assert number_of_common_pitches_classes is number_of_common_pitch_classes


def test_minor_major_third_ratio_uses_jsymbolic_zero_sentinel():
    # Contains minor thirds (+3, -3) and no major thirds.
    pitches = [60, 63, 60]
    assert math.isnan(minor_major_third_ratio(pitches))


def test_direction_metrics_use_different_denominators():
    # Intervals: [0, +2, 0, -2]
    pitches = [60, 60, 62, 62, 60]
    assert direction_of_melodic_motion(pitches) == 0.5
    assert interval_direction_mean(pitches) == 0.0


def test_interval_direction_std_is_population_std():
    # Directions: [1, -1] => population std = 1.0 (sample std would be sqrt(2)).
    pitches = [60, 62, 60]
    value = interval_direction_std(pitches)
    assert math.isclose(value, 1.0, rel_tol=1e-9)
