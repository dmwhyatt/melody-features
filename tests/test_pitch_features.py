import math

from melody_features.features import (
    folded_fifths_pitch_class_histogram,
    mean_pitch,
    mean_pitch_class,
    melodic_pitch_variety,
)


def test_mean_pitch_empty_returns_zero():
    assert mean_pitch([]) == 0.0


def test_mean_pitch_class_is_linear_not_circular():
    # Linear mean over [11, 0] is 5.5 (not a circular centroid near 11.5/ -0.5).
    assert mean_pitch_class([11, 0]) == 5.5


def test_folded_fifths_histogram_uses_pc_keys_in_fifths_order():
    histogram = folded_fifths_pitch_class_histogram([60, 67, 62, 67, 60, 60])  # PCs: 0,7,2
    assert list(histogram.keys()) == [0, 7, 2]
    assert histogram[0] == 3.0
    assert histogram[7] == 2.0
    assert histogram[2] == 1.0


def test_melodic_pitch_variety_counts_onset_positions():
    # Repeated 60 is found one later onset-position (tick) after the first 60.
    pitches = [60, 62, 60]
    starts = [0.0, 0.0, 1.0]
    value = melodic_pitch_variety(pitches=pitches, starts=starts, tempo=120.0, ppqn=480)
    assert math.isclose(value, 1.0)
