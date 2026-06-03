import math

from melody_features.features import (
    get_interpolation_contour_features,
    get_polynomial_contour_features,
    get_step_contour_features,
)
from melody_features.contour import InterpolationContour, StepContour
from melody_features.core.representations import Melody


def test_interpolation_contour_defaults_to_amads_method():
    pitches = [60, 62, 64, 62, 60]
    starts = [0.0, 1.0, 2.0, 3.0, 4.0]

    default_features = get_interpolation_contour_features(pitches, starts)
    amads = InterpolationContour(pitches, starts, method="amads")

    assert default_features == (
        amads.global_direction,
        amads.mean_gradient,
        amads.gradient_std,
        amads.direction_changes,
        amads.class_label,
    )


def test_step_contour_global_variation_defaults_to_amads_population_std():
    contour = StepContour([60, 64], [1.0, 1.0], step_contour_length=4)
    assert contour.contour == [60, 60, 64, 64]

    expected_population_std = math.sqrt(4.0)
    assert contour.global_variation == expected_population_std


def test_step_contour_global_variation_fantastic_uses_sample_std():
    contour = StepContour(
        [60, 64], [1.0, 1.0], step_contour_length=4, method="fantastic"
    )
    expected_sample_std = math.sqrt(16.0 / 3.0)
    assert contour.global_variation == expected_sample_std


def test_get_step_contour_features_returns_three_values():
    result = get_step_contour_features(
        pitches=[60, 62, 64],
        starts=[0.0, 1.0, 2.0],
        ends=[1.0, 2.0, 3.0],
        tempo=120.0,
    )

    assert isinstance(result, tuple)
    assert len(result) == 3


def _build_melody(starts: list[float], pitches: list[int]) -> Melody:
    notes = []
    for s, p in zip(starts, pitches):
        e = s + 1.0
        notes.append(f"Note(start={s}, end={e}, pitch={p}, velocity=100)")
    return Melody({"MIDI Sequence": "".join(notes), "tempo": 120.0})


def test_polynomial_contour_coefficients_match_fantastic_reference_examples():
    # Reference values generated from FANTASTIC poly.contour in
    # Feature_Value_Summary_Statistics.R for the same onset/pitch sequences.
    arch = get_polynomial_contour_features(
        _build_melody(
            starts=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0],
            pitches=[62, 64, 65, 67, 64, 60, 62],
        )
    )
    assert math.isclose(arch[0], -1.50148256328554, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(arch[1], -0.266153301469287, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(arch[2], 0.122057001239158, rel_tol=1e-6, abs_tol=1e-6)

    ascending = get_polynomial_contour_features(
        _build_melody(
            starts=[0.0, 1.0, 2.0, 3.0, 4.0],
            pitches=[60, 62, 64, 65, 67],
        )
    )
    assert math.isclose(ascending[0], 1.7, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(ascending[1], 0.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(ascending[2], 0.0, rel_tol=1e-6, abs_tol=1e-6)
