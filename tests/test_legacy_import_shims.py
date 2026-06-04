"""Legacy top-level import paths kept for backward compatibility."""

import importlib
import sys
import warnings

import pytest

from melody_features.legacy_shims import LEGACY_SHIM_TARGETS, register_legacy_shim
from melody_features.utils.warnings import reset_legacy_shim_warnings_for_tests


@pytest.mark.parametrize(
    "module_name,attr,new_module",
    [
        ("melody_features.representations", "Melody", "melody_features.core.representations"),
        ("melody_features.step_contour", "StepContour", "melody_features.contour.step_contour"),
        ("melody_features.huron_contour", "HuronContour", "melody_features.contour.huron_contour"),
        (
            "melody_features.interpolation_contour",
            "InterpolationContour",
            "melody_features.contour.interpolation_contour",
        ),
        (
            "melody_features.polynomial_contour",
            "PolynomialContour",
            "melody_features.contour.polynomial_contour",
        ),
        (
            "melody_features.meter_estimation",
            "estimate_meter",
            "melody_features.algorithms.meter_estimation",
        ),
        ("melody_features.narmour", "proximity", "melody_features.algorithms.narmour"),
        (
            "melody_features.pitch_spelling",
            "estimate_spelling",
            "melody_features.algorithms.pitch_spelling",
        ),
        (
            "melody_features.tonal_tension",
            "estimate_tonaltension",
            "melody_features.algorithms.tonal_tension",
        ),
        (
            "melody_features.distributional",
            "histogram_bins",
            "melody_features.utils.distributional",
        ),
        ("melody_features.import_mid", "import_midi", "melody_features.io.midi"),
    ],
)
def test_legacy_shim_exports_and_warns(module_name: str, attr: str, new_module: str):
    sys.modules.pop(module_name, None)
    reset_legacy_shim_warnings_for_tests()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=DeprecationWarning)
        register_legacy_shim(module_name, new_module)
        module = importlib.import_module(module_name)
    assert any(module_name in str(record.message) for record in caught)
    assert getattr(module, attr) is not None
    # Do not reload: shims alias shared modules (e.g. core.representations); reload
    # would replace Melody and break isinstance checks in later tests.
    reset_legacy_shim_warnings_for_tests()
    with warnings.catch_warnings(record=True) as caught_again:
        warnings.simplefilter("always", category=DeprecationWarning)
        importlib.import_module(module_name)
    assert not caught_again


def test_all_legacy_targets_registered_on_package_import():
    import melody_features  # noqa: F401

    for old_module in LEGACY_SHIM_TARGETS:
        assert old_module in sys.modules


def test_legacy_shims_do_not_shadow_root_feature_names():
    import melody_features
    import inspect

    assert inspect.isfunction(melody_features.pitch_spelling)
    assert inspect.isfunction(melody_features.tonal_tension)
    assert melody_features.pitch_spelling is not sys.modules["melody_features.pitch_spelling"]


def test_legacy_v107_feature_import_bundle():
    """Typical v1.0.7 import style still resolves after shims load."""
    import melody_features  # noqa: F401

    from melody_features.representations import Melody
    from melody_features.step_contour import StepContour
    from melody_features.distributional import histogram_bins, kurtosis
    from melody_features.meter_estimation import estimate_meter, meter_to_time_signature
    from melody_features.narmour import closure, proximity

    assert Melody is not None
    assert StepContour is not None
    assert callable(histogram_bins)
    assert callable(kurtosis)
    assert callable(estimate_meter)
    assert callable(meter_to_time_signature)
    assert callable(closure)
    assert callable(proximity)
