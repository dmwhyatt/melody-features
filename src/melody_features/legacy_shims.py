"""Register legacy top-level import paths without shadowing root feature names."""

from __future__ import annotations

import importlib
import sys
from typing import Dict

from .utils.warnings import warn_legacy_import_shim

# Old module path -> canonical replacement (removed in 2.0.0).
LEGACY_SHIM_TARGETS: Dict[str, str] = {
    "melody_features.import_mid": "melody_features.io.midi",
    "melody_features.representations": "melody_features.core.representations",
    "melody_features.step_contour": "melody_features.contour.step_contour",
    "melody_features.huron_contour": "melody_features.contour.huron_contour",
    "melody_features.interpolation_contour": "melody_features.contour.interpolation_contour",
    "melody_features.polynomial_contour": "melody_features.contour.polynomial_contour",
    "melody_features.meter_estimation": "melody_features.algorithms.meter_estimation",
    "melody_features.narmour": "melody_features.algorithms.narmour",
    "melody_features.pitch_spelling": "melody_features.algorithms.pitch_spelling",
    "melody_features.tonal_tension": "melody_features.algorithms.tonal_tension",
    "melody_features.distributional": "melody_features.utils.distributional",
}

# Root feature names that must not be shadowed by a compat submodule file.
FEATURE_NAMES_PROTECTED_FROM_SHIM_FILES = frozenset(
    {"pitch_spelling", "tonal_tension"}
)


def register_legacy_shim(old_module: str, new_module: str) -> None:
    """Point `old_module` at `new_module` in :data:`sys.modules`."""
    warn_legacy_import_shim(old_module, new_module)
    sys.modules[old_module] = importlib.import_module(new_module)


def install_legacy_shims() -> None:
    """Install all legacy import aliases (idempotent)."""
    for old_module, new_module in LEGACY_SHIM_TARGETS.items():
        if old_module not in sys.modules:
            register_legacy_shim(old_module, new_module)
