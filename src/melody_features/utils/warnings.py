"""Central warning filters for MIDI import and legacy dependencies."""

from __future__ import annotations

import warnings

_LEGACY_SHIM_WARNED: set[str] = set()


def reset_legacy_shim_warnings_for_tests() -> None:
    """Clear one-time shim warning state (tests only)."""
    _LEGACY_SHIM_WARNED.clear()


def warn_legacy_import_shim(
    old_module: str,
    new_module: str,
    *,
    remove_in: str = "2.0.0",
) -> None:
    """Emit a one-time deprecation warning for a top-level compatibility shim."""
    if old_module in _LEGACY_SHIM_WARNED:
        return
    _LEGACY_SHIM_WARNED.add(old_module)
    warnings.warn(
        (
            f"{old_module} is deprecated; use {new_module} instead. "
            f"This shim will be removed in melody-features {remove_in}."
        ),
        DeprecationWarning,
        stacklevel=2,
    )


def suppress_common_melody_warnings() -> None:
    """Suppress noisy warnings from pretty_midi and deprecated pkg_resources."""
    warnings.filterwarnings("ignore", category=UserWarning, module="pretty_midi")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="pretty_midi")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*pkg_resources is deprecated.*",
    )
