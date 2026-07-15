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
    """Suppress noisy warnings from pretty_midi, deprecated pkg_resources, and
    benign NumPy floating-point warnings triggered by expected degenerate
    inputs (e.g. single-note melodies, zero-variance intervals).

    Many features take a sample standard deviation (`ddof=1`) or similar
    statistic over very short sequences (e.g. a melody with 0 or 1 notes),
    which is a legitimate, already-handled edge case but makes NumPy emit
    RuntimeWarnings such as "Degrees of freedom <= 0 for slice" or "invalid
    value encountered in divide". These are silenced here (scoped to
    NumPy's own internal modules) so end users of this package don't see
    them by default; the resulting `nan`/`0.0` feature values are handled
    explicitly wherever they occur.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="pretty_midi")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="pretty_midi")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*pkg_resources is deprecated.*",
    )
    for message in (
        r".*invalid value encountered.*",
        r".*divide by zero encountered.*",
        r"Degrees of freedom <= 0 for slice.*",
        r"Mean of empty slice.*",
    ):
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=message,
            module=r"numpy(\..*)?",
        )
