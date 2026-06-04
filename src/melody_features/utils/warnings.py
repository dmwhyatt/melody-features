"""Central warning filters for MIDI import and legacy dependencies."""

from __future__ import annotations

import warnings


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
