"""Validation helpers shared by pipeline modules."""

from ..core.representations import Melody


def _check_is_monophonic(melody: Melody) -> bool:
    """Return whether a melody has no overlapping notes."""
    starts = melody.starts
    ends = melody.ends

    if len(starts) < 2:
        return True

    for i in range(1, len(starts)):
        if starts[i] < ends[i - 1]:
            return False

    return True
