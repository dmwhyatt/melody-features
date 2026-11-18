"""Core algorithms for the MUST feature set (Clemente et al., 2020)."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..feature_utils import _get_durations
from ..melody_tokenizer import MustTokenizer

if TYPE_CHECKING:
    from ..core.representations import Melody

_MUST_TOKENIZER = MustTokenizer()


def _zero_for_empty_melody(melody: Melody) -> Optional[float]:
    """Return 0.0 for empty melodies, matching package convention."""
    if len(melody.pitches) == 0:
        return 0.0
    return None


def must_shannon_entropy(distribution: np.ndarray) -> float:
    """Shannon entropy using natural log, matching MUST `shentropy.m`."""
    weights = np.asarray(distribution, dtype=float).ravel()
    total = weights.sum()
    if total == 0.0:
        return 0.0
    probs = weights / total
    return float(-np.sum(probs * np.log(probs)))


def _order_sign(a: int, b: int, c: int) -> int:
    """Order signature for a 3-note pitch sequence (MUST `ordersign.m`)."""
    if a < b:
        if b < c:
            return 1
        if b == c:
            return 2
        if b > c:
            if a < c:
                return 3
            if a == c:
                return 4
            return 5
    elif a == b:
        if b < c:
            return 6
        if b == c:
            return 7
        return 8
    elif a > b:
        if a < c:
            return 9
        if a == c:
            return 10
        if b < c:
            return 11
        if b == c:
            return 12
        return 13


def _duration_accent(durations: np.ndarray, tau: float = 0.5, accent_index: float = 2.0) -> np.ndarray:
    """Parncutt (1994) duration accent (MIDI Toolbox `duraccent.m`)."""
    durations = np.asarray(durations, dtype=float)
    return (1.0 - np.exp(-durations / tau)) ** accent_index


def _pitches(melody: Melody) -> np.ndarray:
    return np.asarray(melody.pitches, dtype=float)


def _onsets_sec(melody: Melody) -> np.ndarray:
    return np.asarray(melody.starts, dtype=float)


def _durations_sec(melody: Melody) -> np.ndarray:
    return np.asarray(melody.ends, dtype=float) - np.asarray(melody.starts, dtype=float)


def _onsets_beats(melody: Melody) -> np.ndarray:
    onsets_sec = _onsets_sec(melody)
    if onsets_sec.size == 0:
        return onsets_sec
    return (onsets_sec - onsets_sec[0]) * melody.tempo / 60.0


def _durations_beats(melody: Melody) -> np.ndarray:
    return np.asarray(_get_durations(melody.starts, melody.ends, melody.tempo), dtype=float)


def _local_unbalance(
    melody: Melody,
    *,
    notes_per_window: int = 2,
    step_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Local unbalance curve (MUST `localunbalance.m` with event-based windows)."""
    onsets = _onsets_beats(melody)
    durations = _durations_beats(melody)
    note_count = len(onsets)
    if note_count == 0:
        return np.array([1.0]), np.array([0.0])
    total_time = onsets[-1] + durations[-1] - onsets[0]
    if note_count <= 1 or total_time <= 0:
        return np.array([1.0]), np.array([0.0])

    expected_duration = total_time / (note_count - 1)
    local_expected_density = (note_count - 1) / total_time
    window_length = expected_duration * notes_per_window
    window_step = step_fraction * window_length

    # Same window-start sequence as before, built by repeated addition (not
    # `np.arange`) so the floating-point values match exactly at the
    # boundary. This part is cheap -- it's the per-window note count below
    # that used to dominate.
    window_starts: list[float] = []
    time = 0.0
    while time < total_time - window_length + window_step * 0.5:
        window_starts.append(time)
        time += window_step

    if not window_starts:
        return np.array([1.0]), np.array([0.0])

    # `onsets` never changes across windows, so round it once up front
    # (previously this rounded the full onset array from scratch on every
    # iteration). Onsets are time-ordered, so rounding preserves that order
    # and each window's note count is `hi - lo` from `searchsorted` on the
    # window bounds -- one batched O(n_windows log n_notes) numpy call
    # instead of an O(n_windows * n_notes) boolean scan repeated per window.
    rounded_onsets = np.round(onsets, 3)
    starts_arr = np.asarray(window_starts, dtype=float)
    lo = np.searchsorted(rounded_onsets, np.round(starts_arr, 3), side="left")
    hi = np.searchsorted(rounded_onsets, np.round(starts_arr + window_length, 3), side="left")
    counts = (hi - lo).astype(float)

    densities = counts / window_length / local_expected_density
    center_weights = np.abs((starts_arr + window_length / 2.0) - total_time / 2.0) / (total_time / 2.0)
    return np.asarray(densities, dtype=float), np.asarray(center_weights, dtype=float)


def _onset_window_bounds(onsets_sec: np.ndarray, min_time: float, max_time: float) -> tuple[int, int]:
    """Sorted-onsets index bounds for a window (MIDI Toolbox `onsetwindow.m`, inclusive upper bound).

    Equivalent to `np.where((onsets_sec >= min_time) & (onsets_sec <= max_time))[0]`
    but returns the matching (contiguous) range in O(log n) instead of scanning
    every onset: onsets are already time-ordered (the mirror-pitch-series and
    meter-estimation helpers in this module rely on the same assumption), so
    the matching notes are exactly `onsets_sec[lo:hi]`.
    """
    lo = int(np.searchsorted(onsets_sec, min_time, side="left"))
    hi = int(np.searchsorted(onsets_sec, max_time, side="right"))
    return lo, hi


def _onset_window_indices(onsets_sec: np.ndarray, min_time: float, max_time: float) -> np.ndarray:
    """Seconds-based onset window (MIDI Toolbox `onsetwindow.m` with inclusive upper bound)."""
    lo, hi = _onset_window_bounds(onsets_sec, min_time, max_time)
    return np.arange(lo, hi)


def bisect_unbalance(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    onsets = _onsets_beats(melody)
    durations = _durations_beats(melody)
    note_count = len(onsets)
    total_time = onsets[-1] + durations[-1] - onsets[0]
    first_half = np.sum(onsets < total_time / 2.0) / note_count
    second_half = np.sum(onsets >= total_time / 2.0) / note_count
    return float(1.0 - 4.0 * first_half * second_half)


def center_mass_offset(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    onsets = _onsets_beats(melody)
    durations = _durations_beats(melody)
    total_time = onsets[-1] + durations[-1] - onsets[0]
    if total_time == 0.0:
        return 0.0
    return float(abs(np.mean(onsets) / total_time - 0.5))


def event_heterogeneity(
    melody: Melody,
    *,
    notes_per_window: int = 2,
    step_fraction: float = 0.5,
) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    densities, center_weights = _local_unbalance(
        melody,
        notes_per_window=notes_per_window,
        step_fraction=step_fraction,
    )
    weight_sum = center_weights.sum()
    if weight_sum == 0.0:
        return 0.0
    return float(np.sum(((densities - 1.0) ** 2) * center_weights) / weight_sum)


def av_abs_interval(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    pitches = _pitches(melody)
    if len(pitches) < 2:
        return 0.0
    intervals = np.abs(np.diff(pitches))
    return float(np.mean(np.log(intervals + 1.0)))


def mel_abruptness(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    pitches = _pitches(melody)
    onsets = _onsets_sec(melody)
    durations = _durations_sec(melody)
    if len(pitches) < 3:
        return 0.0

    total = 0.0
    for index in range(1, len(pitches) - 1):
        if (pitches[index + 1] - pitches[index]) * (pitches[index] - pitches[index - 1]) < 0:
            mean_interval = (
                abs(pitches[index + 1] - pitches[index]) + abs(pitches[index] - pitches[index - 1])
            ) / 2.0
            total += math.log(mean_interval + 1.0)

    normalizer = onsets[-1] + durations[-1]
    return float(total / normalizer) if normalizer else 0.0


def dur_abruptness(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    pitches = _pitches(melody)
    durations = _durations_sec(melody)
    if len(pitches) < 3:
        return 0.0

    total = 0.0
    for index in range(1, len(pitches) - 1):
        if (pitches[index + 1] - pitches[index]) * (pitches[index] - pitches[index - 1]) < 0:
            total += durations[index]

    normalizer = durations.sum()
    return float(total / normalizer) if normalizer else 0.0


def rhythm_abruptness(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    accented = _duration_accent(_durations_beats(melody))
    ratios: list[float] = []
    for index in range(len(accented) - 1):
        if accented[index + 1] > accented[index] and accented[index] > 0:
            ratios.append(accented[index + 1] / accented[index])
        elif accented[index + 1] <= accented[index] and accented[index + 1] > 0:
            ratios.append(accented[index] / accented[index + 1])
    return float(np.mean(ratios)) if ratios else 0.0


@lru_cache(maxsize=256)
def _mirror_pitch_series_cached(
    pitches: tuple[float, ...],
    onsets: tuple[float, ...],
    durations: tuple[float, ...],
) -> tuple[float, ...]:
    """Cached body of `_mirror_pitch_series`.

    `asym_total` and `asym_index` are both `@must complexity` features and
    both need this exact sampled pitch series for a melody, and caching it
    means the two features share one pass instead of each building it
    independently.

    Building the series means classifying every 0.0001-beat sample into
    the note sounding at that instant (or no note, during a rest) -- a
    melody note is "sounding" for samples in `[onset, onset + duration)`.
    That note index is a monotonically non-decreasing step function of
    time, so instead of a Python loop that advances `note_index` one
    sample at a time (the dominant cost of the MUST complexity features:
    a melody spanning tens of beats means hundreds of thousands of
    iterations), `np.searchsorted` locates every sample's note in one
    vectorized pass: `searchsorted(onsets, t, side="right") - 1` gives the
    index of the last note whose onset is at or before `t`, which is
    exactly the `note_index` the loop converges to for that `t`. Samples
    that land in a rest (after a note's duration ends but before the next
    onset) are then dropped with a boolean mask, matching the loop's
    `time < onsets[note_index] + durations[note_index]` check.
    """
    onsets_arr = np.asarray(onsets, dtype=float)
    durations_arr = np.asarray(durations, dtype=float)
    pitches_arr = np.asarray(pitches, dtype=float)
    total_time = onsets_arr[-1] + durations_arr[-1]
    sample_count = int(total_time / 0.0001) + 1
    times = np.arange(sample_count) * 0.0001
    note_index = np.searchsorted(onsets_arr, times, side="right") - 1
    np.clip(note_index, 0, len(onsets_arr) - 1, out=note_index)
    note_end_times = onsets_arr + durations_arr
    sounding = times < note_end_times[note_index]
    return tuple(pitches_arr[note_index[sounding]].tolist())


def _mirror_pitch_series(melody: Melody) -> np.ndarray:
    if len(melody.pitches) == 0:
        return np.array([], dtype=float)
    pitches = _pitches(melody)
    onsets = _onsets_beats(melody) - _onsets_beats(melody)[0]
    durations = _durations_beats(melody)
    series = _mirror_pitch_series_cached(
        tuple(pitches.tolist()), tuple(onsets.tolist()), tuple(durations.tolist())
    )
    return np.asarray(series, dtype=float)


def _mirror_series_key(melody: Melody) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Hashable (pitches, onsets, durations) cache key shared by the mirror-series helpers."""
    pitches = _pitches(melody)
    onsets = _onsets_beats(melody) - _onsets_beats(melody)[0]
    durations = _durations_beats(melody)
    return tuple(pitches.tolist()), tuple(onsets.tolist()), tuple(durations.tolist())


@lru_cache(maxsize=256)
def _mirror_asymmetry_stats(
    pitches: tuple[float, ...],
    onsets: tuple[float, ...],
    durations: tuple[float, ...],
) -> tuple[float, float]:
    """Cached `(asym_total, asym_index)` for a melody.

    `asym_total` and `asym_index` both reduce the same per-sample mirror
    pitch series (one sample per 0.0001 beat -- for a melody spanning tens
    of beats, that's hundreds of thousands to millions of samples) down to
    a single number. Caching `_mirror_pitch_series` itself (see
    `_mirror_pitch_series_cached` below) already avoids rebuilding that
    series twice, but the two features were still each independently
    paying two more O(sample-count) costs: reconstructing a numpy array
    from the cached tuple, and running their own `abs(series -
    series[::-1])` reduction over it -- for a several-hundred-note melody
    that "cheap" reduction is actually the largest remaining cost in the
    whole MUST complexity feature set. Computing both final scalars here,
    directly from the vectorized series construction (no huge tuple/array
    round-trip), means that per-sample numpy work happens exactly once no
    matter which of the two features (or both) get requested.
    """
    onsets_arr = np.asarray(onsets, dtype=float)
    durations_arr = np.asarray(durations, dtype=float)
    pitches_arr = np.asarray(pitches, dtype=float)
    total_time = onsets_arr[-1] + durations_arr[-1]
    sample_count = int(total_time / 0.0001) + 1
    times = np.arange(sample_count) * 0.0001
    note_index = np.searchsorted(onsets_arr, times, side="right") - 1
    np.clip(note_index, 0, len(onsets_arr) - 1, out=note_index)
    note_end_times = onsets_arr + durations_arr
    sounding = times < note_end_times[note_index]
    series = pitches_arr[note_index[sounding]]
    if series.size == 0:
        return 0.0, 0.0
    asymmetry = np.abs(series - series[::-1])
    total = float(asymmetry.sum() / asymmetry.size)
    index = float(np.sum(asymmetry > 0) / asymmetry.size)
    return total, index


def asym_total(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    total, _ = _mirror_asymmetry_stats(*_mirror_series_key(melody))
    return total


def asym_index(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    _, index = _mirror_asymmetry_stats(*_mirror_series_key(melody))
    return index


def event_density(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    onsets = _onsets_sec(melody)
    durations = _durations_sec(melody)
    note_count = len(melody.pitches)
    span = onsets[-1] + durations[-1]
    return float(note_count / span) if span else 0.0


def av_local_p1_entropy(
    melody: Melody,
    *,
    window_length: float = 1.0,
    window_step: float = 0.25,
) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    pitches = _pitches(melody)
    onsets = _onsets_sec(melody)
    durations = _durations_sec(melody)
    total_time = onsets[-1] + durations[-1]
    entropies: list[float] = []
    time = 0.0
    while time <= total_time + 1e-12:
        lo, hi = _onset_window_bounds(onsets, time - window_length, time)
        if hi > lo:
            entropies.append(
                _MUST_TOKENIZER.pitch_distribution(pitches[lo:hi]).entropy()
            )
        time += window_step
    return float(np.mean(entropies)) if entropies else 0.0


def p1_entropy(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    return _MUST_TOKENIZER.pdist1(melody).entropy()


def p2_entropy(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    return _MUST_TOKENIZER.pdist2(melody).entropy()


def p3_entropy(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    return _MUST_TOKENIZER.pdist3(melody).entropy()


def i1_entropy(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    return _MUST_TOKENIZER.idist1(melody).entropy()


def i2_entropy(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    return _MUST_TOKENIZER.idist2(melody).entropy()


def d1_entropy(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    return _MUST_TOKENIZER.ddist1(melody).entropy()


def d2_entropy(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    return _MUST_TOKENIZER.ddist2(melody).entropy()


def d3_entropy(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    return _MUST_TOKENIZER.ddist3(melody).entropy()


def wp_entropy(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    pitches = _pitches(melody).astype(int)
    if len(pitches) < 3:
        return 0.0
    weights = np.zeros(13, dtype=float)
    for index in range(len(pitches) - 2):
        order_index = _order_sign(pitches[index], pitches[index + 1], pitches[index + 2]) - 1
        weights[order_index] += float(np.std(pitches[index : index + 3]))
    total = weights.sum()
    if total == 0.0:
        return 0.0
    weights = weights[weights != 0.0] / total
    return float(-np.sum(weights * np.log(weights)))
