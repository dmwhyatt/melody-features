"""Core algorithms for the MUST feature set (Clemente et al., 2020)."""

from __future__ import annotations

import math
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

    densities: list[float] = []
    center_weights: list[float] = []
    time = 0.0
    while time < total_time - window_length + window_step * 0.5:
        rounded_onsets = np.round(onsets, 3)
        count = np.sum(
            (rounded_onsets >= np.round(time, 3))
            & (rounded_onsets < np.round(time + window_length, 3))
        )
        densities.append(count / window_length / local_expected_density)
        center_weights.append(abs((time + window_length / 2.0) - total_time / 2.0) / (total_time / 2.0))
        time += window_step

    if not densities:
        return np.array([1.0]), np.array([0.0])
    return np.asarray(densities, dtype=float), np.asarray(center_weights, dtype=float)


def _onset_window_indices(onsets_sec: np.ndarray, min_time: float, max_time: float) -> np.ndarray:
    """Seconds-based onset window (MIDI Toolbox `onsetwindow.m` with inclusive upper bound)."""
    return np.where((onsets_sec >= min_time) & (onsets_sec <= max_time))[0]


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


def _mirror_pitch_series(melody: Melody) -> np.ndarray:
    if len(melody.pitches) == 0:
        return np.array([], dtype=float)
    pitches = _pitches(melody)
    onsets = _onsets_beats(melody) - _onsets_beats(melody)[0]
    durations = _durations_beats(melody)
    total_time = onsets[-1] + durations[-1]
    series: list[float] = []
    note_index = 0
    sample_count = int(total_time / 0.0001) + 1
    for sample in range(sample_count):
        time = sample * 0.0001
        if note_index < len(onsets) - 1 and time >= onsets[note_index + 1]:
            note_index += 1
        if time < onsets[note_index] + durations[note_index]:
            series.append(pitches[note_index])
    return np.asarray(series, dtype=float)


def asym_total(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    series = _mirror_pitch_series(melody)
    if series.size == 0:
        return 0.0
    asymmetry = np.abs(series - series[::-1])
    return float(asymmetry.sum() / asymmetry.size)


def asym_index(melody: Melody) -> float:
    if (empty := _zero_for_empty_melody(melody)) is not None:
        return empty
    series = _mirror_pitch_series(melody)
    if series.size == 0:
        return 0.0
    asymmetry = np.abs(series - series[::-1])
    return float(np.sum(asymmetry > 0) / asymmetry.size)


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
        indices = _onset_window_indices(onsets, time - window_length, time)
        if indices.size:
            entropies.append(
                _MUST_TOKENIZER.pitch_distribution(pitches[indices]).entropy()
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
