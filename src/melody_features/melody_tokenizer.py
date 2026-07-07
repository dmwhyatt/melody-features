from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional

import numpy as np

from melody_features.ngram_counter import NGramCounter
from melody_features.core.representations import Melody
from melody_features.feature_utils import _get_durations


class MType:
    """A class representing a melody token based on pitch interval and IOI ratio classifications."""

    def __init__(self, pitch_interval: int, ioi_ratio: float):
        """Initialize an M-Type token.

        Parameters
        ----------
        pitch_interval : int
            The pitch interval classification
        ioi_ratio : float
            The IOI ratio classification
        """
        self.pitch_interval = pitch_interval
        self.ioi_ratio = ioi_ratio

    def __str__(self) -> str:
        """Return string representation of the M-Type token."""
        return f"({self.pitch_interval}, {self.ioi_ratio})"

    def __repr__(self) -> str:
        """Return string representation of the M-Type token."""
        return self.__str__()

    def __eq__(self, other) -> bool:
        """Check if two M-Type tokens are equal."""
        if not isinstance(other, MType):
            return False
        return (
            self.pitch_interval == other.pitch_interval
            and self.ioi_ratio == other.ioi_ratio
        )

    def __hash__(self) -> int:
        """Return hash value of the M-Type token."""
        return hash((self.pitch_interval, self.ioi_ratio))


class MelodyTokenizer:
    """Base class for melody tokenization strategies."""


class FantasticTokenizer(MelodyTokenizer):
    """FANTASTIC melody tokenization using classified interval and IOI-ratio m-types."""

    def __init__(self, scheme: str = "FANTASTIC"):
        """Initialize the tokenizer with a specific interval classification scheme.

        Parameters
        ----------
        scheme : str, optional
            The scheme to use for pitch interval classification, by default "FANTASTIC"
            Options: "FANTASTIC", "SIMILE"
        """
        super().__init__()
        self.scheme = scheme
        self.phrases: list[MType] = []
        self.ngram_counter = NGramCounter()

    def _calculate_iois(self, starts: List[float]) -> List[float]:
        """Calculate inter-onset intervals from start times."""
        return [starts[i] - starts[i - 1] for i in range(1, len(starts))]

    def _calculate_ioi_ratios(self, iois: List[float]) -> List[float]:
        """Calculate IOI ratios from inter-onset intervals."""
        ratios = [None]
        ratios.extend([iois[i] / iois[i - 1] for i in range(1, len(iois))])
        return ratios

    def _classify_pitch_interval(self, interval: int, scheme: Optional[str] = None) -> int:
        """Classify a pitch interval into a category."""
        scheme = scheme or self.scheme
        abs_interval = abs(interval)

        if scheme == "FANTASTIC":
            if abs_interval == 0:
                return 0
            if abs_interval == 1:
                return 1
            if abs_interval == 2:
                return 2
            if abs_interval == 3:
                return 3
            if abs_interval == 4:
                return 4
            if abs_interval == 5:
                return 5
            if abs_interval == 7:
                return 6
            if abs_interval == 8:
                return 7
            if abs_interval == 9:
                return 8
            if abs_interval == 10:
                return 9
            if abs_interval == 11:
                return 10
            if abs_interval == 12:
                return 11
            return 12

        if scheme == "SIMILE":
            if interval == 0:
                return 0
            if 1 <= interval <= 2:
                return 1
            if 3 <= interval <= 4:
                return 2
            if 5 <= interval <= 7:
                return 3
            if interval > 7:
                return 4
            if -2 <= interval <= -1:
                return -1
            if -4 <= interval <= -3:
                return -2
            if -7 <= interval <= -5:
                return -3
            if interval < -7:
                return -4
            return None

        raise ValueError(f"Unknown interval classification scheme: {scheme}")

    def _classify_ioi_ratio(self, ratio: float) -> float:
        """Classify an IOI ratio into a category."""
        if ratio is None:
            return 0
        if ratio < 0.8118987:
            return 1
        if ratio < 1.4945858:
            return 2
        return 3

    def tokenize_melody(
        self, pitches: List[int], starts: List[float], ends: List[float]
    ) -> List[MType]:
        """Tokenize a melody into M-Type tokens."""
        if len(pitches) < 2:
            return []

        pitch_intervals = [pitches[i] - pitches[i - 1] for i in range(1, len(pitches))]
        iois = self._calculate_iois(starts)
        ioi_ratios = self._calculate_ioi_ratios(iois)

        tokens = []
        for index in range(len(pitch_intervals)):
            pitch_class = self._classify_pitch_interval(pitch_intervals[index])
            ioi_class = self._classify_ioi_ratio(ioi_ratios[index])
            tokens.append(MType(pitch_class, ioi_class))

        self.phrases = tokens
        self.ngram_counter.count_ngrams(tokens)
        return tokens

    def ngram_counts(self, n: Optional[int] = None) -> Dict:
        """Get n-gram counts for the current melody."""
        return self.ngram_counter.get_counts(n)

    def segment_melody(
        self, melody: Melody, phrase_gap: float = 1.5, units: str = "quarters"
    ) -> List[Melody]:
        """Segment melody into phrases based on IOI gaps."""
        assert units in ["seconds", "quarters"]
        if units == "seconds":
            raise NotImplementedError(
                "Seconds are not yet implemented, see issue #75: "
                "https://github.com/music-computing/amads/issues/75"
            )

        phrases = []
        current_phrase_pitches = []
        current_phrase_starts = []
        current_phrase_ends = []

        iois = []
        for index in range(1, len(melody.starts)):
            iois.append(melody.starts[index] - melody.starts[index - 1])
        iois.append(None)

        for pitch, start, end, ioi in zip(melody.pitches, melody.starts, melody.ends, iois):
            need_new_phrase = (
                len(current_phrase_pitches) > 0 and ioi is not None and ioi > phrase_gap
            )

            if need_new_phrase:
                start_time = current_phrase_starts[0]
                adjusted_starts = [s - start_time for s in current_phrase_starts]
                adjusted_ends = [e - start_time for e in current_phrase_ends]

                phrase_pitches = current_phrase_pitches.copy()
                midi_data = {
                    "pitches": phrase_pitches,
                    "starts": adjusted_starts,
                    "ends": adjusted_ends,
                    "MIDI Sequence": ", ".join(
                        f"Note(start={s:.6f}, end={e:.6f}, pitch={p}, velocity=90)"
                        for p, s, e in zip(
                            phrase_pitches, adjusted_starts, adjusted_ends
                        )
                    ),
                }

                phrases.append(Melody(midi_data, tempo=melody.tempo))

                current_phrase_pitches = []
                current_phrase_starts = []
                current_phrase_ends = []

            current_phrase_pitches.append(pitch)
            current_phrase_starts.append(start)
            current_phrase_ends.append(end)

        if len(current_phrase_pitches) > 0:
            start_time = current_phrase_starts[0]
            adjusted_starts = [s - start_time for s in current_phrase_starts]
            adjusted_ends = [e - start_time for e in current_phrase_ends]

            midi_data = {
                "pitches": current_phrase_pitches,
                "starts": adjusted_starts,
                "ends": adjusted_ends,
                "MIDI Sequence": ", ".join(
                    f"Note(start={s:.6f}, end={e:.6f}, pitch={p}, velocity=90)"
                    for p, s, e in zip(
                        current_phrase_pitches, adjusted_starts, adjusted_ends
                    )
                ),
            }

            phrases.append(Melody(midi_data, tempo=melody.tempo))

        return phrases


def _must_distribution_key(value: Any) -> Hashable:
    """Convert a MUST distribution label to a hashable dict key."""
    if isinstance(value, np.ndarray):
        flat = value.ravel()
        if flat.size == 1:
            value = flat.item()
        else:
            return tuple(_must_distribution_key(item) for item in flat.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


@dataclass(frozen=True)
class MustDistribution:
    """Normalized MUST distribution weights with their category labels."""

    values: np.ndarray
    weights: np.ndarray

    def as_dict(self) -> dict[Hashable, float]:
        """Map each category label to its normalized weight."""
        result: dict[Hashable, float] = {}
        for value, weight in zip(self.values, self.weights):
            result[_must_distribution_key(value)] = float(weight)
        return result

    def entropy(self) -> float:
        """Shannon entropy (natural log) of the weight vector."""
        from melody_features.algorithms.must import must_shannon_entropy

        return must_shannon_entropy(self.weights)


class MustTokenizer(MelodyTokenizer):
    """MUST distribution tokenization (Clemente et al., 2020).

    Implements `pdist*`, `idist*`, and `ddist*` on notematrix-style timing:
    onsets and durations in beats.
    """

    def pitch_tokens(self, melody: Melody) -> np.ndarray:
        """Raw MIDI pitch values (MUST notematrix column 4)."""
        return np.asarray(melody.pitches, dtype=int)

    def duration_tokens(self, melody: Melody) -> np.ndarray:
        """Beat durations for all notes except the last, rounded to 2 dp."""
        durations = np.asarray(
            _get_durations(melody.starts, melody.ends, melody.tempo),
            dtype=float,
        )
        if durations.size == 0:
            return durations
        return np.round(durations[:-1], 2)

    @staticmethod
    def pitch_distribution(pitches: np.ndarray) -> MustDistribution:
        """Marginal pitch distribution (`pdist1` on a pitch vector)."""
        pitches = np.asarray(pitches, dtype=int)
        if pitches.size == 0:
            return MustDistribution(values=np.array([]), weights=np.array([0.0]))
        values, counts = np.unique(pitches, return_counts=True)
        weights = counts.astype(float) / counts.sum()
        return MustDistribution(values=values, weights=weights)

    @staticmethod
    def _tuple_distribution(rows: np.ndarray) -> MustDistribution:
        if rows.size == 0:
            return MustDistribution(values=np.array([]), weights=np.array([]))
        _, inverse = np.unique(rows, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        values = np.unique(rows, axis=0)
        weights = counts.astype(float) / counts.sum()
        return MustDistribution(values=values, weights=weights)

    @staticmethod
    def _marginalize_intervals(
        pitch_rows: np.ndarray,
        pitch_weights: np.ndarray,
    ) -> MustDistribution:
        """Interval marginal from pitch n-tuple distribution (`idist*`)."""
        if pitch_rows.size == 0:
            return MustDistribution(values=np.array([]), weights=np.array([]))
        interval_diffs = np.diff(pitch_rows, axis=1)
        unique_intervals = np.unique(interval_diffs, axis=0)
        weights = []
        for interval in unique_intervals:
            mask = np.all(interval_diffs == interval, axis=1)
            weights.append(float(pitch_weights[mask].sum()))
        return MustDistribution(values=unique_intervals, weights=np.asarray(weights, dtype=float))

    def pdist1(self, melody: Melody) -> MustDistribution:
        """Pitch distribution (MUST `pdist1.m`)."""
        return self.pitch_distribution(self.pitch_tokens(melody))

    def pdist2(self, melody: Melody) -> MustDistribution:
        """2-tuple pitch distribution (MUST `pdist2.m`)."""
        pitches = self.pitch_tokens(melody)
        if len(pitches) < 2:
            return MustDistribution(values=np.array([]), weights=np.array([]))
        pairs = np.column_stack([pitches[:-1], pitches[1:]])
        return self._tuple_distribution(pairs)

    def pdist3(self, melody: Melody) -> MustDistribution:
        """3-tuple pitch distribution (MUST `pdist3.m`)."""
        pitches = self.pitch_tokens(melody)
        if len(pitches) < 3:
            return MustDistribution(values=np.array([]), weights=np.array([]))
        triples = np.column_stack([pitches[:-2], pitches[1:-1], pitches[2:]])
        return self._tuple_distribution(triples)

    def idist1(self, melody: Melody) -> MustDistribution:
        """Interval distribution marginalized from `pdist2` (MUST `idist1.m`)."""
        pitch_dist = self.pdist2(melody)
        if pitch_dist.values.size == 0:
            return MustDistribution(values=np.array([]), weights=np.array([]))
        return self._marginalize_intervals(pitch_dist.values, pitch_dist.weights)

    def idist2(self, melody: Melody) -> MustDistribution:
        """2-interval distribution marginalized from `pdist3` (MUST `idist2.m`)."""
        pitch_dist = self.pdist3(melody)
        if pitch_dist.values.size == 0:
            return MustDistribution(values=np.array([]), weights=np.array([]))
        return self._marginalize_intervals(pitch_dist.values, pitch_dist.weights)

    def ddist1(self, melody: Melody) -> MustDistribution:
        """Duration distribution in beats (MUST `ddist1.m`)."""
        durations = self.duration_tokens(melody)
        if durations.size == 0:
            return MustDistribution(values=np.array([]), weights=np.array([0.0]))
        values, counts = np.unique(durations, return_counts=True)
        weights = counts.astype(float) / counts.sum()
        return MustDistribution(values=values, weights=weights)

    def ddist2(self, melody: Melody) -> MustDistribution:
        """2-tuple duration distribution (MUST `ddist2.m`)."""
        durations = self.duration_tokens(melody)
        if len(durations) < 2:
            return MustDistribution(values=np.array([]), weights=np.array([]))
        pairs = np.column_stack([durations[:-1], durations[1:]])
        return self._tuple_distribution(pairs)

    def ddist3(self, melody: Melody) -> MustDistribution:
        """3-tuple duration distribution (MUST `ddist3.m`)."""
        durations = self.duration_tokens(melody)
        if len(durations) < 3:
            return MustDistribution(values=np.array([]), weights=np.array([1.0]))
        triples = np.column_stack([durations[:-2], durations[1:-1], durations[2:]])
        unique_triples = np.unique(triples, axis=0)
        weights = []
        for triple in unique_triples:
            weights.append(float(np.sum(triples == triple)))
        weights_arr = np.asarray(weights, dtype=float)
        return MustDistribution(values=unique_triples, weights=weights_arr / weights_arr.sum())


__all__ = [
    "MType",
    "MelodyTokenizer",
    "FantasticTokenizer",
    "MustDistribution",
    "MustTokenizer",
]
