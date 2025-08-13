from typing import Dict, List, Optional, Tuple

from .ngram_counter import NGramCounter
from .representations import Melody


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
    """Base class for melody tokenization."""

    def __init__(self):
        """Initialize the tokenizer."""
        self.phrases = []
        self.ngram_counter = NGramCounter()

    def _calculate_iois(self, starts: List[float]) -> List[float]:
        """Calculate inter-onset intervals from start times.

        Parameters
        ----------
        starts : List[float]
            List of note start times

        Returns
        -------
        List[float]
            List of inter-onset intervals
        """
        return [starts[i] - starts[i - 1] for i in range(1, len(starts))]

    def _calculate_ioi_ratios(self, iois: List[float]) -> List[float]:
        """Calculate IOI ratios from inter-onset intervals.

        Parameters
        ----------
        iois : List[float]
            List of inter-onset intervals

        Returns
        -------
        List[float]
            List of IOI ratios, with None for the first note
        """
        ratios = [None]  # First note has no ratio
        ratios.extend([iois[i] / iois[i - 1] for i in range(1, len(iois))])
        return ratios

    def _classify_pitch_interval(self, interval: int) -> int:
        """Classify a pitch interval into a category.

        Parameters
        ----------
        interval : int
            The pitch interval in semitones

        Returns
        -------
        int
            The interval classification
        """
        abs_interval = abs(interval)
        if abs_interval == 0:
            return 0  # Unison
        elif abs_interval == 1:
            return 1  # Minor second
        elif abs_interval == 2:
            return 2  # Major second
        elif abs_interval == 3:
            return 3  # Minor third
        elif abs_interval == 4:
            return 4  # Major third
        elif abs_interval == 5:
            return 5  # Perfect fourth
        elif abs_interval == 7:
            return 6  # Perfect fifth
        elif abs_interval == 8:
            return 7  # Minor sixth
        elif abs_interval == 9:
            return 8  # Major sixth
        elif abs_interval == 10:
            return 9  # Minor seventh
        elif abs_interval == 11:
            return 10  # Major seventh
        elif abs_interval == 12:
            return 11  # Octave
        else:
            return 12  # Larger than octave

    def _classify_ioi_ratio(self, ratio: float) -> float:
        """Classify an IOI ratio into a category.

        Parameters
        ----------
        ratio : float
            The IOI ratio

        Returns
        -------
        float
            The IOI ratio classification
        """
        if ratio is None:
            return 0  # None
        elif ratio < 0.8118987:
            return 1  # Shorter (q)
        elif ratio < 1.4945858:
            return 2  # Equal (e)
        else:
            return 3  # Longer (l)

    def tokenize_melody(
        self, pitches: List[int], starts: List[float], ends: List[float]
    ) -> List[MType]:
        """Tokenize a melody into M-Type tokens.

        Parameters
        ----------
        pitches : List[int]
            List of MIDI pitch values
        starts : List[float]
            List of note start times
        ends : List[float]
            List of note end times

        Returns
        -------
        List[MType]
            List of M-Type tokens
        """
        if len(pitches) < 2:
            return []

        # Calculate pitch intervals
        pitch_intervals = [pitches[i] - pitches[i - 1] for i in range(1, len(pitches))]

        # Calculate IOIs and ratios
        iois = self._calculate_iois(starts)
        ioi_ratios = self._calculate_ioi_ratios(iois)

        # Create M-Type tokens
        tokens = []
        for i in range(len(pitch_intervals)):
            pitch_class = self._classify_pitch_interval(pitch_intervals[i])
            ioi_class = self._classify_ioi_ratio(ioi_ratios[i])
            tokens.append(MType(pitch_class, ioi_class))

        # Store tokens and update n-gram counts
        self.phrases = tokens
        self.ngram_counter.count_ngrams(tokens)

        return tokens

    def ngram_counts(self, n: Optional[int] = None) -> Dict:
        """Get n-gram counts for the current melody.

        Parameters
        ----------
        n : int, optional
            If provided, only return counts for n-grams of this length.
            If None, return counts for all n-gram lengths.

        Returns
        -------
        Dict
            Dictionary mapping each n-gram to its count
        """
        return self.ngram_counter.get_counts(n)


class FantasticTokenizer(MelodyTokenizer):
    """A tokenizer that implements the FANTASTIC melody tokenization scheme."""

    def __init__(self):
        """Initialize the FANTASTIC tokenizer."""
        super().__init__()

    def _classify_pitch_interval(self, interval: int) -> int:
        """Classify a pitch interval according to FANTASTIC scheme.

        Parameters
        ----------
        interval : int
            The pitch interval in semitones

        Returns
        -------
        int
            The interval classification
        """
        abs_interval = abs(interval)
        if abs_interval == 0:
            return 0  # Unison
        elif abs_interval == 1:
            return 1  # Minor second
        elif abs_interval == 2:
            return 2  # Major second
        elif abs_interval == 3:
            return 3  # Minor third
        elif abs_interval == 4:
            return 4  # Major third
        elif abs_interval == 5:
            return 5  # Perfect fourth
        elif abs_interval == 7:
            return 6  # Perfect fifth
        elif abs_interval == 8:
            return 7  # Minor sixth
        elif abs_interval == 9:
            return 8  # Major sixth
        elif abs_interval == 10:
            return 9  # Minor seventh
        elif abs_interval == 11:
            return 10  # Major seventh
        elif abs_interval == 12:
            return 11  # Octave
        else:
            return 12  # Larger than octave

    def _classify_ioi_ratio(self, ratio: float) -> float:
        """Classify an IOI ratio according to FANTASTIC scheme.

        Parameters
        ----------
        ratio : float
            The IOI ratio

        Returns
        -------
        float
            The IOI ratio classification
        """
        if ratio is None:
            return 0  # None
        elif ratio < 0.8118987:
            return 1  # Shorter (q)
        elif ratio < 1.4945858:
            return 2  # Equal (e)
        else:
            return 3  # Longer (l)

    def segment_melody(
        self, melody: Melody, phrase_gap: float = 1.5, units: str = "quarters"
    ) -> List[Melody]:
        """Segment melody into phrases based on IOI gaps.

        Parameters
        ----------
        melody : Melody
            The melody to segment
        phrase_gap : float, optional
            The minimum IOI gap to consider a new phrase, by default 1.5
        units : str, optional
            The units of the phrase gap, either "seconds" or "quarters", by default "quarters"

        Returns
        -------
        List[Melody]
            List of Melody objects representing phrases
        """
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

        # Calculate IOIs
        iois = []
        for i in range(1, len(melody.starts)):
            iois.append(melody.starts[i] - melody.starts[i - 1])
        iois.append(None)  # Add None for last note

        for i, (pitch, start, end, ioi) in enumerate(
            zip(melody.pitches, melody.starts, melody.ends, iois)
        ):
            # Check if we need to start a new phrase
            need_new_phrase = (
                len(current_phrase_pitches) > 0 and ioi is not None and ioi > phrase_gap
            )

            if need_new_phrase:
                # Create new melody for the phrase
                start_time = current_phrase_starts[0]
                adjusted_starts = [s - start_time for s in current_phrase_starts]
                adjusted_ends = [e - start_time for e in current_phrase_ends]

                # Create MIDI sequence string
                midi_seq = ", ".join(
                    f"Note(start={s:.6f}, end={e:.6f}, pitch={p}, velocity=90)"
                    for p, s, e in zip(
                        current_phrase_pitches.copy(), adjusted_starts, adjusted_ends
                    )
                )

                # Create dictionary with MIDI sequence
                midi_data = {"MIDI Sequence": midi_seq}

                # Create new Melody object
                phrases.append(Melody(midi_data, tempo=melody.tempo))

                # Reset current phrase
                current_phrase_pitches = []
                current_phrase_starts = []
                current_phrase_ends = []

            # Add note to current phrase
            current_phrase_pitches.append(pitch)
            current_phrase_starts.append(start)
            current_phrase_ends.append(end)

        # Handle final phrase
        if len(current_phrase_pitches) > 0:
            start_time = current_phrase_starts[0]
            adjusted_starts = [s - start_time for s in current_phrase_starts]
            adjusted_ends = [e - start_time for e in current_phrase_ends]

            # Create MIDI sequence string
            midi_seq = ", ".join(
                f"Note(start={s:.6f}, end={e:.6f}, pitch={p}, velocity=90)"
                for p, s, e in zip(
                    current_phrase_pitches, adjusted_starts, adjusted_ends
                )
            )

            # Create dictionary with MIDI sequence
            midi_data = {"MIDI Sequence": midi_seq}

            # Create new Melody object
            phrases.append(Melody(midi_data, tempo=melody.tempo))

        return phrases
