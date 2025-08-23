"""
Module for computing n-grams from melodic phrases, similar to FANTASTIC's M-type analysis.
"""

# import pandas as pd
from collections import Counter, namedtuple
from typing import Hashable, List, Optional

Note = namedtuple("Note", ["pitch", "start", "end", "duration", "ioi", "ioi_ratio"])


class MelodyTokenizer:
    """Base class for tokenizing melodies into n-grams."""

    def __init__(self):
        self.precision = 6
        self.phrases = []

    def tokenize_melody(
        self, pitches: List[int], starts: List[float], ends: List[float]
    ) -> List[List]:
        """
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
        List[List]
            List of tokenized phrases
        """
        notes = self.get_notes(pitches, starts, ends)
        self.phrases = self.segment_melody(notes)
        return [self.tokenize_phrase(phrase) for phrase in self.phrases]

    def get_notes(
        self, pitches: List[int], starts: List[float], ends: List[float]
    ) -> List[Note]:
        iois = []
        durations = []
        ioi_ratios = []

        for i, (pitch, start) in enumerate(zip(pitches, starts)):
            end = ends[i]
            duration = round(end - start, self.precision)

            if i < len(pitches) - 1:
                ioi = round(starts[i + 1] - start, self.precision)
            else:
                ioi = None

            durations.append(duration)
            iois.append(ioi)

        for i in range(len(pitches)):
            if i == 0:
                ioi_ratio = None
            else:
                ioi = iois[i]
                prev_ioi = iois[i - 1]
                if ioi is None or prev_ioi is None:
                    ioi_ratio = None
                else:
                    ioi_ratio = round(iois[i] / prev_ioi, self.precision)

            ioi_ratios.append(ioi_ratio)

        notes = [
            Note(pitch, start, ends[i], durations[i], iois[i], ioi_ratios[i])
            for i, (pitch, start) in enumerate(zip(pitches, starts))
        ]

        return notes

    def segment_melody(self, notes: List[Note]) -> List[List]:
        raise NotImplementedError

    def tokenize_phrase(self, phrase) -> List:
        raise NotImplementedError

    def ngram_counts(self, n: int) -> Counter:
        """Count n-grams in all phrases.

        Parameters
        ----------
        n : int
            Length of n-grams to count

        Returns
        -------
        Counter
            Counts of each n-gram
        """
        counts = Counter()
        for phrase in self.phrases:
            tokens = self.tokenize_phrase(phrase)
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i : i + n])
                counts[ngram] += 1
        return counts


class FantasticTokenizer(MelodyTokenizer):
    def __init__(self, phrase_gap: float = 1.0):
        super().__init__()
        self.phrase_gap = phrase_gap
        self.tokens = []

    def tokenize_melody(
        self, pitches: List[int], starts: List[float], ends: List[float]
    ) -> List[List]:
        self.tokens = super().tokenize_melody(pitches, starts, ends)
        return self.tokens

    def segment_melody(self, notes: List[Note]) -> List[List]:
        phrases = []
        current_phrase = []

        for note in notes:
            # Check whether we need to make a new phrase
            need_new_phrase = (
                len(current_phrase) > 0
                and note.start - current_phrase[-1].end > self.phrase_gap
            )
            if need_new_phrase:
                phrases.append(current_phrase)
                current_phrase = []
            current_phrase.append(note)

        if current_phrase:
            phrases.append(current_phrase)

        return phrases

    def tokenize_phrase(self, phrase) -> List:
        tokens = []

        for prev_note, current_note in zip(phrase[:-1], phrase[1:]):
            pitch_interval = current_note.pitch - prev_note.pitch
            ioi_ratio = current_note.ioi_ratio

            pitch_interval_class = self.classify_pitch_interval(pitch_interval)
            ioi_ratio_class = self.classify_ioi_ratio(ioi_ratio)

            token = (pitch_interval_class, ioi_ratio_class)
            tokens.append(token)

        return tokens

    def classify_pitch_interval(self, pitch_interval: Optional[int]) -> Hashable:
        """Classify pitch interval according to Fantastic's interval class scheme.

        Parameters
        ----------
        pitch_interval : int
            Interval in semitones between consecutive notes

        Returns
        -------
        str
            Interval class label (e.g. 'd8', 'd7', 'u2', etc.)
            'd' = downward interval
            'u' = upward interval
            's' = same pitch
            't' = tritone
        """
        # Clamp interval to [-12, 12] semitone range
        if pitch_interval is None:
            return None

        if pitch_interval < -12:
            pitch_interval = -12
        elif pitch_interval > 12:
            pitch_interval = 12

        # Map intervals to class labels based on Fantastic's scheme
        return self.interval_map[pitch_interval]

    interval_map = {
        -12: "d8",  # Descending octave
        -11: "d7",  # Descending major seventh
        -10: "d7",  # Descending minor seventh
        -9: "d6",  # Descending major sixth
        -8: "d6",  # Descending minor sixth
        -7: "d5",  # Descending perfect fifth
        -6: "dt",  # Descending tritone
        -5: "d4",  # Descending perfect fourth
        -4: "d3",  # Descending major third
        -3: "d3",  # Descending minor third
        -2: "d2",  # Descending major second
        -1: "d2",  # Descending minor second
        0: "s1",  # Unison
        1: "u2",  # Ascending minor second
        2: "u2",  # Ascending major second
        3: "u3",  # Ascending minor third
        4: "u3",  # Ascending major third
        5: "u4",  # Ascending perfect fourth
        6: "ut",  # Ascending tritone
        7: "u5",  # Ascending perfect fifth
        8: "u6",  # Ascending minor sixth
        9: "u6",  # Ascending major sixth
        10: "u7",  # Ascending minor seventh
        11: "u7",  # Ascending major seventh
        12: "u8",  # Ascending octave
    }

    def classify_ioi_ratio(self, ioi_ratio: Optional[float]) -> str:
        """Classify an IOI ratio into relative rhythm classes.

        Parameters
        ----------
        ioi_ratio : float
            Inter-onset interval ratio between consecutive notes

        Returns
        -------
        str
            'q' for quicker (<0.8119)
            'e' for equal (0.8119-1.4946)
            'l' for longer (>1.4946)
        """
        if ioi_ratio is None:
            return None
        elif ioi_ratio < 0.8118987:
            return "q"
        elif ioi_ratio < 1.4945858:
            return "e"
        else:
            return "l"

    def count_grams(
        self, sequence: List[Hashable], n: int, existing: Optional[Counter] = None
    ) -> Counter:

        # Count n-grams in a sequence
        if existing is None:
            existing = Counter()

        for i in range(len(sequence) - n + 1):
            # Convert sequence slice to tuple to ensure hashability
            ngram = tuple(
                tuple(x) if isinstance(x, list) else x for x in sequence[i : i + n]
            )
            existing[ngram] += 1

        return existing


# class MType:
#     """Class for analyzing melodic n-grams (m-types) in a melody."""

#     def __init__(self, pitches: List[int], starts: List[float], ends: List[float],
#                  n_limits: Tuple[int, int] = (1, 5)):
#         """
#         Initialize MType analyzer.

#         Parameters
#         ----------
#         pitches : List[int]
#             List of MIDI pitch values
#         starts : List[float]
#             List of note start times
#         ends : List[float]
#             List of note end times
#         n_limits : Tuple[int, int]
#             (min, max) values for n-gram lengths
#         """
#         self.pitches = pitches
#         self.starts = starts
#         self.ends = ends
#         self.n_limits = n_limits
#         self._phrases = None
#         self._ngram_counts = None
#         self._collapsed_counts = None

#     @property
#     def phrases(self) -> List[List[Dict]]:
#         """Segment melody into phrases based on temporal gaps."""
#         if self._phrases is None:
#             phrases = []
#             current_phrase = []

#             # Use gaps > 1 beat as phrase boundaries
#             for i in range(len(self.pitches)-1):
#                 current_phrase.append({
#                     'pitch': self.pitches[i],
#                     'onset': self.starts[i],
#                     'duration': self.ends[i] - self.starts[i]
#                 })

#                 # Check if there's a significant gap to next note
#                 if self.starts[i+1] - self.ends[i] > 1.0:  # 1 beat threshold
#                     if current_phrase:
#                         phrases.append(current_phrase)
#                         current_phrase = []

#             # Add last note and final phrase
#             if self.pitches:
#                 current_phrase.append({
#                     'pitch': self.pitches[-1],
#                     'onset': self.starts[-1],
#                     'duration': self.ends[-1] - self.starts[-1]
#                 })
#                 phrases.append(current_phrase)

#             self._phrases = phrases

#         return self._phrases

#     @staticmethod
#     def _diff_transform(phrase: List[Dict]) -> List[int]:
#         """Transform a phrase into a sequence of pitch intervals."""
#         if not phrase:
#             return []

#         pitches = [note['pitch'] for note in phrase]
#         return [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]

#     @staticmethod
#     def _class_transform(intervals: List[int]) -> List[str]:
#         """Transform intervals into contour classes."""
#         return ['U' if interval > 0 else 'D' if interval < 0 else 'S'
#                 for interval in intervals]

#     @staticmethod
#     def _create_ngrams(contour: List[str], n: int) -> List[str]:
#         """Create n-grams from a contour sequence."""
#         if len(contour) < n:
#             return []

#         return ['_'.join(contour[i:i+n])
#                 for i in range(len(contour)-n+1)]

#     @property
#     def ngram_counts(self) -> pd.DataFrame:
#         """Count n-grams in all phrases of the melody."""
#         if self._ngram_counts is None:
#             ngram_data = []

#             for phrase_id, phrase in enumerate(self.phrases):
#                 # Transform phrase to contour
#                 intervals = self._diff_transform(phrase)
#                 contour = self._class_transform(intervals)

#                 # Generate and count n-grams for each n
#                 for n in range(self.n_limits[0],
#                              min(self.n_limits[1] + 1, len(contour) + 1)):
#                     ngrams = self._create_ngrams(contour, n)
#                     counts = Counter(ngrams)

#                     for ngram, count in counts.items():
#                         ngram_data.append({
#                             'phrase_id': phrase_id,
#                             'n': n,
#                             'ngram': ngram,
#                             'count': count
#                         })

#             self._ngram_counts = pd.DataFrame(ngram_data)

#         return self._ngram_counts

#     @property
#     def collapsed_counts(self) -> pd.DataFrame:
#         """Get collapsed n-gram counts across all phrases."""
#         if self._collapsed_counts is None:
#             self._collapsed_counts = (self.ngram_counts
#                                     .groupby(['ngram', 'n'])['count']
#                                     .sum()
#                                     .reset_index())
#         return self._collapsed_counts


# # Example melody
# p = [60, 62, 64, 62, 60]
# s = [0.0, 1.0, 2.0, 3.0, 4.0]
# e = [0.8, 1.8, 2.8, 3.8, 4.8]

# # Analyze m-types
# mtype = MType(p, s, e)
# results = mtype.collapsed_counts
# print(results)
