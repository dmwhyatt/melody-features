"""Calculates the Step Contour of a melody, along with related features.

Supports both the original FANTASTIC behaviour and an AMADS default variant.
Exemplified in Steinbeck (1982) [2], Juhász (2000) [3], Eerola and Toiviainen (2004) [4].
"""

__author__ = "David Whyatt"

import numpy as np


class StepContour:
    """Class for calculating and analyzing the step contour of a melody.
    A step contour is a list of MIDI pitch values, repeated proportionally to the
    duration (measured in tatums) of each note relative to the total melody length.
    This list is normalized to a user defined length, defaulting to 64 steps as used in
    FANTASTIC. Rests are considered as extending the duration of the previous note.

    Examples
    --------
    >>> pitches = [60, 64, 67]  # C4, E4, G4
    >>> durations = [2.0, 1.0, 1.0]  # First note is 2 beats, others are 1 beat
    >>> sc = StepContour(pitches, durations)
    >>> len(sc.contour)  # Default length is 64
    64
    >>> pitches = [60, 62, 64, 65, 67]  # C4, D4, E4, F4, G4
    >>> durations = [1.0, 1.0, 1.0, 1.0, 1.0]  # Notes have equal durations
    >>> sc = StepContour(pitches, durations)
    >>> sc.contour[:8]  # First 8 values of 64-length contour
    [60, 60, 60, 60, 60, 60, 60, 60]
    >>> sc.global_variation  # Standard deviation of contour  # doctest: +ELLIPSIS
    2.3974...
    >>> sc.global_direction  # Correlation with ascending line  # doctest: +ELLIPSIS
    0.9746...
    >>> sc.local_variation  # Average absolute difference between adjacent values  # doctest: +ELLIPSIS
    0.1111...
    """

    _step_contour_length = 64

    def __init__(
        self,
        pitches: list[int],
        durations: list[float],
        step_contour_length: int = _step_contour_length,
        method: str = "amads",
    ):
        """Initialize StepContour with melody data.

        Parameters
        ----------
        pitches : list[int]
            List of pitch values
        durations : list[float]
            List of note durations measured in tatums
        step_contour_length : int, optional
            Length of the output step contour vector (default is 64)
        method : str, optional
            Method for contour statistics, either "amads" or "fantastic".
            Defaults to "amads". "fantastic" keeps parity with the original
            FANTASTIC implementation where applicable.

        References
        ----------
        [1] Müllensiefen, D. (2009). Fantastic: Feature ANalysis Technology Accessing
        STatistics (In a Corpus): Technical Report v1.5
        [2] W. Steinbeck, Struktur und Ähnlichkeit: Methoden automatisierter
            Melodieanalyse. Bärenreiter, 1982.
        [3] Juhász, Z. 2000. A model of variation in the music of a Hungarian ethnic
            group. Journal of New Music Research 29(2):159-172.
        [4] Eerola, T. & Toiviainen, P. (2004). MIDI Toolbox: MATLAB Tools for Music
            Research. University of Jyväskylä: Kopijyvä, Jyväskylä, Finland.

        Examples
        --------
        >>> sc = StepContour([60, 62], [2.0, 2.0], step_contour_length=4)
        >>> sc.contour
        [60, 60, 62, 62]
        """
        if len(pitches) != len(durations):
            raise ValueError(
                f"The length of pitches (currently {len(pitches)}) must be equal to "
                f"the length of durations (currently {len(durations)})"
            )
        if method not in ("amads", "fantastic"):
            raise ValueError(
                f"Method must be either 'amads' or 'fantastic', got {method}"
            )

        self._step_contour_length = step_contour_length
        self.method = method
        self.contour = self._calculate_contour(pitches, durations)

    def _normalize_durations(self, durations: list[float]) -> list[float]:
        """Helper function to normalize note durations to fit within 4 bars of 4/4 time
        (64 tatums total by default).

        Parameters
        ----------
        durations : list[float]
            List of duration values measured in tatums

        Returns
        -------
        list[float]
            List of normalized duration values

        Examples
        --------
        >>> sc = StepContour([60], [1.0])
        >>> sc._normalize_durations([2.0, 2.0])
        [32.0, 32.0]
        """
        total_duration = sum(durations)
        if total_duration == 0:
            raise ValueError("Total duration is 0, cannot normalize")

        normalized = [
            self._step_contour_length * (duration / total_duration)
            for duration in durations
        ]
        return normalized

    @classmethod
    def _expand_to_vector(
        cls,
        pitches: list[int],
        normalized_durations: list[float],
        step_contour_length: int,
    ) -> list[int]:
        """Helper function that resamples the melody to a vector of length
        step_contour_length.

        Parameters
        ----------
        pitches : list[int]
            List of pitch values
        normalized_durations : list[float]
            List of normalized duration values (should sum to step_contour_length)
        step_contour_length : int
            Length of the output step contour vector

        Returns
        -------
        list[int]
            List of length step_contour_length containing repeated pitch values

        Examples
        --------
        >>> StepContour._expand_to_vector([60, 62], [2.0, 2.0], step_contour_length=4)
        [60, 60, 62, 62]
        """
        if abs(sum(normalized_durations) - step_contour_length) > 1e-6:
            raise ValueError(
                f"The sum of normalized_durations ({sum(normalized_durations)}) must "
                f"be equal to the step contour length ({step_contour_length})"
            )
        # We interpret the output list as a vector of pitch samples taken
        # at times 0, 1, 2, ..., 63 where 63 = step_contour_length - 1
        # and the length of the normalized melody is 64.

        output_length = step_contour_length
        output_pitches = [None for _ in range(output_length)]
        output_times = list(range(output_length))

        output_index = 0
        offset = 0.0

        # Iterate over the input pitches and durations
        for sounding_pitch, duration in zip(pitches, normalized_durations):
            offset += duration

            while True:
                # Step through the output list, and populate any time slots that
                # are occupied by the current note.
                if output_index >= output_length:
                    break
                output_time = output_times[output_index]
                if output_time >= offset:
                    break
                output_pitches[output_index] = sounding_pitch
                output_index += 1

        return output_pitches

    def _calculate_contour(
        self, pitches: list[int], durations: list[float]
    ) -> list[int]:
        """Calculate the step contour from input pitches and durations.

        Examples
        --------
        >>> sc = StepContour([60, 62], [2.0, 2.0], step_contour_length=4)
        >>> sc._calculate_contour([60, 62], [2.0, 2.0])
        [60, 60, 62, 62]
        """
        normalized_durations = self._normalize_durations(durations)
        return self._expand_to_vector(
            pitches, normalized_durations, self._step_contour_length
        )

    @property
    def global_variation(self) -> float:
        """The overall pitch variability of the step contour.

        The step contour is a fixed-length pitch vector in which each note's MIDI
        pitch is repeated in proportion to its duration. This feature is the
        standard deviation of that vector, so long notes influence the result more
        than short notes. Larger values indicate a wider duration-weighted pitch
        spread. The `"amads"` method uses population standard deviation
        (`ddof=0`), while `"fantastic"` uses sample standard deviation
        (`ddof=1`) for compatibility with the original FANTASTIC convention.

        Returns
        -------
        float
            Standard deviation of the duration-weighted step-contour pitch vector.

        Examples
        --------
        >>> sc = StepContour([60, 62, 64], [1.0, 1.0, 1.0])
        >>> sc.global_variation  # doctest: +ELLIPSIS
        1.639...
        """
        if len(self.contour) < 2:
            return 0.0
        ddof = 1 if self.method == "fantastic" else 0
        return float(np.std(self.contour, ddof=ddof))

    @property
    def global_direction(self) -> float:
        """The overall ascending or descending tendency of the step contour.

        This is the Pearson correlation between the fixed-length step-contour
        pitch vector and an ascending linear ramp. Positive values indicate an
        overall upward trajectory, negative values indicate an overall downward
        trajectory, and values near zero indicate little linear pitch direction.
        A flat contour returns `0.0`.

        Returns
        -------
        float
            Correlation with an ascending linear ramp.

        Examples
        --------
        >>> sc = StepContour([60, 62, 64], [1.0, 1.0, 1.0])
        >>> sc.global_direction  # doctest: +ELLIPSIS
        0.942...
        >>> sc = StepContour([60, 60, 60], [1.0, 1.0, 1.0])
        >>> sc.global_direction
        0.0
        >>> sc = StepContour([64, 62, 60], [1.0, 1.0, 1.0])  # Descending melody
        >>> sc.global_direction  # doctest: +ELLIPSIS
        -0.942...
        """
        corr = np.corrcoef(self.contour, np.arange(self._step_contour_length))[0, 1]
        if np.isnan(corr) and len(self.contour) > 1:
            return 0.0
        return float(corr)

    @property
    def local_variation(self) -> float:
        """The average adjacent-sample change in the step contour.

        This feature compares each neighboring pair of samples in the fixed-length
        step-contour pitch vector and averages the absolute pitch differences.
        Because adjacent samples inside the same sustained note have difference
        zero, the measure emphasizes local pitch changes after duration-weighted
        resampling rather than the raw sequence of note-to-note intervals.

        Returns
        -------
        float
            Mean absolute difference between adjacent step-contour samples.

        Examples
        --------
        >>> sc = StepContour([60, 62, 64], [1.0, 1.0, 1.0])
        >>> sc.local_variation  # doctest: +ELLIPSIS
        0.0634...
        """
        pairs = list(zip(self.contour, self.contour[1:]))
        local_variation = sum(abs(c2 - c1) for c1, c2 in pairs) / len(pairs)
        return local_variation
