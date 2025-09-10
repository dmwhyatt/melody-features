"""
Generic histogram analysis module for melodic features.

This module provides a base FeatureHistogram class that can be extended
for different types of histograms (pitch, duration, interval, etc.)
"""

import numpy as np
from typing import List, Dict, Any, Callable, Optional
from abc import ABC, abstractmethod


class FeatureHistogram(ABC):
    """
    Abstract base class for creating and analyzing feature histograms.
    
    This class provides common functionality for histogram analysis including
    statistical properties like skewness, kurtosis, mean, standard deviation,
    and entropy. This seeks to emulate the approach used by jSymbolic for computing
    these features, including specific implementations for skewness and kurtosis that differ
    from numpy default implementations.
    
    Parameters
    ----------
    values : List[Any]
        List of values to create histogram from
    transform_func : Optional[Callable], default=None
        Optional function to transform values before histogram creation
    """
    
    def __init__(self, values: List[Any], transform_func: Optional[Callable] = None):
        self.values = values
        self.transform_func = transform_func
        self._histogram = None
        self._histogram_values = None
        self._compute_histogram()
    
    @abstractmethod
    def _create_histogram(self, transformed_values: List[Any]) -> Dict[int, int]:
        """
        Create the histogram from transformed values.
        
        This method must be implemented by subclasses to define how
        the histogram is created for the specific feature type.
        
        Parameters
        ----------
        transformed_values : List[Any]
            List of transformed values
            
        Returns
        -------
        Dict[int, int]
            Dictionary mapping histogram bins to counts
        """
        raise NotImplementedError("Subclasses must implement _create_histogram method")
    
    def _compute_histogram(self):
        """Compute the histogram based on the values and transform function."""
        if not self.values:
            self._histogram = {}
            self._histogram_values = []
            return

        if self.transform_func:
            transformed_values = [self.transform_func(v) for v in self.values]
        else:
            transformed_values = self.values

        self._histogram = self._create_histogram(transformed_values)
        
        # Folding is used by the PitchClassHistogram class and corresponds to an ordering
        # of pitch classes according to the circle of fifths.
        try:
            is_folded = self.folded
        except AttributeError:
            is_folded = False
        
        if is_folded:
            # For folded fifths, use the histogram bin values weighted by their frequencies
            self._histogram_values = []
            for bin_idx, count in self._histogram.items():
                self._histogram_values.extend([bin_idx] * count)
        else:
            self._histogram_values = transformed_values

    @property
    def histogram(self) -> Dict[int, int]:
        """Get the histogram dictionary."""
        return self._histogram.copy()

    @property
    def histogram_values(self) -> List[int]:
        """Get the values array used for statistical calculations."""
        return self._histogram_values.copy()

    # skewness is implemented in Cory McKay's UtilityClasses.jar, which I decompiled to find this.
    @property
    def skewness(self) -> float:
        """
        Calculate the median skewness using Pearson's median skewness formula.
        
        Returns
        -------
        float
            Median skewness value. Returns 0.0 for empty histograms or when variance is 0.
        """
        if len(self._histogram_values) < 3:
            return 0.0

        mean = np.mean(self._histogram_values)
        # jSymbolic uses the median as the upper middle element of the sorted array
        sorted_vals = sorted(self._histogram_values)
        centre = len(sorted_vals) // 2
        median = float(sorted_vals[centre])
        std = np.std(self._histogram_values, ddof=1)

        if std == 0.0:
            return 0.0

        return float(3 * (mean - median) / std)

    # this was also found by decompiling Cory McKay's UtilityClasses.jar
    @property
    def kurtosis(self) -> float:
        """
        Calculate the sample excess kurtosis using jSymbolic's method.
        
        Interpretation:
        - Positive values: leptokurtic (heavier tails than normal)
        - Negative values: platykurtic (lighter tails than normal)
        - Zero: mesokurtic (similar to normal distribution)
        
        Returns
        -------
        float
            Sample excess kurtosis value. Returns 0.0 for empty histograms or when variance is 0.
        """
        if len(self._histogram_values) < 4:
            return 0.0

        n = len(self._histogram_values)
        mean = np.mean(self._histogram_values)
        std = np.std(self._histogram_values, ddof=1)

        if std == 0.0:
            return 0.0
        
        coefficient = (n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0))

        numerator = sum((x - mean) ** 4 for x in self._histogram_values)
        denominator = std ** 4

        second_term = (3.0 * (n - 1.0) * (n - 1.0)) / ((n - 2.0) * (n - 3.0))

        return float((coefficient * numerator / denominator) + second_term)

    @property
    def mean(self) -> float:
        """
        Calculate the mean of the histogram values.
        
        Returns
        -------
        float
            Mean value. Returns 0.0 for empty histograms.
        """
        if not self._histogram_values:
            return 0.0

        return float(np.mean(self._histogram_values))

    @property
    def std(self) -> float:
        """
        Calculate the standard deviation of the histogram values.
        
        Returns
        -------
        float
            Standard deviation. Returns 0.0 for empty histograms.
        """
        if not self._histogram_values:
            return 0.0

        return float(np.std(self._histogram_values, ddof=1))

    def get_histogram_as_dict(self) -> Dict[str, int]:
        """
        Get the histogram as a dictionary with string keys for compatibility
        with existing feature extraction system.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping bin strings to counts
        """
        return {str(bin_val): count for bin_val, count in self._histogram.items()}

    def __repr__(self) -> str:
        """String representation of the histogram."""
        return f"{self.__class__.__name__}({len(self.values)} values)"

    def __str__(self) -> str:
        """String representation of the histogram."""
        return self.__repr__()


def create_feature_histogram(histogram_class: type, values: List[Any], **kwargs) -> FeatureHistogram:
    """
    Helper function to create a FeatureHistogram instance.

    Parameters
    ----------
    histogram_class : type
        The histogram class to instantiate
    values : List[Any]
        List of values to create histogram from
    **kwargs
        Additional keyword arguments to pass to the histogram class

    Returns
    -------
    FeatureHistogram
        A FeatureHistogram instance
    """
    return histogram_class(values, **kwargs)

class PitchHistogram(FeatureHistogram):
    """
    A generic class for creating and analyzing pitch histograms.

    This class can work with any range of pitch values and creates histograms
    based on the actual pitch values provided.
    
    Parameters
    ----------
    pitches : List[int]
        List of pitch values (can be MIDI pitches, pitch classes, or any integer range)
    """

    def __init__(self, pitches: List[int]):
        super().__init__(pitches)

    def _create_histogram(self, transformed_values: List[int]) -> Dict[int, int]:
        """
        Create the histogram from input values.

        Parameters
        ----------
        transformed_values : List[int]
            List of pitch values

        Returns
        -------
        Dict[int, int]
            Dictionary mapping pitch values to counts
        """
        if not transformed_values:
            return {}

        histogram = {i: 0 for i in range(128)}
        for pitch in transformed_values:
            if 0 <= pitch <= 127:
                histogram[pitch] += 1

        return histogram


class RhythmicValueHistogram(FeatureHistogram):
    """
    A class for creating and analyzing rhythmic value histograms, following
    the implementation in jSymbolic.

    Durations are provided in ticks (pulses per quarter note, or PPQN units).
    Each duration is mapped to the nearest of 12 fixed rhythmic bins using
    thresholds at the midpoints between adjacent ideal values. The final
    histogram is normalized so that bin values sum to 1.0 (or 0.0 if empty).

    Bins:
    [0] 32nd note (or less), [1] 16th note, [2] 8th note, [3] dotted 8th note,
    [4] quarter note, [5] dotted quarter note, [6] half note, [7] dotted half note,
    [8] whole note, [9] dotted whole note, [10] double whole note, [11] dotted double whole note (or more)

    Parameters
    ----------
    durations_in_ticks : List[float]
        Durations in ticks. Ticks are PPQN-relative units.
    ppqn : int
        Pulses per quarter note used to compute ideal rhythmic durations.
    """

    def __init__(self, durations_in_ticks: List[float], ppqn: int):
        self.ppqn = ppqn
        super().__init__(durations_in_ticks)

    def _ideal_ticks(self) -> List[float]:
        q = float(self.ppqn)
        # 32nd, 16th, 8th, dotted 8th, quarter, dotted quarter,
        # half, dotted half, whole, dotted whole, double, dotted double
        return [
            q / 8.0,
            q / 4.0,
            q / 2.0,
            3.0 * q / 4.0,
            q,
            1.5 * q,
            2.0 * q,
            3.0 * q,
            4.0 * q,
            6.0 * q,
            8.0 * q,
            12.0 * q,
        ]

    def bin_values_quarter_notes(self) -> List[float]:
        """
        Ideal rhythmic values expressed in quarter notes for this histogram's PPQN.

        Returns
        -------
        List[float]
            List of 12 rhythmic values in quarter notes corresponding to bins 0..11
        """
        q = float(self.ppqn)
        if q == 0.0:
            return [0.0] * 12
        return [v / q for v in self._ideal_ticks()]

    def bin_midpoints_quarter_notes(self) -> List[float]:
        """Midpoints (in quarter notes) between adjacent ideal rhythmic values.

        Returns
        -------
        List[float]
            List of 11 midpoints defining thresholds between the 12 bins.
        """
        ideals_qn = self.bin_values_quarter_notes()
        return [(ideals_qn[i] + ideals_qn[i + 1]) / 2.0 for i in range(len(ideals_qn) - 1)]

    def map_quarter_notes_to_bin_index(self, duration_qn: float) -> int:
        """Map a duration (in quarter notes) to the nearest rhythmic bin index 0..11.

        Uses midpoints between ideal values as thresholds, matching histogram logic.
        """
        mids = self.bin_midpoints_quarter_notes()
        d = float(duration_qn)
        if d < mids[0]:
            return 0
        for i in range(1, 11):
            if mids[i - 1] <= d < mids[i]:
                return i
        return 11

    def _create_histogram(self, transformed_values: List[float]) -> Dict[int, float]:
        if not transformed_values:
            return {i: 0.0 for i in range(12)}

        ideals = self._ideal_ticks()

        # Precompute midpoints between consecutive ideals for thresholding
        mids = [(ideals[i] + ideals[i + 1]) / 2.0 for i in range(len(ideals) - 1)]

        counts: Dict[int, int] = {i: 0 for i in range(12)}

        for dur in transformed_values:
            d = float(dur)
            if d < mids[0]:
                counts[0] += 1
                continue
            placed = False
            for i in range(1, 11):
                if mids[i - 1] <= d < mids[i]:
                    counts[i] += 1
                    placed = True
                    break
            if placed:
                continue
            counts[11] += 1

        total = float(sum(counts.values()))
        if total == 0.0:
            return {i: 0.0 for i in range(12)}

        normalized: Dict[int, float] = {i: counts[i] / total for i in range(12)}
        return normalized

    def get_histogram_as_dict(self) -> Dict[str, float]:
        """
        Return normalized histogram with string keys, summing to 1.0 (or all zeros).
        """
        return {str(bin_val): float(count) for bin_val, count in self._histogram.items()}


class BeatHistogram:
    """
    Compute jSymbolic informed beat histograms from a per-tick rhythm score. Rhythm_score is the sum of note-on velocities per tick.

    Produces two BPM-indexed histograms (indices 0..max_bpm), normalized to sum to 1:
    - beat_histogram
    - beat_histogram_120_bpm_standardized (computed with ticks_per_second at 120 BPM)

    Parameters
    ----------
    rhythm_score : List[int]
        Total velocity per tick across all pitched channels
    mean_ticks_per_second : float
        Average ticks per second for the sequence (accounts for tempo)
    ppqn : int
        Pulses per quarter note (MIDI resolution)
    min_bpm : int
        Minimum BPM to consider (default 40)
    max_bpm : int
        Maximum BPM to consider (default 200)
    """
    # references to multiple pitched channels here are superfluous, as we only process monophonic music with this package
    # however, across all channels is how the class is described in jSymbolic.
    def __init__(
        self,
        rhythm_score: List[int],
        mean_ticks_per_second: float,
        ppqn: int,
        min_bpm: int = 40,
        max_bpm: int = 200,
    ) -> None:
        self.rhythm_score = list(rhythm_score)
        self.mean_ticks_per_second = float(mean_ticks_per_second)
        self.ppqn = int(ppqn)
        self.min_bpm = int(min_bpm)
        self.max_bpm = int(max_bpm)

        size = self.max_bpm + 1
        self.beat_histogram: List[float] = [0.0] * size
        self.beat_histogram_120_bpm_standardized: List[float] = [0.0] * size

        self._generate()

    @staticmethod
    def _convert_bpm_to_ticks(bpm: int, ticks_per_second: float) -> int:
        if bpm <= 0 or ticks_per_second <= 0.0:
            return 0
        ticks_per_beat = ticks_per_second * 60.0 / float(bpm)
        return max(0, int(round(ticks_per_beat)))

    @staticmethod
    def _autocorrelate(series: List[int]) -> List[float]:
        """Produces autocorrelation following jSymbolic's implementation.
        
        Uses FFT-based autocorrelation for better performance on long sequences.
        """
        series = np.array(series, dtype=np.float64)
        n = len(series)

        if n == 0:
            return []

        # Use FFT-based autocorrelation for better performance
        # This is mathematically equivalent to np.correlate but faster for long sequences
        fft_series = np.fft.fft(series, n=2*n)
        autocorr = np.fft.ifft(fft_series * np.conj(fft_series))
        autocorr = np.real(autocorr[:n])  # Take only positive lags

        # Normalize by the total length
        autocorr = autocorr / n

        return autocorr.tolist()

    @staticmethod
    def _normalize(arr: List[float]) -> List[float]:
        total_sum = float(sum(arr))
        if total_sum == 0.0:
            return [0.0 for _ in arr]
        return [value / total_sum for value in arr]

    def _generate(self) -> None:
        if not self.rhythm_score or self.mean_ticks_per_second <= 0.0:
            return

        upper_len = self._convert_bpm_to_ticks(self.min_bpm - 1, self.mean_ticks_per_second)
        ticks_per_second_120 = float(self.ppqn) * 2.0
        upper_len_std = self._convert_bpm_to_ticks(self.min_bpm - 1, ticks_per_second_120)

        autocorr = self._autocorrelate(self.rhythm_score)

        # Tick-interval histograms over lags corresponding to BPM range
        lower_start = self._convert_bpm_to_ticks(self.max_bpm, self.mean_ticks_per_second)
        tick_histogram = [0.0] * max(upper_len, 0)
        for lag in range(lower_start, len(tick_histogram)):
            if lag < len(autocorr):
                tick_histogram[lag] = autocorr[lag]

        # also for the standardized histogram, standardized to 120 BPM (ticks_per_second = ppqn * 2)
        lower_start_std = self._convert_bpm_to_ticks(self.max_bpm, ticks_per_second_120)
        tick_histogram_std = [0.0] * max(upper_len_std, 0)
        for lag in range(lower_start_std, len(tick_histogram_std)):
            if lag < len(autocorr):
                tick_histogram_std[lag] = autocorr[lag]

        # Aggregate tick-interval bins into BPM-indexed histograms
        for bin_bpm in range(self.min_bpm, self.max_bpm + 1):
            a = self._convert_bpm_to_ticks(bin_bpm, self.mean_ticks_per_second)
            b = self._convert_bpm_to_ticks(bin_bpm - 1, self.mean_ticks_per_second)
            val = 0.0
            for tick in range(a, min(b, len(tick_histogram))):
                val += tick_histogram[tick]
            self.beat_histogram[bin_bpm] = val

            a_std = self._convert_bpm_to_ticks(bin_bpm, ticks_per_second_120)
            b_std = self._convert_bpm_to_ticks(bin_bpm - 1, ticks_per_second_120)
            val_std = 0.0
            for tick in range(a_std, min(b_std, len(tick_histogram_std))):
                val_std += tick_histogram_std[tick]
            self.beat_histogram_120_bpm_standardized[bin_bpm] = val_std

        # Normalize both histograms
        self.beat_histogram = self._normalize(self.beat_histogram)
        self.beat_histogram_120_bpm_standardized = self._normalize(
            self.beat_histogram_120_bpm_standardized
        )


def create_beat_histogram(
    rhythm_score: List[int],
    mean_ticks_per_second: float,
    ppqn: int,
    min_bpm: int = 40,
    max_bpm: int = 200,
) -> BeatHistogram:
    """
    Helper function to compute beat histograms from a per-tick rhythm score.
    """
    return BeatHistogram(
        rhythm_score=rhythm_score,
        mean_ticks_per_second=mean_ticks_per_second,
        ppqn=ppqn,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )

class PitchClassHistogram(FeatureHistogram):
    """
    A class for creating and analyzing pitch class histograms. This class supports
    both regular and folded fifths arrangements. Folded fifths is the name given to a
    histogram where the pitch classes are ordered according to the circle of fifths.
    
    Parameters
    ----------
    pitches : List[int]
        List of MIDI pitch values (will be converted to pitch classes)
    folded : bool, default=False
        If True, creates a folded fifths histogram. If False, creates a regular
        pitch class histogram.
    """

    def __init__(self, pitches: List[int], folded: bool = False):
        self.folded = folded
        super().__init__(pitches, transform_func=lambda x: x % 12)

    def _create_histogram(self, transformed_values: List[int]) -> Dict[int, int]:
        """
        Create the histogram from pitch class values.
        
        Parameters
        ----------
        transformed_values : List[int]
            List of pitch class values (0-11)
            
        Returns
        -------
        Dict[int, int]
            Dictionary mapping pitch classes to counts
        """
        if self.folded:
            return self._create_folded_fifths_histogram(transformed_values)
        else:
            return self._create_regular_histogram(transformed_values)

    def _create_regular_histogram(self, pitch_classes: List[int]) -> Dict[int, int]:
        """Create a regular pitch class histogram."""
        histogram = {i: 0 for i in range(12)}
        for pc in pitch_classes:
            if 0 <= pc <= 11: 
                histogram[pc] += 1
        return histogram

    def _create_folded_fifths_histogram(self, pitch_classes: List[int]) -> Dict[int, int]:
        """
        Create a folded fifths pitch class histogram.
        
        Uses the equation B = (7a) mod 12 to reorder pitch classes
        according to the circle of fifths.
        """
        pc_counts = {i: 0 for i in range(12)}
        for pc in pitch_classes:
            if 0 <= pc <= 11: 
                pc_counts[pc] += 1

        folded_histogram = {i: 0 for i in range(12)}
        for original_pc in range(12):
            folded_pc = (7 * original_pc) % 12
            folded_histogram[folded_pc] = pc_counts[original_pc]

        return folded_histogram


class DurationHistogram(FeatureHistogram):
    """
    A class for creating and analyzing duration histograms.
    
    Parameters
    ----------
    durations : List[float]
        List of duration values (in seconds or beats)
    num_bins : int, default=10
        Number of bins for the histogram
    normalize : bool, default=False
        If True, normalize durations to [0, 1] range before binning
    """

    def __init__(self, durations: List[float], num_bins: int = 10, normalize: bool = False):
        self.num_bins = num_bins
        self.normalize = normalize

        if normalize:
            super().__init__(durations, transform_func=self._normalize_duration)
        else:
            super().__init__(durations)

    def _normalize_duration(self, duration: float) -> float:
        """Normalize duration to [0, 1] range."""
        if not self.values:
            return duration

        min_dur = min(self.values)
        max_dur = max(self.values)

        if max_dur == min_dur:
            return 0.0

        return (duration - min_dur) / (max_dur - min_dur)

    def _create_histogram(self, transformed_values: List[float]) -> Dict[int, int]:
        """
        Create the histogram from duration values.
        
        Parameters
        ----------
        transformed_values : List[float]
            List of duration values (possibly normalized)
            
        Returns
        -------
        Dict[int, int]
            Dictionary mapping bin indices to counts
        """
        if not transformed_values:
            return {}

        min_val = min(transformed_values)
        max_val = max(transformed_values)

        if min_val == max_val:
            return {0: len(transformed_values)}

        bin_width = (max_val - min_val) / self.num_bins
        histogram = {}

        for val in transformed_values:
            bin_idx = min(int((val - min_val) / bin_width), self.num_bins - 1)
            histogram[bin_idx] = histogram.get(bin_idx, 0) + 1

        return histogram

def create_pitch_histogram(pitches: List[int]) -> PitchHistogram:
    """
    Helper function to create a PitchHistogram instance.
    
    Parameters
    ----------
    pitches : List[int]
        List of pitch values
        
    Returns
    -------
    PitchHistogram
        A PitchHistogram instance
    """
    return PitchHistogram(pitches)


def create_pitch_class_histogram(pitches: List[int], folded: bool = False) -> PitchClassHistogram:
    """
    Helper function to create a PitchClassHistogram instance.
    
    Parameters
    ----------
    pitches : List[int]
        List of MIDI pitch values
    folded : bool, default=False
        If True, creates a folded fifths histogram
        
    Returns
    -------
    PitchClassHistogram
        A PitchClassHistogram instance
    """
    return PitchClassHistogram(pitches, folded=folded)


def create_duration_histogram(durations: List[float], num_bins: int = 10, normalize: bool = False) -> DurationHistogram:
    """
    Helper function to create a DurationHistogram instance.
    
    Parameters
    ----------
    durations : List[float]
        List of duration values
    num_bins : int, default=10
        Number of bins for the histogram
    normalize : bool, default=False
        If True, normalize durations to [0, 1] range
        
    Returns
    -------
    DurationHistogram
        A DurationHistogram instance
    """
    return DurationHistogram(durations, num_bins=num_bins, normalize=normalize)


def create_rhythmic_value_histogram(durations_in_ticks: List[float], ppqn: int) -> RhythmicValueHistogram:
    """
    Helper function to create a RhythmicValueHistogram instance.

    Parameters
    ----------
    durations_in_ticks : List[float]
        Durations in ticks (PPQN-based quarter-note units).
    ppqn : int
        Pulses per quarter note.

    Returns
    -------
    RhythmicValueHistogram
        A RhythmicValueHistogram instance
    """
    return RhythmicValueHistogram(durations_in_ticks, ppqn)


class MelodicIntervalHistogram(FeatureHistogram):
    """
    A class for creating and analyzing melodic interval histograms, following
    the jSymbolic implementation.

    This class creates histograms of melodic intervals (pitch differences between
    consecutive notes) with bins for intervals 0-127 (absolute values).
    The histogram is normalized to sum to 1.0.

    Parameters
    ----------
    intervals : List[int]
        List of melodic intervals (can be positive or negative)
    use_absolute : bool, default=True
        If True, uses absolute values of intervals (0-127 range).
        If False, uses signed intervals (-127 to 127 range).
    """
    
    def __init__(self, intervals: List[int], use_absolute: bool = True):
        self.use_absolute = use_absolute
        if use_absolute:
            # Transform to absolute values for histogram creation
            super().__init__(intervals, transform_func=abs)
        else:
            # Use intervals as-is (signed)
            super().__init__(intervals)
    
    def _create_histogram(self, transformed_values: List[int]) -> Dict[int, float]:
        """
        Create the histogram from interval values.
        
        Parameters
        ----------
        transformed_values : List[int]
            List of interval values (absolute if use_absolute=True, signed otherwise)
            
        Returns
        -------
        Dict[int, float]
            Dictionary mapping interval values to normalized counts
        """
        if not transformed_values:
            return {i: 0.0 for i in range(128)}

        # Initialize histogram bins
        if self.use_absolute:
            # For absolute intervals, use bins 0-127
            histogram = {i: 0 for i in range(128)}
            for interval in transformed_values:
                if 0 <= interval <= 127:
                    histogram[interval] += 1
        else:
            # For signed intervals, use bins -127 to 127
            histogram = {i: 0 for i in range(-127, 128)}
            for interval in transformed_values:
                if -127 <= interval <= 127:
                    histogram[interval] += 1

        total = sum(histogram.values())
        if total == 0:
            return {k: 0.0 for k in histogram.keys()}

        normalized = {k: v / total for k, v in histogram.items()}
        return normalized

    def get_histogram_as_dict(self) -> Dict[str, float]:
        """
        Get the histogram as a dictionary with string keys for compatibility
        with existing feature extraction system.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping interval strings to normalized counts
        """
        return {str(interval): float(count) for interval, count in self._histogram.items()}


def create_melodic_interval_histogram(intervals: List[int], use_absolute: bool = True) -> MelodicIntervalHistogram:
    """
    Helper function to create a MelodicIntervalHistogram instance.

    Parameters
    ----------
    intervals : List[int]
        List of melodic intervals
    use_absolute : bool, default=True
        If True, uses absolute values of intervals (0-127 range).
        If False, uses signed intervals (-127 to 127 range).

    Returns
    -------
    MelodicIntervalHistogram
        A MelodicIntervalHistogram instance
    """
    return MelodicIntervalHistogram(intervals, use_absolute=use_absolute)
