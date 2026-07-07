"""Inter-onset interval feature definitions."""

import numpy as np

from ..feature_utils import ioi_from_starts
from ..utils.distributional import histogram_bins
from ..feature_decorators import idyom, interval, jsymbolic, novel, rhythm


__all__ = [
    "ioi",
    "ioi_mean",
    "average_time_between_attacks",
    "ioi_standard_deviation",
    "variability_of_time_between_attacks",
    "ioi_ratio",
    "ioi_ratio_mean",
    "ioi_ratio_standard_deviation",
    "ioi_range",
    "ioi_contour",
    "ioi_contour_mean",
    "ioi_contour_standard_deviation",
    "ioi_histogram",
]


@idyom
@rhythm
@interval
def ioi(starts: list[float]) -> list[float]:
    """The sequence of inter-onset intervals.

    An inter-onset interval (IOI) is the elapsed time from one note onset to the
    next note onset. Unlike note duration, it includes any overlap or silence
    between consecutive notes because it depends only on onset times.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    list[float]
        List of time intervals between consecutive onsets
    """
    return ioi_from_starts(starts)

@idyom
@jsymbolic
@rhythm
@interval
def ioi_mean(starts: list[float]) -> float:
    """
    The arithmetic mean of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Mean of inter-onset intervals

    Note
    ----
    This is called `average_time_between_attacks` in jSymbolic.
    """
    intervals = ioi_from_starts(starts)
    if not intervals:
        return 0.0
    return float(np.mean(intervals))

average_time_between_attacks = ioi_mean

@idyom
@jsymbolic
@rhythm
@interval
def ioi_standard_deviation(starts: list[float]) -> float:
    """
    The standard deviation of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Standard deviation of inter-onset intervals

    Note
    ----
    This is called `variability_of_time_between_attacks` in jSymbolic.
    """
    intervals = ioi_from_starts(starts)
    if not intervals:
        return 0.0
    return float(np.std(intervals, ddof=1))

variability_of_time_between_attacks = ioi_standard_deviation

@idyom
@rhythm
@interval
def ioi_ratio(starts: list[float]) -> list[float]:
    """The sequence of ratios between successive inter-onset intervals.

    First, consecutive onset times are converted to inter-onset intervals (IOIs).
    Each output value is then `IOI[i] / IOI[i - 1]`. Values greater than `1`
    indicate that the current onset gap is longer than the previous one, values
    less than `1` indicate a shorter gap, and `1` indicates no change.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    list[float]
        Sequence of IOI ratios
    """
    intervals = ioi_from_starts(starts)
    if len(intervals) < 2:
        return []

    ratios = [intervals[i] / intervals[i - 1] for i in range(1, len(intervals))]
    return [float(r) for r in ratios]

@novel
@rhythm
@interval
def ioi_ratio_mean(starts: list[float]) -> float:
    """The arithmetic mean of successive inter-onset interval ratios.

    The ratio sequence is computed as `IOI[i] / IOI[i - 1]` for each pair of
    adjacent inter-onset intervals. This summary is above `1` when IOIs tend to
    lengthen, below `1` when they tend to shorten, and close to `1` when
    adjacent IOIs tend to have similar lengths.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Mean of IOI ratios
    """
    ratios = ioi_ratio(starts)
    if not ratios:
        return 0.0
    return float(np.mean(ratios))

@novel
@rhythm
@interval
def ioi_ratio_standard_deviation(starts: list[float]) -> float:
    """The sample standard deviation of successive inter-onset interval ratios.

    This feature measures the variability of `IOI[i] / IOI[i - 1]` across the
    melody. Larger values indicate less regular proportional change between
    neighboring onset gaps.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Standard deviation of IOI ratios
    """
    ratios = ioi_ratio(starts)
    if not ratios:
        return 0.0
    return float(np.std(ratios, ddof=1))

@novel
@rhythm
@interval
def ioi_range(starts: list[float]) -> float:
    """The range of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Range of inter-onset intervals (0.0 if fewer than two onsets)
    """
    intervals = ioi_from_starts(starts)
    if not intervals:
        return 0.0
    return max(intervals) - min(intervals)

@novel
@rhythm
@interval
def ioi_contour(starts: list[float]) -> list[int]:
    """The sequence of IOI-ratio contour values (-1: shorter, 0: same, 1: longer).

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    list[int]
        Sequence of contour values

    Note
    ----
    This contour is computed from ratios of consecutive IOIs, so it requires at
    least three onsets.
    """
    intervals = ioi_from_starts(starts)
    if len(intervals) < 2:
        return []

    ratios = [intervals[i] / intervals[i - 1] for i in range(1, len(intervals))]
    contour = [int(np.sign(ratio - 1)) for ratio in ratios]
    return [int(c) for c in contour]

@novel
@rhythm
@interval
def ioi_contour_mean(starts: list[float]) -> float:
    """The arithmetic mean of ordinal IOI contour values.

    IOI contour values are `-1` for shorter, `0` for unchanged, and `1` for
    longer successive inter-onset intervals. The mean summarizes the balance of
    lengthening versus shortening onset gaps.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Mean of contour values
    """
    contour = ioi_contour(starts)
    if not contour:
        return 0.0
    return float(np.mean(contour))

@novel
@rhythm
@interval
def ioi_contour_standard_deviation(starts: list[float]) -> float:
    """The sample standard deviation of ordinal IOI contour values.

    IOI contour values encode whether successive inter-onset intervals shorten,
    stay the same, or lengthen. This feature measures how variable those ordinal
    changes are across the melody.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Standard deviation of contour values
    """
    contour = ioi_contour(starts)
    if not contour:
        return 0.0
    return float(np.std(contour, ddof=1))

@novel
@rhythm
@interval
def ioi_histogram(starts: list[float]) -> dict:
    """A histogram of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    dict
        Histogram of inter-onset intervals
    """
    intervals = ioi_from_starts(starts)
    num_intervals = len(set(intervals))
    return histogram_bins(intervals, num_intervals)
