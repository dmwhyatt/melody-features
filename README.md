# Melody-Features

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=1023590972)

[![DOI](https://zenodo.org/badge/1023590972.svg)](https://doi.org/10.5281/zenodo.16894207)

[![Tests](https://github.com/dmwhyatt/melody-features/workflows/Tests/badge.svg)](https://github.com/dmwhyatt/melody-features/actions)

[![Coverage](https://codecov.io/gh/dmwhyatt/melody-features/branch/main/graph/badge.svg)](https://codecov.io/gh/dmwhyatt/melody-features)

## Overview
This is a Python package designed to facilitate the use of many different melody analyis tools. 

The main goal of this package is to consolidate a wide range of features from the computational melody analysis literature
into a single place, in a single language.

## Included Contributions

Included in the package are contributions from:

- **FANTASTIC** (Müllensiefen, 2009)
- **SIMILE** (Müllensiefen & Frieler, 2006)
- **melsim** (Silas & Frieler, n.d.)
- **jSymbolic2** (McKay & Fujinaga, 2006)
- **IDyOM** (Pearce, 2005)
- **MIDI Toolbox** (Eerola & Toiviainen, 2004)

## Feature Types

- **Descriptor**: Returns a single scalar value (int, float, bool)
- **Sequence**: Returns a collection (list, tuple, dict, etc.)

<!-- FEATURES_TABLE_START -->

## Melody Features Summary

This table provides an overview of all 231 melody features available in this package.

<div style="overflow-x: auto;">

<table border="1" class="dataframe features-table" id="features-table">
  <thead>
    <tr style="text-align: right;">
      <th>Name</th>
      <th>Pre-existing Implementations</th>
      <th>Further References</th>
      <th>Description</th>
      <th>Type</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Absolute Interval Range</td>
      <td>Fantastic</td>
      <td></td>
      <td>The range between the largest and smallest absolute interval size.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Amount Of Arpeggiation</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of pitch intervals in the melody that constitute triadic movements.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Amount Of Staccato</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of notes with a duration shorter than 0.1 seconds.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Average Interval Span By Melodic Arcs</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The average interval span of melodic arcs.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Average Length Of Melodic Arcs</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The average number of notes that separate peaks and troughs in melodic arcs.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Average Note Duration</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The average note duration in seconds.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Average Time Between Attacks</td>
      <td>Idyom, Jsymbolic</td>
      <td></td>
      <td>The arithmetic mean of inter-onset intervals.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Basic Pitch Histogram</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>A histogram of pitch values within the range of input pitches.</td>
      <td>Sequence</td>
      <td>We use the histogram in the range of input pitches to reduce the output size. An implementation that is truer to the original jSymbolic implementation would return 128 bins (0-127) regardless of how any different pitches are present. However, we believe our approach is more concise and easier to understand for many purposes.</td>
    </tr>
    <tr>
      <td>Chromatic Motion</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of chromatic motion in the melody. Chromatic motion is defined as a melodic interval of 1 semitone.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Combined Strength Of Two Strongest Rhythmic Pulses</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>Sum of the magnitudes of the two strongest rhythmic pulses.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Combined Strength Of Two Strongest Rhythmic Pulses Tempo Standardized</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>Sum of the magnitudes of the two strongest rhythmic pulses using tempo-standardized histogram.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Complebm</td>
      <td>Midi Toolbox</td>
      <td></td>
      <td>Expectancy-based melodic complexity, according to Eerola & North (2000). Calculated using an expectancy-based model that considers pitch patterns, rhythmic features, or both. The complexity score is normalized against the Essen folksong collection, where a score of 5 represents average complexity (standard deviation = 1).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Complete Rests Fraction</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The fraction of the total duration during which no pitched notes are sounding.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Compltrans</td>
      <td>Midi Toolbox</td>
      <td></td>
      <td>The melodic originality measure, according to Simonton (1984). Calculated based on 2nd order pitch-class distribution derived from 15,618 classical music themes. Higher values indicate higher melodic originality (less predictable transitions).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Difference Between Most Common Rhythmic Values</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The absolute difference in bins between most and second most common rhythmic values.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Direction Of Melodic Motion</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of upward melodic motions with regards to the total number of melodic motions.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Distance Between Most Prevalent Melodic Intervals</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The absolute difference between the two most common interval sizes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Dominant Spread</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The longest sequence of pitch classes separated by perfect 5ths that each appear >9% of the time.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Dotted Duration Transitions</td>
      <td>Fantastic</td>
      <td>Steinbeck (1982)</td>
      <td>The proportion of duration transitions that are dotted.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Duration Accent Std</td>
      <td>Novel</td>
      <td></td>
      <td>The standard deviation of duration accents. Duration accent represents the perceptual salience of notes based on their duration, as defined by Parncutt (1994).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Duration Entropy</td>
      <td>Fantastic</td>
      <td></td>
      <td>The zeroth-order base-2 entropy of the duration distribution in quarter notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Duration Histogram</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>A histogram of note durations in quarter notes.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Duration In Seconds</td>
      <td>Fantastic, Jsymbolic</td>
      <td></td>
      <td>The total duration in seconds of the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Duration Range</td>
      <td>Fantastic</td>
      <td></td>
      <td>The range between the longest and shortest note duration in quarter notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Duration Standard Deviation</td>
      <td>Novel</td>
      <td></td>
      <td>The standard deviation of note durations in quarter notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Equal Duration Transitions</td>
      <td>Fantastic</td>
      <td>Steinbeck (1982)</td>
      <td>The proportion of duration transitions that are equal in length.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>First Pitch</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The first pitch number in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>First Pitch Class</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The first pitch class in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Folded Fifths Pitch Class Histogram</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>A histogram of pitch classes arranged according to the circle of fifths.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Get Tempo</td>
      <td>Fantastic, Jsymbolic</td>
      <td></td>
      <td>The first tempo of the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Global Duration</td>
      <td>Fantastic, Jsymbolic</td>
      <td></td>
      <td>The total duration in seconds of the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Gradus</td>
      <td>Midi Toolbox</td>
      <td></td>
      <td>The degree of melodiousness based on Euler's gradus suavitatis (1739).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Half Duration Transitions</td>
      <td>Fantastic</td>
      <td>Steinbeck (1982)</td>
      <td>The proportion of duration transitions that are halved or doubled.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Harmonicity Of Two Strongest Rhythmic Pulses</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The ratio of higher to lower bin index of the two strongest rhythmic pulses.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Harmonicity Of Two Strongest Rhythmic Pulses Tempo Standardized</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The ratio of higher to lower bin index of the two strongest rhythmic pulses (120-BPM standardized histogram).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Honores H</td>
      <td>Fantastic</td>
      <td></td>
      <td>Honoré's H measure corresponds to the observation that the number of tokens occuring exactly once in a sequence is logarithmically related to the total number of tokens in the sequence.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Huron Contour Class Label</td>
      <td>Fantastic</td>
      <td></td>
      <td>The Huron contour classification for the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Importance Of Bass Register</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of pitch numbers in the melody that are between 0 and 54.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Importance Of High Register</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of pitch numbers in the melody that are between 73 and 127.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Importance Of Middle Register</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of pitch numbers in the melody that are between 55 and 72.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Initial Tempo</td>
      <td>Fantastic, Jsymbolic</td>
      <td></td>
      <td>The first tempo of the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Inscale</td>
      <td>Idyom</td>
      <td></td>
      <td>Calculate which pitches are in the estimated key's scale.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Interpolation Contour Class Label</td>
      <td>Fantastic</td>
      <td></td>
      <td>Classify an interpolation contour into gradient categories. The contour is sampled at 4 equally spaced points and each gradient is normalized to units of pitch change per second (scaled to 1 semitone per 0.25 seconds.) The result is then classified into one of 5 categories: - 'a': Strong downward (-2) - normalized gradient <= -1.45 - 'b': Downward (-1) - normalized gradient between -1.45 and -0.45 - 'c': Flat (0) - normalized gradient between -0.45 and 0.45 - 'd': Upward (1) - normalized gradient between 0.45 and 1.45 - 'e': Strong upward (2) - normalized gradient >= 1.45</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Interpolation Contour Direction Changes</td>
      <td>Fantastic</td>
      <td></td>
      <td>The proportion of interpolated gradient values that consistute a change in direction. For instance, a gradient value of -0.5 to 0.25 is a change in direction.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Interpolation Contour Global Direction</td>
      <td>Fantastic</td>
      <td></td>
      <td>The sign of the sum of all contour values.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Interpolation Contour Gradient Std</td>
      <td>Fantastic</td>
      <td></td>
      <td>The standard deviation of the interpolation contour gradients.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Interpolation Contour Mean Gradient</td>
      <td>Fantastic</td>
      <td></td>
      <td>The absolute mean gradient of the interpolation contour.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Interval Between Most Prevalent Pitch Classes</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The number of semitones between the two most prevalent pitch classes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Interval Between Most Prevalent Pitches</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The number of semitones between the two most prevalent pitches.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Interval Direction</td>
      <td>Simile</td>
      <td></td>
      <td>The sequence of interval directions in the melody.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Interval Direction Mean</td>
      <td>Novel</td>
      <td></td>
      <td>The mean of the direction of each interval in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Interval Direction Std</td>
      <td>Novel</td>
      <td></td>
      <td>The standard deviation of the direction of each interval in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Interval Entropy</td>
      <td>Fantastic</td>
      <td></td>
      <td>The zeroth-order base-2 entropy of the interval distribution.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Ioi</td>
      <td>Idyom</td>
      <td></td>
      <td>The time between consecutive onsets (inter-onset interval).</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Ioi Contour</td>
      <td>Novel</td>
      <td></td>
      <td>The sequence of IOI contour values (-1: shorter, 0: same, 1: longer).</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Ioi Contour Mean</td>
      <td>Novel</td>
      <td></td>
      <td>The arithmetic mean of IOI contour values.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Ioi Contour Standard Deviation</td>
      <td>Novel</td>
      <td></td>
      <td>The standard deviation of IOI contour values.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Ioi Histogram</td>
      <td>Novel</td>
      <td></td>
      <td>A histogram of inter-onset intervals.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Ioi Mean</td>
      <td>Idyom, Jsymbolic</td>
      <td></td>
      <td>The arithmetic mean of inter-onset intervals.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Ioi Range</td>
      <td>Novel</td>
      <td></td>
      <td>The range of inter-onset intervals.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Ioi Ratio</td>
      <td>Idyom</td>
      <td></td>
      <td>The sequence of inter-onset interval ratios.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Ioi Ratio Mean</td>
      <td>Novel</td>
      <td></td>
      <td>The arithmetic mean of inter-onset interval ratios.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Ioi Ratio Standard Deviation</td>
      <td>Novel</td>
      <td></td>
      <td>The standard deviation of inter-onset interval ratios.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Ioi Standard Deviation</td>
      <td>Idyom, Jsymbolic</td>
      <td></td>
      <td>The standard deviation of inter-onset intervals.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Ivdirdist1</td>
      <td>Midi Toolbox</td>
      <td></td>
      <td>The proportion of upward intervals for each interval size (1-12 semitones). Returns the proportion of upward intervals for each interval size in the melody as a dictionary mapping interval sizes to their directional bias values.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Ivdist1</td>
      <td>Midi Toolbox</td>
      <td></td>
      <td>The distribution of intervals in the melody, weighted by their durations.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Ivsizedist1</td>
      <td>Midi Toolbox</td>
      <td></td>
      <td>The distribution of interval sizes (0-12 semitones). Returns the distribution of interval sizes by combining upward and downward intervals of the same absolute size. The first component represents a unison (0) and the last component represents an octave (12).</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Last Pitch</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The last pitch number in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Last Pitch Class</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The last pitch class in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Length</td>
      <td>Fantastic</td>
      <td></td>
      <td>The total number of notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Longest Complete Rest</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The longest uninterrupted complete rest in quarter-note units (ignoring rests shorter than 0.1 QN).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Longest Conjunct Scalar Passage</td>
      <td>Novel</td>
      <td></td>
      <td>Calculate the longest conjunct scalar passage.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Longest Monotonic Conjunct Scalar Passage</td>
      <td>Novel</td>
      <td></td>
      <td>Calculate the longest monotonic conjunct scalar passage.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Longest Rhythmic Value</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The longest rhythmic value (in quarter notes) among non-empty bins.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Max Log Df</td>
      <td>Fantastic</td>
      <td></td>
      <td>Calculate maximum log document frequency across all n-grams.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Maximum Note Duration</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The maximum note duration in seconds.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Absolute Interval</td>
      <td>Fantastic, Jsymbolic</td>
      <td></td>
      <td>The arithmetic mean of the absolute intervals in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Complete Rest Duration</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The mean duration of complete rests in quarter-note units (ignoring rests shorter than 0.1 QN).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Duration</td>
      <td>Novel</td>
      <td></td>
      <td>The mean note duration in quarter notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Duration Accent</td>
      <td>Midi Toolbox</td>
      <td></td>
      <td>The mean duration accent across all notes. Duration accent represents the perceptual salience of notes based on their duration, as defined by Parncutt (1994).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Entropy</td>
      <td>Fantastic</td>
      <td></td>
      <td>Calculate the zeroth-order base-2 entropy of m-types across all n-gram lengths.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Global Local Weight</td>
      <td>Fantastic</td>
      <td></td>
      <td>Calculate mean global-local weight using inverse entropy weighting.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Global Weight</td>
      <td>Fantastic</td>
      <td></td>
      <td>Calculate mean global weight using inverse entropy weighting.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Log Df</td>
      <td>Fantastic</td>
      <td></td>
      <td>Calculate mean log document frequency across all n-grams.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Log Tfdf</td>
      <td>Fantastic</td>
      <td></td>
      <td>Calculate mean log TF-DF score across all n-grams.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Melodic Accent</td>
      <td>Novel</td>
      <td></td>
      <td>The arithmetic mean of the melodic accent values across all notes. Melodic accent is defined by Thomassen's model (1982) according to the possible melodic contours arising in 3-pitch windows.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Melodic Attraction</td>
      <td>Novel</td>
      <td></td>
      <td>The arithmetic mean of the melodic attraction values across all notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Melodic Interval</td>
      <td>Fantastic, Jsymbolic</td>
      <td></td>
      <td>The arithmetic mean of the absolute intervals in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Mobility</td>
      <td>Novel</td>
      <td></td>
      <td>The arithmetic mean of the mobility values across all notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Pitch</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The arithmetic mean of the pitch numbers in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Pitch Class</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The arithmetic mean of the pitch classes in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Productivity</td>
      <td>Fantastic</td>
      <td></td>
      <td>Mean productivity is defined as the mean of the number of types occurring only once divided by the total number of tokens. The types occurring only once in a sequence are known as hapax legomena.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Rhythmic Value</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The mean rhythmic value (in quarter notes) using the normalized histogram, weighted by the frequency of the rhythmic value.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Rhythmic Value Offset</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The mean quantized offset from the nearest ideal rhythmic value (in quarter notes).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Rhythmic Value Run Length</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The mean run length of identical rhythmic values across the melody. Run length is the number of consecutive notes with the same rhythmic value. Returns 0.0 if there are fewer than 1 notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Tempo</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The mean tempo of the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mean Tessitura</td>
      <td>Novel</td>
      <td></td>
      <td>The arithmetic mean of the sequence of tessitura values.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Median Complete Rest Duration</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The median duration of complete rests in quarter-note units (ignoring rests shorter than 0.1 QN).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Median Rhythmic Value Offset</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The median quantized offset from the nearest ideal rhythmic value (in quarter notes).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Median Rhythmic Value Run Length</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The median run length of identical rhythmic values across the melody. Run length is the number of consecutive notes with the same rhythmic value.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Melodic Accent Std</td>
      <td>Novel</td>
      <td></td>
      <td>The standard deviation of the melodic accent values across all notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Melodic Attraction</td>
      <td>Midi Toolbox</td>
      <td></td>
      <td>The melodic attraction according to Lerdahl (1996). Each tone in a key has certain anchoring strength ("weight") in tonal pitch space. Melodic attraction strength is affected by the distance between tones and directed motion patterns.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Melodic Attraction Std</td>
      <td>Novel</td>
      <td></td>
      <td>The standard deviation of the melodic attraction values across all notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Melodic Embellishment</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of melodic embellishments in the melody. Melodic embellishments are identified by notes that are surrounded on both sides by notes with durations at least 3 times longer than the central note.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Melodic Interval Histogram</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>A histogram of interval sizes.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Melodic Large Intervals</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of intervals >= 13 semitones.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Melodic Octaves</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of intervals that are octaves (12 semitones).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Melodic Perfect Fifths</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of intervals that are perfect fifths (7 semitones).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Melodic Perfect Fourths</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of intervals that are perfect fourths (5 semitones).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Melodic Pitch Variety</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The average number of notes that pass before a pitch is repeated.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Melodic Sevenths</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of intervals that are sevenths (10 or 11 semitones).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Melodic Sixths</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of intervals that are sixths (8 or 9 semitones).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Melodic Thirds</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of intervals that are thirds (3 or 4 semitones).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Melodic Tritones</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of intervals that are tritones (6 semitones).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Min Log Df</td>
      <td>Fantastic</td>
      <td></td>
      <td>Calculate minimum log document frequency across all n-grams.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Minimum Note Duration</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The minimum note duration in seconds.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Minor Major Third Ratio</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The ratio of minor thirds to major thirds.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mobility</td>
      <td>Midi Toolbox</td>
      <td></td>
      <td>The melodic mobility for each note based on von Hippel (2000). Mobility describes why melodies change direction after large skips by observing that they would otherwise run out of the comfortable melodic range. It uses lag-one autocorrelation between successive pitch heights.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Mobility Std</td>
      <td>Novel</td>
      <td></td>
      <td>The standard deviation of the mobility values across all notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Modal Duration</td>
      <td>Fantastic, Jsymbolic</td>
      <td></td>
      <td>The most common note duration in quarter notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Modal Interval</td>
      <td>Fantastic, Jsymbolic</td>
      <td></td>
      <td>The most common interval size in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Mode</td>
      <td>Fantastic</td>
      <td></td>
      <td>Calculate the mode (major/minor) of a melody.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Most Common Interval</td>
      <td>Fantastic, Jsymbolic</td>
      <td></td>
      <td>The most common interval size in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Most Common Pitch</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The most frequently occurring pitch number in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Most Common Pitch Class</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The most frequently occurring pitch class in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Most Common Rhythmic Value</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The modal rhythmic value (in quarter notes).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Narmour Closure</td>
      <td>Idyom, Midi Toolbox</td>
      <td>Narmour (1990)</td>
      <td>A score of 1 is given if the last three notes in a melody constitute a change in direction. Another score of 1 is given if the final interval is more than one tone smaller than the penultimate. As such, this returns integer values between 0 and 2.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Narmour Intervallic Difference</td>
      <td>Idyom, Midi Toolbox</td>
      <td>Narmour (1990)</td>
      <td>If a large interval is followed by a smaller interval, returns 1 if either: - The smaller interval continues in the same direction and is at least 3 semitones smaller - The smaller interval changes direction and is at least 2 semitones smaller Additionally, returns 1 if a small interval is followed by another interval of the same size. Otherwise returns 0.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Narmour Proximity</td>
      <td>Idyom, Midi Toolbox</td>
      <td>Narmour (1990)</td>
      <td>Proximity is defined as 6 minus the absolute interval between the last two notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Narmour Registral Direction</td>
      <td>Idyom, Midi Toolbox</td>
      <td>Narmour (1990)</td>
      <td>The score is set to zero. If an interval greater than a perfect fifth is followed by a direction change, a score of 1 is given. If an interval smaller than a perfect fourth continues in the same direction, a score of 1 is given. This feature returns either 0 or 1 accordingly.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Narmour Registral Return</td>
      <td>Idyom, Midi Toolbox</td>
      <td>Narmour (1990)</td>
      <td>If the last three notes move away from and then back to the same pitch, a score of 3 is returned. If the pitch returned to is 1 semitone away from the initial, returns 2. If the pitch returned to is 2 semitones away from the initial, returns 1. Otherwise, a score of 0 is returned.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Norm Log Dist</td>
      <td>Fantastic</td>
      <td></td>
      <td>Calculate normalized log distance between TF and DF distributions.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Note Density</td>
      <td>Fantastic, Jsymbolic</td>
      <td></td>
      <td>The average number of notes per second.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Note Density Per Quarter Note</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The average number of note onsets per unit of time corresponding to an idealized quarter note duration based on the tempo.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Note Density Per Quarter Note Variability</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The standard deviation of note density per quarter note. Divides the melody into 8-quarter-note windows and calculates the standard deviation of note density across these windows.</td>
      <td>Descriptor</td>
      <td>Our tests indicate a certain discrepancy between our outputs and JSymbolic's outputs, which may be a consequence of JSymbolic's tick-based approach, or perhaps its idiosyncratic windowing approach.</td>
    </tr>
    <tr>
      <td>Note Density Variability</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The standard deviation of note density across 5-second windows.</td>
      <td>Descriptor</td>
      <td>Our tests indicate a certain discrepancy between our outputs and JSymbolic's outputs, which may be a consequence of JSymbolic's tick-based approach, or perhaps its idiosyncratic windowing approach.</td>
    </tr>
    <tr>
      <td>Npvi</td>
      <td>Midi Toolbox</td>
      <td></td>
      <td>The normalized Pairwise Variability Index (nPVI) of note durations in quarter notes. The nPVI measures the durational variability of events, originally developed for language research to distinguish stress-timed vs. syllable-timed languages. Applied to music by Patel & Daniele (2003) to study the prosodic influences on musical rhythm.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Number Of Common Melodic Intervals</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The number of intervals that appear in at least 9% of melodic transitions.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Number Of Common Pitches</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The number of unique pitch numbers that appear in at least 9% of total notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Number Of Common Pitches Classes</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The number of pitch classes that appear in at least 20% of total notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Number Of Common Rhythmic Values Present</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The number of rhythmic value bins with normalized proportion >= 0.15.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Number Of Different Rhythmic Values Present</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The number of distinct rhythmic value bins that are present in the melody (non-zero).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Number Of Moderate Rhythmic Pulses</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The number of beat histogram peaks with normalized magnitudes over 0.01.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Number Of Moderate Rhythmic Pulses Tempo Standardized</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The number of tempo-standardized beat histogram peaks with normalized magnitudes over 0.01.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Number Of Relatively Strong Rhythmic Pulses</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The number of peaks at least 30% of the max magnitude.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Number Of Relatively Strong Rhythmic Pulses Tempo Standardized</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The number of tempo-standardized peaks at least 30% of the max magnitude.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Number Of Strong Rhythmic Pulses</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The count of BPM bins with pulses greater than 0.001.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Number Of Strong Rhythmic Pulses Tempo Standardized</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The count of BPM bins with pulses greater than 0.001 using the tempo-standardized beat histogram.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Number Of Unique Durations</td>
      <td>Novel</td>
      <td></td>
      <td>The number of unique note durations.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Number Of Unique Pitch Classes</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The number of unique pitch classes in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Number Of Unique Pitches</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The number of unique pitch numbers in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Onset Autocorr Peak</td>
      <td>Novel</td>
      <td></td>
      <td>The maximum onset autocorrelation value (excluding lag 0).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Onset Autocorrelation</td>
      <td>Midi Toolbox</td>
      <td></td>
      <td>The autocorrelation function of onset times weighted by duration accents. This is calculated by weighting the onset times by the duration accents, as defined by Parncutt (1994).</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Pcdist1</td>
      <td>Midi Toolbox</td>
      <td></td>
      <td>The distribution of pitch classes in the melody, weighted by the duration of the notes.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Pitch Class Kurtosis</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The sample excess kurtosis of the pitch class histogram.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Pitch Class Kurtosis After Folding</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The sample excess kurtosis of the pitch class histogram, after arranging the pitch classes according to the circle of fifths.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Pitch Class Skewness</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The skewness of the pitch class histogram, using Pearson's median skewness formula.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Pitch Class Skewness After Folding</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The skewness of the pitch class histogram, using Pearson's median skewness formula, after arranging the pitch classes according to the circle of fifths.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Pitch Class Variability</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>Standard deviation of all pitch classes in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Pitch Class Variability After Folding</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>Standard deviation of all pitch classes after arranging the pitch classes by perfect fifths. Provides a measure of how close the pitch classes are as a whole from the mean pitch class from a dominant-tonic perspective.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Pitch Entropy</td>
      <td>Fantastic</td>
      <td></td>
      <td>The zeroth-order base-2 entropy of the pitch distribution.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Pitch Interval</td>
      <td>Simile</td>
      <td></td>
      <td>The intervals (in semitones) between consecutive pitches in the melody.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Pitch Kurtosis</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The sample excess kurtosis of the pitch histogram.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Pitch Range</td>
      <td>Fantastic, Jsymbolic</td>
      <td></td>
      <td>Subtract the lowest pitch number in the melody from the highest.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Pitch Skewness</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The skewness of the pitch histogram, using Pearson's median skewness formula.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Pitch Standard Deviation</td>
      <td>Fantastic, Jsymbolic</td>
      <td></td>
      <td>Standard deviation of all pitch numbers in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Pitch Variability</td>
      <td>Fantastic, Jsymbolic</td>
      <td></td>
      <td>Standard deviation of all pitch numbers in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Polynomial Contour Coefficients</td>
      <td>Fantastic</td>
      <td></td>
      <td>The first 3 non-constant coefficients of the polynomial contour.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Polyrhythms</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The fraction of beat histogram peaks that are not integer multiples/factors of the highest peak.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Polyrhythms Tempo Standardized</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The fraction of beat histogram peaks that are not integer multiples/factors of the highest peak using tempo-standardized histogram.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Prevalence Of Dotted Notes</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The sum of dotted rhythmic bins: 3, 5, 7, 9, 11.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Prevalence Of Long Rhythmic Values</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The sum of rhythmic bins 6 to 11 (half notes to dotted double whole notes or more).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Prevalence Of Medium Rhythmic Values</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The sum of rhythmic bins 2 to 6 (8th notes to half notes).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Prevalence Of Most Common Melodic Interval</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of intervals that are the most common interval.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Prevalence Of Most Common Pitch</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of pitches that are the most common pitch with regards to the total number of pitches in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Prevalence Of Most Common Pitch Class</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of pitch classes that are the most common pitch class with regards to the total number of pitch classes in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Prevalence Of Most Common Rhythmic Value</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion (0.0 - 1.0) of the modal rhythmic bin.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Prevalence Of Short Rhythmic Values</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The sum of the three shortest rhythmic bins (indexes 0, 1, and 2).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Prevalence Of Very Long Rhythmic Values</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The sum of rhythmic bins 9 to 11 (dotted whole notes to dotted double whole notes or more).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Prevalence Of Very Short Rhythmic Values</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The sum of the two shortest rhythmic bins (indexes 0 and 1).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Proportion Conjunct Scalar</td>
      <td>Novel</td>
      <td></td>
      <td>Calculate the proportion of conjunct scalar motion.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Proportion Scalar</td>
      <td>Novel</td>
      <td></td>
      <td>Calculate the proportion of scalar motion.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Range Of Rhythmic Values</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The range of rhythmic values located within the 12-bin PPQN-based histogram. Durations are converted to quarter notes and mapped to 12 fixed rhythmic bins using midpoints. The returned value is the difference between the highest and lowest non-empty bins.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Referent</td>
      <td>Idyom</td>
      <td></td>
      <td>Calculate the referent (root note) of a melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Relative Prevalence Of Most Common Melodic Intervals</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The ratio of the frequency of the second most common interval to the frequency of the most common interval.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Relative Prevalence Of Most Common Rhythmic Values</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The ratio of the second-most-common rhythmic bin to the most common bin.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Relative Prevalence Of Top Pitch Classes</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The ratio of the frequency of the second most common pitch class to the frequency of the most common pitch class.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Relative Prevalence Of Top Pitches</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The ratio of the frequency of the second most common pitch to the frequency of the most common pitch.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Repeated Notes</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of repeated notes in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Rhythmic Looseness</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The average width of beat histogram peaks. Width is defined as the distance between points at 30% of the peak height.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Rhythmic Looseness Tempo Standardized</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The average width of beat histogram peaks using tempo-standardized histogram. Width is defined as the distance between points at 30% of the peak height.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Rhythmic Variability</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The standard deviation of the beat histogram bin magnitudes, excluding the first 40 bins.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Rhythmic Variability Tempo Standardized</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The standard deviation of the tempo-standardized beat histogram bin magnitudes, excluding the first 40 bins.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Second Strongest Rhythmic Pulse</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The bin index (BPM) of the second-highest magnitude in the beat histogram.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Second Strongest Rhythmic Pulse Tempo Standardized</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The bin index (BPM) of the second-highest magnitude in the 120-BPM standardized beat histogram.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Shortest Rhythmic Value</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The shortest rhythmic value (in quarter notes) among non-empty bins.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Sichels S</td>
      <td>Fantastic</td>
      <td></td>
      <td>Sichel's S measure corresponds to the proportion of m-types that occur exactly twice in a sequence. Higher values indicate a greater amount of m-types that occur exactly twice.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Simpsons D</td>
      <td>Fantastic</td>
      <td></td>
      <td>Simpson's D measure of diversity. This feature measures the rate of m-type repetition in a similar way to Yule's K.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Standard Deviation Absolute Interval</td>
      <td>Fantastic</td>
      <td></td>
      <td>The standard deviation of the absolute intervals in the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Std Global Local Weight</td>
      <td>Fantastic</td>
      <td></td>
      <td>Calculate standard deviation of global-local weight using inverse entropy weighting.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Std Global Weight</td>
      <td>Fantastic</td>
      <td></td>
      <td>Calculate standard deviation of global weight using inverse entropy weighting.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Step Contour Global Direction</td>
      <td>Fantastic</td>
      <td></td>
      <td>The correlation between the step contour vector and an ascending linear function y = x.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Step Contour Global Variation</td>
      <td>Fantastic</td>
      <td></td>
      <td>The standard deviation of the step contour vector.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Step Contour Local Variation</td>
      <td>Fantastic</td>
      <td></td>
      <td>The mean absolute difference between adjacent values of the step contour vector.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Stepwise Motion</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The proportion of stepwise motion in the melody. Stepwise motion is defined as a melodic interval of 1 or 2 semitones.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Strength Of Second Strongest Rhythmic Pulse</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The magnitude of the beat histogram bin with the second-highest magnitude.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Strength Of Second Strongest Rhythmic Pulse Tempo Standardized</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The magnitude of the tempo-standardized beat histogram bin with the second-highest magnitude.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Strength Of Strongest Rhythmic Pulse</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The magnitude of the beat histogram bin with the highest magnitude.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Strength Of Strongest Rhythmic Pulse Tempo Standardized</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The magnitude of the tempo-standardized beat histogram bin with the highest magnitude.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Strength Ratio Of Two Strongest Rhythmic Pulses</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>Ratio of the magnitude of the strongest to second-strongest rhythmic pulse.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Strength Ratio Of Two Strongest Rhythmic Pulses Tempo Standardized</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>Ratio of the magnitude of the strongest to second-strongest rhythmic pulse (120-BPM standardized histogram).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Strong Tonal Centres</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The number of isolated peaks in the pitch class histogram that each account for at least 9% of notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Strongest Rhythmic Pulse</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The bin index (BPM) of the maximum beat histogram magnitude.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Strongest Rhythmic Pulse Tempo Standardized</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The bin index (BPM) of the maximum in the 120-BPM standardized beat histogram.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Tempo Variability</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The variability of tempo of the melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Tessitura</td>
      <td>Midi Toolbox</td>
      <td>von Hippel, C. (2000).</td>
      <td>Tessitura is based on standard deviation from median pitch height. The median range of the melody tends to be favoured and thus more expected. Tessitura predicts whether listeners expect tones close to median pitch height. Higher tessitura values correspond to melodies that have a wider range of pitches.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Tessitura Std</td>
      <td>Novel</td>
      <td></td>
      <td>The standard deviation of the sequence of tessitura values.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Tfdf Kendall</td>
      <td>Fantastic</td>
      <td></td>
      <td>Calculate Kendall's tau correlation between term frequency and document frequency.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Tfdf Spearman</td>
      <td>Fantastic</td>
      <td></td>
      <td>Calculate Spearman correlation between term frequency and document frequency.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Tonal Clarity</td>
      <td>Fantastic</td>
      <td></td>
      <td>Calculate the tonal clarity of a melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Tonal Entropy</td>
      <td>Novel</td>
      <td></td>
      <td>Calculate the tonal entropy of a melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Tonal Spike</td>
      <td>Fantastic</td>
      <td></td>
      <td>Calculate the tonal spike of a melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Tonalness</td>
      <td>Fantastic</td>
      <td></td>
      <td>Calculate the tonalness of a melody.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Tonalness Histogram</td>
      <td>Novel</td>
      <td>Krumhansl (1990)</td>
      <td>A histogram of Krumhansl-Schmuckler correlation values.</td>
      <td>Sequence</td>
      <td></td>
    </tr>
    <tr>
      <td>Total Number Of Notes</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The total number of notes.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Variability In Rhythmic Value Run Lengths</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The standard deviation of rhythmic value run lengths. Run length is the number of consecutive notes with the same rhythmic value.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Variability Of Complete Rest Durations</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The standard deviation of complete rest durations in quarter notes (ignoring rests shorter than 0.1 QN).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Variability Of Note Durations</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The standard deviation of note durations in seconds.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Variability Of Rhythmic Value Offsets</td>
      <td>Jsymbolic</td>
      <td></td>
      <td>The standard deviation of rhythmic value offsets (in quarter notes).</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Variability Of Time Between Attacks</td>
      <td>Idyom, Jsymbolic</td>
      <td></td>
      <td>The standard deviation of inter-onset intervals.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
    <tr>
      <td>Yules K</td>
      <td>Fantastic</td>
      <td></td>
      <td>Yule's K measure of lexical richness. This feature measures the rate at which m-types are repeated in a sequence. Higher values indicate more repetitive sequences.</td>
      <td>Descriptor</td>
      <td></td>
    </tr>
  </tbody>
</table>

</div>

<!-- FEATURES_TABLE_END -->

(This package is strictly for monophonic melodies - it will not compute any features for polyphonic music!)

## Installation

```bash
# Clone the repository
git clone https://github.com/dmwhyatt/melody-features.git
cd melody-features

# Install in development mode
pip install -e .
```

## Quick Start

The feature set can be easily accessed using the top-level function `get_all_features`. Here's a basic example:

```python
from melody_features import get_all_features

# Extract features from a directory of MIDI files, a single MIDI file
# or a list of paths to MIDI files
results = get_all_features(input="path/to/your/midi/files")

# Print the result of all feature calculations
print(results.iloc[:1,].to_json(indent=4, orient="records"))

```

By default, this function will produce a Pandas DataFrame containing the tabulated features, using the Essen Folksong Collection as the reference corpus.


This function can be customised in a number of ways, please see `notebooks/example.ipynb` for a detailed breakdown.

## Melsim

Melsim is an R package for computing the similarity between two or more melodies. It is currently under development by Seb Silas and Klaus Frieler (https://github.com/sebsilas/melsim)

It is included with this feature set through a wrapper approach - take a look at example.py and the supplied MIDI files.

Since calculating similarities is highly modular in Melsim, we leave the user to decide how they wish to construct comparisons. Melsim is not run as part of the `get_all_features` function.

### Available Corpora

The package comes with an example corpus, a MIDI conversion of the well-known Essen Folksong Collection (Eck, 2024; Schaffrath, 1995).

## Development

### Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Run specific test suites
python -m pytest tests/test_module_setup.py -v
python -m pytest tests/test_corpus_import.py -v
python -m pytest tests/test_idyom_setup.py -v
```

## Contributing

Contributions are welcomed, though this project is likely to be migrated into AMADS in the future...

See https://github.com/music-computing/amads

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
