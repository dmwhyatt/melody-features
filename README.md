# Melody-Features

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=1023590972)

[![DOI](https://zenodo.org/badge/1023590972.svg)](https://doi.org/10.5281/zenodo.16894207)

[![Tests](https://github.com/dmwhyatt/melody-features/workflows/Tests/badge.svg)](https://github.com/dmwhyatt/melody-features/actions)

## Overview
This is a Python package designed to facilitate the use of many different melody analyis tools. 

The main goal of this package is to consolidate a wide range of features from the computational melody analysis literature
into a single place, in a single language.

Included in the package are contributions from:

- **FANTASTIC** (Müllensiefen, 2009)
- **SIMILE** (Müllensiefen & Frieler, 2006)
- **melsim** (Silas & Frieler, n.d.)
- **jSymbolic2** (McKay & Fujinaga, 2006)
- **IDyOM** (Pearce, 2005)
- **MIDI Toolbox** (Eerola & Toiviainen, 2004)

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

By default, this function will produce a Pandas DataFrame containing the following features, using the Essen Folksong Collection as the reference corpus.

```json
[
    {
        "melody_num":1,
        "melody_id":"appenzel.mid",
        "pitch_features.pitch_range":24,
        "pitch_features.pitch_standard_deviation":5.2138160258,
        "pitch_features.pitch_entropy":3.2592987841,
        "pitch_features.pcdist1":"{0.0: 0.08796296296296297, 2.0: 0.2361111111111111, 4.0: 0.037037037037037035, 6.0: 0.018518518518518517, 7.0: 0.19907407407407407, 9.0: 0.17592592592592593, 11.0: 0.24537037037037035}",
        "pitch_features.basic_pitch_histogram":"{'62.00-63.85': 9, '63.85-65.69': 3, '65.69-67.54': 14, '67.54-69.38': 27, '69.38-71.23': 25, '71.23-73.08': 13, '73.08-74.92': 23, '74.92-76.77': 5, '76.77-78.62': 0, '78.62-80.46': 9, '80.46-82.31': 2, '82.31-84.15': 9, '84.15-86.00': 1}",
        "pitch_features.mean_pitch":71.7571428571,
        "pitch_features.most_common_pitch":69,
        "pitch_features.number_of_pitches":13,
        "pitch_features.melodic_pitch_variety":0.9071428571,
        "pitch_features.dominant_spread":1,
        "pitch_features.folded_fifths_pitch_class_histogram":"{0: 13.0, 7: 19.0, 2: 33.0, 9: 29.0, 4: 8.0, 11: 34.0, 6: 4.0}",
        "pitch_features.pitch_class_kurtosis_after_folding":-1.1360805712,
        "pitch_features.pitch_class_skewness_after_folding":-0.0770887536,
        "pitch_features.pitch_class_variability_after_folding":3.5799897389,
        "interval_features.pitch_interval":"[0, 5, 0, 4, 0, 0, 3, 2, -2, -3, 0, 0, 3, -2, -3, 0, 0, 2, 8, 4, -21, 0, 5, 0, 2, 2, 0, 1, 2, 2, -2, -3, 0, 0, 3, -2, -3, 0, 0, 2, -4, 2, 0, 0, -2, -1, -2, -2, 7, 0, -2, -1, -2, -2, 7, 0, 5, -8, 3, 0, -2, -3, 2, -4, 0, 0, 5, 0, 4, -2, 2, 1, 2, 2, -2, -3, 1, -1, 3, -2, -3, 0, 0, 2, 8, 4, -12, 0, 3, -2, -3, 0, 0, 2, 0, 3, -2, -3, 3, 9, -12, 3, 4, -2, -3, 12, -4, -8, 3, -2, -3, 3, 9, -12, 3, 4, -2, -3, 12, -4, -5, 9, -4, -5, 9, -9, 5, -5, 9, -9, 5, -5, 9, -9, 5, -5, 12, -3, -4]",
        "interval_features.absolute_interval_range":21,
        "interval_features.mean_absolute_interval":3.4172661871,
        "interval_features.modal_interval":0,
        "interval_features.interval_entropy":3.7118533438,
        "interval_features.ivdist1":"{-21.0: 0.009478672985781991, -12.0: 0.018957345971563982, -9.0: 0.014218009478672987, -8.0: 0.04265402843601896, -5.0: 0.07582938388625593, -4.0: 0.047393364928909956, -3.0: 0.0995260663507109, -2.0: 0.1137440758293839, -1.0: 0.014218009478672987, 0.0: 0.24170616113744078, 1.0: 0.018957345971563982, 2.0: 0.08530805687203792, 3.0: 0.05687203791469195, 4.0: 0.047393364928909956, 5.0: 0.04265402843601896, 7.0: 0.009478672985781991, 8.0: 0.018957345971563982, 9.0: 0.028436018957345974, 12.0: 0.014218009478672987}",
        "interval_features.interval_direction_mean":0.0,
        "interval_features.interval_direction_sd":0.8895880544,
        "interval_features.average_interval_span_by_melodic_arcs":7.2,
        "interval_features.distance_between_most_prevalent_melodic_intervals":2.0,
        "interval_features.melodic_interval_histogram":"{'-21.00--20.00': 1, '-20.00--19.00': 0, '-19.00--18.00': 0, '-18.00--17.00': 0, '-17.00--16.00': 0, '-16.00--15.00': 0, '-15.00--14.00': 0, '-14.00--13.00': 0, '-13.00--12.00': 0, '-12.00--11.00': 3, '-11.00--10.00': 0, '-10.00--9.00': 0, '-9.00--8.00': 3, '-8.00--7.00': 2, '-7.00--6.00': 0, '-6.00--5.00': 0, '-5.00--4.00': 5, '-4.00--3.00': 6, '-3.00--2.00': 13, '-2.00--1.00': 19, '-1.00-0.00': 3, '0.00-1.00': 29, '1.00-2.00': 3, '2.00-3.00': 14, '3.00-4.00': 12, '4.00-5.00': 6, '5.00-6.00': 7, '6.00-7.00': 0, '7.00-8.00': 2, '8.00-9.00': 2, '9.00-10.00': 6, '10.00-11.00': 0, '11.00-12.00': 3}",
        "interval_features.melodic_large_intervals":0.0071942446,
        "interval_features.variable_melodic_intervals":0.1582733813,
        "interval_features.number_of_common_melodic_intervals":4,
        "interval_features.prevalence_of_most_common_melodic_interval":0.2086330935,
        "contour_features.step_contour_global_variation":5.1382973454,
        "contour_features.step_contour_global_direction":0.5462686601,
        "contour_features.step_contour_local_variation":3.8888888889,
        "contour_features.interpolation_contour_global_direction":1,
        "contour_features.interpolation_contour_mean_gradient":16.8087145969,
        "contour_features.interpolation_contour_gradient_std":25.0335871585,
        "contour_features.interpolation_contour_direction_changes":1.0,
        "contour_features.interpolation_contour_class_label":"edea",
        "duration_features.tempo":100,
        "duration_features.duration_range":0.625,
        "duration_features.modal_duration":0.125,
        "duration_features.mean_duration":0.1875,
        "duration_features.duration_standard_deviation":0.1113031574,
        "duration_features.number_of_durations":5,
        "duration_features.global_duration":26.75,
        "duration_features.note_density":5.2336448598,
        "duration_features.duration_entropy":1.2113270711,
        "duration_features.length":140,
        "duration_features.ioi_mean":0.1888489209,
        "duration_features.ioi_std":0.1184117261,
        "duration_features.ioi_ratio_mean":1.2070048309,
        "duration_features.ioi_ratio_std":0.9106849442,
        "duration_features.ioi_contour_mean":0.0,
        "duration_features.ioi_contour_std":0.6593804734,
        "duration_features.ioi_range":0.625,
        "duration_features.ioi_histogram":"{'0.12-0.25': 88, '0.25-0.38': 45, '0.38-0.50': 0, '0.50-0.62': 1, '0.62-0.75': 5}",
        "duration_features.duration_histogram":"{'0.12-0.25': 88, '0.25-0.38': 46, '0.38-0.50': 0, '0.50-0.62': 2, '0.62-0.75': 4}",
        "tonality_features.tonalness":0.7537489436,
        "tonality_features.tonal_clarity":1.010865164,
        "tonality_features.tonal_spike":0.089431857,
        "tonality_features.tonal_entropy":4.5849625007,
        "tonality_features.referent":2,
        "tonality_features.inscale":0,
        "tonality_features.temperley_likelihood":4.965908428e-196,
        "tonality_features.longest_monotonic_conjunct_scalar_passage":5,
        "tonality_features.longest_conjunct_scalar_passage":6,
        "tonality_features.proportion_conjunct_scalar":0.0428571429,
        "tonality_features.proportion_scalar":0.0357142857,
        "tonality_features.tonalness_histogram":"{'0.25-0.30': 0, '0.30-0.34': 0, '0.34-0.38': 0, '0.38-0.42': 0, '0.42-0.46': 0, '0.46-0.50': 0, '0.50-0.55': 0, '0.55-0.59': 0, '0.59-0.63': 0, '0.63-0.67': 0, '0.67-0.71': 0, '0.71-0.75': 0, '0.75-0.80': 1, '0.80-0.84': 0, '0.84-0.88': 0, '0.88-0.92': 0, '0.92-0.96': 0, '0.96-1.00': 0, '1.00-1.05': 0, '1.05-1.09': 0, '1.09-1.13': 0, '1.13-1.17': 0, '1.17-1.21': 0, '1.21-1.25': 0}",
        "narmour_features.registral_direction":1.0,
        "narmour_features.proximity":2.0,
        "narmour_features.closure":0.0,
        "narmour_features.registral_return":0.0,
        "narmour_features.intervallic_difference":1.0,
        "melodic_movement_features.amount_of_arpeggiation":0.5323741007,
        "melodic_movement_features.chromatic_motion":0.0431654676,
        "melodic_movement_features.melodic_embellishment":0.0214285714,
        "melodic_movement_features.repeated_notes":0.2086330935,
        "melodic_movement_features.stepwise_motion":0.2805755396,
        "mtype_features.yules_k":25.642516485,
        "mtype_features.simpsons_d":0.0037852529,
        "mtype_features.sichels_s":0.1789077213,
        "mtype_features.honores_h":2526.2359209404,
        "mtype_features.mean_entropy":6.0212773876,
        "mtype_features.mean_productivity":0.7344632768,
        "corpus_features.tfdf_spearman":0.3090724238,
        "corpus_features.tfdf_kendall":0.2543242062,
        "corpus_features.mean_log_tfdf":0.0026511207,
        "corpus_features.norm_log_dist":0.0110920568,
        "corpus_features.max_log_df":8.9987547695,
        "corpus_features.min_log_df":0.6931471806,
        "corpus_features.mean_log_df":3.1942335627,
        "idyom_default_pitch_features.mean_information_content":3.719061
    }
]
```

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
