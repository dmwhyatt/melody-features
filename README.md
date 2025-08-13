# Melodic Feature Set

[![Tests](https://github.com/Dav8Circle/melodic_feature_set/workflows/Test/badge.svg)](https://github.com/Dav8Circle/melodic_feature_set/actions)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=1023590972)


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

## Installation

```bash
# Clone the repository
git clone https://github.com/Dav8Circle/melodic_feature_set.git
cd melodic_feature_set

# Install in development mode
pip install -e .
```

## Quick Start

The feature set can be easily accessed using the top-level function `get_all_features`. Here's a basic example:

```python
from melodic_feature_set.features import get_all_features, Config, IDyOMConfig, FantasticConfig

# Create a configuration
config = Config(
    corpus="src/melodic_feature_set/corpora/Essen_Corpus",
    idyom={
        "pitch": IDyOMConfig(
            target_viewpoints=["cpitch"],
            source_viewpoints=[("cpint", "cpintfref")],
            ppm_order=2,
            models=":both"
        ),
        "rhythm": IDyOMConfig(
            target_viewpoints=["onset"],
            source_viewpoints=["ioi"],
            ppm_order=1,
            models=":both"
        )
    },
    fantastic=FantasticConfig(
        max_ngram_order=6,
        phrase_gap=1.5
    )
)

# Extract features from a directory of MIDI files
get_all_features(
    input_directory="path/to/your/midi/files",
    output_file="features.csv",
    config=config
)
```

## Advanced Configuration

### Using Corpus Subsets

The package provides easy access to corpus subsets:

```python
from melodic_feature_set import essen_corpus, essen_first_ten

# Use the full Essen corpus
config = Config(corpus=essen_corpus)

# Or use just the first 10 files for quick testing
config = Config(corpus=essen_first_ten)
```

### Skipping Corpus-Dependent Features

You can skip corpus-dependent features by setting `corpus=None`:

```python
config = Config(
    corpus=None,  # Skip corpus-dependent features
    idyom={
    "pitch": IDyOMConfig(
            target_viewpoints=["cpitch"],
            source_viewpoints=[("cpint", "cpintfref")],
            ppm_order=1,
            models=":both"
        )},
    fantastic=FantasticConfig(
        max_ngram_order=6,
        phrase_gap=1.5)
)
```

## Melsim

Melsim is an R package for computing the similarity between two or more melodies. It is currently under development by Seb Silas and Klaus Frieler (https://github.com/sebsilas/melsim)

It is included with this feature set through a wrapper approach - take a look at example.py and the supplied MIDI files.

Since calculating similarities is highly modular in Melsim, we leave the user to decide how they wish to construct comparisons. Melsim is not run as part of the `get_all_features` function.

### Available Corpora

The package comes with an example corpus, a MIDI conversio of the well-known Essen Folksong Collection (Eck, 2024; Schaffrath, 1995).

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
