# Melodic Feature Set

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=1023590972)

[![Tests](https://github.com/Dav8Circle/melodic_feature_set/workflows/Tests/badge.svg)](https://github.com/Dav8Circle/melodic_feature_set/actions)

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

# Extract features from MIDI files
get_all_features(
    input_path="path/to/your/midi/files",  # Can be directory, single file, or list of files
    output_file="features.csv",
    config=config
)
```

## Input Options

The `get_all_features` function supports multiple input types:

### Directory Input
```python
# Process all MIDI files in a directory
get_all_features(
    input_path="path/to/midi/directory",
    output_file="features.csv",
    config=config
)
```

### Single File Input
```python
# Process a single MIDI file
get_all_features(
    input_path="path/to/single/file.mid",
    output_file="features.csv",
    config=config
)
```

### List of Files Input
```python
# Process specific MIDI files
midi_files = [
    "path/to/file1.mid",
    "path/to/file2.mid", 
    "path/to/file3.mid"
]
get_all_features(
    input_path=midi_files,
    output_file="features.csv",
    config=config
)
```

### Using Corpus Files
```python
from melodic_feature_set.corpus import get_corpus_files

# Get first 10 files from the Essen corpus
first_ten_files = get_corpus_files("essen", max_files=10)

get_all_features(
    input_path=first_ten_files,
    output_file="first_ten_features.csv",
    config=config
)
```

## Performance Options

### Skipping IDyOM Analysis

IDyOM analysis can be computationally expensive. You can skip it when not needed:

```python
get_all_features(
    input_path="path/to/midi/files",
    output_file="features_no_idyom.csv",
    config=config,
    skip_idyom=True  # Skip IDyOM analysis if you like
)
```

## Advanced Configuration

### Using Different Corpus Parameters

The package provides an easy way of supplying different corpora to different sets of features. For example:

```python
from melodic_feature_set.corpus import get_corpus_path, get_corpus_files

# Get a subset of files from the corpus
first_ten_files = get_corpus_files("essen", max_files=10)

config = Config(
    corpus=first_ten_files, # will be overriden 
    fantastic=FantasticConfig(
        max_ngram_order=6,
        phrase_gap=1.5,
        corpus=get_corpus_path("essen") # use full Essen corpus
    ),
    idyom={
        "pitch": IDyOMConfig(
            target_viewpoints=["cpitch"],
            source_viewpoints=[("cpint", "cpintfref")],
            ppm_order=1,
            models=":both",
            corpus=None # No corpus at all, no pretraining
        )
    }
)
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
