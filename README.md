# Melody-Features

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=1023590972)

[![DOI](https://zenodo.org/badge/1023590972.svg)](https://doi.org/10.5281/zenodo.16894207)

[![Tests](https://github.com/dmwhyatt/melody-features/workflows/Tests/badge.svg)](https://github.com/dmwhyatt/melody-features/actions)

[![Coverage](https://codecov.io/gh/dmwhyatt/melody-features/branch/main/graph/badge.svg)](https://codecov.io/gh/dmwhyatt/melody-features)

## Overview
This is a Python package designed to facilitate the use of many different melody analysis tools. 

The main goal of this package is to consolidate a wide range of features from the computational melody analysis literature
into a single place, in a single language.

This package is strictly for monophonic melodies - it will not compute any features for polyphonic music!


## Included Contributions

Included in the package are contributions from:

- **FANTASTIC** (Müllensiefen, 2009)
- **SIMILE** (Müllensiefen & Frieler, 2006)
- **melsim** (Silas & Frieler, n.d.)
- **jSymbolic2** (McKay & Fujinaga, 2006)
- **IDyOM** (Pearce, 2005)
- **MIDI Toolbox** (Eerola & Toiviainen, 2004)
- **Partitura** (Cancino-Chacón, 2022)

## Melody Features Summary

This package provides over **200 features** from various computational melody analysis frameworks. For a comprehensive, interactive table with search and sorting capabilities, refer to:

**[Interactive Features Table](https://dmwhyatt.github.io/melody-features/)**

The interactive table allows you to:
- **Search** features by name, implementation, or description
- **Sort** by any column (Name, Implementation, Type, etc.)
- **Browse** all features with detailed descriptions and references

## Installation

```bash

# using pip
pip install melody-features

# or clone the repository
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

By default, this function will produce a Pandas DataFrame containing the tabulated features, using the a collection of 903 Western traditional music melodies as the reference corpus, from Pearce (2006).


This function can be customised in a number of ways, please see `notebooks/example.ipynb` for a detailed breakdown.

## Melsim

Melsim is an R package for computing the similarity between two or more melodies. It is currently under development by Seb Silas and Klaus Frieler (https://github.com/sebsilas/melsim)

It is included with this feature set through a wrapper approach - take a look at example.py and the supplied MIDI files.

Since calculating similarities is highly modular in Melsim, we leave the user to decide how they wish to construct comparisons. Melsim is not run as part of the `get_all_features` function.

### Available Corpora

The package comes with two example corpora: a MIDI conversion of the well-known Essen Folksong Collection (Eck, 2024; Schaffrath, 1995), and 903 Western musical tradition melodies, used by Pearce for IDyOM pretraining (Pearce, 2006). By default, the 903-melody corpus is invoked by `get_all_features`. 

## Development

### Running Tests

```bash
# Simply run pytest
pytest

# or with Python, run all tests
python tests/run_tests.py

# Run specific test suites
python -m pytest tests/test_module_setup.py -v
python -m pytest tests/test_corpus_import.py -v
python -m pytest tests/test_idyom_setup.py -v
```

## Contributing

If you spot something you think ought to be included here, feel free to contribute it!
Simply fork the repo, implement your feature, and submit a Pull Request that explains
the proposed addition(s). If you seek to contribute features that relate to one another, 
you may propose them in a single PR, otherwise, please submit separate PRs for each feature, 
as this makes it simpler to review the code.

Presently, we don't use a formalised style guide. However, we expect that your code will adhere to the following principles:
- Each module should always include a docstring that succintly explains the purpose of the module
- Each function within a module should have its own docstring and type hints. The docstring should include a citation to the relevant literature resource
- Each top-level feature should return using a native Python type
- New features should be accompanied by tests. Where it is possible, implemented features should be validated against their source implementation: see [tests/test_jsymbolic_validation.py](tests/test_jsymbolic_validation.py) for an example.

## License

This project is licensed under the MIT License - see the [LICENSE-MIT](LICENSE-MIT) file for details.

Open-source code adapted from the Partitura Python package is licensed under the Apache-2.0 license, which can be found in the [LICENSE-APACHE](LICENSE-APACHE) file. More details can be found in [NOTICE](NOTICE).
