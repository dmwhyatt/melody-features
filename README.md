# Melodic Feature Set

This is a Python package designed to facilitate the use of many different melody analyis tools. 

The main goal of this package is to consolidate a wide range of features from the computational melody analysis literature
into a single place, in a single language.

Included in the package are contributions from:
- FANTASTIC (Müllensiefen, 2009)
- SIMILE (Müllensiefen & Frieler, 2006)
- melsim (Silas & Frieler, n.d.)
- jSymbolic2 (McKay & Fujinaga, 2006)
- IDyOM (Pearce, 2005)
- MIDI Toolbox (Eerola & Toiviainen, 2004)

The feature set is most readily accessed using the top level function `get_all_features`. 
Below is an example script demonstrating how it may be applied.

```py
# import the top level function
from feature_set.features import get_all_features

# set paths to directory of files to analyse, and name the output file
midi_directory = PATH_TO_MIDI_DIR
output_name = NAME_OF_OUTPUT_FILE

# using name is main guard to prevent circular imports
if __name__ == "__main__":
    get_all_features(midi_directory, output_name)
```

The package also supports a wide range of corpus features from FANTASTIC. These can be computed using a single additional step:

```py
from Feature_Set.corpus import make_corpus_stats
from Feature_Set.features import get_all_features

midi_directory = PATH_TO_MIDI_DIR
output_name = NAME_OF_OUTPUT_FILE

# additional name for corpus dictionary
corpus_name = NAME_OF_CORPUS_DICT

# using the same name is main guard
if __name__ == "__main__":
    make_corpus_stats(midi_directory, corpus_name)

    # We can then use the produced `.json` file as the third argument in our `get_all_features` function
    get_all_features(midi_directory, output_name, corpus_name)
```

## Melsim

Melsim is an R package for computing the similarity between two or more melodies. It is currently under development by Seb Silas and Klaus Frieler (https://github.com/sebsilas/melsim)

It is included with this feature set through a wrapper approach - take a look at example.py and the supplied MIDI files.

Since calculating similarities is highly modular in Melsim, we leave the user to decide how they wish to construct comparisons. Melsim is not run as part of the `get_all_features` function.

## IDyOM

Some of the viewpoints in IDyOM are relevant to melodic analysis in their own right, for example the implementation of the bottom-up Narmour principles (Krumhansl, 1995; Narmour, 1990). These principles are implemented in `get_all_features`.

However, the popular IDyOM measures `entropy` and `ic` must be computed and averaged separately. We include a script to facilitate this analysis, though similar to Melsim, it is a wrapper that runs SBCL in the background; it is not pure Python. Hence, we felt it best to keep distinct from the rest of the implementation, for the time being.