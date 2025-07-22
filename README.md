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
    get_all_features(midi_directory)
```

The package also supports a wide range of corpus features from FANTASTIC. These can be computed using a single additional step:

```py
from Feature_Set.corpus import make_corpus_stats
from Feature_Set.features import get_all_features

# using the same name is main guard
if __name__ == "__main__":
    make_corpus_stats("path_to_midi_file_directory", "name_of_output_dict")

    # We can then use the produced `name_of_output_dict.json` file as the third argument in our `get_all_features` function
    get_all_features("path_to_midi_file_directory", "name_of_output_file", "name_of_output_dict.json")
```
