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

[![Tests](https://github.com/Dav8Circle/melodic_feature_set/workflows/Test/badge.svg)](https://github.com/Dav8Circle/melodic_feature_set/actions)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=1023590972)

The feature set can be easily accessed using the top level function `get_all_features`. 
Below is an example script demonstrating how it may be applied. This can be found in `example.py` in the package source code.

```py
from melodic_feature_set.features import get_all_features, Config, IDyOMConfig, FantasticConfig

# Example usage of the Config class
config = Config(
    # Setting this to None will skip corpus-dependent features, unless
    # we supply a corpus path in the idyom or fantastic configs.
    corpus="src/melodic_feature_set/Essen_Corpus",
    # We can supply multiple IDyOM configs using a dictionary
    # this means we can use different corpora and viewpoints for each config
    idyom={"pitch": IDyOMConfig(
        target_viewpoints=["cpitch"],
        source_viewpoints=["cpint", "cpintfref"],
        ppm_order=2,
        corpus="corpora/Essen_Corpus",
        models=":both"
    ),
    "rhythm": IDyOMConfig(
        target_viewpoints=["onset"],
        source_viewpoints=["ioi"],
        ppm_order=1,
        corpus="corpora/Essen_Corpus",
        models=":both"
    )},
    # Omitting the corpus path in Fantastic here will
    # use the corpus path from the Config object instead.
    fantastic=FantasticConfig(
        max_ngram_order=6,
        phrase_gap=1.5
    )
)

get_all_features(input_directory="PATH",
                output_file="example.csv",
                config=config)
```

## Melsim

Melsim is an R package for computing the similarity between two or more melodies. It is currently under development by Seb Silas and Klaus Frieler (https://github.com/sebsilas/melsim)

It is included with this feature set through a wrapper approach - take a look at example.py and the supplied MIDI files.

Since calculating similarities is highly modular in Melsim, we leave the user to decide how they wish to construct comparisons. Melsim is not run as part of the `get_all_features` function.

## Corpora

We supply the Essen Corpus as an example corpus (Eck, 2024; Schaffrath, 1995), as well as 
"Traditional Flute Dataset for Score Alignment" (https://www.kaggle.com/datasets/jbraga/traditional-flute-dataset/data) for usage with `example.ipynb`.