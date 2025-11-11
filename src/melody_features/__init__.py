"""
A Python package for computing tonnes of melodic features found in computational musicology literature.
"""

from .corpus import (
    essen_corpus,  # noqa: F401
    get_corpus_path,
    list_available_corpora,
    load_melodies_from_directory,
)

from .features import (
    Config,
    FantasticConfig,
    IDyOMConfig,  # noqa: F401
    get_all_features,
)

__all__ = [
    "essen_corpus",
    "get_corpus_path",
    "list_available_corpora",
    "Config",
    "FantasticConfig",
    "IDyOMConfig",
    "get_all_features",
    "load_melodies_from_directory",
]