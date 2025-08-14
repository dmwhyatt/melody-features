"""
A Python package for computing tonnes of melodic features found in computational musicology literature.
"""

from .corpus import (
    essen_corpus,  # noqa: F401
    get_corpus_path,
    list_available_corpora,
)

from .features import (
    Config,
    FantasticConfig,
    IDyOMConfig,  # noqa: F401
    get_all_features,
)
