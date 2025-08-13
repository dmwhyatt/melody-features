"""
A Python package for computing tonnes of melodic features found in computational musicology literature.
"""

# Import corpus paths for easy access
from .corpus import (
    essen_corpus,
    essen_first_ten,  # noqa: F401
    get_corpus_path,
    list_available_corpora,
)

# Import main feature function
from .features import (
    Config,
    FantasticConfig,
    IDyOMConfig,  # noqa: F401
    get_all_features,
)
