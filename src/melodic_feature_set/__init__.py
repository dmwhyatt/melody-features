"""
A Python package for computing tonnes of melodic features found in computational musicology literature.
"""

# Import corpus paths for easy access
from .corpus import essen_corpus, essen_first_ten, get_corpus_path, list_available_corpora

# Import main feature function
from .features import get_all_features, Config, IDyOMConfig, FantasticConfig
