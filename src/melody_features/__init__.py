"""
A Python package for computing tonnes of melodic features found in computational musicology literature.
"""

from .corpus import (
    essen_corpus,  # noqa: F401
    pearce_default_idyom,  # noqa: F401
    get_corpus_path,
    list_available_corpora,
    load_melodies_from_directory,
)

from .feature_definitions.absolute_pitch import *  # noqa: F403
from .feature_definitions.absolute_pitch import __all__ as _absolute_pitch_all
from .feature_definitions.pitch_class import *  # noqa: F403
from .feature_definitions.pitch_class import __all__ as _pitch_class_all
from .feature_definitions.pitch_interval import *  # noqa: F403
from .feature_definitions.pitch_interval import __all__ as _pitch_interval_all
from .feature_definitions.contour import *  # noqa: F403
from .feature_definitions.contour import __all__ as _contour_all
from .feature_definitions.timing import *  # noqa: F403
from .feature_definitions.timing import __all__ as _timing_all
from .feature_definitions.inter_onset_interval import *  # noqa: F403
from .feature_definitions.inter_onset_interval import __all__ as _inter_onset_interval_all
from .feature_definitions.tonality import *  # noqa: F403
from .feature_definitions.tonality import __all__ as _tonality_all
from .feature_definitions.expectation import *  # noqa: F403
from .feature_definitions.expectation import __all__ as _expectation_all
from .feature_definitions.metre import *  # noqa: F403
from .feature_definitions.metre import __all__ as _metre_all
from .feature_definitions.corpus import *  # noqa: F403
from .feature_definitions.corpus import __all__ as _corpus_all
from .feature_definitions.complexity import *  # noqa: F403
from .feature_definitions.complexity import __all__ as _complexity_all
from .feature_definitions.lexical_diversity import *  # noqa: F403
from .feature_definitions.lexical_diversity import __all__ as _lexical_diversity_all

from .features import (
    Config,
    FantasticConfig,
    IDyOMConfig,  # noqa: F401
    get_all_features,
    list_available_features,
)
from .feature_metadata import get_feature_metadata
from .reshape import to_long_format

__all__ = [
    "essen_corpus",
    "pearce_default_idyom",
    "get_corpus_path",
    "list_available_corpora",
    "Config",
    "FantasticConfig",
    "IDyOMConfig",
    "get_all_features",
    "list_available_features",
    "get_feature_metadata",
    "to_long_format",
    "load_melodies_from_directory",
    *_absolute_pitch_all,
    *_pitch_class_all,
    *_pitch_interval_all,
    *_contour_all,
    *_timing_all,
    *_inter_onset_interval_all,
    *_tonality_all,
    *_expectation_all,
    *_metre_all,
    *_corpus_all,
    *_complexity_all,
    *_lexical_diversity_all,
]

del _absolute_pitch_all, _pitch_class_all, _pitch_interval_all, _contour_all, _timing_all, _inter_onset_interval_all, _tonality_all, _expectation_all, _metre_all, _corpus_all, _complexity_all, _lexical_diversity_all

from .legacy_shims import install_legacy_shims

install_legacy_shims()
