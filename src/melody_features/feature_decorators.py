"""
Feature decorators for categorizing melodic features by source.
"""

from functools import wraps
from typing import Callable


class FeatureSource:
    """Class for easy construction of feature decorators."""
    FANTASTIC = "fantastic"
    IDYOM = "idyom"
    MIDI_TOOLBOX = "midi_toolbox"
    JSYMBOLIC = "jsymbolic"
    MELSIM = "melsim"
    SIMILE = "simile"
    CUSTOM = "custom"


def _create_feature_decorator(source: str, citation: str) -> Callable:
    """Create a feature decorator for a specific source."""
    def decorator(func: Callable) -> Callable:
        func._feature_source = source
        func._feature_citation = citation

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._feature_source = source
        wrapper._feature_citation = citation

        return wrapper
    return decorator

fantastic = _create_feature_decorator(
    FeatureSource.FANTASTIC,
    "Müllensiefen, D. (2009). Fantastic: Feature ANalysis Technology Accessing STatistics (In a Corpus): Technical Report v1.5"
)

idyom = _create_feature_decorator(
    FeatureSource.IDYOM,
    "Pearce, M. T. (2005). The construction and evaluation of statistical models of melodic structure in music perception and composition."
)

midi_toolbox = _create_feature_decorator(
    FeatureSource.MIDI_TOOLBOX,
    "Eerola, T., & Toiviainen, P. (2004). MIDI Toolbox: MATLAB Tools for Music Research."
)

melsim = _create_feature_decorator(
    FeatureSource.MELSIM,
    "Silas, S., & Frieler, K. (n.d.). Melsim: Framework for calculating tons of melodic similarities."
)

jsymbolic = _create_feature_decorator(
    FeatureSource.JSYMBOLIC,
    "McKay, C., & Fujinaga, I. (2006). jSymbolic: A Feature Extractor for MIDI Files."
)

simile = _create_feature_decorator(
    FeatureSource.SIMILE,
    "Müllensiefen, D., & Frieler, K. (2004). The Simile algorithms documentation 0.3"
)

novel = _create_feature_decorator(
    FeatureSource.CUSTOM,
    "Novel features do not appear in any of the referenced literature. We introduce them here to extend the contributions of existing feature sets."
)
