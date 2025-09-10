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
    NOVEL = "novel"


class FeatureType:
    """Class for categorizing features by type."""
    PITCH = "pitch"
    INTERVAL = "interval"
    CONTOUR = "contour"
    TONALITY = "tonality"
    DURATION = "duration"
    COMPLEXITY = "complexity"
    CORPUS = "corpus"
    MTYPE = "mtype" 

def _create_feature_decorator(source: str, citation: str, feature_type: str = None) -> Callable:
    """Create a feature decorator for a specific source and optionally a feature type."""
    def decorator(func: Callable) -> Callable:
        # Initialize attributes if they don't exist
        if not hasattr(func, '_feature_sources'):
            func._feature_sources = []
        if not hasattr(func, '_feature_citations'):
            func._feature_citations = []
        if not hasattr(func, '_feature_types'):
            func._feature_types = []

        # Add source information
        if source not in func._feature_sources:
            func._feature_sources.append(source)
            func._feature_citations.append(citation)

        # Add feature type if provided
        if feature_type and feature_type not in func._feature_types:
            func._feature_types.append(feature_type)

        # Set primary attributes
        func._feature_source = func._feature_sources[0]
        func._feature_citation = func._feature_citations[0]
        if func._feature_types:
            func._feature_type = func._feature_types[0]

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Copy all attributes to wrapper
        wrapper._feature_sources = func._feature_sources.copy()
        wrapper._feature_citations = func._feature_citations.copy()
        wrapper._feature_types = func._feature_types.copy()
        wrapper._feature_source = func._feature_source
        wrapper._feature_citation = func._feature_citation
        if hasattr(func, '_feature_type'):
            wrapper._feature_type = func._feature_type

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
    FeatureSource.NOVEL,
    "Novel features do not appear in any of the referenced literature. We introduce them here to extend the contributions of existing feature sets."
)


def feature_type(feature_type: str) -> Callable:
    """Decorator to specify the type/category of a feature.
    
    Parameters
    ----------
    feature_type : str
        The type of feature (e.g., 'pitch', 'rhythm', 'interval', etc.)
    """
    def decorator(func: Callable) -> Callable:
        # Initialize attributes if they don't exist
        if not hasattr(func, '_feature_sources'):
            func._feature_sources = []
        if not hasattr(func, '_feature_citations'):
            func._feature_citations = []
        if not hasattr(func, '_feature_types'):
            func._feature_types = []
        
        # Add feature type
        if feature_type not in func._feature_types:
            func._feature_types.append(feature_type)
        
        # Set primary feature type
        func._feature_type = feature_type
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Copy all attributes to wrapper
        wrapper._feature_sources = getattr(func, '_feature_sources', []).copy()
        wrapper._feature_citations = getattr(func, '_feature_citations', []).copy()
        wrapper._feature_types = func._feature_types.copy()
        wrapper._feature_source = getattr(func, '_feature_source', None)
        wrapper._feature_citation = getattr(func, '_feature_citation', None)
        wrapper._feature_type = func._feature_type
        
        return wrapper
    return decorator


# Feature type decorators - can be combined with source decorators
pitch_feature = feature_type(FeatureType.PITCH)
interval_feature = feature_type(FeatureType.INTERVAL)
contour_feature = feature_type(FeatureType.CONTOUR)
tonality_feature = feature_type(FeatureType.TONALITY)
duration_feature = feature_type(FeatureType.DURATION)
complexity_feature = feature_type(FeatureType.COMPLEXITY)
corpus_feature = feature_type(FeatureType.CORPUS)
mtype_feature = feature_type(FeatureType.MTYPE)
