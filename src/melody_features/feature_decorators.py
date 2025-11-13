"""
Feature decorators for categorizing melodic features by source.

We integrate multiple decorators that are used to label features with their source, category, and domain.
Domain decorators are `pitch`, `rhythm`, and `both`, accounting for the different input representations used to compute the features.
- `pitch` captures anything corresponding to MIDI pitch numbers and re-expressions thereof.
- `rhythm` captures anything that utilises onsets and offsets, including re-expressions thereof.
- `both` is a special category devised to cover any features that are computed from representations 
that combine both pitch and rhythm. This specifially covers the MType features from FANTASTIC.

We also implement decorators corresponding to the original software resource that inspired our present implmentation. This makes 
it possible to calculate features from a specific software resource using `_get_features_by_source`, which is used in the `features.py` to produce
functions like `get_fantastic_features`.

The category decorators are designed to organise features into categories that are useful for analysis and interpretation.
Features from the same category are typically related by the representation used to compute the feature, for example,
`pitch_class` features are computed from MIDI pitch numbers modulo 12.
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
    PARTITURA = "partitura"


class FeatureType:
    """Class for categorizing features by type."""
    INTERVAL = "interval"
    pitch_class = "pitch_class"
    CONTOUR = "contour"
    TONALITY = "tonality"
    METRE = "metre"
    ABSOLUTE = "absolute"
    TIMING = "timing"
    LEXICAL_DIVERSITY = "lexical_diversity"
    EXPECTATION = "expectation"
    COMPLEXITY = "complexity"


class FeatureDomain:
    """Class for categorizing features by domain."""
    PITCH = "pitch"
    RHYTHM = "rhythm"
    BOTH = "both"


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

partitura = _create_feature_decorator(
    FeatureSource.PARTITURA,
    "Cancino-Chacón, C. E., et al., (2022) Partitura: A Python package for symbolic music processing."
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


def domain(domain_value: str) -> Callable:
    """Decorator to specify the domain of a feature.
    
    Parameters
    ----------
    domain_value : str
        The domain of the feature. Must be one of: 'pitch', 'rhythm', or 'both'
        
    Raises
    ------
    ValueError
        If domain_value is not one of the allowed values
    """
    valid_domains = {FeatureDomain.PITCH, FeatureDomain.RHYTHM, FeatureDomain.BOTH}
    if domain_value not in valid_domains:
        raise ValueError(
            f"Domain must be one of {valid_domains}, got '{domain_value}'"
        )
    
    def decorator(func: Callable) -> Callable:
        # Initialize attributes if they don't exist
        if not hasattr(func, '_feature_sources'):
            func._feature_sources = []
        if not hasattr(func, '_feature_citations'):
            func._feature_citations = []
        if not hasattr(func, '_feature_types'):
            func._feature_types = []
        
        # Set feature domain
        func._feature_domain = domain_value
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Copy all attributes to wrapper
        wrapper._feature_sources = getattr(func, '_feature_sources', []).copy()
        wrapper._feature_citations = getattr(func, '_feature_citations', []).copy()
        wrapper._feature_types = getattr(func, '_feature_types', []).copy()
        wrapper._feature_source = getattr(func, '_feature_source', None)
        wrapper._feature_citation = getattr(func, '_feature_citation', None)
        wrapper._feature_domain = func._feature_domain
        if hasattr(func, '_feature_type'):
            wrapper._feature_type = func._feature_type
        
        return wrapper
    return decorator


# Feature type decorators - can be combined with source decorators
interval = feature_type(FeatureType.INTERVAL)
pitch_class = feature_type(FeatureType.pitch_class)
contour = feature_type(FeatureType.CONTOUR)
tonality = feature_type(FeatureType.TONALITY)
metre = feature_type(FeatureType.METRE)
absolute = feature_type(FeatureType.ABSOLUTE)
timing = feature_type(FeatureType.TIMING)
lexical_diversity = feature_type(FeatureType.LEXICAL_DIVERSITY)
expectation = feature_type(FeatureType.EXPECTATION)
complexity = feature_type(FeatureType.COMPLEXITY)

# Domain decorators - can be combined with source decorators
pitch = domain(FeatureDomain.PITCH)
rhythm = domain(FeatureDomain.RHYTHM)
both = domain(FeatureDomain.BOTH)
