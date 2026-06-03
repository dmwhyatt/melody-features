"""Discovery helpers for decorated melodic feature callables."""

import inspect
from typing import Any, Callable, Dict


def _is_canonical_function_binding(name: str, obj: Any) -> bool:
    """Return whether ``name`` is the callable's defining name."""
    return not (inspect.isfunction(obj) and obj.__name__ != name)


def _add_unique_feature(
    features: Dict[str, Callable],
    seen_ids: set[int],
    name: str,
    obj: Callable,
) -> None:
    """Add a feature once, preserving the first canonical binding found."""
    feature_id = id(obj)
    if feature_id in seen_ids:
        return
    seen_ids.add(feature_id)
    features[name] = obj


def get_features_by_type(module: Any, feature_type: str) -> Dict[str, Callable]:
    """Get all canonical feature functions in ``module`` for a feature type."""
    features: Dict[str, Callable] = {}
    seen_ids: set[int] = set()

    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if not inspect.isfunction(obj):
            continue
        if not _is_canonical_function_binding(name, obj):
            continue
        if not hasattr(obj, "_feature_types"):
            continue
        if feature_type not in obj._feature_types:
            continue
        _add_unique_feature(features, seen_ids, name, obj)

    return features


def get_features_by_domain(module: Any, domain: str) -> Dict[str, Callable]:
    """Get all canonical feature functions in ``module`` for a feature domain."""
    features: Dict[str, Callable] = {}
    seen_ids: set[int] = set()

    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if not inspect.isfunction(obj):
            continue
        if not _is_canonical_function_binding(name, obj):
            continue
        if not hasattr(obj, "_feature_domain"):
            continue
        if obj._feature_domain != domain:
            continue
        _add_unique_feature(features, seen_ids, name, obj)

    return features


def get_features_by_domain_and_types(
    module: Any,
    domain: str,
    allowed_types: list[str],
) -> Dict[str, Callable]:
    """Get all canonical features for a domain matching any allowed type."""
    features: Dict[str, Callable] = {}
    seen_ids: set[int] = set()

    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if not inspect.isfunction(obj):
            continue
        if not _is_canonical_function_binding(name, obj):
            continue
        if not hasattr(obj, "_feature_domain") or obj._feature_domain != domain:
            continue
        if not hasattr(obj, "_feature_types"):
            continue
        if not any(feature_type in allowed_types for feature_type in obj._feature_types):
            continue
        _add_unique_feature(features, seen_ids, name, obj)

    return features


def _is_source_candidate(obj: Any) -> bool:
    return (
        inspect.isfunction(obj)
        or inspect.isclass(obj)
        or (hasattr(obj, "__call__") and hasattr(obj, "__name__"))
    )


def get_features_by_source(module: Any, source: str) -> Dict[str, Callable]:
    """Get decorated functions/classes in ``module`` for a source label."""
    source_features: Dict[str, Callable] = {}

    for name, obj in inspect.getmembers(module):
        if not _is_source_candidate(obj):
            continue
        if hasattr(obj, "_feature_sources") and source in obj._feature_sources:
            source_features[name] = obj
        elif hasattr(obj, "_feature_source") and obj._feature_source == source:
            source_features[name] = obj

    return source_features
