"""Discovery helpers for decorated melodic feature callables."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Union


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


_DESCRIPTOR_EXCLUDED_PROPERTIES = frozenset({"count_values", "freq_spec", "total_tokens"})

_TYPE_CATEGORY_LABELS = {
    "absolute": "Absolute Pitch",
    "pitch_class": "Pitch Class",
    "interval": "Pitch Interval",
    "contour": "Contour",
    "timing": "Timing",
    "tonality": "Tonality",
    "metre": "Metre",
    "expectation": "Expectation",
    "complexity": "Complexity",
    "lexical_diversity": "Lexical Diversity",
    "corpus": "Corpus",
}


@dataclass(frozen=True)
class FeatureInfo:
    """Metadata for one individual feature within the package."""

    name: str
    category: str
    domain: Optional[str]
    types: tuple[str, ...]
    sources: tuple[str, ...]
    summary: str
    kind: str  # "function" or "descriptor"


def _metadata_target(obj: Any, *, is_property: bool) -> Any:
    if is_property and hasattr(obj, "fget") and obj.fget is not None:
        return obj.fget
    return obj


def _category_label(types: list[str], domain: Optional[str], name: str) -> str:
    if name in {
        "yules_k",
        "simpsons_d",
        "sichels_s",
        "honores_h",
        "mean_entropy",
        "mean_productivity",
    }:
        return "Lexical Diversity"

    labels: list[str] = []
    for feature_type in types:
        label = _TYPE_CATEGORY_LABELS.get(feature_type)
        if feature_type == "interval" and domain == "rhythm" and "ioi" in name.lower():
            label = "Inter-Onset Interval"
        if label and label not in labels:
            labels.append(label)

    if labels:
        return ", ".join(labels)
    if types:
        return types[0].replace("_", " ").title()
    return "Other"


def _first_docline(obj: Any, *, is_property: bool) -> str:
    doc = inspect.getdoc(_metadata_target(obj, is_property=is_property)) or ""
    line = doc.strip().split("\n")[0] if doc else ""
    return line.strip()


def _iter_descriptor_features() -> Iterator[tuple[str, property]]:
    from .contour import (
        HuronContour,
        InterpolationContour,
        PolynomialContour,
        StepContour,
    )
    from .ngram_counter import NGramCounter

    for cls in (
        StepContour,
        InterpolationContour,
        PolynomialContour,
        HuronContour,
        NGramCounter,
    ):
        class_name = cls.__name__
        for prop_name, prop_obj in inspect.getmembers(cls):
            if not isinstance(prop_obj, property):
                continue
            if prop_name in _DESCRIPTOR_EXCLUDED_PROPERTIES:
                continue
            yield f"{class_name}.{prop_name}", prop_obj


def discover_atomic_features(
    module: Any,
    *,
    include_descriptors: bool = True,
) -> List[FeatureInfo]:
    """Collect metadata for atomic features exported from ``module``.

    Aggregator helpers (names starting with ``get_``) and alias bindings are
    excluded, matching the feature summary table conventions.
    """
    seen_function_ids: set[int] = set()
    entries: list[FeatureInfo] = []

    candidates: list[tuple[str, Any]] = [
        (name, obj)
        for name, obj in inspect.getmembers(module)
        if inspect.isfunction(obj)
    ]
    if include_descriptors:
        candidates.extend(_iter_descriptor_features())

    for name, obj in candidates:
        if name.startswith("get_") or name.startswith("_"):
            continue
        if name == "InverseEntropyWeighting":
            continue

        is_property = isinstance(obj, property)
        target = _metadata_target(obj, is_property=is_property)
        feature_types = getattr(target, "_feature_types", None)
        if not feature_types and not is_property:
            continue

        if not is_property:
            if not _is_canonical_function_binding(name, obj):
                continue
            try:
                function_id = id(inspect.unwrap(obj))
            except (AttributeError, ValueError):
                function_id = id(obj)
            if function_id in seen_function_ids:
                continue
            seen_function_ids.add(function_id)

        domain = getattr(target, "_feature_domain", None)
        types = tuple(feature_types or ())
        sources = tuple(getattr(target, "_feature_sources", []) or [])
        feature_name = name.split(".", 1)[-1] if "." in name else name

        entries.append(
            FeatureInfo(
                name=name,
                category=_category_label(list(types), domain, feature_name),
                domain=domain,
                types=types,
                sources=sources,
                summary=_first_docline(obj, is_property=is_property),
                kind="descriptor" if is_property else "function",
            )
        )

    return sorted(entries, key=lambda entry: entry.name.lower())


def list_available_features(
    *,
    domain: Optional[str] = None,
    feature_type: Optional[str] = None,
    source: Optional[str] = None,
    detailed: bool = False,
    module: Any = None,
) -> Union[List[str], List[dict]]:
    """List atomic features available in the package.

    Parameters
    ----------
    domain : str, optional
        Filter by feature domain (``pitch``, ``rhythm``, or ``both``)
    feature_type : str, optional
        Filter by decorator type (e.g. ``absolute``, ``interval``, ``contour``)
    source : str, optional
        Filter by implementation source (e.g. ``fantastic``, ``jsymbolic``)
    detailed : bool, optional
        If True, return metadata dictionaries instead of names only
    module : module, optional
        Module to scan (defaults to :mod:`melody_features.features`)

    Returns
    -------
    list[str] or list[dict]
        Feature names, or metadata records when ``detailed=True``
    """
    if module is None:
        import melody_features.features as features_module

        module = features_module

    entries = discover_atomic_features(module)

    if domain is not None:
        entries = [entry for entry in entries if entry.domain == domain]
    if feature_type is not None:
        entries = [entry for entry in entries if feature_type in entry.types]
    if source is not None:
        entries = [entry for entry in entries if source in entry.sources]

    if detailed:
        return [
            {
                "name": entry.name,
                "category": entry.category,
                "domain": entry.domain,
                "types": list(entry.types),
                "sources": list(entry.sources),
                "summary": entry.summary,
                "kind": entry.kind,
            }
            for entry in entries
        ]

    return [entry.name for entry in entries]
