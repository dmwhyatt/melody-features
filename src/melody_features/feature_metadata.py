"""Shared metadata for melodic features: name, source, type, family, description.

This module is the single source of truth for feature metadata used both by
the Quarto documentation table (`docs/quarto_table_build.py`) and by the
package-level :func:`get_feature_metadata` API, which lets users join
per-feature metadata (source, family, domain, type, description) onto a long
format feature table by `feature_name`.

Collects labelled feature callables from `melody_features.features`, extracts
metadata such as pre-existing implementations, references, descriptions,
notes, and categories.
"""

from __future__ import annotations

import inspect
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote

import pandas as pd

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent


@dataclass
class FeatureRow:
    """One row of metadata for a single atomic feature (function or descriptor)."""

    python_name: str
    name: str
    source_url: str
    implementations: str
    references: str
    description: str
    type_label: str
    notes: str
    category: str
    domain: str
    sort_name: str
    feature_types: tuple = ()
    has_corpus_stats_param: bool = False


SECTION_RE = re.compile(r"^([A-Za-z ]+)\n[-]+$", re.MULTILINE)

FEATURE_ALIAS_EXPORTS: dict[str, tuple[str, str]] = {
    "ambitus": ("pitch_range", "MIDI Toolbox"),
    "average_time_between_attacks": ("ioi_mean", "jSymbolic"),
    "duration_in_seconds": ("global_duration", "jSymbolic"),
    "mean_melodic_interval": ("mean_absolute_interval", "jSymbolic"),
    "most_common_interval": ("modal_interval", "jSymbolic"),
    "number_of_common_pitches_classes": ("number_of_common_pitch_classes", "local legacy export"),
    "pitch_variability": ("pitch_standard_deviation", "jSymbolic"),
    "variability_of_time_between_attacks": ("ioi_standard_deviation", "jSymbolic"),
    "total_number_of_notes": ("length", "jSymbolic"),
}

FEATURE_DISPLAY_NAME_OVERRIDES: dict[str, str] = {
    "compltrans": "Melodic Originality (Compltrans)",
    "complebm_pitch": "Expectancy Complexity Pitch (Complebm)",
    "complebm_rhythm": "Expectancy Complexity Rhythm (Complebm)",
    "complebm_optimal": "Expectancy Complexity Optimal (Complebm)",
}

_CANONICAL_ALIAS_NOTES: dict[str, list[tuple[str, str]]] = {}
for _alias, (_canonical, _source) in FEATURE_ALIAS_EXPORTS.items():
    _CANONICAL_ALIAS_NOTES.setdefault(_canonical, []).append((_alias, _source))


# The wide-format DataFrame returned by `get_all_features` names contour
# descriptor features after the composite keys assembled in
# `feature_definitions/contour.py::get_contour_features`, which don't match
# the descriptor property's registry name (`ClassName.prop_name`). This is
# the small, closed set of aliases needed to bridge that gap so that
# `get_feature_metadata()` rows line up with real wide-format column names.
CONTOUR_COMPOSITE_ALIASES: dict[str, str] = {
    "StepContour.global_variation": "step_contour_global_variation",
    "StepContour.global_direction": "step_contour_global_direction",
    "StepContour.local_variation": "step_contour_local_variation",
    "InterpolationContour.global_direction": "interpolation_contour_global_direction",
    "InterpolationContour.mean_gradient": "interpolation_contour_mean_gradient",
    "InterpolationContour.gradient_std": "interpolation_contour_gradient_std",
    "InterpolationContour.direction_changes": "interpolation_contour_direction_changes",
    "InterpolationContour.class_label": "interpolation_contour_class_label",
    "PolynomialContour.coefficients": "polynomial_contour_coefficients",
    "HuronContour.class_label": "huron_contour",
}


def _alias_display_name(python_name: str) -> str:
    return fix_possessive_feature_names(normalize_feature_text(snake_to_title(python_name)))


def _alias_note(alternate_python_name: str, source: str) -> str:
    return (
        f'This feature is named `{alternate_python_name}` '
        f"({_alias_display_name(alternate_python_name)}) in {source}."
    )


def _notes_for_feature(python_name: str, docstring_notes: str) -> str:
    parts: list[str] = []
    if docstring_notes:
        parts.append(docstring_notes)
    doc_lower = docstring_notes.lower()
    for alternate_name, source in _CANONICAL_ALIAS_NOTES.get(python_name, []):
        if alternate_name.lower() in doc_lower:
            continue
        note = _alias_note(alternate_name, source)
        if note not in parts and all(note not in p for p in parts):
            parts.append(note)
    return " ".join(parts)


def snake_to_title(name: str) -> str:
    return name.replace("_", " ").strip().title()


def capitalize_ioi(text: str) -> str:
    """Capitalize 'IOI' in text while preserving other formatting."""
    if not text:
        return text
    return re.sub(r'\bioi\b', 'IOI', text, flags=re.IGNORECASE)


def normalize_feature_text(text: str) -> str:
    """Normalize acronyms and tokens in free text.
    - IOI -> IOI
    - df -> DF (word-boundary)
    - tfdf -> TFDF (word-boundary)
    - npvi -> NPVI (word-boundary)
    """
    if not text:
        return text
    text = capitalize_ioi(text)
    text = re.sub(r"\bstm\b", "STM", text, flags=re.IGNORECASE)
    text = re.sub(r"\bltm\b", "LTM", text, flags=re.IGNORECASE)
    text = re.sub(r"\btfdf\b", "TFDF", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdf\b", "DF", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnpvi\b", "NPVI", text, flags=re.IGNORECASE)
    return text


def fix_possessive_feature_names(text: str) -> str:
    """Fix known feature name possessives that are lost by title-casing.
    E.g., 'Honores H' -> "Honore's H", 'Sichels S' -> "Sichel's S", etc.
    """
    if not text:
        return text
    replacements = [
        (r"\bHonores H\b", "Honore's H"),
        (r"\bSichels S\b", "Sichel's S"),
        (r"\bSimpsons D\b", "Simpson's D"),
        (r"\bYules K\b", "Yule's K"),
    ]
    result = text
    for pattern, repl in replacements:
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
    return result


def extract_sections_from_docstring(doc: str) -> dict[str, str]:
    """Parse simple NumPy-style sections (Parameters, Returns, Notes, Citation, etc.)."""
    if not doc:
        return {}
    text = inspect.cleandoc(doc)
    sections: dict[str, str] = {}
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        sections["Preamble"] = text.strip()
        return sections
    first_start = matches[0].start()
    preamble = text[:first_start].strip()
    if preamble:
        sections["Preamble"] = preamble
    for idx, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections[title] = body
    return sections


def determine_type_from_return_annotation(obj) -> str:
    try:
        ann = inspect.signature(obj).return_annotation
    except (TypeError, ValueError):
        return "Descriptor"

    if ann is inspect.Signature.empty:
        return "Descriptor"

    scalar_types = (int, float, bool)
    sequence_type_names = {"list", "tuple", "dict", "set", "ndarray", "Series", "DataFrame"}

    if isinstance(ann, type):
        return "Descriptor" if issubclass(ann, scalar_types) else "Sequence"

    if isinstance(ann, str):
        lowered = ann.lower()
        if any(t in lowered for t in ("int", "float", "bool")) and not any(t in lowered for t in ("list", "tuple", "dict", "set")):
            return "Descriptor"
        if any(t in lowered for t in ("list", "tuple", "dict", "set", "ndarray")):
            return "Sequence"
        return "Descriptor"

    name = getattr(getattr(ann, "__origin__", None), "__name__", "") or getattr(ann, "_name", "") or str(ann)
    for seq_name in sequence_type_names:
        if seq_name in str(name):
            return "Sequence"
    return "Descriptor"


def _get_feature_category(obj, domain: str = None, feature_name: str = None) -> str:
    if feature_name in {"mean_melodic_accent", "melodic_accent_std"}:
        return "Complexity"

    """Determine the feature category based on the actual feature type decorator.
    Returns a comma-separated string of categories for features that belong to multiple categories.

    Parameters
    ----------
    obj : object
        The feature object (function or property)
    domain : str, optional
        The feature domain ('pitch', 'rhythm', 'both', etc.)
    feature_name : str, optional
        The feature name to help determine category (e.g., for mtype features or IOI features)
    """
    mtype_features = {"yules_k", "simpsons_d", "sichels_s", "honores_h", "mean_entropy", "mean_productivity"}

    corpus_feature_names = {
        "tfdf_spearman", "tfdf_kendall", "norm_log_dist", "max_log_df", "min_log_df",
        "mean_log_df", "mean_global_local_weight",
        "mean_global_weight", "mean_log_tfdf",
        "mean_document_frequency"
    }

    if feature_name and feature_name in corpus_feature_names:
        return 'Corpus'

    is_property = isinstance(obj, property)
    target = obj.fget if is_property else obj
    try:
        sig = inspect.signature(target)
        if 'corpus_stats' in sig.parameters:
            if feature_name and feature_name not in mtype_features:
                return 'Corpus'
    except (TypeError, ValueError):
        pass

    if feature_name and feature_name in mtype_features:
        return 'Lexical Diversity'

    if is_property and hasattr(obj, 'fget') and obj.fget is not None:
        feature_types = getattr(obj.fget, "_feature_types", None)
    else:
        feature_types = getattr(obj, "_feature_types", None)

    type_mapping = {
        'pitch': 'Absolute Pitch',
        'interval': 'Pitch Interval',
        'contour': 'Contour',
        'rhythm': 'Timing',
        'complexity': 'Complexity',
        'tonality': 'Tonality',
        'metre': 'Metre',
        'expectation': 'Expectation',
        'lexical_diversity': 'Lexical Diversity',
        'corpus': 'Corpus',
        'mtype': 'Lexical Diversity',
        'pitch_class': 'Pitch Class',
        'absolute': 'Absolute Pitch',
        'timing': 'Timing',
    }

    if feature_types and len(feature_types) > 0:
        categories = []
        for feature_type in feature_types:
            mapped = type_mapping.get(feature_type)

            if feature_type == 'absolute' and domain == 'pitch':
                mapped = 'Absolute Pitch'
            elif feature_type == 'pitch_class' and domain == 'pitch':
                mapped = 'Pitch Class'
            elif feature_type == 'interval' and domain == 'pitch':
                mapped = 'Pitch Interval'
            elif feature_type == 'interval' and domain == 'rhythm':
                if feature_name and 'ioi' in feature_name.lower():
                    mapped = 'Inter-Onset Interval'
                else:
                    mapped = 'Timing'

            if mapped and mapped not in categories:
                categories.append(mapped)

        if categories:
            return ', '.join(categories)
        return feature_types[0].replace('_', ' ').title()

    if hasattr(obj, '__name__'):
        name = obj.__name__
        if name in ['honores_h', 'yules_k', 'simpsons_d', 'sichels_s', 'mean_entropy', 'mean_productivity']:
            return 'Lexical Diversity'
        elif name in ['class_label', 'global_variation', 'global_direction', 'local_variation', 'coefficients']:
            return 'Contour'

    if isinstance(obj, property):
        if hasattr(obj, 'fget') and obj.fget:
            if hasattr(obj.fget, '__qualname__'):
                qualname = obj.fget.__qualname__
                if 'NGramCounter' in qualname:
                    return 'Complexity'
                elif any(cls in qualname for cls in ['HuronContour', 'StepContour', 'InterpolationContour', 'PolynomialContour']):
                    return 'Contour'
        return 'Other'

    return 'Other'


def _detect_repo_info() -> tuple[str, str]:
    """Return (repo_url, branch). Tries env, then pyproject, falls back to defaults."""
    repo_url = os.getenv("REPO_URL") or os.getenv("FEATURES_REPO_URL")
    branch = os.getenv("REPO_BRANCH") or os.getenv("FEATURES_REPO_BRANCH") or "main"
    if not repo_url:
        try:
            import tomllib
        except ImportError:
            tomllib = None
        if tomllib is not None:
            pyproj = _REPO_ROOT / "pyproject.toml"
            if pyproj.exists():
                try:
                    with open(pyproj, "rb") as f:
                        data = tomllib.load(f)
                    repo_url = (
                        data.get("project", {})
                        .get("urls", {})
                        .get("Homepage")
                        or data.get("project", {})
                        .get("urls", {})
                        .get("Repository")
                        or ""
                    )
                except (OSError, tomllib.TOMLDecodeError):
                    repo_url = None
    if not repo_url:
        repo_url = "https://github.com/dmwhyatt/melody-features"
    return repo_url.rstrip("/"), branch


def _build_source_url(obj: object) -> str:
    repo_url, repo_branch = _detect_repo_info()
    target = obj.fget if isinstance(obj, property) else obj

    try:
        target_unwrapped = inspect.unwrap(target)
    except (AttributeError, ValueError):
        target_unwrapped = target
    try:
        file_path_str = inspect.getsourcefile(target_unwrapped) or inspect.getfile(target_unwrapped)
        if not file_path_str:
            return ""
        file_path = Path(file_path_str)
        _, start_line = inspect.getsourcelines(target_unwrapped)
    except (OSError, TypeError):
        return ""

    try:
        rel_path = file_path.relative_to(_REPO_ROOT)
    except ValueError:
        parts = file_path.parts
        rel_path = None
        if "src" in parts:
            idx = parts.index("src")
            rel_path = Path(*parts[idx:])
        elif "melody_features" in parts:
            idx = parts.index("melody_features")
            rel_path = Path("src") / Path(*parts[idx:])
        if rel_path is None:
            return ""

    quoted_path = quote(rel_path.as_posix())
    return f"{repo_url}/blob/{repo_branch}/{quoted_path}#L{start_line}"


def _format_source_name(raw_name: str) -> str:
    """Return canonical display names for pre-existing implementations.
    Falls back to Title Case when not explicitly mapped.
    """
    if not raw_name:
        return ""
    normalized = raw_name.replace("_", " ").strip().lower()
    mapping = {
        "fantastic": "FANTASTIC",
        "jsymbolic": "jSymbolic",
        "midi toolbox": "MIDI Toolbox",
        "midi_toolbox": "MIDI Toolbox",
        "must": "MUST",
        "simile": "SIMILE",
        "idyom": "IDyOM",
    }
    return mapping.get(normalized, raw_name.replace("_", " ").strip().title())


def collect_feature_rows(objs: Iterable[tuple[str, object]]) -> list[FeatureRow]:
    """Collect metadata rows for a set of (name, callable/property) pairs."""
    rows: list[FeatureRow] = []
    seen_function_ids: set[int] = set()

    for name, obj in objs:
        if name.startswith("get_") or name.startswith("_"):
            continue
        if name in FEATURE_ALIAS_EXPORTS:
            continue

        if name == "InverseEntropyWeighting":
            continue

        is_property = isinstance(obj, property)

        if is_property and hasattr(obj, 'fget') and obj.fget is not None:
            feature_types = getattr(obj.fget, "_feature_types", None)
        else:
            feature_types = getattr(obj, "_feature_types", None)
        if not feature_types and not is_property:
            continue

        if not is_property:
            try:
                function_id = id(inspect.unwrap(obj))
            except (AttributeError, ValueError):
                function_id = id(obj)
            if function_id in seen_function_ids:
                continue
            seen_function_ids.add(function_id)

        if "." in name:
            class_name, prop_name = name.split(".", 1)
            class_display = class_name
            if class_name == "StepContour":
                class_display = "Step Contour"
            elif class_name == "InterpolationContour":
                class_display = "Interpolation Contour"
            elif class_name == "PolynomialContour":
                class_display = "Polynomial Contour"
            elif class_name == "HuronContour":
                class_display = "Huron Contour"
            elif class_name == "NGramCounter":
                class_display = ""
            class_part = f"{class_display} " if class_display else ""
            pretty_name = f"{class_part}{snake_to_title(prop_name)}".strip()
        else:
            pretty_name = snake_to_title(name)

        pretty_name = FEATURE_DISPLAY_NAME_OVERRIDES.get(name, pretty_name)

        pretty_name = fix_possessive_feature_names(normalize_feature_text(pretty_name))

        source_url = _build_source_url(obj)

        feature_sources = getattr(obj, "_feature_sources", [])
        if feature_sources:
            implementations = ", ".join(sorted({_format_source_name(s) for s in feature_sources}))
        else:
            implementations = ""
            if is_property and "." in name:
                class_name = name.split(".", 1)[0]
                class_source_map = {
                    "StepContour": "FANTASTIC",
                    "InterpolationContour": "FANTASTIC",
                    "PolynomialContour": "FANTASTIC",
                    "HuronContour": "FANTASTIC",
                    "NGramCounter": "FANTASTIC",
                }
                mapped = class_source_map.get(class_name)
                if mapped:
                    implementations = mapped

        doc_string = inspect.getdoc(obj.fget) if is_property else inspect.getdoc(obj)
        sections = extract_sections_from_docstring(doc_string or "")
        description = normalize_feature_text(" ".join(sections.get("Preamble", "").split()))
        doc_notes = normalize_feature_text(" ".join(sections.get("Note", "").split()))
        feature_python_name = name.split(".", 1)[-1] if "." in name else name
        notes = _notes_for_feature(feature_python_name, doc_notes)

        citation_section = sections.get("Citation", "").strip()
        if citation_section:
            references = normalize_feature_text(
                " | ".join([" ".join(p.split()) for p in re.split(r"\n\s*\n", citation_section) if p.strip()])
            )
        else:
            references = ""

        if is_property:
            type_label = "Descriptor"
            if "." in name:
                class_name, prop_name = name.split(".", 1)
                if class_name == "PolynomialContour" and prop_name == "coefficients":
                    type_label = "Sequence"  # list[float]
                if class_name == "HuronContour" and prop_name == "huron_contour":
                    type_label = "Descriptor"  # str
        else:
            type_label = determine_type_from_return_annotation(obj)

        if is_property and hasattr(obj, 'fget') and obj.fget is not None:
            domain_attr = getattr(obj.fget, "_feature_domain", None)
        else:
            domain_attr = getattr(obj, "_feature_domain", None)

        feature_name = name.split(".", 1)[-1] if "." in name else name
        category = _get_feature_category(obj, domain_attr, feature_name)

        if is_property and "." in name:
            class_name = name.split(".", 1)[0]
            contour_classes = ["StepContour", "InterpolationContour", "PolynomialContour", "HuronContour"]
            if class_name in contour_classes and not domain_attr:
                domain_attr = "pitch"

        domain_for_filter = domain_attr if domain_attr else ""

        target_for_sig = obj.fget if (is_property and hasattr(obj, "fget") and obj.fget is not None) else obj
        try:
            has_corpus_stats_param = "corpus_stats" in inspect.signature(target_for_sig).parameters
        except (TypeError, ValueError):
            has_corpus_stats_param = False

        rows.append(
            FeatureRow(
                python_name=name,
                name=pretty_name,
                source_url=source_url,
                implementations=implementations,
                references=references,
                description=description,
                type_label=type_label,
                notes=notes,
                category=category,
                domain=domain_for_filter,
                sort_name=pretty_name,
                feature_types=tuple(feature_types or ()),
                has_corpus_stats_param=has_corpus_stats_param,
            )
        )
    return rows


def to_dataframe(rows: list[FeatureRow]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in rows])
    sort_cols = ['sort_name'] if 'sort_name' in df.columns else ['name']
    df = df.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)

    return df


def _iter_documented_features():
    """Yield (python_name, obj) pairs for every documented atomic feature.

    Mirrors the member set scanned by the Quarto docs table: functions in
    `melody_features.features` plus descriptor properties on the contour and
    n-gram classes.
    """
    from . import features as features_module
    from .contour import (
        HuronContour,
        InterpolationContour,
        PolynomialContour,
        StepContour,
    )
    from .ngram_counter import NGramCounter

    members = inspect.getmembers(features_module)
    for n, o in members:
        if inspect.isfunction(o):
            yield n, o

    feature_classes = [StepContour, InterpolationContour, PolynomialContour, HuronContour, NGramCounter]
    excluded_properties = {"count_values", "freq_spec", "total_tokens"}

    for cls in feature_classes:
        class_name = cls.__name__
        for prop_name, prop_obj in inspect.getmembers(cls):
            if isinstance(prop_obj, property) and prop_name not in excluded_properties:
                yield f"{class_name}.{prop_name}", prop_obj


def build_table() -> pd.DataFrame:
    """Build the exported feature table from atomic feature callables only.

    Convention: helper/aggregator wrappers (for example `get_*`) are intentionally
    excluded so the table contains only user-facing scalar/sequence feature atoms.
    """
    rows = collect_feature_rows(_iter_documented_features())
    return to_dataframe(rows)


def count_features() -> int:
    """Return the number of features included in the summary table."""
    return len(build_table())


_IDYOM_METADATA_ROW = {
    "feature_name": "idyom",
    "family": "idyom",
    "source": "IDyOM",
    "domain": "",
    "type": "",
    "description": (
        "Information-theoretic feature computed by IDyOM for a configured "
        "model. The feature name suffix is `<config_name>_<idyom_metric>` "
        "and depends on the `Config.idyom` settings supplied to "
        "`get_all_features`, so individual metrics are not enumerated here."
    ),
    "notes": "",
    "references": "Pearce, M. T. (2005). The construction and evaluation of statistical models of melodic structure in music perception and composition.",
}


# Names collected under `get_lexical_diversity_features`/`get_mtype_features`
# regardless of whether they're reached via a decorated top-level function or
# an `NGramCounter` descriptor property.
_MTYPE_FEATURE_NAMES = frozenset(
    {"yules_k", "simpsons_d", "sichels_s", "honores_h", "mean_entropy", "mean_productivity"}
)

# Corpus-dependent feature names that take a `corpus_stats` argument; these
# are placed under the `corpus` family regardless of their decorator type.
_CORPUS_FEATURE_NAMES = frozenset(
    {
        "tfdf_spearman", "tfdf_kendall", "norm_log_dist", "max_log_df", "min_log_df",
        "mean_log_df", "mean_global_local_weight", "mean_global_weight",
        "mean_log_tfdf", "mean_document_frequency",
    }
)

_CONTOUR_CLASS_NAMES = frozenset(
    {"StepContour", "InterpolationContour", "PolynomialContour", "HuronContour"}
)


def _resolve_family_key(
    python_name: str,
    feature_types: tuple,
    domain: Optional[str],
    feature_suffix: str,
    has_corpus_stats_param: bool,
) -> str:
    """Derive the wide-format family prefix directly from decorator metadata.

    Mirrors how `pipeline/processing.py` actually buckets each feature into
    `get_pitch_features`, `get_interval_features`, etc. (rather than the
    docs-table's display grouping, which intentionally regroups a couple of
    features for presentation purposes and would otherwise produce a family
    that never matches a real wide-format column).
    """
    if "." in python_name:
        class_name = python_name.split(".", 1)[0]
        if class_name in _CONTOUR_CLASS_NAMES:
            return "contour"
        if class_name == "NGramCounter":
            return "lexical_diversity" if feature_suffix in _MTYPE_FEATURE_NAMES else "complexity"

    if feature_suffix in _MTYPE_FEATURE_NAMES:
        return "lexical_diversity"

    if has_corpus_stats_param or feature_suffix in _CORPUS_FEATURE_NAMES:
        return "corpus"

    if not feature_types:
        return "other"

    feature_type = feature_types[0]
    if feature_type == "interval":
        if domain == "rhythm":
            return "inter_onset_interval" if "ioi" in feature_suffix.lower() else "timing"
        return "pitch_interval"
    if feature_type == "absolute":
        return "absolute_pitch"
    # pitch_class, contour, timing, tonality, metre, expectation, complexity,
    # lexical_diversity, and corpus are already valid family keys as-is.
    return feature_type


def get_feature_metadata() -> pd.DataFrame:
    """Return a metadata table for every feature produced by `get_all_features`.

    One row per feature `family.feature_name` combination (matching the
    dotted column names in the wide-format DataFrame returned by
    :func:`melody_features.get_all_features`), with columns describing the
    feature's source, family/category, domain, return type, and description.
    This table can be joined onto a long-format feature table by
    `feature_name` (see :func:`melody_features.to_long_format`).

    `to_long_format` falls back to inferring `family`/`source` from the
    column prefix for any wide-format column (e.g. dynamic IDyOM columns)
    that doesn't have an exact match here.

    Returns
    -------
    pd.DataFrame
        Columns: `feature_name`, `family`, `source`, `domain`, `type`,
        `description`, `notes`, `references`.
    """
    rows = collect_feature_rows(_iter_documented_features())

    records = []
    for row in rows:
        feature_suffix = CONTOUR_COMPOSITE_ALIASES.get(
            row.python_name,
            row.python_name.split(".", 1)[-1] if "." in row.python_name else row.python_name,
        )
        family_key = _resolve_family_key(
            row.python_name,
            row.feature_types,
            row.domain or None,
            feature_suffix,
            row.has_corpus_stats_param,
        )

        records.append(
            {
                "feature_name": f"{family_key}.{feature_suffix}",
                "family": family_key,
                "source": row.implementations,
                "domain": row.domain,
                "type": row.type_label,
                "description": row.description,
                "notes": row.notes,
                "references": row.references,
            }
        )

    records.append(dict(_IDYOM_METADATA_ROW))

    df = pd.DataFrame.from_records(records)
    df = df.drop_duplicates(subset="feature_name", keep="first").reset_index(drop=True)
    return df
