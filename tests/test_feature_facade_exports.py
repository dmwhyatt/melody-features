import importlib.util
import inspect

import melody_features
import melody_features.feature_definitions.absolute_pitch as absolute_pitch_definitions
import melody_features.feature_definitions.complexity as complexity_definitions
import melody_features.feature_definitions.contour as contour_definitions
import melody_features.feature_definitions.corpus as corpus_definitions
import melody_features.feature_definitions.expectation as expectation_definitions
import melody_features.feature_definitions.inter_onset_interval as inter_onset_interval_definitions
import melody_features.feature_definitions.lexical_diversity as lexical_diversity_definitions
import melody_features.feature_definitions.metre as metre_definitions
import melody_features.feature_definitions.pitch_class as pitch_class_definitions
import melody_features.feature_definitions.pitch_interval as pitch_interval_definitions
import melody_features.feature_definitions.timing as timing_definitions
import melody_features.feature_definitions.tonality as tonality_definitions
import melody_features.features as features_module


MOVED_MODULES = (
    absolute_pitch_definitions,
    pitch_class_definitions,
    pitch_interval_definitions,
    contour_definitions,
    timing_definitions,
    inter_onset_interval_definitions,
    tonality_definitions,
    expectation_definitions,
    metre_definitions,
    corpus_definitions,
    complexity_definitions,
    lexical_diversity_definitions,
)

REMOVED_ROOT_MODULES = (
    "melody_features.absolute_pitch_features",
    "melody_features.pitch_class_features",
    "melody_features.pitch_interval_features",
    "melody_features.contour_features",
    "melody_features.timing_features",
    "melody_features.inter_onset_interval_features",
    "melody_features.tonality_features",
    "melody_features.expectation_features",
    "melody_features.metre_features",
    "melody_features.corpus_features",
    "melody_features.complexity_features",
    "melody_features.lexical_diversity_features",
)

ALIASES = {
    absolute_pitch_definitions: {
        "ambitus": "pitch_range",
        "pitch_variability": "pitch_standard_deviation",
    },
    pitch_class_definitions: {
        "number_of_common_pitches_classes": "number_of_common_pitch_classes",
    },
    pitch_interval_definitions: {
        "mean_melodic_interval": "mean_absolute_interval",
        "most_common_interval": "modal_interval",
    },
    contour_definitions: {
        "get_comb_contour_matrix": "comb_contour_matrix",
    },
    timing_definitions: {
        "total_number_of_notes": "length",
        "duration_in_seconds": "global_duration",
        "short_note_fraction": "amount_of_staccato",
    },
    inter_onset_interval_definitions: {
        "average_time_between_attacks": "ioi_mean",
        "variability_of_time_between_attacks": "ioi_standard_deviation",
    },
    tonality_definitions: {
        "keyname": "key",
    },
}

DOMAIN_DISCOVERY = (
    ("pitch", ["absolute"], (absolute_pitch_definitions,)),
    ("pitch", ["pitch_class"], (pitch_class_definitions,)),
    ("pitch", ["interval"], (pitch_interval_definitions,)),
    ("rhythm", ["timing"], (timing_definitions,)),
    ("rhythm", ["interval"], (inter_onset_interval_definitions,)),
)

TYPE_DISCOVERY = (
    ("contour", (contour_definitions,)),
    ("tonality", (tonality_definitions,)),
    ("expectation", (expectation_definitions,)),
    ("metre", (metre_definitions,)),
    ("corpus", (corpus_definitions,)),
    ("complexity", (complexity_definitions,)),
    ("lexical_diversity", (lexical_diversity_definitions,)),
)


def _module_exports(module):
    return tuple(getattr(module, "__all__", ()))


def _canonical_feature_names(modules, feature_type=None, domain=None):
    names = []
    for module in modules:
        for name in _module_exports(module):
            obj = getattr(module, name)
            if not inspect.isfunction(obj):
                continue
            if obj.__name__ != name:
                continue
            if feature_type is not None and feature_type not in getattr(obj, "_feature_types", []):
                continue
            if domain is not None and getattr(obj, "_feature_domain", None) != domain:
                continue
            names.append(name)
    return names


def test_root_feature_modules_are_not_importable():
    for module_name in REMOVED_ROOT_MODULES:
        assert importlib.util.find_spec(module_name) is None


def test_moved_exports_are_reexported_from_facade():
    for module in MOVED_MODULES:
        for name in _module_exports(module):
            assert getattr(features_module, name) is getattr(module, name)


def test_moved_exports_are_available_from_package_root():
    for module in MOVED_MODULES:
        for name in _module_exports(module):
            assert getattr(melody_features, name) is getattr(module, name)
            assert name in melody_features.__all__


def test_moved_feature_aliases_are_reexported_from_facade_and_root():
    for module, aliases in ALIASES.items():
        for alias, canonical_name in aliases.items():
            assert getattr(features_module, alias) is getattr(features_module, canonical_name)
            assert getattr(features_module, alias) is getattr(module, alias)
            assert getattr(melody_features, alias) is getattr(module, alias)


def test_moved_feature_facade_domain_discovery_uses_canonical_names():
    for domain, feature_types, modules in DOMAIN_DISCOVERY:
        discovered = features_module._get_features_by_domain_and_types(domain, feature_types)
        expected = _canonical_feature_names(modules, feature_types[0], domain)

        for name in expected:
            assert discovered[name] is getattr(features_module, name)

        for aliases in ALIASES.values():
            for alias in aliases:
                assert alias not in discovered


def test_moved_feature_facade_type_discovery_uses_canonical_names():
    for feature_type, modules in TYPE_DISCOVERY:
        discovered = features_module._get_features_by_type(feature_type)
        expected = _canonical_feature_names(modules, feature_type)

        for name in expected:
            assert discovered[name] is getattr(features_module, name)


def test_moved_labelled_callables_stay_visible_to_facade_introspection():
    labelled_callables = {
        name
        for name, obj in inspect.getmembers(features_module)
        if callable(obj) and hasattr(obj, "_feature_source")
    }

    expected = set()
    for module in MOVED_MODULES:
        for name in _module_exports(module):
            obj = getattr(module, name)
            if callable(obj) and hasattr(obj, "_feature_source"):
                expected.add(name)

    assert expected.issubset(labelled_callables)
