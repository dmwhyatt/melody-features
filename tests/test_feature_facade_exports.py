import inspect

import melody_features.absolute_pitch_features as absolute_pitch_features
import melody_features.complexity_features as complexity_features
import melody_features.contour_features as contour_features
import melody_features.corpus_features as corpus_features
import melody_features.expectation_features as expectation_features
import melody_features.features as features_module
import melody_features.inter_onset_interval_features as inter_onset_interval_features
import melody_features.lexical_diversity_features as lexical_diversity_features
import melody_features.metre_features as metre_features
import melody_features.pitch_class_features as pitch_class_features
import melody_features.pitch_interval_features as pitch_interval_features
import melody_features.timing_features as timing_features
import melody_features.tonality_features as tonality_features


MOVED_MODULES = (
    absolute_pitch_features,
    pitch_class_features,
    pitch_interval_features,
    contour_features,
    timing_features,
    inter_onset_interval_features,
    tonality_features,
    expectation_features,
    metre_features,
    corpus_features,
    complexity_features,
    lexical_diversity_features,
)

ALIASES = {
    absolute_pitch_features: {
        "ambitus": "pitch_range",
        "pitch_variability": "pitch_standard_deviation",
    },
    pitch_class_features: {
        "number_of_common_pitches_classes": "number_of_common_pitch_classes",
    },
    pitch_interval_features: {
        "mean_melodic_interval": "mean_absolute_interval",
        "most_common_interval": "modal_interval",
    },
    contour_features: {
        "get_comb_contour_matrix": "comb_contour_matrix",
    },
    timing_features: {
        "total_number_of_notes": "length",
        "duration_in_seconds": "global_duration",
        "short_note_fraction": "amount_of_staccato",
    },
    inter_onset_interval_features: {
        "average_time_between_attacks": "ioi_mean",
        "variability_of_time_between_attacks": "ioi_standard_deviation",
    },
    tonality_features: {
        "keyname": "key",
    },
}

DOMAIN_DISCOVERY = (
    ("pitch", ["absolute"], (absolute_pitch_features,)),
    ("pitch", ["pitch_class"], (pitch_class_features,)),
    ("pitch", ["interval"], (pitch_interval_features,)),
    ("rhythm", ["timing"], (timing_features,)),
    ("rhythm", ["interval"], (inter_onset_interval_features,)),
)

TYPE_DISCOVERY = (
    ("contour", (contour_features,)),
    ("tonality", (tonality_features,)),
    ("expectation", (expectation_features,)),
    ("metre", (metre_features,)),
    ("corpus", (corpus_features,)),
    ("complexity", (complexity_features,)),
    ("lexical_diversity", (lexical_diversity_features,)),
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


def test_moved_exports_are_reexported_from_facade():
    for module in MOVED_MODULES:
        for name in _module_exports(module):
            assert getattr(features_module, name) is getattr(module, name)


def test_moved_feature_aliases_are_reexported_from_facade():
    for module, aliases in ALIASES.items():
        for alias, canonical_name in aliases.items():
            assert getattr(features_module, alias) is getattr(features_module, canonical_name)
            assert getattr(features_module, alias) is getattr(module, alias)


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
