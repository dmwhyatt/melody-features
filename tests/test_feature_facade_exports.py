import inspect

import melody_features.absolute_pitch_features as absolute_pitch_features
import melody_features.features as features_module
import melody_features.pitch_class_features as pitch_class_features
import melody_features.pitch_interval_features as pitch_interval_features


ABSOLUTE_PITCH_FEATURES = (
    "pitch_range",
    "pitch_standard_deviation",
    "first_pitch",
    "last_pitch",
    "basic_pitch_histogram",
    "melodic_pitch_variety",
    "mean_pitch",
    "most_common_pitch",
    "number_of_unique_pitches",
    "number_of_common_pitches",
    "tessitura",
    "mean_tessitura",
    "tessitura_std",
    "prevalence_of_most_common_pitch",
    "relative_prevalence_of_top_pitches",
    "interval_between_most_prevalent_pitches",
    "pitch_skewness",
    "pitch_kurtosis",
    "importance_of_bass_register",
    "importance_of_middle_register",
    "importance_of_high_register",
    "pitch_spelling",
    "repeated_notes",
    "stepwise_motion",
)

PITCH_CLASS_FEATURES = (
    "pitch_class_variability",
    "pitch_class_variability_after_folding",
    "pcdist1",
    "first_pitch_class",
    "last_pitch_class",
    "dominant_spread",
    "mean_pitch_class",
    "most_common_pitch_class",
    "number_of_unique_pitch_classes",
    "number_of_common_pitch_classes",
    "prevalence_of_most_common_pitch_class",
    "relative_prevalence_of_top_pitch_classes",
    "interval_between_most_prevalent_pitch_classes",
    "folded_fifths_pitch_class_histogram",
    "pitch_class_skewness",
    "pitch_class_kurtosis",
    "pitch_class_skewness_after_folding",
    "pitch_class_kurtosis_after_folding",
    "strong_tonal_centres",
)

PITCH_INTERVAL_FEATURES = (
    "pitch_interval",
    "absolute_interval_range",
    "mean_absolute_interval",
    "standard_deviation_absolute_interval",
    "modal_interval",
    "ivdist1",
    "ivdirdist1",
    "ivsizedist1",
    "interval_direction",
    "interval_direction_mean",
    "interval_direction_std",
    "average_length_of_melodic_arcs",
    "average_interval_span_by_melodic_arcs",
    "distance_between_most_prevalent_melodic_intervals",
    "melodic_interval_histogram",
    "melodic_large_intervals",
    "melodic_thirds",
    "melodic_perfect_fourths",
    "melodic_tritones",
    "melodic_perfect_fifths",
    "melodic_sixths",
    "melodic_sevenths",
    "melodic_octaves",
    "minor_major_third_ratio",
    "direction_of_melodic_motion",
    "number_of_common_melodic_intervals",
    "prevalence_of_most_common_melodic_interval",
    "relative_prevalence_of_most_common_melodic_intervals",
    "amount_of_arpeggiation",
    "chromatic_motion",
)

PITCH_INTERVAL_HELPERS = ("variable_melodic_intervals",)

FEATURE_MODULES = (
    (absolute_pitch_features, ABSOLUTE_PITCH_FEATURES),
    (pitch_class_features, PITCH_CLASS_FEATURES),
    (pitch_interval_features, PITCH_INTERVAL_FEATURES + PITCH_INTERVAL_HELPERS),
)

FEATURE_DISCOVERY = (
    ("pitch", ["absolute"], ABSOLUTE_PITCH_FEATURES),
    ("pitch", ["pitch_class"], PITCH_CLASS_FEATURES),
    ("pitch", ["interval"], PITCH_INTERVAL_FEATURES),
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
}


def test_moved_features_are_reexported_from_facade():
    for module, names in FEATURE_MODULES:
        for name in names:
            assert getattr(features_module, name) is getattr(module, name)


def test_moved_feature_aliases_are_reexported_from_facade():
    for module, aliases in ALIASES.items():
        for alias, canonical_name in aliases.items():
            assert getattr(features_module, alias) is getattr(features_module, canonical_name)
            assert getattr(features_module, alias) is getattr(module, alias)


def test_moved_feature_facade_discovery_uses_canonical_names():
    for domain, feature_types, names in FEATURE_DISCOVERY:
        discovered = features_module._get_features_by_domain_and_types(domain, feature_types)

        for name in names:
            assert discovered[name] is getattr(features_module, name)

        for aliases in ALIASES.values():
            for alias in aliases:
                assert alias not in discovered


def test_moved_features_stay_visible_to_facade_introspection():
    labelled = {
        name
        for name, obj in inspect.getmembers(features_module)
        if inspect.isfunction(obj) and hasattr(obj, "_feature_source")
    }

    expected = set(ABSOLUTE_PITCH_FEATURES).union(PITCH_CLASS_FEATURES, PITCH_INTERVAL_FEATURES)
    expected_aliases = {alias for aliases in ALIASES.values() for alias in aliases}

    assert expected.issubset(labelled)
    assert expected_aliases.issubset(labelled)
