import inspect

import melody_features.absolute_pitch_features as absolute_pitch_features
import melody_features.corpus_features as corpus_features
import melody_features.expectation_features as expectation_features
import melody_features.features as features_module
import melody_features.metre_features as metre_features
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

EXPECTATION_FEATURES = (
    "narmour_registral_direction",
    "narmour_proximity",
    "narmour_closure",
    "narmour_registral_return",
    "narmour_intervallic_difference",
    "melodic_embellishment",
    "mobility",
    "mean_mobility",
    "mobility_std",
    "melodic_attraction",
    "mean_melodic_attraction",
    "melodic_attraction_std",
    "melodic_accent",
    "mean_melodic_accent",
    "melodic_accent_std",
    "compltrans",
    "pitch_stm_mean_information_content",
    "pitch_ltm_mean_information_content",
    "rhythm_stm_mean_information_content",
    "rhythm_ltm_mean_information_content",
)

EXPECTATION_HELPERS = (
    "_get_key_distances",
    "get_narmour_features",
    "_stability_distance",
    "_get_simonton_transition_matrix",
)

METRE_FEATURES = (
    "metric_hierarchy",
    "meter_accent",
    "meter_numerator",
    "meter_denominator",
    "proportion_of_time_in_first_meter",
    "number_of_unique_time_signatures",
    "syncopation",
    "syncopicity",
)

METRE_HELPERS = ("_meter_accent_mean",)

CORPUS_FEATURES = (
    "get_ngram_document_frequency",
    "tfdf_spearman",
    "tfdf_kendall",
    "mean_log_tfdf",
    "norm_log_dist",
    "max_log_df",
    "min_log_df",
    "mean_log_df",
    "mean_global_local_weight",
    "std_global_local_weight",
    "mean_global_weight",
    "std_global_weight",
    "get_corpus_features",
)

CORPUS_DECORATED_FEATURES = (
    "tfdf_spearman",
    "tfdf_kendall",
    "mean_log_tfdf",
    "norm_log_dist",
    "max_log_df",
    "min_log_df",
    "mean_log_df",
    "mean_global_local_weight",
    "std_global_local_weight",
    "mean_global_weight",
    "std_global_weight",
)

CORPUS_CLASSES = ("InverseEntropyWeighting",)

CORPUS_LABELLED_EXPORTS = CORPUS_DECORATED_FEATURES + (
    "get_ngram_document_frequency",
    "InverseEntropyWeighting",
)

CORPUS_HELPERS = (
    "_fantastic_melody_tokens",
    "_fantastic_melody_tf_df",
    "_fantastic_log_normalized_tf_df",
    "_fantastic_melody_ngram_counts",
    "_fantastic_min_tie_ranks",
    "_compute_corpus_feature_bundle",
    "_setup_corpus_statistics",
)

FEATURE_MODULES = (
    (absolute_pitch_features, ABSOLUTE_PITCH_FEATURES),
    (pitch_class_features, PITCH_CLASS_FEATURES),
    (pitch_interval_features, PITCH_INTERVAL_FEATURES + PITCH_INTERVAL_HELPERS),
    (expectation_features, EXPECTATION_FEATURES + EXPECTATION_HELPERS),
    (metre_features, METRE_FEATURES + METRE_HELPERS),
    (corpus_features, CORPUS_FEATURES + CORPUS_CLASSES + CORPUS_HELPERS),
)

DOMAIN_DISCOVERY = (
    ("pitch", ["absolute"], ABSOLUTE_PITCH_FEATURES),
    ("pitch", ["pitch_class"], PITCH_CLASS_FEATURES),
    ("pitch", ["interval"], PITCH_INTERVAL_FEATURES),
)

TYPE_DISCOVERY = (
    ("expectation", EXPECTATION_FEATURES),
    ("metre", METRE_FEATURES),
    ("corpus", CORPUS_DECORATED_FEATURES),
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


def test_moved_feature_facade_domain_discovery_uses_canonical_names():
    for domain, feature_types, names in DOMAIN_DISCOVERY:
        discovered = features_module._get_features_by_domain_and_types(domain, feature_types)

        for name in names:
            assert discovered[name] is getattr(features_module, name)

        for aliases in ALIASES.values():
            for alias in aliases:
                assert alias not in discovered


def test_moved_feature_facade_type_discovery_uses_canonical_names():
    for feature_type, names in TYPE_DISCOVERY:
        discovered = features_module._get_features_by_type(feature_type)

        for name in names:
            assert discovered[name] is getattr(features_module, name)


def test_moved_features_stay_visible_to_facade_introspection():
    labelled = {
        name
        for name, obj in inspect.getmembers(features_module)
        if inspect.isfunction(obj) and hasattr(obj, "_feature_source")
    }

    expected = set(ABSOLUTE_PITCH_FEATURES).union(
        PITCH_CLASS_FEATURES,
        PITCH_INTERVAL_FEATURES,
        EXPECTATION_FEATURES,
        METRE_FEATURES,
        CORPUS_LABELLED_EXPORTS,
    )
    expected_aliases = {alias for aliases in ALIASES.values() for alias in aliases}

    assert expected.issubset(labelled)
    assert expected_aliases.issubset(labelled)


def test_moved_classes_stay_visible_to_facade_introspection():
    labelled_callables = {
        name
        for name, obj in inspect.getmembers(features_module)
        if (inspect.isclass(obj) or (hasattr(obj, "__call__") and hasattr(obj, "__name__")))
        and hasattr(obj, "_feature_source")
    }

    assert set(CORPUS_CLASSES).issubset(labelled_callables)
