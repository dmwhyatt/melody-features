import inspect

import melody_features.absolute_pitch_features as absolute_pitch_features
import melody_features.features as features_module


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

ABSOLUTE_PITCH_ALIASES = {
    "ambitus": "pitch_range",
    "pitch_variability": "pitch_standard_deviation",
}


def test_absolute_pitch_features_are_reexported_from_facade():
    for name in ABSOLUTE_PITCH_FEATURES:
        assert getattr(features_module, name) is getattr(absolute_pitch_features, name)


def test_absolute_pitch_aliases_are_reexported_from_facade():
    for alias, canonical_name in ABSOLUTE_PITCH_ALIASES.items():
        assert getattr(features_module, alias) is getattr(features_module, canonical_name)
        assert getattr(features_module, alias) is getattr(absolute_pitch_features, alias)


def test_absolute_pitch_facade_discovery_uses_canonical_names():
    discovered = features_module._get_features_by_domain_and_types("pitch", ["absolute"])

    for name in ABSOLUTE_PITCH_FEATURES:
        assert discovered[name] is getattr(absolute_pitch_features, name)

    for alias in ABSOLUTE_PITCH_ALIASES:
        assert alias not in discovered


def test_absolute_pitch_features_stay_visible_to_facade_introspection():
    labelled = {
        name
        for name, obj in inspect.getmembers(features_module)
        if inspect.isfunction(obj) and hasattr(obj, "_feature_source")
    }

    assert set(ABSOLUTE_PITCH_FEATURES).issubset(labelled)
    assert set(ABSOLUTE_PITCH_ALIASES).issubset(labelled)
