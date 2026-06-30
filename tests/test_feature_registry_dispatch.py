from types import SimpleNamespace

import melody_features.features as features_module
from melody_features.feature_decorators import FeatureType
from melody_features.feature_dispatch import collect_feature_values, invoke_feature
from melody_features.feature_registry import (
    discover_atomic_features,
    get_features_by_domain_and_types,
    get_features_by_source,
    get_features_by_type,
    list_available_features,
)


def canonical():
    return 1


canonical._feature_types = [FeatureType.ABSOLUTE]
canonical._feature_domain = "pitch"
canonical._feature_sources = ["jsymbolic"]


def other():
    return 2


other._feature_types = [FeatureType.TIMING]
other._feature_domain = "rhythm"
other._feature_source = "legacy"


class _SourceFeature:
    _feature_sources = ["jsymbolic"]


def test_registry_deduplicates_aliases_for_category_discovery():
    module = SimpleNamespace(
        canonical=canonical,
        alias=canonical,
        other=other,
    )

    by_type = get_features_by_type(module, FeatureType.ABSOLUTE)
    by_domain_and_type = get_features_by_domain_and_types(module, "pitch", [FeatureType.ABSOLUTE])

    assert by_type == {"canonical": canonical}
    assert by_domain_and_type == {"canonical": canonical}


def test_source_registry_preserves_aliases_and_legacy_source_metadata():
    module = SimpleNamespace(
        canonical=canonical,
        alias=canonical,
        source_class=_SourceFeature,
        legacy=other,
    )

    jsymbolic_features = get_features_by_source(module, "jsymbolic")
    legacy_features = get_features_by_source(module, "legacy")

    assert jsymbolic_features["canonical"] is canonical
    assert jsymbolic_features["alias"] is canonical
    assert jsymbolic_features["source_class"] is _SourceFeature
    assert legacy_features == {"legacy": other}


def test_list_available_features_excludes_aliases_and_aggregators():
    names = list_available_features()

    assert "pitch_range" in names
    assert "ambitus" not in names
    assert not any(name.startswith("get_") for name in names)


def test_list_available_features_detailed_filter_by_source():
    fantastic = list_available_features(detailed=True, source="fantastic")

    assert fantastic
    assert all("fantastic" in entry["sources"] for entry in fantastic)
    assert any(entry["summary"] for entry in fantastic)


def test_discover_atomic_features_includes_contour_descriptors():
    entries = discover_atomic_features(features_module)
    names = {entry.name for entry in entries}

    assert "StepContour.global_variation" in names
    assert "NGramCounter.yules_k" in names


def test_features_module_facade_uses_registry_without_changing_discovery_surface():
    absolute_features = features_module._get_features_by_type(FeatureType.ABSOLUTE)
    pitch_absolute_features = features_module._get_features_by_domain_and_types(
        "pitch", [FeatureType.ABSOLUTE]
    )
    source_features = features_module._get_features_by_source("jsymbolic")

    assert absolute_features["pitch_range"] is features_module.pitch_range
    assert "ambitus" not in absolute_features
    assert pitch_absolute_features["pitch_range"] is features_module.pitch_range
    assert source_features["pitch_range"] is features_module.pitch_range
    assert source_features["ambitus"] is features_module.pitch_range


def test_get_novel_features_uses_novel_source_label():
    novel_features = features_module._get_features_by_source("novel")
    assert novel_features
    assert all("novel" in func._feature_sources for func in novel_features.values())


class _Melody:
    pitches = [60, 62]
    starts = [0.0, 1.0]
    ends = [0.5, 1.5]
    tempo = 120


def test_invoke_feature_binds_melody_fields_and_defaults():
    def feature(pitches, starts, ends, tempo, ppqn, phrase_gap, max_ngram_order):
        return pitches, starts, ends, tempo, ppqn, phrase_gap, max_ngram_order

    result = invoke_feature(feature, _Melody(), default_max_ngram_order=7)

    assert result == ([60, 62], [0.0, 1.0], [0.5, 1.5], 120, 480, 1.5, 7)


def test_collect_feature_values_expands_two_item_tuples():
    def summary(pitches):
        return min(pitches), max(pitches)

    values = collect_feature_values(
        {"summary": summary},
        _Melody(),
        tuple_suffix="high",
        default_max_ngram_order=5,
    )

    assert values == {"summary_mean": 60, "summary_high": 62}
