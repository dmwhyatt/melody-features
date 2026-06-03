from types import SimpleNamespace

import melody_features.features as features_module
from melody_features.feature_decorators import FeatureType
from melody_features.feature_dispatch import collect_feature_values, invoke_feature
from melody_features.feature_registry import (
    get_features_by_domain_and_types,
    get_features_by_source,
    get_features_by_type,
)


def _decorated_feature():
    return 1


_decorated_feature._feature_types = [FeatureType.ABSOLUTE]
_decorated_feature._feature_domain = "pitch"
_decorated_feature._feature_sources = ["jsymbolic"]


def _other_feature():
    return 2


_other_feature._feature_types = [FeatureType.TIMING]
_other_feature._feature_domain = "rhythm"
_other_feature._feature_source = "legacy"


class _SourceFeature:
    _feature_sources = ["jsymbolic"]


def test_registry_deduplicates_aliases_for_category_discovery():
    module = SimpleNamespace(
        canonical=_decorated_feature,
        alias=_decorated_feature,
        other=_other_feature,
    )

    by_type = get_features_by_type(module, FeatureType.ABSOLUTE)
    by_domain_and_type = get_features_by_domain_and_types(module, "pitch", [FeatureType.ABSOLUTE])

    assert by_type == {"canonical": _decorated_feature}
    assert by_domain_and_type == {"canonical": _decorated_feature}


def test_source_registry_preserves_aliases_and_legacy_source_metadata():
    module = SimpleNamespace(
        canonical=_decorated_feature,
        alias=_decorated_feature,
        source_class=_SourceFeature,
        legacy=_other_feature,
    )

    jsymbolic_features = get_features_by_source(module, "jsymbolic")
    legacy_features = get_features_by_source(module, "legacy")

    assert jsymbolic_features["canonical"] is _decorated_feature
    assert jsymbolic_features["alias"] is _decorated_feature
    assert jsymbolic_features["source_class"] is _SourceFeature
    assert legacy_features == {"legacy": _other_feature}


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
